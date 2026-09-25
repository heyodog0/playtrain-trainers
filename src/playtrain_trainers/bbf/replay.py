"""Subsequence prioritized replay for BBF.

Three things make this different from a plain DQN buffer, and each is a
correctness trap worth stating:

1. **Frames are stored once, not stacked.** A 4-frame stack written per
   transition would store every frame four times: at the gin's
   `replay_capacity = 200000` and 84x84 grayscale that is 5.6 GB instead of
   1.4 GB. The stack is rebuilt at sample time from the frame ring.

2. **Samples are windows, not transitions.** The SPR loss needs the `jumps`
   observations and actions that FOLLOW the sampled transition, and the C51
   loss needs an n-step return whose horizon n changes during the run
   (annealed 10 -> 3 after each reset). So a start index is only usable if it
   has enough valid successors for both, and `n` is an argument to `sample`,
   not a property of the stored data.

3. **Episode ends inside a window are masked, not skipped.** Truncating the
   n-step return at a terminal and masking the SPR targets past it keeps the
   transitions near an episode end usable; dropping them instead would bias
   the buffer away from exactly the states where the reward is.

Backward stacking repeats an episode's first frame rather than reading into
the previous episode, which is what `PlayTrainEnv.reset` does when it fills
the stack -- so a replayed observation at an episode start is the same tensor
the agent actually saw.
"""
from __future__ import annotations

import numpy as np
import torch

from playtrain_trainers.bbf.config import BBFConfig


class SumTree:
    """Fixed-capacity sum tree for O(log n) proportional sampling.

    Stored as a flat array of size ``2 * capacity`` with leaves in the upper
    half, so a parent is at ``i // 2`` and no pointer chasing is needed.
    """

    def __init__(self, capacity: int):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self.nodes = np.zeros(2 * capacity, dtype=np.float64)

    @property
    def total(self) -> float:
        return float(self.nodes[1]) if self.capacity > 1 else float(self.nodes[1])

    def set(self, index: int, value: float) -> None:
        if not 0 <= index < self.capacity:
            raise IndexError(f"index {index} out of range for capacity {self.capacity}")
        if value < 0:
            raise ValueError("priority must be >= 0")
        i = index + self.capacity
        self.nodes[i] = value
        i //= 2
        while i >= 1:
            self.nodes[i] = self.nodes[2 * i] + self.nodes[2 * i + 1]
            i //= 2

    def set_many(self, indices: np.ndarray, values: np.ndarray) -> None:
        for i, v in zip(np.asarray(indices), np.asarray(values), strict=True):
            self.set(int(i), float(v))

    def get(self, index: int) -> float:
        return float(self.nodes[index + self.capacity])

    def query(self, prefix: float) -> int:
        """Smallest leaf index whose cumulative sum exceeds ``prefix``."""
        i = 1
        while i < self.capacity:
            left = self.nodes[2 * i]
            if prefix < left:
                i = 2 * i
            else:
                prefix -= left
                i = 2 * i + 1
        return i - self.capacity


class SubsequenceReplayBuffer:
    """Prioritized replay over windows of a single env's frame stream.

    Index convention. Slot ``i`` holds the transition
    ``(frame_i, action_i, reward_i, terminal_i)``: ``action_i`` was taken in
    the state whose newest frame is ``frame_i``, it produced ``reward_i``, and
    ``terminal_i`` says the episode ended as a result. The successor state's
    newest frame is therefore ``frame_{i+1}``.
    """

    def __init__(self, cfg: BBFConfig, frame_shape: tuple[int, int, int] | None = None):
        self.cfg = cfg
        self.capacity = cfg.replay_capacity
        self.frame_stack = cfg.frame_stack
        self.jumps = cfg.jumps
        self.prioritized = cfg.replay_scheme == "prioritized"
        # One FRAME, not one stacked observation: (C_frame, H, W).
        c_frame = 1 if cfg.obs_mode == "grayscale" else 3
        self.frame_shape = frame_shape or (c_frame, cfg.obs_size, cfg.obs_size)

        self.frames = np.zeros((self.capacity, *self.frame_shape), dtype=np.uint8)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.terminals = np.zeros(self.capacity, dtype=bool)
        # An episode id per slot makes every boundary question a comparison
        # rather than a scan: same-episode checks for stacking and for window
        # validity both reduce to equality.
        self.episode_ids = np.full(self.capacity, -1, dtype=np.int64)

        self.cursor = 0          # next slot to write
        self.size = 0            # slots filled (<= capacity)
        self.total_added = 0     # transitions ever added
        self._episode_id = 0
        self._tree = SumTree(self.capacity) if self.prioritized else None
        self._max_priority = 1.0

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------
    def add(
        self,
        frame: np.ndarray,
        action: int,
        reward: float,
        terminal: bool,
        *,
        episode_start: bool = False,
    ) -> int:
        """Append one transition; returns the slot written.

        ``episode_start`` begins a new episode id, so backward stacking will
        not read across the boundary. It is also implied by the previous
        transition being terminal, which is the common case; passing it
        explicitly covers a truncation that was not a terminal.
        """
        frame = np.asarray(frame)
        if frame.shape != self.frame_shape:
            raise ValueError(
                f"frame shape {frame.shape} != buffer's {self.frame_shape}"
            )
        if episode_start or self.total_added == 0:
            self._episode_id += 1
        i = self.cursor
        self.frames[i] = frame
        self.actions[i] = action
        self.rewards[i] = reward
        self.terminals[i] = bool(terminal)
        self.episode_ids[i] = self._episode_id
        if terminal:
            # The next add starts a fresh episode without the caller saying so.
            self._episode_id += 1
        self.cursor = (self.cursor + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.total_added += 1
        if self._tree is not None:
            # A new transition enters at the highest priority seen, so it is
            # sampled at least once before its priority reflects its own loss.
            self._tree.set(i, self._max_priority)
        return i

    # ------------------------------------------------------------------
    # Validity
    # ------------------------------------------------------------------
    def _oldest(self) -> int:
        """Index of the oldest live slot."""
        return 0 if self.size < self.capacity else self.cursor

    def _window_len(self, n: int) -> int:
        """Slots from the start index that a window may need, inclusive.

        Both paths reach one slot further than their step count suggests:

        * SPR's j-th target is the OBSERVATION at ``t + j + 1``, so for
          ``j = jumps - 1`` it needs slot ``t + jumps``.
        * The n-step return reads rewards at ``t .. t+n-1`` but bootstraps
          from the STATE at ``t + n``, so it needs slot ``t + n``.

        Hence ``max(jumps, n) + 1`` slots in total. Dropping the ``+ 1`` made
        the last usable start look valid while its final SPR target did not
        exist yet, and the mask silently hid it.
        """
        return max(self.jumps, n) + 1

    def valid_start(self, index: int, n: int) -> bool:
        """Can slot ``index`` start a window with horizon ``n``?

        A start is valid when it is live, is not itself the slot the cursor is
        about to overwrite, and either its whole window is present in the same
        episode or the episode ENDS inside the window (in which case the tail
        is masked rather than needed).
        """
        if self.size == 0 or not 0 <= index < self.capacity:
            return False
        if self.episode_ids[index] < 0:
            return False
        ep = self.episode_ids[index]
        need = self._window_len(n)
        for k in range(need):
            j = (index + k) % self.capacity
            if not self._is_live(j) or self.episode_ids[j] != ep:
                # The episode must have ended at or before the last live slot
                # of the window; otherwise the window runs into unwritten or
                # foreign data and the sample would be fabricated.
                return k > 0 and bool(self.terminals[(index + k - 1) % self.capacity])
            if self.terminals[j]:
                return True
        return True

    def _is_live(self, index: int) -> bool:
        """Is slot ``index`` written and not yet overwritten?

        While the buffer is filling, live means below the cursor. Once it has
        wrapped, the slot AT the cursor is the next to be clobbered and its
        successor relationship is broken, so it is excluded.
        """
        if self.size < self.capacity:
            return 0 <= index < self.cursor
        return index != self.cursor and self.episode_ids[index] >= 0

    def valid_indices(self, n: int) -> np.ndarray:
        """Every slot that can start a window at horizon ``n``.

        O(size * reach), so this is a DIAGNOSTIC and a test helper -- it must
        not be called on the sampling path. At the gin's capacity of 200k it
        is ~2M operations, and running it per sample for 200k gradient steps
        would dominate the whole run.
        """
        return np.array(
            [i for i in range(self.size) if self.valid_start(i, n)], dtype=np.int64
        )

    def can_sample(self, batch_size: int, n: int) -> bool:
        """Cheap readiness check: enough data, and some valid start near 0.

        Deliberately does not enumerate the valid set; it probes forward for
        the first valid start, which for a healthy buffer succeeds at once.
        """
        if self.size < max(batch_size, self._window_len(n)):
            return False
        return self._probe_forward(0, n) >= 0

    def _probe_forward(self, start: int, n: int) -> int:
        """First valid start at or after ``start``, scanning forward; -1 if none.

        Bounded by ``size``, and only reached when a prioritized draw lands on
        an invalid slot -- which at the real capacity means the handful of
        slots within one window of the cursor.
        """
        if self.size == 0:
            return -1
        for k in range(self.size):
            i = (start + k) % self.size
            if self.valid_start(i, n):
                return i
        return -1

    # ------------------------------------------------------------------
    # Observation reconstruction
    # ------------------------------------------------------------------
    def stacked_obs(self, index: int) -> np.ndarray:
        """Rebuild the ``frame_stack`` observation whose newest frame is ``index``.

        Frames before the episode's first are the episode's first, repeated --
        matching what the env's own reset does, and never reading into the
        previous episode.
        """
        ep = self.episode_ids[index]
        out = np.empty((self.frame_stack, *self.frame_shape), dtype=np.uint8)
        j = index
        for back in range(self.frame_stack):
            out[self.frame_stack - 1 - back] = self.frames[j]
            prev = (j - 1) % self.capacity
            # Step back only while the previous slot is the same episode AND
            # still live; otherwise hold the current frame.
            if self._is_live(prev) and self.episode_ids[prev] == ep:
                j = prev
        # (stack, C, H, W) -> (stack * C, H, W), oldest first, matching the
        # channel order the env's wrapper produces.
        return out.reshape(self.frame_stack * self.frame_shape[0], *self.frame_shape[1:])

    # ------------------------------------------------------------------
    # n-step return
    # ------------------------------------------------------------------
    def n_step(self, index: int, n: int, gamma: float) -> tuple[float, float, int, bool]:
        """Accumulate the n-step return from ``index``.

        Returns ``(return, discount, steps, done)``:
        ``return`` is sum_k gamma^k r_{t+k}, truncated at a terminal;
        ``discount`` is gamma^steps, the factor on the bootstrap;
        ``steps`` is how many transitions were actually consumed;
        ``done`` is True when a terminal was reached, so the bootstrap is zero.
        """
        ep = self.episode_ids[index]
        total = 0.0
        steps = 0
        done = False
        for k in range(n):
            j = (index + k) % self.capacity
            if not self._is_live(j) or self.episode_ids[j] != ep:
                break
            total += (gamma**k) * float(self.rewards[j])
            steps = k + 1
            if self.terminals[j]:
                done = True
                break
        return total, gamma**steps, steps, done

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def sample_indices(
        self, batch_size: int, n: int, rng: np.random.Generator, max_attempts: int = 16
    ) -> np.ndarray:
        """Draw ``batch_size`` valid start indices.

        Prioritized draws are stratified over the tree's total mass, as in
        Dopamine: the mass is cut into ``batch_size`` equal bands and one draw
        is taken per band, so a single huge priority cannot crowd out the rest
        of the batch.

        An invalid draw is retried within its band. Only the slots within one
        window of the cursor are persistently invalid -- eleven of 200k at the
        gin's capacity -- so retries are rare in a healthy buffer. If a band
        keeps failing, the draw walks forward to the next valid slot rather
        than falling back to a uniform pick: a uniform fallback silently
        replaces prioritized sampling with uniform sampling, which is a
        correctness change that no test would see.
        """
        if self.size == 0:
            raise RuntimeError("cannot sample an empty buffer")
        if self._tree is None or self._tree.total <= 0.0:
            out = np.empty(batch_size, dtype=np.int64)
            for b in range(batch_size):
                i = self._probe_forward(int(rng.integers(self.size)), n)
                if i < 0:
                    raise RuntimeError(
                        f"no valid windows at horizon n={n} (size={self.size})"
                    )
                out[b] = i
            return out

        out = np.empty(batch_size, dtype=np.int64)
        band = self._tree.total / batch_size
        for b in range(batch_size):
            picked = -1
            for _ in range(max_attempts):
                idx = self._tree.query(float(rng.uniform(b * band, (b + 1) * band)))
                if self.valid_start(idx, n):
                    picked = idx
                    break
            if picked < 0:
                picked = self._probe_forward(idx, n)
            if picked < 0:
                raise RuntimeError(
                    f"no valid windows at horizon n={n} (size={self.size})"
                )
            out[b] = picked
        return out

    def sampling_probabilities(self, indices: np.ndarray) -> np.ndarray:
        if self._tree is None or self._tree.total <= 0.0:
            return np.full(len(indices), 1.0 / max(self.size, 1), dtype=np.float64)
        total = self._tree.total
        return np.array([self._tree.get(int(i)) / total for i in indices])

    def loss_weights(self, indices: np.ndarray) -> np.ndarray:
        """Dopamine's importance weights: ``1/sqrt(p)``, normalized by the max.

        Dopamine's Rainbow does not use the usual beta-annealed form; it takes
        the reciprocal square root of the sampling probability and divides by
        the batch max so the largest weight is 1. Kept identical here (D-021).
        """
        if self._tree is None:
            return np.ones(len(indices), dtype=np.float64)
        probs = self.sampling_probabilities(indices)
        w = 1.0 / np.sqrt(probs + 1e-10)
        return w / w.max()

    def update_priorities(self, indices: np.ndarray, losses: np.ndarray) -> None:
        """Set priority to ``loss ** priority_exponent`` (0.5 = Dopamine's sqrt)."""
        if self._tree is None:
            return
        losses = np.abs(np.asarray(losses, dtype=np.float64)) + 1e-10
        prios = losses**self.cfg.priority_exponent
        self._tree.set_many(np.asarray(indices), prios)
        self._max_priority = max(self._max_priority, float(prios.max()))

    def sample(
        self,
        batch_size: int,
        n: int,
        gamma: float,
        rng: np.random.Generator,
        device: torch.device | str = "cpu",
    ) -> dict[str, torch.Tensor]:
        """One training batch.

        Shapes, with S = frame_stack * frame channels and J = jumps:
          obs            [B, S, H, W] uint8
          action         [B] int64
          n_step_return  [B] float32
          discount       [B] float32   gamma ** steps_taken
          next_obs       [B, S, H, W] uint8, the bootstrap state
          done           [B] bool, True when the return hit a terminal
          spr_actions    [B, J] int64
          spr_obs        [B, J, S, H, W] uint8, the SPR targets
          spr_mask       [B, J] bool, False past an episode end
          indices        [B] int64, for `update_priorities`
          weights        [B] float32
        """
        if n < 1:
            raise ValueError("n must be >= 1")
        idx = self.sample_indices(batch_size, n, rng)
        B, J = batch_size, self.jumps

        obs = np.empty((B, *self._obs_shape()), dtype=np.uint8)
        next_obs = np.empty((B, *self._obs_shape()), dtype=np.uint8)
        spr_obs = np.zeros((B, J, *self._obs_shape()), dtype=np.uint8)
        action = np.empty(B, dtype=np.int64)
        ret = np.empty(B, dtype=np.float32)
        disc = np.empty(B, dtype=np.float32)
        done = np.zeros(B, dtype=bool)
        spr_actions = np.zeros((B, J), dtype=np.int64)
        spr_mask = np.zeros((B, J), dtype=bool)

        for b, t in enumerate(idx):
            t = int(t)
            ep = self.episode_ids[t]
            obs[b] = self.stacked_obs(t)
            action[b] = self.actions[t]
            r, d, steps, is_done = self.n_step(t, n, gamma)
            ret[b], disc[b], done[b] = r, d, is_done
            # Bootstrap from the state `steps` later. When the episode ended
            # the bootstrap is masked by `done`, so the observation stored
            # here is never used -- the last valid frame keeps shapes right.
            # D-042: the official buffer bootstraps from t+n-1 with gamma^n.
            offset = steps - 1 if self.cfg.official_offby1_bootstrap else steps
            boot = (t + max(offset, 0)) % self.capacity
            if is_done or not self._is_live(boot) or self.episode_ids[boot] != ep:
                next_obs[b] = self.stacked_obs((t + max(steps - 1, 0)) % self.capacity)
            else:
                next_obs[b] = self.stacked_obs(boot)
            # SPR: the j-th target is the observation j+1 steps ahead, and the
            # action that led into it. Masked once the episode has ended.
            alive = True
            for j in range(J):
                cur = (t + j) % self.capacity
                nxt = (t + j + 1) % self.capacity
                if not alive or not self._is_live(cur) or self.episode_ids[cur] != ep:
                    break
                spr_actions[b, j] = self.actions[cur]
                if self.terminals[cur]:
                    alive = False
                    break
                if not self._is_live(nxt) or self.episode_ids[nxt] != ep:
                    break
                spr_obs[b, j] = self.stacked_obs(nxt)
                spr_mask[b, j] = True
        weights = self.loss_weights(idx)

        dev = torch.device(device)
        return {
            "obs": torch.from_numpy(obs).to(dev),
            "action": torch.from_numpy(action).to(dev),
            "n_step_return": torch.from_numpy(ret).to(dev),
            "discount": torch.from_numpy(disc).to(dev),
            "next_obs": torch.from_numpy(next_obs).to(dev),
            "done": torch.from_numpy(done).to(dev),
            "spr_actions": torch.from_numpy(spr_actions).to(dev),
            "spr_obs": torch.from_numpy(spr_obs).to(dev),
            "spr_mask": torch.from_numpy(spr_mask).to(dev),
            "indices": torch.from_numpy(idx).to(dev),
            "weights": torch.from_numpy(weights.astype(np.float32)).to(dev),
        }

    def _obs_shape(self) -> tuple[int, int, int]:
        c, h, w = self.frame_shape
        return (self.frame_stack * c, h, w)

    def nbytes(self) -> int:
        """Bytes the ring actually occupies, for the run record."""
        return int(
            self.frames.nbytes
            + self.actions.nbytes
            + self.rewards.nbytes
            + self.terminals.nbytes
            + self.episode_ids.nbytes
        )

    def __len__(self) -> int:
        return self.size
