"""Fast-weight lens encoder: state contract, unroll/step equivalence, resets,
and the store/retrieve behaviour the design rests on."""
import pytest
import torch

from playtrain_trainers.impala.lens import LENS_VARIANTS, LensEncoder, lens_scan
from playtrain_trainers.impala.net import ImpalaNet

OBS = (3, 64, 64)


def _inputs(T, B, done=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    frame = torch.randint(0, 256, (T, B, *OBS), dtype=torch.uint8, generator=g)
    if done is None:
        done = torch.zeros(T, B, dtype=torch.bool)
    return {"frame": frame, "reward": torch.zeros(T, B), "done": done,
            "last_action": torch.zeros(T, B, dtype=torch.int64)}


def _net(net="impala_lens"):
    torch.manual_seed(0)
    m = ImpalaNet(OBS, 8, use_lstm=True, net=net).eval()
    # Open the write gate so the test exercises real writes, not beta ~ 0.1.
    if net in LENS_VARIANTS:
        torch.nn.init.constant_(m.encoder.write_beta.bias, 1.0)
    return m


def test_state_contract():
    m = _net()
    s = m.initial_state(5)
    assert [tuple(t.shape) for t in s] == [(1, 5, 256), (1, 5, 256), (1, 5, 16 * 32), (1, 5, 32 * 16 * 16)]
    out, s2 = m(_inputs(3, 5), s)
    assert out["policy_logits"].shape == (3, 5, 8)
    assert [t.shape for t in s2] == [t.shape for t in s]
    assert len(_net("impala").initial_state(2)) == 2      # the baseline is untouched


def test_lens_needs_lstm():
    with pytest.raises(ValueError):
        ImpalaNet(OBS, 8, use_lstm=False, net="impala_lens")


def test_unroll_equals_single_steps():
    """The learner replays an unroll in one call; the actor steps one frame at a
    time carrying the state. Both must produce the same logits and state."""
    T, B = 7, 3
    done = torch.zeros(T, B, dtype=torch.bool)
    done[0, 1] = done[3, 0] = done[5, 2] = True
    inp = _inputs(T, B, done)
    m = _net()
    with torch.no_grad():
        full, s_full = m(inp, m.initial_state(B))
        s = m.initial_state(B)
        steps = []
        for t in range(T):
            o, s = m({k: v[t:t + 1] for k, v in inp.items()}, s)
            steps.append(o["policy_logits"])
    assert torch.allclose(full["policy_logits"], torch.cat(steps), atol=1e-5)
    for a, b in zip(s_full, s):
        assert torch.allclose(a, b, atol=1e-5)
    assert s_full[2].abs().sum() > 0                       # something was written


def test_done_resets_memory():
    """A frame flagged done starts an episode: what came before must not matter."""
    m = _net()
    B = 2
    with torch.no_grad():
        _, dirty = m(_inputs(4, B, seed=1), m.initial_state(B))
        first = _inputs(1, B, torch.ones(1, B, dtype=torch.bool), seed=2)
        a, sa = m(first, dirty)
        b, sb = m(first, m.initial_state(B))
    assert torch.allclose(a["policy_logits"], b["policy_logits"], atol=1e-6)
    for x, y in zip(sa, sb):
        assert torch.allclose(x, y, atol=1e-6)


def test_nowrite_reads_nothing():
    m = _net("impala_lens_nowrite")
    with torch.no_grad():
        _, s = m(_inputs(4, 2), m.initial_state(2))
    assert s[2].abs().sum() == 0 and m.encoder.last_read == 0


def test_scan_stores_and_retrieves_by_similarity():
    """Bind two orthogonal keys to two values, then read. A key never written
    reads zero; a key similar to a written one reads its value scaled by the
    similarity. This is the whole generalisation argument: the read sees a
    sprite only through its similarity to keys written this episode."""
    K, R = 8, 4
    e = torch.eye(K)
    v1, v2 = torch.tensor([1., 0, 0, 0]), torch.tensor([0., 1, 0, 0])
    kw = torch.stack([e[0], e[1]]).view(2, 1, 1, K)
    v = torch.stack([v1, v2]).view(2, 1, 1, R)
    M = lens_scan(torch.zeros(1, R, K), kw, v, torch.ones(2, 1, 1), torch.ones(2, 1))[-1, 0]
    assert torch.allclose(M @ e[0], v1) and torch.allclose(M @ e[1], v2)
    assert torch.allclose(M @ e[5], torch.zeros(R))
    near = torch.nn.functional.normalize(e[0] + 0.5 * e[5], dim=0)
    assert torch.allclose(M @ near, v1 * near[0])
    # Rebinding the same key overwrites (delta rule with a unit key, beta 1).
    M2 = lens_scan(M.unsqueeze(0), e[0].view(1, 1, 1, K), v2.view(1, 1, 1, R),
                   torch.ones(1, 1, 1), torch.ones(1, 1))[-1, 0]
    assert torch.allclose(M2 @ e[0], v2, atol=1e-6)


def test_gradient_reaches_keys_and_write_heads():
    m = _net().train()
    out, _ = m(_inputs(5, 2), m.initial_state(2))
    out["policy_logits"][-1].sum().backward()
    enc = m.encoder
    for name in ("key_stem.0.weight", "write_attn.weight", "write_value.weight", "write_beta.weight"):
        g = enc.get_parameter(name).grad
        assert g is not None and g.abs().sum() > 0, name


def test_frozen_keys_have_no_grad():
    enc = LensEncoder(freeze_keys=True)
    assert not any(p.requires_grad for p in enc.key_stem.parameters())


def test_keys_separate_different_content():
    """Two different patches on a black frame must not share a key; the same
    patch at two places must. Plain unit-norm keys fail the first half."""
    torch.manual_seed(0)
    enc = LensEncoder().eval()
    x = torch.zeros(1, 3, 64, 64, dtype=torch.uint8)
    a = torch.randint(0, 256, (3, 8, 8), dtype=torch.uint8)
    b = torch.randint(0, 256, (3, 8, 8), dtype=torch.uint8)
    x[0, :, 8:16, 8:16], x[0, :, 40:48, 16:24], x[0, :, 24:32, 48:56] = a, a, b
    with torch.no_grad():
        k = enc.normalise_keys(enc.key_stem(x.float() / 255))[0]
    cos = lambda p, q: torch.nn.functional.cosine_similarity(k[:, p[0], p[1]], k[:, q[0], q[1]], dim=0)
    assert cos((2, 2), (10, 4)) > 0.95          # same patch, two places
    assert cos((2, 2), (6, 12)) < 0.8           # different patches
    assert k.norm(dim=0).max() < 1.0


def test_hard_write_stores_one_location():
    """The hard variant writes the key of a single location, never a blend."""
    torch.manual_seed(0)
    enc = LensEncoder(hard_write=True).eval()
    x = torch.randint(0, 256, (2, 3, 64, 64), dtype=torch.uint8).float() / 255
    with torch.no_grad():
        d = enc.s1(enc.s0(x))
        kc = enc.normalise_keys(enc.key_stem(x))
        kw, _, _ = enc.proposals(d, kc, kc)      # previous map = current: every candidate is a real key
    unit = torch.nn.functional.normalize(kc.flatten(2), dim=1)        # [N, K, n]
    best = torch.einsum("nhk,nks->nhs", kw, unit).max(-1).values
    assert torch.allclose(best, torch.ones_like(best), atol=1e-5)


def test_hard_write_unroll_equals_steps_and_has_grad():
    T, B = 6, 2
    done = torch.zeros(T, B, dtype=torch.bool); done[2, 1] = True
    inp = _inputs(T, B, done)
    m = _net("impala_lens_hard")
    with torch.no_grad():
        full, s_full = m(inp, m.initial_state(B))
        s, steps = m.initial_state(B), []
        for t in range(T):
            o, s = m({k: v[t:t + 1] for k, v in inp.items()}, s)
            steps.append(o["policy_logits"])
    assert torch.allclose(full["policy_logits"], torch.cat(steps), atol=1e-5)
    m.train()
    out, _ = m(inp, m.initial_state(B))
    out["policy_logits"][-1].sum().backward()
    assert m.encoder.write_attn.weight.grad.abs().sum() > 0
