"""The PlayTrain random baseline for one game (D-008, D-044).

The paper's Frostbite floor is 65.2 on ALE. This measures the same quantity on
a PlayTrain game: 100 episodes of a uniform-random policy over the declared 8
actions, under the identical wrapper stack the agent will see. It is what
`score_over_random` is measured from, and for frostbite what PROTOCOL
section 5 B's normalized progress is measured from.

The floor is per GAME, not per backend (D-044), so each game needs its own run
and writes to `results/bbf/random_<game>/`.

Run: uv run --no-sync python tools/bbf_random_baseline.py --game venture
"""
from __future__ import annotations

import argparse
from pathlib import Path

from playtrain_trainers.bbf.config import BBFConfig, results_dir
from playtrain_trainers.bbf.evaluate import (
    evaluate_policy,
    policy_rng,
    random_policy,
    write_metrics,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0, help="policy RNG seed")
    ap.add_argument("--game", default="frostbite")
    ap.add_argument("--run-id", default=None,
                    help="defaults to random_<game>, which is where "
                         "`measured_random_baseline` looks for it")
    args = ap.parse_args()

    run_id = args.run_id or f"random_{args.game}"
    cfg = BBFConfig(run_id=run_id, seed=args.seed, game=args.game,
                    num_eval_episodes=args.episodes)
    cfg.validate()

    # One generator per episode, derived from the episode's game seed, so any
    # single episode can be replayed on its own (tools/bbf_filmstrip.py).
    episodes, summary = evaluate_policy(
        cfg,
        lambda env, s: random_policy(env.action_space.n, policy_rng(cfg, s)),
        random_mean=None,
        progress_every=25,
    )

    out = results_dir(cfg)
    payload = write_metrics(
        out / "metrics.json",
        cfg,
        episodes,
        summary,
        unit="U03",
        extra={
            "note": (
                f"D-008: the PlayTrain {args.game} floor, the analogue of the "
                "paper's ALE random column. Uniform over the 8 declared actions. "
                "On frostbite note F-008: only 5 are distinguishable, so "
                "LEFT/RIGHT/NOOP each carry 2/8 of the mass and UP/DOWN 1/8. "
                "venture, amidar and star_gunner all read SPACE, so for them all "
                "8 are distinguishable and the draw is uniform over effects too."
            ),
        },
    )
    _report(payload["summary"], episodes, out)


def _report(s: dict, episodes, out: Path) -> None:
    print("\n" + "=" * 62)
    print("U03  PlayTrain frostbite, uniform-random over 8 actions (D-008)")
    print("=" * 62)
    rows = [
        ("episodes", f"{s['episodes']}"),
        ("score mean", f"{s['score_mean']:.2f}"),
        ("score 95% CI (bootstrap)", f"[{s['score_ci95'][0]:.2f}, {s['score_ci95'][1]:.2f}]"),
        ("score std / sem", f"{s['score_std']:.2f} / {s['score_sem']:.2f}"),
        ("score min / median / max", f"{s['score_min']:.0f} / {s['score_median']:.0f} / {s['score_max']:.0f}"),
        ("agent steps (mean)", f"{s['agent_steps_mean']:.1f}"),
        ("game frames (mean)", f"{s['game_frames_mean']:.1f}"),
        ("lives left (mean)", f"{s['lives_left_mean']:.2f}"),
        ("terminal states", ", ".join(f"{k}={v}" for k, v in s["game_states"].items())),
        ("truncated at cap", f"{s['truncated']}"),
    ]
    # Only for games whose profile says the field means something (D-044).
    if "win_rate" in s:
        rows.insert(3, ("win rate", f"{s['win_rate']:.3f}  ({s['wins']}/{s['episodes']})"))
    if "floes_visited_mean" in s:
        rows.insert(4, ("floes visited (mean)", f"{s['floes_visited_mean']:.2f} / 16"))
    if "score_ceiling" in s:
        rows.append(("score ceiling", f"{s['score_ceiling']:.0f}"))
    w = max(len(k) for k, _ in rows)
    for k, v in rows:
        print(f"  {k:<{w}} : {v}")
    hist: dict[float, int] = {}
    for e in episodes:
        hist[e.score] = hist.get(e.score, 0) + 1
    print("\n  score histogram (raw score -> episodes):")
    for score in sorted(hist):
        print(f"    {score:>5.0f} : {'#' * hist[score]} ({hist[score]})")
    print(f"\n  written: {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
