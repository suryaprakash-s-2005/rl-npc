# FINAL RESULTS SUMMARY (pre-audit, historical)

> **Superseded.** This is the original positive-result record. The audit did not
> reproduce the claimed +0.25 benefit under the corrected protocol; see
> `results/AUDIT_SUMMARY.md` and `paper/revised_draft.pdf`.

Project: Strategy-Transition-Aware Experience Sharing for Personalized
Reinforcement Learning NPCs (ICAIDIET'26)

## Setup (combat redesign, within-episode personality switch)

- 2D pursuit/combat env (Gymnasium-style), double-DQN NPC pursuer/killer.
- HP changed 10 -> 20 so a kill is slow enough for a mid-episode personality
  transition to reliably occur.
- 4 player strategies (combat param profiles): aggressive, defensive, evasive,
  balanced.
- Within-episode strategy switch: each episode starts under one personality and
  switches once at `switch_step = 100` to a paired partner personality
  (`(L[i], L[(i+1)%4])`). This is the core "strategy-transition" protocol the
  NPC's experience sharing must cope with.
- Domain randomization: `params_jitter = 0.2` perturbs the player's combat
  params each reset so the NPC cannot memorize a fixed opponent.
- Replay variants:
    - single       : one shared replay buffer (baseline)
    - partitioned  : per-strategy replay partitions (strategy-aware sharing)
- Tagging / conditioning:
    - strategy_source: groundtruth (gt) or online detector (det)
    - cond_on_strategy: whether the policy conditions on the strategy label
      (on) or must infer it from dynamics (off)
- Detector: temporal MLP (retrained on HP-20 telemetry, 91.5% val acc) + CUSUM
  transition detector.
- Seeds: gt/cond-on n=3; gt/cond-off n=6; det n=6. Eval 10-12 episodes/seed with
  mid-episode switch.
- Metric: win_rate = fraction of eval episodes NPC kills player (adaptive win);
  adapt_time = steps after switch until performance recovers.

## Grid results (mean over seeds)

| condition     | variant     | n | win_rate | adapt_time |
|---------------|---|---:|---:|---:|
| GT / cond-on  | single      | 3 | 0.944 | 32.4 |
| GT / cond-on  | partitioned | 3 | 0.972 | 40.0 |
| GT / cond-off | single      | 6 | 0.708 | 27.8 |
| GT / cond-off | partitioned | 6 | 0.958 | 26.8 |
| DET / cond-on | single      | 6 | 0.908 | 33.3 |
| DET / cond-on | partitioned | 6 | 0.750 | 25.2 |
| DET / cond-off| single      | 6 | 0.800 | 28.8 |
| DET / cond-off| partitioned | 6 | 0.717 | 24.5 |

## Headline findings

1. **Ground truth, no label in obs (GT/cond-off) is the clean headline.**
   Strategy-partitioned replay raises adaptive win rate from **0.708 -> 0.958**
   (n=6, +0.25) with comparable adaptation time. This is the core contribution:
   when the NPC must infer the player's personality from dynamics, sharing
   experience within strategy partitions materially improves how quickly/well
   it adapts to a mid-episode personality transition.

2. **Ground truth, label given (GT/cond-on):** partitioned (0.972) >= single
   (0.944). With the exact label in the observation the advantage of shared
   partitions is small, as expected -- both variants adapt.

3. **Detector mode (DET) does NOT reproduce the partitioned benefit.** Over
   n=6 seeds, single >= partitioned in both conditions:
   - cond-on: 0.908 vs 0.750
   - cond-off: 0.800 vs 0.717
   When replay partitions are tagged by the *imperfect online detector* (91%
   acc, CUSUM warm-up latency), mis-attributed experience contaminates each
   partition, eroding -- and in cond-on exceeding -- the sharing benefit. This
   is an honest limitation: strategy-aware sharing helps most when partition
   labels are reliable.

## Interpretation for the author's report

- The mechanism is validated end-to-end in the clean setting (GT/off): the
  paper's central claim "strategy-transition-aware experience sharing improves
  adaptive NPC win rate under mid-episode personality switching" is supported
  (+0.25 win rate, consistent across 6 seeds).
- Detector integration is functional but shows that imperfect labels reduce the
  advantage; this motivates confidence-gated / soft partition assignment as a
  clear future-work direction and should be framed as such (not hidden).

## Reproduce

    python -m src.strategy.train_detector          # HP-20 telemetry -> data/detector.pt
    python experiments/run_grid.py                  # GT on/off (seeds 0-2), DET on (seeds 0-2)
    python experiments/run_det_ext.py               # DET cond-on/off seeds 3-5
    python experiments/run_gt_off_ext.py            # GT cond-off seeds 3-5
    python experiments/plot_final.py                # results/final_win_adapt.png + grid_summary.md

## Artifacts

- results/grid_full.json, grid_gt_cond_off.json, grid_det_cond_on.json,
  grid_det_cond_off.json (per-run rows)
- results/final_win_adapt.png (grouped bar chart: win rate + adaptation time)
- results/grid_summary.md (summary table)
- results/train_{single,partitioned}-{gt,det}-seed{...}.json (per-run history)
