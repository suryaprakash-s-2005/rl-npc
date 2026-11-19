# 2nd Review Prep — 16/09/2026, VCET Madurai

Deck used: `D:\IEEE\2nd-review-ppt.pptx` (15 slides)
Paper: `D:\IEEE\..._Partitioned_Experience_Replay...pdf` (finalized, ready for submission)

> IMPORTANT: The PPT was updated from "PPO / Actor-Critic / Stable-Baselines3" to
> **Double DQN** (slides 4, 6, 8) to match the paper and code. The patched file is at
> `C:\Users\THIS-PC\AppData\Local\Temp\opencode\rev2_dqn.pptx`; the original PPT is still
> locked open in PowerPoint. See Step 4 below on swapping it in.

---

## 1. Thirty-second opening pitch

"Reinforcement-learning NPCs learn to fight, but the moment a player switches strategy
mid-encounter they degrade — a single replay buffer keeps sampling the *old* player's
experience, which we call **strategy interference**. Our framework first *recognizes* the
player's strategy from gameplay telemetry, *detects* transitions with a confirm-gated CUSUM,
then *stores experience in per-strategy partitions* that are sampled proportionally to how
similar the strategy is to the one we currently detect. Across 6 seeds, under the hardest
setting (the NPC must infer the style purely from dynamics), partitioned experience sharing
lifts the adaptive win rate from **0.708 to 0.958** — plus 0.25 absolute — with the *same*
adaptation time. And we report honestly where it fails: when partition labels come from the
imperfect online detector, the benefit shrinks and reverses, which isolates label reliability
as the crux of the mechanism and our next iteration."

---

## 2. Talk script, slide by slide

**Slide 1 — Title**
- Team names + roll numbers, supervisor, "Second Review, VCET Madurai". (15 sec, don't linger.)

**Slide 2 — Contents**
- "Ten parts: problem, objectives, literature gap, methodology, architecture, modules, the
  algorithm & math, implementation + live demo, experimental results, and paper status."
  Highlight that today you'll go deeper on the math (slide 9) than at the 1st review and that
  the demo is live.

**Slide 3 — Problem Statement & Motivation**
- "An RL NPC trained with a vanilla replay buffer is fine against a stable opponent, but a
  player switches tactics mid-fight. One buffer mixes aggressive and defensive experience,
  which interferes: the NPC adapts slowly or not at all."
- Quote the 3 challenges verbatim: **Recognize, Detect, Adapt** — "recognize the current
  style from noisy telemetry, detect when it changes, adapt fast afterwards."
- Possible judge probe: "why is this a real failure?" → "negative transfer domes from the
  old policy/behavior; it's exactly what a uniform replay buffer cannot structure."

**Slide 4 — Objectives & Motivation**
- Read the 6 objectives briefly; stress the last two: "integrate with a Double DQN NPC" and
  "evaluate under both ground-truth and detected labels." Mention the paper's *full-factorial*
  protocol (2×2×2: replay × label source × observation) — sets up rigor later.

**Slide 5 — Literature Review**
- Point at the table: "Lee 2025 learns human strategy but not mid-fight switches; Ayoub 2025
  assumes fixed behaviour; Zheng 2024 reviews DRL-DDA but has no strategy-aware replay;
  Hare & Tang share experience but don't separate it by strategy."
- Research gap line: "no existing method makes the replay buffer itself strategy-aware."
- If asked about Sestini (roguelike design-parameter adaptation): that's in the paper's
  related work — "we address the *opponent behaviour* distribution, not the environment design."

**Slide 6 — Proposed Methodology**
- Read the pipeline line: "detect → partition → prioritize relevant experience → adapt the
  NPC using Double DQN." Name the four strategies: aggressive, defensive, evasive, balanced.

**Slide 7 — Architecture / Workflow**
- Describe the runtime loop: game telemetry → 10-feature sliding window → MLP classifier →
  confirm-gated CUSUM → current detected strategy → partition weights (Eq. 3) → Double DQN
  samples batch → policy action. "Everything upstream of the RL agent is light — classifier is
  a 2×64 MLP, CUSUM is a running sum — so this runs in real time."

**Slide 8 — Module-wise Implementation**
- "All five modules are complete." Read the list; explicitly say module 4 is
  "Double DQN + reward shaping + soft target update" and stack is PyTorch-based custom agent
  (not SB3). Emphasize 6-seed evaluation done (module 5).

**Slide 9 — Strategy-Transition Algorithm & Mathematical Model**
- This is the slide judges will probe. Rehearse these three blocks:
  1. Classifier: "10 telemetry features over a W=15-step window, context of 8 windows = an
     80-dim input, MLP → 4-class softmax. Trained on 29,920 synthesized windows (7,480 per
     strategy), 92.7% validation accuracy."
  2. CUSUM (Eq. 2): "binary change indicator **I_t = 1 if the current label differs from the
     active segment label**, and **S_t = max(0, S_{t-1} + I_t − k)** with k=0.5. It only
     accumulates on disagreement; a stable label gives I_t=0 so S decays to 0 — no drift.
     Alarm at h=6.0, then we require the new label to persist **confirm = 10** consecutive
     steps before we actually switch the partition." -> "10 steps at dt=0.1s is 1 second."
  3. Partition weights (Eq. 3): "**w_j(c) = 1 if j is the detected strategy**, otherwise
     **δ · s_cj^(−p)** where s_cj is the cosine similarity between the canonical persona
     parameter vectors, **δ=0.5** is the stale floor, **p=1.6** is the prior power. We then
     sample batch entries multinomially across partitions (Algorithm 1)."
- If they ask "why similarity-weighted and not hard cut-off?": "so related strategies transfer
  smoothly and we never starve a partition to zero once it has data."

**Slide 10 — Implementation Results (win rate, confusion matrix, CUSUM trace)**
- Confusion matrix: "92.7% overall. Aggressive is essentially separable (precision 0.989);
  residual errors run along the evasive/defensive/balanced midline, which is expected — they
  differ mainly in distance-keeping parameters."
- CUSUM traces: "the S statistic stays quiet while the label is stable, arms only right after
  the style switch, and the partition switch is emitted after the 10-step confirmation."

**Slide 11 — Detector fidelity / training rewards**
- Detector fidelity (Table IV / Fig. 3): "every evaluation episode has exactly one true
  switch, yet the deployed pipeline reports 1.23 transitions/episode (single) vs 1.63
  (partitioned). Spurious confirmations fracture the memory — this is *why* DET labels erode
  the benefit, and it frames our future work."
- Training rewards (Fig. 6): "under ground-truth labels, partitioned runs climb to a higher,
  steadier plateau; under detector labels the two curves converge — mirrors the win-rate
  reversal." -> this sentence is your bridge if asked "so does it always help?"

**Slide 12 — Live Arena**
- One-liner: "a playable 2D melee arena — the same 20×20 gym environment and the same
  trained Double DQN policy + detector, with combat re-tuned so a human can survive hits
  (60 HP, 3 damage, relaxed cooldown, asymmetric reach). The research numbers come from the
  controlled HP-20 evaluation." Then go to the live demo (Section 4 below).

**Slide 13 — Expected Outcomes**
- Read the bullets; land on the headline: "target improvement 0.708 → 0.958 in the cleanest
  setting — already achieved and reported in the paper."

**Slide 14 — Paper Status**
- "Manuscript finalized, ready for submission; venues shortlisted (IEEE AICSSYC, TEMSMET,
  ICICSCS), final venue to be locked." Be ready with the *justification* line (AI/RL relevance,
  scope alignment). Mention submission deadline awareness (e.g., ICAIDIET was the 1st-review
  target; venues updated for 2nd review).

**Slide 15 — References**
- Don't read; "citations available offline." If challenged on list alignment: the PPT refs and
  paper refs differ slightly because they were assembled at different stages — offer to align
  them now, or just say the paper carries the canonical list.

---

## 3. Q&A dossier

### A. Concept & novelty

**Q: What is new versus a plain replay buffer or Prioritized Experience Replay (PER)?**
A: PER re-ranks *individual transitions* by TD-error; it is agnostic to the opponent's
distribution. We partition by the *player's strategy* — actually by the strategy we *currently
detect* — and damp stale partitions via a parameter-similarity prior (Eq. 3). The two are
orthogonal; we could add PER *inside* partitions later. The paper's control is a single shared
buffer (with expert warm-up), so the +0.25 is attributable to the partitioning itself.

**Q: What is "negative transfer / strategy interference" concretely here?**
A: After the player switches aggressive→defensive, the buffer still mostly stores aggressive
experience; every gradient batch re-trains the policy on the wrong behaviour, diluting the
new style. In numbers: a single buffer under GT/cond-off wins 0.708 while the partitioned
memory wins 0.958 — 35% relative improvement.

**Q: Why not just hand the NPC the label (cond-on) and move on?**
A: Two reasons. (1) In a real game the label is hidden — the developer doesn't know the
player's "style"; that's the GT/cond-off condition, the demanding one. (2) Our own
cond-on result (0.972 vs 0.944) shows partitioning still helps a little there, so it's not
an artifact of a weak baseline; it works by *disambiguation*.

**Q: Is "strategy" just difficulty tuning (DDA)?**
A: DDA tunes one scalar difficulty signal. We ground adaptation in the player's *tactical
profile* — an 8-parameter persona vector and a 4-class behavioural label — and adapt the
NPC's own learned policy, not just a difficulty knob.

### B. Method / math

**Q: Walk us through the CUSUM — why k=0.5, h=6.0, confirm=10?**
A: I_t is 1 only when the smoothed classifier label disagrees with the active segment.
S accumulates a *drift* I_t − k, so it rises only during sustained disagreement and decays to
0 under a stable label (no false-alarm drift). h=6.0 with k=0.5 means roughly a dozen
disagreeing frames to alarm; then confirm=10 consecutive steps (1 s at dt=0.1) guarantees we
only switch partitions on a genuine switch, suppressing the residual 7% classifier error.

**Q: Why does the demo look like it reacts in ~0.5 s when the paper says confirm=10 steps?**
A: 10 steps × dt 0.1 s = 1 s nominal. The live front-end additionally blends an adaptive
"soft-lock" on classifier confidence so the plan badge updates almost immediately for the
eye; the *published* adaptation numbers are from the CUSUM-gated pipeline, which is
conservative by design.

**Q: Explain Eq. (3) and Algorithm 1 — why δ=0.5 and p=1.6?**
A: The detected strategy's partition always gets weight 1. A non-current partition j is
down-weighted by δ·s_cj^(−p): s_cj is the cosine similarity between the canonical persona
θ-vectors c and j (identical = 1.0), p>1 sharpens the decay, and δ is a floor so no partition
ever receives zero once it has data. Batch sizes n_j are drawn multinomially, then topped up
from the current partition if a partition is empty. δ=0.5 and p=1.6 came from a small sweep —
aggressive best immediately usable values — and the per-seed spread shows it is not fragile.

**Q: What 10 features / how big the training set?**
A: Approach frequency, mean engagement distance, retreat frequency, retreat-vs-approach ratio,
normalized longest engagement run, sword-swing frequency, hit rate, normalized damage dealt,
fraction of steps inside sword range, and low-health retreat ratio — over W=15 steps. Context
of 8 consecutive windows = an 80-dim input to a 2×64 ReLU MLP. Trained on 29,920 windows
(7,480 per persona) synthesized by rolling out the HP-20 personas; 92.7% held-out accuracy.

**Q: Why Double DQN rather than PPO?**
A: Double DQN is off-policy, which is what makes experience replay meaningful in the first
place — the whole mechanism *is* the replay structure. It also needs only a tiny network
(2×128) and trains fast on CPU, so a 6-seed factorial is cheap. PPO was dropped early for
exactly the same reason PER/partitioning matter: on-policy learners throw away old
experience, so there's no buffer artefact to fix. (The PPT's earlier "PPO" wording was a
stale slide; it now reads Double DQN everywhere.)

**Q: Reward design — isn't it heavily shaped?**
A: Sparse terminal (+150 / −50) plus interpretable shaping: ±8 per hit, +0.6 heading, +1.2
closing per unit, +1.0 in-range, −0.005 per step. Same shaping in *every* condition, so it
cannot create the partitioned-vs-single gap. Expert demonstrations (30 greedy chasers per
strategy) pre-fill every partition so no partition is starved at the start.

**Q: What is "adaptation time", exactly?**
A: The number of steps *after* the mid-episode switch until the NPC's short-window
performance recovers to its pre-switch level. In GT/cond-off, partitioned = 26.8 steps vs
single = 27.8 — statistically a tie, which is why we frame the win as "same adaptation speed,
much higher win rate."

### C. Results & trustworthiness

**Q: Why does partitioning help only under ground-truth labels?**
A: Because the mechanism's value is proportional to label reliability. Under DET, the CUSUM
needs warm-up (segment seeding + confirmation), so the first stretch of a new persona is often
tagged wrong; misattributed transitions land in the wrong partition with full weight, and the
confirmation latency delays the moment the sampling distribution actually moves. Table IV
quantifies it: 1.23–1.63 detector-reported transitions per episode where there is exactly one
true switch.

**Q: 92.7% classifier accuracy seems good — why do DET results reverse then?**
A: The error is not uniform: it lies on the evasive/defensive/balanced midline, and a single
mislabel during warm-up corrupts an entire partition segment, which then stings repeatedly
through sampling. 92.7% per-frame is much weaker than "clean partition labels", and the
failure concentrates at the exact moment — right after a switch — when the agent must be right.

**Q: Only 6 seeds — is that convincing?**
A: n=6 per cell (n=3 for GT/cond-on), each a fresh training run + re-eval over 10–12 held-out
episodes. Two looking points: (1) the GT/cond-off gain is consistent across *all six* seeds;
(2) per-seed spread — partitioned wins cluster in [0.83, 1.00] while single spans
[0.33, 1.00] — so raising the mean didn't come from outlier collapse. CI on a 6-seed mean is
wide; we report it honestly and rely on the consistency + spread argument.

**Q: Why HP=20, switch at step 100, 400-step episodes?**
A: 20 HP keeps the episode long enough that a mid-episode switch is observable before a fight
ends; switching once per episode at step 100 forces the policy to *track a transition* rather
than memorize a fixed opponent; 400-step cap bounds episode length so eval is reproducible.
Under HP-20/dt=0.1 the fight dynamics match the demo world, just with the *research* melee
numbers (range 1.6, cooldown 4, 2 dmg).

**Q: Compute/time for the whole grid?**
A: Double DQN with a 2×128 net on CPU; 400 episodes × ≤400 steps, ~a few minutes per seed,
so the 2×2×2 grid (seeds n=3/6) is hours, not days. Everything is CPU-trainable — a selling
point for undergraduate/game-studio feasibility.

### D. System & real-deployment

**Q: Is the demo environment the same as the paper's?**
A: The live demo is a *playable front-end* over the same 20×20 gym environment, the same
trained policy, detector, and CUSUM. Combat was re-tuned for a human to survive a showcase
(60 HP, 3 damage per hit, cooldown 6, attack range 1.0 with a slight human-reach bonus), and
it pre-loads the cond-on model so the plan badge is legible. All *published* numbers come from
the controlled HP-20 factorial. (Say this before the demo so the HP-20 chart and HP-60 screen
never collide.)

**Q: What's in the 14-dim observation?**
A: Normalized relative position, distance, the player's velocity and heading, closing rate,
both healths, both attack cooldowns, both speeds, and a swing-in-range flag. cond-on appends a
4-d one-hot of the active strategy (18-d); cond-off withholds it — that's the demanding case.

**Q: Will this work with an actual human and unseen strategies?**
A: The four personas are hand-authored prototypical styles; a real player won't be a pure
canonical persona. That's explicitly our future work (iii): discover personas online from
human play and evaluate perceived adaptation, not just win rate. The detection pipeline and
partitioned memory transfer as-is; only the persona inventory needs to become data-driven.

**Q: What are the three future-work lines?**
A: (i) confidence-gated sharing — soft, posterior-based partition assignment and gating the
buffer re-weighting on detector confidence; (ii) rehearsal-based warm-up to shorten the
detector's cold start; (iii) online persona discovery + human-player evaluation.

### E. Prior-review follow-ups (in case they reference the 1st review)

- 1st review asked about experience-sharing novelty and validation plan. Answer: the 1st
  review promised a switch-awareness evaluation protocol; this deliverable *is* that — a
  full-factorial 6-seed protocol with GT/DET and label on/off, plus per-seed spread and
  training dynamics, matching phase 8–10 of the timeline.
- The 1st-review deck mentioned "Hierarchical RL" in the timeline; that was superseded by the
  Double DQN + partitioned memory design — clarify if asked: "HRL was dropped; the final
  design is a single off-policy agent with a strategy-structured replay."

---

## 4. Demo run script (live, ~90 seconds)

### Pre-flight (do this morning in the review room)
```
cd /d D:\IEEE\project2
D:\torch_venv\Scripts\python.exe demo\smoke_test.py
```
Expect rc=0. If the window can't open (remote session), fall back to:
```
D:\torch_venv\Scripts\python.exe demo\fight.py --frames 300 --shot shot_live.png
```
Model files: `data\demo_npc_partitioned.pt` (used by default) or fallback
`data\npc_partitioned-gt-seed0.pt` — the demo auto-picks the fallback if needed.

### Choreography
Launch: `D:\torch_venv\Scripts\python.exe demo\fight.py` (run from `D:\IEEE\project2`).

- **0:00–0:10 — intro.** "This is the live arena: red is you, blue is the trained
  strategy-aware Double DQN NPC. The world is the same 20×20 gym environment from the paper."
- **0:10–0:25 — play yourself.** Move with WASD; swing with SPACE (auto-aimed). Note the
  **press-per-swing & amber reload arc** ("one clean swing every 0.6 s — cadence is visible,
  no mashing"). Touch the NPC once so it reacts, then back off.
- **0:25–0:50 — watch mode (press T).** "Now the player goes autopilot and the personality
  auto-switches at sim-step 100: aggressive → defensive → evasive → balanced." Point at the
  screen as each plan flips: **KITE → PRESS → HUNT → SWAY** (floating badge + "ADAPTS TO
  <style>" banner subtitle + chevrons: retreat arrows for KITE, chase arrows for HUNT/PRESS).
  Mention the top detection strip: live classifier confidence + CUSUM switch markers.
- **0:50–1:10 — force a switch (keys 1–4).** "I'll force one myself: press **2** (defensive)
  → the NPC should flip to PRESS within about a second; press **3** (evasive) → HUNT."
  Mix keys 1–4 to show it tracks *you*, not the clock.
- **1:10–1:20 — style-test menu (press M).** Pick **AGGRESSIVE**, play a few seconds of
  rushing: NPC KITEs (backs off, slices at range). Then **M → EVASIVE**: you run, NPC HUNTs.
- **1:20–1:30 — exit.** Q (or M → close). "Everything you saw — plan commit, chevrons, the
  detection timeline — is the *scoreboard* for the paper's adaptation mechanism; the win-rate
  and adaptation-time numbers are the controlled evaluation in Table III."

### Talking points while the demo runs
- Each screen element = one paper component: badge = detected plan (Eq. 3 priority),
  chevrons = policy intent, detection strip = classifier + CUSUM (Eq. 2), banner = the
  transition being acted on.
- Rehearse the honest caveat sentence (Section 3.D above) so HP-60 screen vs HP-20 charts
  never seem contradictory.

---

## 5. Final notes

1. **Swap the PPT** — original `D:\IEEE\2nd-review-ppt.pptx` is still open in PowerPoint;
   close it, then copy `C:\Users\THIS-PC\AppData\Local\Temp\opencode\rev2_dqn.pptx` over it,
   or open that patched file directly. Backup kept at `2nd-review-ppt-ORIG-ppofix.pptx.bak`.
2. **Likely hot questions today:** PPO-vs-DQN (now aligned), why DET reverses (have Table IV
   + future-work (i) ready), the Eq. (2)/Eq. (3) math (rehearse the walk-throughs), and the
   demo-vs-paper environment question (use the caveat sentence).
3. The paper's per-seed plot, confusion matrix, CUSUM trace, and training curves are already
   embedded on slides 10–11 — keep those open when defending DET results.