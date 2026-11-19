"""On-line strategy transition detection (CUSUM over label change + confirmation).

Robust two-stage detector that does NOT fire on stable input:

1) A CUSUM statistic accumulates on a *binary change indicator* -- 1 when the
   current argmax label differs from the active segment label, else 0:

       S_t = max(0, S_{t-1} + indicator_t - k)

   With k in (0,1) the statistic decays to 0 whenever the label is stable
   (indicator=0), so a constant input can never accumulate drift.

2) When S_t >= h a candidate change is armed; it is confirmed as a genuine
   transition only if the incoming label persists for `confirm` consecutive
   steps and differs from the active segment label. Otherwise it is discarded
   as noise and the statistic resets.

`min_segment` prevents re-arming immediately after a confirmed change.
"""

import numpy as np


class CusumTransitionDetector:
    def __init__(self, cfg, n_classes):
        t = cfg["transitions"]
        self.k = float(t.get("cusum_k", 0.5))       # decay below indicator=1
        self.h = float(t.get("cusum_h", 6.0))       # alarm threshold
        self.min_segment = int(t.get("min_segment", 10))
        self.confirm = int(t.get("confirm", 10))
        self.smoothing = float(t.get("score_smoothing", 4))
        self.n_classes = n_classes

        self._smoothed = None
        self.segment_label = None
        self.current_label = None
        self._cusum = 0.0
        self._decided = False

        self._candidate = False
        self._cand_label = None
        self._cand_count = 0
        self._cand_span = 0
        self._since_change = 0
        self.noise_alarms = 0

    def reset(self):
        self._smoothed = None
        self.segment_label = None
        self.current_label = None
        self._cusum = 0.0
        self._decided = False
        self._candidate = False
        self._cand_label = None
        self._cand_count = 0
        self._cand_span = 0
        self._since_change = 0
        self.noise_alarms = 0

    def _ema(self, scores):
        alpha = 1.0 / (self.smoothing + 1.0)  # e.g. smoothing=4 -> alpha=0.2
        if self._smoothed is None:
            self._smoothed = scores.copy()
        else:
            self._smoothed = alpha * scores + (1.0 - alpha) * self._smoothed
        return int(self._smoothed.argmax())

    def update(self, scores):
        label = self._ema(np.asarray(scores, dtype=np.float64))

        if not self._decided:
            self.segment_label = label
            self.current_label = label
            self._decided = True
            return label, False

        transition = False
        self._since_change += 1

        # ----- candidate tracking ----------------------------------- #
        if self._candidate:
            self._cand_span += 1
            if label == self._cand_label:
                self._cand_count += 1
            # require label differs from the active segment to confirm
            if (
                self._cand_label != self.segment_label
                and self._cand_count >= self.confirm
            ):
                transition = True
                self.segment_label = self._cand_label
                self._since_change = 0
                self._reset_candidate()
                self._cusum = 0.0
            elif self._cand_span >= max(3 * self.confirm, 40):
                # expired candidate -> noise, forget it and lower the statistic
                self.noise_alarms += 1
                self._reset_candidate()
                self._cusum = max(0.0, self._cusum - 1.0)
        # ----- CUSUM drift on binary label-change indicator ---------- #
        indicator = 1.0 if label != self.segment_label else 0.0
        if not self._candidate and self._since_change >= self.min_segment:
            self._cusum = max(0.0, self._cusum + indicator - self.k)
            if self._cusum >= self.h:
                self._candidate = True
                self._cand_label = label
                self._cand_count = 0
                self._cand_span = 0
                self._cusum = 0.0

        self.current_label = label
        return label, transition

    def _reset_candidate(self):
        self._candidate = False
        self._cand_label = None
        self._cand_count = 0
        self._cand_span = 0

    @property
    def decided(self):
        return self._decided