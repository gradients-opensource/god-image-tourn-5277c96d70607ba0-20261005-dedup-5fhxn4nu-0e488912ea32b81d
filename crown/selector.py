"""Candidate selection and plateau detection on measured statistics.

Two problems, both created by the evaluator's structure:

1. Selection noise. The holdout screen uses 1 of the evaluator's 16 noise draws
   per stratum; the argmin over many candidates of a noisy estimate is biased
   toward lucky candidates (winner's curse). Fix: race — screen every
   candidate cheaply, confirm the top few with more draws, and pick with a
   one-standard-error rule using *paired* differences (same noise draws for
   both candidates, so the SE is the SE of the difference, not of the levels).

2. Plateau. Training past the holdout optimum wastes the rest of the budget
   (50-90% in the current champion's own runs). Fix: declare a plateau when
   recent screens fail to beat the best by more than the paired SE, so the
   controller can spend the remaining budget on something that still helps.

No constant here is tuned on a dataset: the SE is measured from the screen's
own per-case losses, and the small integers (top-k, window) are counts.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .scoring import ScoreReport


@dataclass
class Candidate:
    tag: str
    step: int
    state: dict
    scale: float
    screen: Optional[ScoreReport] = None
    confirm: Optional[ScoreReport] = None
    member: int = 0
    unloadable: bool = False      # the evaluator twin refused this state: never ship or rank it (v6.1 M6)

    @property
    def score(self):
        rep = self.confirm if self.confirm is not None else self.screen
        return sel_score(rep) if rep is not None else float("inf")


SEL_METRIC = "mean"
# 1-SE tie-break inside the best candidate's own trajectory. "earliest" is the shipped rule (less-overfit direction).
# "latest" is the v10 experiment: on the 09-27/28 z-image and krea2 plus-image trajectories the last saved state was
# the best in every run and the high-sigma band kept improving, so "latest within 1 SE" recovered +0.039 % (SE 0.015)
# of 0.045 % selection regret (research/plus5/selection_noise_research.md). Found on those same splits: it ships only
# after a fresh-split test that includes an early-overfitting family (flux, ideogram4).
PICK_TIEBREAK = "earliest"


def sel_score(rep) -> float:
    """The number the selector ranks by: the evaluator's mean, or a robust reduction of the
    per-image losses (p75 / worst) that prefers candidates without a bad image."""
    if SEL_METRIC == "mean" or len(rep.per_image) == 1:
        return float(rep.score)
    if SEL_METRIC == "p75":
        return float(np.quantile(rep.per_image, 0.75))
    if SEL_METRIC == "worst":
        return float(max(rep.per_image))
    raise ValueError(f"unknown select metric {SEL_METRIC!r}")


class Selector:
    def __init__(self):
        self.cands: list[Candidate] = []
        self.history: list[tuple[int, float]] = []   # (step, screen score) in time order

    def add(self, cand: Candidate):
        self.cands.append(cand)
        if cand.screen is not None:
            self.history.append((cand.step, sel_score(cand.screen)))

    def best(self, confirmed_only=False) -> Optional[Candidate]:
        pool = [c for c in self.cands if not c.unloadable and (c.confirm if confirmed_only else (c.confirm or c.screen)) is not None]
        return min(pool, key=lambda c: c.score) if pool else None

    def top(self, k: int, member=None):
        """The k best screened candidates, one per eval point (member, step): raw/ema twins of one
        step must not crowd out other steps in the confirm stage (v6.1 m4)."""
        pool = [c for c in self.cands if c.screen is not None and not c.unloadable and (member is None or c.member == member)]
        per_point = {}
        for c in sorted(pool, key=lambda c: sel_score(c.screen)):
            per_point.setdefault((c.member, c.step), c)
        return list(per_point.values())[:k]

    # --- 1-SE rule ------------------------------------------------------------------
    def pick(self, use_confirm=True) -> Optional[Candidate]:
        """Among candidates whose paired difference to the best is within one SE, return the earliest
        one OF THE BEST CANDIDATE'S OWN TRAJECTORY. Earlier = less overfit direction, by the measured
        monotone overfit curves - which order the checkpoints of one member's run; a soup (member -1,
        its step is the members' total) and another member (its own step count) are neither earlier nor
        later than it, so they only win by score. The rule never picks something significantly worse."""
        pool = [c for c in self.cands if not c.unloadable and (c.confirm if use_confirm else c.screen) is not None]
        if not pool:
            return self.best()
        rep = (lambda c: c.confirm) if use_confirm else (lambda c: c.screen)
        b = min(pool, key=lambda c: rep(c).score)
        within = []
        for c in pool:
            if c is b:
                within.append(c); continue
            if c.member != b.member:
                continue
            mean, se, n = rep(c).paired_diff(rep(b))
            if n > 1 and mean <= se:      # c - b <= SE  → not distinguishably worse
                within.append(c)
        if PICK_TIEBREAK == "latest":
            return min(within, key=lambda c: (-c.step, rep(c).score))  # same step: the better one
        if PICK_TIEBREAK != "earliest":
            raise ValueError(f"unknown pick tie-break {PICK_TIEBREAK!r}")
        return min(within, key=lambda c: (c.step, rep(c).score))     # same step: the better one (v6.1 m3)

    # --- plateau -------------------------------------------------------------------
    def _points(self, member=None):
        """Eval POINTS in step order: the best-screened candidate at each (member, step)."""
        screened = [c for c in self.cands if c.screen is not None and (member is None or c.member == member)]
        points = {}
        for c in screened:
            key = (c.member, c.step)
            if key not in points or c.screen.score < points[key].screen.score:
                points[key] = c
        return [points[k] for k in sorted(points)]

    def trailing_improvements(self, member=None) -> int:
        """How many of the most recent eval points, newest first, each beat the best of ALL earlier
        points by more than the paired SE; stops at the first that did not. 0 = the curve has
        stopped descending (or too few points)."""
        ordered = [c for c in self._points(member) if c.step > 0]   # the identity point is not an improvement to count (v6.1 m9)
        r = 0
        for i in range(len(ordered) - 1, 0, -1):
            b = min(ordered[:i], key=lambda c: c.screen.score)
            mean, se, n = ordered[i].screen.paired_diff(b.screen)
            if n > 1 and mean < -se:
                r += 1
            else:
                break
        return r

    def best_point_index(self, member=None) -> int:
        """Index (0-based, step order, identity excluded) of the best screened eval point of this member;
        -1 when there are no real points yet."""
        ordered = [c for c in self._points(member) if c.step > 0]
        if not ordered:
            return -1
        return min(range(len(ordered)), key=lambda i: ordered[i].screen.score)

    def plateaued(self, window: int = 3, member=None) -> bool:
        """True when the last `window` eval POINTS (a point = all candidates screened at one step of
        one member) each failed to beat the running best of the earlier points by more than the
        paired SE of that comparison. Counting points, not candidates, keeps raw/ema pairs from
        making a two-point window look like three."""
        ordered = self._points(member)
        if len(ordered) < window + 1:
            return False
        recent, earlier = ordered[-window:], ordered[:-window]
        b = min(earlier, key=lambda c: c.screen.score)
        for c in recent:
            mean, se, n = c.screen.paired_diff(b.screen)
            if n > 1 and mean < -se:       # a real improvement
                return False
        return True
