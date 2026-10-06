"""Wall-clock budget controller.

The validator kills the container at `hours_to_complete`; there is no reward for
finishing early and total loss for finishing late. So the controller's job is to
spend everything up to a safety margin, and to know at every moment whether the
next action (a step, an eval, a save) still fits.

All timings are measured from this run — step time and eval time are EMAs of
what actually happened, never assumed.
"""

import time

from . import contract as C


class Budget:
    def __init__(self, deadline_ts: float, kill_margin_s: float = C.KILL_MARGIN_S, publish_reserve_s: float = C.PUBLISH_RESERVE_S):
        self.t0 = time.time()
        self.deadline = deadline_ts - kill_margin_s        # hard stop for any compute
        self.publish_reserve = publish_reserve_s
        self.step_time = None    # EMA seconds per optimizer step
        self.eval_time = None    # EMA seconds per holdout screen (1 noise)
        self.n_steps = 0
        self.n_evals = 0

    # --- measurement -------------------------------------------------------------
    def observe_step(self, dt: float):
        dt = max(0.0, dt)                # a clock step backwards must not make the EMA negative
        self.step_time = dt if self.step_time is None else 0.9 * self.step_time + 0.1 * dt
        self.n_steps += 1

    def observe_eval(self, dt: float):
        self.eval_time = dt if self.eval_time is None else 0.7 * self.eval_time + 0.3 * dt
        self.n_evals += 1

    # --- queries -----------------------------------------------------------------
    def remaining(self) -> float:
        """Seconds left before the hard stop, minus the publish reserve."""
        return self.deadline - time.time() - self.publish_reserve

    def progress(self) -> float:
        """Fraction of the usable budget consumed, in [0, 1]."""
        total = max(1.0, self.deadline - self.publish_reserve - self.t0)
        return min(1.0, max(0.0, (time.time() - self.t0) / total))

    def fits(self, seconds: float) -> bool:
        return self.remaining() >= seconds

    def est_step(self) -> float:
        # floored: callers divide by it, and a first step timed at 0 s (a trivially fast mock step on a
        # coarse clock) raised ZeroDivisionError in _eval_every (VPS mock suite, 2026-09-27); real steps
        # take 0.5-1.5 s, so the floor never binds on a GPU
        return max(1e-4, self.step_time) if self.step_time is not None else 3.0

    def est_eval(self, noises: int = 1) -> float:
        # holdout cost scales linearly in noise draws; 60 s is a conservative prior before the first measurement
        base = self.eval_time if self.eval_time is not None else 60.0
        return base * noises

    def steps_affordable(self, reserve_s: float = 0.0) -> int:
        return max(0, int((self.remaining() - reserve_s) / self.est_step()))

    def elapsed(self) -> float:
        return time.time() - self.t0
