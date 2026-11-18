"""
Velocity-adaptive One-Euro Filter for low-latency, jitter-free signal smoothing.
Reference: Casiez, Roussel, Vogel (CHI 2012) "1€ Filter: A Simple Speed-based
Low-pass Filter for Noisy Input in Interactive Systems".
"""

import math
import time


class OneEuroFilter:
    """
    Smooths noisy landmark/expression signals:
      - Low cutoff (min_cutoff) eliminates jitter when stationary.
      - Higher beta reduces lag immediately during fast intentional movements.
    """
    def __init__(self, min_cutoff=1.2, beta=0.08, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * max(cutoff, 1e-5))
        return 1.0 / (1.0 + tau / max(dt, 1e-5))

    def reset(self):
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None

    def filter(self, x, timestamp=None):
        """Filters a new scalar observation `x` at optional `timestamp` (seconds)."""
        now = time.perf_counter() if timestamp is None else float(timestamp)
        x = float(x)

        if self._x_prev is None or self._t_prev is None:
            self._x_prev = x
            self._dx_prev = 0.0
            self._t_prev = now
            return x

        dt = max(now - self._t_prev, 1e-4)
        self._t_prev = now

        # Estimate derivative (velocity) and smooth it
        dx = (x - self._x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev
        self._dx_prev = dx_hat

        # Adapt cutoff frequency based on velocity
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self._x_prev
        self._x_prev = x_hat
        return x_hat
