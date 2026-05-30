"""
Hawkes Process Simulator for Realistic Order Arrivals
=====================================================

Implements univariate and multivariate Hawkes processes to generate
realistic, self-exciting order arrival times for market microstructure
simulation using O(1) recursive state updates instead of O(N^2) past-event lookups.
"""

from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class HawkesProcess:
    """
    Univariate Hawkes process with exponential kernel.
    """

    def __init__(self, mu: float, alpha: float, omega: float) -> None:
        if mu <= 0:
            raise ValueError(f"Base intensity mu must be positive, got {mu}")
        if alpha < 0:
            raise ValueError(f"Excitation alpha must be non-negative, got {alpha}")
        if omega <= 0:
            raise ValueError(f"Decay omega must be positive, got {omega}")
        if alpha >= omega:
            raise ValueError(
                f"Branching ratio α/ω = {alpha/omega:.3f} ≥ 1 — "
                f"process is non-stationary."
            )

        self.mu = mu
        self.alpha = alpha
        self.omega = omega
        self.branching_ratio = alpha / omega

    def simulate(self, T: float, seed: Optional[int] = None) -> np.ndarray:
        """
        Simulate event times on [0, T] using Ogata's thinning algorithm in O(1) space/time per step.
        """
        rng = np.random.default_rng(seed)
        events: List[float] = []
        t = 0.0
        t_last = 0.0
        r_last = 0.0

        lambda_star = self.mu + self.alpha

        while t < T:
            u1 = rng.random()
            dt = -np.log(u1) / lambda_star
            t += dt

            if t >= T:
                break

            decayed_r = r_last * np.exp(-self.omega * (t - t_last))
            lambda_t = self.mu + self.alpha * decayed_r

            u2 = rng.random()
            if u2 <= lambda_t / lambda_star:
                events.append(t)
                r_last = decayed_r + 1.0
                t_last = t
                lambda_star = lambda_t + self.alpha
            else:
                lambda_star = max(lambda_t, self.mu + 0.1)

        return np.array(events, dtype=np.float64)

    def expected_rate(self) -> float:
        return self.mu / (1.0 - self.branching_ratio)

    def __repr__(self) -> str:
        return (
            f"HawkesProcess(mu={self.mu:.2f}, alpha={self.alpha:.2f}, "
            f"omega={self.omega:.2f}, branching_ratio={self.branching_ratio:.3f})"
        )


class MultivariateHawkes:
    """
    Multivariate Hawkes process for correlated order-flow events.
    """

    STREAMS = ("limit_orders", "market_orders", "cancellations")

    def __init__(
        self,
        params: Optional[Dict[str, Dict[str, float]]] = None,
        cross_excitation: Optional[Dict[Tuple[str, str], Dict[str, float]]] = None,
    ) -> None:
        self.params: Dict[str, Dict[str, float]] = params or {
            "limit_orders": {"mu": 40.0, "alpha": 15.0, "omega": 50.0},
            "market_orders": {"mu": 8.0, "alpha": 6.0, "omega": 25.0},
            "cancellations": {"mu": 35.0, "alpha": 12.0, "omega": 45.0},
        }

        self.cross_excitation: Dict[Tuple[str, str], Dict[str, float]] = (
            cross_excitation or {
                ("market_orders", "cancellations"): {"alpha": 20.0, "omega": 40.0},
                ("cancellations", "limit_orders"): {"alpha": 8.0, "omega": 30.0},
                ("market_orders", "limit_orders"): {"alpha": 5.0, "omega": 20.0},
            }
        )

        for stream, p in self.params.items():
            br = p["alpha"] / p["omega"]
            if br >= 1.0:
                raise ValueError(
                    f"Self-excitation branching ratio for '{stream}' "
                    f"is {br:.3f} ≥ 1 — non-stationary."
                )

    def simulate_all(
        self,
        T: float,
        seed: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Simulate all streams jointly using multivariate thinning in O(1) step complexity.
        """
        rng = np.random.default_rng(seed)
        all_events: Dict[str, List[float]] = {s: [] for s in self.STREAMS}
        t = 0.0
        t_last = 0.0

        r = {src: {tgt: 0.0 for tgt in self.STREAMS} for src in self.STREAMS}
        alphas = {src: {tgt: 0.0 for tgt in self.STREAMS} for src in self.STREAMS}
        omegas = {src: {tgt: 1.0 for tgt in self.STREAMS} for src in self.STREAMS}

        for s in self.STREAMS:
            alphas[s][s] = self.params[s]["alpha"]
            omegas[s][s] = self.params[s]["omega"]

        for (src, tgt), kernel in self.cross_excitation.items():
            alphas[src][tgt] = kernel["alpha"]
            omegas[src][tgt] = kernel["omega"]

        total_alpha_sum = sum(sum(alphas[src].values()) for src in self.STREAMS)
        lambda_star = sum(self.params[s]["mu"] for s in self.STREAMS) + total_alpha_sum

        max_events_per_stream = int(T * 500)

        while t < T:
            u1 = rng.random()
            if u1 == 0.0:
                u1 = 1e-15
            dt = -np.log(u1) / lambda_star
            t += dt

            if t >= T:
                break

            decayed_r = {src: {tgt: 0.0 for tgt in self.STREAMS} for src in self.STREAMS}
            intensities = {tgt: self.params[tgt]["mu"] for tgt in self.STREAMS}

            decay_dt = t - t_last
            for src in self.STREAMS:
                for tgt in self.STREAMS:
                    decayed_r[src][tgt] = r[src][tgt] * np.exp(-omegas[src][tgt] * decay_dt)
                    intensities[tgt] += alphas[src][tgt] * decayed_r[src][tgt]

            total_lambda = sum(intensities.values())

            u2 = rng.random()
            if u2 <= total_lambda / lambda_star:
                u3 = rng.random() * total_lambda
                cumsum = 0.0
                chosen_stream = self.STREAMS[-1]
                for s in self.STREAMS:
                    cumsum += intensities[s]
                    if u3 <= cumsum:
                        chosen_stream = s
                        break

                if len(all_events[chosen_stream]) < max_events_per_stream:
                    all_events[chosen_stream].append(t)

                for src in self.STREAMS:
                    for tgt in self.STREAMS:
                        r[src][tgt] = decayed_r[src][tgt]
                for tgt in self.STREAMS:
                    r[chosen_stream][tgt] += 1.0

                t_last = t
                lambda_star = total_lambda + total_alpha_sum
            else:
                lambda_star = max(total_lambda, sum(self.params[s]["mu"] for s in self.STREAMS) + 0.1)

        return {s: np.array(evts, dtype=np.float64) for s, evts in all_events.items()}

    def __repr__(self) -> str:
        lines = ["MultivariateHawkes("]
        for s in self.STREAMS:
            p = self.params[s]
            lines.append(f"  {s}: mu={p['mu']:.1f}, alpha={p['alpha']:.1f}, omega={p['omega']:.1f}")
        lines.append(")")
        return "\n".join(lines)


def generate_intraday_arrivals(
    symbol_config: Dict[str, Any],
    date: datetime.date,
    volatility_profile: Any,
    hawkes_params: Optional[Dict[str, Dict[str, float]]] = None,
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Generate realistic intraday order arrival times for a given symbol.
    """
    from python.data_generation.nse_config import TRADING_SECONDS

    rng_base = np.random.default_rng(seed)

    avg_vol = symbol_config.get("avg_daily_volume", 10_000_000)
    volume_scale = np.clip(avg_vol / 10_000_000, 0.3, 3.0)

    default_params = {
        "limit_orders": {"mu": 40.0 * volume_scale, "alpha": 15.0, "omega": 50.0},
        "market_orders": {"mu": 8.0 * volume_scale, "alpha": 6.0, "omega": 25.0},
        "cancellations": {"mu": 35.0 * volume_scale, "alpha": 12.0, "omega": 45.0},
    }
    params = hawkes_params or default_params

    window_secs = 300.0
    num_windows = int(np.ceil(TRADING_SECONDS / window_secs))

    all_events: Dict[str, List[float]] = {
        "limit_orders": [],
        "market_orders": [],
        "cancellations": [],
    }

    for w in range(num_windows):
        window_start = w * window_secs
        window_end = min((w + 1) * window_secs, TRADING_SECONDS)
        window_len = window_end - window_start

        mid_time = window_start + window_len / 2.0
        vol_mult = float(volatility_profile(mid_time))

        scaled_params = {}
        for stream, p in params.items():
            scaled_params[stream] = {
                "mu": p["mu"] * vol_mult,
                "alpha": p["alpha"],
                "omega": p["omega"],
                "branching_ratio": p["alpha"] / p["omega"] if p["omega"] > 0 else 0
            }

        child_seed = rng_base.integers(0, 2**31)
        mh = MultivariateHawkes(params=scaled_params)
        window_events = mh.simulate_all(T=window_len, seed=int(child_seed))

        for stream in all_events:
            shifted = window_events[stream] + window_start
            all_events[stream].extend(shifted.tolist())

    result: Dict[str, np.ndarray] = {}
    for stream, times in all_events.items():
        arr = np.array(sorted(times), dtype=np.float64)
        arr = arr[arr <= TRADING_SECONDS]
        result[stream] = arr

    return result
