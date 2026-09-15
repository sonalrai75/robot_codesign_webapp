from __future__ import annotations

"""Local singularity / Thom-style catastrophe diagnostics.

This module is deliberately conservative.  A small singular value is a trigger
for deeper analysis, not proof of a catastrophe.  The routines report
candidates and numerical support from directional second/third derivatives.
Definitive Thom classification generally requires the appropriate equilibrium
or potential formulation plus non-degeneracy/transversality tests.
"""

from dataclasses import dataclass
from typing import Callable, Sequence
import numpy as np


@dataclass(frozen=True)
class CatastropheThresholds:
    near_sigma_ratio: float = 0.08
    near_condition: float = 12.0
    rank_sigma_ratio: float = 1.0e-3
    fold_curvature_alignment: float = 0.15
    cusp_curvature_alignment: float = 0.15
    cubic_alignment: float = 0.15


def _directional_derivatives(
    evaluate: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    v: np.ndarray,
    h: float = 2.0e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Central directional second and third derivatives along normalized v."""
    v = np.asarray(v, float)
    v = v / max(float(np.linalg.norm(v)), 1e-15)
    x = np.asarray(x, float)
    f0 = np.asarray(evaluate(x), float)
    fp = np.asarray(evaluate(x + h*v), float)
    fm = np.asarray(evaluate(x - h*v), float)
    fpp = np.asarray(evaluate(x + 2*h*v), float)
    fmm = np.asarray(evaluate(x - 2*h*v), float)
    d2 = (fp - 2.0*f0 + fm) / (h*h)
    d3 = (fpp - 2.0*fp + 2.0*fm - fmm) / (2.0*h**3)
    return d2, d3


def diagnose_map_singularity(
    J: np.ndarray,
    evaluate: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    metric_names: Sequence[str],
    *,
    force_deep: bool = False,
    derivative_step: float = 2.0e-3,
    thresholds: CatastropheThresholds | None = None,
) -> dict:
    """Diagnose a possible fold/cusp of a performance map near ``x``.

    ``J`` must be the Jacobian of the same (normally log) metrics returned by
    ``evaluate`` with respect to the design coordinates in ``x``.
    """
    th = thresholds or CatastropheThresholds()
    J = np.asarray(J, float)
    x = np.asarray(x, float)
    names = list(metric_names)
    if J.ndim != 2 or J.shape[0] == 0:
        return {"level": "not_applicable", "message": "No active performance map was supplied."}

    U, s, VT = np.linalg.svd(J, full_matrices=False)
    smax = float(s[0]) if len(s) else 0.0
    smin = float(s[-1]) if len(s) else 0.0
    ratio = smin / smax if smax > 0 else 0.0
    condition = smax / smin if smin > 0 else None
    near = bool(ratio <= th.near_sigma_ratio or (condition is not None and condition >= th.near_condition))
    critical = bool(ratio <= th.rank_sigma_ratio)
    u = U[:, -1]
    v = VT[-1, :]

    order = np.argsort(np.abs(u))[::-1]
    suggested = [names[i] for i in order[:min(2, len(names))]]
    out = {
        "level": "near_rank_loss" if near else "regular",
        "classification": "none",
        "confidence": "diagnostic",
        "sigma_min": smin,
        "sigma_max": smax,
        "sigma_ratio": float(ratio),
        "condition": (float(condition) if condition is not None else None),
        "near_rank_loss": near,
        "numerical_rank_loss": critical,
        "critical_left_vector": u.tolist(),
        "critical_right_vector": v.tolist(),
        "suggested_control_metrics": suggested,
        "deep_analysis_performed": False,
        "message": (
            "Near-rank-loss trigger reached; higher-order directional tests are indicated."
            if near else
            "No near-rank-loss trigger at this design. Catastrophe classification was not attempted."
        ),
        "caution": (
            "A small singular value or fold-shaped plot alone does not establish a Thom catastrophe. "
            "The reported labels are numerical diagnostics of the performance map."
        ),
    }
    if not (near or force_deep):
        return out

    d2, d3 = _directional_derivatives(evaluate, x, v, derivative_step)
    a2 = float(np.dot(u, d2))
    a3 = float(np.dot(u, d3))
    n2 = float(np.linalg.norm(d2))
    n3 = float(np.linalg.norm(d3))
    q2 = abs(a2) / max(n2, 1e-12)
    q3 = abs(a3) / max(n3, 1e-12)

    out.update({
        "deep_analysis_performed": True,
        "directional_second": d2.tolist(),
        "directional_third": d3.tolist(),
        "projected_second_a2": a2,
        "projected_third_a3": a3,
        "second_alignment": float(q2),
        "third_alignment": float(q3),
        "derivative_step": float(derivative_step),
    })

    # Conservative local classification.  Unless numerical rank loss is very
    # strong, we intentionally retain the word "candidate".
    if near and q2 >= th.fold_curvature_alignment:
        out["level"] = "fold_candidate"
        out["classification"] = "A2 fold" if critical else "A2 fold candidate"
        out["confidence"] = "numerically_supported" if critical else "candidate"
        out["message"] = (
            "Weak active direction plus nonzero projected quadratic curvature supports a local fold candidate. "
            "Two-parameter continuation is the next test for cusp structure."
        )
    elif near and q2 < th.cusp_curvature_alignment and q3 >= th.cubic_alignment:
        out["level"] = "cusp_candidate"
        out["classification"] = "A3 cusp" if critical else "A3 cusp candidate"
        out["confidence"] = "numerically_supported" if critical else "candidate"
        out["message"] = (
            "The projected quadratic term is weak while the cubic term is nonzero along the critical direction; "
            "this is a cusp candidate and requires two-control-parameter fold continuation/transversality checks."
        )
    else:
        out["message"] = (
            "Near-rank-loss geometry was detected, but the local derivative tests do not support a fold or cusp label at this point."
        )
    return out


def scalar_normal_form_diagnostics(
    f: Callable[[float], float], z0: float = 0.0, h: float = 1e-3
) -> dict:
    """Small test helper for canonical scalar fold/cusp normal forms."""
    fm2, fm1, f0, fp1, fp2 = [float(f(z0+k*h)) for k in (-2, -1, 0, 1, 2)]
    d1 = (fp1-fm1)/(2*h)
    d2 = (fp1-2*f0+fm1)/(h*h)
    d3 = (fp2-2*fp1+2*fm1-fm2)/(2*h**3)
    if abs(d1) < 1e-5 and abs(d2) > 1e-3:
        kind = "A2 fold"
    elif abs(d1) < 1e-5 and abs(d2) <= 1e-3 and abs(d3) > 1e-2:
        kind = "A3 cusp"
    else:
        kind = "unclassified"
    return {"classification": kind, "d1": d1, "d2": d2, "d3": d3}


def catastrophe_search_scores(diagnostic: dict) -> dict:
    """Dimensionless ranking scores used by the automated control-space search.

    Lower is better.  The fold score rewards rank loss plus a strong quadratic
    critical term.  The cusp score rewards rank loss, a weak quadratic term,
    and a nonzero cubic term.  These are discovery scores, not classifications.
    """
    ratio = max(float(diagnostic.get("sigma_ratio", 1.0)), 0.0)
    q2 = max(float(diagnostic.get("second_alignment", 0.0)), 0.0)
    q3 = max(float(diagnostic.get("third_alignment", 0.0)), 0.0)
    fold = ratio + 0.02 / max(q2, 1.0e-4)
    cusp = ratio + q2 + max(0.0, 0.15 - q3)
    return {"fold_score": float(fold), "cusp_score": float(cusp)}
