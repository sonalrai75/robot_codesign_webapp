from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np

from robot_codesign.analysis.svd_design import analyze_design_space, secondary_metric
from robot_codesign.paths import PLANNERS
from robot_codesign.analysis import path_min_time_control

PRIMARY_INDEX = {
    "H_min": 0,
    "H_max": 1,
    "f1_Hz": 2,
    "f2_Hz": 3,
}


@dataclass(frozen=True)
class TargetSpec:
    metric: str
    relation: str  # equal | min | max
    value: float
    tolerance: float = 0.01


def _travel_time(robot, context):
    planner = PLANNERS[context["path_type"]]
    path = planner.plan(
        robot,
        context["start_xy"],
        context["end_xy"],
        branch=context.get("branch", 1),
        n=101,
    )
    timing = path_min_time_control(
        robot,
        path,
        context["torque_limits"],
        n_nodes=context.get("timing_nodes", 27),
    )
    return float(timing["time_s"])


def metric_value(robot, metric: str, context: dict) -> float:
    if metric in PRIMARY_INDEX:
        return float(robot.performance_vector()[PRIMARY_INDEX[metric]])
    if metric == "travel_time_s":
        return _travel_time(robot, context)
    if metric == "mass_kg":
        return float(robot.mass())
    if metric in {"smoothness", "peak_width", "rms_width"}:
        return float(secondary_metric(robot, metric))
    raise ValueError(f"Unsupported target metric: {metric}")


def all_metrics(robot, context: dict) -> dict:
    p = robot.performance_vector()
    return {
        "H_min": float(p[0]),
        "H_max": float(p[1]),
        "f1_Hz": float(p[2]),
        "f2_Hz": float(p[3]),
        "travel_time_s": _travel_time(robot, context),
        "mass_kg": float(robot.mass()),
        "smoothness": float(secondary_metric(robot, "smoothness")),
        "peak_width": float(secondary_metric(robot, "peak_width")),
        "rms_width": float(secondary_metric(robot, "rms_width")),
    }

def target_metrics(robot, targets: list[TargetSpec], context: dict) -> dict:
    """Evaluate only metrics needed to score a candidate line-search step."""
    out = {}
    p = None
    for t in targets:
        if t.metric in PRIMARY_INDEX:
            if p is None:
                p = robot.performance_vector()
            out[t.metric] = float(p[PRIMARY_INDEX[t.metric]])
        else:
            out[t.metric] = metric_value(robot, t.metric, context)
    return out


def target_delta_log(current: float, target: TargetSpec) -> tuple[float, bool, float]:
    """Return desired log-metric correction, satisfied flag, and normalized error.

    The normalized error is zero inside an inequality/tolerance band and otherwise
    the magnitude of the required logarithmic correction.
    """
    if current <= 0 or target.value <= 0:
        raise ValueError(f"Target metric {target.metric} must remain positive")
    tol = max(0.0, float(target.tolerance))
    ratio = current / target.value
    log_ratio = float(np.log(ratio))

    if target.relation == "equal":
        satisfied = abs(log_ratio) <= np.log1p(tol)
        return (-log_ratio if not satisfied else 0.0), satisfied, (0.0 if satisfied else abs(log_ratio))
    if target.relation == "max":
        # Satisfied up to the requested tolerance above the limit.
        satisfied = current <= target.value * (1.0 + tol)
        return (-log_ratio if not satisfied else 0.0), satisfied, (0.0 if satisfied else abs(log_ratio))
    if target.relation == "min":
        # Satisfied up to the requested tolerance below the limit.
        satisfied = current >= target.value / (1.0 + tol)
        return (-log_ratio if not satisfied else 0.0), satisfied, (0.0 if satisfied else abs(log_ratio))
    raise ValueError(f"Unknown target relation: {target.relation}")


def _metric_log_gradient(robot, metric: str, context: dict, rel_step: float = 8e-4) -> np.ndarray:
    """Gradient of log(metric) with respect to log-design variables.

    Travel-time sensitivities use the current path as a *frozen-path local
    linearization*.  Re-solving a nonlinear geodesic for every one of the 40
    central-difference probes is prohibitively expensive for an interactive
    search.  Accepted trial steps are still checked with the full path planner,
    so this approximation guides the local SVD step without replacing the
    nonlinear validation.
    """
    x = robot.design_vector()
    grad = np.zeros_like(x)
    h = float(rel_step)
    frozen_path = None
    if metric == "travel_time_s":
        planner = PLANNERS[context["path_type"]]
        frozen_path = planner.plan(robot, context["start_xy"], context["end_xy"],
                                   branch=context.get("branch", 1), n=101)

    def eval_metric(r):
        if metric == "travel_time_s":
            return float(path_min_time_control(r, frozen_path, context["torque_limits"],
                                               n_nodes=context.get("timing_nodes", 27))["time_s"])
        return metric_value(r, metric, context)

    for j in range(len(x)):
        xp = x.copy(); xm = x.copy()
        xp[j] += h; xm[j] -= h
        fp = eval_metric(robot.with_design_vector(xp))
        fm = eval_metric(robot.with_design_vector(xm))
        grad[j] = (np.log(fp) - np.log(fm)) / (2.0 * h)
    return grad


def target_jacobian(robot, targets: list[TargetSpec], context: dict) -> np.ndarray:
    """Jacobian of selected log performance metrics wrt log design.

    The existing four-metric sensitivity model is reused directly; only metrics
    outside that set (notably travel time) require additional finite differences.
    """
    base = analyze_design_space(robot)
    rows = []
    cache: dict[str, np.ndarray] = {}
    for t in targets:
        if t.metric in PRIMARY_INDEX:
            rows.append(base["jacobian"][PRIMARY_INDEX[t.metric]])
        else:
            if t.metric not in cache:
                cache[t.metric] = _metric_log_gradient(robot, t.metric, context)
            rows.append(cache[t.metric])
    return np.vstack(rows) if rows else np.zeros((0, len(robot.design_vector())))


def _secondary_gradient(robot, objective: str) -> np.ndarray:
    x = robot.design_vector()
    g = np.zeros_like(x)
    h = 1e-4
    for j in range(len(x)):
        xp = x.copy(); xm = x.copy(); xp[j] += h; xm[j] -= h
        fp = secondary_metric(robot.with_design_vector(xp), objective)
        fm = secondary_metric(robot.with_design_vector(xm), objective)
        # Use a stable relative gradient when the objective is positive.
        if fp > 0 and fm > 0:
            g[j] = (np.log(fp) - np.log(fm)) / (2*h)
        else:
            g[j] = (fp - fm) / (2*h)
    return g


def _null_projector(J: np.ndarray, n: int) -> np.ndarray:
    if J.size == 0:
        return np.eye(n)
    return np.eye(n) - np.linalg.pinv(J, rcond=1e-9) @ J


def _target_state(metrics: dict, targets: list[TargetSpec]):
    delta = []
    errors = []
    satisfied = []
    details = []
    for t in targets:
        d, ok, err = target_delta_log(float(metrics[t.metric]), t)
        delta.append(d); errors.append(err); satisfied.append(ok)
        details.append({
            "metric": t.metric,
            "relation": t.relation,
            "target": t.value,
            "tolerance": t.tolerance,
            "current": float(metrics[t.metric]),
            "satisfied": bool(ok),
            "log_error": float(err),
        })
    merit = float(np.linalg.norm(errors)) if errors else 0.0
    return np.asarray(delta, float), merit, all(satisfied) if satisfied else True, details


def _snapshot(iteration: int, robot, metrics: dict, targets: list[TargetSpec], J: np.ndarray | None,
              step_norm: float = 0.0, alpha: float = 0.0, note: str = "") -> dict:
    _, merit, satisfied, details = _target_state(metrics, targets)
    if J is None or J.size == 0:
        singular_values = []
        rank = 0
        condition = 0.0
    else:
        s = np.linalg.svd(J, compute_uv=False)
        tol = max(J.shape) * np.finfo(float).eps * (s[0] if len(s) else 0.0)
        rank = int(np.sum(s > tol))
        singular_values = s.tolist()
        nz = s[s > tol]
        condition = float(nz[0] / nz[-1]) if len(nz) > 1 else (1.0 if len(nz) == 1 else 0.0)
    return {
        "iteration": int(iteration),
        "metrics": {k: float(v) for k, v in metrics.items()},
        "targets": details,
        "target_merit": merit,
        "targets_satisfied": bool(satisfied),
        "singular_values": singular_values,
        "rank": rank,
        "condition": condition,
        "step_norm": float(step_norm),
        "line_search_alpha": float(alpha),
        "design_vector": robot.design_vector().tolist(),
        "t1": list(map(float, robot.t1)),
        "t2": list(map(float, robot.t2)),
        "note": note,
    }


def run_target_search(
    robot,
    targets: list[TargetSpec],
    context: dict,
    max_iterations: int | None = 10,
    step_limit: float = 0.16,
    secondary_objective: str = "none",
    secondary_weight: float = 0.25,
    progress: Callable[[dict], None] | None = None,
    safety_cap: int = 100,
):
    """Iterative global-target search using evolving local SVD subspaces.

    A global target is approached through a sequence of local, trust-region SVD
    steps.  The Jacobian/SVD is recomputed after every accepted nonlinear step.
    When ``max_iterations`` is None, iteration continues until convergence or
    stall, subject to a hard numerical safety cap.
    """
    if not targets:
        raise ValueError("Select at least one global performance target")
    if step_limit <= 0:
        raise ValueError("step_limit must be positive")
    if secondary_objective not in {"none", "mass", "smoothness", "peak_width", "rms_width"}:
        raise ValueError("Unsupported secondary objective")

    limit = safety_cap if max_iterations is None else min(int(max_iterations), safety_cap)
    if limit < 1:
        raise ValueError("max_iterations must be at least 1, or blank for unconstrained")

    current = robot
    trace = []
    metrics = all_metrics(current, context)
    J = target_jacobian(current, targets, context)
    snap = _snapshot(0, current, metrics, targets, J, note="Initial design")
    trace.append(snap)
    if progress: progress(snap)

    status = "running"
    message = ""
    stalled_count = 0

    for k in range(1, limit + 1):
        delta_y, merit0, targets_ok, _ = _target_state(metrics, targets)

        # If all mandatory targets are met and no secondary objective was asked
        # for, the global target search is complete.
        if targets_ok and secondary_objective == "none":
            status = "targets_satisfied"
            message = "All enabled performance targets are within tolerance."
            break

        J = target_jacobian(current, targets, context)
        if J.size:
            U, s, VT = np.linalg.svd(J, full_matrices=False)
            # Damped inverse prevents tiny singular values from causing huge moves.
            damping = max(1e-4, (s[0] if len(s) else 1.0) * 1e-5)
            inv = s / (s*s + damping*damping)
            active_step = VT.T @ (inv * (U.T @ delta_y))
        else:
            active_step = np.zeros_like(current.design_vector())

        sec_step = np.zeros_like(active_step)
        if secondary_objective != "none":
            grad = _secondary_gradient(current, secondary_objective)
            Pn = _null_projector(J, len(active_step))
            d = -(Pn @ grad)
            nd = np.linalg.norm(d)
            if nd > 1e-10:
                sec_step = d / nd * min(step_limit * float(secondary_weight), step_limit * 0.5)

        step = active_step + sec_step
        ns = float(np.linalg.norm(step))
        if ns > step_limit:
            step *= step_limit / ns
            ns = step_limit
        if ns < 1e-9:
            status = "stalled"
            message = "The evolving SVD produced no meaningful feasible local step."
            break

        x0 = current.design_vector()
        best = None
        # Nonlinear line search: accept only a step that improves target merit.
        # If targets are already satisfied and a secondary objective is active,
        # preserve feasibility and improve that objective.
        base_secondary = secondary_metric(current, secondary_objective) if secondary_objective != "none" else None
        for alpha in (1.0, 0.5, 0.25, 0.125, 0.0625):
            trial = current.with_design_vector(x0 + alpha * step)
            try:
                m_target = target_metrics(trial, targets, context)
                _, merit, ok, _ = _target_state(m_target, targets)
                if secondary_objective != "none":
                    sec = secondary_metric(trial, secondary_objective)
                else:
                    sec = 0.0
            except Exception:
                continue

            if targets_ok and secondary_objective != "none":
                # Once feasible, reject steps that leave the target tolerance band.
                score = (0.0 if ok else 1e3 + merit, float(sec))
                base_score = (0.0, float(base_secondary))
                improved = score < base_score
            else:
                score = (merit, float(sec))
                improved = merit < merit0 - max(1e-8, 1e-4 * max(merit0, 1e-8))

            if improved and (best is None or score < best[0]):
                best = (score, alpha, trial, m_target)

        if best is None:
            stalled_count += 1
            if stalled_count >= 1:
                status = "stalled"
                message = "No trust-region/backtracking step improved the nonlinear target error."
                break
            continue

        _, alpha, current, _ = best
        stalled_count = 0
        # Full metrics, including an exact replanned travel time, are evaluated
        # only for the accepted nonlinear design.
        metrics = all_metrics(current, context)
        Jnew = target_jacobian(current, targets, context)
        snap = _snapshot(
            k, current, metrics, targets, Jnew,
            step_norm=float(alpha * ns), alpha=float(alpha),
            note="Accepted nonlinear step; Jacobian/SVD recomputed at this design.",
        )
        trace.append(snap)
        if progress: progress(snap)
        # A target can become satisfied on the final allowed iteration.  Report
        # convergence rather than incorrectly labelling that case an iteration-limit stop.
        if snap["targets_satisfied"] and secondary_objective == "none":
            status = "targets_satisfied"
            message = "All enabled performance targets are within tolerance."
            break

    else:
        status = "iteration_limit"
        if max_iterations is None:
            message = f"Stopped at the numerical safety cap of {safety_cap} iterations."
        else:
            message = f"Stopped after the requested maximum of {limit} iterations."

    # Loop may terminate before entering because iteration 0 was already feasible.
    if status == "running":
        _, _, ok, _ = _target_state(metrics, targets)
        if ok:
            status = "targets_satisfied"
            message = "All enabled performance targets are within tolerance."
        else:
            status = "iteration_limit"
            message = "Iteration limit reached."

    return {
        "status": status,
        "message": message,
        "iterations_completed": max(0, len(trace) - 1),
        "max_iterations_requested": max_iterations,
        "unconstrained": max_iterations is None,
        "safety_cap": safety_cap,
        "secondary_objective": secondary_objective,
        "trace": trace,
        "final_robot": current,
    }
