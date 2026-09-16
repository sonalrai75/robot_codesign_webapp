from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np

from robot_codesign.analysis.svd_design import analyze_design_space, secondary_metric
from robot_codesign.paths import PLANNERS
from robot_codesign.analysis import path_min_time_control
from robot_codesign.analysis.catastrophe import diagnose_map_singularity, catastrophe_search_scores

PRIMARY_INDEX = {
    "H_min": 0,
    "H_max": 1,
    "f1_Hz": 2,
    "f2_Hz": 3,
}

SECONDARY_METRICS = {"mass", "smoothness", "peak_width", "rms_width", "maneuverability"}


@dataclass(frozen=True)
class TargetSpec:
    metric: str
    relation: str  # equal | min | max
    value: float
    tolerance: float = 0.01


@dataclass(frozen=True)
class SecondarySpec:
    metric: str
    direction: str = "min"  # min | max
    weight: float = 1.0


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


def secondary_value(robot, metric: str) -> float:
    if metric == "mass":
        return float(robot.mass())
    if metric in {"smoothness", "peak_width", "rms_width"}:
        return float(secondary_metric(robot, metric))
    if metric == "maneuverability":
        s = np.asarray(analyze_design_space(robot)["singular_values"], float)
        return float(s[-1]) if len(s) else 0.0
    raise ValueError(f"Unsupported secondary metric: {metric}")


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
    """Return desired log-metric correction, satisfied flag, and normalized error."""
    if current <= 0 or target.value <= 0:
        raise ValueError(f"Target metric {target.metric} must remain positive")
    tol = max(0.0, float(target.tolerance))
    ratio = current / target.value
    log_ratio = float(np.log(ratio))

    if target.relation == "equal":
        satisfied = abs(log_ratio) <= np.log1p(tol)
        return (-log_ratio if not satisfied else 0.0), satisfied, (0.0 if satisfied else abs(log_ratio))
    if target.relation == "max":
        satisfied = current <= target.value * (1.0 + tol)
        return (-log_ratio if not satisfied else 0.0), satisfied, (0.0 if satisfied else abs(log_ratio))
    if target.relation == "min":
        satisfied = current >= target.value / (1.0 + tol)
        return (-log_ratio if not satisfied else 0.0), satisfied, (0.0 if satisfied else abs(log_ratio))
    raise ValueError(f"Unknown target relation: {target.relation}")


def _metric_log_gradient(robot, metric: str, context: dict, rel_step: float = 8e-4) -> np.ndarray:
    """Gradient of log(metric) with respect to log-design variables.

    Travel-time sensitivities use the current path as a frozen-path local
    linearization. Accepted trial steps are still checked with the full planner.
    """
    x = robot.design_vector()
    grad = np.zeros_like(x)
    h = float(rel_step)
    frozen_path = None
    if metric == "travel_time_s":
        planner = PLANNERS[context["path_type"]]
        frozen_path = planner.plan(
            robot,
            context["start_xy"],
            context["end_xy"],
            branch=context.get("branch", 1),
            n=101,
        )

    def eval_metric(r):
        if metric == "travel_time_s":
            return float(
                path_min_time_control(
                    r,
                    frozen_path,
                    context["torque_limits"],
                    n_nodes=context.get("timing_nodes", 27),
                )["time_s"]
            )
        return metric_value(r, metric, context)

    for j in range(len(x)):
        xp = x.copy(); xm = x.copy()
        xp[j] += h; xm[j] -= h
        fp = eval_metric(robot.with_design_vector(xp))
        fm = eval_metric(robot.with_design_vector(xm))
        grad[j] = (np.log(fp) - np.log(fm)) / (2.0 * h)
    return grad


def target_jacobian(robot, targets: list[TargetSpec], context: dict) -> np.ndarray:
    """Jacobian of selected log performance metrics wrt log design."""
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




def catastrophe_diagnostics(robot, targets: list[TargetSpec], context: dict, force_deep: bool = False) -> dict:
    """Local Thom-style singularity diagnostics for the selected target map.

    The target map is expressed in log-performance/log-design coordinates,
    matching the evolving-SVD target search.  This is intentionally a
    diagnostic layer: it does not promote a small singular value to a
    catastrophe without higher-order evidence.
    """
    if not targets:
        return {"level": "not_applicable", "message": "Select at least one performance target."}
    J = target_jacobian(robot, targets, context)
    x0 = robot.design_vector()
    names = [t.metric for t in targets]

    def evaluate(x):
        r = robot.with_design_vector(np.asarray(x, float))
        vals = target_metrics(r, targets, context)
        y = np.asarray([float(vals[n]) for n in names], float)
        if np.any(y <= 0):
            raise ValueError("Catastrophe diagnostics require positive target metrics for log scaling")
        return np.log(y)

    return diagnose_map_singularity(J, evaluate, x0, names, force_deep=force_deep)



def search_catastrophes(
    robot, targets: list[TargetSpec], context: dict,
    control_metrics: list[str] | None = None,
    span_fraction: float = 0.15, grid_points: int = 5,
    solve_iterations: int = 10, step_limit: float = 0.16,
) -> dict:
    """Automatically search a two-control target region for fold/cusp candidates.

    Each grid point changes two selected performance target values, solves back
    toward that level set with the existing evolving-SVD target solver, then
    evaluates the local higher-order singularity diagnostic.  This is a
    discovery scan; candidates still require continuation/transversality checks.
    """
    if len(targets) < 2:
        raise ValueError("Catastrophe search requires at least two enabled performance targets")
    if not (0.01 <= span_fraction <= 0.50):
        raise ValueError("span_fraction must be between 0.01 and 0.50")
    if grid_points not in {3, 5, 7}:
        raise ValueError("grid_points must be 3, 5, or 7")
    if not (1 <= solve_iterations <= 30):
        raise ValueError("solve_iterations must be 1..30")

    # Use the local SVD to propose controls when the caller does not choose them.
    seed_diag = catastrophe_diagnostics(robot, targets, context, force_deep=True)
    available = [t.metric for t in targets]
    controls = list(control_metrics or seed_diag.get("suggested_control_metrics", []))
    controls = [m for m in controls if m in available]
    for m in available:
        if m not in controls:
            controls.append(m)
        if len(controls) == 2:
            break
    controls = controls[:2]
    if len(controls) != 2 or controls[0] == controls[1]:
        raise ValueError("Choose two distinct enabled target metrics as catastrophe-search controls")

    base = {t.metric: float(t.value) for t in targets}
    factors = np.linspace(1.0-span_fraction, 1.0+span_fraction, grid_points)
    points = []
    warm = robot
    for i, fa in enumerate(factors):
        row_factors = factors if i % 2 == 0 else factors[::-1]  # serpentine warm starts
        for fb in row_factors:
            varied=[]
            for t in targets:
                value=base[t.metric]
                relation=t.relation
                if t.metric == controls[0]:
                    value *= float(fa); relation="equal"
                elif t.metric == controls[1]:
                    value *= float(fb); relation="equal"
                varied.append(TargetSpec(t.metric, relation, value, t.tolerance))
            try:
                solved=run_target_search(warm,varied,context,max_iterations=solve_iterations,
                                         step_limit=step_limit,secondary_objectives=[],
                                         null_step_fraction=0.0,safety_cap=max(30,solve_iterations))
                candidate=solved["final_robot"]
                diag=catastrophe_diagnostics(candidate,varied,context,force_deep=True)
                scores=catastrophe_search_scores(diag)
                _, merit, feasible, _ = _target_state(target_metrics(candidate,varied,context),varied)
                rec={
                    "control_1":controls[0], "control_1_value":float(base[controls[0]]*fa),
                    "control_2":controls[1], "control_2_value":float(base[controls[1]]*fb),
                    "factor_1":float(fa), "factor_2":float(fb),
                    "feasible":bool(feasible), "target_merit":float(merit),
                    "solver_status":solved["status"],
                    "classification":diag.get("classification","none"),
                    "level":diag.get("level","regular"),
                    "sigma_min":float(diag.get("sigma_min",0.0)),
                    "sigma_ratio":float(diag.get("sigma_ratio",1.0)),
                    "condition":diag.get("condition"),
                    "a2":diag.get("projected_second_a2"),
                    "a3":diag.get("projected_third_a3"),
                    "second_alignment":diag.get("second_alignment"),
                    "third_alignment":diag.get("third_alignment"),
                    **scores,
                    "t1":list(map(float,candidate.t1)), "t2":list(map(float,candidate.t2)),
                }
                # Penalize points that did not reach the requested level set.
                if not feasible:
                    rec["fold_score"] += 10.0 + merit
                    rec["cusp_score"] += 10.0 + merit
                points.append(rec)
                if feasible:
                    warm=candidate
            except Exception as exc:
                points.append({
                    "control_1":controls[0],"control_1_value":float(base[controls[0]]*fa),
                    "control_2":controls[1],"control_2_value":float(base[controls[1]]*fb),
                    "factor_1":float(fa),"factor_2":float(fb),
                    "feasible":False,"error":str(exc),
                    "fold_score":1e9,"cusp_score":1e9,
                })

    valid=[p for p in points if p.get("feasible") and p.get("sigma_ratio") is not None]
    fold=sorted(valid,key=lambda p:p["fold_score"])[:5]
    cusp=sorted(valid,key=lambda p:p["cusp_score"])[:5]

    # A sign change in a2 between nearby feasible points is particularly useful
    # for locating a possible cusp on a fold locus.
    crossings=[]
    for a in valid:
        if a.get("a2") is None: continue
        for b in valid:
            if b is a or b.get("a2") is None: continue
            di=abs(a["factor_1"]-b["factor_1"]); dj=abs(a["factor_2"]-b["factor_2"])
            adjacent=(di < 1e-12 and dj <= 2*span_fraction/(grid_points-1)+1e-12) or (dj < 1e-12 and di <= 2*span_fraction/(grid_points-1)+1e-12)
            if adjacent and a["a2"]*b["a2"] < 0:
                key=tuple(sorted(((a["factor_1"],a["factor_2"]),(b["factor_1"],b["factor_2"]))))
                if not any(c["key"]==key for c in crossings):
                    crossings.append({"key":key,"point_a":a,"point_b":b})
    for c in crossings: c.pop("key",None)

    return {
        "controls":controls, "span_fraction":float(span_fraction),
        "grid_points":int(grid_points), "evaluated_points":len(points),
        "feasible_points":len(valid), "points":points,
        "best_fold_candidates":fold, "best_cusp_candidates":cusp,
        "a2_sign_crossings":crossings[:10],
        "seed_diagnostic":seed_diag,
        "message":"Search completed. Rankings identify regions for follow-up continuation; they are not catastrophe classifications.",
        "caution":"A cusp claim requires following the fold locus and verifying nondegeneracy/transversality. Search scores only prioritize numerical experiments.",
    }


def verify_fold_candidate(
    robot, targets: list[TargetSpec], context: dict,
    control_metrics: list[str], control_values: list[float],
    *, solve_iterations: int = 15, step_limit: float = 0.12,
    refinement_span: float = 0.15, refinement_points: int = 9,
) -> dict:
    """Numerically verify a corank-one fold candidate conservatively.

    The routine first refines the candidate by varying the second control while
    holding the first control and all remaining enabled targets fixed.  It then
    checks rank loss/corank, quadratic nondegeneracy, output-control
    transversality, derivative-step robustness, and a two-seed branch probe.
    Design-bound and pseudo-arclength turning-point tests are reported as
    inconclusive when the production model does not expose those structures.
    """
    if len(control_metrics) != 2 or len(control_values) != 2:
        raise ValueError("Fold verification requires exactly two control metrics and values")
    names=[t.metric for t in targets]
    if any(m not in names for m in control_metrics):
        raise ValueError("Fold verification controls must be enabled target metrics")
    if refinement_points not in {5,7,9,11}:
        raise ValueError("refinement_points must be 5, 7, 9, or 11")

    base={t.metric:float(t.value) for t in targets}
    base[control_metrics[0]]=float(control_values[0])
    base[control_metrics[1]]=float(control_values[1])
    factors=np.linspace(1.0-refinement_span,1.0+refinement_span,refinement_points)
    refined=[]; warm=robot
    for fac in factors:
        specs=[]
        for t in targets:
            v=base[t.metric]
            rel=t.relation
            if t.metric in control_metrics:
                rel="equal"
            if t.metric==control_metrics[1]:
                v=base[t.metric]*float(fac)
            specs.append(TargetSpec(t.metric,rel,v,t.tolerance))
        try:
            solved=run_target_search(warm,specs,context,max_iterations=solve_iterations,
                step_limit=step_limit,secondary_objectives=[],null_step_fraction=0.0,
                safety_cap=max(30,solve_iterations))
            cand=solved["final_robot"]
            _,merit,feasible,_=_target_state(target_metrics(cand,specs,context),specs)
            diag=catastrophe_diagnostics(cand,specs,context,force_deep=True)
            refined.append({"factor":float(fac),"target_value":float(base[control_metrics[1]]*fac),
                "feasible":bool(feasible),"target_merit":float(merit),"robot":cand,"targets":specs,"diag":diag})
            if feasible: warm=cand
        except Exception as exc:
            refined.append({"factor":float(fac),"target_value":float(base[control_metrics[1]]*fac),"feasible":False,"error":str(exc)})
    valid=[r for r in refined if r.get("feasible") and r.get("diag")]
    if not valid:
        return {"status":"inconclusive","classification":"Fold verification inconclusive","tests":[],
                "message":"No feasible points were obtained during local singularity refinement."}
    best=min(valid,key=lambda r:float(r["diag"].get("sigma_ratio",1.0)))
    cand=best["robot"]; specs=best["targets"]
    J=target_jacobian(cand,specs,context)
    U,sv,VT=np.linalg.svd(J,full_matrices=False)
    smax=float(sv[0]); smin=float(sv[-1]); ratio=smin/max(smax,1e-15)
    second_ratio=float(sv[-2]/smax) if len(sv)>=2 else 1.0
    u=U[:,-1]; v=VT[-1,:]
    idx=names.index(control_metrics[1])
    trans=float(abs(u[idx]))

    # Recompute higher-order terms over several finite-difference scales.
    robustness=[]
    for h in (1e-3,2e-3,4e-3):
        d=catastrophe_diagnostics(cand,specs,context,force_deep=True)
        # catastrophe_diagnostics uses the production default derivative step;
        # call the lower-level routine when a different h is required.
        if h != 2e-3:
            def ev(x):
                rr=cand.with_design_vector(np.asarray(x,float)); vals=target_metrics(rr,specs,context)
                return np.log(np.asarray([vals[n] for n in names],float))
            d=diagnose_map_singularity(J,ev,cand.design_vector(),names,force_deep=True,derivative_step=h)
        robustness.append({"step":h,"a2":float(d.get("projected_second_a2",0.0)),
                           "a3":float(d.get("projected_third_a3",0.0))})
    a2s=np.asarray([r["a2"] for r in robustness],float)
    robust=bool(np.all(np.sign(a2s)==np.sign(a2s[0])) and np.ptp(a2s)/max(abs(np.mean(a2s)),1e-12)<0.35)
    a2=float(best["diag"].get("projected_second_a2",0.0))

    tests=[]
    def add(name,status,value,criterion,note): tests.append({"name":name,"status":status,"value":value,"criterion":criterion,"note":note})
    add("Singularity localization","PASS" if ratio<=1e-2 else "INCONCLUSIVE",
        f"sigma_min/sigma_max={ratio:.3e}","<= 1e-2",
        "Local refinement reached numerical rank loss." if ratio<=1e-2 else "Candidate strengthened but did not reach the verification threshold.")
    add("Corank-one spectrum","PASS" if ratio<=1e-2 and second_ratio>=5e-2 else "INCONCLUSIVE",
        f"smallest ratio={ratio:.3e}; next ratio={second_ratio:.3e}","one singular value collapses while the next remains separated","Full active singular spectrum checked.")
    add("Quadratic nondegeneracy","PASS" if abs(a2)>=1e-3 else "FAIL",f"a2={a2:.4e}","|a2| >= 1e-3","A generic fold requires nonzero projected quadratic curvature.")
    add("Control transversality","PASS" if trans>=0.05 else "INCONCLUSIVE",f"|u^T e_control|={trans:.4f}",">= 0.05","Tests whether the selected unfolding control cuts across the singular image locally.")
    add("Derivative robustness","PASS" if robust else "INCONCLUSIVE",str(robustness),"a2 sign stable and spread < 35%","Repeated at three finite-difference scales.")

    # Two-seed local branch probe at the refined target level.
    branches=[]
    for sign in (-1.0,1.0):
        try:
            seed=cand.with_design_vector(cand.design_vector()+sign*0.06*v)
            sol=run_target_search(seed,specs,context,max_iterations=max(15,solve_iterations),step_limit=step_limit,
                secondary_objectives=[],null_step_fraction=0.0,safety_cap=max(30,solve_iterations))
            rr=sol["final_robot"]; _,mer,ok,_=_target_state(target_metrics(rr,specs,context),specs)
            branches.append({"feasible":bool(ok),"merit":float(mer),"x":rr.design_vector(),"t1":list(map(float,rr.t1)),"t2":list(map(float,rr.t2))})
        except Exception as exc: branches.append({"feasible":False,"error":str(exc)})
    dist=None; branch_pass=False
    if len(branches)==2 and all(b.get("feasible") for b in branches):
        dist=float(np.linalg.norm(branches[0]["x"]-branches[1]["x"]))
        branch_pass=dist>=0.02
    add("Two-branch probe","PASS" if branch_pass else "INCONCLUSIVE",
        "distance="+(f"{dist:.4f}" if dist is not None else "not resolved"),"two feasible distinct local designs","Opposite critical-direction seeds are solved back to the same target level.")
    add("Active design bounds","INCONCLUSIVE","bounds not exposed by current robot model","no active bound causes rank loss","The production Planar2DOFRobot currently has no explicit section-variable bounds to test.")
    add("Pseudo-arclength turning point","INCONCLUSIVE","not run","turning point on continued solution branch","Production catastrophe path does not yet expose pseudo-arclength continuation.")

    required=[t for t in tests if t["name"] in {"Singularity localization","Corank-one spectrum","Quadratic nondegeneracy","Control transversality","Derivative robustness"}]
    verified=all(t["status"]=="PASS" for t in required) and branch_pass
    classification="Fold singularity numerically verified" if verified else "Fold candidate — verification incomplete"
    return {"status":"verified" if verified else "candidate","classification":classification,
        "controls":control_metrics,"refined_control_values":[float(base[control_metrics[0]]),float(best["target_value"])],
        "sigma_values":list(map(float,sv)),"sigma_ratio":float(ratio),"condition":float(smax/smin) if smin>0 else None,
        "a2":a2,"a3":best["diag"].get("projected_third_a3"),"transversality":trans,
        "tests":tests,"robustness":robustness,
        "refinement":[{"factor":r["factor"],"target_value":r["target_value"],"feasible":r.get("feasible",False),
                       "sigma_ratio":(r.get("diag") or {}).get("sigma_ratio"),"sigma_min":(r.get("diag") or {}).get("sigma_min"),
                       "condition":(r.get("diag") or {}).get("condition"),"error":r.get("error")} for r in refined],
        "design":{"t1":list(map(float,cand.t1)),"t2":list(map(float,cand.t2))},
        "branches":[{k:v for k,v in b.items() if k!="x"} for b in branches],
        "message":"Verification is deliberately conservative. A Thom-catastrophe claim additionally requires an appropriate smooth potential/equilibrium interpretation; this endpoint verifies fold geometry of the selected design-to-performance map."}

def _secondary_gradient(robot, metric: str) -> np.ndarray:
    """Gradient of a secondary metric in log-design coordinates.

    Positive metrics use log-gradients so very differently scaled quantities can
    be combined meaningfully after null-space projection. Smoothness can be zero,
    so it falls back to an ordinary central difference near zero.
    """
    x = robot.design_vector()
    g = np.zeros_like(x)
    h = 1e-4
    base = secondary_value(robot, metric)
    use_log = base > 1e-12
    for j in range(len(x)):
        xp = x.copy(); xm = x.copy(); xp[j] += h; xm[j] -= h
        fp = secondary_value(robot.with_design_vector(xp), metric)
        fm = secondary_value(robot.with_design_vector(xm), metric)
        if use_log and fp > 0 and fm > 0:
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


def _secondary_values(robot, specs: list[SecondarySpec]) -> dict[str, float]:
    return {s.metric: secondary_value(robot, s.metric) for s in specs}


def _secondary_loss_change(base: dict[str, float], trial: dict[str, float], specs: list[SecondarySpec]) -> float:
    """Weighted local relative loss change; negative means improvement."""
    total = 0.0
    wsum = 0.0
    for s in specs:
        w = max(0.0, float(s.weight))
        if w == 0:
            continue
        b = float(base[s.metric]); v = float(trial[s.metric])
        scale = max(abs(b), 1e-8)
        rel = (v - b) / scale
        total += w * (rel if s.direction == "min" else -rel)
        wsum += w
    return total / wsum if wsum else 0.0


def _secondary_details(values: dict[str, float], specs: list[SecondarySpec]) -> list[dict]:
    return [{
        "metric": s.metric,
        "direction": s.direction,
        "weight": float(s.weight),
        "current": float(values[s.metric]),
    } for s in specs]


def _snapshot(
    iteration: int,
    robot,
    metrics: dict,
    targets: list[TargetSpec],
    J: np.ndarray | None,
    secondary_specs: list[SecondarySpec],
    secondary_values: dict[str, float] | None = None,
    step_norm: float = 0.0,
    active_step_norm: float = 0.0,
    null_step_norm: float = 0.0,
    alpha: float = 0.0,
    note: str = "",
) -> dict:
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
    secvals = secondary_values if secondary_values is not None else _secondary_values(robot, secondary_specs)
    return {
        "iteration": int(iteration),
        "metrics": {k: float(v) for k, v in metrics.items()},
        "targets": details,
        "target_merit": merit,
        "targets_satisfied": bool(satisfied),
        "secondary_objectives": _secondary_details(secvals, secondary_specs),
        "singular_values": singular_values,
        "rank": rank,
        "condition": condition,
        "step_norm": float(step_norm),
        "active_step_norm": float(active_step_norm),
        "null_step_norm": float(null_step_norm),
        "line_search_alpha": float(alpha),
        "design_vector": robot.design_vector().tolist(),
        "t1": list(map(float, robot.t1)),
        "t2": list(map(float, robot.t2)),
        "note": note,
    }


def _combined_null_direction(robot, J: np.ndarray, specs: list[SecondarySpec]) -> np.ndarray:
    """Weighted blend of normalized projected secondary directions.

    Each objective is projected into the current target-null space first and
    normalized before weighting. This makes user weights comparable even when
    objective units and raw gradient magnitudes differ substantially.
    """
    n = len(robot.design_vector())
    if not specs:
        return np.zeros(n)
    Pn = _null_projector(J, n)
    combined = np.zeros(n)
    for spec in specs:
        grad = _secondary_gradient(robot, spec.metric)
        signed = -grad if spec.direction == "min" else grad
        d = Pn @ signed
        nd = float(np.linalg.norm(d))
        if nd > 1e-10:
            combined += max(0.0, float(spec.weight)) * (d / nd)
    nc = float(np.linalg.norm(combined))
    return combined / nc if nc > 1e-10 else np.zeros(n)


def run_target_search(
    robot,
    targets: list[TargetSpec],
    context: dict,
    max_iterations: int | None = 10,
    step_limit: float = 0.16,
    secondary_objectives: list[SecondarySpec] | None = None,
    null_step_fraction: float = 0.25,
    # Backward-compatible v0.12 arguments:
    secondary_objective: str = "none",
    secondary_weight: float = 0.25,
    progress: Callable[[dict], None] | None = None,
    safety_cap: int = 100,
):
    """Iterative target search with active SVD correction + null-space optimization.

    The active step seeks the global performance targets. A separate weighted
    null-space step improves selected secondary metrics while preserving those
    targets to first order. Both subspaces are recomputed after every accepted
    nonlinear design update.
    """
    if not targets:
        raise ValueError("Select at least one global performance target")
    if step_limit <= 0:
        raise ValueError("step_limit must be positive")
    if not (0.0 <= null_step_fraction <= 1.0):
        raise ValueError("null_step_fraction must be between 0 and 1")

    specs = list(secondary_objectives or [])
    if not specs and secondary_objective != "none":
        if secondary_objective not in SECONDARY_METRICS:
            raise ValueError("Unsupported secondary objective")
        direction = "max" if secondary_objective == "maneuverability" else "min"
        specs = [SecondarySpec(secondary_objective, direction, max(0.0, float(secondary_weight)))]
    for s in specs:
        if s.metric not in SECONDARY_METRICS:
            raise ValueError(f"Unsupported secondary objective: {s.metric}")
        if s.direction not in {"min", "max"}:
            raise ValueError(f"Unsupported secondary direction: {s.direction}")
        if s.weight < 0:
            raise ValueError("Secondary objective weights must be nonnegative")
    specs = [s for s in specs if s.weight > 0]

    limit = safety_cap if max_iterations is None else min(int(max_iterations), safety_cap)
    if limit < 1:
        raise ValueError("max_iterations must be at least 1, or blank for unconstrained")

    current = robot
    trace = []
    metrics = all_metrics(current, context)
    J = target_jacobian(current, targets, context)
    secvals = _secondary_values(current, specs)
    snap = _snapshot(0, current, metrics, targets, J, specs, secvals, note="Initial design")
    trace.append(snap)
    if progress:
        progress(snap)

    status = "running"
    message = ""

    for k in range(1, limit + 1):
        delta_y, merit0, targets_ok, _ = _target_state(metrics, targets)

        if targets_ok and not specs:
            status = "targets_satisfied"
            message = "All enabled performance targets are within tolerance."
            break

        J = target_jacobian(current, targets, context)
        if J.size:
            U, s, VT = np.linalg.svd(J, full_matrices=False)
            damping = max(1e-4, (s[0] if len(s) else 1.0) * 1e-5)
            inv = s / (s*s + damping*damping)
            active_step = VT.T @ (inv * (U.T @ delta_y))
        else:
            active_step = np.zeros_like(current.design_vector())

        null_dir = _combined_null_direction(current, J, specs)
        null_step = np.zeros_like(active_step)
        if np.linalg.norm(null_dir) > 0:
            null_budget = step_limit * float(null_step_fraction)
            # Once all primary targets are feasible, allow the whole trust region
            # to be used for secondary navigation along the target manifold.
            if targets_ok:
                null_budget = step_limit
            null_step = null_dir * null_budget

        # Keep target correction dominant while the design is still infeasible.
        step = active_step + null_step
        ns = float(np.linalg.norm(step))
        if ns > step_limit:
            step *= step_limit / ns
            scale = step_limit / ns
            active_step = active_step * scale
            null_step = null_step * scale
            ns = step_limit

        active_ns = float(np.linalg.norm(active_step))
        null_ns = float(np.linalg.norm(null_step))
        if ns < 1e-9:
            status = "stalled"
            message = "The evolving SVD produced no meaningful feasible active or null-space step."
            break

        x0 = current.design_vector()
        best = None
        base_secondary = secvals

        for alpha in (1.0, 0.5, 0.25, 0.125, 0.0625):
            trial = current.with_design_vector(x0 + alpha * step)
            try:
                m_target = target_metrics(trial, targets, context)
                _, merit, ok, _ = _target_state(m_target, targets)
                trial_secondary = _secondary_values(trial, specs)
                sec_change = _secondary_loss_change(base_secondary, trial_secondary, specs)
            except Exception:
                continue

            if targets_ok and specs:
                # Once feasible, remain inside every primary tolerance band and
                # require an actual improvement in the weighted secondary goals.
                improved = ok and sec_change < -1e-7
                score = (0.0 if ok else 1e3 + merit, sec_change)
            else:
                # Before feasibility, target progress is lexicographically first;
                # secondary improvement is used only to choose among target-improving moves.
                target_improved = merit < merit0 - max(1e-8, 1e-4 * max(merit0, 1e-8))
                improved = target_improved
                score = (merit, sec_change)

            if improved and (best is None or score < best[0]):
                best = (score, alpha, trial, m_target, trial_secondary)

        if best is None:
            status = "stalled"
            if targets_ok and specs:
                message = "Targets remain satisfied, but no null-space/backtracking step improved the selected secondary objectives."
            else:
                message = "No trust-region/backtracking step improved the nonlinear target error."
            break

        _, alpha, current, _, secvals = best
        metrics = all_metrics(current, context)
        Jnew = target_jacobian(current, targets, context)
        snap = _snapshot(
            k,
            current,
            metrics,
            targets,
            Jnew,
            specs,
            secvals,
            step_norm=float(alpha * ns),
            active_step_norm=float(alpha * active_ns),
            null_step_norm=float(alpha * null_ns),
            alpha=float(alpha),
            note=(
                "Accepted nonlinear step; active target correction and weighted target-null-space optimization were recomputed at this design."
                if specs else
                "Accepted nonlinear target step; Jacobian/SVD recomputed at this design."
            ),
        )
        trace.append(snap)
        if progress:
            progress(snap)

        if snap["targets_satisfied"] and not specs:
            status = "targets_satisfied"
            message = "All enabled performance targets are within tolerance."
            break

    else:
        status = "iteration_limit"
        if max_iterations is None:
            message = f"Stopped at the numerical safety cap of {safety_cap} iterations."
        else:
            message = f"Stopped after the requested maximum of {limit} iterations."

    if status == "running":
        _, _, ok, _ = _target_state(metrics, targets)
        if ok and not specs:
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
        "secondary_objectives": [
            {"metric": s.metric, "direction": s.direction, "weight": float(s.weight)} for s in specs
        ],
        "null_step_fraction": float(null_step_fraction),
        "trace": trace,
        "final_robot": current,
    }
