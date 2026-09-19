from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np
from scipy.linalg import eigvalsh
from robot_codesign.fem.two_link_variable import assemble_two_link_variable_sections

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


def continue_fold_candidate(
    robot, targets: list[TargetSpec], context: dict,
    control_metrics: list[str], control_values: list[float],
    *, steps_each_side: int = 6, arclength_step: float = 0.025,
    corrector_iterations: int = 5, section_min: float | None = None,
    section_max: float | None = None,
) -> dict:
    """Follow a selected fold candidate with a minimum-norm pseudo-arclength path.

    The augmented unknown is y=[log(section variables), eta], where eta is the
    log multiplier of the second selected control target.  Because the design
    problem is redundant, the augmented level set is not intrinsically 1-D.
    We therefore select a reproducible 1-D slice: the initial tangent is the
    projection of the control direction into null([J,-e]), subsequent tangents
    are the closest null-space projections of the previous tangent, and Newton
    correctors use minimum-norm updates plus the pseudo-arclength hyperplane.
    This is a numerical continuation slice, not a claim of a globally unique branch.
    """
    if len(control_metrics)!=2 or len(control_values)!=2:
        raise ValueError("Fold continuation requires exactly two controls and values")
    if not (2 <= steps_each_side <= 12): raise ValueError("steps_each_side must be 2..12")
    if not (0.002 <= arclength_step <= 0.10): raise ValueError("arclength_step must be 0.002..0.10")
    if not (2 <= corrector_iterations <= 8): raise ValueError("corrector_iterations must be 2..8")
    if section_min is not None and section_min <= 0: raise ValueError("section_min must be positive")
    if section_max is not None and section_max <= 0: raise ValueError("section_max must be positive")
    if section_min is not None and section_max is not None and section_min >= section_max:
        raise ValueError("section_min must be less than section_max")

    names=[t.metric for t in targets]
    if any(m not in names for m in control_metrics):
        raise ValueError("Continuation controls must be enabled targets")
    cidx=names.index(control_metrics[1])
    base={t.metric:float(t.value) for t in targets}
    base[control_metrics[0]]=float(control_values[0]); base[control_metrics[1]]=float(control_values[1])
    # Both controls are level-set coordinates during continuation.
    def specs_for_eta(eta):
        out=[]
        for t in targets:
            v=base[t.metric]; rel=t.relation
            if t.metric in control_metrics: rel='equal'
            if t.metric==control_metrics[1]: v=base[t.metric]*float(np.exp(eta))
            out.append(TargetSpec(t.metric,rel,v,t.tolerance))
        return out
    def residual(x,eta):
        rr=robot.with_design_vector(x); vals=target_metrics(rr,specs_for_eta(eta),context)
        actual=np.asarray([float(vals[n]) for n in names],float)
        targ=np.asarray([float(next(t.value for t in specs_for_eta(eta) if t.metric==n)) for n in names],float)
        return np.log(actual)-np.log(targ)
    def augmented_jac(x,eta):
        rr=robot.with_design_vector(x); J=target_jacobian(rr,specs_for_eta(eta),context)
        col=np.zeros((len(names),1)); col[cidx,0]=-1.0
        return np.hstack([J,col])
    def tangent(A, preferred):
        P=np.eye(A.shape[1])-np.linalg.pinv(A)@A
        q=P@preferred
        nq=np.linalg.norm(q)
        if nq<1e-10: return None
        return q/nq
    def point_payload(y,tan=None,converged=True,iters=0):
        x=y[:-1]; eta=float(y[-1]); rr=robot.with_design_vector(x); sp=specs_for_eta(eta)
        J=target_jacobian(rr,sp,context); sv=np.linalg.svd(J,compute_uv=False)
        vals=target_metrics(rr,sp,context); r=residual(x,eta)
        thick=np.r_[np.asarray(rr.t1,float),np.asarray(rr.t2,float)]
        active=[]
        if section_min is not None:
            active += [f"t{i+1}:lower" for i,v in enumerate(thick) if v <= section_min*1.01]
        if section_max is not None:
            active += [f"t{i+1}:upper" for i,v in enumerate(thick) if v >= section_max*.99]
        return {"eta":eta,"control_value":float(base[control_metrics[1]]*np.exp(eta)),
            "sigma_min":float(sv[-1]),"sigma_ratio":float(sv[-1]/max(sv[0],1e-15)),
            "condition":float(sv[0]/sv[-1]) if sv[-1]>0 else None,
            "residual_inf":float(np.max(np.abs(r))),"converged":bool(converged),"corrector_iterations":iters,
            "tangent_control":None if tan is None else float(tan[-1]),
            "t1":list(map(float,rr.t1)),"t2":list(map(float,rr.t2)),
            "section_min_actual":float(np.min(thick)),"section_max_actual":float(np.max(thick)),
            "active_bounds":active,"performance":{n:float(vals[n]) for n in names}}

    y0=np.r_[robot.design_vector(),0.0]
    A0=augmented_jac(y0[:-1],y0[-1]); pref=np.zeros_like(y0); pref[-1]=1.0
    t0=tangent(A0,pref)
    if t0 is None: raise ValueError("Could not construct an initial continuation tangent")

    def march(sign):
        y=y0.copy(); t=sign*t0.copy(); out=[]
        for _ in range(steps_each_side):
            yp=y+arclength_step*t; yn=yp.copy(); conv=False; nit=0
            for nit in range(1,corrector_iterations+1):
                rr=residual(yn[:-1],yn[-1]); arc=float(np.dot(t,yn-yp)); g=np.r_[rr,arc]
                if np.max(np.abs(g))<2e-5: conv=True; break
                A=augmented_jac(yn[:-1],yn[-1]); B=np.vstack([A,t])
                dy=-np.linalg.pinv(B)@g
                # keep a failed Newton step from exploding the positive section model
                nd=np.linalg.norm(dy)
                if nd>.20: dy*=.20/nd
                yn+=dy
            A=augmented_jac(yn[:-1],yn[-1]); tn=tangent(A,t)
            if tn is None: break
            if np.dot(tn,t)<0: tn=-tn
            out.append(point_payload(yn,tn,conv,nit)); y=yn; t=tn
            if not conv: break
        return out
    neg=march(-1.0); pos=march(1.0)
    center=point_payload(y0,t0,True,0)
    trace=list(reversed(neg))+[center]+pos
    # Turning point: tangent component of the continued control changes sign.
    turns=[]
    for i in range(1,len(trace)):
        a=trace[i-1].get('tangent_control'); b=trace[i].get('tangent_control')
        if a is not None and b is not None and a*b<0:
            turns.append(i)
    best=min(trace,key=lambda q:q['sigma_ratio'])

    # Construct two distinct designs at one common performance level on the
    # two-solution side of the detected fold.  This is deliberately derived
    # from the computed continuation trace rather than from an illustrative
    # perturbation.  The common level is chosen inside the overlap of the two
    # local branches and each branch is corrected independently to that level.
    paired_designs=None
    if turns and len(trace) >= 5:
        ib=int(np.argmin([q['sigma_ratio'] for q in trace]))
        left=trace[:ib]; right=trace[ib+1:]
        if left and right:
            fold_val=float(trace[ib]['control_value'])
            left_hi=max(float(q['control_value']) for q in left)
            right_hi=max(float(q['control_value']) for q in right)
            overlap_hi=min(left_hi,right_hi)
            if overlap_hi > fold_val*(1.0+1e-7):
                pair_level=fold_val + 0.35*(overlap_hi-fold_val)
                qa=min(left,key=lambda q:abs(float(q['control_value'])-pair_level))
                qb=min(right,key=lambda q:abs(float(q['control_value'])-pair_level))
                pair_specs=[]
                for t in targets:
                    v=base[t.metric]; rel=t.relation
                    if t.metric in control_metrics: rel='equal'
                    if t.metric==control_metrics[1]: v=pair_level
                    pair_specs.append(TargetSpec(t.metric,rel,float(v),t.tolerance))
                solved_pair=[]
                for qseed in (qa,qb):
                    try:
                        xseed=np.log(np.maximum(np.r_[np.asarray(qseed['t1'],float),np.asarray(qseed['t2'],float)],1e-15))
                        seed=robot.with_design_vector(xseed)
                        sol=run_target_search(seed,pair_specs,context,max_iterations=max(20,corrector_iterations*4),
                            step_limit=min(0.08,max(0.02,arclength_step*2.0)),secondary_objectives=[],
                            null_step_fraction=0.0,safety_cap=max(30,corrector_iterations*5))
                        rr=sol['final_robot']; vals=target_metrics(rr,pair_specs,context)
                        actual=np.asarray([float(vals[n]) for n in names],float)
                        targ=np.asarray([float(next(t.value for t in pair_specs if t.metric==n)) for n in names],float)
                        resid=float(np.max(np.abs(np.log(actual)-np.log(targ))))
                        solved_pair.append({'robot':rr,'performance':{n:float(vals[n]) for n in names},'residual_inf':resid})
                    except Exception:
                        solved_pair.append(None)
                if all(z is not None for z in solved_pair):
                    ra,rb=solved_pair
                    xa=ra['robot'].design_vector(); xb=rb['robot'].design_vector()
                    dist=float(np.linalg.norm(xa-xb))
                    pa=ra['performance']; pb=rb['performance']
                    diffs={n:float(abs(pa[n]-pb[n])) for n in names}
                    rel_diffs={n:float(abs(pa[n]-pb[n])/max(0.5*(abs(pa[n])+abs(pb[n])),1e-15)) for n in names}
                    ta=np.r_[np.asarray(ra['robot'].t1,float),np.asarray(ra['robot'].t2,float)]
                    tb=np.r_[np.asarray(rb['robot'].t1,float),np.asarray(rb['robot'].t2,float)]
                    dt=tb-ta
                    dt_rel=dt/np.maximum(0.5*(np.abs(ta)+np.abs(tb)),1e-15)
                    # Compare branch separation with the local weakest design direction.
                    # The sign of an SVD vector is arbitrary, so the alignment magnitude is reported.
                    xmid=0.5*(xa+xb); rmid=robot.with_design_vector(xmid)
                    Jmid=target_jacobian(rmid,pair_specs,context)
                    _,_,Vtmid=np.linalg.svd(Jmid,full_matrices=False)
                    vcrit=np.asarray(Vtmid[-1],float)
                    dx=xb-xa
                    parallel=float(np.dot(vcrit,dx))
                    perp_vec=dx-parallel*vcrit
                    perp=float(np.linalg.norm(perp_vec))
                    frac=float(abs(parallel)/max(np.linalg.norm(dx),1e-15))
                    paired_designs={
                        'control_metric':control_metrics[1],'common_control_target':float(pair_level),
                        'design_distance_log':dist,
                        'design_distance_relative':float(np.linalg.norm(dt)/max(0.5*(np.linalg.norm(ta)+np.linalg.norm(tb)),1e-15)),
                        'critical_direction_projection':parallel,
                        'critical_direction_projection_abs':abs(parallel),
                        'orthogonal_distance_log':perp,
                        'critical_direction_fraction':frac,
                        'branch_a':{'t1':list(map(float,ra['robot'].t1)),'t2':list(map(float,ra['robot'].t2)),
                                    'performance':pa,'residual_inf':ra['residual_inf']},
                        'branch_b':{'t1':list(map(float,rb['robot'].t1)),'t2':list(map(float,rb['robot'].t2)),
                                    'performance':pb,'residual_inf':rb['residual_inf']},
                        'section_absolute_differences':list(map(float,dt)),
                        'section_relative_differences':list(map(float,dt_rel)),
                        'absolute_performance_differences':diffs,
                        'relative_performance_differences':rel_diffs,
                        'interpretation':'Two independently corrected structural designs on opposite local branches at the same selected performance level.'
                    }

    # Local fold normal-form diagnostic.  Use the best rank-loss point as the
    # critical design x_c, project nearby designs onto its weakest right-singular
    # direction, and test whether the continued control obeys mu ~ C z^2.
    # This is a numerical local-reduction diagnostic, not by itself a proof that
    # the full engineering model is a classical gradient-potential catastrophe.
    normal_form=None
    try:
        xc_robot=robot.with_design_vector(np.log(np.maximum(np.r_[np.asarray(best['t1'],float),np.asarray(best['t2'],float)],1e-15)))
        eta_c=float(best['eta'])
        Jc=target_jacobian(xc_robot,specs_for_eta(eta_c),context)
        _,svc,Vtc=np.linalg.svd(Jc,full_matrices=False)
        vc=np.asarray(Vtc[-1],float); xc=xc_robot.design_vector()
        control_c=float(best['control_value'])
        pts=[]
        for i,q in enumerate(trace):
            if not q.get('converged',False) or float(q.get('residual_inf',1.0))>1e-4: continue
            xq=np.log(np.maximum(np.r_[np.asarray(q['t1'],float),np.asarray(q['t2'],float)],1e-15))
            z=float(np.dot(vc,xq-xc))
            mu=float(q['control_value'])-control_c
            mu_log=float(np.log(float(q['control_value'])/control_c))
            pts.append({'trace_index':i,'z':z,'z2':z*z,'mu':mu,'mu_log':mu_log,
                        'control_value':float(q['control_value']),'residual_inf':float(q['residual_inf'])})
        fits=[]
        ordered=sorted(pts,key=lambda q:abs(q['z']))
        for frac,label in ((1.0,'100%'),(.75,'75%'),(.50,'50%')):
            n=max(5,int(np.ceil(len(ordered)*frac)))
            use=ordered[:min(n,len(ordered))]
            if len(use)<5: continue
            z=np.asarray([q['z'] for q in use],float); mu=np.asarray([q['mu'] for q in use],float)
            # General cubic fit diagnoses coordinate leakage and higher order terms.
            X=np.column_stack([np.ones_like(z),z,z*z,z*z*z])
            coef=np.linalg.lstsq(X,mu,rcond=None)[0]; pred=X@coef
            ss=float(np.sum((mu-np.mean(mu))**2)); r2=1.0-float(np.sum((mu-pred)**2))/max(ss,1e-30)
            # Canonical fold test constrained through the critical point: mu=C z^2.
            z2=z*z; C=float(np.dot(z2,mu)/max(np.dot(z2,z2),1e-30)); pred2=C*z2
            r2q=1.0-float(np.sum((mu-pred2)**2))/max(float(np.sum(mu*mu)),1e-30)
            zscale=float(np.max(np.abs(z)))
            quad=abs(float(coef[2]))*zscale*zscale
            cubic=abs(float(coef[3]))*zscale**3
            linear=abs(float(coef[1]))*zscale
            fits.append({'window':label,'points':len(use),'max_abs_z':zscale,
                         'c0':float(coef[0]),'c1':float(coef[1]),'c2':float(coef[2]),'c3':float(coef[3]),
                         'r2_cubic':r2,'quadratic_C':C,'r2_mu_equals_Cz2':r2q,
                         'linear_to_quadratic':float(linear/max(quad,1e-30)),
                         'cubic_to_quadratic':float(cubic/max(quad,1e-30))})
        closest=fits[-1] if fits else None
        supported=bool(closest and abs(closest['quadratic_C'])>1e-12 and closest['r2_mu_equals_Cz2']>=.90 and closest['cubic_to_quadratic']<=.35)
        normal_form={'status':'supported' if supported else 'diagnostic',
            'critical_control_metric':control_metrics[1],'critical_control_value':control_c,
            'critical_sigma_values':list(map(float,svc)),'critical_direction':list(map(float,vc)),
            'points':pts,'fits':fits,
            'canonical_relation':'mu ≈ C z^2, with z = v_c^T(x-x_c) and mu = control-control_c',
            'potential_relation':'Integrating the reduced scalar form g(z,mu)=mu-C z^2 gives V(z,mu)=mu*z-(C/3)z^3 up to smooth rescaling and higher-order terms.',
            'interpretation':('The shrinking-neighborhood continuation data are consistent with a local quadratic fold normal form.' if supported else 'The continuation data do not yet satisfy the configured numerical normal-form support criteria.'),
            'caution':'This projection test supports local fold normal-form equivalence. A rigorous Thom classification still requires a justified smooth reduction/unfolding interpretation; the constructed reduced potential is mathematical, not mechanical potential energy.'}
    except Exception as e:
        normal_form={'status':'unavailable','error':str(e),'interpretation':'Normal-form diagnostic could not be evaluated.'}


    # Critical-point coordinate-invariance diagnostic.
    #
    # Keep the SAME physical critical design x_c and physical scalar critical
    # coordinate z = v_c^T (x-x_c). Under a smooth invertible linear change
    # x-x_c = A y, the covector representing this scalar coordinate transforms
    # as ell_y = A^T v_c, so z = ell_y^T y exactly.  This is a coordinate-
    # invariance check of the projected fold germ; it is deliberately NOT
    # described as a Lyapunov-Schmidt complement-invariance test, because a
    # genuine complement test would have to construct the regular complement
    # and re-solve the regular equations in each decomposition.
    coordinate_invariance=None
    try:
        if normal_form and normal_form.get('status') in {'supported','diagnostic'}:
            good=[]
            for i,q in enumerate(trace):
                if not q.get('converged',False) or float(q.get('residual_inf',1.0))>1e-4:
                    continue
                xq=np.log(np.maximum(np.r_[np.asarray(q['t1'],float),np.asarray(q['t2'],float)],1e-15))
                good.append((i,q,xq))
            nvar=len(xc)
            transforms=[]
            I=np.eye(nvar)
            transforms.append(("identity",I))
            if nvar:
                d1=np.linspace(0.85,1.15,nvar)
                d2=np.linspace(1.15,0.85,nvar)
                transforms.append(("scaled-A",np.diag(d1)))
                transforms.append(("scaled-B",np.diag(d2)))
            if nvar>=2:
                A=I.copy(); A[0,-1]=0.18
                transforms.append(("shear-1",A))
                B=I.copy(); B[-1,0]=-0.18
                transforms.append(("shear-2",B))

            tests=[]
            for label,A in transforms:
                Ainv=np.linalg.inv(A)
                # z is a scalar physical projection.  v_c acts here as a
                # covector; therefore its coordinate representation is A^T v_c.
                ell_y=A.T@vc
                pts=[]
                max_z_error=0.0
                for i,q,xq in good:
                    dx=xq-xc
                    dy=Ainv@dx
                    z=float(np.dot(ell_y,dy))
                    z_phys=float(np.dot(vc,dx))
                    max_z_error=max(max_z_error,abs(z-z_phys))
                    mu=float(q['control_value'])-control_c
                    pts.append((abs(z),z,mu,i))
                use=sorted(pts,key=lambda r:r[0])[:max(5,int(np.ceil(.5*len(pts))))]
                z=np.asarray([r[1] for r in use],float)
                mu=np.asarray([r[2] for r in use],float)
                z2=z*z
                C=float(np.dot(z2,mu)/max(np.dot(z2,z2),1e-30))
                pred=C*z2
                r2q=1.0-float(np.sum((mu-pred)**2))/max(float(np.sum(mu*mu)),1e-30)
                X=np.column_stack([np.ones_like(z),z,z*z,z*z*z])
                coef=np.linalg.lstsq(X,mu,rcond=None)[0]
                zscale=float(np.max(np.abs(z)))
                linear=abs(float(coef[1]))*zscale
                quad=abs(float(coef[2]))*zscale*zscale
                cubic=abs(float(coef[3]))*zscale**3
                tests.append({
                    'coordinate_system':label,
                    'max_physical_z_error':float(max_z_error),
                    'quadratic_C':C,
                    'r2_mu_equals_Cz2':r2q,
                    'linear_to_quadratic':float(linear/max(quad,1e-30)),
                    'cubic_to_quadratic':float(cubic/max(quad,1e-30)),
                    'points':len(use),
                })

            reliable=[t for t in tests if np.isfinite(t['quadratic_C']) and np.isfinite(t['r2_mu_equals_Cz2'])]
            if reliable:
                signs=[np.sign(t['quadratic_C']) for t in reliable if abs(t['quadratic_C'])>1e-12]
                same_sign=bool(signs and all(s==signs[0] for s in signs))
                max_zerr=max(t['max_physical_z_error'] for t in reliable)
                min_r2=min(t['r2_mu_equals_Cz2'] for t in reliable)
                max_lq=max(t['linear_to_quadratic'] for t in reliable)
                max_cq=max(t['cubic_to_quadratic'] for t in reliable)
                supported=bool(
                    same_sign and max_zerr<=1e-10 and min_r2>=.90
                    and max_lq<=.35 and max_cq<=.35
                )
                coordinate_invariance={
                    'status':'supported' if supported else 'diagnostic',
                    'tests':tests,
                    'same_quadratic_sign':same_sign,
                    'max_physical_z_error':float(max_zerr),
                    'min_quadratic_r2':float(min_r2),
                    'max_linear_to_quadratic':float(max_lq),
                    'max_cubic_to_quadratic':float(max_cq),
                    'interpretation':(
                        'The same physical projected fold germ is invariant under the tested smooth invertible coordinate changes.'
                        if supported else
                        'At least one tested coordinate representation does not preserve the configured projected fold diagnostics.'
                    ),
                    'caution':(
                        'This is a coordinate-invariance check of the scalar physical projection z=v_c^T(x-x_c), not a Lyapunov-Schmidt complement-invariance proof. '
                        'A genuine complement test requires explicitly constructing each regular complement and re-solving the regular equations.'
                    ),
                }
            else:
                coordinate_invariance={'status':'unavailable','tests':tests,'interpretation':'No reliable critical-point coordinate-invariance fits were available.'}
    except Exception as e:
        coordinate_invariance={'status':'unavailable','error':str(e),'interpretation':'Critical-point coordinate-invariance diagnostic could not be evaluated.'}

    # Backward-compatible response key for the current front end/API clients.
    complement_invariance=coordinate_invariance

    # Local spectral-smoothness audit at the critical point and its nearest
    # well-corrected continuation neighbors.  This is not another fold search:
    # it checks the eigenvalue simplicity / mode-selection assumptions needed
    # for f1 and f2 to be smooth local performance coordinates.
    spectral_smoothness=None
    try:
        good_spec=[]
        for i,q in enumerate(trace):
            if not q.get('converged',False) or float(q.get('residual_inf',1.0))>1e-4:
                continue
            good_spec.append((abs(float(q['control_value'])-control_c),i,q))
        selected=sorted(good_spec,key=lambda a:a[0])[:5]
        rows=[]
        for _,i,q in selected:
            rr=robot.with_design_vector(np.log(np.maximum(
                np.r_[np.asarray(q['t1'],float),np.asarray(q['t2'],float)],1e-15)))
            M,K=assemble_two_link_variable_sections(
                rr.l1,rr.l2,np.asarray(rr.t1,float),np.asarray(rr.t2,float),
                rr.width1_out,rr.width2_out,rr.E,rr.rho,
                M_joint2=rr.joint_mass,M_tip=rr.tip_mass)
            lam=np.real(eigvalsh(K,M))
            positive=lam[lam>1e-3]
            raw_f=np.sqrt(positive)/(2*np.pi)
            flex=raw_f[raw_f>1.0]
            H=np.linalg.eigvalsh(rr.inertia_matrix(np.array([0.0,0.0])))
            first=[float(v) for v in flex[:5]]
            f1=first[0] if len(first)>0 else None
            f2=first[1] if len(first)>1 else None
            f3=first[2] if len(first)>2 else None
            rows.append({
                'trace_index':int(i),'control_value':float(q['control_value']),
                'frequencies_hz':first,
                'f1_f2_gap_hz':None if f1 is None or f2 is None else float(f2-f1),
                'f2_f3_gap_hz':None if f2 is None or f3 is None else float(f3-f2),
                'f1_threshold_margin_hz':None if f1 is None else float(f1-1.0),
                'modes_above_1hz':int(len(flex)),
                'inertia_eigenvalues':[float(v) for v in H],
                'inertia_gap':float(H[-1]-H[0]),
                'residual_inf':float(q.get('residual_inf',0.0)),
            })
        if rows:
            gaps12=[r['f1_f2_gap_hz'] for r in rows if r['f1_f2_gap_hz'] is not None]
            gaps23=[r['f2_f3_gap_hz'] for r in rows if r['f2_f3_gap_hz'] is not None]
            margins=[r['f1_threshold_margin_hz'] for r in rows if r['f1_threshold_margin_hz'] is not None]
            hgaps=[r['inertia_gap'] for r in rows]
            # Relative separation is reported rather than imposing a theorem-level
            # numerical threshold.  "supported" only requires strict positive
            # separation and no mode-selection threshold contact in this sample.
            supported=bool(gaps12 and min(gaps12)>0 and margins and min(margins)>0 and min(hgaps)>0)
            spectral_smoothness={
                'status':'supported' if supported else 'diagnostic',
                'rows':rows,
                'min_f1_f2_gap_hz':float(min(gaps12)) if gaps12 else None,
                'min_f2_f3_gap_hz':float(min(gaps23)) if gaps23 else None,
                'min_f1_threshold_margin_hz':float(min(margins)) if margins else None,
                'min_inertia_gap':float(min(hgaps)) if hgaps else None,
                'interpretation':(
                    'The selected flexible modes remain ordered and separated from the 1 Hz selection threshold in the sampled critical neighborhood; the inertia eigenvalues also remain distinct.'
                    if supported else
                    'At least one sampled spectral separation or mode-selection condition needs inspection before local smoothness is claimed.'
                ),
                'caution':'This is a local numerical audit of the eigenvalue-selection assumptions. Analytic smoothness follows for simple eigenvalues of the smooth symmetric/generalized-symmetric matrix families; the audit checks that the implementation is operating in that regime near the computed fold.'
            }
        else:
            spectral_smoothness={'status':'unavailable','rows':[],'interpretation':'No sufficiently corrected continuation points were available for the local spectral audit.'}
    except Exception as e:
        spectral_smoothness={'status':'unavailable','rows':[],'error':str(e),'interpretation':'Local spectral-smoothness audit could not be evaluated.'}

    bounds_configured=section_min is not None or section_max is not None
    any_active=any(q['active_bounds'] for q in trace)
    return {"status":"completed","controls":control_metrics,"trace":trace,"turning_indices":turns,
        "turning_point_detected":bool(turns),"best":best,"paired_designs":paired_designs,"normal_form":normal_form,"complement_invariance":complement_invariance,"spectral_smoothness":spectral_smoothness,
        "bounds":{"configured":bounds_configured,"section_min":section_min,"section_max":section_max,
                  "active_anywhere":any_active,
                  "interpretation":("No configured finite section bound became active on the computed trace." if bounds_configured and not any_active else
                                    "At least one configured section bound became active; boundary-induced singularity must be considered." if any_active else
                                    "No finite section bounds were supplied. The model enforces positivity through log design variables, so a finite-bound exclusion test remains unavailable.")},
        "message":"Pseudo-arclength continuation completed on a reproducible one-dimensional slice of the redundant level-set manifold. A turning point plus isolated rank loss strengthens fold evidence; absence of a turning point on this finite trace is inconclusive."}


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
