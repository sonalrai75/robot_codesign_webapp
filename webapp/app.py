from __future__ import annotations
from pathlib import Path
import threading
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from robot_codesign.models import Planar2DOFRobot
from robot_codesign.paths import PLANNERS
from robot_codesign.analysis import metric_length, euclidean_joint_length, path_min_time_control
from robot_codesign.analysis.svd_design import analyze_design_space, secondary_direction, secondary_metric, corrected_null_move
from robot_codesign.analysis.target_search import TargetSpec, SecondarySpec, run_target_search, catastrophe_diagnostics

ROOT=Path(__file__).resolve().parent
app=FastAPI(title="Robot Co-Design Laboratory",version="0.14.0")
app.mount("/static",StaticFiles(directory=ROOT/"static"),name="static")

class RobotInput(BaseModel):
    l1: float=.1419; l2: float=.4581; rho: float=2700.; E: float=2.7e10
    joint_mass: float=.09966; tip_mass: float=.000122
    width1_out: float=.010; width2_out: float=.038
    t1: list[float]=Field(default_factory=lambda:[.19]*10)
    t2: list[float]=Field(default_factory=lambda:[.005263157894736842]*10)

class AnalyzeInput(BaseModel):
    robot: RobotInput=Field(default_factory=RobotInput)
    start_xy: list[float]=Field(default_factory=lambda:[.4,0.])
    end_xy: list[float]=Field(default_factory=lambda:[.4,.2])
    path_type: str="geodesic"
    branch: int=1
    torque_limits: list[float]=Field(default_factory=lambda:[40.,40.])


class GlobalTarget(BaseModel):
    metric: str
    relation: str = "equal"  # equal | min | max
    value: float
    tolerance: float = 0.01

class NullObjective(BaseModel):
    metric: str
    direction: str = "min"  # min | max
    weight: float = 1.0

class TargetSearchInput(AnalyzeInput):
    targets: list[GlobalTarget] = Field(default_factory=list)
    # None means continue until targets are met or the search stalls.  The
    # numerical implementation still has a hard safety cap.
    max_iterations: int | None = 10
    step_limit: float = 0.16
    # v0.13: multiple weighted secondary metrics can be optimized together in
    # the evolving null space of the selected primary target Jacobian.
    secondary_objectives: list[NullObjective] = Field(default_factory=list)
    null_step_fraction: float = 0.25
    # v0.12 compatibility fields; ignored when secondary_objectives is supplied.
    secondary_objective: str = "none"
    secondary_weight: float = 0.25

class DesignMoveInput(AnalyzeInput):
    # Optimistic-concurrency metadata.  The browser owns a stable client_id and
    # sends the revision it believes is current.  The server serializes move
    # requests per client and rejects stale revisions with HTTP 409.
    client_id: str="default"
    expected_revision: int=0
    space: str="null"  # null | active | secondary
    direction_index: int=1
    amplitude: float=.08
    objective: str="mass"
    nonlinear_correction: bool=True
    active_weights: list[float]=Field(default_factory=lambda:[1.,0.,0.,0.])


# Per-client move serialization.  This protects the numerical state transition
# from rapid/repeated requests in this process.  The UI also disables all
# state-changing controls while a move is running.
_move_registry_lock = threading.Lock()
_client_move_locks: dict[str, threading.Lock] = {}
_client_revisions: dict[str, int] = {}

def _move_lock_for(client_id: str) -> threading.Lock:
    key=(client_id or "default")[:128]
    with _move_registry_lock:
        lock=_client_move_locks.get(key)
        if lock is None:
            lock=threading.Lock(); _client_move_locks[key]=lock
        return lock

def build_robot(r:RobotInput): return Planar2DOFRobot(**r.model_dump())

def design_payload(robot):
    t1=np.asarray(robot.t1,float); t2=np.asarray(robot.t2,float)
    # element mass excludes lumped joint/tip masses; values support distribution plots.
    m1=(robot.rho*robot.width1_out*t1*(robot.l1/len(t1))).tolist()
    m2=(robot.rho*robot.width2_out*t2*(robot.l2/len(t2))).tolist()
    return {"t1":t1.tolist(),"t2":t2.tolist(),"element_mass1":m1,"element_mass2":m2,
            "joint_mass":robot.joint_mass,"tip_mass":robot.tip_mass}

def performance_payload(robot,req=None,path=None,timing=None):
    p=robot.performance_vector()
    out={"H_min":float(p[0]),"H_max":float(p[1]),"f1_Hz":float(p[2]),"f2_Hz":float(p[3]),"mass_kg":robot.mass(),
         "smoothness":secondary_metric(robot,"smoothness"),"peak_width":secondary_metric(robot,"peak_width"),"rms_width":secondary_metric(robot,"rms_width")}
    if req is not None:
        if path is None:
            planner=PLANNERS[req.path_type]; path=planner.plan(robot,req.start_xy,req.end_xy,branch=req.branch,n=101)
        if timing is None:
            timing=path_min_time_control(robot,path,req.torque_limits,n_nodes=27)
        out.update({"travel_time_s":timing["time_s"],"time_solver_success":timing["success"],"max_torque_ratio":timing["max_torque_ratio"],"riemannian_length":metric_length(robot,path),"joint_path_length":euclidean_joint_length(path)})
    return out

@app.get("/")
def index(): return FileResponse(ROOT/"static"/"index.html")

@app.get("/target-search")
def target_search_page(): return FileResponse(ROOT/"static"/"target_search.html")

@app.get("/api/v1/capabilities")
def capabilities():
    return {"robot_models":["planar_2dof"],"path_types":list(PLANNERS),"primary_metrics":["H_min","H_max","f1_Hz","f2_Hz"],
            "secondary_metrics":["mass","smoothness","peak_width","rms_width","maneuverability"],
            "catastrophe_analysis":["near-rank-loss screen","higher-order fold/cusp diagnostics"],
            "architecture":"DOF-agnostic model/path/metric interfaces; evolving-SVD target/null-space search with conservative manifold singularity diagnostics"}

@app.post("/api/v1/analyze")
def analyze(req:AnalyzeInput):
    try:
        robot=build_robot(req.robot); planner=PLANNERS[req.path_type]
        path=planner.plan(robot,req.start_xy,req.end_xy,branch=req.branch,n=101)
        xy=np.array([robot.forward_kinematics(q) for q in path.q])
        timing=path_min_time_control(robot,path,req.torque_limits,n_nodes=27)
        # Per-point rigid-body inertia and its eigenvalues are inexpensive enough
        # to expose for the interactive click inspector.
        H=[robot.inertia_matrix(qi) for qi in path.q]
        He=[np.linalg.eigvalsh(h) for h in H]
        return {"path":{"type":path.name,"u":path.u.tolist(),"q":path.q.tolist(),"xy":xy.tolist(),
                        "q_u":path.q_u.tolist(),"q_uu":path.q_uu.tolist(),"metadata":path.metadata,
                        "inertia":[h.tolist() for h in H],"inertia_eigs":[e.tolist() for e in He],
                        "timing":timing},
                "performance":performance_payload(robot,req,path=path,timing=timing),"design":design_payload(robot)}
    except Exception as e: raise HTTPException(400,str(e))

def _link_participation(v, n_link):
    p1=float(np.linalg.norm(v[:n_link])); p2=float(np.linalg.norm(v[n_link:]))
    tot=p1+p2
    return (p1/tot if tot else 0.0, p2/tot if tot else 0.0)

def _balanced_active_direction(a, n_link):
    """Choose a normalized direction in the active right-singular subspace with near-equal link participation."""
    V=np.asarray(a["active_right_vectors"],float)
    r=V.shape[1]
    if r==1:
        return V[:,0], np.array([1.0])
    from scipy.optimize import minimize
    best=None
    starts=[np.eye(r)[k] for k in range(r)]
    starts += [np.ones(r)/np.sqrt(r), np.array([(-1.0)**k for k in range(r)])/np.sqrt(r)]
    def obj(c):
        nc=np.linalg.norm(c)
        if nc<1e-12:return 1e6
        c=c/nc; d=V@c
        p1=np.linalg.norm(d[:n_link]); p2=np.linalg.norm(d[n_link:])
        return float((p1-p2)**2)
    for c0 in starts:
        res=minimize(obj,c0,method="BFGS",options={"maxiter":200,"gtol":1e-10})
        c=np.asarray(res.x,float); c/=max(np.linalg.norm(c),1e-15)
        score=obj(c)
        if best is None or score<best[0]: best=(score,c)
    c=best[1]; d=V@c; d/=max(np.linalg.norm(d),1e-15)
    return d,c

@app.post("/api/v1/svd")
def svd(req:AnalyzeInput):
    try:
        robot=build_robot(req.robot); a=analyze_design_space(robot); x=robot.design_vector()
        active=[]
        for k in range(a["rank"]):
            v=a["active_right_vectors"][:,k]
            p1,p2=_link_participation(v,len(robot.t1))
            active.append({"index":k+1,"vector":v.tolist(),"link1_participation":p1,"link2_participation":p2,"plus":design_payload(robot.with_design_vector(x+.08*v)),"minus":design_payload(robot.with_design_vector(x-.08*v))})
        # The numerical SVD null basis is not unique.  Reorder it for the UI so
        # the first directions are physically balanced between the two links.
        # This does not change the null space; it only changes presentation/order.
        N=a["null_basis"]
        n_link=len(robot.t1)
        records=[]
        for k in range(N.shape[1]):
            v=N[:,k]
            p1=float(np.linalg.norm(v[:n_link])); p2=float(np.linalg.norm(v[n_link:]))
            balance=min(p1,p2)/max(p1,p2,1e-15)
            records.append((balance,k,p1,p2,v))
        records.sort(key=lambda z:z[0],reverse=True)
        null=[]
        for ui_index,(balance,k,p1,p2,v) in enumerate(records,1):
            tot=p1+p2
            null.append({"index":ui_index,"raw_index":k+1,"vector":v.tolist(),
                         "link1_participation":p1/tot if tot else 0.0,
                         "link2_participation":p2/tot if tot else 0.0,
                         "balance_score":balance,
                         "plus":design_payload(robot.with_design_vector(x+.08*v)),
                         "minus":design_payload(robot.with_design_vector(x-.08*v))})
        return {"singular_values":a["singular_values"].tolist(),"rank":a["rank"],"condition":a["condition"],"n_design_variables":len(x),
                "null_dimension":int(len(x)-a["rank"]),"primary_metrics":a["primary_names"],"active_modes":active,"null_modes":null,
                "design":design_payload(robot),
                "balanced_active":(lambda dc:(lambda pp:{"vector":dc[0].tolist(),"weights":dc[1].tolist(),"link1_participation":pp[0],"link2_participation":pp[1],"plus":design_payload(robot.with_design_vector(x+.08*dc[0])),"minus":design_payload(robot.with_design_vector(x-.08*dc[0]))})(_link_participation(dc[0],len(robot.t1))))(_balanced_active_direction(a,len(robot.t1)))}
    except Exception as e: raise HTTPException(400,str(e))


@app.post("/api/v1/target-search")
def target_search(req:TargetSearchInput):
    try:
        robot=build_robot(req.robot)
        specs=[]
        allowed_metrics={"H_min","H_max","f1_Hz","f2_Hz","travel_time_s","mass_kg"}
        for t in req.targets:
            if t.metric not in allowed_metrics:
                raise ValueError(f"Unsupported target metric: {t.metric}")
            if t.relation not in {"equal","min","max"}:
                raise ValueError(f"Unsupported target relation: {t.relation}")
            if t.value <= 0:
                raise ValueError(f"Target {t.metric} must be positive")
            if not (0 <= t.tolerance <= .5):
                raise ValueError("Target tolerance must be between 0 and 0.5")
            specs.append(TargetSpec(t.metric,t.relation,float(t.value),float(t.tolerance)))
        if req.max_iterations is not None and not (1 <= req.max_iterations <= 100):
            raise ValueError("max_iterations must be 1..100, or null for unconstrained")
        secondary_specs=[]
        allowed_secondary={"mass","smoothness","peak_width","rms_width","maneuverability"}
        for o in req.secondary_objectives:
            if o.metric not in allowed_secondary:
                raise ValueError(f"Unsupported secondary objective: {o.metric}")
            if o.direction not in {"min","max"}:
                raise ValueError(f"Unsupported secondary direction: {o.direction}")
            if not (0 <= o.weight <= 100):
                raise ValueError("Secondary objective weight must be between 0 and 100")
            secondary_specs.append(SecondarySpec(o.metric,o.direction,float(o.weight)))
        if not (0 <= req.null_step_fraction <= 1):
            raise ValueError("null_step_fraction must be between 0 and 1")
        search_context={"start_xy":req.start_xy,"end_xy":req.end_xy,"path_type":req.path_type,"branch":req.branch,"torque_limits":req.torque_limits,"timing_nodes":27}
        result=run_target_search(
            robot,specs,
            search_context,
            max_iterations=req.max_iterations,step_limit=float(req.step_limit),
            secondary_objectives=secondary_specs,null_step_fraction=float(req.null_step_fraction),
            secondary_objective=req.secondary_objective,secondary_weight=float(req.secondary_weight),
            safety_cap=100,
        )
        final=result.pop("final_robot")
        result["final_design"]=design_payload(final)
        result["final_performance"]=performance_payload(final,req)
        # Automated manifold/catastrophe screen. Deep derivative tests run only
        # when the selected target Jacobian is near rank loss.
        result["catastrophe_analysis"]=catastrophe_diagnostics(final,specs,search_context,force_deep=False)
        return result
    except Exception as e:
        raise HTTPException(400,str(e))

@app.post("/api/v1/catastrophe-analysis")
def catastrophe_analysis(req:TargetSearchInput):
    """Force higher-order local singularity diagnostics at the supplied design."""
    try:
        robot=build_robot(req.robot)
        specs=[]
        allowed={"H_min","H_max","f1_Hz","f2_Hz","travel_time_s","mass_kg"}
        for t in req.targets:
            if t.metric not in allowed: raise ValueError(f"Unsupported target metric: {t.metric}")
            if t.relation not in {"equal","min","max"}: raise ValueError(f"Unsupported target relation: {t.relation}")
            if t.value <= 0: raise ValueError(f"Target {t.metric} must be positive")
            specs.append(TargetSpec(t.metric,t.relation,float(t.value),float(t.tolerance)))
        context={"start_xy":req.start_xy,"end_xy":req.end_xy,"path_type":req.path_type,"branch":req.branch,"torque_limits":req.torque_limits,"timing_nodes":27}
        return catastrophe_diagnostics(robot,specs,context,force_deep=True)
    except Exception as e:
        raise HTTPException(400,str(e))

@app.post("/api/v1/design/move")
def design_move(req:DesignMoveInput):
    client_id=(req.client_id or "default")[:128]
    lock=_move_lock_for(client_id)
    # Hold the per-client lock for the complete transaction: structure move,
    # path/control/performance recomputation, and new SVD.  A second request
    # from the same browser waits, then fails its stale expected_revision check.
    with lock:
      try:
        current_revision=_client_revisions.get(client_id,0)
        if req.expected_revision != current_revision:
            raise HTTPException(409, f"Stale design revision: expected {req.expected_revision}, current is {current_revision}. Refresh/recompute before applying another move.")
        robot=build_robot(req.robot); a=analyze_design_space(robot); oldp=robot.performance_vector(); amp=float(np.clip(req.amplitude,-.6,.6))
        note=""
        if req.space=="active":
            k=req.direction_index-1
            if k<0 or k>=a["rank"]: raise ValueError("Active direction index out of range")
            d=a["active_right_vectors"][:,k]; new=robot.with_design_vector(robot.design_vector()+amp*d)
            conv=False; res=float(np.linalg.norm(np.log(new.performance_vector()/oldp),np.inf)); moved=abs(amp)
            note="Active SVD directions deliberately change primary performance."
        elif req.space=="active_combo":
            V=a["active_right_vectors"]; w=np.asarray(req.active_weights,float)[:a["rank"]]
            if len(w)<a["rank"]: w=np.pad(w,(0,a["rank"]-len(w)))
            if np.linalg.norm(w)<1e-12: raise ValueError("At least one active SVD weight must be nonzero")
            w=w/np.linalg.norm(w); d=V@w; d=d/max(np.linalg.norm(d),1e-15)
            new=robot.with_design_vector(robot.design_vector()+amp*d); conv=False
            res=float(np.linalg.norm(np.log(new.performance_vector()/oldp),np.inf)); moved=abs(amp)
            note="Moved along a normalized user-weighted combination of active SVD directions."
        elif req.space=="active_balanced":
            d,w=_balanced_active_direction(a,len(robot.t1)); new=robot.with_design_vector(robot.design_vector()+amp*d); conv=False
            res=float(np.linalg.norm(np.log(new.performance_vector()/oldp),np.inf)); moved=abs(amp)
            p1,p2=_link_participation(d,len(robot.t1))
            note=f"Balanced active direction uses both links (L1 {100*p1:.1f}% / L2 {100*p2:.1f}%) while remaining inside the active SVD subspace."
        elif req.space=="null":
            k=req.direction_index-1
            N=a["null_basis"]; n_link=len(robot.t1)
            order=[]
            for j in range(N.shape[1]):
                p1=float(np.linalg.norm(N[:n_link,j])); p2=float(np.linalg.norm(N[n_link:,j]))
                balance=min(p1,p2)/max(p1,p2,1e-15)
                order.append((balance,j))
            order.sort(key=lambda z:z[0],reverse=True)
            if k<0 or k>=len(order): raise ValueError("Null direction index out of range")
            raw_k=order[k][1]
            d=N[:,raw_k]
            if req.nonlinear_correction:
                new,conv,res,moved=corrected_null_move(robot,d,amp,target=oldp)
                note="Moved in the selected local null direction, then applied minimum-norm nonlinear correction to restore the four primary metrics."
            else:
                new=robot.with_design_vector(robot.design_vector()+amp*d); conv=False
                res=float(np.linalg.norm(np.log(new.performance_vector()/oldp),np.inf)); moved=abs(amp)
                note="Pure local null-space move without nonlinear correction; finite moves can drift from the target manifold."
        elif req.space=="secondary":
            d,_=secondary_direction(robot,req.objective)
            if not np.linalg.norm(d):
                new=robot; conv=False; res=0.; moved=0.; note=f"No meaningful first-order null-space descent direction for {req.objective} at this design."
            elif req.nonlinear_correction:
                new,conv,res,moved=corrected_null_move(robot,d,abs(amp),target=oldp)
                note=f"Projected descent for {req.objective}, followed by nonlinear correction of primary performance."
            else:
                new=robot.with_design_vector(robot.design_vector()+abs(amp)*d); conv=False
                res=float(np.linalg.norm(np.log(new.performance_vector()/oldp),np.inf)); moved=abs(amp)
                note=f"Projected first-order descent for {req.objective}, without nonlinear correction."
        else: raise ValueError("space must be active, active_combo, active_balanced, null, or secondary")
        # Complete every expensive part before committing the revision.  A failed
        # geodesic/control/performance calculation must not advance client state.
        anew=analyze_design_space(new)
        new_design=design_payload(new)
        new_performance=performance_payload(new,req)
        old_performance=performance_payload(robot)
        next_revision=current_revision+1
        _client_revisions[client_id]=next_revision
        return {"design":new_design,"performance":new_performance,"old_performance":old_performance,
                "singular_values":anew["singular_values"].tolist(),"rank":anew["rank"],"null_dimension":int(len(new.design_vector())-anew["rank"]),
                "condition":anew["condition"],"correction_converged":conv,"primary_log_residual_inf":res,"log_design_move_norm":moved,"note":note,
                "design_revision":next_revision}
      except HTTPException:
        raise
      except Exception as e:
        raise HTTPException(400,str(e))
