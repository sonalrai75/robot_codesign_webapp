# Robot Co-Design Laboratory v0.1

This is the first deployable web layer around the Chapter-5 reconstruction and modern SVD/redundant-space analysis.

## Local run

From this folder:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Windows Git Bash: source .venv/Scripts/activate
pip install -e .
pip install -r requirements-web.txt
python run_local.py
```

Open `http://127.0.0.1:8000`.

## What V0.1 does

- configurable 2-DOF planar robot and endpoint task;
- inertia-metric geodesic, joint-space straight line, Cartesian straight line + IK, and an experimental cubic path optimizer;
- common approximate torque-constrained fixed-path timing model for path comparison;
- Chapter-5-style 20-variable flexible FE model;
- primary metrics H_min, H_max, f1, f2;
- normalized-log Jacobian and SVD;
- active rank, singular values, conditioning, and null-space dimension;
- first projected redundant-space secondary move: mass reduction;
- browser visualization of robot/task path.

## Important V0.1 limitations

The app is a research prototype, not an exact reconstruction of every unpublished 1993 numerical implementation. The arbitrary-path timing method is an interactive discretized approximation. The `optimized_cubic` planner minimizes Riemannian length inside a small cubic-Bezier family; it is not yet direct-collocation global time optimization. The mass secondary step is currently a single projected null-space step, not yet the nonlinear correction/continuation loop used in the later research milestones.

## Hosted deployment

The numerical engine is separated from the HTTP/UI layer. The included Dockerfile can be used on a container host. For a public version, add job queues for long analyses, persistent experiment storage, rate limits, authentication if needed, and a replacement React/Three.js frontend without changing the numerical interfaces.

## V0.2 hyperspace explorer

V0.2 adds a physical 20-element section/mass-distribution view and interactive SVD design-space controls. After computing SVD, choose either an active right-singular direction V_i, a redundant/null direction N_i, or a secondary objective projected into the null space. The app shows +/- mode previews and permits finite log-design-space moves. Null-space and secondary moves can be followed by a nonlinear minimum-norm correction that restores the four primary metrics (H_min, H_max, f1, f2). Recompute the SVD after each move because the Jacobian and null space evolve with design.


## v0.5 path point inspector
Click any point on the task-space path to freeze it. The selected point is marked in purple and the UI reports Cartesian/joint coordinates, the local inertia matrix and eigenvalues, minimum-time path speed/acceleration, approximate actuator torques and utilization, plus the current structural width/mass state and flexible frequencies.


## v0.8 control correction
For `path_type=geodesic`, the web app now uses the Chapter-5 geodesic reduction directly. The geodesic is parameterized by Riemannian arc length `s`; the allowable `s_ddot` is computed from the two independent joint torque limits, followed by a forward/backward rest-to-rest speed-envelope construction. The joint torque vector is `tau = H(q) q_s s_ddot`, so at least one actuator is on its positive or negative torque limit during each acceleration/deceleration interval. The UI marks the active saturated actuator, the accel-to-brake switching time, and the Riemannian path-speed profile. Non-geodesic paths continue to use the generic fixed-path SLSQP approximation and are explicitly labeled as such.


## v0.9 move-safety update
Design moves are serialized in the browser and on the FastAPI backend. The UI disables state-changing controls while a move is running, ignores stale browser responses, and sends a per-client expected revision. The backend executes the complete structure → path/control → performance → SVD transaction under a per-client lock and rejects stale revisions with HTTP 409. Failed computations do not advance the committed revision.


## v0.10 distributed-mass rigid inertia

The rigid-body inertia now integrates the full 10-element mass distribution on each link instead of collapsing each link to its mean cross-sectional area. For equal-length piecewise-constant elements, the implementation computes exact mass moments (mass, first moment and second moment) and reduces exactly to the Chapter 5 uniform-area equations when all sections are uniform. As a result, redistributing material along a link can change H(q), the inertia-metric geodesic, the torque-constrained bang-bang solution and travel time even when total link mass is unchanged.


## v0.11 — global performance target search

A separate `/target-search` page accepts global inequality/equality performance targets and automatically navigates the nonlinear design space through repeated local SVD steps. The performance Jacobian/SVD is recomputed after every accepted move. Users may set a maximum number of iterations or leave it blank to continue until convergence/stall (with a numerical safety cap of 100). Every accepted iteration is retained in the result trace and can be inspected individually; any viewed iteration can be adopted as the starting design for the next run. Travel-time sensitivity uses a frozen-current-path local linearization for interactive speed, while every accepted candidate is validated by recomputing the full selected path and time-optimal control.


## v0.12 — metric percentage-change charts
Both the main laboratory and global target-search page now display signed percentage changes from the relevant starting design. The main-page reference is the initial analyzed design (reset when uniform sections are restored). The global-search reference is iteration 0 of the current search, and selecting an iteration updates the chart immediately.
