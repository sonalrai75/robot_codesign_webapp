# Robot Co-Design V0

Initial reconstruction of the two-link planar robot in Chapter 5 of the 1993 MIT thesis.

## Current validated milestone

- two-link rigid-body inertia matrix
- nominal inertia eigenvalues
- intermediate-design inertia eigenvalues
- configuration sweep confirming the maximum inertia eigenvalue at q1-q2 = 0

## Next milestone

Assemble the 20-element flexible model and identify the nominal rectangular
section dimensions against the reported 116.3 Hz and 268.2 Hz frequencies.

Run:

```bash
python -m robot_codesign.examples.thesis_ch5.run_milestone1
pytest -q
```

## Chapter 5 provenance note

The travel times printed in Figures 5-4 and 5-5 are labeled **Final Design** and
therefore belong to the post-shape-optimization design, not the uniform
intermediate design.  The intermediate design is still retained because its
lengths and areas are explicitly tabulated, but it must not be used as if it
were the published final geometry.

Because the thesis prints only the final inertia eigenvalues (0.0066, 0.0281)
and the final shape graph, not the complete element-by-element final mass
distribution, `fit_final_equivalent_metric.py` reconstructs an *equivalent*
cosine-coupled inertia metric.  It is explicitly an inverse reconstruction,
not a claim that the original 1993 distributed geometry has yet been recovered.

## Figure 5-6 reconstruction status

A high-resolution digitization check was added for Figure 5-6.  Treating the
upper/lower plotted outlines literally as the in-plane rectangular-section
thickness, together with the stated fixed out-of-plane widths of 1 cm and
3.8 cm, does **not** reproduce Table 5.1.  It predicts much larger inertia and
much higher flexible frequencies.  Therefore the figure-to-FE-section mapping
is currently classified as an unresolved historical modeling detail rather
than silently assumed.

The package keeps the digitized profile and a variable-section FE assembler so
that alternative interpretations can be tested reproducibly.



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
