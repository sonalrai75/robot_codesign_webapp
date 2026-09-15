import numpy as np
from robot_codesign.analysis.catastrophe import scalar_normal_form_diagnostics, diagnose_map_singularity


def test_fold_normal_form():
    d = scalar_normal_form_diagnostics(lambda z: z*z)
    assert d["classification"] == "A2 fold"
    assert abs(d["d2"] - 2.0) < 1e-5


def test_cusp_normal_form():
    d = scalar_normal_form_diagnostics(lambda z: z**3)
    assert d["classification"] == "A3 cusp"
    assert abs(d["d3"] - 6.0) < 1e-4


def test_vector_map_fold_candidate():
    # F(x,y)=(x, y^2): rank loss at origin with quadratic curvature in weak direction.
    J=np.array([[1.0,0.0],[0.0,0.0]])
    def ev(q): return np.array([q[0], q[1]**2])
    d=diagnose_map_singularity(J,ev,np.zeros(2),["x","y"],force_deep=True)
    assert d["level"] == "fold_candidate"
    assert d["classification"] == "A2 fold"


def test_vector_map_cusp_candidate():
    # F(x,y)=(x, y^3): vanishing quadratic and nonzero cubic critical term.
    J=np.array([[1.0,0.0],[0.0,0.0]])
    def ev(q): return np.array([q[0], q[1]**3])
    d=diagnose_map_singularity(J,ev,np.zeros(2),["x","y"],force_deep=True)
    assert d["level"] == "cusp_candidate"
    assert d["classification"] == "A3 cusp"

from robot_codesign.analysis.catastrophe import catastrophe_search_scores


def test_search_score_prefers_stronger_fold_geometry():
    strong=catastrophe_search_scores({"sigma_ratio":.002,"second_alignment":.8,"third_alignment":.1})
    weak=catastrophe_search_scores({"sigma_ratio":.08,"second_alignment":.2,"third_alignment":.1})
    assert strong["fold_score"] < weak["fold_score"]


def test_search_score_prefers_cusp_geometry():
    cusp=catastrophe_search_scores({"sigma_ratio":.002,"second_alignment":.01,"third_alignment":.8})
    ordinary=catastrophe_search_scores({"sigma_ratio":.002,"second_alignment":.7,"third_alignment":.8})
    assert cusp["cusp_score"] < ordinary["cusp_score"]
