import numpy as np
from robot_codesign.geometry.equivalent_metric import metric_from_eigenvalues_and_angle

def test_equivalent_metric_preserves_published_zero_configuration_eigenvalues():
    metric=metric_from_eigenvalues_and_angle(0.0066,0.0281,-0.73)
    vals=np.linalg.eigvalsh(metric.matrix([0.0,0.0]))
    assert np.allclose(vals,[0.0066,0.0281],atol=1e-12)
