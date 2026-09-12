from fastapi import HTTPException
from webapp.app import DesignMoveInput, design_move, _client_revisions


def test_stale_design_revision_is_rejected():
    cid="pytest-stale-revision"
    _client_revisions.pop(cid, None)
    req=DesignMoveInput(client_id=cid, expected_revision=0, space="active", direction_index=1, amplitude=0.001)
    out=design_move(req)
    assert out["design_revision"] == 1
    stale=DesignMoveInput(client_id=cid, expected_revision=0, space="active", direction_index=1, amplitude=0.001)
    try:
        design_move(stale)
    except HTTPException as e:
        assert e.status_code == 409
        assert "Stale design revision" in str(e.detail)
    else:
        raise AssertionError("stale request was not rejected")
