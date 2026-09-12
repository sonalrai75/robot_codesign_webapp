from fastapi.testclient import TestClient
from webapp.app import app


def test_target_search_page_and_already_satisfied_unconstrained():
    c=TestClient(app)
    page=c.get('/target-search')
    assert page.status_code==200
    assert 'Global Performance Target Search' in page.text

    a=c.post('/api/v1/analyze',json={}).json()
    f1=a['performance']['f1_Hz']
    r=c.post('/api/v1/target-search',json={
        'targets':[{'metric':'f1_Hz','relation':'min','value':f1*0.9,'tolerance':0.01}],
        'max_iterations':None,
        'step_limit':0.08,
        'secondary_objective':'none',
        'secondary_weight':0.25,
    })
    assert r.status_code==200
    d=r.json()
    assert d['unconstrained'] is True
    assert d['status']=='targets_satisfied'
    assert d['iterations_completed']==0
    assert len(d['trace'])==1


def test_target_search_takes_evolving_svd_step():
    c=TestClient(app)
    a=c.post('/api/v1/analyze',json={}).json()
    f1=a['performance']['f1_Hz']
    r=c.post('/api/v1/target-search',json={
        'targets':[{'metric':'f1_Hz','relation':'min','value':f1*1.01,'tolerance':0.001}],
        'max_iterations':1,
        'step_limit':0.08,
        'secondary_objective':'none',
        'secondary_weight':0.25,
    })
    assert r.status_code==200
    d=r.json()
    assert d['iterations_completed']==1
    assert len(d['trace'])==2
    assert d['trace'][1]['metrics']['f1_Hz'] > d['trace'][0]['metrics']['f1_Hz']
    assert d['trace'][1]['step_norm'] > 0
    assert d['trace'][1]['singular_values']
