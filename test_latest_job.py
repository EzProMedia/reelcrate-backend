import os
import time
import uuid
from test_regressions import client, isolated, headers
import main


def job(email='dj@example.test', started=1, status='done'):
    jid = str(uuid.uuid4())
    main.job_dir(jid).mkdir()
    main.write_state(jid, {'owner_email': email, 'started_at': started, 'status': status})
    return jid


def test_latest_job_requires_auth(client):
    assert client.get('/api/latest-job').status_code == 401


def test_latest_job_is_account_scoped_and_not_cached(client):
    job(started=10)
    newest = job(started=20)
    job(email='other@example.test', started=30)
    response = client.get('/api/latest-job', headers=headers())
    assert response.json() == {'job_id': newest}
    assert response.headers['cache-control'] == 'private, no-store'


def test_latest_job_empty_and_expired(client):
    assert client.get('/api/latest-job', headers=headers()).json() == {'job_id': None}
    jid = job()
    old = time.time() - (main.CLIP_TTL_HOURS + 1) * 3600
    os.utime(main.job_dir(jid), (old, old))
    assert client.get('/api/latest-job', headers=headers()).json() == {'job_id': None}
    active = job(status='rendering')
    os.utime(main.job_dir(active), (old, old))
    assert client.get('/api/latest-job', headers=headers()).json() == {'job_id': active}


def test_latest_job_skips_invalid_state(client):
    jid = job()
    (main.job_dir(jid) / 'state.json').write_text('broken')
    (main.JOBS_DIR / 'not-a-job').mkdir()
    assert client.get('/api/latest-job', headers=headers()).json() == {'job_id': None}
