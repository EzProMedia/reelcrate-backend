import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

os.environ['JWT_SECRET'] = 'test-only-' + 'x'*40
os.environ['REELCRATE_DATA'] = tempfile.mkdtemp(prefix='reelcrate-tests-')
os.environ['RESEND_API_KEY'] = ''
os.environ['STRIPE_SECRET_KEY'] = ''
import pytest
from fastapi.testclient import TestClient
import auth, billing, main, reliability, email_service

@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(auth, 'USERS_FILE', tmp_path/'users.json')
    monkeypatch.setattr(reliability, 'ROOT', tmp_path)
    monkeypatch.setattr(main, 'JOBS_DIR', tmp_path/'jobs')
    main.JOBS_DIR.mkdir()
    monkeypatch.setattr(main, 'ADMIN_TOKEN', 'a'*40)
    monkeypatch.setattr(billing, 'STRIPE_SECRET_KEY', 'sk_test_mock')
    monkeypatch.setattr(billing, '_PRICE_BY_PLAN', {'monthly':'price_mock'})
    auth._save_users({'dj@example.test': {'email':'dj@example.test','verified':True,'stripe_customer_id':'cus_mock'}})

@pytest.fixture
def client():
    return TestClient(main.app)

def headers(email='dj@example.test'):
    return {'Authorization':'Bearer '+auth.make_token(email)}

@pytest.mark.parametrize('value', ['', auth.DEFAULT_JWT_SECRET, 'short'])
def test_bad_secret_fails_startup(value):
    env={**os.environ,'JWT_SECRET':value}
    result=subprocess.run([sys.executable,'-c','import auth'],env=env,capture_output=True)
    assert result.returncode != 0

@pytest.mark.parametrize('path', ['/api/admin/waitlist','/api/admin/disk','/api/admin/cleanup'])
def test_admin_headers_only(client,path):
    call=client.post if path.endswith('cleanup') else client.get
    assert call(path+'?token='+'a'*40).status_code==401
    assert call(path, headers=headers()).status_code==401
    assert call(path, headers={'Authorization':'Bearer '+'a'*40}).status_code==200

def test_admin_disabled_without_dedicated_secret(client, monkeypatch):
    monkeypatch.setattr(main,'ADMIN_TOKEN','')
    assert client.get('/api/admin/disk',headers={'Authorization':'Bearer '+auth.JWT_SECRET}).status_code==503

def test_unsigned_webhook_rejected(client,monkeypatch):
    monkeypatch.setattr(billing,'STRIPE_WEBHOOK_SECRET','')
    assert client.post('/api/billing/webhook',json={'type':'customer.subscription.created'}).status_code==503

def test_bad_webhook_signature(client,monkeypatch):
    monkeypatch.setattr(billing,'STRIPE_WEBHOOK_SECRET','whsec_test')
    assert client.post('/api/billing/webhook',json={}).status_code==400

@pytest.mark.parametrize('promo,found,expected',[(None,None,False),('BAD',None,False),('FOUNDER40','promo_1',True)])
def test_promotion_modes(client,monkeypatch,promo,found,expected):
    monkeypatch.setattr(billing,'_resolve_promotion_code',lambda code:found)
    create=Mock(return_value=SimpleNamespace(url='https://checkout.stripe.com/mock',id='cs_mock'))
    monkeypatch.setattr(billing.stripe.checkout.Session,'create',create)
    r=client.post('/api/billing/checkout',headers=headers(),json={'promo':promo})
    assert r.status_code==200
    kw=create.call_args.kwargs
    assert ('discounts' in kw)==expected
    assert ('allow_promotion_codes' in kw)!=expected

def test_rejected_discount_falls_back(client,monkeypatch):
    monkeypatch.setattr(billing,'_resolve_promotion_code',lambda code:'promo_bad')
    create=Mock(side_effect=[billing.stripe.error.InvalidRequestError('expired','discounts[0][promotion_code]'),SimpleNamespace(url='https://checkout.stripe.com/mock',id='cs_mock')])
    monkeypatch.setattr(billing.stripe.checkout.Session,'create',create)
    r=client.post('/api/billing/checkout',headers=headers(),json={'promo':'EXPIRED'})
    assert r.status_code==200 and r.json()['promo_applied'] is None
    assert create.call_args.kwargs['allow_promotion_codes']

def test_unrelated_payment_error_not_retried(client,monkeypatch):
    monkeypatch.setattr(billing,'_resolve_promotion_code',lambda code:'promo_ok')
    create=Mock(side_effect=billing.stripe.error.InvalidRequestError('bad price','line_items'))
    monkeypatch.setattr(billing.stripe.checkout.Session,'create',create)
    assert client.post('/api/billing/checkout',headers=headers(),json={'promo':'OK'}).status_code==502
    assert create.call_count==1

def test_stripe_calls_do_not_block_event_loop(monkeypatch):
    def slow(**kw):
        time.sleep(.15)
        return SimpleNamespace(url='mock',id='mock')
    monkeypatch.setattr(billing.stripe.checkout.Session,'create',slow)
    async def run():
        task=asyncio.create_task(billing.checkout(billing.CheckoutReq(), 'dj@example.test'))
        start=time.monotonic()
        await asyncio.sleep(.02)
        assert time.monotonic()-start < .10
        await task
    asyncio.run(run())

def test_email_persist_retry_and_deduplicate(monkeypatch):
    send=Mock(side_effect=[False,True])
    monkeypatch.setattr(email_service,'send_verify_email',send)
    reliability.queue_email(auth.send_verify_email,'dj@example.test','DJ','secret-token',delivery_id='verify-one')
    assert send.call_count==0
    reliability.deliver_once()
    with reliability.connect() as db:
        assert db.execute('SELECT attempts FROM mail').fetchone()[0]==1
        db.execute('UPDATE mail SET due=0')
    reliability.deliver_once()
    reliability.queue_email(auth.send_verify_email,'dj@example.test','DJ','secret-token',delivery_id='verify-one')
    assert not reliability.deliver_once()
    with reliability.connect() as db:
        assert db.execute('SELECT args FROM mail').fetchone()[0]=='[]'

def test_corrupt_accounts_not_silently_erased():
    auth.USERS_FILE.write_text('{broken')
    with pytest.raises(RuntimeError): auth._load_users()
    assert auth.USERS_FILE.read_text()=='{broken'

def test_backup_keeps_previous_valid_accounts():
    before=auth.USERS_FILE.read_text()
    auth._save_users({'new':{}})
    assert auth.USERS_FILE.with_suffix('.json.bak').read_text()==before

def test_repeated_signin_limited(client):
    for _ in range(10):
        assert client.post('/api/auth/signin',json={'email':'missing@test.com','password':'wrong'}).status_code==401
    assert client.post('/api/auth/signin',json={'email':'missing@test.com','password':'wrong'}).status_code==429

def job(status='done'):
    jid=str(uuid.uuid4());main.job_dir(jid).mkdir()
    main.write_state(jid,{'status':status,'owner_email':'dj@example.test','clips':[{'rank':1}]})
    (main.job_dir(jid)/'clip_01.mp4').write_bytes(b'0123456789')
    return jid

def test_private_jobs_and_signed_phone_ranges(client):
    jid=job()
    assert client.get('/api/jobs/'+jid).status_code==401
    assert client.get('/api/jobs/'+jid,headers=headers('other@test.com')).status_code==403
    state=client.get('/api/jobs/'+jid,headers=headers()).json()
    assert 'owner_email' not in state
    url=state['clips'][0]['url']
    assert client.get(url.split('?')[0]).status_code==401
    r=client.get(url,headers={'Range':'bytes=2-5'})
    assert r.status_code==206 and r.content==b'2345'
    assert client.get(url,headers={'Range':'bytes=99-'}).status_code==416
    assert client.get(url+'&download=1').headers['content-disposition'].startswith('attachment')

def test_cleanup_preserves_running_jobs(client):
    active=job('rendering');done=job()
    assert client.post('/api/admin/cleanup',headers={'Authorization':'Bearer '+'a'*40}).status_code==200
    assert main.job_dir(active).exists() and not main.job_dir(done).exists()

def test_job_traversal_rejected():
    with pytest.raises(Exception):main.job_dir('..')

def test_phone_frontend(client):
    if not main.FRONTEND.exists(): pytest.skip('Frontend is distributed separately for Netlify')
    r=client.get('/app/')
    assert r.status_code==200 and 'manifest.webmanifest' in r.text
    assert client.get('/app/config.js').text=='window.REELCRATE_BACKEND = "";'
    assert client.get('/app/manifest.webmanifest').json()['display']=='standalone'

def test_rerender_preserves_original_on_failure(client,monkeypatch):
    jid=job(); directory=main.job_dir(jid)
    (directory/'clip_01_src.mp4').write_bytes(b'source')
    def fail(*args,**kwargs):
        assert main.read_state(jid)['status']=='rerendering'
        main.emergency_cleanup()
        return False
    monkeypatch.setattr(main,'render_clip',fail)
    r=client.post(f'/api/clips/{jid}/1/rerender',headers=headers(),json={'caption':'New caption'})
    assert r.status_code==500
    assert (directory/'clip_01.mp4').read_bytes()==b'0123456789'
    assert main.read_state(jid)['status']=='done'

def test_missing_email_key_is_retryable(monkeypatch):
    monkeypatch.setattr(email_service,'RESEND_API_KEY','')
    assert email_service.send_email('test@example.test','test','test') is False
