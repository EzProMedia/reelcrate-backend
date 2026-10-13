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
    monkeypatch.setattr(billing.stripe.Customer, 'retrieve', Mock(return_value={'id':'cus_mock'}))
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


def test_missing_portal_customer_routes_to_setup_with_cors(client, monkeypatch):
    create = Mock(side_effect=billing.stripe.error.InvalidRequestError(
        "No such customer: 'cus_mock'; a similar object exists in test mode", 'customer', code='resource_missing'))
    monkeypatch.setattr(billing.stripe.billing_portal.Session, 'create', create)
    r = client.post('/api/billing/portal', headers={**headers(), 'Origin':'https://reelcrate.app'})
    assert r.status_code == 409
    assert r.json()['detail']['code'] == 'billing_setup_required'
    assert r.headers['access-control-allow-origin'] == 'https://reelcrate.app'
    u = auth._load_users()['dj@example.test']
    assert u['stripe_customer_id'] == 'cus_mock' and u['billing_setup_required']
    assert not billing.is_paying('dj@example.test')
    assert client.get('/api/auth/me', headers=headers()).json()['billing_setup_required']


@pytest.mark.parametrize('deleted', [False, True])
def test_checkout_replaces_missing_customer_preserving_history(client, monkeypatch, deleted):
    u = auth._load_users()['dj@example.test']
    u.update(subscription_id='sub_old', subscription_status='trialing',
             subscription_current_period_end=1, billing_setup_required=True)
    auth._save_users({'dj@example.test':u})
    retrieve = Mock(return_value={'id':'cus_mock','deleted':True}) if deleted else Mock(
        side_effect=billing.stripe.error.InvalidRequestError('No such customer: cus_mock', 'id', code='resource_missing'))
    monkeypatch.setattr(billing.stripe.Customer, 'retrieve', retrieve)
    customer = Mock(return_value=SimpleNamespace(id='cus_new'))
    monkeypatch.setattr(billing.stripe.Customer, 'create', customer)
    checkout = Mock(return_value=SimpleNamespace(id='cs_new', url='https://checkout.stripe.com/mock'))
    monkeypatch.setattr(billing.stripe.checkout.Session, 'create', checkout)
    r = client.post('/api/billing/checkout', headers=headers(), json={})
    assert r.status_code == 200
    assert checkout.call_args.kwargs['customer'] == 'cus_new'
    u = auth._load_users()['dj@example.test']
    assert u['verified'] and u['stripe_customer_id'] == 'cus_new'
    assert u['billing_history'][0]['subscription_id'] == 'sub_old'
    assert not u.get('billing_setup_required') and not u.get('subscription_status')
    assert not billing.is_paying('dj@example.test')  # A checkout draft grants no access.
    billing._apply_subscription({'id':'sub_old','customer':'cus_mock','status':'active',
        'metadata':{'reelcrate_email':'dj@example.test'}})
    assert auth._load_users()['dj@example.test']['stripe_customer_id'] == 'cus_new'
    assert not billing.is_paying('dj@example.test')


@pytest.mark.parametrize('where', ['portal', 'retrieve', 'create', 'checkout_retry'])
def test_stripe_failures_are_safe_readable_and_do_not_rebind_accounts(client, monkeypatch, where):
    failure = billing.stripe.error.APIConnectionError('private diagnostic must not reach browser')
    path = '/api/billing/checkout'
    if where == 'portal':
        path = '/api/billing/portal'
        monkeypatch.setattr(billing.stripe.billing_portal.Session, 'create', Mock(side_effect=failure))
    elif where == 'retrieve':
        monkeypatch.setattr(billing.stripe.Customer, 'retrieve', Mock(side_effect=failure))
    elif where == 'create':
        auth._save_users({'dj@example.test':{'email':'dj@example.test','verified':True}})
        monkeypatch.setattr(billing.stripe.Customer, 'create', Mock(side_effect=failure))
    else:
        monkeypatch.setattr(billing, '_resolve_promotion_code', lambda code:'promo_bad')
        monkeypatch.setattr(billing.stripe.checkout.Session, 'create', Mock(side_effect=[
            billing.stripe.error.InvalidRequestError('expired', 'discounts[0][promotion_code]'), failure]))
    before = auth._load_users()
    r = client.post(path, headers={**headers(), 'Origin':'https://reelcrate.app'}, json={'promo':'EXPIRED'})
    assert r.status_code == 502
    assert r.headers['access-control-allow-origin'] == 'https://reelcrate.app'
    assert 'private diagnostic' not in r.text and 'support@reelcrate.app' in r.text
    assert auth._load_users() == before


def test_existing_customer_opens_portal_without_replacement(client, monkeypatch):
    create = Mock(return_value=SimpleNamespace(url='https://billing.stripe.com/mock'))
    monkeypatch.setattr(billing.stripe.billing_portal.Session, 'create', create)
    r = client.post('/api/billing/portal', headers=headers())
    assert r.status_code == 200 and create.call_args.kwargs['customer'] == 'cus_mock'


def test_portal_configuration_error_does_not_reset_customer(client, monkeypatch):
    monkeypatch.setattr(billing.stripe.billing_portal.Session, 'create', Mock(
        side_effect=billing.stripe.error.InvalidRequestError('Missing portal configuration', 'configuration')))
    before = auth._load_users()
    assert client.post('/api/billing/portal', headers=headers()).status_code == 502
    assert auth._load_users() == before

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

@pytest.mark.parametrize('field,value', [('bpm_min','nan'),('bpm_max','221'),('bpm_min','-5'),('framing','bad'),('watermark','x'*41)])
def test_bad_clip_settings_rejected(client,monkeypatch,field,value):
    monkeypatch.setattr(main,'is_paying',lambda email:True)
    r=client.post('/api/upload',headers=headers(),data={field:value},files={'file':('set.mp4',b'fake')})
    assert r.status_code==400
    assert not list(main.JOBS_DIR.iterdir())

@pytest.mark.parametrize('status,expected',[('active',True),('trialing',False)])
def test_upload_applies_real_controls(client,monkeypatch,status,expected):
    auth._save_users({'dj@example.test':dict(verified=True,subscription_status=status)})
    monkeypatch.setattr(main,'is_paying',lambda email:True)
    received=[]
    async def capture(*args,**kwargs): received.append((args,kwargs))
    monkeypatch.setattr(main,'process_job',capture)
    r=client.post('/api/upload',headers=headers(),data={'show_caption':'false','framing':'crop','watermark':'DJ Test','hide_logo':'true'},files={'file':('set.mp4',b'fake')})
    assert r.status_code==200
    st=main.read_state(r.json()['job_id'])
    assert (st['show_caption'],st['framing'],st['watermark'],st['hide_logo'])==(False,'crop','DJ Test',expected)
    assert received[0][1]['show_caption'] is False
    assert received[0][1]['hide_logo'] is expected


def test_zip_is_private_and_contains_only_finished_clips(client):
    import io,zipfile
    jid=job()
    (main.job_dir(jid)/'clip_01_src.mp4').write_bytes(b'private source')
    (main.job_dir(jid)/'users.json').write_text('private')
    st=client.get('/api/jobs/'+jid,headers=headers()).json()
    assert client.get('/api/downloads/'+jid).status_code==401
    assert client.get('/api/downloads/'+jid+'?ticket='+auth.make_token(jid+'/clip_01.mp4',kind='clip')).status_code==401
    r=client.get(st['download_url'])
    assert r.status_code==200
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        assert z.namelist()==['clip_01.mp4']
        assert z.read('clip_01.mp4')==b'0123456789'


def test_rerender_removes_caption_and_name_and_retains_tempo(client,monkeypatch):
    jid=job(); d=main.job_dir(jid);(d/'clip_01_src.mp4').write_bytes(b'source')
    st=main.read_state(jid);st.update(watermark='old',framing='fit')
    st['clips'][0].update(duration_sec=15,local_bpm=123,local_bpm_confidence=.8)
    main.write_state(jid,st)
    calls=[]
    def render(src,clip,path,wm,viz,**kw):
        calls.append((clip,wm,kw));Path(path).write_bytes(b'updated');return True
    monkeypatch.setattr(main,'render_clip',render)
    r=client.post(f'/api/clips/{jid}/1/rerender',headers=headers(),json={'caption':'','watermark':''})
    assert r.status_code==200
    clip,wm,kw=calls[0]
    assert clip['custom_title']==wm==''
    assert clip['local_bpm_confidence']==.8
    assert not kw['show_caption'] and kw['framing']=='fit'
    assert main.read_state(jid)['clips'][0]['show_caption'] is False


def test_rerender_reports_the_actual_framing_after_tracking_fallback(client,monkeypatch):
    jid=job();d=main.job_dir(jid);(d/'clip_01_src.mp4').write_bytes(b'source')
    def render(src,clip,path,*args,**kwargs):
        clip['framing_applied']='fit';clip['tracking']={'coverage':.2}
        Path(path).write_bytes(b'updated');return True
    monkeypatch.setattr(main,'render_clip',render)
    r=client.post(f'/api/clips/{jid}/1/rerender',headers=headers(),json={'caption':'from the booth','framing':'crop'})
    assert r.status_code==200
    c=main.read_state(jid)['clips'][0]
    assert c['framing']=='fit' and c['framing_requested']=='crop'
    assert c['tracking']['coverage']==.2


def test_analysis_grid_stays_aligned_across_chunks(tmp_path):
    import numpy as np,soundfile as sf
    from analyze import _stream_rms_and_flux
    p=tmp_path/'audio.wav'; sr=11025
    rng=np.random.default_rng(123); sf.write(p,rng.normal(0,.1,sr*5),sr)
    whole=_stream_rms_and_flux(str(p),chunk_sec=10)
    chunked=_stream_rms_and_flux(str(p),chunk_sec=.63)
    for a,b in zip(whole[:2],chunked[:2]):
        np.testing.assert_allclose(a,b,atol=1e-5)


def test_short_set_does_not_duplicate_clips_and_silence_rejected(tmp_path):
    import numpy as np,soundfile as sf
    from analyze import detect_drops
    p=tmp_path/'short.wav';sr=11025;t=np.arange(sr*4)/sr
    sf.write(p,.3*np.sin(t*2*np.pi*100)*(0.6+0.4*np.sin(t*2*np.pi*2)),sr)
    clips=detect_drops(str(p),num_clips=10,clip_len_sec=30)
    assert len(clips)==1 and clips[0]['start_sec']==0 and clips[0]['end_sec']==4
    sf.write(p,np.zeros(sr*4),sr)
    with pytest.raises(ValueError,match='no usable audio'):detect_drops(str(p))

@pytest.mark.parametrize('bpm',[30,40,50,60,70,80,95,105,110,120,128,140,174,190])
def test_known_beat_rates_are_not_quantized_or_octave_shifted(bpm):
    import numpy as np
    from analyze import _stable_bpm
    sr=11025;t=np.arange(sr*20)/sr;y=np.zeros_like(t)
    for beat in np.arange(0,20,60/bpm):
        dt=t-beat
        y+=np.where((dt>=0)&(dt<.12),np.sin(2*np.pi*90*dt)*np.exp(-np.maximum(dt,0)*40),0)
    actual,confidence=_stable_bpm(y,sr)
    assert abs(actual-bpm)<.8,(bpm,actual)
    assert confidence>=.55


def test_bar_length_repeat_prefers_a_strong_observed_beat_peak():
    from analyze import _preferred_tempo_candidate
    candidates=[(34.84,.531,.480),(139.48,.467,.465),(70.33,.376,.359)]
    assert _preferred_tempo_candidate(candidates)==candidates[1]
    # A genuine slow pulse, a weak subdivision, or an unrelated faster rhythm
    # must not be arbitrarily doubled just to land in a typical DJ tempo range.
    for candidates in [
        [(40,.7,.63),(80,.2,.19)],
        [(40,.7,.63),(80,.4,.38)],
        [(40,.7,.63),(110,.62,.61)],
        [(70,.7,.66),(140,.68,.65)],
    ]:
        assert _preferred_tempo_candidate(candidates)==candidates[0]


@pytest.mark.parametrize('bpm',[70,130,140,174])
def test_accented_beats_do_not_report_the_four_beat_bar(bpm):
    import numpy as np
    from analyze import _stable_bpm
    sr=11025;t=np.arange(sr*30)/sr;y=np.zeros_like(t)
    for i,beat in enumerate(np.arange(0,30,60/bpm)):
        dt=t-beat;accent=(1,.6,.8,.6)[i%4]
        y+=accent*np.where((dt>=0)&(dt<.12),np.sin(2*np.pi*90*dt)*np.exp(-np.maximum(dt,0)*40),0)
    actual,confidence=_stable_bpm(y,sr)
    assert abs(actual-bpm)<.8,(bpm,actual)
    assert confidence>=.55


def test_noise_is_not_confident_and_range_returns_actual_interpretation():
    import numpy as np
    from analyze import _stable_bpm,_bpm_for_range
    _,confidence=_stable_bpm(np.random.default_rng(42).normal(0,.1,11025*20),11025)
    assert confidence<.55
    assert _bpm_for_range(70,135,145)==140
    assert _bpm_for_range(140,60,90)==70
    assert _bpm_for_range(128,90,110)==0


def test_crop_favors_foreground_over_background_crowd():
    from render import _select_foreground_face
    dj=(400,100,80,90); crowd=[(100+i,120,30,35) for i in range(20)]
    assert _select_foreground_face([dj]+crowd,None,640)==dj
    assert _select_foreground_face([dj]+crowd,.2,640)==dj
    # Retain a nearby foreground subject when a similar face crosses.
    other=(100,100,85,90)
    assert _select_foreground_face([dj,other],.7,640)==dj
    assert _select_foreground_face([dj,other],None,640)==other
    # Looking down must not transfer the track to a smaller crowd face.
    assert _select_foreground_face(crowd,.7,640,7200) is None
    assert _select_foreground_face([other],.7,640,7200) is None


def test_bundled_foreground_detector_loads_and_ignores_blank_frames():
    import cv2,numpy as np,hashlib
    from render import FACE_MODEL
    assert hashlib.sha256(FACE_MODEL.read_bytes()).hexdigest()=='8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4'
    detector=cv2.FaceDetectorYN.create(str(FACE_MODEL),'',(640,360),.8,.3,1000)
    _,faces=detector.detect(np.zeros((360,640,3),dtype=np.uint8))
    assert faces is None


def test_tracking_follows_movement_and_resets_at_camera_cuts():
    from render import _crop_track
    observations=[(i*.25,.25+i*.015,.07) for i in range(8)]
    observations += [(2+i*.25,.75-i*.015,.07) for i in range(8)]
    points=_crop_track(observations,4,.316,[2.])
    assert points is not None
    assert points[0][1] < .3 and points[-1][1] > .6
    before=next(x for t,x in points if abs(t-1.9999)<.00001)
    after=next(x for t,x in points if t==2.)
    assert after-before > .3  # cut, no pan across the empty middle
    for t,cx,width in observations:
        x=min(points,key=lambda p:abs(p[0]-t))[1]
        assert x-.158 <= cx-width/2 and x+.158 >= cx+width/2


def test_tracking_refuses_blind_intervals_and_faces_too_wide():
    from render import _crop_track
    assert _crop_track([(0,.5,.1),(.25,.5,.1),(3,.5,.1)],4,.316) is None
    assert _crop_track([(0,None,0),(.25,None,0)],.5,.316) is None
    assert _crop_track([(0,.5,.4),(.25,.5,.4)],.5,.316) is None
    assert _crop_track([(0,.5,.1),(.25,None,0),(.5,.52,.1)],.75,.316)


def test_occluded_face_motion_uses_local_pixels():
    import cv2,numpy as np
    from render import _flow_face_box
    previous=np.random.default_rng(17).integers(0,255,(180,320),dtype=np.uint8)
    current=cv2.warpAffine(previous,np.float32([[1,0,12],[0,1,5]]),(320,180))
    box=_flow_face_box(previous,current,np.array([100,50,60,70],dtype=np.float32))
    assert box is not None
    assert abs(box[0]-112)<1 and abs(box[1]-55)<1
    assert _flow_face_box(np.zeros_like(previous),np.zeros_like(previous),np.array([100,50,60,70])) is None


def test_new_shot_uses_foreground_evidence_beyond_its_first_frame(monkeypatch):
    import numpy as np,render
    gray=np.zeros((180,320),dtype=np.uint8)
    crowd=(230,40,20,25);dj=(70,40,60,70)
    monkeypatch.setattr(render,'_flow_face_box',lambda previous,current,box:box)
    samples=[(0.,gray,[crowd]),(.25,gray,[crowd]),(.5,gray,[crowd,dj]),(.75,gray,[crowd,dj])]
    path=render._track_shot(samples,320)
    assert all(abs(x-100/320)<.001 for _,x,_ in path)


def test_crop_commands_preserve_long_motion_paths_without_expression_limits(tmp_path):
    from render import _crop_commands
    import math
    points=[(i*.25,.5+.2*math.sin(i)) for i in range(241)]
    path=tmp_path/'motion.txt'
    _crop_commands(points,3412,60,path)
    lines=path.read_text().splitlines()
    assert len(lines)==1801 and max(map(len,lines))<45
    offsets=[int(line.split()[-1].rstrip(';')) for line in lines]
    assert min(offsets)>=0 and max(offsets)<=2332 and max(offsets)-min(offsets)>1000


def test_audio_energy_does_not_invent_transitions_drops_or_crowd_reactions():
    from analyze import assign_hooks
    clips=assign_hooks([dict(energy=e,flux=1.,flux_sustain=1.,peak_sec=i*30)
                        for i,e in enumerate([.9,.7,.4]*4)])
    assert {c['tag'] for c in clips}=={'ENERGY','MOMENT','VIBE'}
    forbidden=['transition','blend','drop','build','scream','room went','two tracks']
    assert all(not any(word in c['hook'] for word in forbidden) for c in clips)


@pytest.mark.parametrize('cut_frame', [10, 11, 12])
def test_camera_cut_command_arrives_on_its_first_frame(tmp_path, monkeypatch, cut_frame):
    """Exercise actual FFmpeg commands, including fractional timestamps."""
    import numpy as np, render
    monkeypatch.setattr(render, 'W', 108)
    cut = cut_frame / render.FPS
    path = tmp_path / 'motion.txt'
    initial = render._crop_commands([(0,.25),(cut-.0001,.25),(cut,.75),(1,.75)],640,1,path)
    pixels = subprocess.check_output([
        'ffmpeg','-v','error','-f','lavfi','-i',
        'color=red:s=640x180:r=24000/1001:d=1,drawbox=x=320:y=0:w=320:h=180:color=blue:t=fill',
        '-vf',f"{render.TRACK_CLOCK},sendcmd=f='{path}',crop@follow=108:180:x={initial}:y=0",
        '-frames:v','25','-pix_fmt','rgb24','-f','rawvideo','-'])
    frames = np.frombuffer(pixels,np.uint8).reshape(-1,180,108,3)
    colors = frames.mean(axis=(1,2))
    assert np.all(colors[:cut_frame,0] > 200)
    assert np.all(colors[cut_frame:,2] > 200)


def test_logo_validation_and_sanitized_storage(client,monkeypatch):
    from PIL import Image
    import io
    monkeypatch.setattr(main,'is_paying',lambda email:True)
    async def noop(*a,**kw):pass
    monkeypatch.setattr(main,'process_job',noop)
    logo=io.BytesIO();Image.new('RGBA',(120,80),(255,200,0,100)).save(logo,format='PNG')
    r=client.post('/api/upload',headers=headers(),files={'file':('set.mp4',b'fake'),'logo':('custom.png',logo.getvalue(),'image/png')})
    assert r.status_code==200
    jid=r.json()['job_id'];im=Image.open(main.job_dir(jid)/'dj-logo.png')
    assert im.mode=='RGBA' and im.size==(120,80)
    assert main.read_state(jid)['has_custom_logo']
    main.write_state(jid,{**main.read_state(jid),'status':'done'})
    r=client.post('/api/upload',headers=headers(),files={'file':('set.mp4',b'fake'),'logo':('bad.png',b'<script>bad</script>','image/png')})
    assert r.status_code==400


def test_caption_wrap_preserves_wide_handles_inside_frame():
    from render import wrap_caption,FONT_BOLD
    from PIL import ImageFont
    text='W'*80+' DJ: 100% [LIVE]'
    lines=wrap_caption(text,52)
    assert ''.join(lines).replace(' ','')==text.replace(' ','')
    font=ImageFont.truetype(FONT_BOLD,52)
    assert all(font.getlength(line)<=850 for line in lines)
