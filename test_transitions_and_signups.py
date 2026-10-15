import json
import numpy as np
import pytest
import soundfile as sf
from unittest.mock import Mock
import main, auth
from test_regressions import client, isolated, headers
from transitions import extract_features, find_boundaries, select_transition_clips


def music(frequencies, seconds=80, sr=11025):
    t=np.arange(int(seconds*sr))/sr
    notes=sum(np.sin(2*np.pi*f*t)*(1+.25*np.sin(2*np.pi*(i+1)*.23*t)) for i,f in enumerate(frequencies))
    return (.1*notes*(.7+.3*np.cos(2*np.pi*2*t)**4)).astype('float32')


def test_known_crossfade_has_both_tracks(tmp_path):
    a=music([110,220,330,440]);b=music([155.56,311.12,466.68,622.24])
    overlap=8*11025; ramp=np.linspace(0,1,overlap)
    y=np.concatenate([a[:-overlap],a[-overlap:]*(1-ramp)+b[:overlap]*ramp,b[overlap:]])
    path=tmp_path/'mix.wav';sf.write(path,y,11025)
    f=extract_features(path); events=find_boundaries(f)
    clips=select_transition_clips(events,f['duration'],3,30)
    assert clips
    clip=min(clips,key=lambda c:abs(c['peak_sec']-76))
    assert abs(clip['peak_sec']-76)<6
    assert clip['start_sec']<72 and clip['end_sec']>80
    assert clip['end_sec']-clip['start_sec']==30


def test_volume_change_is_not_a_track_transition(tmp_path):
    y=music([110,220,330,440],seconds=150)
    y[75*11025:]*=.25
    path=tmp_path/'gain.wav';sf.write(path,y,11025)
    assert find_boundaries(extract_features(path))==[]


def test_short_change_returning_to_same_music_is_not_transition(tmp_path):
    y=music([110,220,330,440],seconds=150)
    y[70*11025:80*11025]=music([155.56,311.12,466.68,622.24],seconds=10)
    path=tmp_path/'fill.wav';sf.write(path,y,11025)
    assert find_boundaries(extract_features(path))==[]


def test_requested_number_never_filled_with_nontransitions():
    events=[dict(transition_start_sec=50,transition_end_sec=70,peak_sec=60,score=2),
            dict(transition_start_sec=90,transition_end_sec=92,peak_sec=91,score=1)]
    clips=select_transition_clips(events,150,10,30)
    assert len(clips)==1 # 20s blend + 12s context does not fit a 30s clip.
    assert clips[0]['peak_sec']==91
    clips=select_transition_clips(events,150,10,60)
    assert len(clips)==1 # overlapping exports are not used to pad the number.


@pytest.fixture(autouse=True)
def signup_storage(tmp_path,monkeypatch):
    monkeypatch.setattr(main,'WAITLIST_FILE',tmp_path/'waitlist.json')
    monkeypatch.setattr(main,'ADMIN_EMAIL','owner@djmail.com')
    monkeypatch.setattr(main,'queue_email',Mock())


def test_counts_merge_duplicates_and_exclude_owner(client):
    main._save_waitlist([{'email':'ONE@djmail.com','joined_at':1}, {'email':'one@djmail.com','joined_at':2},
                        {'email':'owner@djmail.com','joined_at':0}])
    auth._save_users({'one@djmail.com':{'verified':True},'two@djmail.com':{'verified':True},
                     'unverified@djmail.com':{'verified':False}})
    r=client.get('/api/founder-spots')
    assert r.json()=={'capacity':40,'signed_up':2,'remaining':38}
    assert '@' not in r.text and r.headers['cache-control']=='no-store'


def test_owner_only_import_and_read(client):
    payload={'entries':[{'email':'dj@djmail.com','joined_at':123}]}
    for path in ['/api/admin/signups','/api/admin/signups/import']:
        r=client.post(path,json=payload,headers=headers()) if path.endswith('import') else client.get(path,headers=headers())
        assert r.status_code==403
    auth._save_users({'owner@djmail.com':{'verified':True}})
    h=headers('owner@djmail.com')
    assert client.post('/api/admin/signups/import',json=payload,headers=h).json()['imported']==1
    assert client.post('/api/admin/signups/import',json=payload,headers=h).json()['imported']==0
    assert client.get('/api/admin/signups',headers=h).json()['remaining']==39
    main.queue_email.assert_not_called()


def test_failed_storage_never_claims_signup_saved(client,monkeypatch):
    monkeypatch.setattr(main,'_save_waitlist',Mock(side_effect=OSError('disk full')))
    with pytest.raises(OSError):
        client.post('/api/waitlist',json={'email':'dj@djmail.com'})
    main.queue_email.assert_not_called()


def test_corrupt_storage_not_replaced(client):
    main.WAITLIST_FILE.write_text('broken')
    assert client.get('/api/founder-spots').status_code==503
    assert main.WAITLIST_FILE.read_text()=='broken'


def test_honeypot_does_not_register_or_send(client):
    assert client.post('/api/waitlist',json={'email':'bot@djmail.com','bot_field':'spam'}).status_code==200
    assert not main.WAITLIST_FILE.exists()
    main.queue_email.assert_not_called()


def test_full_offer_never_goes_negative_or_sends_founder_code(client):
    main._save_waitlist([{'email':f'dj{i}@djmail.com','joined_at':i+1} for i in range(40)])
    r=client.post('/api/waitlist',json={'email':'next@djmail.com'}).json()
    assert r['remaining']==0 and r['founder'] is False
    assert main.queue_email.call_count==1 # owner alert, no false founder offer
