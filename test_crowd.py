import cv2
import numpy as np
from crowd import motion_evidence, select_reactions
from analyze import assign_hooks


def texture():
    return np.random.default_rng(17).integers(30,210,(180,320),dtype=np.uint8)


def test_static_and_camera_pan_are_not_crowd_reactions():
    a=texture()
    assert motion_evidence(a,a)[0]==0
    b=cv2.warpAffine(a,np.float32([[1,0,6],[0,1,2]]),(320,180))
    assert motion_evidence(a,b)[0]<.12


def test_light_change_is_not_a_reaction():
    a=texture(); b=np.minimum(a.astype(int)+25,255).astype(np.uint8)
    assert motion_evidence(a,b)[0]<.12


def test_distributed_independent_motion_is_detected():
    a=texture();b=a.copy()
    for y,x in [(25,30),(80,130),(120,220)]:
        b[y:y+45,x:x+60]=np.roll(a[y:y+45,x:x+60],5,axis=1)
    score,coverage=motion_evidence(a,b)
    assert score>.12
    assert coverage>=.25


def features():
    scores=np.zeros(900)
    for p in (90,270,450,630,810): scores[p-6:p+7]=.8
    return dict(scores=scores,coverage=np.full(900,.5),duration=300,fps=3)


def test_counts_lengths_context_and_nonoverlap():
    for length in (15,30,60):
        clips=select_reactions(features(),3,length)
        assert len(clips)==3
        for c in clips:
            assert abs(c['end_sec']-c['start_sec']-length)<.001
            assert c['start_sec']<c['peak_sec']<c['end_sec']
            assert c['start_sec']>=0 and c['end_sec']<=300
        assert all(a['end_sec']<=b['start_sec'] for a,b in zip(clips,clips[1:]))


def test_no_filler_for_insufficient_or_static_video():
    f=features()
    assert len(select_reactions(f,12,30))==5
    f['scores']=np.zeros(900)
    assert select_reactions(f,5,30)==[]


def test_single_frame_spikes_not_selected():
    f=features(); f['scores']=np.zeros(900); f['scores'][270]=10
    assert select_reactions(f,5,30)==[]


def test_crowd_hooks_do_not_claim_music_transitions():
    clips=assign_hooks(select_reactions(features(),3,30))
    assert {c['tag'] for c in clips}=={'CROWD'}
    assert len({c['hook'] for c in clips})==3


def test_crowd_render_never_tracks_one_face(monkeypatch,tmp_path):
    from types import SimpleNamespace
    import render
    calls=[]
    def run(cmd,**kwargs):
        calls.append(cmd)
        if cmd[0]=='ffprobe': return SimpleNamespace(stdout='codec_type=video\nwidth=1280\nheight=720\n',returncode=0)
        return SimpleNamespace(stdout='',stderr='',returncode=0)
    monkeypatch.setattr(render.subprocess,'run',run)
    monkeypatch.setattr(render,'_face_crop_plan',lambda *args: (_ for _ in ()).throw(AssertionError('Crowd mode must not track a face')))
    clip=assign_hooks(select_reactions(features(),1,30))[0]
    render.render_clip('test.mp4',clip,str(tmp_path/'out.mp4'),'DJ','freq_bars',framing='crop')
    command=next(c for c in calls if c[0]=='ffmpeg')
    graph=command[command.index('-filter_complex')+1]
    assert 'force_original_aspect_ratio=decrease' in graph
    assert 'gblur' not in graph and 'crop@follow' not in graph
    assert clip['framing_applied']=='fit'


def test_crowd_region_rejects_invalid_coordinates():
    import pytest
    from crowd import validate_region,region_filter
    assert validate_region(None) is None
    assert validate_region([0,.35,.68,.65])==[0,.35,.68,.65]
    assert region_filter([0,.35,.68,.65]).startswith('crop=')
    for bad in ([0,0,1.1,1],[0,0,.01,1],[0,0,float('nan'),1],[-.1,0,.5,.5],['x',0,1,1],{'x':1},[True,0,1,1]):
        with pytest.raises(ValueError): validate_region(bad)


def test_crowd_area_is_used_for_rendering(monkeypatch,tmp_path):
    import render
    from types import SimpleNamespace
    commands=[]
    def run(cmd,**kw):
        commands.append(cmd)
        return SimpleNamespace(stdout='codec_type=video\nwidth=1280\nheight=720\n',returncode=0,stderr='')
    monkeypatch.setattr(render.subprocess,'run',run)
    clip=assign_hooks(select_reactions(features(),1,15))[0]
    clip['crowd_region']=[0,.355,.685,.645]
    render.render_clip('test.mp4',clip,str(tmp_path/'clip.mp4'),'DJ','freq_bars')
    command=next(c for c in commands if c[0]=='ffmpeg');graph=command[command.index('-filter_complex')+1]
    assert 'crop=iw*0.685:ih*0.645:iw*0.0:ih*0.355,scale=' in graph
