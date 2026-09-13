"""Exact video clocks, retained pixels and explicit generated descriptions."""
import hashlib
import struct
from fractions import Fraction
import pytest
from scone import Scone, SconeError
from stub_server import StubScone

PNG=b'\x89PNG\r\n\x1a\n'+struct.pack('>I4sII',13,b'IHDR',64,32)+bytes(9)
SHA=hashlib.sha256(PNG).hexdigest()
START=9007199254740993


def catalogue():
    return {'schema_version':1,'timestamp_encoding':'decimal-string','space':'alpha','episode_id':'7',
      'evidence':{'original':{'attachment_id':'a'*64,'media_type':'video/mp4','bytes':100,'filename':'clip.mp4'},
       'manifest':{'attachment_id':'b'*64,'media_type':'application/json','bytes':1000,'filename':'manifest.json'},
       'filename':'clip.mp4','format':'mp4','parser':'video-frame-ocr','segments':[],
       'metadata':{'extraction':'ocr','coverage':'sampled-video-frames'},'download_path':'/v1/attachments/'+'a'*64,
       'video':{'source_sha256':'a'*64,'decoder_revision':'c'*64,'policy_revision':'decoded-video-frames-v1','model_revision':'test-v1',
        'policy':{'interval_seconds':1,'max_frames':64,'max_duration_seconds':600,'max_pixels':20000000,'max_frame_bytes':10000000,'max_total_bytes':32000000},
        'stream_index':0,'time_base':'1/3','start_timestamp':str(START),'duration_ticks':'3','decoded_frames':3,'unavailable_requests':0,
        'frames':[{'ordinal':1,'presentation_timestamp':str(START+1),'requested_seconds':[0],'width':64,'height':32,
                   'png_sha256':SHA,'png_bytes':len(PNG),'ocr_engine':'test','empty':True}]}}}


def interpretation():
    return {'schema_version':1,'space':'alpha','episode_id':'7','persisted':False,'original_sha256':'a'*64,'manifest_sha256':'b'*64,
      'frame':{'ordinal':1,'presentation_timestamp':str(START+1),'time_base':'1/3','png_sha256':SHA,'width':64,'height':32},
      'understanding':{'text':'A generated description.','model':'my-local-model','origin':'model_generated','attachment_id':SHA,
       'source':'video:episode:7/stream:0/frame:1','media_type':'image/png','width':64,'height':32}}


@pytest.fixture
def fixture():
    with StubScone() as server,Scone(server.base_url,'key') as client:
        server.route('GET','/v1/capabilities',200,{'schema_version':1,'implementation':'python',
          'features':{'documents.provenance':True,'documents.video.understand':True}})
        server.route('GET','/v1/status',200,{'space':'alpha'})
        server.route('GET','/v1/episodes/7/document/video/catalogue',200,catalogue())
        server.route('POST','/v1/episodes/7/document/video/frames/1/understand',200,interpretation())
        yield server,client.video_documents(expected_space='alpha'),client


def test_catalogue_keeps_lossless_clock_without_inference(fixture):
    server,videos,_=fixture;saved=videos.catalogue(7)
    assert saved.start_timestamp==START and saved.frames[0].presentation_timestamp==START+1
    assert saved.frame_time(1)==Fraction(1,3) and saved.frames[0].empty
    assert saved.original.attachment_id=='a'*64
    assert all(r.method=='GET' for r in server.requests)


@pytest.mark.parametrize('patch',[{'episode_id':7},{'episode_id':'8'},{'space':'beta'},{'schema_version':True},{'timestamp_encoding':'number'}])
def test_wrong_envelope_refuses(fixture,patch):
    server,videos,_=fixture;server.route('GET','/v1/episodes/7/document/video/catalogue',200,{**catalogue(),**patch})
    with pytest.raises(SconeError):videos.catalogue(7)


@pytest.mark.parametrize('patch',[{'source_sha256':'e'*64},{'start_timestamp':str(2**63)},{'duration_ticks':'0'},
 {'time_base':'0/3'},{'unavailable_requests':1},{'frames':[]}])
def test_inconsistent_sampling_refuses(fixture,patch):
    server,videos,_=fixture;body=catalogue();body['evidence']['video'].update(patch)
    server.route('GET','/v1/episodes/7/document/video/catalogue',200,body)
    with pytest.raises(SconeError):videos.catalogue(7)


def test_single_explicit_post_returns_actual_model(fixture):
    server,videos,_=fixture;result=videos.interpret(videos.catalogue(7),1,prompt='😀'*16000)
    assert result.model=='my-local-model' and result.text=='A generated description.'
    assert result.frame.presentation_timestamp==START+1 and not result.persisted
    posts=[r for r in server.requests if r.method=='POST']
    assert len(posts)==1 and posts[0].json=={'prompt':'😀'*16000}
    assert server.requests[-1].path.endswith('/catalogue')


@pytest.mark.parametrize('patch',[{'original_sha256':'e'*64},{'manifest_sha256':'e'*64},{'persisted':True},{'episode_id':'8'}])
def test_substituted_interpretation_refuses(fixture,patch):
    server,videos,_=fixture;server.route('POST','/v1/episodes/7/document/video/frames/1/understand',200,{**interpretation(),**patch})
    with pytest.raises(SconeError):videos.interpret(videos.catalogue(7),1,prompt='Describe')


def test_changed_source_refuses_before_inference(fixture):
    server,videos,_=fixture;saved=videos.catalogue(7);changed=catalogue();changed['evidence']['manifest']['attachment_id']='e'*64
    server.route('GET','/v1/episodes/7/document/video/catalogue',200,changed)
    with pytest.raises(SconeError):videos.interpret(saved,1,prompt='Describe')
    assert not any(r.method=='POST' for r in server.requests)


def test_invalid_task_and_frame_refuse_before_requests(fixture):
    server,videos,_=fixture;saved=videos.catalogue(7);count=len(server.requests)
    for prompt in ['',' ','\ud800','\0','😀'*16001]:
        with pytest.raises(SconeError):videos.interpret(saved,1,prompt=prompt)
    with pytest.raises(SconeError):videos.interpret(saved,0,prompt='Describe')
    assert len(server.requests)==count


def test_checked_png_requires_bound_headers_hash_and_dimensions(fixture,monkeypatch):
    from requests import Response
    server,videos,client=fixture;saved=videos.catalogue(7);request=client.session.request
    headers={'Content-Type':'image/png','X-Scone-Video-Frame-SHA256':SHA,'X-Scone-Video-Frame-Ordinal':'1',
      'X-Scone-Video-PTS':str(START+1),'X-Scone-Video-Time-Base':'1/3'}
    current={'headers':headers,'data':PNG,'status':200}
    def serve(method,url,**kwargs):
        if not url.endswith('/frames/1'):return request(method,url,**kwargs)
        assert kwargs['allow_redirects'] is False and kwargs['stream'] is True
        response=Response();response.status_code=current['status'];response._content=current['data'];response._content_consumed=True
        response.headers.update(current['headers']);return response
    monkeypatch.setattr(client.session,'request',serve)
    assert videos.frame(saved,1)==PNG
    for change in [{'X-Scone-Video-PTS':str(START+2)},{'Content-Type':'image/jpeg'},{'X-Scone-Video-Frame-SHA256':'e'*64}]:
        current['headers']={**headers,**change}
        with pytest.raises(SconeError):videos.frame(saved,1)
    current['headers']=headers;current['data']=PNG+b'extra'
    with pytest.raises(SconeError):videos.frame(saved,1)
    current['data']=PNG;current['status']=302
    with pytest.raises(SconeError):videos.frame(saved,1)


def test_source_changed_during_inference_refuses_result(fixture):
    server,videos,_=fixture;saved=videos.catalogue(7);respond=server._respond_with
    def replacing(method,path):
        answer=respond(method,path)
        if method=='POST':
            changed=catalogue();changed['evidence']['manifest']['attachment_id']='e'*64
            server.route('GET','/v1/episodes/7/document/video/catalogue',200,changed)
        return answer
    server._respond_with=replacing
    with pytest.raises(SconeError):videos.interpret(saved,1,prompt='Describe')
    assert len([r for r in server.requests if r.method=='POST'])==1


def test_modified_snapshot_and_wrong_space_refuse_locally(fixture):
    from dataclasses import replace
    server,videos,_=fixture;saved=videos.catalogue(7);before=len(server.requests)
    for source in [replace(saved,space='beta'),replace(saved,frames=(replace(saved.frames[0],width=65),))]:
        with pytest.raises(SconeError):videos.interpret(source,1,prompt='Describe')
    assert len(server.requests)==before


def test_nonempty_ocr_regions_preserve_utf8_byte_spans(fixture):
    server,videos,_=fixture;body=catalogue();body['evidence']['video']['frames'][0]['empty']=False
    body['evidence']['segments']=[{'locator':'video:stream:0/frame:1','text':'😀','metadata':{'video_frame_ordinal':'1','extraction':'ocr','engine':'test'},
      'regions':[{'text':'😀','start':0,'end':4,'box':[0,0,1,1],'coordinate_space':'normalized_displayed_frame_top_left','score':0.9}]}]
    server.route('GET','/v1/episodes/7/document/video/catalogue',200,body)
    saved=videos.catalogue(7);assert saved.frames[0].text=='😀' and saved.frames[0].regions[0].end==4
    body['evidence']['segments'][0]['regions'][0]['end']=3
    with pytest.raises(SconeError):videos.catalogue(7)


@pytest.mark.parametrize('engine',['é'*49,'😀'*96])
def test_native_ocr_engine_codepoint_limit(fixture,engine):
    server,videos,_=fixture;body=catalogue();body['evidence']['video']['frames'][0]['ocr_engine']=engine
    body['evidence']['video']['policy_revision']='é'*96
    server.route('GET','/v1/episodes/7/document/video/catalogue',200,body)
    assert videos.catalogue(7).frames[0].ocr_engine==engine


def test_ocr_regions_cannot_omit_nonwhitespace_text(fixture):
    server,videos,_=fixture
    for content,start,end,label in [('prefix 😀',7,11,'😀'),('😀 tail',0,4,'😀')]:
        body=catalogue();body['evidence']['video']['frames'][0]['empty']=False
        body['evidence']['segments']=[{'locator':'video:stream:0/frame:1','text':content,'metadata':{'video_frame_ordinal':'1','extraction':'ocr','engine':'test'},
          'regions':[{'text':label,'start':start,'end':end,'box':[0,0,1,1],'coordinate_space':'normalized_displayed_frame_top_left','score':None}]}]
        server.route('GET','/v1/episodes/7/document/video/catalogue',200,body)
        with pytest.raises(SconeError):videos.catalogue(7)


def test_huge_region_coordinate_refuses_as_client_error(fixture):
    server,videos,_=fixture;body=catalogue();body['evidence']['video']['frames'][0]['empty']=False
    body['evidence']['segments']=[{'locator':'video:stream:0/frame:1','text':'x','metadata':{'video_frame_ordinal':'1','extraction':'ocr','engine':'test'},
      'regions':[{'text':'x','start':0,'end':1,'box':[0,0,10**400,1],'coordinate_space':'normalized_displayed_frame_top_left','score':None}]}]
    server.route('GET','/v1/episodes/7/document/video/catalogue',200,body)
    with pytest.raises(SconeError):videos.catalogue(7)


def test_one_frame_may_use_the_native_total_region_allowance():
    from scone import VideoCatalogue
    body=catalogue();body['evidence']['video']['frames'][0]['empty']=False
    regions=[{'text':'x','start':2*i,'end':2*i+1,'box':[0,0,1,1],'coordinate_space':'normalized_displayed_frame_top_left','score':None} for i in range(10001)]
    body['evidence']['segments']=[{'locator':'video:stream:0/frame:1','text':'\n'.join('x' for _ in regions),
      'metadata':{'video_frame_ordinal':'1','extraction':'ocr','engine':'test'},'regions':regions}]
    assert len(VideoCatalogue.from_json(body,expected_space='alpha',episode_id=7).frames[0].regions)==10001


def test_native_unicode_catalogue_filename_is_not_a_write_byte_limit():
    from scone import VideoCatalogue
    body=catalogue();body['evidence']['filename']='é'*512+'.mp4'
    assert VideoCatalogue.from_json(body,expected_space='alpha',episode_id=7).filename==body['evidence']['filename']
