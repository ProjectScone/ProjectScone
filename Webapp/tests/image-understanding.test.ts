import {test} from 'node:test';
import assert from 'node:assert/strict';
import {parseImageUnderstanding,imageMemoryBody,sourceUnderstandingImages} from '../src/memory/image-understanding.ts';
const id='a'.repeat(64);
const receipt={schema_version:1,episode_id:7,attachment_id:id,persisted:false,understanding:{text:'A red rectangle.',attachment_id:id,source:'slide.png',media_type:'image/png',model:'local-vision',width:3,height:2,origin:'model_generated'}};
test('image analysis must match the selected source, attachment and generated origin',()=>{
  assert.equal(parseImageUnderstanding(receipt,7,id).understanding.text,'A red rectangle.');
  for(const invalid of [{...receipt,episode_id:8},{...receipt,persisted:true},{...receipt,attachment_id:'b'.repeat(64)},
    {...receipt,understanding:{...receipt.understanding,origin:'approved'}},{...receipt,understanding:{...receipt.understanding,attachment_id:'b'.repeat(64)}},
    {...receipt,understanding:{...receipt.understanding,text:''}},{...receipt,understanding:{...receipt.understanding,width:0}},
  ])assert.throws(()=>parseImageUnderstanding(invalid,7,id));
});
test('explicit memory save carries source image provenance and a stable request identity',()=>{
  const result=parseImageUnderstanding(receipt,7,id),body=imageMemoryBody(result,'Describe the colors.','request-one');
  assert.deepEqual(body.attachment_ids,[id]);assert.equal(body.metadata.origin,'model_generated');assert.equal(body.metadata.source_episode_id,'7');
  assert.equal(body.metadata.source_attachment_id,id);assert.equal(body.metadata.vision_model,'local-vision');
  assert.match(body.content,/Model-generated image interpretation/);assert.match(body.content,/A red rectangle/);assert.match(body.content,/Describe the colors/);
  assert.equal(body.dedup_key,'image-analysis:request-one');assert.equal('approved' in body,false);
});
test('only valid supported retained images become understanding choices',()=>{
  assert.deepEqual(sourceUnderstandingImages([{attachment_id:id,media_type:'image/png',bytes:40}]),[{attachment_id:id,media_type:'image/png',bytes:40}]);
  assert.deepEqual(sourceUnderstandingImages([{attachment_id:id,media_type:'image/gif',bytes:40}]),[]);
  assert.deepEqual(sourceUnderstandingImages(undefined),[]);
  assert.throws(()=>sourceUnderstandingImages([{attachment_id:'invalid',media_type:'image/png',bytes:40}]));
});
