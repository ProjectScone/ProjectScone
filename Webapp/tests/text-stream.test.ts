import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readTextStream} from '../src/conversations/text-stream.ts';

function bytes(text:string,step=1){
  const encoded=new TextEncoder().encode(text);
  return new ReadableStream<Uint8Array>({start(controller){for(let i=0;i<encoded.length;i+=step)controller.enqueue(encoded.slice(i,i+step));controller.close();}});
}
async function collect(text:string,after=0){const events=[];for await(const event of readTextStream(bytes(text,text.length>100000?4096:1),'turn',after))events.push(event);return events;}
const chunk=(sequence:number,text:string)=>`event: text\nid: ${sequence}\ndata: ${JSON.stringify({sequence,text,provisional:true})}\n\n`;
const terminal='event: terminal\ndata: {"request_id":"turn","status":"completed","read_receipt":true}\n\n';

test('incremental SSE preserves literal Unicode across byte splits and ignores comments',async()=>{
  const events=await collect(': keep-alive\r\n\r\n'+chunk(1,'Hello 🌿\n<script>x</script>').replaceAll('\n','\r\n')+terminal);
  assert.deepEqual(events,[{kind:'text',sequence:1,text:'Hello 🌿\n<script>x</script>'},{kind:'terminal',status:'completed'}]);
});
test('a resumed stream skips acknowledged chunks, but only explicit gaps allow missing sequences',async()=>{
  assert.deepEqual(await collect(chunk(7,'old')+chunk(8,'new')+terminal,7),[{kind:'text',sequence:8,text:'new'},{kind:'terminal',status:'completed'}]);
  assert.deepEqual(await collect('event: gap\ndata: {"after":0,"next_sequence":5}\n\n'+chunk(5,'tail')+terminal),[{kind:'gap',next:5},{kind:'text',sequence:5,text:'tail'},{kind:'terminal',status:'completed'}]);
  await assert.rejects(collect(chunk(2,'missing prefix')+terminal),/sequence/i);
});
test('terminal-only and service end are readable without inventing text',async()=>{
  assert.deepEqual(await collect(terminal),[{kind:'terminal',status:'completed'}]);
  assert.deepEqual(await collect('event: end\ndata: {"request_id":"turn","reason":"service_shutdown","read_receipt":true}\n\n'),[{kind:'end',reason:'service_shutdown'}]);
});
test('malformed, contradictory or unbounded events fail closed and premature EOF is not completion',async()=>{
  for(const wire of [chunk(1,'hi').replace('id: 1','id: 2'),chunk(1,'hi').replace('true','false'),
    terminal.replace('"turn"','"other"'),terminal.replace('completed','pending'),
    'event: gap\ndata: {"after":8,"next_sequence":10}\n\n',
    'event: thought\ndata: {"text":"private"}\n\n',
    'event: text\ndata: '+ 'x'.repeat(524289)+'\n\n',
    chunk(1,'partial'),terminal.trim(),': keep-alive\n\n'])await assert.rejects(collect(wire));
});
test('stopping consumption cancels the owned reader instead of leaving the HTTP stream open',async()=>{
  let cancelled=false;
  const stream=new ReadableStream<Uint8Array>({start(controller){controller.enqueue(new TextEncoder().encode(chunk(1,'first')));},cancel(){cancelled=true;}});
  for await(const event of readTextStream(stream,'turn',0)){assert.equal(event.kind,'text');break;}
  assert.equal(cancelled,true);
});
