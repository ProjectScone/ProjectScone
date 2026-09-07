import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readTextSource} from '../src/memory/text-import.ts';

test('text imports preserve Unicode, BOM, line endings and code as literal source text',async()=>{
  const content='\ufeff# Juniper\r\n\r\nconst label = "星";\r\n<script>neverExecute()</script>\n';
  const file=new File([content],'calibration.MD',{type:'application/octet-stream'});
  const source=await readTextSource(file);
  assert.deepEqual(source,{name:'calibration.MD',content,bytes:new TextEncoder().encode(content).length});
});

test('text imports reject unsupported, invalid UTF-8, binary and empty content',async()=>{
  for(const [name,bytes] of [
    ['scan.pdf',Buffer.from('%PDF-1.7')], ['note.txt',Buffer.from([0xc3,0x28])],
    ['file.rs',Buffer.from('text\0binary')], ['empty.txt',Buffer.alloc(0)],
    ['blank.md',Buffer.from(' \r\n\t')],
  ] as const){
    await assert.rejects(readTextSource(new File([bytes],name)));
  }
});

test('oversized imports fail before reading file contents',async()=>{
  let read=false;
  await assert.rejects(readTextSource({name:'large.txt',size:1_048_577,arrayBuffer:async()=>{read=true;return new ArrayBuffer(0);}}));
  assert.equal(read,false);
});
