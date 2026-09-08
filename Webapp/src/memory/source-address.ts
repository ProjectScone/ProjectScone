export function sourceAddress(episodeId:number,space:string):string {
  if(!Number.isSafeInteger(episodeId)||episodeId<=0||!space)throw Error('Invalid source address');
  return `/memory/sources/${episodeId}?${new URLSearchParams({space})}`;
}
export function readSourceAddress(id:string|undefined,search:string):{episodeId:number;space:string}|null {
  const params=new URLSearchParams(search),spaces=params.getAll('space');
  if(!id||! /^[1-9]\d*$/.test(id)||!Number.isSafeInteger(Number(id))||spaces.length!==1||!spaces[0])return null;
  return {episodeId:Number(id),space:spaces[0]};
}
export function verifiedSpace(value:unknown):string {
  if(!value||typeof value!=='object'||!('space' in value)||typeof value.space!=='string'||!value.space)throw Error('The server did not identify this memory space.');
  return value.space;
}
