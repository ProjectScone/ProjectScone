export interface TextSource {name:string;content:string;bytes:number}
export const TEXT_FILE_ACCEPT='.txt,.md,.markdown,.json,.csv,.log,.ts,.tsx,.js,.jsx,.py,.rs,.html,.css,.yaml,.yml,.toml,.xml,.sh,.sql';
const extensions=new Set(TEXT_FILE_ACCEPT.split(','));

/** Read one explicitly selected source. No network, parser or code execution. */
export async function readTextSource(file:Pick<File,'name'|'size'|'arrayBuffer'>):Promise<TextSource>{
  const extension=file.name.slice(file.name.lastIndexOf('.')).toLowerCase();
  if(!extensions.has(extension))throw Error('Choose a supported text, Markdown or code file. PDF and binary formats are not supported here.');
  if(!file.size||file.size>1_048_576)throw Error('Choose a nonempty text file up to 1 MB.');
  const bytes=await file.arrayBuffer();
  if(bytes.byteLength!==file.size)throw Error('The selected file changed while reading. Select it again.');
  let content:string;
  try{content=new TextDecoder('utf-8',{fatal:true,ignoreBOM:true}).decode(bytes);}
  catch{throw Error('This file is not valid UTF-8 text. Convert its encoding before importing.');}
  if(content.includes('\0'))throw Error('This file contains binary data, not plain text.');
  if(!content.trim())throw Error('The selected file contains no searchable text.');
  return {name:file.name,content,bytes:file.size};
}
