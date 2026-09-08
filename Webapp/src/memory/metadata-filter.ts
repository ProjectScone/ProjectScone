export const comparisons=[['is','Equals text'],['has','Contains text'],['in','One of these values'],['above','Greater than'],['at_least','At least'],['below','Less than'],['at_most','At most'],['present','Key is present']] as const;
export type FilterOperator=typeof comparisons[number][0];
export interface FilterRule {kind:'rule';field:string;operator:FilterOperator;value:string;negated:boolean}
export interface FilterGroup {kind:'group';mode:'all'|'any';children:FilterNode[]}
export type FilterNode=FilterRule|FilterGroup;
export interface AppliedFilter {draft:FilterGroup;json:string}
type WireRule={field:string;not?:true;is?:string;has?:string;in?:string[];present?:true;above?:number;below?:number;at_least?:number;at_most?:number};
type WireNode=WireRule|{all:WireNode[]}|{any:WireNode[]};
export const blankRule=():FilterRule=>({kind:'rule',field:'',operator:'is',value:'',negated:false});
export const blankGroup=():FilterGroup=>({kind:'group',mode:'all',children:[blankRule()]});

export function compileFilter(draft:FilterGroup):string {
  let count=0;
  function visit(node:FilterNode,depth:number):WireNode {
    if(node.kind==='group'){
      if(depth>8)throw Error('Use at most 8 nested filter groups.');
      if(!node.children.length)throw Error('Each group needs a rule. Remove an empty group or add a rule.');
      const parts=node.children.map(child=>visit(child,depth+1));
      return node.mode==='all'?{all:parts}:{any:parts};
    }
    if(++count>200)throw Error('Use at most 200 metadata rules.');
    const field=node.field.trim();
    if(!/^[a-z][a-z0-9_]{0,31}$/.test(field))throw Error('Metadata keys start with a lowercase letter and use up to 32 letters, numbers or underscores.');
    const result:WireRule={field,...(node.negated?{not:true}:{})};
    switch(node.operator){
      case 'present':result.present=true;break;
      case 'is':case 'has':result[node.operator]=node.value;break;
      case 'in':{
        const values=node.value.split('\n');
        if(values.some(value=>!value.length))throw Error('Enter one value per line, with no empty lines.');
        result.in=values;break;
      }
      default:{
        const value=node.value.trim();
        if(!/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$/i.test(value)||!Number.isFinite(Number(value)))throw Error('Numeric comparisons need a finite number, such as 10 or 2.5.');
        result[node.operator]=Number(value);
      }
    }
    return result;
  }
  const json=JSON.stringify(visit(draft,1));
  if(Array.from(json).length>8000)throw Error('This filter exceeds the server’s 8,000-character limit. Shorten its values or remove rules.');
  return json;
}
