/* Binary-mask operations shared by the browser editor and deterministic tests. */
(function(root){
  'use strict';
  function pack(mask){const bytes=new Uint8Array(Math.ceil(mask.length/8));for(let i=0;i<mask.length;i++){if(mask[i]!==0&&mask[i]!==1)throw Error('Mask is not binary');bytes[i>>3]|=mask[i]<<(i&7);}let raw='';for(const byte of bytes)raw+=String.fromCharCode(byte);return btoa(raw);}
  function unpack(text,count){const raw=atob(text);if(raw.length!==Math.ceil(count/8))throw Error('Wrong mask dimensions');const out=new Uint8Array(count);for(let i=0;i<count;i++)out[i]=(raw.charCodeAt(i>>3)>>(i&7))&1;return out;}
  function stamp(mask,width,height,x,y,diameter,value){const radius=Math.max(.5,diameter/2);x=Math.round(x);y=Math.round(y);for(let yy=Math.max(0,Math.ceil(y-radius));yy<=Math.min(height-1,Math.floor(y+radius));yy++)for(let xx=Math.max(0,Math.ceil(x-radius));xx<=Math.min(width-1,Math.floor(x+radius));xx++)if((xx-x)**2+(yy-y)**2<=radius**2)mask[yy*width+xx]=value;}
  function line(mask,width,height,a,b,diameter,value){const steps=Math.max(1,Math.ceil(Math.max(Math.abs(a[0]-b[0]),Math.abs(a[1]-b[1]))));for(let i=0;i<=steps;i++)stamp(mask,width,height,a[0]+(b[0]-a[0])*i/steps,a[1]+(b[1]-a[1])*i/steps,diameter,value);}
  function polygon(mask,width,height,points,value){if(points.length<3)return;const xs=points.map(p=>p[0]),ys=points.map(p=>p[1]);for(let y=Math.max(0,Math.floor(Math.min(...ys)));y<=Math.min(height-1,Math.ceil(Math.max(...ys)));y++)for(let x=Math.max(0,Math.floor(Math.min(...xs)));x<=Math.min(width-1,Math.ceil(Math.max(...xs)));x++){let inside=false;const px=x+.5,py=y+.5;for(let i=0,j=points.length-1;i<points.length;j=i++){const a=points[i],b=points[j];if((a[1]>py)!==(b[1]>py)&&px<(b[0]-a[0])*(py-a[1])/(b[1]-a[1])+a[0])inside=!inside;}if(inside)mask[y*width+x]=value;}}
  function clone(masks){return {building:masks.building.slice(),valid:masks.valid.slice()};}
  class History{constructor(limit=60){this.limit=limit;this.past=[];this.future=[];}push(masks){this.past.push(clone(masks));if(this.past.length>this.limit)this.past.shift();this.future=[];}undo(masks){if(!this.past.length)return null;this.future.push(clone(masks));return this.past.pop();}redo(masks){if(!this.future.length)return null;this.past.push(clone(masks));return this.future.pop();}}
  function scoringCoverage(masks,originals){
    if(masks.building.length!==masks.valid.length||masks.building.length!==originals.building.length)throw Error('Wrong mask dimensions');
    let added=0,scored=0;
    for(let i=0;i<masks.building.length;i++)if(masks.building[i]&&!originals.building[i]){added++;if(masks.valid[i])scored++;}
    return {added,scored,excluded:added-scored};
  }
  const api={pack,unpack,stamp,line,polygon,clone,History,scoringCoverage};if(typeof module!=='undefined'&&module.exports)module.exports=api;else root.MaskCore=api;
})(typeof window!=='undefined'?window:globalThis);
