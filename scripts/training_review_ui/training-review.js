/* Add context decisions without changing the established drawing implementation. */
(()=>{'use strict';
  const $=id=>document.getElementById(id),originalFetch=window.fetch.bind(window);
  const token=document.querySelector('meta[name="editor-token"]').content;
  const query=new URLSearchParams(location.search),view=query.get('view');
  const suffix=view?'?view='+encodeURIComponent(view):'';
  const fields=['pairing','support','mask_review','reviewer','evidence'];
  let state=null,review=null,saved='',queue=Promise.resolve(),timer,blocked=false;
  const values=()=>Object.fromEntries(fields.map(k=>[k,$('scene-'+k).value]));
  function status(message,error=false){$('scene-status').textContent=message;$('scene-status').classList.toggle('error',error);}
  async function flush(){
    clearTimeout(timer);if(!review)throw new Error('Wait for scene notes to load.');
    const current=values();if(JSON.stringify(current)===saved)return;
    if(blocked)throw new Error('Scene notes have a conflict. Reload before saving.');
    const response=await originalFetch('/api/review'+suffix,{method:'POST',headers:{'Content-Type':'application/json','X-Mask-Editor-Token':token},body:JSON.stringify({...current,sample_id:state.sample_id,source_packet_sha256:state.source_packet_sha256,expected_revision:review.revision})});
    const result=await response.json();if(!response.ok){if(response.status===409)blocked=true;throw new Error(result.error||'Could not save scene notes.');}
    review=result.review;saved=JSON.stringify(current);status('Scene notes saved locally.');
    // Editing scene evidence invalidates a previously finished review.
    if(!result.batch.all_complete)$('batch-finished').hidden=true;
    $('batch-progress').textContent=`${result.batch.ready+result.batch.accepted} / ${result.batch.total} finished - ${result.batch.remaining} left`;
  }
  function saveNotes(){queue=queue.catch(()=>{}).then(flush);queue.catch(error=>status(error.message,true));return queue;}
  window.fetch=async function(input,options){
    const url=typeof input==='string'?input:input.url;
    if(options?.method==='POST'&&/^\/api\/(draft|version)(\?|$)/.test(url))await saveNotes();
    return originalFetch(input,options);
  };
  for(const key of fields)$('scene-'+key).addEventListener('input',()=>{status('Scene notes not saved yet.');clearTimeout(timer);timer=setTimeout(()=>saveNotes(),350);});
  $('save-scene').addEventListener('click',()=>saveNotes());
  // Flush notes even when no mask was changed before Previous/Next/image selection.
  for(const id of ['previous-image','next-image','image-select','save-next']){
    const element=$(id),event=id==='image-select'?'change':'click';let replay=false;
    element.addEventListener(event,async e=>{if(replay)return;e.stopImmediatePropagation();e.preventDefault();try{
      const v=values();
      if(id==='save-next'&&(['pairing','support','mask_review'].some(k=>v[k]==='pending')||!v.reviewer.trim()||v.evidence.trim().length<10)){
        status('Choose all three answers, enter your name and add a short evidence note. Uncertain answers are allowed.',true);$('scene-fields').scrollIntoView({behavior:'smooth',block:'center'});return;
      }
      await saveNotes();replay=true;element.dispatchEvent(new Event(event,{bubbles:true}));
    }catch(error){status(error.message,true);}finally{replay=false;}},true);
  }
  window.addEventListener('beforeunload',e=>{if(review&&JSON.stringify(values())!==saved){e.preventDefault();e.returnValue='';}});
  async function load(){try{
    const response=await originalFetch('/api/state'+suffix,{headers:{'X-Mask-Editor-Token':token}});if(!response.ok)throw new Error('Could not load training review.');
    state=await response.json();review=state.training_review;
    for(const key of fields)$('scene-'+key).value=review[key];saved=JSON.stringify(values());
    const actual='?view='+state.view_number;
    $('training-context-image').src='/training-context'+actual;
    $('context-link').href=$('context-preview-link').href='/training-context'+actual;
    $('satellite-link').href='/training-satellite'+actual;
    $('scene-fields').disabled=false;status(review.saved_utc?'Saved scene notes restored.':'Choose what you can verify; uncertain answers are allowed.');
    $('page-title').textContent='Training review · view '+state.view_number;document.title='SatGround · Training review '+state.view_number;
  }catch(error){status(error.message,true);}}
  load();
})();
