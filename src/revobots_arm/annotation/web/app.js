'use strict';
const $ = id => document.getElementById(id);
const token = new URLSearchParams(location.hash.slice(1)).get('token') || '';
let project, source, frame = 0, draft = null, playing = false, playTime = 0, playFrame = 0;
let pendingSaves = 0, pendingEdit = null, editTimer, previewBusy = false, previewQueued = false;
let history = [], saveQueue = Promise.resolve(), saveFailed = false, requestId = 0, refreshTimer;
const clone = value => JSON.parse(JSON.stringify(value));
const escapeHTML = text => String(text).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const timecode = seconds => `${String(Math.floor(seconds/60)).padStart(2,'0')}:${(seconds%60).toFixed(3).padStart(6,'0')}`;
const notice = text => { $('notice').textContent = text; };
async function api(path, data) {
  const response = await fetch(path, {method: data === undefined ? 'GET' : 'POST', headers: {'X-Annotation-Token': token, 'Content-Type':'application/json'}, ...(data === undefined ? {} : {body:JSON.stringify(data)})});
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || 'Request failed');
  return result;
}
function setFrame(value) {
  frame = Math.max(0, Math.min(source.frames-1, Math.round(Number(value)||0)));
  $('scrubber').value = frame; $('goto').value = frame;
  $('timecode').textContent = timecode(frame/source.fps);
  $('frame-label').textContent = `FRAME ${frame} / ${source.frames-1}`;
  requestId++; $('mark').disabled=source.cameras.length>0; $('frame-status').textContent=source.cameras.length?`Loading frame ${frame}…`:'';
  renderSegments(); clearTimeout(refreshTimer); refreshTimer = setTimeout(refreshFrames, playing ? 0 : 45);
}
async function refreshFrames() {
  if(previewBusy){previewQueued=true;requestId++;return;}
  previewBusy=true;
  try {await loadFrames();}
  finally {previewBusy=false;if(previewQueued){previewQueued=false;refreshFrames();}}
}
async function loadFrames() {
  const generation = ++requestId, index = frame, sourceId = source.id;
  const cameras = $('all-cameras').checked ? source.cameras : [$('camera').value].filter(Boolean);
  if (!cameras.length) { $('frames').replaceChildren(); $('frame-status').textContent = 'This recording has no camera streams. Frame marking and export are still available.'; $('mark').disabled=false; return; }
  const target = $('frames'); target.classList.toggle('multi', cameras.length > 1);
  try {
    const blobs = await Promise.all(cameras.map(async camera => {
      const query = new URLSearchParams({source:sourceId, camera, frame:index});
      const response = await fetch(`/api/frame?${query}`, {headers:{'X-Annotation-Token':token}});
      if (!response.ok) throw new Error((await response.json()).error || 'Preview failed');
      return {camera, blob:await response.blob()};
    }));
    if (generation !== requestId || index !== frame || sourceId !== source.id) return;
    const previousURLs = [...target.querySelectorAll('img')].map(img => img.src);
    target.replaceChildren(...blobs.map(({camera, blob}) => {
      const card = document.createElement('div'); card.className='camera-frame';
      const image = document.createElement('img'); image.alt=`${camera}, frame ${index}`;
      image.src=URL.createObjectURL(blob); const label=document.createElement('span'); label.textContent=camera;
      card.append(image,label); return card;
    }));
    previousURLs.forEach(url=>URL.revokeObjectURL(url)); $('frame-status').textContent=''; $('mark').disabled=false;
  } catch(error) { if(generation===requestId) { $('frame-status').textContent=error.message; stopPlayback(); } }
}
function selectSource(id) {
  flushEdit(); stopPlayback(); source=project.sources.find(s=>s.id===id); draft=null;
  $('camera').replaceChildren(...source.cameras.map(camera=>new Option(camera,camera)));
  $('scrubber').max=source.frames-1; $('goto').max=source.frames-1;
  $('duration').textContent=timecode(source.duration); $('task').value=source.task||'';
  updateMark(); setFrame(0);
}
function renderSegments() {
  if(!source) return;
  const ranges=project.episodes.filter(e=>e.source===source.id);
  $('segments').innerHTML=ranges.map(e=>`<div class="segment" title="${escapeHTML(e.task||'Episode')} · ${e.start}–${e.end-1}" style="left:${e.start/source.frames*100}%;width:${(e.end-e.start)/source.frames*100}%"></div>`).join('') + (draft!==null?`<div class="segment draft" style="left:${Math.min(draft,frame)/source.frames*100}%;width:${(Math.abs(frame-draft)+1)/source.frames*100}%"></div>`:'');
}
function updateMark() {
  const active=draft!==null; $('mark').classList.toggle('active',active); $('cancel').hidden=!active;
  $('mark').innerHTML=active?'✓ Mark end <kbd>Space</kbd>':'＋ Mark start <kbd>Space</kbd>';
  $('mark-state').textContent=active?`START FRAME ${draft} · ${timecode(draft/source.fps)}`:'SELECT AN EPISODE';
  $('mark-description').textContent=active?'Move to the last included frame, then mark the end.':'Set a start, then move to the last frame you want to keep.';
}
function validate(episodes) {
  for(const s of project.sources) {
    const sorted=episodes.filter(e=>e.source===s.id).sort((a,b)=>a.start-b.start);
    for(let i=0;i<sorted.length;i++) {
      const e=sorted[i]; if(!Number.isInteger(e.start)||!Number.isInteger(e.end)||e.start<0||e.start>=e.end||e.end>s.frames) throw new Error('Frame boundaries are outside the recording.');
      if(i&&sorted[i-1].end>e.start) throw new Error('Episodes cannot overlap. Adjust the start or end.');
    }
  }
}
function commit(episodes, remember=true) {
  try {validate(episodes);} catch(error) {notice(error.message);renderEpisodes();return false;}
  if(remember) history.push(clone(project.episodes));
  project.episodes=clone(episodes); renderEpisodes(); notice('');
  const snapshot=clone(episodes); pendingSaves++; $('save-status').textContent='Saving annotations…';
  saveQueue=saveQueue.then(async()=>{
    try {
      const saved=await api('/api/annotations',{revision:project.revision,episodes:snapshot});
      project.revision=saved.revision; saveFailed=false; $('save-status').textContent='● All changes saved locally';
    } catch(error) {saveFailed=true; notice(`${error.message} — reload if this recording is open in another tab.`);}
    finally {pendingSaves--; $('save-status').textContent=saveFailed?'Unsaved changes':pendingSaves?'Saving annotations…':'● All changes saved locally';}
  });
  return true;
}
function mark() {
  stopPlayback();
  if($('mark').disabled){notice('Wait for the current preview frame to load before marking.');return;}
  if(draft===null) {draft=frame;updateMark();renderSegments();return;}
  if(frame<draft) {notice('Move to a frame at or after the marked start.');return;}
  const episode={id:crypto.randomUUID(),source:source.id,start:draft,end:frame+1,task:$('task').value.trim()};
  if(commit([...project.episodes,episode])) {draft=null;updateMark();renderSegments();}
}
function renderEpisodes() {
  $('episode-count').textContent=project.episodes.length; $('undo').disabled=!history.length;
  $('episodes').innerHTML=project.episodes.length ? project.episodes.map((e,i)=>{
    const s=project.sources.find(s=>s.id===e.source);
    return `<article class="episode" data-id="${escapeHTML(e.id)}"><div class="episode-header"><strong>Episode ${String(i+1).padStart(2,'0')}</strong><button class="quiet jump" title="Jump to episode">↗</button><button class="quiet delete" title="Delete episode">×</button></div><div class="bounds">Frames <input aria-label="Episode ${i+1} start" class="start" type="number" min="0" max="${s.frames-1}" value="${e.start}"> → <input aria-label="Episode ${i+1} last included frame" class="end" type="number" min="0" max="${s.frames-1}" value="${e.end-1}"></div><small>${escapeHTML(s.name)} · ${((e.end-e.start)/s.fps).toFixed(2)}s · ${e.end-e.start} frames</small><input class="task" aria-label="Episode ${i+1} task" placeholder="Task description" value="${escapeHTML(e.task)}"></article>`;
  }).join('') : '<div class="empty">Your first episode starts here.<br><small>Press Space to mark its beginning.</small></div>';
  renderSegments();
}
function stopPlayback(){playing=false;$('play').textContent='▶ Play';}
function togglePlayback(){if(playing)return stopPlayback();if(frame>=source.frames-1)setFrame(0);playing=true;playTime=performance.now();playFrame=frame;$('play').textContent='Ⅱ Pause';}
setInterval(()=>{if(playing){const next=playFrame+Math.floor((performance.now()-playTime)/1000*source.fps);if(next!==frame)setFrame(next);if(next>=source.frames-1)stopPlayback();}},100);
async function exportSelection(convert){
  stopPlayback(); if(!flushEdit())return; if(draft!==null){notice('Finish or cancel the current marked range before exporting.');return;} await saveQueue;
  if(saveFailed){notice('Annotations have unsaved changes. Resolve the save error before exporting.');return;}
  try {await api('/api/export',{revision:project.revision,convert,repo_id:$('repo-id').value.trim()}); await pollJob();}catch(error){notice(error.message);}
}
async function pollJob(){
  const job=await api('/api/job'); $('job').textContent=job.message; $('job').classList.toggle('error',job.state==='error');
  $('export').disabled=$('convert').disabled=job.state==='running';
  if(job.state==='running')setTimeout(()=>pollJob().catch(e=>notice(e.message)),700);
}
$('source').onchange=()=>selectSource($('source').value);
$('camera').onchange=refreshFrames; $('all-cameras').onchange=refreshFrames;
$('scrubber').oninput=()=>{stopPlayback();setFrame($('scrubber').value);};
$('goto').onchange=()=>{stopPlayback();setFrame($('goto').value);};
$('previous').onclick=()=>{stopPlayback();setFrame(frame-1);}; $('next').onclick=()=>{stopPlayback();setFrame(frame+1);};
$('play').onclick=togglePlayback; $('mark').onclick=mark;
$('cancel').onclick=()=>{draft=null;updateMark();renderSegments();};
$('undo').onclick=()=>{clearTimeout(editTimer);pendingEdit=null;if(history.length)commit(history.pop(),false);};
$('episodes').onclick=event=>{
  const card=event.target.closest('.episode');if(!card)return;
  const episode=project.episodes.find(e=>e.id===card.dataset.id);
  if(event.target.closest('.delete'))commit(project.episodes.filter(e=>e.id!==episode.id));
  if(event.target.closest('.jump')){ $('source').value=episode.source;selectSource(episode.source);setFrame(episode.start); }
};
function flushEdit(){
  clearTimeout(editTimer); if(!pendingEdit)return true;
  const edit=pendingEdit; pendingEdit=null;
  if(edit.start.trim()===''||edit.end.trim()===''){notice('Both frame boundaries are required.');renderEpisodes();return false;}
  const episodes=clone(project.episodes),episode=episodes.find(e=>e.id===edit.id);
  if(!episode)return true;
  episode.start=Number(edit.start);episode.end=Number(edit.end)+1;episode.task=edit.task;
  return commit(episodes);
}
$('episodes').oninput=event=>{
  const card=event.target.closest('.episode');if(!card)return;
  pendingEdit={id:card.dataset.id,start:card.querySelector('.start').value,end:card.querySelector('.end').value,task:card.querySelector('.task').value};
  clearTimeout(editTimer);editTimer=setTimeout(flushEdit,500);
};
$('export').onclick=()=>exportSelection(false); $('convert').onclick=()=>exportSelection(true);
document.addEventListener('keydown',event=>{
  if(!source||(/INPUT|TEXTAREA|SELECT/.test(document.activeElement.tagName)&&document.activeElement.type!=='range')||event.ctrlKey||event.metaKey||event.altKey)return;
  if(event.repeat&&event.code==='Space')return;
  if(event.code==='Space'){event.preventDefault();mark();}
  else if(event.code==='ArrowLeft'||event.code==='ArrowRight'){event.preventDefault();stopPlayback();setFrame(frame+(event.code==='ArrowLeft'?-1:1)*(event.shiftKey?Math.round(source.fps):1));}
  else if(event.code==='KeyK'){event.preventDefault();togglePlayback();}
  else if(event.code==='Escape')$('cancel').click();
});
window.addEventListener('beforeunload',event=>{if(saveFailed||pendingSaves>0||pendingEdit!==null){event.preventDefault();event.returnValue='';}});
async function init(){
  try {project=await api('/api/project');$('source').replaceChildren(...project.sources.map(s=>new Option(`${s.name} · ${s.frames.toLocaleString()} frames`,s.id)));$('source-count').textContent=`${project.sources.length} RECORDING${project.sources.length>1?'S':''}`;$('source-path').textContent=project.path;$('source-path').title=project.path;selectSource(project.sources[0].id);renderEpisodes();$('save-status').textContent='● All changes saved locally';await pollJob();}
  catch(error){notice(error.message);$('save-status').textContent='Unable to load';$('frame-status').textContent=error.message;}
}
init();
