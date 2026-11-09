/* Mano3V: three-view camera video (head camera on top, wrist cameras below) of one take segment with the fitted hand,
   switches for the skeleton (drawn on a canvas, frame-synced) and the mesh (base video with / without the mesh).
   Mano3V.mount(element, baseURL, segmentLabel, {key:"best", color:"#2ee06a", controls:true})  - data made by skel_compare.py.
   Returns a controller {video, duration, seek(t), play(), pause(), time(), onTime(fn), setControls(on), seekReal(s), realTime(), realDuration}
   (seekReal / realTime use the frames' real times: s = seconds of real time from the segment start): with controls:false the player shows only
   the skeleton / mesh switches, and a shared bar on the page (e.g. the 3D viewer's) drives it through seek/play/pause. */
(function(){
const metaCache={};
function getMeta(base){ if(!metaCache[base]) metaCache[base]=fetch(base+"meta.json").then(r=>r.json()); return metaCache[base]; }
async function mount(el, base, label, opt){
  opt=Object.assign({key:"best", color:"#2ee06a", controls:true}, opt||{});
  const M=await getMeta(base); const ep=M.episodes.find(e=>e.label===label); if(!ep){el.textContent="no segment "+label;return;}
  el.innerHTML=`<div style="position:relative;width:100%;background:#000;border-radius:6px;overflow:hidden">
    <video playsinline muted preload=auto style="width:100%;display:block"></video>
    <canvas style="position:absolute;left:0;top:0;width:100%;height:100%;pointer-events:none"></canvas></div>
    <div style="display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin:6px 0;font:13px system-ui,sans-serif">
    <span data-a=own style="display:${opt.controls?"contents":"none"}"><button data-a=pp style="font:inherit;padding:2px 10px">Play</button>
    <button data-a=pv style="font:inherit;padding:2px 8px" title="previous frame">◀</button><button data-a=nx style="font:inherit;padding:2px 8px" title="next frame">▶</button></span>
    <label style="cursor:pointer"><input type=checkbox data-a=sk checked> Skeleton</label>
    <label style="cursor:pointer"><input type=checkbox data-a=me checked> Mesh</label>
    <span data-a=tm style="opacity:.7;font-variant-numeric:tabular-nums"></span></div>
    <input type=range data-a=sb min=0 max=${ep.n-1} value=0 step=1 style="width:100%;display:${opt.controls?"block":"none"}">`;
  const q=a=>el.querySelector(`[data-a=${a}]`), v=el.querySelector("video"), cv=el.querySelector("canvas"), cx=cv.getContext("2d");
  const A=new Int16Array(await (await fetch(base+label+"_"+opt.key+".bin")).arrayBuffer()), FPS=M.fps||30;
  const src=m=>base+label+"_mesh_"+(m?opt.key:"none")+".mp4", listeners=[];
  const idx=t=>Math.max(0,Math.min(ep.n-1,Math.round(t*FPS)));
  function draw(t){ const r=v.getBoundingClientRect(), d=window.devicePixelRatio||1;
    if(cv.width!==Math.round(r.width*d)){cv.width=Math.round(r.width*d);cv.height=Math.round(r.height*d)}
    cx.clearRect(0,0,cv.width,cv.height); const i=idx(t); q("tm").textContent=`${(t).toFixed(2)} s / ${(ep.n/FPS).toFixed(1)} s`; q("sb").value=i; listeners.forEach(f=>f(t));
    if(!q("sk").checked) return; const s=cv.width/M.W; cx.strokeStyle=opt.color; cx.fillStyle=opt.color; cx.lineWidth=2*d; cx.lineCap="round";
    for(let vi=0;vi<3;vi++){ const p=M.panels[vi]; cx.save(); cx.beginPath(); cx.rect(p[0]*s,p[1]*s,p[2]*s,p[3]*s); cx.clip();
      for(let h=0;h<2;h++){ const o=(((i*3+vi)*2+h)*21)*2, X=k=>A[o+2*k]*s, Y=k=>A[o+2*k+1]*s, ok=k=>A[o+2*k]!==-32768;
        cx.beginPath(); for(const [a,b] of M.bones) if(ok(a)&&ok(b)){cx.moveTo(X(a),Y(a));cx.lineTo(X(b),Y(b))} cx.stroke();
        for(let k=0;k<21;k++) if(ok(k)){cx.beginPath();cx.arc(X(k),Y(k),2.2*d,0,6.3);cx.fill()} } cx.restore(); } }
  function loop(n,m){draw(m.mediaTime); v.requestVideoFrameCallback(loop)}
  v.src=src(true); v.onloadeddata=()=>draw(v.currentTime); v.onseeked=()=>draw(v.currentTime);
  if(v.requestVideoFrameCallback) v.requestVideoFrameCallback(loop); else v.ontimeupdate=()=>draw(v.currentTime);
  v.onplay=()=>q("pp").textContent="Pause"; v.onpause=()=>q("pp").textContent="Play";
  q("pp").onclick=()=>v.paused?v.play():v.pause();
  const step=k=>{v.pause(); v.currentTime=Math.max(0,(idx(v.currentTime)+k+0.5)/FPS)}; q("pv").onclick=()=>step(-1); q("nx").onclick=()=>step(1);
  q("sb").oninput=e=>{v.pause(); v.currentTime=(+e.target.value+0.5)/FPS};
  q("sk").onchange=()=>draw(v.currentTime);
  q("me").onchange=e=>{const t=v.currentTime, play=!v.paused; v.src=src(e.target.checked); v.addEventListener("loadeddata",()=>{v.currentTime=t; if(play) v.play()},{once:true})};
  window.addEventListener("resize",()=>draw(v.currentTime));
  const setControls=on=>{q("own").style.display=on?"contents":"none"; q("sb").style.display=on?"block":"none"};
  const ft=ep.t_s||null;                       // real time of each video frame from the segment start (the camera dropped some frames)
  const frameAt=s=>{ if(!ft) return idx(s); let lo=0, hi=ft.length-1; while(lo<hi){const m=(lo+hi+1)>>1; if(ft[m]<=s+1e-4) lo=m; else hi=m-1} return lo; };
  return {video:v, duration:ep.n/FPS, seek:t=>{v.currentTime=Math.max(0,Math.min(ep.n/FPS-1e-3,t))}, play:()=>v.play(), pause:()=>v.pause(),
          time:()=>v.currentTime, onTime:f=>listeners.push(f), setControls,
          seekReal:s=>{v.currentTime=(frameAt(s)+0.5)/FPS}, realTime:()=>ft?ft[idx(v.currentTime)]:v.currentTime,
          realDuration:ft?ft[ft.length-1]+1/FPS:ep.n/FPS};
}
window.Mano3V={mount};
})();
