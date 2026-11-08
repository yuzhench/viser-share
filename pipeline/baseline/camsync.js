// Camera sync for the two viser playback iframes (#viserA, #viserB, same origin): rotate / pan / zoom one view and the other
// follows with the same camera. Both scenes are in the same upright station frame, so the same camera pose shows the same
// place. The view the user last touched leads (camera-controls 'controlstart' / 'control'); on each of its 'update's (also
// during the damping after release) its current position + target are copied to the other with setLookAt (no transition).
// Re-attaches by itself when an iframe reloads (part switch). Uses window.__viserMutable.cameraControl of each client.
(function () {
  const ids = ['viserA', 'viserB'];
  const cc = [null, null];
  let leader = -1;
  function copy(from) {
    const a = cc[from], b = cc[1 - from];
    if (!a || !b) return;
    const p = a.camera.position.clone(), t = a.camera.position.clone();
    a.getPosition(p, false); a.getTarget(t, false);
    b.setLookAt(p.x, p.y, p.z, t.x, t.y, t.z, false);
  }
  function attach(i) {
    let c = null;
    try { const w = document.getElementById(ids[i]).contentWindow; c = w && w.__viserMutable && w.__viserMutable.cameraControl; } catch (e) { c = null; }
    if (!c || c === cc[i]) return;
    cc[i] = c;
    const lead = () => { leader = i; };
    c.addEventListener('controlstart', lead);
    c.addEventListener('control', lead);
    c.addEventListener('update', () => { if (leader === i) copy(i); });
    if (leader === 1 - i) copy(1 - i);                          // a view that (re)loads late takes the leader's camera
  }
  setInterval(() => { attach(0); attach(1); }, 400);
  window.__viserCamSync = { cc, copy, leader: () => leader, reset: () => { leader = -1; } };
})();
