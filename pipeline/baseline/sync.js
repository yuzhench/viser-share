// Keeps the two viser playback iframes (same origin) on one timeline: a shared play / pause / seek bar, and whichever viewer
// was touched last leads (its own bar works too); the 2D video below follows the same time. Reads each viser client's Mantine
// slider (aria-valuenow / aria-valuemax, seconds) and its play / pause icon; seeks by a mouse press on the slider track.
(function () {
  const ids = ['viserA', 'viserB'];
  let master = 0, dragging = false, lastSeek = [0, 0];
  const $ = (id) => document.getElementById(id);
  function ctl(i) {
    const f = $(ids[i]); let d; try { d = f.contentDocument; } catch (e) { return null; }
    if (!d) return null;
    const th = d.querySelector('.mantine-Slider-thumb[role="slider"]'); if (!th) return null;
    const max = parseFloat(th.getAttribute('aria-valuemax')), t = parseFloat(th.getAttribute('aria-valuenow'));
    if (!(max > 0) || !Number.isFinite(t)) return null;
    const ic = d.querySelector('svg.tabler-icon-player-pause-filled, svg.tabler-icon-player-play-filled');
    if (!d.__syncHooked) { d.__syncHooked = true; d.addEventListener('pointerdown', () => { master = i; }, true); }
    return { d, th, max, t, playing: ic ? ic.classList.contains('tabler-icon-player-pause-filled') : null,
             btn: ic && ic.closest('button'), track: th.closest('.mantine-Slider-trackContainer') || th.parentElement };
  }
  function seek(c, t) {
    const w = c.d.defaultView, r = c.track.getBoundingClientRect();
    const x = r.left + r.width * Math.min(1, Math.max(0, t / c.max)), y = r.top + r.height / 2;
    const o = { bubbles: true, cancelable: true, clientX: x, clientY: y, button: 0, buttons: 1, view: w };
    c.track.dispatchEvent(new w.MouseEvent('mousedown', o));
    c.d.dispatchEvent(new w.MouseEvent('mouseup', Object.assign({}, o, { buttons: 0 })));
  }
  function setPlaying(c, want) { if (c.playing !== null && c.playing !== want && c.btn) c.btn.click(); }
  const btn = $('syncPlay'), rng = $('syncRange'), lab = $('syncTime');
  btn.onclick = () => { const a = ctl(master) || ctl(1 - master); if (!a) return; const want = !a.playing;
    for (const i of [0, 1]) { const c = ctl(i); if (c) setPlaying(c, want); } };
  rng.addEventListener('pointerdown', () => { dragging = true; });
  window.addEventListener('pointerup', () => { dragging = false; });
  rng.addEventListener('input', () => { const v = parseFloat(rng.value); for (const i of [0, 1]) { const c = ctl(i); if (c) seek(c, v); }
    lastSeek = [performance.now(), performance.now()]; });
  function loop(now) {
    const m = ctl(master), s = ctl(1 - master);
    if (m) {
      rng.max = m.max.toFixed(3); if (!dragging) rng.value = m.t; lab.textContent = m.t.toFixed(1) + ' / ' + m.max.toFixed(1) + ' s';
      btn.textContent = m.playing ? '❚❚  pause both' : '▶  play both';
      if (s && !dragging) {
        if (m.playing !== null && s.playing !== m.playing) setPlaying(s, m.playing);
        const tol = m.playing ? 0.3 : 0.06, j = 1 - master;
        if (Math.abs(s.t - Math.min(m.t, s.max)) > tol && now - lastSeek[j] > 400) { seek(s, Math.min(m.t, s.max)); lastSeek[j] = now; }
      }
      const v = $('v2d');                                        // the 2D video follows the same time
      if (v && v.readyState >= 2 && Number.isFinite(v.duration)) {
        const tv = Math.min(m.t + (window.SYNC_OFFSET || 0), v.duration - 0.05);
        if (m.playing) { if (v.paused) v.play().catch(() => {}); if (Math.abs(v.currentTime - tv) > 0.3) v.currentTime = tv; }
        else { if (!v.paused) v.pause(); if (Math.abs(v.currentTime - tv) > 0.05) v.currentTime = tv; }
      }
    }
    requestAnimationFrame(loop);
  }
  requestAnimationFrame(loop);
  window.__viserSync = { ctl, seek, setPlaying, reset() { master = 0; lastSeek = [0, 0]; } };
})();
