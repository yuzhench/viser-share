// TouchScale baseline page: one 3D view (viser recording per clip, setup/ts_viser_record.py) with every baseline setting's hand under its
// own scene node (/vitra_est_hand, /vitra_true_hand, /wilor_hand, /hamer_hand, /fact_hand). The buttons switch those nodes with the
// viser client's own scene-tree action (the call behind the eye icon in its scene tree), reached through React's fiber of the client's
// canvas; re-applied every 300 ms, so the choice holds when the clip changes or playback restarts. Per-clip numbers from stats.json
// (setup/ts_prep_viser.py, from setup/ts_depth_check.py). Under it, the head camera with all five settings' keypoints, with its own player.
(function () {
  'use strict';
  var METH = [
    { key: 'vitra_est', name: 'VITRA (HaWoR)', sub: 'FoV estimated from the frames (VITRA’s own setting)', mesh: '#eb2323', kp: '#00e1ff' },
    { key: 'vitra_true', name: 'VITRA (HaWoR)', sub: 'the camera’s true FoV', mesh: '#e6b400', kp: '#00e1ff' },
    { key: 'wilor', name: 'WiLoR', sub: 'its own hand detector (how TouchScale uses it)', mesh: '#2864ff', kp: '#ffa000' },
    { key: 'hamer', name: 'HaMeR', sub: 'ViTPose+ hand boxes', mesh: '#aa3ce6', kp: '#ffffff' },
    { key: 'fact', name: 'FACT-style', sub: 'HaMeR + smoothing + moved onto the depth camera’s surface', mesh: '#ff69b4', kp: '#141e78' }
  ];
  var CLIPS = [
    { ep: 'e0de6ab8', task: 'pour water from a measuring cup' }, { ep: '8f839902', task: 'screw a bottle cap tight' },
    { ep: '806d9d6e', task: 'wet filter paper with a wash bottle' }, { ep: 'cc6d9831', task: 'scoop rice with a spoon' }
  ];
  var want = { vitra_est: true, vitra_true: false, wilor: false, hamer: false, fact: false }, cur = CLIPS[0].ep, stats = null;
  var frame = document.getElementById('ts3d'), full = document.getElementById('tsFull');
  var clipBox = document.getElementById('clips'), methBox = document.getElementById('meths');

  function src(ep) { return '../../viser-client/index.html?playbackPath=' + encodeURIComponent('../pipeline/touchscale/recordings/' + ep + '.viser'); }
  function pct(x) { return x == null ? '–' : Math.round(100 * x) + '%'; }
  function cm(x) { return x == null ? '–' : (x >= 0 ? '+' : '') + x.toFixed(1) + ' cm'; }

  function buildClips() {
    CLIPS.forEach(function (c) {
      var b = document.createElement('button'); b.type = 'button'; b.dataset.ep = c.ep;
      b.innerHTML = c.task + '<small>' + c.ep + '</small>';
      b.addEventListener('click', function () { setClip(c.ep); });
      clipBox.appendChild(b);
    });
  }
  function buildMeths() {
    METH.forEach(function (m) {
      var b = document.createElement('button'); b.type = 'button'; b.dataset.key = m.key; b.className = 'meth';
      b.innerHTML = '<span class="sw" style="background:' + m.mesh + ';border-color:' + m.kp + '"></span><span class="mt"><b>' + m.name +
        '</b><span class="ms">' + m.sub + '</span><span class="mn"></span></span>';
      b.addEventListener('click', function () { want[m.key] = !want[m.key]; paint(); apply(); });
      methBox.appendChild(b);
    });
  }
  function paint() {
    Array.prototype.forEach.call(clipBox.children, function (b) { b.setAttribute('aria-pressed', String(b.dataset.ep === cur)); });
    Array.prototype.forEach.call(methBox.children, function (b) {
      var k = b.dataset.key; b.setAttribute('aria-pressed', String(!!want[k]));
      var s = stats && stats[cur] && stats[cur].methods[k], el = b.querySelector('.mn');
      if (!s) { el.textContent = ''; return; }
      var n = stats[cur].frames, shown = Math.max(s.shown_frames.left, s.shown_frames.right);
      el.textContent = 'this clip: palm ' + cm(s.behind_cm_median) + ' behind the measured surface (median) · ' + pct(s.within_0_3cm) +
        ' in the 0-3 cm range · hand found in ' + shown + ' of ' + n + ' frames';
    });
  }
  function setClip(ep) {
    cur = ep; paint();
    frame.src = src(ep); full.href = src(ep);
  }
  // scene-node switching inside the viser client
  function actions(w) {
    try {
      var el = w.document.querySelector('canvas'); if (!el) return null;
      if (w.__handActions && w.__handActions.el === el) return w.__handActions.a;
      var key = Object.keys(el).find(function (k) { return k.indexOf('__reactFiber$') === 0; }); if (!key) return null;
      for (var f = el[key]; f; f = f.return) {
        var v = f.memoizedProps && f.memoizedProps.value;
        if (v && typeof v === 'object' && v.sceneTreeActions) { w.__handActions = { el: el, a: v.sceneTreeActions }; return v.sceneTreeActions; }
      }
    } catch (e) { /* client not ready yet */ }
    return null;
  }
  function apply() {
    var w = null; try { w = frame.contentWindow; } catch (e) { return; }
    if (!w || !w.__viserSceneTree) return;
    var st = w.__viserSceneTree.getState(), a = null;
    METH.forEach(function (m) {
      var node = '/' + m.key + '_hand', n = st[node]; if (!n) return;
      var v = !!want[m.key];
      if (n.overrideVisibility !== v) { a = a || actions(w); if (a) a.updateNodeAttributes(node, { overrideVisibility: v }); }
    });
  }
  // head-camera videos of all five settings (video2d/<clip>_<setting>.mp4, setup/ts_render_2d.py): always shown, their own player
  // (play / pause all, one slider); the first video leads and the others are kept on its time
  var vbox = document.getElementById('v2d'), vPlay = document.getElementById('vPlay'), vRange = document.getElementById('vRange'),
      vTime = document.getElementById('vTime'), vkey = '', vids = [], vPaused = false, vDrag = false;
  function buildVideos() {
    if (!vbox || vkey === cur) return;
    vkey = cur; vbox.innerHTML = ''; vids = [];
    METH.forEach(function (m) {
      var f = document.createElement('figure');
      f.innerHTML = '<figcaption><span class="sw" style="background:' + m.mesh + ';border-color:' + m.kp + '"></span><b>' + m.name + '</b><span class="ms">' + m.sub + '</span></figcaption>' +
        '<video muted playsinline loop preload="auto"></video>';
      var v = f.querySelector('video'); v.src = 'video2d/' + cur + '_' + m.key + '.mp4';
      vbox.appendChild(f); vids.push(v);
      if (!vPaused) v.play().catch(function () {});
    });
  }
  function paintV() { if (vPlay) vPlay.textContent = vPaused ? '▶  play all' : '❚❚  pause all'; }
  if (vPlay) vPlay.addEventListener('click', function () {
    vPaused = !vPaused; var t = vids[0] ? vids[0].currentTime : 0;
    vids.forEach(function (v) { if (vPaused) { v.pause(); v.currentTime = t; } else v.play().catch(function () {}); });
    paintV();
  });
  if (vRange) {
    vRange.addEventListener('pointerdown', function () { vDrag = true; });
    window.addEventListener('pointerup', function () { vDrag = false; });
    vRange.addEventListener('input', function () { var t = parseFloat(vRange.value); vids.forEach(function (v) { v.currentTime = t; }); });
  }
  function tickV() {
    try {
      buildVideos();
      var m = vids[0];
      if (m && isFinite(m.duration) && m.duration > 0) {
        vRange.max = m.duration.toFixed(2); if (!vDrag) vRange.value = m.currentTime;
        vTime.textContent = m.currentTime.toFixed(1) + ' / ' + m.duration.toFixed(1) + ' s';
        if (!vPaused && m.paused) m.play().catch(function () {});
        vids.slice(1).forEach(function (v) {
          if (v.readyState < 2) return;
          if (Math.abs(v.currentTime - m.currentTime) > (vPaused ? 0.02 : 0.15)) v.currentTime = m.currentTime;
          if (vPaused && !v.paused) v.pause(); if (!vPaused && v.paused) v.play().catch(function () {});
        });
      }
      paintV();
    } catch (e) { if (window.console) console.warn('2D videos', e); }
  }
  setInterval(tickV, 100);
  buildClips(); buildMeths(); setClip(cur);
  fetch('stats.json').then(function (r) { return r.json(); }).then(function (j) { stats = j; paint(); }).catch(function () {});
  setInterval(apply, 300);
  window.TSPage = { setClip: setClip, want: want, apply: apply };
})();
