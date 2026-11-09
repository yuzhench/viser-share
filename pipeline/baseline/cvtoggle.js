// Baseline buttons under the Ours card's bar. Each one shows that method's hand in our point cloud (the ours recordings, made with
// setup/viser_record_1025.py --cv-toggle, carry /vitra_hand (HaWoR), /wilor_hand, /hamer_hand, /fact_hand, all switched off); the right
// view follows the last one picked (the baseline recording carries VITRA's hand (/hands) and the other methods' hands in VITRA's world,
// switched off), and the card under it shows the head camera with each picked method's hand (VITRA when none is picked): one Mano3V
// player per method on cvview/ (setup/render_cvview_1025.py), driven by the 3D timeline like the camera views on the left.
// Scene nodes are switched with the viser client's own scene-tree action (the call behind the eye icon in its scene tree), reached
// through React's fiber of the client's canvas; re-applied every 300 ms, so the choice holds when the part changes or playback restarts.
(function () {
  'use strict';
  var btns = Array.prototype.slice.call(document.querySelectorAll('[data-node]')), want = {}, order = [];
  var RIGHT = {
    vitra: { node: '/hands', title: 'VITRA', sub: 'its own point cloud, camera and hand (HaWoR), in its own world' },
    wilor: { node: '/wilor_hand', title: 'WiLoR', sub: 'its hand (oracle boxes; blue mesh, orange keypoints) in VITRA’s world: its camera and depth' },
    hamer: { node: '/hamer_hand', title: 'HaMeR', sub: 'its hand (oracle boxes; purple mesh, white keypoints) in VITRA’s world: its camera and depth' },
    fact: { node: '/fact_hand', title: 'FACT-style', sub: 'its hand (ViTPose boxes, depth anchor; pink mesh, navy keypoints) in VITRA’s world: its camera and depth' }
  };
  var CVV = { vitra: { name: 'VITRA', label: 'VITRA (HaWoR)', mesh: '#eb2323', kp: '#00e1ff' }, wilor: { name: 'WiLoR', label: 'WiLoR (oracle boxes)', mesh: '#2864ff', kp: '#ffa000' },
              hamer: { name: 'HaMeR', label: 'HaMeR (oracle boxes)', mesh: '#aa3ce6', kp: '#ffffff' },
              fact: { name: 'FACT-style', label: 'FACT-style (ViTPose boxes, depth anchor)', mesh: '#ff69b4', kp: '#141e78' } };
  btns.forEach(function (b) { want[b.dataset.node] = false; });
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
  function win(id) { var fr = document.getElementById(id); try { return fr && fr.contentWindow; } catch (e) { return null; } }
  function setNodes(w, wantMap) {                      // wantMap: node -> true (force on) / false (force off) / undefined (as recorded)
    if (!w || !w.__viserSceneTree) return;
    var st = w.__viserSceneTree.getState(), a = null;
    Object.keys(wantMap).forEach(function (node) {
      var n = st[node]; if (!n) return;
      if (n.overrideVisibility !== wantMap[node]) { a = a || actions(w); if (a) a.updateNodeAttributes(node, { overrideVisibility: wantMap[node] }); }
    });
  }
  function rightKey() { for (var i = order.length - 1; i >= 0; i--) if (want['/' + order[i] + '_hand']) return order[i]; return 'vitra'; }
  function apply() {
    var mine = {}; Object.keys(want).forEach(function (n) { mine[n] = want[n] ? true : undefined; });
    setNodes(win('viserA'), mine);
    var rk = rightKey(), right = {};
    Object.keys(RIGHT).forEach(function (k) { right[RIGHT[k].node] = k === rk ? (k === 'vitra' ? undefined : true) : (k === 'vitra' ? false : undefined); });
    setNodes(win('viserB'), right);
    var tb = document.getElementById('titleB'), sb = document.getElementById('subB');
    if (tb && tb.textContent !== RIGHT[rk].title) { tb.textContent = RIGHT[rk].title; if (sb) sb.textContent = RIGHT[rk].sub; }
  }
  // head-camera views of the picked baselines
  var cvvKey = '', players = [], cvvGen = 0;
  function picked() { var ks = btns.filter(function (b) { return want[b.dataset.node]; }).map(function (b) { return b.dataset.key; }); return ks.length ? ks : ['vitra']; }
  function curPart() { try { return cur; } catch (e) { return null; } }                 // the page's current part (its global 'cur')
  function cvvSync() {
    var el = document.getElementById('cvv'), part = curPart(); if (!el || !window.Mano3V || !part) return;
    var ks = picked(), key = part + '|' + ks.join(','); if (key === cvvKey) return;
    cvvKey = key; var gen = ++cvvGen; players = []; el.innerHTML = '';
    var t = document.getElementById('cvvTitle');
    if (t) t.textContent = 'Head camera with the ' + ks.map(function (k) { return CVV[k].name; }).join(' and ') + (ks.length > 1 ? ' hands' : ' hand');
    ks.forEach(function (k) {
      var box = document.createElement('div'); box.className = 'cvv-item';
      box.innerHTML = '<div class="cvv-lab"><i style="background:' + CVV[k].mesh + ';border-color:' + CVV[k].kp + '"></i>' + CVV[k].label + '</div><div class="cvv-m"></div>';
      el.appendChild(box);
      Promise.resolve(window.Mano3V.mount(box.querySelector('.cvv-m'), 'cvview/', part, { key: k, color: CVV[k].kp, controls: false }))
        .then(function (p) { if (gen === cvvGen && p) players.push(p); });
    });
  }
  function follow() {
    try {
      cvvSync();
      var S = window.__viserSync, c = S && (S.ctl(0) || S.ctl(1)), part = curPart();
      var o = (typeof OFFC !== 'undefined' && part && OFFC[part]) || 0;
      if (c) players.forEach(function (p) {
        var end = p.realDuration - 1e-3, t;
        if (c.playing) { t = Math.min(Math.max(0, c.t + o), end); if (p.video.playbackRate !== (c.rate || 1)) p.video.playbackRate = c.rate || 1;   // the 3D view's speed
          if (p.video.paused) p.play(); if (Math.abs(p.realTime() - t) > 0.25) p.seekReal(t); p.lastT = null; }
        else { if (!p.video.paused) p.pause(); t = Math.min(Math.max(0, Math.floor(c.t / 0.2 + 1e-6) * 0.2 + o), end);
          if (t !== p.lastT || Math.abs(p.realTime() - t) > 0.1) { p.seekReal(t); p.lastT = t; } }
      });
    } catch (e) { if (window.console) console.warn('baseline camera views', e); }
    requestAnimationFrame(follow);
  }
  requestAnimationFrame(follow);
  function mark() { btns.forEach(function (b) { b.setAttribute('aria-pressed', String(!!want[b.dataset.node])); }); }
  btns.forEach(function (b) {
    b.addEventListener('click', function () {
      var n = b.dataset.node, k = b.dataset.key; want[n] = !want[n];
      order = order.filter(function (x) { return x !== k; }); if (want[n]) order.push(k);
      mark(); apply();
    });
  });
  setInterval(apply, 300);
  window.__handToggle = { state: function () { return { want: JSON.parse(JSON.stringify(want)), right: rightKey(), views: picked(), players: players.length }; } };
})();
