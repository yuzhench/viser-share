// Touch panel for the ours vs VITRA page: both gloves' taxels, drawn beside our 3D view and following its timeline:
// window.__viserSync.ctl(0) gives seconds from the part's first 3D frame; while paused the panel snaps to the 3D frame on screen
// (5 a second), like the camera views. No amplitude cut: every filtered contact is drawn (a 0.15 cut removed light contact, e.g. a
// two-finger pinch). Two views: 'hand' (default) paints the taxels on a MANO hand seen from the palm, the left glove on the mirrored
// hand (positions approximate: tactile/hand.json + hand.bin from setup/hand_taxel_map.py; per pixel a Gaussian mix, sigma 0.45 taxel,
// of the working taxels around the pixel's grid position), and 'grid', the glove's 24 x 31 slot grid (fingers up).
// Data from setup/export_tactile_web_1025.py: tactile/meta.json + tactile/part<k>.bin (uint8 = value x 255; per part: left
// filtered, left raw, right filtered, right raw, each n x slots). 'filled' view: tactile/part<k>_filled.bin (left, right; filtered with
// the dead taxels filled from the same glove's working neighbours, setup/tactile_fill_web_1025.py).   Touch.mount(element, 'tactile/')
(function () {
  'use strict';
  var VMAX = 0.3;                                              // colour scale 0 -> 0.3, as in the tactile figure
  var STOPS = [[0, [50, 55, 64]], [0.012, [96, 26, 30]], [0.06, [158, 30, 36]], [0.13, [226, 44, 40]], [0.21, [250, 138, 36]], [0.3, [255, 232, 92]]];
  function rgb(v) {
    if (v <= 0) return STOPS[0][1];
    for (var i = 1; i < STOPS.length; i++) {
      var a = STOPS[i - 1], b = STOPS[i];
      if (v <= b[0]) { var w = (v - a[0]) / (b[0] - a[0]); return a[1].map(function (x, c) { return Math.round(x + w * (b[1][c] - x)); }); }
    }
    return STOPS[STOPS.length - 1][1];
  }
  var LUT = [], LUTB = new Uint8Array(256 * 3);
  for (var q = 0; q < 256; q++) { var cq = rgb(q / 255); LUT.push('rgb(' + cq.join(',') + ')'); LUTB[3 * q] = cq[0]; LUTB[3 * q + 1] = cq[1]; LUTB[3 * q + 2] = cq[2]; }
  var NOTE = {
    filtered: 'Only taxels in contact: each taxel’s resting level removed, then a threshold on its own noise (on above 4σ, off below 2σ, at least 25 ms); single specks dropped.',
    raw: 'The calibrated pressure as recorded: many taxels sit above 0 even with the hand free.',
    filled: 'Filtered contact, with the dead taxels filled from the same glove’s working neighbours: for each one, a linear fit from its 12 nearest working taxels, learned at the other places of this glove in this take. An estimate, not a measurement'
  };
  var HNOTE = ' Hand view: taxel positions are approximate (the grid’s finger strips laid along the fingers, its palm block over the palm; the glove’s true taxel positions were not recorded); the left glove is on a mirrored right hand. Drag to turn the hands, double-click to reset.';
  var SKIN = [0.31, 0.33, 0.37];                               // the hand where there is no taxel, or a dead one (also the legend swatch)
  var CSS = '#touch.tc{background:#14171c;color:#e6e9ed;padding:10px 12px 12px}' +
    '.tc-lab{display:flex;justify-content:space-between;align-items:baseline;gap:6px;font-size:12.5px;margin:0 0 4px}' +
    '.tc-lab b{font-weight:600}.tc-lab span{color:#aab3bd;font-variant-numeric:tabular-nums;white-space:nowrap}' +
    '.tc canvas{display:block}.tc-glove+.tc-glove{margin-top:10px}.tc-cbar{margin-top:12px}' +
    '.tc canvas.tc-hand{touch-action:none;cursor:grab}.tc canvas.tc-hand:active{cursor:grabbing}' +
    '.tc-view{display:flex;align-items:center;gap:8px;font-size:11.5px;color:#8f99a4;margin:0 0 8px}' +
    '.tc-seg{display:inline-flex;border:1px solid #2f3640;border-radius:6px;overflow:hidden}.tc-seg button{font:inherit;font-size:11.5px;line-height:1.4;padding:1px 10px;border:0;background:#1c2027;color:#aab3bd;cursor:pointer}' +
    '.tc-seg button[aria-pressed="true"]{background:#3b6fd8;color:#fff}' +
    '.tc-ticks{display:flex;justify-content:space-between;font-size:11px;color:#aab3bd;margin-top:2px}' +
    '.tc-time{font-size:11.5px;color:#aab3bd;margin-top:8px;font-variant-numeric:tabular-nums}' +
    '.tc-note{font-size:11.5px;line-height:1.45;color:#8f99a4;margin:6px 0 0}' +
    '.tc-dead{display:flex;align-items:center;gap:6px;font-size:11px;color:#8f99a4;margin-top:6px}.tc-dead i{width:7px;height:7px;border-radius:50%;border:1.5px solid #6b7480;display:inline-block;flex:none}' +
    '.tc-dead i.sq{width:9px;height:9px;border:0;border-radius:2px;background:rgb(' + SKIN.map(function (x) { return Math.round(x * 255); }).join(',') + ')}' +
    '.tc-fill{display:none;align-items:center;gap:6px;font-size:11px;color:#8f99a4;margin-top:4px}.tc-fill label{display:flex;align-items:center;gap:6px;cursor:pointer}.tc-fill input{margin:0;accent-color:#6ea8ff}.tc-fill i{width:7px;height:7px;border-radius:50%;background:#e22c28;box-shadow:0 0 0 1.5px #c9d1d9;display:inline-block;flex:none}' +
    '@media (min-width:1101px){.grid3>.view.touch{max-height:calc(100vh - 16px);overflow-y:auto}}' +
    '@media (max-width:1100px){#touch.tc{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);column-gap:18px}' +
    '#touch.tc .tc-glove+.tc-glove{margin-top:0}#touch.tc .tc-view,#touch.tc .tc-cbar,#touch.tc .tc-dead,#touch.tc .tc-fill,#touch.tc .tc-time,#touch.tc .tc-note{grid-column:1/-1}}';
  var markFilled = false;                                     // 'filled' grid view: ring the filled taxels (off: the plain picture)
  var view = 'hand', HAND = null, GLS = [null, null], CELL = null, rot = [0, 0], handErr = '';
  var meta = null, base = '', want = null, mode = 'filtered', drawn = '', sizeGen = 0, root = null, last = -1;
  var cache = {}, E = {}, DEAD = [{}, {}], UNF = [{}, {}], ABS = [{}, {}];   // ABS: slots of meta.slots this glove does not carry; UNF: dead taxels the 'filled' view leaves empty; DEAD: never respond in this take

  // ---- hand view (WebGL) ----
  var VS = 'attribute vec3 aP; attribute vec3 aN; attribute vec3 aG;' +
    'uniform mat3 uR; uniform vec3 uC; uniform vec2 uS; uniform float uM; varying vec3 vN; varying vec3 vG;' +
    'void main() { vec3 p = uR * (vec3(aP.x * uM, aP.y, aP.z) - uC); vN = uR * vec3(aN.x * uM, aN.y, aN.z); vG = aG;' +
    ' gl_Position = vec4(p.x * uS.x, p.y * uS.y, -p.z * 4.0, 1.0); }';
  var FS = 'precision mediump float; uniform sampler2D uVal; uniform sampler2D uMask; uniform sampler2D uLut;' +
    'uniform vec3 uSkin; uniform vec3 uL; uniform float uK; varying vec3 vN; varying vec3 vG;' +
    'void main() { vec2 g = vG.xy; vec2 c0 = floor(g + 0.5); float ws = 0.0; float vs = 0.0;' +
    ' for (int dy = -1; dy <= 1; dy++) { for (int dx = -1; dx <= 1; dx++) {' +
    '  vec2 c = c0 + vec2(float(dx), float(dy)); if (c.x < 0.0 || c.y < 0.0 || c.x > 30.5 || c.y > 23.5) continue;' +
    '  vec2 uv = (c + 0.5) / 32.0; vec2 d = g - c; float w = exp(-dot(d, d) * uK) * texture2D(uMask, uv).r;' +
    '  ws += w; vs += w * texture2D(uVal, uv).r; } }' +
    ' float v = vs / max(ws, 0.0001); float t = clamp((ws - 0.15) / 0.4, 0.0, 1.0); float a = t * t * (3.0 - 2.0 * t) * vG.z;' +
    ' vec3 heat = texture2D(uLut, vec2((v * 255.0 + 0.5) / 256.0, 0.5)).rgb;' +
    ' vec3 n = normalize(vN); float lam = max(dot(n, uL), 0.0); vec3 hv = normalize(uL + vec3(0.0, 0.0, 1.0));' +
    ' float sp = pow(max(dot(n, hv), 0.0), 24.0) * 0.10;' +
    ' vec3 col = mix(uSkin * (0.38 + 0.62 * lam), heat * (0.72 + 0.28 * lam), a) + sp;' +
    ' gl_FragColor = vec4(col, 1.0); }';

  function decodeHand(j, b) {
    var o = j.offsets, n = j.nV, m = j.nF, i;
    var qp = new Int16Array(b, o.pos, n * 3), qn = new Int8Array(b, o.normal, n * 3), qg = new Uint16Array(b, o.grid, n * 3);
    var pos = new Float32Array(n * 3), nrm = new Float32Array(n * 3), grid = new Float32Array(n * 3);
    for (i = 0; i < n * 3; i++) { pos[i] = qp[i] / j.qpos; nrm[i] = qn[i] / 127; }
    for (i = 0; i < n; i++) { grid[3 * i] = qg[3 * i] / j.qgrid - 1; grid[3 * i + 1] = qg[3 * i + 1] / j.qgrid - 1; grid[3 * i + 2] = qg[3 * i + 2] / 65535; }
    var lo = j.bbox[0], hi = j.bbox[1], pad = 1.05;
    return { j: j, n: n, m: m, pos: pos, nrm: nrm, grid: grid, faces: new Uint16Array(b, o.faces, m * 3),
      box: { cx: (lo[0] + hi[0]) / 2, cy: (lo[1] + hi[1]) / 2, cz: (lo[2] + hi[2]) / 2, w: (hi[0] - lo[0]) * pad, h: (hi[1] - lo[1]) * pad } };
  }

  function initGL(c, mirror) {
    var gl = c.getContext('webgl', { antialias: true, alpha: true, premultipliedAlpha: true }) || c.getContext('experimental-webgl');
    if (!gl) throw new Error('no WebGL');
    function sh(t, src) { var s = gl.createShader(t); gl.shaderSource(s, src); gl.compileShader(s); if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s)); return s; }
    var p = gl.createProgram(); gl.attachShader(p, sh(gl.VERTEX_SHADER, VS)); gl.attachShader(p, sh(gl.FRAGMENT_SHADER, FS)); gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    function buf(arr, tgt) { var b = gl.createBuffer(); gl.bindBuffer(tgt, b); gl.bufferData(tgt, arr, gl.STATIC_DRAW); return b; }
    function tex(w, h, fmt, data) {
      var t = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, t); gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
      gl.texImage2D(gl.TEXTURE_2D, 0, fmt, w, h, 0, fmt, gl.UNSIGNED_BYTE, data);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      return t;
    }
    var G = { gl: gl, prog: p, m: mirror ? -1 : 1, vals: new Uint8Array(1024), mask: new Uint8Array(1024), maskKey: '', L: {} };
    G.bP = buf(HAND.pos, gl.ARRAY_BUFFER); G.bN = buf(HAND.nrm, gl.ARRAY_BUFFER); G.bG = buf(HAND.grid, gl.ARRAY_BUFFER); G.bF = buf(HAND.faces, gl.ELEMENT_ARRAY_BUFFER);
    G.tVal = tex(32, 32, gl.LUMINANCE, G.vals); G.tMask = tex(32, 32, gl.LUMINANCE, G.mask); G.tLut = tex(256, 1, gl.RGB, LUTB);
    ['aP', 'aN', 'aG'].forEach(function (a) { G.L[a] = gl.getAttribLocation(p, a); });
    ['uR', 'uC', 'uS', 'uM', 'uVal', 'uMask', 'uLut', 'uSkin', 'uL', 'uK'].forEach(function (u) { G.L[u] = gl.getUniformLocation(p, u); });
    return G;
  }

  function maskKey() { return mode + '|' + want; }

  function setMask(h) {
    var G = GLS[h], gl = G.gl, S = meta.slots.length, fl = mode === 'filled';
    G.mask.fill(0);
    for (var s = 0; s < S; s++) if (!ABS[h][s] && (!DEAD[h][s] || (fl && !UNF[h][s]))) G.mask[CELL[s]] = 255;
    gl.bindTexture(gl.TEXTURE_2D, G.tMask); gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 32, 32, gl.LUMINANCE, gl.UNSIGNED_BYTE, G.mask);
    G.maskKey = maskKey();
  }

  function drawHand(h, d, off) {
    var G = GLS[h], gl = G.gl, S = meta.slots.length, B = HAND.box, c = gl.canvas, L = G.L;
    G.vals.fill(0);
    for (var s = 0; s < S; s++) G.vals[CELL[s]] = d[off + s];
    gl.viewport(0, 0, c.width, c.height); gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST); gl.useProgram(G.prog);
    if (G.maskKey !== maskKey()) setMask(h);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, G.tVal); gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 32, 32, gl.LUMINANCE, gl.UNSIGNED_BYTE, G.vals);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, G.tMask);
    gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, G.tLut);
    gl.uniform1i(L.uVal, 0); gl.uniform1i(L.uMask, 1); gl.uniform1i(L.uLut, 2);
    var a = rot[0], b = rot[1], ca = Math.cos(a), sa = Math.sin(a), cb = Math.cos(b), sb = Math.sin(b);
    gl.uniformMatrix3fv(L.uR, false, new Float32Array([cb, sa * sb, -ca * sb, 0, ca, sa, sb, -sa * cb, ca * cb]));   // Rx(a) Ry(b), column-major
    gl.uniform3f(L.uC, B.cx * G.m, B.cy, B.cz); gl.uniform2f(L.uS, 2 / B.w, 2 / B.h); gl.uniform1f(L.uM, G.m);
    var sg = HAND.j.sigma || 0.45; gl.uniform1f(L.uK, 1 / (2 * sg * sg));
    gl.uniform3f(L.uSkin, SKIN[0], SKIN[1], SKIN[2]);
    var lx = -0.35, ly = 0.5, lz = 0.8, ln = Math.sqrt(lx * lx + ly * ly + lz * lz); gl.uniform3f(L.uL, lx / ln, ly / ln, lz / ln);
    [['aP', G.bP, gl.FLOAT], ['aN', G.bN, gl.FLOAT], ['aG', G.bG, gl.FLOAT]].forEach(function (x) {
      gl.bindBuffer(gl.ARRAY_BUFFER, x[1]); gl.enableVertexAttribArray(L[x[0]]); gl.vertexAttribPointer(L[x[0]], 3, x[2], false, 0, 0);
    });
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, G.bF); gl.drawElements(gl.TRIANGLES, HAND.m * 3, gl.UNSIGNED_SHORT, 0);
  }

  function handReady() { return !!(HAND && GLS[0] && GLS[1]); }

  function loadHand() {
    Promise.all([fetch(base + 'hand.json').then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); }),
                 fetch(base + 'hand.bin').then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.arrayBuffer(); })])
      .then(function (a) {
        HAND = decodeHand(a[0], a[1]);
        GLS = [initGL(E.hc[0], true), initGL(E.hc[1], false)];
        size(); setView(view);
      }).catch(function (e) {
        GLS = [null, null]; handErr = e.message; if (window.console) console.warn('touch panel: hand view off', e);
        if (E.vw) E.vw.style.display = 'none';
        setView('grid');
      });
  }

  function setView(v) {
    view = v === 'grid' ? 'grid' : 'hand'; drawn = '';
    var hand = view === 'hand' && handReady();
    if (E.vbtn) E.vbtn.forEach(function (b) { b.setAttribute('aria-pressed', String(b.dataset.view === view)); });
    if (E.cv) for (var h = 0; h < 2; h++) { E.cv[h].style.display = hand ? 'none' : 'block'; E.hc[h].style.display = hand ? 'block' : 'none'; }
    legend();
    if (last >= 0) draw(last);
  }

  function legend() {
    if (!E.note || !meta) return;
    var hand = view === 'hand' && handReady();
    E.note.textContent = NOTE[mode] + (mode === 'filled' ? (hand ? '.' : '; tick the box above to ring the filled taxels.') + fillNote() : '') + (hand ? HNOTE : '');
    E.deadI.className = hand ? 'sq' : '';
    E.deadTxt.textContent = hand ? (mode === 'filled' ? 'grey: no taxel there, or a dead taxel left empty (its neighbours do not predict it)' : 'grey: no taxel there, or a dead one (no reading in this whole take)')
      : (mode === 'filled' ? 'empty ring: a dead taxel left empty (its neighbours do not predict it)' : 'no reading in this whole take (dead row or column of the glove)');
    E.fill.style.display = mode === 'filled' && !hand ? 'flex' : 'none';
  }

  function build() {
    if (!document.getElementById('tc-style')) { var s = document.createElement('style'); s.id = 'tc-style'; s.textContent = CSS; document.head.appendChild(s); }
    root.classList.add('tc');
    root.innerHTML = '<div class="tc-view"><span>view</span><span class="tc-seg" role="group" aria-label="Touch view"><button type="button" data-view="hand" aria-pressed="false">hand</button>' +
      '<button type="button" data-view="grid" aria-pressed="false">grid</button></span></div>' +
      '<div class="tc-glove"><div class="tc-lab"><b>left glove</b><span></span></div><canvas class="tc-grid"></canvas><canvas class="tc-hand" style="display:none"></canvas></div>' +
      '<div class="tc-glove"><div class="tc-lab"><b>right glove</b><span></span></div><canvas class="tc-grid"></canvas><canvas class="tc-hand" style="display:none"></canvas></div>' +
      '<div class="tc-cbar"><canvas></canvas><div class="tc-ticks"><span>0</span><span>0.1</span><span>0.2</span><span>0.3</span></div></div>' +
      '<div class="tc-dead"><i></i><span></span></div>' +
      '<div class="tc-fill"><label><input type="checkbox"><i></i><span>ring the filled taxels (dead taxels, value from their neighbours, not measured)</span></label></div>' +
      '<div class="tc-time"></div><p class="tc-note"></p>';
    E.cv = Array.prototype.slice.call(root.querySelectorAll('canvas.tc-grid')); E.hc = Array.prototype.slice.call(root.querySelectorAll('canvas.tc-hand'));
    E.cb = root.querySelector('.tc-cbar canvas');
    E.n = Array.prototype.slice.call(root.querySelectorAll('.tc-lab span'));
    E.time = root.querySelector('.tc-time'); E.note = root.querySelector('.tc-note');
    E.deadTxt = root.querySelector('.tc-dead span'); E.deadI = root.querySelector('.tc-dead i'); E.fill = root.querySelector('.tc-fill'); E.mark = root.querySelector('.tc-fill input');
    E.vw = root.querySelector('.tc-view'); E.vbtn = Array.prototype.slice.call(root.querySelectorAll('[data-view]'));
    E.vbtn.forEach(function (b) { b.addEventListener('click', function () { setView(b.dataset.view); }); });
    E.mark.checked = markFilled; E.mark.addEventListener('change', function () { markFilled = E.mark.checked; drawn = ''; if (last >= 0) draw(last); });
    E.btn = Array.prototype.slice.call((root.closest('.view') || document).querySelectorAll('[data-tmode]'));
    E.btn.forEach(function (b) { b.addEventListener('click', function () { setMode(b.dataset.tmode); }); if (b.dataset.tmode === 'filled' && !meta.filled) b.style.display = 'none'; });
    var drag = null;
    E.hc.forEach(function (c) {
      c.addEventListener('pointerdown', function (e) { drag = [e.clientX, e.clientY, rot[0], rot[1]]; try { c.setPointerCapture(e.pointerId); } catch (err) {} e.preventDefault(); });
      c.addEventListener('pointermove', function (e) {
        if (!drag) return;
        rot[1] = drag[3] + (e.clientX - drag[0]) * 0.012; rot[0] = Math.max(-1.3, Math.min(1.3, drag[2] + (e.clientY - drag[1]) * 0.012));
        sizeGen++; drawn = '';
      });
      c.addEventListener('pointerup', function () { drag = null; }); c.addEventListener('pointercancel', function () { drag = null; });
      c.addEventListener('dblclick', function () { rot = [0, 0]; sizeGen++; drawn = ''; });
    });
    if (window.ResizeObserver) new ResizeObserver(size).observe(root);
    window.addEventListener('resize', size);
    setMode(mode); setView(view);
  }

  function size() {
    if (!meta || !E.cv) return;
    var w = E.cv[0].parentElement.clientWidth, bw = E.cb.parentElement.clientWidth;   // a glove's column, the colour bar's row
    if (!(w > 0) || !(bw > 0)) return;
    var dpr = window.devicePixelRatio || 1, h = w * meta.H / meta.W;
    E.cv.forEach(function (c) { c.style.width = w + 'px'; c.style.height = h + 'px'; c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); });
    if (HAND) {                                                // stacked gloves (wide page): keep both hands on screen
      var asp = HAND.box.h / HAND.box.w, hw = w, hh = w * asp;
      if (window.innerWidth > 1100) { var cap = Math.max(200, (window.innerHeight - 320) / 2); if (hh > cap) { hh = cap; hw = hh / asp; } }
      E.hc.forEach(function (c) { c.style.width = hw + 'px'; c.style.height = hh + 'px'; c.style.margin = '0 auto'; c.width = Math.round(hw * dpr); c.height = Math.round(hh * dpr); });
    }
    var cb = E.cb; cb.style.width = bw + 'px'; cb.style.height = '10px'; cb.width = Math.round(bw * dpr); cb.height = Math.round(10 * dpr);
    var g = cb.getContext('2d');
    for (var x = 0; x < cb.width; x++) { g.fillStyle = LUT[Math.min(255, Math.round((x + 0.5) / cb.width * VMAX * 255))]; g.fillRect(x, 0, 1, cb.height); }
    sizeGen++; drawn = '';
    if (last >= 0) draw(last);
  }

  function cur() { return cache[mode === 'filled' ? want + '|f' : want]; }

  function draw(j) {
    var filled = mode === 'filled', d = cur(); if (!(d instanceof Uint8Array)) return;
    var P = meta.parts[want], S = meta.slots.length, k = mode === 'raw' ? 1 : 0, hand = view === 'hand' && handReady();
    for (var h = 0; h < 2; h++) {
      var cnt = 0, mx = 0, nf = 0, off = filled ? (h * P.n + j) * S : ((2 * h + k) * P.n + j) * S, s, qq;
      if (hand) {
        for (s = 0; s < S; s++) {
          if (ABS[h][s] || (DEAD[h][s] && (!filled || UNF[h][s]))) continue;
          qq = d[off + s]; if (DEAD[h][s] && qq > 0) nf++; if (k ? qq >= 26 : qq > 0) cnt++; if (qq > mx) mx = qq;
        }
        try { drawHand(h, d, off); } catch (e) { if (window.console) console.warn('touch panel: hand draw', e); }
      } else {
        var c = E.cv[h], g = c.getContext('2d'), cs = c.width / meta.W, r = cs * 0.42;
        g.clearRect(0, 0, c.width, c.height);
        for (s = 0; s < S; s++) {
          if (ABS[h][s]) continue;
          qq = d[off + s]; var slot = meta.slots[s];
          var x = (slot % meta.W + 0.5) * cs, y = (Math.floor(slot / meta.W) + 0.5) * cs;
          if (DEAD[h][s] && (!filled || UNF[h][s])) { g.strokeStyle = '#5b636e'; g.lineWidth = Math.max(1, cs * 0.12); g.beginPath(); g.arc(x, y, r * 0.75, 0, 2 * Math.PI); g.stroke(); continue; }
          var ring = DEAD[h][s] && markFilled;
          g.fillStyle = LUT[qq]; g.beginPath(); g.arc(x, y, ring ? r * 0.78 : r, 0, 2 * Math.PI); g.fill();
          if (ring) { g.strokeStyle = '#c9d1d9'; g.lineWidth = Math.max(1, cs * 0.08); g.beginPath(); g.arc(x, y, r * 0.98, 0, 2 * Math.PI); g.stroke(); }
          if (DEAD[h][s] && qq > 0) nf++;
          if (k ? qq >= 26 : qq > 0) cnt++;
          if (qq > mx) mx = qq;
        }
      }
      E.n[h].textContent = (k ? cnt + ' above 0.1' : (cnt ? cnt + ' in contact' : 'no contact')) + (nf ? ' (' + nf + ' filled)' : '') + (mx ? ' · max ' + (mx / 255).toFixed(2) : '');
    }
    E.time.textContent = 'take ' + (P.t0 + j / meta.fps).toFixed(1) + ' s';
    last = j;
  }

  function frameAt(t, playing) {
    var P = meta.parts[want], tau = playing ? t : Math.floor(t / 0.2 + 1e-6) * 0.2;
    return Math.max(0, Math.min(P.n - 1, Math.round(tau * meta.fps)));
  }

  function tick() {
    try {
      if (meta && want && cur() instanceof Uint8Array) {
        var S = window.__viserSync, c = S && (S.ctl(0) || S.ctl(1));
        var j = c ? frameAt(c.t, c.playing) : 0, key = want + '|' + mode + '|' + view + '|' + j + '|' + sizeGen;
        if (key !== drawn) { drawn = key; draw(j); }
      }
    } catch (e) { if (window.console) console.warn('touch panel', e); }
    requestAnimationFrame(tick);
  }

  function load(k) {
    if (cache[k]) return cache[k];
    var P = meta.parts[k];
    cache[k] = fetch(base + P.file).then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.arrayBuffer(); }).then(function (b) {
      var a = new Uint8Array(b); if (a.length !== 4 * P.n * meta.slots.length) throw new Error('size ' + a.length);
      cache[k] = a; drawn = ''; if (k === want) { last = 0; draw(0); } return a;
    }).catch(function (e) { delete cache[k]; if (E.time && k === want) E.time.textContent = 'touch data did not load (' + e.message + ')'; });
    return cache[k];
  }

  function loadF(k) {
    var key = k + '|f'; if (cache[key] || !meta.filled || !meta.filled.files[k]) return cache[key];
    var P = meta.parts[k];
    cache[key] = fetch(base + meta.filled.files[k]).then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.arrayBuffer(); }).then(function (b) {
      var a = new Uint8Array(b); if (a.length !== 2 * P.n * meta.slots.length) throw new Error('size ' + a.length);
      cache[key] = a; drawn = ''; if (k === want && mode === 'filled') { if (last < 0) last = 0; draw(last); } return a;
    }).catch(function (e) { delete cache[key]; if (E.time && k === want) E.time.textContent = 'filled taxels did not load (' + e.message + ')'; });
    return cache[key];
  }

  function fillNote() {
    var f = meta.filled, parts = [], empty = 0;
    ['left', 'right'].forEach(function (sd) {
      var I = f.info[sd]; empty += I.left_empty || 0;
      parts.push(I.dead ? 'the ' + sd + ' glove ' + I.filled + ' of its ' + I.dead + ' dead taxels (held-out check r ≈ ' + I.heldout_r_median_filled + ')' : 'the ' + sd + ' glove has no dead taxel');
    });
    return ' Filled on this take: ' + parts.join('; ') + '.' + (empty ? ' ' + empty + ' stay empty, mostly on dead rows: their held-out check stays below r = ' + f.r_min + ', so the neighbours do not predict them.' : '');
  }

  function setPart(k) {
    want = k; drawn = ''; last = -1;
    if (meta && meta.parts[k]) { if (!(cache[k] instanceof Uint8Array) && E.time) E.time.textContent = 'loading touch…'; load(k); if (mode === 'filled') loadF(k); if (cur() instanceof Uint8Array) draw(0); }
  }

  function setMode(m) {
    mode = m === 'raw' ? 'raw' : (m === 'filled' && meta && meta.filled ? 'filled' : 'filtered'); drawn = '';
    E.btn.forEach(function (b) { b.setAttribute('aria-pressed', String(b.dataset.tmode === mode)); });
    legend();
    if (mode === 'filled' && want) loadF(want);
    if (last >= 0) draw(last);
  }

  function mount(el, b) {
    root = el; base = b;
    return fetch(base + 'meta.json').then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); }).then(function (m) {
      meta = m; var pos = {}; meta.slots.forEach(function (sl, i) { pos[sl] = i; });
      CELL = new Int16Array(meta.slots.length); meta.slots.forEach(function (sl, i) { CELL[i] = Math.floor(sl / meta.W) * 32 + sl % meta.W; });
      DEAD = ['left', 'right'].map(function (sd) { var o = {}; ((meta.dead || {})[sd] || []).forEach(function (sl) { o[pos[sl]] = true; }); return o; });
      ABS = ['left', 'right'].map(function (sd) { var o = {}; ((meta.absent || {})[sd] || []).forEach(function (sl) { o[pos[sl]] = true; }); return o; });
      UNF = ['left', 'right'].map(function (sd) { var o = {}; (((meta.filled || {}).unfilled || {})[sd] || []).forEach(function (sl) { o[pos[sl]] = true; }); return o; });
      build(); size(); loadHand();
      if (want) setPart(want);
      setTimeout(function () { Object.keys(meta.parts).forEach(load); }, 4000);   // the other parts (about 0.6 MB each)
      requestAnimationFrame(tick);
    }).catch(function (e) { el.textContent = 'touch data not found (' + e.message + ')'; });
  }

  window.Touch = {
    mount: mount, setPart: setPart, setMode: setMode, setView: setView,
    renderAt: function (t, playing) { if (meta && cache[want] instanceof Uint8Array) { var j = frameAt(t, !!playing); draw(j); return j; } return -1; },
    state: function () { return { part: want, mode: mode, view: view, hand: handReady(), handErr: handErr, rot: rot.slice(), frame: last, loaded: Object.keys(cache).filter(function (k) { return cache[k] instanceof Uint8Array; }) }; }
  };
})();
