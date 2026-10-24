/* On-video lane editor (lives on the Dashboard, no separate page).
   Pure helpers are unit-tested in node; UI attaches to the dashboard's live-feed card. */
(function (root) {
  function unit(a, b) { const dx = b[0] - a[0], dy = b[1] - a[1], n = Math.hypot(dx, dy) || 1; return [+(dx / n).toFixed(3), +(dy / n).toFixed(3)]; }
  function affine(M, p) { return [M[0][0] * p[0] + M[0][1] * p[1] + M[0][2], M[1][0] * p[0] + M[1][1] * p[1] + M[1][2]]; }
  function invert(M) {
    const [a, b, tx] = M[0], [c, d, ty] = M[1], det = a * d - b * c || 1;
    return [[d / det, -b / det, (b * ty - d * tx) / det], [-c / det, a / det, (c * tx - a * ty) / det]];
  }
  function mapState(S, fn) {
    const m = P => P.map(fn);
    return Object.assign({}, S, { lanes: S.lanes.map(L => Object.assign({}, L, { polygon: m(L.polygon), stop: m(L.stop || []), dir: m(L.dir || []) })),
      nopark: S.nopark.map(m), restricted: S.restricted.map(m) });
  }

  function buildConfig(S) {
    const n = p => [+(p[0] / S.W).toFixed(4), +(p[1] / S.H).toFixed(4)];
    const cfg = { signalized: !!S.signalized, stabilize: true, speed_limit_kmh: Number(S.limit) || 40,
      meters_per_pixel: S.mpp || 0.05, lanes: {},
      no_parking_zones: S.nopark.map(z => z.map(n)), restricted_zones: S.restricted.map(z => z.map(n)) };
    if (S.persp) cfg.perspective = S.persp;
    if (S.north && !(S.north[0] === 0 && S.north[1] === -1)) cfg.north_direction = S.north;
    S.lanes.forEach(L => {
      const e = { polygon: L.polygon.map(n), num_lanes: L.n || 1 };
      if (L.dir && L.dir.length === 2) e.direction = unit(L.dir[0], L.dir[1]);      // optional: engine learns it if absent
      if (L.stop && L.stop.length === 2) e.stop_line = L.stop.map(n);
      cfg.lanes[L.name] = e;
    });
    return cfg;
  }

  function fromConfig(cfg, W, H) {
    const p = q => [q[0] * W, q[1] * H];
    const S = { W, H, lanes: [], nopark: (cfg.no_parking_zones || []).map(z => z.map(p)),
      restricted: (cfg.restricted_zones || []).map(z => z.map(p)), mpp: cfg.meters_per_pixel || 0.05,
      signalized: cfg.signalized !== false, limit: cfg.speed_limit_kmh || 40, persp: cfg.perspective || null,
      north: cfg.north_direction || [0, -1] };
    Object.entries(cfg.lanes || {}).forEach(([name, L]) => {
      const poly = L.polygon.map(p);
      const c = poly.reduce((a, q) => [a[0] + q[0] / poly.length, a[1] + q[1] / poly.length], [0, 0]);
      const d = L.direction;
      S.lanes.push({ name, n: L.num_lanes || 1, polygon: poly, stop: L.stop_line ? L.stop_line.map(p) : [],
        dir: d ? [c, [c[0] + d[0] * 70, c[1] + d[1] * 70]] : [] });
    });
    return S;
  }

  function initUI() {
    const $ = id => document.getElementById(id);
    if (!$('cv')) return;
    const cv = $('cv'), ctx = cv.getContext('2d'), img = new Image();
    const COL = ['#3b82f6', '#22c55e', '#ef4444', '#eab308', '#a855f7', '#06b6d4'];
    let S = { W: cv.width, H: cv.height, lanes: [], nopark: [], restricted: [], mpp: 0.05, signalized: true, limit: 40, persp: null, north: [0, -1] };
    let tool = null, pts = [], cur = null, M = [[1, 0, 0], [0, 1, 0]];

    const hint = t => { $('hint').textContent = t; };
    function poly(P, col, close) {
      if (!P.length) return;
      ctx.strokeStyle = col; ctx.fillStyle = col + '33'; ctx.lineWidth = 2; ctx.beginPath();
      P.forEach((q, i) => i ? ctx.lineTo(q[0], q[1]) : ctx.moveTo(q[0], q[1]));
      if (close && P.length > 2) { ctx.closePath(); ctx.fill(); }
      ctx.stroke();
      P.forEach(q => { ctx.beginPath(); ctx.arc(q[0], q[1], 4, 0, 7); ctx.fillStyle = col; ctx.fill(); });
    }
    function draw() {
      ctx.clearRect(0, 0, cv.width, cv.height);
      if (img.complete && img.naturalWidth) ctx.drawImage(img, 0, 0, cv.width, cv.height);
      S.lanes.forEach((L, i) => {
        poly(L.polygon, COL[i % 6], true);
        if (L.stop.length === 2) { ctx.strokeStyle = '#fff'; ctx.lineWidth = 5; ctx.beginPath(); ctx.moveTo(L.stop[0][0], L.stop[0][1]); ctx.lineTo(L.stop[1][0], L.stop[1][1]); ctx.stroke(); }
        ctx.fillStyle = '#fff'; ctx.strokeStyle = '#000'; ctx.lineWidth = 3; ctx.font = 'bold 22px sans-serif';
        const c = L.polygon.reduce((a, q) => [a[0] + q[0] / L.polygon.length, a[1] + q[1] / L.polygon.length], [0, 0]);
        ctx.strokeText(L.name, c[0] - 30, c[1]); ctx.fillText(L.name, c[0] - 30, c[1]);
      });
      S.nopark.forEach(z => poly(z, '#f97316', true)); S.restricted.forEach(z => poly(z, '#dc2626', true));
      if (cur && cur.polygon.length) poly(cur.polygon, '#ffffff', true);
      poly(pts, '#ffff00', false);
    }
    function lanesList() {
      $('lanelist').innerHTML = S.lanes.map((L, i) => `<span class="tag" style="background:${COL[i % 6]}">${L.name} ×${L.n}
        <a href="#" onclick="zren(${i});return false" style="color:#fff" title="rename">✎</a>
        <a href="#" onclick="zdel(${i});return false" style="color:#fff" title="delete">✕</a></span> `).join('') || '<span class="muted">No lanes yet</span>';
    }
    root.zdel = i => { S.lanes.splice(i, 1); lanesList(); draw(); };
    root.zren = i => { const n = prompt('New name (e.g. North, South, East, West):', S.lanes[i].name); if (n && n.trim()) { S.lanes[i].name = n.trim(); lanesList(); draw(); } };

    function startLane(name) {
      const n = parseInt(prompt('How many real lanes inside this area?', '1')) || 1;
      cur = { name, n, polygon: [], stop: [], dir: [] }; pts = []; tool = 'lane_poly';
      hint(`"${name}": click the corners of its area on the video, then press "Finish shape"`); draw();
    }
    function finishTool() {
      if (tool === 'lane_poly') {
        if (pts.length < 3) return hint('Need at least 3 points for the lane area');
        cur.polygon = pts; pts = []; tool = 'lane_stop'; hint('Click 2 points on the STOP LINE, or press "Skip stop line" if there is no signal');
      } else if (tool === 'nopark' || tool === 'restricted') {
        if (pts.length < 3) return hint('Need at least 3 points');
        (tool === 'nopark' ? S.nopark : S.restricted).push(pts); pts = []; tool = null; hint('Zone added');
      }
      draw();
    }
    function addCur() { S.lanes.push(cur); cur = null; tool = null; pts = []; hint('Lane added. Add another, or press Save.'); lanesList(); }
    function click(x, y) {
      if (!tool) return;
      pts.push([x, y]);
      if (tool === 'lane_stop' && pts.length === 2) { cur.stop = pts; addCur(); }
      else if (tool === 'scale' && pts.length === 2) {
        const m = parseFloat(prompt('Real distance between the two points in metres (e.g. lane width 3.5):', '3.5'));
        const d = Math.hypot(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]);
        if (m > 0 && d > 0) { S.mpp = +(m / d).toFixed(5); S.persp = null; hint('Scale set: ' + S.mpp + ' m/pixel'); }
        pts = []; tool = null;
      }
      draw();
    }

    async function loadCurrent() {
      const cfg = await (await fetch('/api/zones/config')).json();
      const inv = invert(M);
      S = mapState(fromConfig(cfg, cv.width, cv.height), p => affine(inv, p));      // saved zones are in reference coords
      S.W = cv.width; S.H = cv.height;
      $('sig').checked = S.signalized; $('limit').value = S.limit; $('north').value = JSON.stringify(S.north);
      lanesList(); draw();
    }
    async function open() {
      const r = await fetch('/api/live/snapshot');
      if (!r.ok) { alert('Start a video or camera first, then edit lanes on it.'); return; }
      const j = await r.json(); M = j.M;
      $('feedimg').style.display = 'none'; cv.style.display = 'block'; $('editbar').style.display = 'block';
      img.onload = async () => { cv.height = Math.round(cv.width * img.naturalHeight / img.naturalWidth); S.H = cv.height; S.W = cv.width; await loadCurrent(); hint('Frame frozen. Rename lanes with ✎, delete with ✕, or add new lanes. The North/East/South/West buttons name a lane by direction.'); };
      img.src = j.image;
    }
    function close() { cv.style.display = 'none'; $('editbar').style.display = 'none'; $('feedimg').style.display = 'block'; tool = null; pts = []; cur = null; }

    cv.addEventListener('click', e => { const r = cv.getBoundingClientRect(); click((e.clientX - r.left) * cv.width / r.width, (e.clientY - r.top) * cv.height / r.height); });
    $('b_edit').onclick = open;
    $('b_cancel').onclick = close;
    $('b_lane').onclick = () => { const name = prompt('Lane name:'); if (name && name.trim()) startLane(name.trim()); };
    ['North', 'East', 'South', 'West'].forEach(d => { $('b_' + d[0].toLowerCase()).onclick = () => startLane(d); });
    $('b_nopark').onclick = () => { tool = 'nopark'; pts = []; hint('Click corners of the NO-PARKING zone, then "Finish shape"'); };
    $('b_restr').onclick = () => { tool = 'restricted'; pts = []; hint('Click corners of the RESTRICTED zone (divider / bus lane), then "Finish shape"'); };
    $('b_scale').onclick = () => { tool = 'scale'; pts = []; hint('Click 2 points a known distance apart (e.g. across one lane)'); };
    $('b_finish').onclick = finishTool;
    $('b_skip').onclick = () => { if (tool === 'lane_stop') addCur(); };
    $('b_undo').onclick = () => { pts.pop(); draw(); };
    $('b_clear').onclick = () => { if (confirm('Remove all zones?')) { S.lanes = []; S.nopark = []; S.restricted = []; pts = []; cur = null; tool = null; lanesList(); draw(); } };
    $('sig').onchange = e => { S.signalized = e.target.checked; };
    $('limit').onchange = e => { S.limit = +e.target.value; };
    $('north').onchange = e => { S.north = JSON.parse(e.target.value); };
    $('b_save').onclick = async () => {
      S.limit = +$('limit').value || 40; S.signalized = $('sig').checked; S.north = JSON.parse($('north').value);
      const ref = mapState(S, p => affine(M, p));                                    // back to reference coordinates
      const r = await fetch('/api/zones/config', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(buildConfig(ref)) });
      const j = await r.json();
      if (j.ok) { close(); const m = $('editmsg'); m.textContent = 'Saved - applied to the live video.'; setTimeout(() => m.textContent = '', 5000); }
      else hint('Fix: ' + j.errors.join('; '));
    };
    lanesList();
  }

  if (typeof module !== 'undefined') module.exports = { buildConfig, fromConfig, affine, invert, mapState };
  if (typeof document !== 'undefined') document.addEventListener('DOMContentLoaded', initUI);
})(typeof window !== 'undefined' ? window : globalThis);
