(function () {
  'use strict';
  var $ = function (s, r) { return (r || document).querySelector(s); };
  var esc = function (s) { return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) { return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]; }); };
  var VORDER = ['FAKE', 'MISLEADING', 'INCONCLUSIVE', 'LEGIT', 'UNRATED'];
  var VNAME = {}; VORDER.forEach(function (v) { VNAME[v] = Labels.verdict(v); });
  var SN = Labels.site, LN = Labels.lang, term = Labels.term;
  var state = { stats: null, tab: 'desk' };

  function api(path, opts) {
    return fetch(path, opts).then(function (r) {
      return r.json().then(function (j) { if (!r.ok) throw new Error(j.error || r.status); return j; });
    });
  }
  function fmtDate(s) { return s ? Labels.dateLong(s) : 'undated'; }
  function fmtN(n) { return Number(n || 0).toLocaleString('en-US'); }
  function pct(p) { return p == null ? 'n/a' : p.toFixed(2); }
  function band(p) { return p >= 0.8 ? 'hi' : p >= 0.5 ? 'mid' : 'lo'; }
  function chip(v) { return '<span class="chip v-' + esc(v) + '">' + esc(VNAME[v] || v) + '</span>'; }
  /* bar to scale 0..1 with ticks at 0.5 and 0.8, the cut points that set the colour */
  function meter(p) {
    if (p == null) return '<span class="score"><span class="score-na">no score</span></span>';
    return '<span class="score"><span class="meter ' + band(p) + '" role="img" aria-label="Plausibility ' + p.toFixed(2) + '"><b style="width:' + (p * 100).toFixed(1) + '%"></b><u style="left:50%"></u><u style="left:80%"></u></span><span class="score-num">' + p.toFixed(2) + '</span></span>';
  }
  function nSources(st) { var m = {}; Object.keys(st.sites).forEach(function (c) { m[SN(c)] = 1; }); return Object.keys(m).length; }
  function safeUrl(u) { return /^https?:\/\//i.test(u || '') ? u : ''; }


  /* ---------- routing: the shell owns the hash. The Desk, Claim Check and Overview register themselves in their own files ---------- */
  Shell.register('sources', { enter: function () { renderSources(); } });
  Shell.register('lab', {
    enter: function (iso) { if (window.PropLab) { window.PropLab.setActive(true); window.PropLab.select(iso || null); } },
    leave: function () { if (window.PropLab) window.PropLab.setActive(false); }
  });
  Shell.on(function (r) { state.tab = r.route; });


  /* ---------- sources / charts: axes are linear, ticks rounded to 1/2/5 steps, text styled by CSS tokens ---------- */
  function nice(max, n) {
    max = max || 1;
    var raw = max / n, mag = Math.pow(10, Math.floor(Math.log(raw) / Math.LN10)), f = raw / mag;
    var step = (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10) * mag;
    return { step: step, top: Math.ceil(max / step) * step };
  }
  function ticks(sc) { var a = []; for (var v = 0; v <= sc.top + sc.step / 1000; v += sc.step) a.push(v); return a; }
  function tl(v) { return v >= 10000 ? (v / 1000) + 'k' : fmtN(v); }

  function chartSites(st) {
    var W = Math.max(300, $('#ch-sites').clientWidth || 320);
    var byName = {}, mixByName = {};
    Object.keys(st.sites).forEach(function (c) {
      var n = SN(c); byName[n] = (byName[n] || 0) + st.sites[c]; mixByName[n] = mixByName[n] || {};
      var mx = st.site_verdict_mix[c] || {}; Object.keys(mx).forEach(function (v) { mixByName[n][v] = (mixByName[n][v] || 0) + mx[v]; });
    });
    var sites = Object.keys(byName).sort(function (a, b) { return byName[b] - byName[a]; });
    var maxv = Math.max.apply(null, sites.map(function (s) { return byName[s]; })) || 1;
    var labW = Math.min(150, Math.round(W * 0.34)), rowH = 28, top = 6, bot = 38, R = 44;
    var plotW = W - labW - R, sc = nice(maxv, W < 500 ? 3 : 5), x = function (v) { return labW + v / sc.top * plotW; };
    var H = top + sites.length * rowH + bot, ay = top + sites.length * rowH;
    var s = '<svg viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="Items per source by verdict">';
    ticks(sc).forEach(function (v) {
      s += '<line class="grid-line" x1="' + x(v).toFixed(1) + '" y1="' + top + '" x2="' + x(v).toFixed(1) + '" y2="' + ay + '"/>' +
        '<text x="' + x(v).toFixed(1) + '" y="' + (ay + 16) + '" text-anchor="middle">' + tl(v) + '</text>';
    });
    sites.forEach(function (site, i) {
      var y = top + i * rowH + 4, cx = labW, mix = mixByName[site] || {};
      s += '<text class="t-strong" x="' + (labW - 8) + '" y="' + (y + 14) + '" text-anchor="end"><title>' + esc(site) + '</title>' + esc(site.length > 22 ? site.slice(0, 21) + '…' : site) + '</text>';
      VORDER.forEach(function (v) {
        var n = mix[v] || 0; if (!n) return;
        var w = Math.max(1.5, n / sc.top * plotW);
        s += '<rect class="f-' + v + '" x="' + cx.toFixed(1) + '" y="' + y + '" width="' + w.toFixed(1) + '" height="19"><title>' + esc(site + ': ' + fmtN(n) + ' ' + VNAME[v]) + '</title></rect>';
        cx += w;
      });
      s += '<text x="' + (cx + 6).toFixed(1) + '" y="' + (y + 14) + '">' + fmtN(byName[site]) + '</text>';
    });
    s += '<line class="axis" x1="' + labW + '" y1="' + top + '" x2="' + labW + '" y2="' + ay + '"/><line class="axis" x1="' + labW + '" y1="' + ay + '" x2="' + (labW + plotW) + '" y2="' + ay + '"/>' +
      '<text class="t-axis" x="' + (labW + plotW / 2) + '" y="' + (H - 4) + '" text-anchor="middle">Items</text></svg>';
    $('#ch-sites').innerHTML = s;
    $('#lg-verdict').innerHTML = VORDER.filter(function (v) { return st.verdicts[v]; }).map(function (v) {
      return '<span class="v-' + v + '"><i></i>' + esc(VNAME[v]) + ' ' + fmtN(st.verdicts[v]) + '</span>';
    }).join('');
  }

  function chartHist(st) {
    var hh = st.score_histogram, W = Math.max(280, $('#ch-hist').clientWidth || 300), H = 230;
    var L = 44, R = 8, T = 12, B = 46, pw = W - L - R, ph = H - T - B;
    var mh = Math.max.apply(null, hh.map(function (b) { return b.count; })) || 1, sc = nice(mh, 4);
    var y = function (v) { return T + ph - v / sc.top * ph; }, bw = pw / hh.length, x = function (v) { return L + v * pw; };
    var g = '<svg viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="Plausibility score histogram, items per 0.1 band">';
    ticks(sc).forEach(function (v) {
      g += '<line class="grid-line" x1="' + L + '" y1="' + y(v).toFixed(1) + '" x2="' + (L + pw) + '" y2="' + y(v).toFixed(1) + '"/>' +
        '<text x="' + (L - 6) + '" y="' + (y(v) + 4).toFixed(1) + '" text-anchor="end">' + tl(v) + '</text>';
    });
    hh.forEach(function (b, i) {
      var h = b.count / sc.top * ph, bx = L + i * bw, c = band(b.lo);
      g += '<rect class="f-' + c + '" x="' + (bx + 1.5).toFixed(1) + '" y="' + (T + ph - h).toFixed(1) + '" width="' + Math.max(1, bw - 3).toFixed(1) + '" height="' + h.toFixed(1) + '"><title>' + b.lo.toFixed(1) + ' to ' + b.hi.toFixed(1) + ': ' + fmtN(b.count) + ' items</title></rect>';
      if (b.count) g += '<text x="' + (bx + bw / 2).toFixed(1) + '" y="' + (T + ph - h - 4).toFixed(1) + '" text-anchor="middle">' + fmtN(b.count) + '</text>';
    });
    var every = bw < 30 ? 2 : 1;
    for (var i = 0; i <= hh.length; i += every) g += '<text x="' + (L + i * bw).toFixed(1) + '" y="' + (T + ph + 16) + '" text-anchor="middle">' + (i / hh.length).toFixed(1) + '</text>';
    g += '<line class="thr" x1="' + x(0.5).toFixed(1) + '" y1="' + T + '" x2="' + x(0.5).toFixed(1) + '" y2="' + (T + ph) + '"/><line class="thr" x1="' + x(0.8).toFixed(1) + '" y1="' + T + '" x2="' + x(0.8).toFixed(1) + '" y2="' + (T + ph) + '"/>' +
      '<line class="axis" x1="' + L + '" y1="' + (T + ph) + '" x2="' + (L + pw) + '" y2="' + (T + ph) + '"/><line class="axis" x1="' + L + '" y1="' + T + '" x2="' + L + '" y2="' + (T + ph) + '"/>' +
      '<text class="t-axis" x="' + (L + pw / 2) + '" y="' + (H - 4) + '" text-anchor="middle">Plausibility score</text>' +
      '<text class="t-axis" transform="translate(11 ' + (T + ph / 2) + ') rotate(-90)" text-anchor="middle">Items</text></svg>';
    $('#ch-hist').innerHTML = g + (st.unscored ? '<p class="note">' + fmtN(st.unscored) + ' unrated posts have no score and are not shown.</p>' : '');
    $('#lg-hist').innerHTML = '<span><i class="v-INCONCLUSIVE"></i>below 0.5</span><span><i class="v-MISLEADING"></i>0.5 to 0.8</span><span><i class="v-FAKE"></i>0.8 and above</span>';
  }

  function chartMonths(st) {
    var keys = Object.keys(st.months).sort(); if (!keys.length) { $('#ch-months').innerHTML = '<p class="note">No dated items.</p>'; return; }
    var mi = function (k) { return +k.slice(0, 4) * 12 + (+k.slice(5, 7) - 1); };
    var m0 = mi(keys[0]), m1 = mi(keys[keys.length - 1]), N = m1 - m0 + 1;
    var W = Math.max(280, $('#ch-months').clientWidth || 300), H = 230, L = 44, R = 8, T = 12, B = 46, pw = W - L - R, ph = H - T - B;
    var mx = Math.max.apply(null, keys.map(function (k) { return st.months[k]; })) || 1, sc = nice(mx, 4);
    var y = function (v) { return T + ph - v / sc.top * ph; }, bw = pw / N, x = function (m) { return L + (m - m0) * bw; };
    var g = '<svg viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="Items per publication month">';
    ticks(sc).forEach(function (v) {
      g += '<line class="grid-line" x1="' + L + '" y1="' + y(v).toFixed(1) + '" x2="' + (L + pw) + '" y2="' + y(v).toFixed(1) + '"/>' +
        '<text x="' + (L - 6) + '" y="' + (y(v) + 4).toFixed(1) + '" text-anchor="end">' + tl(v) + '</text>';
    });
    keys.forEach(function (k) {
      var n = st.months[k], h = n / sc.top * ph;
      g += '<rect class="f-accent" x="' + x(mi(k)).toFixed(2) + '" y="' + (T + ph - h).toFixed(1) + '" width="' + Math.max(1, bw - 0.6).toFixed(2) + '" height="' + h.toFixed(1) + '"><title>' + esc(k + ': ' + fmtN(n) + ' items') + '</title></rect>';
    });
    var y0 = Math.floor(m0 / 12), y1 = Math.floor(m1 / 12), pxYear = bw * 12, ystep = Math.max(1, Math.ceil(46 / pxYear));
    for (var yr = y0; yr <= y1 + 1; yr++) {
      var mpos = yr * 12; if (mpos < m0 || mpos > m1 + 1) continue;
      if ((yr - y0) % ystep !== 0) continue;
      g += '<line class="axis" x1="' + x(mpos).toFixed(1) + '" y1="' + (T + ph) + '" x2="' + x(mpos).toFixed(1) + '" y2="' + (T + ph + 4) + '"/>' +
        '<text x="' + x(mpos).toFixed(1) + '" y="' + (T + ph + 16) + '" text-anchor="middle">' + yr + '</text>';
    }
    g += '<line class="axis" x1="' + L + '" y1="' + (T + ph) + '" x2="' + (L + pw) + '" y2="' + (T + ph) + '"/><line class="axis" x1="' + L + '" y1="' + T + '" x2="' + L + '" y2="' + (T + ph) + '"/>' +
      '<text class="t-axis" x="' + (L + pw / 2) + '" y="' + (H - 4) + '" text-anchor="middle">Publication month</text>' +
      '<text class="t-axis" transform="translate(11 ' + (T + ph / 2) + ') rotate(-90)" text-anchor="middle">Items</text></svg>';
    $('#ch-months').innerHTML = g;
  }

  function renderSources() {
    var st = state.stats; if (!st) return;
    var rated = st.meta.n_factchecks || 0, fm = (st.verdicts.FAKE || 0) + (st.verdicts.MISLEADING || 0);
    var tiles = [
      [st.total, 'Items indexed', 'all sources, rated and unrated', ''],
      [rated, 'Fact-checks', 'rated by a fact-checking site', ''],
      [st.meta.n_candidates, 'Unrated social posts', 'candidates, scored by the engine', 'v-UNRATED'],
      [nSources(st), 'Sources', 'distinct sites and platforms', ''],
      [fm, 'Fake or misleading', (st.total ? (fm / st.total * 100).toFixed(1) : '0') + '% of all items', 'v-FAKE']
    ];
    $('#tiles').innerHTML = tiles.map(function (t) {
      return '<div class="panel tile ' + t[3] + '"><span class="lbl">' + esc(t[1]) + '</span><b>' + fmtN(t[0]) + '</b><small>' + esc(t[2]) + '</small></div>';
    }).join('');
    chartSites(st); chartHist(st); chartMonths(st);
  }
  var rt; window.addEventListener('resize', function () { clearTimeout(rt); rt = setTimeout(function () { if (state.tab === 'sources') renderSources(); }, 150); });

  /* ---------- boot: nav counts and the engine banner; the Desk fills its own filters ---------- */
  Shell.getStats().then(function (st) {
    state.stats = st;
    var eng = st.meta.engine || {};
    $('#nc-gallery').textContent = fmtN(st.total);
    $('#nc-sources').textContent = fmtN(nSources(st));
    if (!eng.available) {
      var b = $('#banner'); b.hidden = false;
      b.textContent = 'Engine scores pending. Plausibility currently comes from fact-checker labels only; it will blend in the verdict engine after the next rebuild.';
    }
    if (state.tab === 'sources') renderSources();
  }).catch(function (e) { var b = $('#banner'); b.hidden = false; b.textContent = 'The fact-check index could not be loaded. Reload the page to try again.'; });

  /* ---------- helpers other scripts use (case.js, desk.js, claimcheck.js, overview.js) ---------- */
  window.FNPUI = { esc: esc, fmtN: fmtN, chip: chip, meter: meter, band: band, safeUrl: safeUrl, api: api, VNAME: VNAME, VORDER: VORDER };
})();
