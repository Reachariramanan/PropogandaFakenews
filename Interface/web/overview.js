/* Overview page: key figures, highest-risk claims, 30-day trend, verdict mix, languages and a Propaganda Lab teaser.
   Reads /api/stats, /api/proplab/meta, /api/proplab/countries and /api/items only. Every figure comes from those responses;
   a panel whose source is unavailable shows a designed empty state instead of a guess. */
(function () {
  'use strict';
  var $ = function (s) { return document.querySelector(s); };
  var L = window.Labels, F = window.FNPUI, esc = L.esc, fmtN = F.fmtN, term = L.term;
  var rendered = false, trendData = null, trendTimer = null;

  function empty(title, text) { return '<div class="ov-empty"><b>' + esc(title) + '</b><span>' + esc(text) + '</span></div>'; }
  function pctOf(n, d) { return d ? (n / d * 100) : 0; }
  function put(sel, html) { var el = $(sel); if (el) el.innerHTML = html; }
  function groupBy(counts, nameOf) {
    var g = {}; Object.keys(counts || {}).forEach(function (c) { var n = nameOf(c); g[n] = (g[n] || 0) + counts[c]; });
    return Object.keys(g).map(function (k) { return { name: k, n: g[k] }; }).sort(function (a, b) { return b.n - a.n; });
  }

  /* ---------- KPI strip ---------- */
  /* line icons for the Command Center theme only (CSS hides .ico in the other themes) */
  var ICONS = {
    layers: '<path d="m12 3 9 5-9 5-9-5 9-5Z"/><path d="m3 12.5 9 5 9-5"/><path d="m3 17 9 5 9-5"/>',
    shield: '<path d="M12 3 4.5 6v5.5c0 4.6 3 7.9 7.5 9.5 4.5-1.6 7.5-4.9 7.5-9.5V6L12 3Z"/><path d="m8.8 12 2.3 2.3 4.2-4.6"/>',
    trans: '<path d="M4 6h9M8.5 4v2M6 6c.5 3.5 3 6 6.5 7.5M12 6c-.5 3-3 6-7 8"/><path d="m13 20 4-9 4 9M14.5 17h5"/>',
    warn: '<path d="M12 3.5 21.5 20h-19L12 3.5Z"/><path d="M12 10v4.5M12 17.2v.1"/>',
    db: '<ellipse cx="12" cy="6" rx="7.5" ry="3"/><path d="M4.5 6v6c0 1.7 3.4 3 7.5 3s7.5-1.3 7.5-3V6M4.5 12v6c0 1.7 3.4 3 7.5 3s7.5-1.3 7.5-3v-6"/>'
  };
  function kpi(label, value, sub, cls, icon) {
    return '<div class="kpi panel ' + (cls || '') + '"><span class="ico" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">' + (ICONS[icon] || '') + '</svg></span><span class="lbl">' + label + '</span><b>' + esc(value) + '</b><small>' + sub + '</small></div>';
  }
  function renderKpis(st, lab) {
    var h = '';
    if (st) {
      var rated = st.total - (st.verdicts.UNRATED || 0), fm = (st.verdicts.FAKE || 0) + (st.verdicts.MISLEADING || 0);
      var codes = Object.keys(st.sites), checkers = {}, social = {};
      codes.forEach(function (c) { (L.isSocial(c) ? social : checkers)[L.site(c)] = 1; });
      var langs = groupBy(st.languages, L.lang).filter(function (x) { return x.name !== 'Unknown'; });
      h += kpi('Claims indexed', fmtN(st.total), esc(fmtN(rated)) + ' rated fact-checks and ' + esc(fmtN(st.verdicts.UNRATED || 0)) + ' unrated social posts', '', 'layers');
      h += kpi('Fact-checkers covered', fmtN(Object.keys(checkers).length), 'plus ' + esc(fmtN(Object.keys(social).length)) + ' social platforms', '', 'shield');
      h += kpi('Languages', fmtN(langs.length), 'led by ' + esc(langs.slice(0, 2).map(function (x) { return x.name; }).join(' and ')), '', 'trans');
      h += kpi('Likely false or misleading', pctOf(fm, rated).toFixed(1) + '%', 'of rated fact-checks. The index is built from debunks, so this is not the share of all news.', 'v-FAKE', 'warn');
    } else h += kpi('Claims indexed', 'n/a', 'The fact-check index could not be loaded.', '', 'layers');
    if (lab) {
      var g = lab.global && lab.global.about && lab.global.about.all && lab.global.about.all.all;
      h += kpi('Posts analysed in the Lab', fmtN(lab.n_posts), (g && typeof g.rate === 'number' ? (g.rate * 100).toFixed(0) + '% carry a propaganda signal. ' : '') + (lab.n_articles > 0 ? '' : 'News source offline: social posts only.'), '', 'db');
    } else h += kpi('Posts analysed in the Lab', 'n/a', 'Propaganda Lab data is not built yet.', '', 'db');
    put('#ov-kpis', h);
  }

  /* ---------- verdict mix and languages ---------- */
  /* Command Center: the same shares as the stacked bar, drawn as a donut (circumference = 100 units, so every arc is to scale) */
  function donut(st, tot) {
    var vs = st.verdicts, R = 46, acc = 0, g = '<svg viewBox="0 0 120 120" aria-hidden="true" focusable="false"><circle class="dn-track" cx="60" cy="60" r="' + R + '"/>';
    F.VORDER.filter(function (v) { return vs[v]; }).forEach(function (v) {
      var len = pctOf(vs[v], tot);
      g += '<circle class="dn-' + v + '" data-verdict="' + v + '" cx="60" cy="60" r="' + R + '" pathLength="100" stroke-dasharray="' + len.toFixed(3) + ' ' + (100 - len).toFixed(3) + '" stroke-dashoffset="' + (-acc).toFixed(3) + '" transform="rotate(-90 60 60)"><title>' + esc(F.VNAME[v] + ': ' + fmtN(vs[v]) + ' (' + len.toFixed(1) + '%)') + '</title></circle>';
      acc += len;
    });
    return '<div class="mixdonut">' + g + '<text class="dn-total" x="60" y="60" text-anchor="middle">' + esc(fmtN(st.total)) + '</text><text class="dn-unit" x="60" y="75" text-anchor="middle">items</text></svg></div>';
  }
  function renderMix(st) {
    if (!st) { put('#ov-mix', empty('Verdict mix unavailable', 'The fact-check index could not be loaded.')); return; }
    var tot = st.total || 1, vs = st.verdicts;
    var bar = '<div class="mixbar" role="img" aria-label="Share of items by verdict">' + F.VORDER.filter(function (v) { return vs[v]; }).map(function (v) {
      return '<i class="v-' + v + '" style="width:' + pctOf(vs[v], tot).toFixed(2) + '%" title="' + esc(F.VNAME[v] + ': ' + fmtN(vs[v])) + '"></i>';
    }).join('') + '</div>';
    var rows = F.VORDER.filter(function (v) { return vs[v]; }).map(function (v) {
      var name = v === 'UNRATED' ? 'Unrated social posts' : F.VNAME[v];
      return '<button type="button" class="mix-row v-' + v + '" data-verdict="' + v + '"><span class="mix-k"><i></i>' + (L.glossaryEntry(v) ? '<span class="mix-name">' + esc(name) + '</span>' : esc(name)) + '</span><b>' + esc(fmtN(vs[v])) + '</b><small>' + pctOf(vs[v], tot).toFixed(1) + '%</small></button>';
    }).join('');
    put('#ov-mix', bar + donut(st, tot) + '<div class="mix-rows">' + rows + '</div>' +
      '<p class="note">Definitions: ' + ['FAKE', 'MISLEADING', 'INCONCLUSIVE', 'LEGIT'].map(function (v) { return term(v, L.verdict(v)); }).join(', ') + '.</p>');
  }
  function renderLangs(st) {
    if (!st) { put('#ov-lang', empty('Languages unavailable', 'The fact-check index could not be loaded.')); return; }
    var gs = groupBy(st.languages, L.lang), top = gs.slice(0, 6), max = top.length ? top[0].n : 1, tot = st.total || 1;
    put('#ov-lang', '<div class="bar-rows">' + top.map(function (x) {
      return '<button type="button" class="bar-row" data-lang="' + esc(x.name) + '"><span class="bar-k">' + esc(x.name) + '</span><span class="bar-track"><b style="width:' + pctOf(x.n, max).toFixed(1) + '%"></b></span><span class="bar-v">' + esc(fmtN(x.n)) + ' <small>' + pctOf(x.n, tot).toFixed(0) + '%</small></span></button>';
    }).join('') + '</div>' + (gs.length > 6 ? '<p class="note">and ' + (gs.length - 6) + ' more languages.</p>' : ''));
  }
  document.addEventListener('click', function (e) {
    var v = e.target.closest('#ov-mix [data-verdict]'), l = e.target.closest('#ov-lang [data-lang]');
    if (v) { window.Desk.filter('verdict', v.getAttribute('data-verdict')); Shell.navigate('#desk'); }
    else if (l) { window.Desk.filter('verdict', ''); window.Desk.filter('lang', l.getAttribute('data-lang')); Shell.navigate('#desk'); }
  });

  /* ---------- highest-risk claims ---------- */
  function ymd(d) { return d.toISOString().slice(0, 10); }
  function addDays(s, n) { var d = new Date(s.slice(0, 10) + 'T00:00:00Z'); d.setUTCDate(d.getUTCDate() + n); return d; }
  /* curated strip only: skip shouting, very short titles and unrated social posts */
  function readable(it) {
    var t = String(it.title || it.claim || '').trim();
    if (t.length < 25 || it.verdict_norm === 'UNRATED' || (it.kind === 'candidate' && !it.verdict_raw)) return false;
    var letters = t.replace(/[^A-Za-zÀ-ɏ]/g, '');
    if (letters.length >= 8 && letters.replace(/[^A-ZÀ-Þ]/g, '').length / letters.length > 0.6) return false;
    return true;
  }
  function pickRisk(items) {
    var seen = {}, out = [];
    items.forEach(function (it) {
      var k = String(it.title || it.claim).toLowerCase().slice(0, 60);
      if (out.length < 6 && readable(it) && !seen[k]) { seen[k] = 1; out.push(it); }
    });
    return out;
  }
  function riskTile(it) {
    var v = it.verdict_norm;
    return '<a class="risk s-' + esc(v) + '" href="#desk/' + encodeURIComponent(it.id) + '"><span class="risk-top">' + F.chip(v) + '<span class="risk-score" title="Plausibility score">' + (it.plaus == null ? 'n/a' : it.plaus.toFixed(2)) + '</span></span>' +
      '<span class="risk-title">' + esc(it.title || it.claim) + '</span><span class="risk-meta">' + esc(L.site(it.site)) + ' - ' + esc(L.dateLong(it.published)) + ' - ' + esc(L.lang(it.lang)) + '</span></a>';
  }
  function renderRisk(anchor) {
    if (!anchor) { put('#ov-risk', empty('No dated claims', 'Highest-risk claims appear here once the index has claims with a publication date.')); return; }
    var q = function (days) { return Shell.getJSON('/api/items?sort=plaus&verdict=FAKE,MISLEADING&page_size=80&from=' + ymd(addDays(anchor, -(days - 1)))); };
    q(7).then(function (d) {
      var picks = pickRisk(d.items), span = 7;
      if (picks.length >= 6) return { picks: picks, span: span };
      return q(30).then(function (d2) { return { picks: pickRisk(d2.items), span: 30 }; });
    }).then(function (r) {
      var to = L.dateLong(anchor), from = L.dateLong(addDays(anchor, -(r.span - 1)));
      $('#ov-risk-h').textContent = r.span === 7 ? 'Highest-risk claims this week' : 'Highest-risk claims, last 30 days';
      $('#ov-risk-sub').textContent = 'Top fact-checked claims by plausibility score among those published ' + from + ' to ' + to + '. Titles in capitals, very short titles and unrated social posts are left out of this strip.';
      put('#ov-risk', r.picks.length ? r.picks.map(riskTile).join('') : empty('Nothing to show', 'No readable fake or misleading claims were published in this period.'));
    }).catch(function (e) { put('#ov-risk', empty('Could not load claims', 'Reload the page to try again.')); });
  }

  /* ---------- 30-day trend by verdict: counted from /api/items by publication date, six 5-day periods ---------- */
  var GROUPS = [['FAKE', 'FAKE', 'Fake'], ['MISLEADING', 'MISLEADING', 'Misleading'], ['INCONCLUSIVE', 'INCONCLUSIVE,LEGIT', 'Inconclusive or legit']];
  function niceTop(max) { var mag = Math.pow(10, Math.floor(Math.log(max) / Math.LN10)), f = max / mag; return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10) * mag; }
  function loadTrend(anchor) {
    if (!anchor) { put('#ov-trend', empty('Trend unavailable', 'The index has no dated claims to count.')); return; }
    var bins = [], i, g, jobs = [];
    for (i = 0; i < 6; i++) bins.push({ from: ymd(addDays(anchor, -29 + i * 5)), to: ymd(addDays(anchor, -25 + i * 5)), v: {} });
    bins.forEach(function (b) {
      GROUPS.forEach(function (gr) {
        jobs.push(Shell.getJSON('/api/items?page_size=1&verdict=' + gr[1] + '&from=' + b.from + '&to=' + b.to).then(function (d) { b.v[gr[0]] = d.total; }));
      });
    });
    Promise.all(jobs).then(function () {
      trendData = { bins: bins, anchor: anchor };
      $('#ov-trend-sub').textContent = 'Fact-checks per 5-day period, ' + L.dateLong(bins[0].from) + ' to ' + L.dateLong(anchor) + ', counted from the index by publication date.';
      drawTrend();
    }).catch(function (e) { put('#ov-trend', empty('Trend could not be loaded', 'Reload the page to try again.')); });
  }
  function drawTrend(w) {
    if (!trendData) return;
    var el = $('#ov-trend'), bins = trendData.bins, W = w || Math.max(280, el.clientWidth || 360), H = 210, Lm = 40, R = 6, T = 14, B = 34, pw = W - Lm - R, ph = H - T - B;
    var tot = bins.map(function (b) { return GROUPS.reduce(function (a, g) { return a + (b.v[g[0]] || 0); }, 0); });
    var mx = Math.max.apply(null, tot);
    if (!(mx > 0)) { el.innerHTML = empty('No fact-checks in this period', 'Nothing was published in the last 30 days of data.'); return; }
    var top = niceTop(mx), y = function (v) { return T + ph - v / top * ph; }, bw = pw / bins.length, g = '';
    g += '<svg viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="Fact-checks per 5-day period by verdict, last 30 days">';
    [0, .5, 1].forEach(function (f) { var v = top * f; g += '<line class="grid-line" x1="' + Lm + '" x2="' + (Lm + pw) + '" y1="' + y(v).toFixed(1) + '" y2="' + y(v).toFixed(1) + '"/><text x="' + (Lm - 6) + '" y="' + (y(v) + 4).toFixed(1) + '" text-anchor="end">' + fmtN(v) + '</text>'; });
    bins.forEach(function (b, i) {
      var x = Lm + i * bw + bw * .16, w = bw * .68, acc = 0;
      GROUPS.forEach(function (gr) {
        var n = b.v[gr[0]] || 0; if (!n) return;
        g += '<rect class="f-' + gr[0] + '" x="' + x.toFixed(1) + '" y="' + y(acc + n).toFixed(1) + '" width="' + w.toFixed(1) + '" height="' + (n / top * ph).toFixed(1) + '"><title>' + esc(L.dateLong(b.from, true) + ' to ' + L.dateLong(b.to, true) + ': ' + fmtN(n) + ' ' + gr[2].toLowerCase()) + '</title></rect>';
        acc += n;
      });
      g += '<text x="' + (x + w / 2).toFixed(1) + '" y="' + (y(tot[i]) - 4).toFixed(1) + '" text-anchor="middle" class="t-strong">' + fmtN(tot[i]) + '</text>' +
        '<text x="' + (x + w / 2).toFixed(1) + '" y="' + (T + ph + 15) + '" text-anchor="middle">' + esc(L.dateLong(b.from, true)) + '</text>';
    });
    g += '<line class="axis" x1="' + Lm + '" x2="' + (Lm + pw) + '" y1="' + (T + ph) + '" y2="' + (T + ph) + '"/><text class="t-axis" x="' + (Lm + pw / 2) + '" y="' + (H - 4) + '" text-anchor="middle">Period starting</text></svg>';
    var gt = GROUPS.map(function (gr) { return bins.reduce(function (a, b) { return a + (b.v[gr[0]] || 0); }, 0); }), all = gt.reduce(function (a, n) { return a + n; }, 0) || 1;
    el.innerHTML = '<div class="chart">' + g + '</div><div class="legend">' + GROUPS.map(function (gr, i) { return '<span class="v-' + gr[0] + '"><i></i>' + esc(gr[2]) + '<em class="lg-pct">' + (gt[i] / all * 100).toFixed(1) + '%</em></span>'; }).join('') + '</div>';
    /* when the legend sits beside the chart (Command Center) the chart column is narrower than the panel: redraw once at its real width */
    var cw = el.querySelector('.chart').clientWidth;
    if (!w && cw > 0 && Math.abs(cw - W) > 6) drawTrend(Math.max(240, cw));
  }
  window.addEventListener('resize', function () { clearTimeout(trendTimer); trendTimer = setTimeout(function () { if (Shell.current().route === 'overview') drawTrend(); }, 150); });

  /* ---------- Command Center only: faint dotted world map behind the page heading ----------
     Built once from the Lab's country outlines (/api/proplab/world): the outlines are painted on an off-screen canvas, sampled on a grid, and kept as dots. */
  var mapState = 0; /* 0 = not built, 1 = building, 2 = done */
  function buildDots(w) {
    var vb = String(w.viewBox || '-426 -213 859 381').split(/\s+/).map(Number), X0 = vb[0], Y0 = vb[1], VW = vb[2], VH = vb[3], step = 8;
    var cv = document.createElement('canvas'); cv.width = Math.ceil(VW); cv.height = Math.ceil(VH);
    var cx = cv.getContext('2d'); if (!cx || typeof Path2D !== 'function') return '';
    cx.translate(-X0, -Y0);
    Object.keys(w.countries).forEach(function (k) { var c = w.countries[k]; if (c && c.d) { try { cx.fill(new Path2D(c.d)); } catch (e) {} } });
    var data = cx.getImageData(0, 0, cv.width, cv.height).data, d = '';
    for (var y = step / 2; y < VH; y += step) {
      if (Y0 + y > 125) continue; /* leave Antarctica out */
      for (var x = step / 2; x < VW; x += step) { if (data[((Math.floor(y) * cv.width) + Math.floor(x)) * 4 + 3] > 100) d += 'M' + (X0 + x).toFixed(0) + ' ' + (Y0 + y).toFixed(0) + 'h.01'; }
    }
    return d ? '<svg viewBox="' + vb.join(' ') + '" preserveAspectRatio="xMaxYMid meet" focusable="false"><path d="' + d + '"/></svg>' : '';
  }
  function ensureMap() {
    var host = $('#ov-map'); if (!host || mapState || document.documentElement.getAttribute('data-theme') !== 'command') return;
    mapState = 1;
    Shell.getJSON('/api/proplab/world').then(function (w) { host.innerHTML = w && w.countries ? buildDots(w) : ''; mapState = 2; }).catch(function () { mapState = 0; });
  }
  document.addEventListener('fnp-theme', function () {
    if (Shell.current().route !== 'overview') return;
    ensureMap(); drawTrend();
  });

  /* ---------- Propaganda Lab teaser ---------- */
  function renderLab(lab) {
    if (!lab) { put('#ov-lab', empty('Propaganda Lab not built yet', 'This panel fills in when the aggregate build is available on this machine.')); return; }
    var min = lab.min_n || 30;
    Shell.getJSON('/api/proplab/countries').then(function (d) {
      var st = (d.countries && d.countries.about && d.countries.about.all && d.countries.about.all.all) || {};
      var rows = Object.keys(st).filter(function (k) { var s = st[k]; return k !== (lab.unlocated_key || 'UNL') && s && s.n >= min && !s.insufficient && typeof s.lift === 'number'; })
        .map(function (k) { return { iso: k, s: st[k] }; }).sort(function (a, b) { return b.s.lift - a.s.lift || b.s.n - a.s.n; }).slice(0, 5);
      if (!rows.length) { put('#ov-lab', empty('No country has enough posts yet', 'Countries need at least ' + min + ' posts to be ranked.')); return; }
      var mx = Math.max.apply(null, rows.map(function (r) { return r.s.lift; })) || 1;
      var h = '<div class="lab-head"><span>Country</span><span aria-hidden="true"></span><span>' + term('lift', 'Lift') + '</span><span>' + term('rate', 'Rate') + '</span><span>Posts</span></div><ol class="lab-rows">' + rows.map(function (r) {
        return '<li><a href="#lab/' + esc(r.iso) + '"><span class="lab-c">' + esc((lab.names && lab.names[r.iso]) || r.iso) + '</span><span class="lab-bar"><b style="width:' + (r.s.lift / mx * 100).toFixed(1) + '%"></b></span><span class="lab-n lab-lift">' + r.s.lift.toFixed(2) + '</span><span class="lab-n">' + (r.s.rate * 100).toFixed(0) + '%</span><span class="lab-n">' + esc(fmtN(r.s.n)) + '</span></a></li>';
      }).join('') + '</ol>';
      h += '<p class="note">Countries named in posts, all time, social posts; only countries with at least ' + term('min_n', esc(min) + ' posts') + '. A signal for triage, not a verdict.' + (lab.n_articles > 0 ? '' : ' News source offline: social posts only.') + '</p>';
      h += '<p class="ov-more"><a href="#lab">Open the Propaganda Lab</a></p>';
      put('#ov-lab', h);
    }).catch(function (e) { put('#ov-lab', empty('Country figures could not be loaded', 'Reload the page to try again.')); });
  }

  function render() {
    Promise.all([Shell.getStats().catch(function () { return null; }), Shell.getLabMeta(), Shell.getLatest()]).then(function (r) {
      var st = r[0], lab = r[1], latest = r[2];
      renderKpis(st, lab); renderMix(st); renderLangs(st); renderLab(lab);
      renderRisk(latest); loadTrend(latest);
    });
  }
  Shell.register('overview', { enter: function () { ensureMap(); if (!rendered) { rendered = true; render(); } else drawTrend(); } });
})();
