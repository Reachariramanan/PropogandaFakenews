/* Propaganda Lab tab. Reads /api/proplab/* only (read-only, same origin, no external requests).
   Every field may be missing or null: render '-' or hide, never throw. */
(function () {
  'use strict';
  var $ = function (s, r) { return (r || document).querySelector(s); };
  var esc = function (s) { return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]; }); };
  var API = '/api/proplab/';
  var SVGNS = 'http://www.w3.org/2000/svg';
  var UNL = 'UNL';
  var S = {
    active: false, loaded: false, loading: false, meta: null, world: null, countries: null, heat: null, built: null,
    detail: {}, dB: {}, view: 'map', mode: 'about', layer: 'signal', metric: 'lift', tech: 0, win: 'all', src: 'all',
    sel: null, ctab: 'overview', sort: null, find: '', rmax: 1, roving: null, centers: null,
    posts: { key: '', items: [], next: 0, more: false, loading: false, err: '', fT: '', sigOnly: false, auto: 0 }
  };

  /* ---------- helpers ---------- */
  function isNum(v) { return typeof v === 'number' && isFinite(v); }
  function fx(v, d) { return isNum(v) ? v.toFixed(d == null ? 2 : d) : '-'; }
  function fn(v) { return isNum(v) ? v.toLocaleString('en-US') : '-'; }
  function fp(v, d) { return isNum(v) ? (v * 100).toFixed(d == null ? 1 : d) + '%' : '-'; }
  function fci(ci) { return Array.isArray(ci) && isNum(ci[0]) && isNum(ci[1]) ? '[' + fp(ci[0]) + ' to ' + fp(ci[1]) + ']' : ''; }
  function tn(t) { return String(t == null ? '-' : t).replace(/_/g, ' ').replace(/,/g, ' / '); }
  function get(o) { for (var i = 1; i < arguments.length; i++) { if (o == null) return null; o = o[arguments[i]]; } return o == null ? null : o; }
  function safeUrl(u) { return /^https?:\/\//i.test(u || '') ? u : ''; }
  function fmtUTC(s) {
    var d = new Date(s); if (!s || isNaN(d)) return '-';
    return d.toISOString().slice(0, 16).replace('T', ' ') + ' UTC';
  }
  function age(s) {
    var d = new Date(s); if (!s || isNaN(d)) return '';
    var m = Math.max(0, Math.round((Date.now() - d.getTime()) / 60000));
    return m < 90 ? m + ' min ago' : m < 2880 ? Math.round(m / 60) + ' h ago' : Math.round(m / 1440) + ' days ago';
  }
  function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }
  function put(el, html) { if (el && el.__h !== html) { el.__h = html; el.innerHTML = html; } }
  var T = function (k, t) { return Labels.term(k, t); };
  var MQ_OVERLAY = window.matchMedia('(max-width: 1279px)'), MQ_WIDE = window.matchMedia('(min-width: 960px)');
  var pushed = false;
  /* verdict-style codes from the data (FAKE, plausibly_fake) as readable words */
  function vname(l) {
    var u = String(l == null ? '' : l).toUpperCase(), n = Labels.verdict(u);
    if (n === u || /[A-Z]{2}/.test(n)) n = u.replace(/_/g, ' ').toLowerCase();
    return n.charAt(0).toUpperCase() + n.slice(1);
  }
  function tterm(t) { return T(tn(t)); } /* a technique name with its (?) definition when the glossary has one */
  function tdef(t) { var g = Labels.glossaryEntry(tn(t)); return g ? g.definition : ''; }
  var lsGet = function (k) { try { return localStorage.getItem(k); } catch (e) { return null; } };
  var lsSet = function (k, v) { try { localStorage.setItem(k, v); } catch (e) {} };
  function name(iso) { return get(S.meta, 'names', iso) || get(S.world, 'countries', iso, 'name') || iso; }
  function fetchJSON(p) {
    return fetch(API + p, { cache: 'no-cache' }).then(function (r) {
      return r.json().catch(function () { return {}; }).then(function (j) {
        if (!r.ok) { var e = new Error(j.error || ('HTTP ' + r.status)); e.status = r.status; e.code = j.code; throw e; }
        return j;
      });
    });
  }

  /* ---------- data ---------- */
  function stats() { return get(S.countries, 'countries', S.mode, S.win, S.src) || {}; }
  function gstat() { return get(S.meta, 'global', S.mode, S.win, S.src); }
  function effMetric() { return S.layer === 'fake' ? (S.metric === 'volume' ? 'volume' : 'rate') : S.metric; }
  function minN() { return get(S.meta, 'min_n') || 30; }
  function heatView() { return get(S.heat, 'views', S.mode, S.win, S.src); }
  function techList() { return (S.heat && S.heat.techniques) || get(S.meta, 'techniques') || []; }
  function curTech() { var t = techList(); return t[S.tech] != null ? t[S.tech] : (t[0] != null ? t[0] : null); }
  function techShare(iso) { /* selected technique's share, only for countries in the heatmap (n >= min_n) */
    var hv = heatView(), t = techList(), ti = t.indexOf(curTech()), i;
    if (!hv || !Array.isArray(hv.countries) || !Array.isArray(hv.share) || ti < 0) return null;
    i = hv.countries.indexOf(iso); if (i < 0) return null;
    var v = (hv.share[i] || [])[ti]; return isNum(v) ? v : null;
  }
  function val(st, iso) {
    if (!st || !(st.n > 0)) return null;
    var m = effMetric(), v;
    if (m === 'volume') v = st.n;
    else if (S.layer === 'fake') v = st.debunk_rate;
    else if (m === 'technique') v = (st.insufficient || st.n < minN()) ? null : techShare(iso);
    else v = m === 'lift' ? st.lift : st.rate;
    return isNum(v) ? v : null;
  }
  function fv(v) { /* value formatted for the active metric */
    var m = effMetric();
    return m === 'volume' ? fn(v) : m === 'lift' ? fx(v) : fp(v, S.layer === 'fake' ? 2 : 1);
  }
  function metricName() {
    var m = effMetric();
    return m === 'volume' ? 'posts' : m === 'technique' ? tn(curTech()) + ' share' : S.layer === 'fake' ? 'debunk rate' : m === 'lift' ? 'lift' : 'signal rate';
  }
  function kfmt(v) { return v >= 1e6 ? (v / 1e6) + 'M' : v >= 1e3 ? (v / 1e3) + 'k' : String(v); }
  function newsOff() { return !(get(S.meta, 'n_articles') > 0); }

  function load(initial) {
    if (S.loading) return Promise.resolve();
    S.loading = true;
    var tries = 0;
    function once() {
      return fetchJSON('meta').then(function (meta) {
        if (!initial && S.loaded && meta.built_at === S.built) return null;
        return Promise.all([fetchJSON('world'), fetchJSON('countries'), fetchJSON('heatmap').catch(function () { return null; }), fetchJSON('meta')])
          .then(function (r) {
            if (r[3].built_at !== meta.built_at && tries++ < 2) return once(); /* a build swapped mid-load: start over */
            S.meta = r[3]; S.world = r[0]; S.countries = r[1]; S.heat = r[2]; S.built = r[3].built_at; S.centers = null;
            return true;
          });
      });
    }
    return once().then(function (changed) {
      if (changed) {
        S.loaded = true; showApp(true);
        buildMap(); renderAll();
        if (S.sel) { ensureDetail(S.sel); revealRow(); }
      } else if (S.loaded) renderFresh();
    }).catch(function (e) {
      if (!S.loaded) showEmpty(e);
      else renderFresh(e.message);
    }).then(function () { S.loading = false; });
  }
  function showEmpty(e) {
    $('#pl-app').hidden = true; var el = $('#pl-empty'); el.hidden = false;
    var nb = !e || e.status === 503 || e.status === 404;
    put(el, '<h3>' + (nb ? 'Propaganda Lab data is not ready yet' : 'Propaganda Lab data could not be loaded') + '</h3><p>' +
      (nb ? 'The country figures have not been built on this machine yet. This page checks again every minute and fills in when they appear.' : 'Something went wrong while loading the country figures. This page tries again every minute; you can also reload it.') + '</p>');
  }
  function showApp() { $('#pl-empty').hidden = true; $('#pl-app').hidden = false; }

  function poll() {
    if (!S.active || document.hidden) return;
    load(false);
  }
  setInterval(poll, 60000);
  document.addEventListener('visibilitychange', function () { if (!document.hidden) poll(); });

  /* ---------- fresh strip + figures ---------- */
  function renderFresh(err) {
    var m = S.meta || {};
    put($('#pl-fresh'), '<span><span class="lbl">Data through</span> <b>' + esc(m.data_through ? Labels.dateLong(m.data_through) : '-') + '</b></span>' +
      '<span><span class="lbl">Built</span> <b>' + esc(m.built_at ? Labels.dateTimeUTC(m.built_at) : '-') + '</b> <span class="pl-age">' + esc(age(m.built_at)) + '</span></span>' +
      (err ? '<span class="pl-age">Could not refresh, showing the last build.</span>' : ''));
  }
  function renderKpis() {
    var m = S.meta || {}, g = gstat() || {}, st = stats(), nc = 0, ns = 0, k;
    for (k in st) { if (k !== UNL && st[k] && st[k].n > 0) { nc++; if (!st[k].insufficient) ns++; } }
    var loc = S.mode === 'origin' ? m.n_located_origin : m.n_located_about;
    var share = isNum(loc) && m.n_posts > 0 ? fp(loc / m.n_posts, 0) : '-';
    var cards = [
      ['Posts in view', fn(g.n), (S.mode === 'origin' ? 'origin known for ' : 'tied to a country for ') + share + ' of ' + fn(m.n_posts), ''],
      ['Countries', fn(nc), fn(ns) + ' with enough posts to rank', ''],
      ['Signal rate', fp(g.rate), fci(g.ci) || 'all countries, this view', '', T('rate', 'Signal rate')],
      ['Debunk rate', fp(g.debunk_rate, 2), fn(g.debunk_n) + ' posts near a debunked claim', 'fake', T('debunk rate', 'Debunk rate')]
    ];
    if (m.n_articles > 0) cards.splice(1, 0, ['News articles', fn(m.n_articles), 'included in the All source view', '']);
    if (isNum(m.n_gemma_verified) && m.n_gemma_verified > 0) cards.push(['Checked against fact-checks', fn(m.n_gemma_verified), 'posts with a candidate match', '']);
    if (isNum(m.n_gemma_span_checked) && m.n_gemma_span_checked > 0) cards.push(['Second-opinion check', fn(m.n_gemma_span_checked), 'posts sampled', '', T('Gemma check', 'Gemma check')]);
    put($('#pl-kpis'), cards.map(function (c) {
      return '<div class="pl-kpi ' + c[3] + '"><span class="lbl">' + (c[4] || esc(c[0])) + '</span><b>' + esc(c[1]) + '</b><small title="' + esc(c[2]) + '">' + esc(c[2]) + '</small></div>';
    }).join(''));
  }

  /* ---------- controls: View and Metric stay in the toolbar; Attribution, Layer, Window and Source live in the Filters popover ---------- */
  var WIN_NAME = { '7d': 'Last 7 days', '30d': 'Last 30 days', all: 'All time' };
  var SRC_NAME = { social: 'Social', news: 'News', all: 'All sources' };
  function renderControls() {
    var f = $('#pl-controls'), fake = S.layer === 'fake', noNews = newsOff();
    if (noNews && S.src === 'news') S.src = 'all';
    var cur = { view: S.view, mode: S.mode, layer: S.layer, metric: effMetric(), window: S.win, source: S.src };
    Object.keys(cur).forEach(function (k) {
      f.querySelectorAll('input[name="pl-' + k + '"]').forEach(function (i) {
        i.checked = i.value === cur[k];
        var off = (k === 'metric' && (i.value === 'lift' || i.value === 'technique') && fake) || (k === 'source' && i.value === 'news' && noNews);
        i.disabled = off; i.parentNode.classList.toggle('dis', off);
        i.parentNode.title = off ? (k === 'metric' ? 'Lift and technique apply to the propaganda signal layer only' : 'No news articles in this build yet') : '';
      });
    });
    var tsel = $('#pl-tech'), techOn = !fake && S.metric === 'technique', tl = techList();
    if (tsel) {
      var th = tl.map(function (t, i) { return '<option value="' + i + '">' + esc(tn(t)) + '</option>'; }).join('');
      if (tsel.__h !== th) { tsel.__h = th; tsel.innerHTML = th; }
      tsel.value = String(tl.indexOf(curTech()) < 0 ? 0 : tl.indexOf(curTech()));
      tsel.parentNode.hidden = !techOn;
    }
    var em = effMetric();
    put($('#pl-metric-term'), em === 'volume' ? T('volume', 'Volume') : em === 'technique' ? T('propaganda technique', 'Technique') : fake ? T('debunk rate', 'Debunk rate') : T(em === 'lift' ? 'lift' : 'signal rate', em === 'lift' ? 'Lift' : 'Signal rate'));
    put($('#pl-tech-term'), techOn && curTech() != null ? tterm(curTech()) : '');
    /* the Filters button counts the settings that differ from the defaults; chips beside it always show all four */
    var nd = (S.mode !== 'about') + (S.layer !== 'signal') + (S.win !== 'all') + (S.src !== 'all');
    put($('#pl-filters-btn'), '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" aria-hidden="true"><path d="M4 7h10M18 7h2M4 17h2M10 17h10"/><circle cx="16" cy="7" r="2"/><circle cx="8" cy="17" r="2"/></svg>Filters' + (nd ? '<span class="n" aria-hidden="true">' + nd + '</span><span class="sr"> (' + nd + ' changed)</span>' : ''));
    var chips = [['Countries', S.mode === 'origin' ? 'Posted from' : 'Mentioned'], ['Layer', fake ? 'Fake / misleading' : 'Propaganda signal'], ['Window', WIN_NAME[S.win] || S.win], ['Source', SRC_NAME[S.src] || S.src]];
    put($('#pl-chips'), chips.map(function (c) { return '<button type="button" class="fchip pchip" data-filters="1" title="Change in Filters"><small>' + esc(c[0]) + '</small><b>' + esc(c[1]) + '</b></button>'; }).join(''));
    put($('#pl-news-note'), noNews ? 'News articles are not in this build yet, so only social posts are counted.' : '');
    var notes = [];
    if (techOn) notes.push('Technique share is the fraction of a country\'s signal posts that use the chosen technique. Only countries with at least ' + minN() + ' posts are shown.');
    if (S.mode === 'origin') {
      var m = S.meta || {};
      notes.push('Where a post came from is known for ' + (isNum(m.n_located_origin) && m.n_posts > 0 ? fp(m.n_located_origin / m.n_posts, 0) : '-') + ' of posts (registered channels only). The rest are listed under Unlocated.');
    }
    if (fake) notes.push('The fake / misleading layer shows the share of posts that closely match a debunked claim. Lift and technique are not defined here, so rate is shown (or volume, if selected).');
    put($('#pl-ctl-note'), esc(notes.join(' ')));
    renderVizHead();
  }
  $('#pl-controls').addEventListener('change', function (e) {
    var t = e.target;
    if (t && t.id === 'pl-tech') { S.tech = +t.value || 0; S.sort = null; renderAll(); return; }
    if (!t || t.type !== 'radio') return;
    var k = t.name.replace('pl-', '');
    if (k === 'view') S.view = t.value; else if (k === 'mode') S.mode = t.value; else if (k === 'layer') S.layer = t.value;
    else if (k === 'metric') S.metric = t.value; else if (k === 'window') S.win = t.value; else if (k === 'source') S.src = t.value;
    if (k === 'layer' || k === 'metric') S.sort = null;
    if (k === 'mode' || k === 'window' || k === 'source') S.posts.key = '';
    renderAll();
  });
  $('#pl-chips').addEventListener('click', function (e) { if (e.target.closest('[data-filters]')) $('#pl-filters-btn').click(); });
  if (window.Shell) window.Shell.addPop($('#pl-filters-btn'), $('#pl-filters'), null, 'input:checked');

  /* panel title and one-line explanation above the map follow the metric */
  function renderVizHead() {
    var em = effMetric(), t, sub;
    if (S.view === 'heat') {
      t = 'Techniques by country';
      sub = S.layer !== 'fake' && em === 'lift' ? 'Each row is a country with enough posts, each column a propaganda technique. Colour shows how far a technique is used above or below the global average.' : 'Each row is a country with enough posts, each column a propaganda technique. Colour shows the share of its signal posts that use it.';
    } else if (em === 'volume') { t = 'Posts by country'; sub = 'The number of posts attributed to each country in this view, on a log scale. Select a country to open its file.'; }
    else if (em === 'technique') { t = tterm(curTech()) + ' share by country'; sub = 'The share of a country\'s propaganda-signal posts that use this technique. Countries with too few posts are hatched.'; }
    else if (S.layer === 'fake') { t = T('debunk rate', 'Debunk rate') + ' by country'; sub = 'The share of posts that closely match a claim fact-checkers rated fake or misleading. A lead for review, not a ruling.'; }
    else if (em === 'lift') { t = T('lift', 'Lift') + ' by country'; sub = 'How much more often posts carry propaganda techniques than expected for posts of the same length. 1 is as expected; higher is more.'; }
    else { t = T('signal rate', 'Signal rate') + ' by country'; sub = 'The share of posts with at least one strong propaganda span. It over-fires on short or emotive posts, so compare countries by lift.'; }
    put($('#pl-viz-h'), t); put($('#pl-viz-sub'), esc(sub));
  }

  /* ---------- colour ---------- */
  function colorFor(v) {
    if (S.layer !== 'fake' && effMetric() === 'lift') {
      var t = v > 0 ? clamp(Math.log(v) / Math.LN2, -1, 1) : -1;
      return t >= 0 ? 'color-mix(in srgb, var(--dv-hi) ' + (t * 100).toFixed(1) + '%, var(--dv-mid))' : 'color-mix(in srgb, var(--dv-lo) ' + (-t * 100).toFixed(1) + '%, var(--dv-mid))';
    }
    if (effMetric() === 'volume') {
      var lo = Math.log(S.vlo || 1), hi = Math.log(S.vhi || 10), tv = v > 0 ? clamp((Math.log(v) - lo) / ((hi - lo) || 1), 0, 1) : 0;
      return 'color-mix(in srgb, ' + (S.layer === 'fake' ? 'var(--fake)' : 'var(--accent)') + ' ' + (tv * 100).toFixed(1) + '%, var(--paper-2))';
    }
    return 'color-mix(in srgb, ' + (S.layer === 'fake' ? 'var(--fake)' : 'var(--accent)') + ' ' + (clamp(v / (S.rmax || 1), 0, 1) * 100).toFixed(1) + '%, var(--paper-2))';
  }
  function niceCeil(x) {
    if (!(x > 0)) return 1;
    var mag = Math.pow(10, Math.floor(Math.log(x) / Math.LN10)), f = x / mag;
    return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 2.5 ? 2.5 : f <= 5 ? 5 : 10) * mag;
  }
  function computeRmax() {
    var st = stats(), mx = 0, k, v;
    var mn = Infinity, vol = effMetric() === 'volume';
    for (k in st) {
      if (k === UNL || !st[k] || (st[k].insufficient && !vol)) continue; v = val(st[k], k);
      if (v != null && v > mx) mx = v; if (v != null && v > 0 && v < mn) mn = v;
    }
    S.rmax = niceCeil(mx);
    if (vol) { S.vlo = isFinite(mn) ? Math.pow(10, Math.floor(Math.log(mn) / Math.LN10)) : 1; S.vhi = mx > 0 ? Math.pow(10, Math.ceil(Math.log(mx) / Math.LN10)) : 10; if (S.vhi <= S.vlo) S.vhi = S.vlo * 10; }
  }
  function pctTick(v) { var p = v * 100; return p.toFixed(p === 0 ? 0 : p < 2 ? 2 : p < 10 ? 1 : 0) + '%'; }

  function legend(kind) { /* kind: 'map' or 'heat' */
    var lift = S.layer !== 'fake' && effMetric() === 'lift', vol = kind === 'map' && effMetric() === 'volume', h = '', ticks = '', bar;
    if (kind === 'heat') {
      if (lift) { bar = 'lift'; } else { bar = 'seq'; }
    } else bar = lift ? 'lift' : 'seq';
    if (bar === 'lift') {
      [[0.5, '\u22640.5'], [0.7, '0.7'], [1, '1'], [1.4, '1.4'], [2, '\u22652']].forEach(function (t) {
        ticks += '<span style="left:' + ((Math.log(t[0]) / Math.LN2 + 1) / 2 * 100).toFixed(1) + '%">' + esc(t[1]) + '</span>';
      });
    } else if (vol) {
      var lo = S.vlo || 1, hi = S.vhi || 10, span = Math.log(hi) - Math.log(lo) || 1;
      for (var tv = lo; tv <= hi * 1.0001; tv *= 10) ticks += '<span style="left:' + ((Math.log(tv) - Math.log(lo)) / span * 100).toFixed(1) + '%">' + esc(kfmt(Math.round(tv))) + '</span>';
    } else {
      var top = kind === 'heat' ? S.hmax : S.rmax;
      [0, 0.25, 0.5, 0.75, 1].forEach(function (f) { ticks += '<span style="left:' + (f * 100) + '%">' + esc(pctTick(top * f)) + '</span>'; });
    }
    var cap = kind === 'heat'
      ? (bar === 'lift' ? 'Technique share vs global share' : 'Share of signal posts using the technique')
      : (vol ? 'Posts per country (log scale)' : effMetric() === 'technique' ? 'Share of signal posts using ' + tn(curTech()) : S.layer === 'fake' ? 'Debunk rate' : (lift ? 'Lift: 1 = as expected, length-adjusted' : 'Signal rate'));
    var capHtml = lift && kind === 'map' ? T('lift', 'Lift') + ': 1 = as expected, length-adjusted' : esc(cap);
    h += '<div class="pl-ramp"><div class="lbl">' + capHtml + '</div><div class="pl-ramp-bar ' + bar + (S.layer === 'fake' && kind === 'map' ? ' fake' : '') + '" role="img" aria-label="' + esc(cap) + ' colour scale"></div><div class="pl-ramp-ticks">' + ticks + '</div></div>';
    if (kind === 'map') {
      if (!vol) h += '<span class="pl-sw"><i class="hatch"></i>Too few posts (' + T('min_n', 'under ' + minN()) + ')</span>';
      h += '<span class="pl-sw"><i></i>No data</span>';
    } else h += '<span class="pl-sw"><i></i>No value</span>';
    put($('#pl-legend'), h);
  }

  /* ---------- map ---------- */
  var mapEl = $('#pl-map'), tip = $('#pl-tip');
  function buildMap() {
    var w = S.world; if (!w || !w.countries) { put(mapEl, '<p class="note">The map outline is not available in this build. The country list still works.</p>'); return; }
    var ver = w.viewBox + '|' + Object.keys(w.countries).reduce(function (a, k) { return a + k + ((w.countries[k] || {}).d || '').length; }, '');
    if (mapEl.__ver === ver) return;
    mapEl.__ver = ver;
    var svg = '<svg xmlns="' + SVGNS + '" viewBox="' + esc(w.viewBox || '0 0 1000 500') + '" role="group" aria-label="World map of countries. Use the country list for a text version." id="pl-svg">' +
      '<defs><pattern id="pl-hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect class="hatch-bg" width="6" height="6"/><line class="hatch-ln" x1="0" y1="0" x2="0" y2="6"/></pattern></defs>';
    Object.keys(w.countries).forEach(function (iso) {
      var c = w.countries[iso]; if (!c || !c.d) return;
      svg += '<path class="c nodata" data-iso="' + esc(iso) + '" d="' + esc(c.d) + '" role="button" tabindex="-1" aria-label="' + esc(name(iso)) + '"></path>';
    });
    mapEl.__h = null; mapEl.innerHTML = svg + '</svg>';
    mapEl.__svgBuilt = true;
  }
  function paintMap() {
    var svg = $('#pl-svg'); if (!svg) return;
    var st = stats(), metricNm = metricName();
    var first = null, rov = null;
    svg.querySelectorAll('path.c').forEach(function (p) {
      var iso = p.getAttribute('data-iso'), s = st[iso], v = val(s, iso), lab;
      p.classList.toggle('sel', iso === S.sel);
      if (!s || !(s.n > 0)) {
        p.style.fill = ''; p.classList.add('nodata'); p.removeAttribute('tabindex'); lab = name(iso) + ': no data in this view';
      } else {
        p.classList.remove('nodata');
        if ((s.insufficient && effMetric() !== 'volume') || v == null) { p.style.fill = 'url(#pl-hatch)'; lab = name(iso) + ': ' + (s.insufficient || s.n < minN() ? 'too few posts (' + fn(s.n) + '), not coloured' : metricNm + ' not available, ' + fn(s.n) + ' posts'); }
        else { p.style.fill = colorFor(v); lab = name(iso) + ': ' + metricNm + ' ' + fv(v) + (effMetric() === 'volume' ? '' : ', ' + fn(s.n) + ' posts'); }
        p.setAttribute('tabindex', '-1'); if (!first) first = p; if (iso === S.roving) rov = p;
      }
      p.setAttribute('aria-label', lab); p.__lab = lab;
    });
    var r = (iso2el(S.sel) && !iso2el(S.sel).classList.contains('nodata') ? iso2el(S.sel) : rov) || first;
    if (r) { r.setAttribute('tabindex', '0'); S.roving = r.getAttribute('data-iso'); }
  }
  function iso2el(iso) { return iso ? $('#pl-svg path[data-iso="' + iso + '"]') : null; }

  function showTip(p, ev) {
    var iso = p.getAttribute('data-iso'), s = stats()[iso], v = val(s, iso), wrap = $('#pl-map-wrap'), r = wrap.getBoundingClientRect();
    var lines = '<b>' + esc(name(iso)) + '</b>';
    if (!s || !(s.n > 0)) lines += '<span>No data in this view</span>';
    else {
      lines += '<span>' + fn(s.n) + ' posts' + (s.insufficient ? ' (too few to colour)' : '') + '</span>';
      if (effMetric() === 'technique') lines += '<span>' + esc(tn(curTech())) + ' share ' + (v == null ? 'n/a (too few posts)' : fp(v)) + '</span>';
      lines += '<span>Lift ' + fx(s.lift) + ' - rate ' + fp(s.rate) + '</span>';
      if (S.layer === 'fake') lines += '<span>Debunk rate ' + fp(s.debunk_rate, 2) + ' (' + fn(s.debunk_n) + ' posts)</span>';
    }
    tip.innerHTML = lines; tip.hidden = false;
    var x, y;
    if (ev && ev.clientX != null) { x = ev.clientX - r.left + 14; y = ev.clientY - r.top + 14; }
    else { var b = p.getBoundingClientRect(); x = b.left - r.left + b.width / 2; y = b.top - r.top + b.height / 2; }
    x = clamp(x, 4, Math.max(4, r.width - tip.offsetWidth - 4)); y = clamp(y, 4, Math.max(4, r.height - tip.offsetHeight - 4));
    tip.style.left = x + 'px'; tip.style.top = y + 'px';
  }
  function hideTip() { tip.hidden = true; }
  mapEl.addEventListener('pointermove', function (e) {
    var p = e.target.closest && e.target.closest('path.c'); if (!p || e.pointerType === 'touch') { hideTip(); return; } showTip(p, e);
  });
  mapEl.addEventListener('pointerleave', hideTip);
  mapEl.addEventListener('click', function (e) { var p = e.target.closest && e.target.closest('path.c'); if (p) select(p.getAttribute('data-iso'), true); });
  mapEl.addEventListener('focusin', function (e) {
    var p = e.target.closest && e.target.closest('path.c'); if (!p) return;
    var old = iso2el(S.roving); if (old && old !== p && old.getAttribute('tabindex') === '0') old.setAttribute('tabindex', '-1');
    p.setAttribute('tabindex', '0'); S.roving = p.getAttribute('data-iso'); showTip(p);
  });
  mapEl.addEventListener('focusout', hideTip);
  mapEl.addEventListener('keydown', function (e) {
    var p = e.target.closest && e.target.closest('path.c'); if (!p) return;
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); select(p.getAttribute('data-iso'), true); return; }
    var dir = { ArrowRight: [1, 0], ArrowLeft: [-1, 0], ArrowDown: [0, 1], ArrowUp: [0, -1] }[e.key]; if (!dir) return;
    e.preventDefault();
    if (!S.centers) {
      S.centers = {};
      mapEl.querySelectorAll('path.c').forEach(function (q) { var b = q.getBBox(); S.centers[q.getAttribute('data-iso')] = [b.x + b.width / 2, b.y + b.height / 2]; });
    }
    var me = S.centers[p.getAttribute('data-iso')], best = null, bs = Infinity;
    mapEl.querySelectorAll('path.c:not(.nodata)').forEach(function (q) {
      if (q === p) return;
      var c = S.centers[q.getAttribute('data-iso')], dx = c[0] - me[0], dy = c[1] - me[1], along = dx * dir[0] + dy * dir[1], perp = Math.abs(dx * dir[1]) + Math.abs(dy * dir[0]);
      if (along <= 0) return;
      var sc = along * along + 3 * perp * perp; if (sc < bs) { bs = sc; best = q; }
    });
    if (best) best.focus();
  });
  mapEl.addEventListener('keyup', function (e) { if (e.key === ' ') e.preventDefault(); });

  /* ---------- heatmap ---------- */
  function renderViz() {
    var heat = S.view === 'heat';
    $('#pl-map-wrap').hidden = heat; $('#pl-heat').hidden = !heat;
    computeRmax();
    var note = '';
    if (heat) { renderHeat(); note = S.layer === 'fake' ? 'The heatmap always shows propaganda techniques among signal posts; countries are sorted by the debunk rate of the fake / misleading layer.' : ''; }
    else {
      paintMap();
      var u = stats()[UNL], g = gstat();
      var ut = u && u.n > 0 ? '<span class="pl-unl">' + T('unlocated', 'Unlocated') + ': ' + fn(u.n) + ' posts' + (g && g.n > 0 ? ' (' + fp(u.n / g.n, 0) + ' of this view)' : '') + ' <button type="button" data-unl="1">Open details</button></span>' : '';
      note = '';
      put($('#pl-viz-note'), ut + esc(note));
    }
    legend(heat ? 'heat' : 'map');
    if (heat) put($('#pl-viz-note'), esc(note));
  }
  function renderHeat() {
    var el = $('#pl-heat'), hv = get(S.heat, 'views', S.mode, S.win, S.src), techs = (S.heat && S.heat.techniques) || get(S.meta, 'techniques') || [];
    if (!hv || !hv.countries || !hv.countries.length) { put(el, '<p class="note" style="padding:16px">No countries to show for this selection. A country needs at least ' + esc(get(S.meta, 'min_n') || 30) + ' posts to appear in the heatmap.</p>'); return; }
    var lift = S.layer !== 'fake' && effMetric() === 'lift', st = stats(), rows = [];
    hv.countries.forEach(function (iso, i) { rows.push({ iso: iso, i: i, v: val(st[iso], iso), n: get(st[iso], 'n') || 0 }); });
    rows.sort(function (a, b) { return (b.v == null ? -Infinity : b.v) - (a.v == null ? -Infinity : a.v) || b.n - a.n; });
    var mx = 0;
    if (!lift) hv.share.forEach(function (r) { (r || []).forEach(function (x) { if (isNum(x) && x > mx) mx = x; }); });
    S.hmax = niceCeil(mx);
    var tcol = effMetric() === 'technique' ? techs.indexOf(curTech()) : -1;
    var h = '<table class="pl-heat" aria-label="Countries by propaganda technique' + (tcol >= 0 ? ', sorted by ' + esc(tn(curTech())) + ' share' : '') + '"><thead><tr><th class="rowh" scope="col"><span class="lbl" style="padding:6px 8px;display:block">Country</span></th>' +
      techs.map(function (t, j) { return '<th scope="col"' + (j === tcol ? ' class="colsel" aria-sort="descending"' : '') + ' title="' + esc(tn(t) + (tdef(t) ? ': ' + tdef(t) : '')) + '"><span>' + esc(tn(t)) + '</span></th>'; }).join('') + '</tr></thead><tbody>';
    rows.forEach(function (r) {
      var sh = (hv.share || [])[r.i] || [], ra = (hv.ratio || [])[r.i] || [];
      h += '<tr' + (r.iso === S.sel ? ' class="sel"' : '') + '><th scope="row" class="rowh"><button type="button" data-iso="' + esc(r.iso) + '" title="' + esc(name(r.iso)) + '">' + esc(name(r.iso)) + '</button></th>';
      techs.forEach(function (t, j) {
        var s = sh[j], q = ra[j], v = lift ? q : s, tt = name(r.iso) + ' / ' + tn(t) + ': share ' + fp(s) + ' of signal posts, ' + fx(q) + 'x global';
        if (!isNum(v)) h += '<td class="nul' + (j === tcol ? ' colsel' : '') + '" title="' + esc(tt) + '">-</td>';
        else {
          var light = !lift && v / (S.hmax || 1) > 0.55;
          var col = lift ? colorFor(v) : 'color-mix(in srgb, var(--accent) ' + (clamp(v / (S.hmax || 1), 0, 1) * 100).toFixed(1) + '%, var(--paper-2))';
          h += '<td class="' + (light ? 'light' : '') + (j === tcol ? ' colsel' : '') + '" style="background:' + col + '" title="' + esc(tt) + '">' + (lift ? fx(v, 1) : (v * 100).toFixed(0)) + '</td>';
        }
      });
      h += '</tr>';
    });
    put(el, h + '</tbody></table>');
  }
  $('#pl-heat').addEventListener('click', function (e) { var b = e.target.closest('button[data-iso]'); if (b) select(b.getAttribute('data-iso'), true); });
  $('#pl-viz-note').addEventListener('click', function (e) { if (e.target.closest('[data-unl]')) select(UNL, true); });

  /* ---------- country list (leaderboard) ---------- */
  var KEYS = {
    name: 'Country (A to Z)', n: 'Posts', rate: 'Signal rate', lift: 'Lift', debunk: 'Debunk rate', tech: 'Top technique', ver: 'Checked against fact-checks'
  };
  function keyLabel(k) { return k === 'tshare' ? tn(curTech()) + ' share' : KEYS[k] || k; }
  function techOn() { return effMetric() === 'technique'; }
  function defKey() { var m = effMetric(); return S.layer === 'fake' && m !== 'volume' ? 'debunk' : m === 'volume' ? 'n' : m === 'technique' ? 'tshare' : m; }
  function cellVal(iso, s, k) {
    if (k === 'name') return name(iso).toLowerCase();
    if (k === 'tshare') return s.insufficient || s.n < minN() ? null : techShare(iso);
    if (k === 'n') return s.n; if (k === 'rate') return isNum(s.rate) ? s.rate : null; if (k === 'lift') return isNum(s.lift) ? s.lift : null;
    if (k === 'debunk') return isNum(s.debunk_rate) ? s.debunk_rate : null; if (k === 'tech') return s.top_technique ? tn(s.top_technique).toLowerCase() : null;
    if (k === 'ver') return isNum(s.verified_n) ? s.verified_n : null;
    return null;
  }
  function showKey(iso, s, k) { /* the value a row shows for the key the list is sorted by */
    var v = cellVal(iso, s, k);
    if (k === 'name') return fn(s.n);
    if (k === 'tech') return s.top_technique ? tn(s.top_technique) : '-';
    if (k === 'n' || k === 'ver') return fn(v);
    if (k === 'lift') return fx(v);
    return fp(v, k === 'debunk' ? 2 : 1);
  }
  function sortKeys() { var ks = ['n', 'rate', 'lift', 'debunk', 'ver', 'tech', 'name']; if (techOn()) ks.unshift('tshare'); return ks; }
  function renderSortControls(key, dir) {
    var sel = $('#pl-sort'), opts = '<option value="">Selected metric: ' + esc(keyLabel(defKey())) + '</option>' + sortKeys().filter(function (k) { return k !== defKey() || S.sort; }).map(function (k) {
      return '<option value="' + k + '">' + esc(keyLabel(k)) + '</option>';
    }).join('');
    if (sel.__h !== opts) { sel.__h = opts; sel.innerHTML = opts; }
    sel.value = S.sort ? S.sort.key : '';
    if (sel.value !== (S.sort ? S.sort.key : '')) sel.value = '';
    var txt = key === 'name' || key === 'tech' ? (dir > 0 ? 'A to Z' : 'Z to A') : (dir > 0 ? 'Low to high' : 'High to low');
    var b = $('#pl-dir'); b.textContent = txt; b.setAttribute('aria-label', 'Sort order: ' + txt + '. Select to reverse.');
  }
  function renderBoard() {
    var st = stats(), key, dir, rows = [], q = S.find.trim().toLowerCase(), unl = null, nAll = 0, nRank = 0;
    if (S.sort) { key = S.sort.key; dir = S.sort.dir; } else { key = defKey(); dir = -1; }
    Object.keys(st).forEach(function (iso) {
      var s = st[iso]; if (!s || !(s.n > 0)) return;
      if (iso !== UNL) { nAll++; if (!s.insufficient) nRank++; }
      if (q && name(iso).toLowerCase().indexOf(q) < 0 && iso.toLowerCase() !== q) return;
      if (iso === UNL) { unl = [iso, s]; return; }
      rows.push([iso, s]);
    });
    var valueKey = key === 'lift' || key === 'rate' || key === 'debunk' || key === 'tshare';
    rows.sort(function (a, b) {
      if (valueKey && !!a[1].insufficient !== !!b[1].insufficient) return a[1].insufficient ? 1 : -1;
      var x = cellVal(a[0], a[1], key), y = cellVal(b[0], b[1], key);
      if (x == null && y == null) return 0; if (x == null) return 1; if (y == null) return -1;
      return (x < y ? -1 : x > y ? 1 : 0) * dir || (b[1].n - a[1].n);
    });
    if (unl) rows.push(unl);
    renderSortControls(key, dir);
    put($('#pl-count'), esc(fn(nAll)) + ' countries, ' + esc(fn(nRank)) + ' with at least ' + T('min_n', fn(minN()) + ' posts'));
    var board = $('#pl-board'), had = document.activeElement && board.contains(document.activeElement) ? document.activeElement.getAttribute('data-iso') : null;
    var cur = rows.some(function (r) { return r[0] === S.sel; }) ? S.sel : (rows[0] && rows[0][0]);
    var h = rows.map(function (r) {
      var iso = r[0], s = r[1], isU = iso === UNL, on = iso === S.sel, cv = cellVal(iso, s, key);
      var big = key === 'name' ? fn(s.n) + ' posts' : valueKey && cv == null ? '-' : showKey(iso, s, key);
      var sub = [];
      if (key !== 'n' && key !== 'name') sub.push(fn(s.n) + ' posts');
      if (key !== 'lift' && !isU && isNum(s.lift)) sub.push('lift ' + fx(s.lift));
      if (key !== 'tech' && s.top_technique && !isU) sub.push(tn(s.top_technique));
      var pct = key === defKey() && valueKey && isNum(cv) && !s.insufficient ? clamp(cv / (S.rmax || 1), 0, 1) : null;
      return '<li class="row lb' + (s.insufficient ? ' few' : '') + (isU ? ' unl' : '') + '" role="option" data-iso="' + esc(iso) + '" tabindex="' + (iso === cur ? 0 : -1) + '" aria-selected="' + on + '"' + (on ? ' aria-current="true"' : '') + '>' +
        '<span class="lb-name">' + esc(isU ? 'Unlocated posts' : name(iso)) + '</span><span class="lb-val">' + esc(big) + '</span>' +
        (pct != null ? '<span class="lb-bar" aria-hidden="true"><b style="width:' + (pct * 100).toFixed(1) + '%"></b></span>' : '') +
        '<span class="row-meta">' + (s.insufficient ? '<span class="lb-few" title="Fewer than ' + esc(minN()) + ' posts: figures swing too much to trust">few posts</span>' : '') + esc(sub.join(' - ')) + '</span></li>';
    }).join('');
    if (!rows.length) h = '<li class="lb-none" role="presentation">' + (q ? 'No country matches &ldquo;' + esc(S.find.trim()) + '&rdquo;.' : 'No countries have posts in this view.') + '</li>';
    put(board, h);
    if (had) { var nr = board.querySelector('.row[data-iso="' + had + '"]'); if (nr && document.activeElement !== nr) nr.focus({ preventScroll: true }); }
  }
  function boardRows() { return Array.prototype.slice.call($('#pl-board').querySelectorAll('.row[data-iso]')); }
  function boardCursor(row, focus) {
    boardRows().forEach(function (r) { r.tabIndex = r === row ? 0 : -1; });
    if (row && focus) row.focus();
  }
  function boardMove(from, dir) {
    var all = boardRows(); if (!all.length) return;
    var i = from ? all.indexOf(from) : -1, t = dir === 'first' ? 0 : dir === 'last' ? all.length - 1 : Math.max(0, Math.min(all.length - 1, i < 0 ? 0 : i + dir));
    boardCursor(all[t], true);
  }
  function revealRow() { /* keep the selected country in view when it was chosen on the map; only where the list scrolls on its own */
    if (!MQ_WIDE.matches) return;
    var r = $('#pl-board .row[aria-selected="true"]'); if (r) r.scrollIntoView({ block: 'nearest' });
  }
  $('#pl-board').addEventListener('click', function (e) {
    var r = e.target.closest('.row[data-iso]'); if (!r) return;
    boardCursor(r, false); select(r.getAttribute('data-iso'), true);
  });
  $('#pl-board').addEventListener('keydown', function (e) {
    var row = e.target.closest && e.target.closest('.row[data-iso]'); if (!row || e.altKey || e.ctrlKey || e.metaKey) return;
    var k = e.key;
    if (k === 'ArrowDown' || k === 'j') { e.preventDefault(); boardMove(row, 1); }
    else if (k === 'ArrowUp' || k === 'k') { e.preventDefault(); boardMove(row, -1); }
    else if (k === 'Home') { e.preventDefault(); boardMove(row, 'first'); }
    else if (k === 'End') { e.preventDefault(); boardMove(row, 'last'); }
    else if (k === 'Enter' || k === ' ') { e.preventDefault(); select(row.getAttribute('data-iso'), true); }
  });
  $('#pl-board').addEventListener('focusin', function (e) { var r = e.target.closest && e.target.closest('.row[data-iso]'); if (r && r.tabIndex !== 0) boardCursor(r, false); });
  $('#pl-find').addEventListener('input', function (e) { S.find = e.target.value; renderBoard(); });
  $('#pl-find').addEventListener('keydown', function (e) {
    if (e.key === 'Escape' && e.target.value) { e.target.value = ''; S.find = ''; renderBoard(); e.stopPropagation(); }
    else if (e.key === 'ArrowDown') { var f = boardRows()[0]; if (f) { e.preventDefault(); boardCursor($('#pl-board .row[tabindex="0"]') || f, true); } }
  });
  $('#pl-sort').addEventListener('change', function (e) {
    var k = e.target.value;
    S.sort = k ? { key: k, dir: k === 'name' || k === 'tech' ? 1 : -1 } : null; renderBoard();
  });
  $('#pl-dir').addEventListener('click', function () {
    var cur = S.sort || { key: defKey(), dir: -1 }; S.sort = { key: cur.key, dir: -cur.dir }; renderBoard();
  });
  /* j and k from anywhere on the Lab (outside form fields and the country file) move through the list */
  document.addEventListener('keydown', function (e) {
    if (!S.active || (e.key !== 'j' && e.key !== 'k') || e.ctrlKey || e.metaKey || e.altKey) return;
    var t = e.target, tag = t && t.tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || (t && t.isContentEditable) || (t && t.closest && t.closest('#pl-country, #pl-filters, #pl-board, #pl-map'))) return;
    e.preventDefault();
    var cur = $('#pl-board .row[aria-selected="true"]') || $('#pl-board .row[tabindex="0"]');
    if (cur) boardMove(cur, e.key === 'j' ? 1 : -1); else boardMove(null, 1);
  });

  /* ---------- selection + country panel ---------- */
  function syncSel() { var a = $('#pl-app'); if (a) a.setAttribute('data-sel', S.sel ? '1' : '0'); }
  function focusTitle() { var t = $('#lab-ctitle'); if (t) t.focus({ preventScroll: true }); }
  function select(iso, user) {
    if (!/^[A-Z]{3}$/.test(iso || '')) return;
    var same = S.sel === iso, wasNone = !S.sel; S.sel = iso;
    if (!same) { S.ctab = 'overview'; S.posts.key = ''; var pc = $('#pl-country'); if (pc) pc.scrollTop = 0; }
    syncSel(); paintMap(); renderBoard(); if (!user || !$('#pl-board').contains(document.activeElement)) revealRow();
    if (S.view === 'heat') renderHeat();
    renderCountry(); ensureDetail(iso);
    var live = $('#route-live'); if (live && !same) live.textContent = 'Country file opened: ' + name(iso);
    if (user && !same && window.Shell) {
      /* the first selection pushes a history entry so Back returns to the map; later selections replace it */
      if (wasNone) { pushed = true; window.Shell.navigate('#lab/' + iso); } else window.Shell.navigate('#lab/' + iso, true);
    }
    if (user && !same && MQ_OVERLAY.matches) focusTitle();
  }
  function goBack() {
    var iso = S.sel;
    if (pushed && history.length > 1) history.back(); else if (window.Shell) window.Shell.navigate('#lab', true);
    setTimeout(function () { var r = iso && $('#pl-board .row[data-iso="' + iso + '"]'); if (r && (MQ_OVERLAY.matches || $('#pl-country').contains(document.activeElement))) r.focus({ preventScroll: true }); }, 60);
  }
  function ensureDetail(iso) {
    if (S.dB[iso] === S.built) return;
    S.dB[iso] = S.built;
    var built = S.built;
    fetchJSON('country/' + iso).then(function (d) { return d; }, function (e) { if (e.status === 404) return null; throw e; })
      .then(function (d) { if (S.built !== built) return; S.detail[iso] = d; if (S.sel === iso) renderCountry(); })
        .catch(function () { delete S.dB[iso]; if (S.sel === iso) renderCountry('The detail for this country could not be loaded. Select the country again to retry.'); });
  }
  function dview(iso) { return get(S.detail[iso], 'modes', S.mode, S.win); }

  var CTABS = [['overview', 'Overview'], ['narratives', 'Narratives'], ['posts', 'Posts'], ['fake', 'Fake / misleading']];
  var ICON_BACK = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m15 5-7 7 7 7"/></svg>';
  function ctx() {
    return (S.mode === 'origin' ? 'Posts from channels based in this country' : 'Posts that mention this country') + ' - ' + (WIN_NAME[S.win] || S.win).toLowerCase() + ' - ' + (SRC_NAME[S.src] || S.src).toLowerCase();
  }
  function renderCountry(err) {
    var el = $('#pl-country'), iso = S.sel;
    if (!iso) { put(el, ''); return; }
    if (S.active && window.Shell) window.Shell.setTitle(name(iso));
    var loading = !(iso in S.detail), d = dview(iso), s = stats()[iso];
    var h = '<div class="case-bar"><button type="button" class="btn ghost case-close" data-act="close" aria-label="Close the country file">' + ICON_BACK + '<span class="cc-wide">Close</span><span class="cc-narrow">Back to map</span></button>' +
      '<div class="grow"><span class="case-kicker">Country file</span></div></div><div class="case-in lc-in">' +
      '<header class="lc-head"><h2 class="case-title" id="lab-ctitle" tabindex="-1">' + esc(iso === UNL ? 'Unlocated posts' : name(iso)) + '</h2><p class="lc-ctx">' + esc(ctx()) + '</p></header>';
    h += '<div class="pl-ctabs" role="tablist" aria-label="Country file sections">' + CTABS.map(function (t) {
      return '<button type="button" role="tab" id="pl-ct-' + t[0] + '" data-ct="' + t[0] + '" aria-selected="' + (S.ctab === t[0]) + '" aria-controls="pl-cpanel" tabindex="' + (S.ctab === t[0] ? 0 : -1) + '">' + esc(t[1]) + '</button>';
    }).join('') + '</div><div role="tabpanel" id="pl-cpanel" aria-labelledby="pl-ct-' + S.ctab + '">';
    if (err) h += '<p class="note">' + esc(err) + '</p>';
    if (S.ctab === 'overview') h += cOverview(iso, d, s, loading);
    else if (S.ctab === 'posts') h += cPostsShell(iso);
    else if (loading) h += '<p class="note">Loading this country\'s figures.</p>';
    else if (S.ctab === 'narratives') h += cNarratives(iso, d);
    else h += cFake(iso, d, s);
    put(el, h + '</div></div>');
    if (S.ctab === 'posts') initPosts(iso);
  }
  $('#pl-country').addEventListener('click', function (e) {
    if (e.target.closest('[data-act="close"]')) { goBack(); return; }
    var t = e.target.closest('button[data-ct]');
    if (t) { S.ctab = t.getAttribute('data-ct'); renderCountry(); var n = $('#pl-ct-' + S.ctab); if (n) n.focus(); return; }
    var fb = e.target.closest('button[data-fb]');
    if (fb) { var bx = fb.closest('.pl-fb'); if (bx) sendFlag(bx.getAttribute('data-uid'), fb.getAttribute('data-fb')); return; }
    var m = e.target.closest('.pl-more');
    if (m) { var p = m.previousElementSibling; if (p) { var c = p.classList.toggle('clamp'); m.textContent = c ? 'Show more' : 'Show less'; m.setAttribute('aria-expanded', !c); $('#pl-country').__h = null; } return; }
    if (e.target.closest('#pl-load-more')) loadPosts(false);
  });
  $('#pl-country').addEventListener('keydown', function (e) {
    var t = e.target.closest('button[data-ct]'); if (!t) return;
    var ids = CTABS.map(function (x) { return x[0]; }), i = ids.indexOf(S.ctab);
    if (e.key === 'ArrowRight') i = (i + 1) % ids.length; else if (e.key === 'ArrowLeft') i = (i + ids.length - 1) % ids.length;
    else if (e.key === 'Home') i = 0; else if (e.key === 'End') i = ids.length - 1; else return;
    e.preventDefault(); S.ctab = ids[i]; renderCountry(); var n = $('#pl-ct-' + S.ctab); if (n) n.focus();
  });
  $('#pl-country').addEventListener('change', function (e) {
    if (e.target.id === 'pl-f-tech') { S.posts.fT = e.target.value; S.posts.auto = 0; drawPosts(); }
    if (e.target.id === 'pl-f-sig') { S.posts.sigOnly = e.target.checked; S.posts.auto = 0; drawPosts(); }
  });

  function statBox(label, v, sub) { return '<div class="pl-stat"><span>' + (label && label.html ? label.html : esc(label)) + '</span><b>' + esc(v) + '</b>' + (sub ? '<small>' + esc(sub) + '</small>' : '') + '</div>'; }
  function bars(items, fmtv, markerMax) {
    if (!items.length) return '';
    var mx = Math.max.apply(null, items.map(function (i) { return Math.max(i.v || 0, i.m || 0); })) || 1;
    return '<div class="pl-bars">' + items.map(function (i) {
      return '<div class="pl-bar"><span title="' + esc(i.l) + '">' + (i.h || esc(i.l)) + '</span><span class="pl-bar-track"><b style="width:' + ((i.v || 0) / mx * 100).toFixed(1) + '%"></b>' +
        (i.m != null ? '<u style="left:' + (i.m / mx * 100).toFixed(1) + '%" title="Global share ' + esc(fp(i.m)) + '"></u>' : '') + '</span><small>' + esc(fmtv(i)) + '</small></div>';
    }).join('') + '</div>';
  }
  /* sum counts whose stored codes share a display name (hi + hi-IN = Hindi) */
  function mergeNames(o, nameOf) {
    var out = {}; Object.keys(o || {}).forEach(function (k) { out[nameOf(k)] = (out[nameOf(k)] || 0) + (isNum(o[k]) ? o[k] : 0); }); return out;
  }
  function topObj(o, n) {
    return Object.keys(o || {}).map(function (k) { return { l: k, v: o[k] }; }).filter(function (x) { return isNum(x.v); }).sort(function (a, b) { return b.v - a.v; }).slice(0, n);
  }
  function cOverview(iso, d, s, loading) {
    var k = get(d, 'kpis', S.src) || ((!d) ? s : null), h = '';
    if (!k) {
      h += '<p class="note">' + (loading ? 'Loading this country\'s figures.' : (S.detail[iso] === null ? 'Fewer than 5 posts here, so no detail was built.' : 'No posts for this combination of source, attribution and time window. Try widening the time window or source in Filters.')) + '</p>';
      if (!k && s && s.n > 0) k = s; else return h;
    }
    if (k.insufficient) h += '<p class="banner" style="margin:0 0 10px">Only ' + esc(fn(k.n)) + ' posts, below the ' + T('min_n', 'minimum of ' + esc(get(S.meta, 'min_n') || 30)) + '. Rates swing too much to trust, so the country is not coloured on the map.</p>';
    h += '<div class="pl-stats">' + statBox('Posts', fn(k.n), isNum(k.n_weighted) ? 'weighted ' + fn(Math.round(k.n_weighted)) : '') +
      statBox({ html: T('signal rate', 'Signal rate') }, fp(k.rate), fci(k.ci)) + statBox({ html: T('lift', 'Lift') }, fx(k.lift), isNum(k.lift_raw) ? 'before length adjustment ' + fx(k.lift_raw) : '') +
      statBox({ html: T('debunk rate', 'Debunk rate') }, fp(k.debunk_rate, 2), fn(k.debunk_n) + ' posts') +
      statBox({ html: T('propaganda technique', 'Top technique') }, k.top_technique ? tn(k.top_technique) : '-', isNum(k.top_technique_ratio) ? fx(k.top_technique_ratio) + 'x the global share' : '') +
      statBox('Checked against fact-checks', fn(k.verified_n), fn(k.verified_fake) + ' fake, ' + fn(k.verified_misleading) + ' misleading, ' + fn(k.verified_plausibly_fake) + ' plausibly fake') + '</div>';
    if (isNum(k.agreement_n) && k.agreement_n > 0) {
      var ag = isNum(k.agreement_post) ? Math.round(k.agreement_post * k.agreement_n) : null;
      h += '<p class="note">' + T('Gemma check', 'Gemma check') + ': the second opinion agrees on ' + (ag == null ? '-' : fn(ag)) + ' of ' + fn(k.agreement_n) + ' sampled posts' + (isNum(k.agreement_span) ? ' (average wording overlap ' + fp(k.agreement_span, 0) + ')' : '') + '.</p>';
    } else h += '<p class="note pl-muted">' + T('Gemma check', 'Gemma check') + ': not run on this country yet.</p>';
    if (!d) return h;
    var te = (d.techniques || []).slice().sort(function (a, b) { return (b.share || 0) - (a.share || 0); }).slice(0, 8);
    if (te.length) {
      h += '<div class="pl-sub">Techniques among signal posts (tick = global share)</div>' + bars(te.map(function (t) { return { l: tn(t.technique), h: tterm(t.technique), v: t.share, m: isNum(t.global_share) ? t.global_share : null, r: t.ratio, n: t.n }; }),
        function (i) { return fp(i.v, 0) + (isNum(i.r) ? ' (' + fx(i.r) + 'x)' : ''); });
    }
    var pl = topObj(mergeNames(d.platforms, Labels.site), 6), lg = topObj(mergeNames(d.languages, Labels.lang), 6);
    if (pl.length) h += '<div class="pl-sub">Platforms</div>' + bars(pl, function (i) { return fn(i.v); });
    if (lg.length) h += '<div class="pl-sub">Languages</div>' + bars(lg, function (i) { return fn(i.v); });
    var en = (d.entities || []).slice(0, 10);
    if (en.length) h += '<div class="pl-sub">Entities mentioned</div><div class="pl-chips">' + en.map(function (x) { return '<span class="chip tag" title="' + esc(x.type || '') + '">' + esc(x.name) + ' ' + esc(fn(x.n)) + '</span>'; }).join('') + '</div>';
    var tr = get(S.detail[iso], 'modes', S.mode, 'all', 'trend');
    if (Array.isArray(tr) && tr.length > 1) h += '<div class="pl-sub">Weekly signal rate, last ' + tr.length + ' weeks (all time)</div>' + spark(tr);
    return h;
  }
  function spark(tr) {
    var W = 320, H = 100, L = 30, R = 6, T = 8, B = 20, pts = tr.filter(function (t) { return isNum(t.rate); });
    if (pts.length < 2) return '';
    var mx = niceCeil(Math.max.apply(null, pts.map(function (t) { return t.rate; })) * 1.05), pw = W - L - R, ph = H - T - B;
    var x = function (i) { return L + i / (tr.length - 1) * pw; }, y = function (v) { return T + ph - v / mx * ph; };
    var g = get(S.meta, 'global', S.mode, 'all', S.src, 'rate');
    var d = '', i = 0;
    tr.forEach(function (t, j) { if (isNum(t.rate)) { d += (i++ ? 'L' : 'M') + x(j).toFixed(1) + ' ' + y(t.rate).toFixed(1); } });
    return '<svg class="pl-spark" viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="Weekly signal rate from ' + esc(tr[0].week) + ' to ' + esc(tr[tr.length - 1].week) + '">' +
      '<line class="ax" x1="' + L + '" y1="' + (T + ph) + '" x2="' + (L + pw) + '" y2="' + (T + ph) + '"/><line class="ax" x1="' + L + '" y1="' + T + '" x2="' + L + '" y2="' + (T + ph) + '"/>' +
      (isNum(g) && g <= mx ? '<line class="gl" x1="' + L + '" y1="' + y(g).toFixed(1) + '" x2="' + (L + pw) + '" y2="' + y(g).toFixed(1) + '"><title>global rate ' + esc(fp(g)) + '</title></line>' : '') +
      '<path class="ln" d="' + d + '"/>' +
      '<text x="' + (L - 4) + '" y="' + (T + 4) + '" text-anchor="end">' + esc(pctTick(mx)) + '</text><text x="' + (L - 4) + '" y="' + (T + ph) + '" text-anchor="end">0%</text>' +
      '<text x="' + L + '" y="' + (H - 4) + '">' + esc(tr[0].week) + '</text><text x="' + (L + pw) + '" y="' + (H - 4) + '" text-anchor="end">' + esc(tr[tr.length - 1].week) + '</text></svg>' +
      '<p class="note">Dashed line: rate across all countries. Weeks with few posts are noisy.</p>';
  }

  /* ---------- posts ---------- */
  function textHTML(text, spans) {
    var cp = Array.from(String(text == null ? '' : text)), n = cp.length, out = '', cur = 0;
    var sp = (Array.isArray(spans) ? spans : []).filter(function (s) { return Array.isArray(s) && isNum(s[0]) && isNum(s[1]); })
      .map(function (s) { return [clamp(s[0], 0, n), clamp(s[1], 0, n), s[2]]; }).filter(function (s) { return s[1] > s[0]; })
      .sort(function (a, b) { return a[0] - b[0]; });
    sp.forEach(function (s) {
      if (s[0] < cur) { if (s[1] <= cur) return; s = [cur, s[1], s[2]]; }
      out += esc(cp.slice(cur, s[0]).join('')) + '<mark title="' + esc(tn(s[2])) + '">' + esc(cp.slice(s[0], s[1]).join('')) + '</mark>';
      cur = s[1];
    });
    return out + esc(cp.slice(cur).join(''));
  }
  function authorName(a) { /* some platforms store the profile address as the author: show the handle, not the URL */
    a = String(a);
    if (/^https?:\/\//i.test(a)) { var seg = a.replace(/[?#].*$/, '').replace(/\/+$/, '').split('/').pop(); return seg || 'Unknown author'; }
    return a;
  }
  function vclass(l) { return /^[A-Z_]+$/.test(l || '') ? 'v-' + l : ''; }
  function langName(l) { return Labels.lang(l); }
  function postHTML(p, o) {
    o = o || {};
    if (!p) return '';
    var url = safeUrl(p.url), long = String(p.text || '').length > 260 || (String(p.text || '').match(/\n/g) || []).length > 3, h;
    h = '<article class="pl-post' + (p.signal ? ' sig' : '') + '"><div class="pl-post-top"><span class="badge">' + esc(Labels.site(p.platform || p.source)) + '</span>' +
      (p.author ? '<span>' + esc(authorName(p.author)) + '</span>' : '') + '<span>' + esc(p.posted_at ? Labels.dateLong(p.posted_at) : 'undated') + '</span>' +
      (p.lang ? '<span>' + esc(langName(p.lang)) + '</span>' : '') + (url ? '<a href="' + esc(url) + '" target="_blank" rel="noopener noreferrer">Open original<span class="sr"> (opens in a new tab)</span></a>' : '') + '</div>' +
      '<p class="pl-text' + (long ? ' clamp' : '') + '">' + textHTML(p.text, p.spans) + '</p>' + (long ? '<button type="button" class="pl-more" aria-expanded="false">Show more</button>' : '');
    var tags = (Array.isArray(p.techniques) ? p.techniques : []).map(function (t) { return '<span class="chip tag"' + (tdef(t) ? ' title="' + esc(tdef(t)) + '"' : '') + '>' + esc(tn(t)) + '</span>'; }).join('');
    if (p.signal && !tags) tags = '<span class="chip tag">Propaganda signal</span>';
    if (tags) h += '<div class="pl-post-tags">' + tags + '</div>';
    if (p.debunk) {
      var u = safeUrl(p.debunk.kb_url);
      h += '<p class="pl-kb"><span class="chip ' + vclass(p.debunk.kb_label) + '">' + esc(vname(p.debunk.kb_label || 'fact-check')) + '</span> Close to a fact-checked claim' + (isNum(p.debunk.rerank) ? ' (match strength ' + fx(p.debunk.rerank) + ')' : '') + ': ' + esc(p.debunk.kb_claim || '-') + (u ? ' <a href="' + esc(u) + '" target="_blank" rel="noopener noreferrer">Read the fact-check<span class="sr"> (opens in a new tab)</span></a>' : '') + '</p>';
    }
    var vf = p.verified;
    if (vf && typeof vf === 'object') {
      h += '<div class="pl-kb"><span class="chip ' + vclass(vf.label) + '">' + esc(vname(vf.label || 'unrated')) + '</span>' + (vf.plausibly_fake ? ' <span class="chip tag">Plausibly fake</span>' : '') +
        (isNum(vf.fake_score) ? ' <span class="chip tag" title="Plausibility score from 0 to 1">Score ' + esc(fx(vf.fake_score)) + '</span>' : '') + (vf.basis && vf.basis !== 'none' ? ' <span class="pl-muted">' + esc(Labels.verdictBasis(vf.basis)) + '</span>' : '') +
        (vf.rationale ? '<details><summary>Why it was rated this way</summary><p>' + esc(vf.rationale) + '</p></details>' : '') + '</div>';
    }
    var gc = p.gemma_check;
    if (gc && typeof gc === 'object') {
      var ga = gc.agreement || null, agrees = ga ? (ga.gemma_flags != null && ga.encoder_flags != null ? !!ga.gemma_flags === !!ga.encoder_flags : (isNum(ga.overlap) ? ga.overlap >= 0.5 : null)) : null;
      var gs = Array.isArray(gc.gemma_spans) ? gc.gemma_spans : [], snip = '';
      if (Array.isArray(p.spans) && p.spans.length) {
        var cpt = Array.from(String(p.text || ''));
        snip = gs.filter(function (x) { return Array.isArray(x) && isNum(x[0]) && isNum(x[1]) && x[1] > x[0]; }).slice(0, 4).map(function (x) {
          return '<mark class="gm" title="' + esc(tn(x[2])) + '">' + esc(cpt.slice(clamp(x[0], 0, cpt.length), clamp(x[1], 0, cpt.length)).join('')) + '</mark>';
        }).join(' ');
      }
      h += '<p class="pl-kb pl-muted">' + T('Gemma check', 'Gemma check') + ': ' + (agrees == null ? 'unclear' : agrees ? 'agrees with the flagged wording' : 'differs from the flagged wording') + (ga && isNum(ga.overlap) ? ' (overlap ' + fp(ga.overlap, 0) + ')' : '') + (snip ? '. Also flagged: ' + snip : '') + '</p>';
    }
    if (o.flag && typeof p.post_uid === 'string' && p.post_uid) h += '<div class="pl-fb" data-uid="' + esc(p.post_uid) + '">' + flagInner(p.post_uid) + '</div>';
    return h + '</article>';
  }
  /* ---------- feedback (in-memory only, this page session) ---------- */
  var FLAGS = {};
  function flagInner(uid) {
    var f = FLAGS[uid] || {}, dis = f.busy ? ' disabled' : '', h;
    if (f.done && f.done !== 'clear') {
      h = '<span class="pl-fb-ok" role="status">Thanks, recorded (' + (f.done === 'confirm' ? 'confirmed' : 'not propaganda') + ')</span>' +
        '<button type="button" class="btn ghost pl-fb-b" data-fb="clear" aria-label="Undo my feedback on this post"' + dis + '>Undo</button>';
    } else {
      h = '<span class="pl-fb-l">Is this propaganda?</span>' +
        '<button type="button" class="btn ghost pl-fb-b" data-fb="not_propaganda" aria-label="Mark this post as not propaganda"' + dis + '>Not propaganda</button>' +
        '<button type="button" class="btn ghost pl-fb-b" data-fb="confirm" aria-label="Confirm this post as propaganda"' + dis + '>Confirm</button>';
    }
    if (f.busy) h += '<span class="pl-muted" role="status">Sending</span>';
    if (f.err) h += '<span class="pl-fb-err" role="alert">Could not record, try again' + (f.err !== true ? ': ' + esc(f.err) : '') + '</span>';
    return h;
  }
  function flagBox(uid) {
    var all = document.querySelectorAll('#pl-post-list .pl-fb');
    for (var i = 0; i < all.length; i++) if (all[i].getAttribute('data-uid') === uid) return all[i];
    return null;
  }
  function sendFlag(uid, action) {
    var f = FLAGS[uid] = FLAGS[uid] || {};
    if (f.busy) return;
    f.busy = true; f.err = null;
    var body = { post_uid: uid, action: action };
    if (action === 'confirm' && S.posts.fT) body.technique = S.posts.fT;
    var box = flagBox(uid), had = box && box.contains(document.activeElement);
    if (box) box.innerHTML = flagInner(uid);
    fetch(API.replace('proplab/', 'proplab/flag'), { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
      .then(function (r) { return r.json().catch(function () { return {}; }).then(function (j) { return { ok: r.ok && j && j.ok !== false, j: j || {} }; }); })
      .then(function (x) {
        f.busy = false;
        if (x.ok) { f.done = action; f.err = null; } else f.err = (x.j && typeof x.j.error === 'string' && x.j.error) || true;
      }, function () { f.busy = false; f.err = true; })
      .then(function () {
        var b = flagBox(uid); if (!b) return;
        b.innerHTML = flagInner(uid);
        if (had) { var nb = b.querySelector('button:not([disabled])'); if (nb) nb.focus(); }
      });
  }
  function cPostsShell(iso) {
    var opts = '<option value="">All techniques</option>' + (get(S.meta, 'techniques') || []).map(function (t) { return '<option value="' + esc(t) + '"' + (S.posts.fT === t ? ' selected' : '') + '>' + esc(tn(t)) + '</option>'; }).join('');
    return '<div class="pl-filters"><label class="f"><span>Technique</span><select id="pl-f-tech">' + opts + '</select></label>' +
      '<label class="f"><span>Show</span><span class="pl-chk"><input type="checkbox" id="pl-f-sig"' + (S.posts.sigOnly ? ' checked' : '') + '> Propaganda-signal posts only</span></label></div>' +
      '<p class="note" id="pl-post-count" aria-live="polite"></p><div id="pl-post-list"></div><div id="pl-post-more" style="margin-top:10px"></div>';
  }
  function postKey() { return S.mode + '|' + S.sel; }
  function initPosts(iso) {
    if (S.posts.key !== postKey()) { S.posts = { key: postKey(), items: [], next: 0, more: false, loading: false, err: '', fT: S.posts.fT, sigOnly: S.posts.sigOnly, auto: 0, done: false }; loadPosts(true); }
    else drawPosts();
  }
  function passes(p) {
    var th = get(S.meta, 'data_through'), days = S.win === '7d' ? 7 : S.win === '30d' ? 30 : 0;
    if (days && th && p.posted_at) { var t = Date.parse(p.posted_at), c = Date.parse(th) - days * 864e5; if (!isNaN(t) && !isNaN(c) && t < c) return false; }
    if (S.src !== 'all' && p.source && p.source !== S.src) return false;
    if (S.posts.fT && (p.techniques || []).indexOf(S.posts.fT) < 0) return false;
    if (S.posts.sigOnly && !p.signal) return false;
    return true;
  }
  function drawPosts() {
    var list = $('#pl-post-list'); if (!list) return;
    var P = S.posts, shown = P.items.filter(passes);
    var h = shown.map(function (p) { return postHTML(p, { flag: true }); }).join('');
    if (!shown.length) h = '<p class="note">' + (P.loading ? 'Loading posts.' : P.err ? esc(P.err) : P.items.length ? 'No loaded posts match these filters. Load more posts or relax the filters.' : 'No posts for this country in this view.') + '</p>';
    list.innerHTML = h;
    $('#pl-post-count').textContent = 'Showing ' + shown.length + ' of ' + P.items.length + ' loaded posts, strongest signal first (' + (WIN_NAME[S.win] || S.win).toLowerCase() + ', ' + (SRC_NAME[S.src] || S.src).toLowerCase() + ').';
    $('#pl-post-more').innerHTML = P.more ? '<button class="btn ghost" type="button" id="pl-load-more"' + (P.loading ? ' disabled' : '') + '>' + (P.loading ? 'Loading' : 'Load more') + '</button>' : '';
  }
  function loadPosts(first) {
    var P = S.posts, key = P.key;
    if (P.loading) return;
    P.loading = true; P.err = ''; drawPosts();
    fetchJSON('posts?mode=' + encodeURIComponent(S.mode) + '&iso=' + encodeURIComponent(S.sel) + '&offset=' + P.next + '&limit=50').then(function (r) {
      if (S.posts.key !== key) return;
      P.items = P.items.concat(r.posts || []); P.next = isNum(r.next_offset) ? r.next_offset : P.next + 50; P.more = !!r.has_more;
    }).catch(function (e) { if (S.posts.key === key) { P.err = e.status === 404 ? 'No posts are stored for this country.' : 'Posts could not be loaded. Try again in a moment.'; P.more = false; } })
      .then(function () {
        if (S.posts.key !== key) return;
        P.loading = false; drawPosts();
        if (P.more && P.items.filter(passes).length < 8 && P.auto++ < 3) loadPosts(false);
      });
  }

  /* ---------- narratives ---------- */
  function cNarratives(iso, d) {
    var ns = get(d, 'narratives') || [], h = '';
    if (!ns.length) return '<p class="note">No narratives were found for this country in this view. Narratives need enough posts to form a group; try the all-time window.</p>';
    if (isNum(d.narratives_coverage)) h += '<p class="note">' + T('narrative', 'Narratives') + ' cover ' + fp(d.narratives_coverage, 0) + ' of this country\'s posts.</p>';
    ns.forEach(function (n) {
      var nm = n.name || n.terms || ('Unnamed group ' + (n.cluster != null ? Number(n.cluster) + 1 : '')), ex = (n.exemplars || []).filter(Boolean);
      h += '<div class="pl-narr"><h3>' + esc(nm) + '</h3>' + (n.summary ? '<p class="note">' + esc(n.summary) + '</p>' : '') + (n.name && n.terms && n.name !== n.terms ? '<p class="terms">' + esc(n.terms) + '</p>' : '') +
        '<div class="pl-chips"><span class="chip tag">' + esc(fn(n.n)) + ' posts' + (isNum(n.n_unique) ? ', ' + esc(fn(n.n_unique)) + ' distinct' : '') + '</span>' +
        (n.templated ? '<span class="chip tag pl-templ" tabindex="0" title="' + esc((isNum(n.n_unique) ? fn(n.n_unique) : 'few') + ' distinct texts out of ' + fn(n.n) + ' posts') + '" aria-label="Templated: ' + esc((isNum(n.n_unique) ? fn(n.n_unique) : 'few') + ' distinct texts out of ' + fn(n.n) + ' posts') + '">Templated</span>' : '') +
        '<span class="chip tag">Signal ' + esc(fp(n.rate, 0)) + '</span>' +
        (n.top_technique ? '<span class="chip tag"' + (tdef(n.top_technique) ? ' title="' + esc(tdef(n.top_technique)) + '"' : '') + '>' + esc(tn(n.top_technique)) + '</span>' : '') +
        (isNum(n.cohesion) ? '<span class="chip tag" title="' + esc((Labels.glossaryEntry('cohesion') || {}).definition || '') + '">Cohesion ' + esc(fx(n.cohesion)) + '</span>' : '') + '</div>' +
        (ex.length ? '<details><summary>Example posts (' + ex.length + ')</summary>' + ex.map(function (p) { return postHTML(p); }).join('') + '</details>' : '') + '</div>';
    });
    return h;
  }

  /* ---------- fake / misleading ---------- */
  function cFake(iso, d, s) {
    var fl = get(d, 'fake_layer'), k = get(d, 'kpis', S.src) || s, h = '';
    if (!fl) return '<p class="note">No fake or misleading detail for this country in this view.</p>';
    h += '<div class="pl-stats">' + statBox('Posts near a debunked claim', fn(fl.debunk_n), 'of ' + fn(fl.n)) + statBox({ html: T('debunk rate', 'Debunk rate') }, fp(get(k, 'debunk_rate'), 2), fci(get(k, 'debunk_ci'))) + '</div>';
    var v = fl.verified;
    if (v && isNum(v.n) && v.n > 0) {
      h += '<div class="pl-sub">' + fn(v.n) + ' checked against fact-checks</div><div class="pl-chips">' + Object.keys(v).filter(function (l) { return l !== 'n' && isNum(v[l]) && v[l] > 0; }).map(function (l) { return '<span class="chip ' + vclass(l) + '">' + esc(vname(l)) + ' ' + esc(fn(v[l])) + '</span>'; }).join('') + '</div>';
    } else h += '<p class="note">No posts from this country have been checked against fact-checks yet.</p>';
    var vp = (fl.verified_posts || []).filter(Boolean);
    if (vp.length) h += '<div class="pl-sub">Posts checked against fact-checks</div>' + vp.map(function (p) { return postHTML(p); }).join('');
    var ms = (fl.matches || []).filter(Boolean);
    h += '<div class="pl-sub">Posts close to a claim rated fake or misleading</div>';
    if (!ms.length) h += '<p class="note">No close matches in this view.</p>';
    ms.forEach(function (m) {
      var u = safeUrl(m.kb_url);
      h += '<article class="pl-post sig"><p class="pl-text clamp">' + esc(m.text || '-') + '</p><button type="button" class="pl-more" aria-expanded="false">Show more</button>' +
        '<p class="pl-kb"><span class="chip ' + vclass(m.kb_label) + '">' + esc(vname(m.kb_label || 'fact-check')) + '</span> ' + esc(m.kb_claim || '-') + (isNum(m.rerank) ? ' (match strength ' + esc(fx(m.rerank)) + ')' : '') + (u ? ' <a href="' + esc(u) + '" target="_blank" rel="noopener noreferrer">Read the fact-check<span class="sr"> (opens in a new tab)</span></a>' : '') + '</p></article>';
    });
    return h + '<p class="note">A match is a lead for review, not a ruling on the post.</p><details class="pl-rule"><summary>How a match is decided</summary><p>' + esc(get(S.meta, 'method', 'debunk_rule') || 'Rule not available in this build.') + '</p></details>';
  }

  /* ---------- how to read (explainer card, collapsible, remembered) ---------- */
  var METHOD_NAME = { engine: 'Model', signal_rule: 'What counts as a signal post', debunk_rule: 'What counts as a debunk match', about: '"Mentioned" attribution', origin: '"Posted from" attribution', lift: 'How lift is calculated', narratives: 'How narratives are found', windows: 'Time windows' };
  function caveat(c) { /* customer wording: no internal service names, no shouting */
    c = String(c == null ? '' : c);
    if (/shaktidb|reachable/i.test(c)) return 'News articles: none yet. The news source is not connected, so the Lab covers social posts only.';
    return c.replace(/\bSIGNAL\b/g, 'signal');
  }
  function renderHow() {
    var m = S.meta || {}, cav = (m.caveats || []).map(caveat).filter(Boolean);
    var h = cav.length ? '<ul class="pl-cav">' + cav.map(function (c) { return '<li>' + esc(c) + '</li>'; }).join('') + '</ul>' : '<p class="note">No caveats were published with this build.</p>';
    h += '<p class="pl-terms"><span class="lbl">Key terms</span>' + [['lift', 'Lift'], ['signal rate', 'Signal rate'], ['min_n', 'Minimum posts'], ['span encoder', 'Span encoder'], ['Gemma check', 'Gemma check']].map(function (t) { return T(t[0], t[1]); }).join('') + '</p>';
    if (m.n_gemma_verified > 0 || m.n_gemma_span_checked > 0) h += '<p class="note">Checked against fact-checks: ' + fn(m.n_gemma_verified) + ' posts. ' + T('Gemma check', 'Gemma check') + ': ' + (m.n_gemma_span_checked > 0 ? fn(m.n_gemma_span_checked) + ' posts sampled.' : 'not run yet.') + '</p>';
    h += '<p class="note">Gemma agrees a post is persuasive about 85% of the time, but often on different phrases. The exact phrases it marks overlap the encoder\'s only about 25 to 30% of the time.</p>';
    h += '<p class="note">Verified verdicts come from posts chosen because they matched a fact-check, so they skew FAKE and are not prevalence. They are counts of posts checked against fact-checks, never a rate over all posts.</p>';
    if (m.method) {
      h += '<details class="pl-rule"><summary>Method details</summary><dl>' + Object.keys(m.method).map(function (k) { return '<dt>' + esc(METHOD_NAME[k] || k.replace(/_/g, ' ')) + '</dt><dd>' + esc(m.method[k]) + '</dd>'; }).join('') + '</dl></details>';
    }
    put($('#pl-how-body'), h);
  }
  function setHow(open, save) {
    $('#pl-how-body').hidden = !open; var b = $('#pl-how-toggle');
    b.setAttribute('aria-expanded', open ? 'true' : 'false'); b.textContent = open ? 'Hide' : 'Show';
    $('#pl-how').classList.toggle('closed', !open);
    if (save) lsSet('fnp-lab-how', open ? 'open' : 'closed');
  }
  $('#pl-how-toggle').addEventListener('click', function () { setHow($('#pl-how-body').hidden, true); });
  setHow(lsGet('fnp-lab-how') !== 'closed', false);

  function renderAll() {
    if (!S.loaded) return;
    renderFresh(); renderKpis(); renderControls(); renderViz(); renderBoard(); renderCountry(); renderHow();
  }

  /* Escape closes an open country file, after popovers, the tooltip and the search boxes have had their turn */
  document.addEventListener('keydown', function (e) {
    if (e.key !== 'Escape' || !S.active || !S.sel || e.defaultPrevented) return;
    var t = e.target;
    if (t && t.closest && t.closest('#pl-filters, #gs-list')) return;
    e.preventDefault(); goBack();
  });

  window.PropLab = {
    /* select(iso) follows the hash (#lab/<ISO3>); select(null) clears the selection */
    select: function (iso) {
      if (!iso) { pushed = false; if (S.sel) { S.sel = null; syncSel(); if (S.loaded) { paintMap(); renderBoard(); if (S.view === 'heat') renderHeat(); renderCountry(); } } return; }
      if (!/^[A-Z]{3}$/.test(iso)) return;
      if (S.sel === iso) { if (S.loaded && window.Shell) window.Shell.setTitle(name(iso)); return; }
      if (!S.loaded) { S.sel = iso; S.ctab = 'overview'; S.posts.key = ''; syncSel(); return; }
      select(iso, false);
    },
    setActive: function (on) {
      S.active = !!on;
      if (on) { if (!S.loaded) load(true); else load(false).then(function () { if (S.view === 'map') paintMap(); }); }
    }
  };
})();
