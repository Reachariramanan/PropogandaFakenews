/* Evidence Base: Workbench and Insights tabs (routes #evidence/workbench and #evidence/insights). Plugs into evidence.js through Evidence.register and Evidence.h.
   Workbench: find items | selection and runs | result. Selection (at most 25) -> POST /api/rt/analyse -> poll GET /api/rt/jobs/<id> every 2 s while the tab is visible ->
   per-item results drawn with ClaimCheck.renderResult (verdict bars, 14 technique bars, highlighted spans). Items are run one at a time by the server.
   Insights: POST /api/rt/insights -> template insights that cite rows or numbers from the connected database, plus an optional Gemma phrasing that the server has already validated.
   Everything in an item (title, text, source, technique names, spans) is untrusted data and only ever reaches the page through esc(). No external requests, no files downloaded.
   Nothing is kept on the server beyond the stored analysis rows; in this browser only named selections are saved (localStorage, inside try/catch; the page works without it). */
(function () {
  'use strict';
  var Ev = window.Evidence, H = Ev && Ev.h, L = window.Labels, F = window.FNPUI, C = window.Case, CC = window.ClaimCheck, Shell = window.Shell;
  if (!H || !CC) return;
  var $ = function (s, r) { return (r || document).querySelector(s); };
  var esc = L.esc, fmtN = F.fmtN, term = L.term, n2 = H.n2;
  var MAX_SEL = 25, POLL_MS = 2000, STORE_KEY = 'fnp-wb-selections', MAX_SETS = 20, MAX_ENGINE = 8000;
  var TERMINAL = { done: 1, cached: 1, degraded: 1, skipped: 1, error: 1 };
  var el = { wb: $('#ev-workbench'), ins: $('#ev-insights') };

  /* ---------- helpers ---------- */
  function call(method, path, body) {
    var o = { method: method, headers: { Accept: 'application/json' } };
    if (body !== undefined) { o.headers['Content-Type'] = 'application/json'; o.body = JSON.stringify(body); }
    return fetch(path, o).then(function (r) {
      return r.json().catch(function () { return {}; }).then(function (j) {
        if (!r.ok) { var e = new Error((j && j.error) || ('HTTP ' + r.status)); e.status = r.status; e.code = j && j.error; e.hint = j && j.hint; e.retry = r.headers.get('Retry-After'); throw e; }
        return j;
      });
    }, function () { var e = new Error('network'); e.network = true; throw e; });
  }
  function ob(cls, text, title) { return '<span class="ob ' + cls + '"' + (title ? ' title="' + esc(title) + '"' : '') + '>' + esc(text) + '</span>'; }
  var P = window.FNPPure, secs = P.secs, clip = P.clip;
  function bandName(p) { return p >= 0.8 ? 'high' : p >= 0.5 ? 'moderate' : 'low'; }
  function metaOf(it) { return { id: it.id, title: clip(H.itemTitle(it), 160), kind: it.kind, platform: it.platform, source: it.source, lang: it.lang, published_at: it.published_at }; }
  function compose(it) { var t = String(it.title || '').trim(), e = String(it.text_excerpt || '').trim(), s = t && e ? t + '\n\n' + e : (t || e); return s.slice(0, MAX_ENGINE).trim(); }
  function keepFocus(fn) {
    var a = document.activeElement, k = a && a.getAttribute && a.getAttribute('data-fk');
    fn();
    if (k) { var n = document.querySelector('[data-fk="' + k.replace(/"/g, '') + '"]'); if (n && n !== document.activeElement) n.focus({ preventScroll: true }); }
  }
  function live(t) { var n = $('#wb-live'); if (n) { n.textContent = ''; setTimeout(function () { n.textContent = t; }, 30); } }
  function errText(e) {
    if (e.network) return 'The server could not be reached. Check your connection and try again.';
    if (e.status === 429) return (e.message && !/^HTTP/.test(e.message) ? e.message : 'Too many requests') + '. Try again in ' + (e.retry ? e.retry + ' seconds' : 'a moment') + '.';
    return e.message || 'Something went wrong.';
  }

  /* =====================================================================================================================
     WORKBENCH STATE (module level: survives switching tabs, lost on reload)
     ===================================================================================================================== */
  var S = {
    sel: [],          /* [{id,title,kind,platform,source,lang,published_at}] in the order added */
    res: {},          /* id -> {state, reason, secs, result, cached_at, engine_state, retry_after_s, jobId} from a run in this session */
    items: {},        /* id -> full /api/rt/item record, for the result panel */
    itemErr: {},
    view: null,       /* id shown in the Result panel */
    run: null,        /* current or last job: {id, ids, state, total, done, items:{id:{...}}, qpos, cancelling, force, needFull, fails} */
    off: 0,           /* server time minus browser time (ms), from the job response */
    sort: { key: '', dir: 'desc' },
    cmp: null,        /* selection comparison: {state, rows:{id:...}, done, total} */
    cmpFor: null,     /* id whose Compare section is open in the Result panel */
    note: null, find: null, polling: false, pollT: null, tickT: null, active: false, built: false, sig: ''
  };
  var Fd = { q: '', kind: '', platform: '', topic: '', lang: '', analysed: '', sort: '', page: 0, pages: 1, total: null, capped: false, items: [], loading: false, seq: 0, error: null };
  var SORTS = [['', 'Best match'], ['recent', 'Newest first'], ['p_prop', 'Top propaganda signal'], ['p_fake', 'Top fake probability']];

  function inSel(id) { return P.selHas(S.sel, id); }
  function selAdd(m) { return P.selAdd(S.sel, m, MAX_SEL); }
  function selRemove(id) {
    P.selRemove(S.sel, id);
    if (S.view === id) { S.view = S.sel.length ? S.sel[0].id : null; }
    S.cmp = null;
  }
  function runActive() { return !!(S.run && (S.run.state === 'queued' || S.run.state === 'running')); }
  function stateOf(id) {
    var r = S.run && S.run.items[id], rs = S.res[id];
    if (r && (!TERMINAL[r.state] || !rs || rs.jobId === S.run.id)) return r;
    return rs || null;
  }
  function resultOf(id) {
    var rs = S.res[id];
    if (rs && rs.result && (rs.state === 'done' || rs.state === 'cached' || rs.state === 'degraded')) return { src: rs.state === 'cached' ? 'cached' : 'run', degraded: rs.state === 'degraded', result: rs.result, at: rs.cached_at || null, secs: rs.secs };
    var it = S.items[id], a = it && it.analyses && it.analyses[0];
    if (a && a.result && a.result.step1) return { src: 'cached', degraded: !!a.result.partial, partial: !!a.result.partial, result: a.result, at: a.created_at, stored: true };
    return null;
  }
  function summarise(id) {
    var r = resultOf(id); if (!r) return null;
    var res = r.result, s1 = res.step1 || {}, p = res.step2 && res.step2.shown && res.step2.propaganda ? res.step2.propaganda : null, cls = p && Array.isArray(p.classes) ? p.classes : [];
    var found = cls.filter(function (c) { return Number(c.probability) > 0; }).sort(function (a, b) { return b.probability - a.probability; });
    var n = null, top = null;
    if (p) {
      if (cls.length) n = found.length; else { var seen = {}; (p.techniques || []).forEach(function (t) { seen[L.technique(t.technique)] = 1; }); n = Object.keys(seen).length; }
      if (found[0]) top = { name: L.technique(found[0].technique), p: found[0].probability };
    }
    return { label: s1.label || null, plaus: s1.plausibility == null ? null : Number(s1.plausibility), n: n, top: top, src: r.src, degraded: r.degraded, at: r.at };
  }

  /* ---------- named selections in this browser (localStorage inside try/catch; an in-memory copy keeps the feature working without storage) ---------- */
  var memSets = [], storeOk = true;
  function loadSets() {
    try {
      var raw = localStorage.getItem(STORE_KEY); if (!raw) return memSets.slice();
      var j = JSON.parse(raw), a = j && Array.isArray(j.sets) ? j.sets : [];
      return a.filter(function (s) { return s && typeof s.name === 'string' && Array.isArray(s.items); }).slice(0, MAX_SETS);
    } catch (e) { storeOk = false; return memSets.slice(); }
  }
  function saveSets(a) {
    memSets = a.slice();
    try { localStorage.setItem(STORE_KEY, JSON.stringify({ version: 1, sets: a })); storeOk = true; } catch (e) { storeOk = false; }
  }

  /* =====================================================================================================================
     BUILD
     ===================================================================================================================== */
  function build() {
    if (S.built) return; S.built = true;
    el.wb.innerHTML =
      '<div class="wb" id="wb">' +
        '<p class="wb-intro">Pick items from the Realtime Database, run Claim Check on them and read each result. Claim Check reads the title and stored excerpt of an item: it gives the fake-or-not verdict with a probability for each class, and a probability for each of the 14 propaganda techniques.</p>' +
        '<div class="wb-grid">' +
          '<section class="panel wb-col wb-find" id="wb-find" aria-labelledby="wb-find-h">' +
            '<header class="wb-h"><h2 id="wb-find-h" class="panel-title">Find items</h2><p class="wb-sub">Search the stored items and add the ones you want to check.</p></header>' +
            '<div class="wb-find-ctl">' +
              '<div class="dl-q" role="search" aria-label="Find items"><label class="sr" for="wb-q">Search titles and text</label>' +
                '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="10.5" cy="10.5" r="6"/><path d="m15 15 5.5 5.5"/></svg>' +
                '<input type="search" id="wb-q" placeholder="Search titles and text" autocomplete="off" maxlength="300"></div>' +
              '<div class="vchips" id="wb-kind" role="group" aria-label="Filter by kind">' + [['', 'All'], ['news', 'News'], ['social', 'Social']].map(function (c) { return '<button type="button" class="vchip" data-kind="' + c[0] + '" aria-pressed="false">' + esc(c[1]) + '</button>'; }).join('') + '</div>' +
              '<div class="wb-sels">' +
                '<label class="f"><span class="sr">Platform</span><select id="wb-platform"><option value="">All platforms</option></select></label>' +
                '<label class="f"><span class="sr">Topic</span><select id="wb-topic"><option value="">All topics</option></select></label>' +
                '<label class="f"><span class="sr">Language</span><select id="wb-lang"><option value="">All languages</option></select></label>' +
                '<label class="f"><span class="sr">Analysed state</span><select id="wb-analysed"><option value="">Any state</option><option value="yes">Analysed (batch or deep)</option><option value="no">Not analysed yet</option></select></label>' +
                '<label class="f"><span class="sr">Sort</span><select id="wb-sort">' + SORTS.map(function (s) { return '<option value="' + s[0] + '">' + esc(s[1]) + '</option>'; }).join('') + '</select></label>' +
              '</div>' +
              '<div class="wb-find-line"><span class="dl-count" id="wb-count" role="status" aria-live="polite">Loading</span>' +
                '<button type="button" class="btn ghost sm" id="wb-addall" data-fk="addall">Add all shown (max 25)</button></div>' +
            '</div>' +
            '<ul class="wb-list" id="wb-list" aria-label="Items found"></ul>' +
            '<div class="wb-foot" id="wb-more"></div>' +
          '</section>' +
          '<section class="panel wb-col wb-sel" id="wb-sel" aria-labelledby="wb-sel-h">' +
            '<header class="wb-h"><h2 id="wb-sel-h" class="panel-title">Selection and runs</h2><p class="wb-sub" id="wb-sel-sub"></p></header>' +
            '<div class="wb-tools" role="toolbar" aria-label="Selection tools">' +
              '<button type="button" class="btn" id="wb-run" data-fk="run">Run Claim Check on selection</button>' +
              '<button type="button" class="btn ghost" id="wb-cmp" data-fk="cmp">Compare with the Fake News Database</button>' +
              '<button type="button" class="btn ghost" id="wb-ins" data-fk="ins">Insights for selection</button>' +
              '<button type="button" class="btn ghost" id="wb-copy" data-fk="copy">Copy results as JSON</button>' +
            '</div>' +
            '<p class="wb-why" id="wb-why"></p>' +
            '<label class="wb-force"><input type="checkbox" id="wb-force"> <span>Ignore stored results and run again <small>(uses the engine even for items already checked)</small></span></label>' +
            '<div id="wb-note" role="status"></div>' +
            '<div id="wb-prog"></div>' +
            '<h3 class="wb-h3">Selection <span class="wb-n" id="wb-seln"></span></h3>' +
            '<div id="wb-tray"></div>' +
            '<details class="wb-saved" id="wb-saved"><summary>Saved selections</summary><div class="wb-saved-in" id="wb-saved-in"></div></details>' +
            '<div id="wb-table"></div>' +
            '<div id="wb-cmpout"></div>' +
            '<div id="wb-json"></div>' +
          '</section>' +
          '<section class="panel wb-col wb-res" id="wb-resp" aria-labelledby="wb-res-h">' +
            '<header class="wb-h"><h2 id="wb-res-h" class="panel-title">Result</h2><p class="wb-sub">Select an item in the selection or the table to read its result.</p></header>' +
            '<div id="wb-result"></div>' +
          '</section>' +
        '</div><div class="sr" id="wb-live" aria-live="polite"></div>' +
      '</div>';
    wire();
  }

  /* =====================================================================================================================
     FIND ITEMS
     ===================================================================================================================== */
  function fillSelects() {
    var f = H.facets(); if (!f) return;
    var fill = function (id, first, items, val, label) {
      var s = $(id), cur = s.value;
      s.innerHTML = '<option value="">' + esc(first) + '</option>' + items.map(function (x) { return '<option value="' + esc(val(x)) + '">' + esc(label(x)) + '</option>'; }).join('');
      s.value = cur;
    };
    fill('#wb-platform', 'All platforms', f.by_platform || [], function (x) { return x.platform; }, function (x) { return L.platform(x.platform) + ' (' + fmtN(x.count) + ')'; });
    var tp = f.by_topic || { top: [], unclassified: { count: 0 } };
    fill('#wb-topic', 'All topics', tp.top.concat([{ topic_id: 'unclassified', name: 'Unclassified', count: tp.unclassified.count }]), function (x) { return x.topic_id; }, function (x) { return x.name + ' (' + fmtN(x.count) + ')'; });
    fill('#wb-lang', 'All languages', (f.by_language && f.by_language.top) || [], function (x) { return x.lang; }, function (x) { return L.lang(x.lang) + ' (' + fmtN(x.count) + ')'; });
  }
  function findParams(page) {
    var p = new URLSearchParams(), set = function (k, v) { if (v) p.set(k, v); };
    set('q', Fd.q.trim()); set('kind', Fd.kind); set('platform', Fd.platform); set('topic', Fd.topic); set('lang', Fd.lang); set('analysed', Fd.analysed); set('sort', Fd.sort);
    p.set('size', 25); p.set('page', page);
    return p.toString();
  }
  function findRow(it) {
    var added = inSel(it.id), full = !added && S.sel.length >= MAX_SEL, t = itemTitleOf(it), k = it.kind === 'news' ? 'news' : 'social';
    return '<li class="wb-fr' + (added ? ' in' : '') + '" data-id="' + esc(it.id) + '">' +
      '<button type="button" class="wb-fr-open" data-act="view" data-id="' + esc(it.id) + '" data-fk="v:' + esc(it.id) + '"><span class="wb-fr-t">' + esc(t) + '</span>' +
      '<span class="wb-fr-m"><span class="ev-kind" title="' + esc(H.kindName(it.kind)) + '">' + H.ICON[k] + '<span class="sr">' + esc(H.kindName(it.kind)) + '. </span></span><span class="ev-plat" title="Source table: ' + esc(L.platformTable(it.platform)) + '">' + esc(L.platform(it.platform)) + '</span><span aria-hidden="true">&middot;</span><span class="wb-fr-s">' + esc(H.sourceName(it.source)) + '</span><span aria-hidden="true">&middot;</span><span>' + esc(L.lang(it.lang)) + '</span><span aria-hidden="true">&middot;</span><span>' + esc(it.published_at ? H.ago(it.published_at) : 'undated') + '</span></span>' +
      '<span class="wb-fr-a">' + H.scoreHTML(it.inferred) + '</span></button>' +
      '<button type="button" class="btn sm ' + (added ? '' : 'ghost') + ' wb-add" data-act="toggle" data-id="' + esc(it.id) + '" data-fk="a:' + esc(it.id) + '" aria-pressed="' + added + '"' + (full ? ' disabled title="The selection holds at most ' + MAX_SEL + ' items"' : '') +
      ' aria-label="' + (added ? 'Remove from selection: ' : 'Add to selection: ') + esc(clip(t, 80)) + '">' + (added ? 'Added' : 'Add') + '</button></li>';
  }
  function itemTitleOf(it) { return clip(H.itemTitle(it), 200); }
  function renderFind() {
    var box = $('#wb-list'); if (!box) return;
    keepFocus(function () {
      var h = Fd.items.map(findRow).join('');
      if (!Fd.items.length && Fd.loading) h = '<li class="wb-fr sk" aria-hidden="true"><span class="skel w1"></span><span class="skel w2"></span></li><li class="wb-fr sk" aria-hidden="true"><span class="skel w1"></span><span class="skel w2"></span></li><li class="wb-fr sk" aria-hidden="true"><span class="skel w1"></span><span class="skel w2"></span></li>';
      if (!Fd.items.length && !Fd.loading) h = '<li class="wb-empty">' + (Fd.error ? '<b>The list could not be loaded</b><span>' + esc(Fd.error) + '</span><button type="button" class="btn ghost sm" data-act="find-retry">Try again</button>' : '<b>No items match these filters</b><span>Try fewer words or remove a filter.</span>') + '</li>';
      box.innerHTML = h; box.setAttribute('aria-busy', Fd.loading ? 'true' : 'false');
    });
    var cnt = $('#wb-count');
    cnt.textContent = Fd.loading && !Fd.items.length ? 'Loading' : Fd.total == null ? '' : Fd.total === 0 ? 'No items' : (Fd.capped ? '10,000+' : fmtN(Fd.total)) + (Fd.total === 1 ? ' item' : ' items') + ', showing ' + fmtN(Fd.items.length);
    var all = $('#wb-addall'), room = MAX_SEL - S.sel.length, addable = Fd.items.filter(function (x) { return !inSel(x.id); }).length;
    all.disabled = !Fd.items.length || !addable || room <= 0;
    all.title = room <= 0 ? 'The selection is full' : !addable && Fd.items.length ? 'Everything shown is already selected' : '';
    $('#wb-more').innerHTML = Fd.page < Fd.pages && Fd.items.length ? '<button type="button" class="btn ghost sm" data-act="find-more" data-fk="more"' + (Fd.loading ? ' disabled' : '') + '>' + (Fd.loading ? 'Loading' : 'Show more items') + '</button>' : (Fd.items.length ? 'End of the list.' : '');
  }
  function fetchFind(replace) {
    var my = ++Fd.seq; Fd.loading = true; Fd.error = null;
    if (replace) { Fd.page = 0; Fd.pages = 1; }
    var page = replace ? 1 : Fd.page + 1;
    renderFind();
    H.rt('/api/rt/items?' + findParams(page)).then(function (d) {
      if (my !== Fd.seq) return;
      Fd.loading = false; Fd.page = d.page; Fd.pages = Math.min(d.pages, Math.ceil(10000 / d.size)); Fd.total = d.total; Fd.capped = !!d.total_capped;
      Fd.items = replace ? d.items : Fd.items.concat(d.items);
      renderFind();
    }).catch(function (e) {
      if (my !== Fd.seq) return;
      Fd.loading = false;
      if (H.failure(e)) return;
      Fd.error = e.status === 400 ? (e.message || 'These filters are not valid.') : 'Check your connection, then try again.';
      renderFind();
    });
  }
  var findT = null;
  function findSoon() { clearTimeout(findT); findT = setTimeout(function () { fetchFind(true); }, 260); }

  /* =====================================================================================================================
     SELECTION, RUNS, RESULTS TABLE
     ===================================================================================================================== */
  function stateHTML(id) {
    var st = stateOf(id); if (!st) return resultOf(id) ? '<span class="wb-st ok">Stored result</span>' : '<span class="wb-st idle">Not run yet</span>';
    var s = st.state, sc = (st.secs != null ? secs(st.secs) : '');
    if (s === 'queued') return '<span class="wb-st q">' + (S.run && S.run.qpos > 0 ? 'Waiting for an earlier run' : 'Queued') + '</span>';
    if (s === 'running') return '<span class="wb-st run"><span class="cc-spin" aria-hidden="true"></span>Running <span class="wb-el" data-t0="' + esc(st.started_at || '') + '">' + elapsed(st) + '</span>' + (st.note ? '<span class="wb-st-n">' + esc(st.note) + '</span>' : '') + '</span>';
    if (s === 'done') return '<span class="wb-st ok">Done' + (sc ? ' in ' + esc(sc) : '') + '</span>';
    if (s === 'cached') return '<span class="wb-st ok">Stored result, instant</span>';
    if (s === 'degraded') return '<span class="wb-st warn">Degraded: deep verdict timed out</span>';
    if (s === 'skipped') return '<span class="wb-st idle">Skipped: ' + esc(st.reason || 'not run') + '</span>';
    return '<span class="wb-st err">Not finished: ' + esc(st.reason || 'analysis failed') + '</span>';
  }
  function elapsed(st) {
    var t0 = Date.parse(st.started_at); if (isNaN(t0)) return '';
    return secs((Date.now() + S.off - t0) / 1000);
  }
  function rowClass(id) { var sm = summarise(id); return sm && sm.label ? ' s-' + esc(sm.label) : ''; }
  function renderTray() {
    var box = $('#wb-tray'); if (!box) return;
    $('#wb-seln').textContent = S.sel.length + ' of ' + MAX_SEL;
    var sub = !S.sel.length ? 'Nothing selected yet. Add items from the list, or open an item in the Browse tab and choose Analyse.' : fmtN(S.sel.length) + ' item' + (S.sel.length === 1 ? '' : 's') + ' selected (at most ' + MAX_SEL + ').';
    $('#wb-sel-sub').textContent = sub;
    keepFocus(function () {
      if (!S.sel.length) { box.innerHTML = '<div class="wb-empty"><b>The selection is empty</b><span>Use Add in Find items, then run Claim Check here. Each item takes about 15 to 40 s and items run one at a time.</span></div>'; return; }
      box.innerHTML = '<div class="wb-tray-tools"><button type="button" class="link-btn" data-act="clear" data-fk="clear">Clear selection</button></div>' +
        '<ul class="wb-tray" id="wb-tray-l" role="listbox" aria-label="Selected items. Select one to read its result; press Delete to remove it.">' + S.sel.map(function (m) {
          return '<li class="wb-tr-li" role="presentation"><div class="row wb-tr' + rowClass(m.id) + '" role="option" data-id="' + esc(m.id) + '" data-fk="t:' + esc(m.id) + '" tabindex="-1" aria-selected="' + (m.id === S.view) + '">' +
            '<span class="row-title">' + esc(m.title || m.id) + '</span>' +
            '<span class="row-meta"><span class="ev-plat" title="Source table: ' + esc(L.platformTable(m.platform)) + '">' + esc(L.platform(m.platform)) + '</span><span aria-hidden="true">&middot;</span><span class="row-src">' + esc(H.sourceName(m.source)) + '</span></span>' +
            '<span class="wb-tr-st">' + stateHTML(m.id) + '</span></div>' +
            '<button type="button" class="wb-x" data-act="remove" data-id="' + esc(m.id) + '" data-fk="x:' + esc(m.id) + '" aria-label="Remove from selection: ' + esc(clip(m.title || m.id, 80)) + '"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><path d="M6 6l12 12M18 6 6 18"/></svg></button></li>';
        }).join('') + '</ul>';
      var l = $('#wb-tray-l'), cur = l.querySelector('.row[aria-selected="true"]') || l.querySelector('.row'); if (cur) C.setCursor(l, cur, false);
    });
  }
  function renderTools() {
    var run = $('#wb-run'), why = $('#wb-why'), n = S.sel.length, act = runActive(), hasRes = S.sel.some(function (m) { return !!resultOf(m.id); });
    var reason = '';
    if (!n) reason = 'Add at least one item to run Claim Check.'; else if (act) reason = 'A run is in progress. You can cancel it below.';
    run.disabled = !n || act;
    run.textContent = act ? 'Running...' : 'Run Claim Check on selection';
    $('#wb-cmp').disabled = !n || !!(S.cmp && S.cmp.state === 'running');
    $('#wb-cmp').title = n ? '' : 'Select items first';
    $('#wb-ins').disabled = !n;
    $('#wb-ins').title = n ? '' : 'Select items first';
    $('#wb-copy').disabled = !hasRes;
    $('#wb-copy').title = hasRes ? '' : 'Run Claim Check first: there are no results to copy yet';
    why.textContent = reason; why.hidden = !reason;
    run.setAttribute('aria-describedby', reason ? 'wb-why' : '');
    $('#wb-force').disabled = act;
  }
  function renderProg() {
    var box = $('#wb-prog'), r = S.run; if (!box) return;
    if (!r) {
      var n = S.sel.length;
      box.innerHTML = '<p class="wb-honest">About 15 to 40 s per item. Items run one at a time' + (n ? ', so up to ' + esc(secs(n * 15)) + ' to ' + esc(secs(n * 40)) + ' for ' + esc(fmtN(n)) + ' item' + (n === 1 ? '' : 's') + ' that have no stored result' : '') + '. Items already checked come back from ' + term('cached result', 'stored results') + ' at once.</p>';
      return;
    }
    var act = runActive(), total = r.total || 1, done = r.done || 0, running = Object.keys(r.items).filter(function (k) { return r.items[k].state === 'running'; });
    var line;
    if (r.state === 'queued') line = 'Waiting in the queue' + (r.qpos > 1 ? ' (' + (r.qpos - 1) + ' run' + (r.qpos - 1 === 1 ? '' : 's') + ' ahead)' : '') + '. ' + done + ' of ' + total + ' items finished.';
    else if (r.state === 'running') line = done + ' of ' + total + ' items finished' + (running.length ? ', 1 running' : '') + (r.cancelling ? '. Cancelling: the item in progress finishes first, the rest are skipped' : '') + '.';
    else if (r.state === 'cancelled') line = 'Cancelled after ' + done + ' of ' + total + ' items.';
    else if (r.state === 'lost') line = 'The server no longer has this run (it was restarted). Items that finished are kept in the stored results.';
    else if (r.state === 'error') line = 'The run stopped because of a server problem. ' + done + ' of ' + total + ' items finished.';
    else line = 'Finished: ' + done + ' of ' + total + ' items.';
    box.innerHTML = '<div class="wb-pr" role="group" aria-label="Run progress"><div class="wb-pr-top"><span class="wb-pr-t">' + (act ? '<span class="cc-spin" aria-hidden="true"></span>' : '') + esc(line) + '</span>' +
      (act ? '<button type="button" class="btn ghost sm" data-act="cancel" data-fk="cancel"' + (r.cancelling ? ' disabled' : '') + '>' + (r.cancelling ? 'Cancelling' : 'Cancel') + '</button>' : '') + '</div>' +
      '<div class="wb-bar" role="progressbar" aria-label="Items finished" aria-valuemin="0" aria-valuemax="' + total + '" aria-valuenow="' + done + '"><b style="width:' + (done / total * 100).toFixed(1) + '%"></b></div>' +
      (act ? '<p class="wb-honest">About 15 to 40 s per item; items run one at a time. This page checks progress every 2 s while it is visible. <span class="wb-el-all" data-t0="' + esc(r.started_at || r.created_at || '') + '"></span></p>' : '') + '</div>';
  }
  function renderNote() {
    var box = $('#wb-note'); if (!box) return;
    var n = S.note; box.innerHTML = n ? '<div class="cc-note' + (n.kind === 'warn' ? ' warn' : '') + '"><div>' + n.html + '</div>' + (n.retry ? '<button type="button" class="btn sm" data-act="retry-failed" data-fk="retryf">' + esc(n.retry) + '</button>' : '') + '</div>' : '';
  }
  function setNote(kind, html, retry) { S.note = html ? { kind: kind, html: html, retry: retry } : null; renderNote(); }

  /* the results table: one row per selected item that has a result */
  function tableRows() {
    var rows = S.sel.map(function (m) { var s = summarise(m.id); return s ? { m: m, s: s } : null; }).filter(Boolean);
    var k = S.sort.key; if (k) rows.sort(function (a, b) {
      var x = k === 'score' ? a.s.plaus : a.s.n, y = k === 'score' ? b.s.plaus : b.s.n;
      if (x == null && y == null) return 0; if (x == null) return 1; if (y == null) return -1;
      return S.sort.dir === 'desc' ? y - x : x - y;
    });
    return rows;
  }
  function sortTh(key, label, tip) {
    var on = S.sort.key === key;
    return '<th scope="col" aria-sort="' + (on ? (S.sort.dir === 'desc' ? 'descending' : 'ascending') : 'none') + '"><button type="button" class="wb-sort' + (on ? ' on' : '') + '" data-act="sort" data-key="' + key + '" data-fk="s:' + key + '" title="' + esc(tip) + '">' + esc(label) + '<span aria-hidden="true">' + (on ? (S.sort.dir === 'desc' ? ' ↓' : ' ↑') : '') + '</span></button></th>';
  }
  function srcBadge(s) {
    return s.src === 'cached' ? ob('ob-db', 'cached result', 'Served from the stored results of an earlier run; the engine was not called.') : ob('ob-fnp', 'Run now', 'FNP read this item on request in this run.');
  }
  function renderTable() {
    var box = $('#wb-table'); if (!box) return;
    var rows = tableRows();
    if (!rows.length) { box.innerHTML = ''; return; }
    keepFocus(function () {
      box.innerHTML = '<h3 class="wb-h3">Results <span class="wb-n">' + rows.length + ' of ' + S.sel.length + ' selected items</span></h3>' +
        '<div class="wb-tbl-wrap"><table class="wb-tbl"><caption class="sr">Claim Check results for the selected items. Sort by fake-or-not score or number of techniques.</caption><thead><tr><th scope="col">Item</th><th scope="col">Verdict</th>' +
        sortTh('score', 'Fake-or-not score', 'Sort by the fake-or-not score (probability that it is fake or misleading)') + sortTh('tech', 'Techniques', 'Sort by the number of propaganda techniques found') +
        '<th scope="col">Top technique</th><th scope="col">Result</th></tr></thead><tbody>' + rows.map(function (r) {
          var s = r.s, b = s.plaus == null ? null : s.plaus >= 0.8 ? 'FAKE' : s.plaus >= 0.5 ? 'MISLEADING' : 'INCONCLUSIVE';
          return '<tr' + (r.m.id === S.view ? ' aria-current="true" class="cur"' : '') + '><th scope="row" data-l="Item"><button type="button" class="link-btn wb-tbl-t" data-act="view" data-id="' + esc(r.m.id) + '" data-fk="r:' + esc(r.m.id) + '">' + esc(clip(r.m.title || r.m.id, 90)) + '</button></th>' +
            '<td data-l="Verdict">' + (s.label ? H.vchip(s.label) : '<span class="chip tag">No verdict</span>') + '</td>' +
            '<td data-l="Fake-or-not score">' + (s.plaus == null ? '<span class="chip num v-UNRATED">No score</span>' : '<span class="chip num v-' + b + '" title="Plausibility of being fake or misleading: ' + esc(bandName(s.plaus)) + '">' + s.plaus.toFixed(2) + '</span>') + '</td>' +
            '<td data-l="Techniques" class="num">' + (s.n == null ? 'n/a' : esc(fmtN(s.n))) + '</td>' +
            '<td data-l="Top technique">' + (s.top ? esc(s.top.name) + ' <small class="num">' + Math.round(s.top.p * 100) + '%</small>' : '<span class="wb-none">None found</span>') + '</td>' +
            '<td data-l="Result">' + srcBadge(s) + (s.degraded ? ob('ob-db warn', 'degraded', 'The deep verdict timed out; this is the fast result only and it was not stored.') : '') + '</td></tr>';
        }).join('') + '</tbody></table></div><p class="ev-denom">Denominator: ' + rows.length + ' of the ' + S.sel.length + ' selected items have a result. The score is a ranking aid, not a calibrated probability. Techniques count the distinct technique types found.</p>';
    });
  }

  /* ---------- running a job and following it ---------- */
  function submit(ids, force) {
    if (runActive()) return;
    setNote(null);
    ids.forEach(function (id) { delete S.res[id]; });
    $('#wb-run').disabled = true;
    call('POST', '/api/rt/analyse', { ids: ids, force: !!force }).then(function (j) {
      S.run = { id: j.job_id, ids: ids.slice(), state: j.state, total: j.total, done: j.done || 0, items: {}, qpos: j.queue_position, cancelling: false, force: !!force, needFull: false, fails: 0, created_at: j.created_at, started_at: j.started_at };
      S.off = isNaN(Date.parse(j.server_time)) ? S.off : Date.parse(j.server_time) - Date.now();
      j.items.forEach(function (it) { S.run.items[it.id] = it; });
      live('Started checking ' + ids.length + (ids.length === 1 ? ' item.' : ' items.'));
      renderAll(); startTimers(); poll(true);
    }).catch(function (e) {
      renderTools();
      if (H.failure(e)) return;
      if (e.status === 404) setNote('warn', '<b>Some items are no longer in the cache.</b> ' + esc(e.message) + '. They may have aged out of the synced window; remove them and run again.');
      else if (e.status === 429) setNote('warn', '<b>Too many runs at once.</b> ' + esc(errText(e)));
      else setNote('warn', '<b>The run could not be started.</b> ' + esc(errText(e)));
    });
  }
  function applyJob(j) {
    var r = S.run; if (!r || r.id !== j.job_id) return;
    if (!isNaN(Date.parse(j.server_time))) S.off = Date.parse(j.server_time) - Date.now();
    r.state = j.state; r.total = j.total; r.done = j.done; r.cancelling = !!j.cancelling; r.qpos = j.queue_position; r.started_at = j.started_at; r.finished_at = j.finished_at;
    var need = false;
    j.items.forEach(function (it) {
      var prev = r.items[it.id] || {}, copy = {}; Object.keys(it).forEach(function (k) { if (k !== 'result') copy[k] = it[k]; });
      r.items[it.id] = copy;
      if (!TERMINAL[it.state]) return;
      var cur = S.res[it.id], same = cur && cur.jobId === r.id;
      var result = it.result || (same ? cur.result : null);
      if (it.has_result && !result) need = true;
      S.res[it.id] = { state: it.state, reason: it.reason, secs: it.secs, result: result, cached_at: it.cached_at || null, engine_state: it.engine_state || null, retry_after_s: it.retry_after_s || null, jobId: r.id };
      if (prev.state !== it.state && S.sel.some(function (m) { return m.id === it.id; })) { var t = (S.sel.filter(function (m) { return m.id === it.id; })[0] || {}).title; live('Finished ' + r.done + ' of ' + r.total + ': ' + (t || 'an item') + ' (' + it.state + ').'); }
    });
    r.needFull = need;
    var sig = j.state + '|' + j.done + '|' + j.cancelling + '|' + j.queue_position + '|' + j.items.map(function (it) { return it.state + (it.note || '') + (it.has_result ? 'r' : ''); }).join(',');
    if (sig !== S.sig) { S.sig = sig; renderAll(true); }
    return need;
  }
  function poll(force) {
    var r = S.run; if (!r) return;
    if (!runActive() && !r.needFull) { stopTimers(); return; }
    if ((document.hidden || !S.active) && !force) return;
    if (S.polling) return; S.polling = true;
    call('GET', '/api/rt/jobs/' + encodeURIComponent(r.id) + (r.needFull ? '' : '?results=0')).then(function (j) {
      S.polling = false; r.fails = 0;
      var need = applyJob(j);
      if (need) { poll(true); return; }
      if (!runActive()) finish();
    }).catch(function (e) {
      S.polling = false;
      if (e.status === 404) { r.state = 'lost'; stopTimers(); renderAll(); setNote('warn', '<b>The server no longer knows this run.</b> It was probably restarted. Items that finished before that are kept in the stored results; run the rest again.', 'Run the unfinished items again'); return; }
      if (H.failure(e)) { stopTimers(); return; }
      r.fails++; if (r.fails === 3) setNote('warn', '<b>Connection to the server was lost.</b> This page keeps trying every 2 s while it is open.');
    });
  }
  function finish() {
    stopTimers(); renderAll();
    var r = S.run, c = { done: 0, cached: 0, degraded: 0, skipped: 0, error: 0 }, notes = [], retry = [], eng = 0, busy = 0, cancelled = 0, short = 0;
    r.ids.forEach(function (id) {
      var st = r.items[id]; if (!st) return;
      c[st.state] = (c[st.state] || 0) + 1;
      if (st.state === 'degraded') retry.push(id);
      if (st.state === 'error') { retry.push(id); if (st.engine_state === 'busy') busy++; else if (st.engine_state) eng++; }
      if (st.state === 'skipped') { if (st.reason === 'cancelled') cancelled++; else short++; }
    });
    var ok = c.done + c.cached, parts = [];
    parts.push('<b>' + (r.state === 'cancelled' ? 'Cancelled.' : 'Finished.') + '</b> ' + ok + ' of ' + r.total + ' item' + (r.total === 1 ? '' : 's') + ' checked' + (c.cached ? ' (' + c.cached + ' from stored results)' : '') + '.');
    if (c.degraded) notes.push(c.degraded + ' timed out in the deep verdict, so ' + (c.degraded === 1 ? 'it shows' : 'they show') + ' the fast result and techniques only and ' + (c.degraded === 1 ? 'was' : 'were') + ' not stored.');
    if (eng) notes.push(eng + ' could not be checked because the verdict engine was not reachable or returned an error.');
    if (busy) notes.push(busy + ' could not be checked because the engine stayed busy with other work.');
    if (c.error - eng - busy > 0) notes.push((c.error - eng - busy) + ' failed for another reason (see the item rows).');
    if (short) notes.push(short + ' had too little text to check and ' + (short === 1 ? 'was' : 'were') + ' skipped.');
    if (cancelled) notes.push(cancelled + ' not started because the run was cancelled.');
    var warn = c.degraded || c.error || short || cancelled;
    S.runRetry = retry;
    setNote(warn ? 'warn' : 'ok', parts.join(' ') + (notes.length ? '<span class="sub">' + notes.map(esc).join(' ') + '</span>' : ''), retry.length ? 'Retry ' + retry.length + ' unfinished item' + (retry.length === 1 ? '' : 's') : '');
    live((r.state === 'cancelled' ? 'Run cancelled. ' : 'Run finished. ') + ok + ' of ' + r.total + ' items checked.');
    if (!S.view || !S.sel.some(function (m) { return m.id === S.view; })) { var first = S.sel.filter(function (m) { return resultOf(m.id); })[0]; if (first) { S.view = first.id; renderAll(); loadView(); } }
    var ex = S.view; if (ex) paintResult();
  }
  function tick() {
    if (document.hidden) return;
    var ns = document.querySelectorAll('.wb-el');
    Array.prototype.forEach.call(ns, function (n) { var t0 = Date.parse(n.getAttribute('data-t0')); if (!isNaN(t0)) n.textContent = secs((Date.now() + S.off - t0) / 1000); });
    var all = $('.wb-el-all'); if (all && S.run) { var t = Date.parse(S.run.started_at || S.run.created_at); if (!isNaN(t)) all.textContent = 'Run time so far ' + secs((Date.now() + S.off - t) / 1000) + '.'; }
  }
  function startTimers() {
    if (!S.tickT) S.tickT = setInterval(tick, 1000);
    if (!S.pollT) S.pollT = setInterval(function () { poll(false); }, POLL_MS);
  }
  function stopTimers() { clearInterval(S.tickT); clearInterval(S.pollT); S.tickT = S.pollT = null; }
  document.addEventListener('visibilitychange', function () { if (!document.hidden && S.active && runActive()) { startTimers(); poll(true); } });

  /* =====================================================================================================================
     RESULT PANEL
     ===================================================================================================================== */
  function loadView() {
    var id = S.view; if (!id) return;
    if (S.items[id] || S.itemErr[id]) { paintResult(); return; }
    paintResult();
    Ev.getItem(id).then(function (it) { S.items[id] = it; if (S.view === id) paintResult(); })
      .catch(function (e) { if (H.failure(e)) return; S.itemErr[id] = e.status === 404 ? 'This item is not in the Evidence Base copy any more (it may have aged out of the window).' : 'The item could not be loaded. Check your connection and try again.'; if (S.view === id) paintResult(); });
  }
  function viewMeta(id) {
    var m = S.sel.filter(function (x) { return x.id === id; })[0]; if (m) return m;
    var f = Fd.items.filter(function (x) { return x.id === id; })[0]; if (f) return metaOf(f);
    return S.items[id] ? metaOf(S.items[id]) : { id: id, title: id };
  }
  function srcBadgeBig(r) {
    if (r.src === 'cached') return ob('ob-db solid', 'cached result', 'Served from the stored results of an earlier run' + (r.at ? ' (' + L.dateTimeUTC(r.at) + ')' : '') + '; the engine was not called.') + (r.at ? '<span class="wb-when">stored ' + esc(H.ago(r.at)) + '</span>' : '');
    return ob('ob-fnp solid', 'Run by FNP now', 'FNP read this item on request, in this session.') + (r.secs ? '<span class="wb-when">' + esc(secs(r.secs)) + ' engine time</span>' : '');
  }
  function dbBlock(id, it) {
    var rows = [];
    var add = function (k, v) { rows.push([k, v]); };
    if (!it) return C.section('wb-s-db', 'From the database', H.BADGE_DB + ' Loading what the connected database stores about this item.', '<p class="note">Loading</p>');
    add('Kind', esc(H.kindName(it.kind)));
    add('Platform', esc(L.platform(it.platform)), 'Source table: ' + esc(it.source_table || L.platformTable(it.platform)));
    add('Source', esc(H.sourceName(it.source)));
    add('Language', esc(L.lang(it.lang)));
    add('Published', it.published_at ? esc(L.dateTimeUTC(it.published_at)) + '<span class="sub">' + esc(H.ago(it.published_at)) + '</span>' : 'Unknown');
    add(term('database classification', 'Topic'), it.topic && it.topic.name ? esc(it.topic.name) : 'Unclassified');
    return C.section('wb-s-db', 'From the database', H.BADGE_DB + ' What the connected database stores about this item, shown unchanged.', C.kv(rows));
  }
  function batchBlock(it) {
    var inf = it && it.inferred, b = inf && inf.batch; if (!b) return '';
    var rows = [['Propaganda signal', b.p_prop == null ? 'Not computed' : '<span class="ev-big">' + esc(n2(b.p_prop)) + '</span> ' + H.mini('Propaganda signal', b.p_prop, 'acc') + '<span class="sub">Uncalibrated: it ranks items, it is not a share.</span>'],
      ['Verdict', b.verdict ? H.vchip(b.verdict) + '<span class="sub">Only items that closely match a known fact-check get one.</span>' : 'No verdict']];
    return C.section('wb-s-batch', 'Inferred by FNP: batch cheap pass', ob('ob-fnp', 'Inferred by FNP (batch)') + ' From FNP\'s fast background analysis, which has not reached every item yet, not from a Claim Check run. Run Claim Check for the full result.', C.kv(rows) + H.inferredBars(inf) + '<p class="cs-sub-2">Propaganda techniques found</p>' + H.techniquesHTML(inf.techniques));
  }
  function inferredBadge(it) {
    var inf = it && it.inferred;
    return !inf ? ob('ob-db', 'Not analysed yet') : inf.deep ? ob('ob-fnp', 'Inferred by FNP (deep)', 'A stored Claim Check result, run ' + (inf.deep.created_at ? H.ago(inf.deep.created_at) : 'earlier')) : ob('ob-fnp', 'Inferred by FNP (batch)');
  }
  function paintResult() {
    var box = $('#wb-result'); if (!box) return;
    var id = S.view;
    if (!id) {
      box.innerHTML = '<div class="wb-guide"><h3>Nothing to show yet</h3><p>Add items to the selection and run Claim Check. Each result has two parts, always shown together:</p><ol><li><b>Is it fake?</b> The verdict with a probability for each class: Fake, Misleading, Legit and Inconclusive.</li><li><b>What kind of propaganda?</b> A probability bar for each of the 14 techniques, and the wording marked in the text. It describes how the text is worded, whether or not it is true.</li></ol><p class="note">Select an item in the list at the left to preview what the database stores about it.</p></div>';
      return;
    }
    var top = box.scrollTop, m = viewMeta(id), it = S.items[id], r = resultOf(id), st = stateOf(id), h = '';
    var kicker = [L.platform(m.platform), H.sourceName(m.source), L.lang(m.lang), m.published_at ? H.ago(m.published_at) : 'undated'].join(' - ');
    var added = inSel(id), act = runActive();
    h += '<div class="wb-rh"><p class="case-kicker">' + esc(kicker) + '</p><h3 class="wb-rt" id="wb-rt" tabindex="-1">' + esc(clip(it ? H.itemTitle(it) : m.title, 130)) + '</h3><div class="wb-rbadges">' +
      (r ? srcBadgeBig(r) + (r.degraded ? ob('ob-db warn', 'degraded') : '') + (r.result && r.result.dup_of ? ob('ob-db', 'reused from a near-duplicate', 'An item with almost the same text was analysed; its result was reused.') : '') : inferredBadge(it)) + '</div>' +
      '<div class="wb-ract"><button type="button" class="btn ghost sm" data-act="toggle-view" data-id="' + esc(id) + '" data-fk="tv">' + (added ? 'Remove from selection' : 'Add to selection') + '</button>' +
      '<button type="button" class="btn sm" data-act="run-one" data-id="' + esc(id) + '" data-fk="r1"' + (act ? ' disabled title="A run is in progress"' : '') + '>' + (r && !r.degraded ? 'Run again' : 'Run Claim Check on this item') + '</button>' +
      '<button type="button" class="btn ghost sm" data-act="compare-one" data-id="' + esc(id) + '" data-fk="c1" aria-expanded="' + (S.cmpFor === id) + '">Compare with the Fake News Database</button>' +
      '<a class="btn ghost sm" href="#evidence/item/' + encodeURIComponent(id) + '">Open the case file</a>' + (it ? H.itemUrl(it.url) : '') + '</div></div>';
    if (S.itemErr[id]) h += '<div class="cc-note warn" role="status"><div><b>' + esc(S.itemErr[id]) + '</b></div></div>';
    if (st && !r && (st.state === 'error' || st.state === 'skipped')) {
      h += '<div class="cc-note warn" role="status"><div><b>' + (st.state === 'skipped' ? 'This item was skipped.' : 'This item could not be checked.') + '</b> ' + esc((st.reason || '').replace(/^./, function (c) { return c.toUpperCase(); }) + (st.reason ? '.' : '')) + (st.state === 'error' && st.engine_state ? '<span class="sub">The verdict engine state was: ' + esc(L.titleCase(st.engine_state)) + '.</span>' : '') + '</div>' +
        (st.state === 'error' && !act ? '<button type="button" class="btn sm" data-act="run-one" data-id="' + esc(id) + '" data-fk="r2">Try again</button>' : '') + '</div>';
    }
    if (r) {
      var rr = CC.renderResult(r.result, { text: it ? compose(it) : m.title || '', kicker: kicker, howOpen: false });
      if (r.degraded) h += '<div class="cc-note warn" role="status"><div>' + (r.partial ? '<b>Partial result: propaganda only.</b> The deep verdict timed out earlier, so only the propaganda signals are shown and there is no verdict. The scheduled analysis retries it on later runs (up to 3 times); run it again here to try now.' : '<b>Degraded result.</b> The deep verdict timed out, so this shows the fast result and the propaganda techniques only. It was not stored.') + '</div><button type="button" class="btn sm" data-act="run-one" data-id="' + esc(id) + '" data-fk="r3"' + (act ? ' disabled' : '') + '>Retry</button></div>';
      h += rr.stepper + '<div class="wb-cc">' + rr.s1 + rr.s2 + '</div>' + rr.how;
    } else if (it && it.inferred) h += batchBlock(it);
    else if (it && !(st && (st.state === 'error' || st.state === 'skipped'))) h += '<div class="ev-notan"><b>Not analysed yet</b><p>Neither a batch analysis nor a Claim Check result exists for this item. That says nothing about whether it is true. Run Claim Check on it to see both parts.</p></div>';
    h += dbBlock(id, it);
    if (S.cmpFor === id && it) h += '<div class="wb-cmp-one">' + H.compareSection(it) + '</div>';
    box.innerHTML = h;
    box.scrollTop = top;
  }

  /* ---------- compare with the Fake News Database: this item (shown in the Result panel) and the whole selection (a compact table) ---------- */
  function compareSelection() {
    if (!S.sel.length) return;
    var ids = S.sel.map(function (m) { return m.id; }), c = S.cmp = { state: 'running', rows: {}, done: 0, total: ids.length }, i = 0, active = 0;
    renderCmp(); renderTools();
    live('Comparing ' + ids.length + ' items with the Fake News Database.');
    function next() {
      while (active < 3 && i < ids.length) {
        (function (id) {
          i++; active++;
          Ev.getItem(id).then(function (it) { S.items[id] = S.items[id] || it; c.rows[id] = { it: it }; }).catch(function (e) { c.rows[id] = { err: e.status === 404 ? 'No longer in the cache' : e.status === 429 ? 'Too many requests: wait a moment and compare again' : 'Could not load' }; })
            .then(function () { active--; c.done++; if (S.cmp !== c) return; renderCmp(); if (c.done >= c.total) { c.state = 'done'; renderTools(); live('Comparison finished.'); } else next(); });
        })(ids[i]);
      }
    }
    next();
  }
  function renderCmp() {
    var box = $('#wb-cmpout'), c = S.cmp; if (!box) return;
    if (!c) { box.innerHTML = ''; return; }
    var rows = S.sel.map(function (m) {
      var r = c.rows[m.id], cell;
      if (!r) return '<tr><th scope="row" data-l="Item">' + esc(clip(m.title || m.id, 90)) + '</th><td colspan="4" class="wb-none">' + (c.state === 'running' ? 'Waiting' : 'Not compared') + '</td></tr>';
      if (r.err) return '<tr><th scope="row" data-l="Item">' + esc(clip(m.title || m.id, 90)) + '</th><td colspan="4" class="wb-none">' + esc(r.err) + '</td></tr>';
      var fc = r.it.nearest_factchecks || {}, mt = fc.matches && fc.matches[0], nd = r.it.near_duplicates || {};
      cell = fc.available === false ? '<span class="wb-none">The Fake News Database could not be searched</span>' : mt ? '<a href="#desk/' + encodeURIComponent(mt.id) + '">' + esc(clip(mt.title || mt.claim || mt.id, 90)) + '</a>' : '<span class="wb-none">No fact-check is close enough</span>';
      return '<tr><th scope="row" data-l="Item"><button type="button" class="link-btn wb-tbl-t" data-act="view" data-id="' + esc(m.id) + '">' + esc(clip(m.title || m.id, 90)) + '</button></th>' +
        '<td data-l="Nearest fact-check">' + cell + '</td><td data-l="Rating">' + (mt ? F.chip(mt.verdict_norm || 'UNRATED') : '') + '</td>' +
        '<td data-l="Similarity" class="num">' + (mt && mt.similarity != null ? esc(n2(mt.similarity)) : 'n/a') + '</td>' +
        '<td data-l="Near-duplicates" class="num">' + (nd.count == null ? 'n/a' : esc(fmtN(nd.count))) + '</td></tr>';
    }).join('');
    box.innerHTML = '<h3 class="wb-h3">Compared with the Fake News Database ' + H.BADGE_FN + (c.state === 'running' ? '<span class="wb-n" role="status">' + c.done + ' of ' + c.total + '</span>' : '') + '<button type="button" class="link-btn wb-close" data-act="cmp-close">Hide</button></h3>' +
      '<div class="wb-tbl-wrap"><table class="wb-tbl wb-cmp"><caption class="sr">Nearest fact-check and near-duplicate count for each selected item</caption><thead><tr><th scope="col">Item</th><th scope="col">Nearest fact-check</th><th scope="col">Rating</th><th scope="col">' + term('similarity', 'Similarity') + '</th><th scope="col">' + term('near-duplicate', 'Near-duplicates') + '</th></tr></thead><tbody>' + rows + '</tbody></table></div>' +
      '<p class="ev-denom">Similarity is how close a past fact-check is to the item\'s text (0 to 1). No close match does not mean the item is true. Near-duplicates are other items in the Realtime Database with almost the same text. Select a fact-check to open it in the Desk.</p>';
  }

  /* ---------- copy the results as JSON: clipboard API, then a visible select-all textarea ---------- */
  function jsonText() {
    var items = [];
    S.sel.forEach(function (m) {
      var r = resultOf(m.id); if (!r) return;
      var res = r.result, s1 = res.step1 || {}, p = res.step2 && res.step2.propaganda, it = S.items[m.id] || {};
      items.push({
        id: m.id, title: m.title, platform: m.platform, source: m.source, url: it.url || null,
        result_source: r.src === 'cached' ? 'cached result' : 'run by FNP now', degraded: !!r.degraded,
        verdict: s1.label || null, fake_or_not_score: s1.plausibility == null ? null : s1.plausibility, verdict_classes: s1.classes || [],
        technique_classes: p && Array.isArray(p.classes) ? p.classes.filter(function (c) { return c.probability > 0; }) : [], technique_basis: p && p.basis_note || null,
        spans_found: p && p.techniques ? p.techniques.length : null
      });
    });
    return JSON.stringify({ exported_at: new Date().toISOString(), note: 'Probabilities are uncalibrated ranking aids. Source: FNP Claim Check on items from the Realtime Database.', items: items }, null, 2);
  }
  function copyJson() {
    var text = jsonText(), box = $('#wb-json');
    var show = function (msg) {
      box.innerHTML = '<div class="wb-jsonbox"><p class="note" role="status">' + esc(msg) + '</p><label class="sr" for="wb-json-ta">Results as JSON</label><textarea id="wb-json-ta" class="wb-ta" readonly rows="10"></textarea></div>';
      var ta = $('#wb-json-ta'); ta.value = text; ta.focus(); ta.select();
      return ta;
    };
    var fallback = function () {
      var ta = show('Select and copy the text below (the browser did not allow automatic copying).');
      try { if (document.execCommand && document.execCommand('copy')) { box.querySelector('.note').textContent = 'Copied. The same text is shown below in case you need it again.'; live('Copied the results as JSON.'); } } catch (e) {}
      return ta;
    };
    try {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(function () { show('Copied the results as JSON. The same text is shown below in case you need it again.'); live('Copied the results as JSON.'); }, fallback);
        return;
      }
    } catch (e) {}
    fallback();
  }

  /* ---------- saved selections ---------- */
  function renderSaved() {
    var box = $('#wb-saved-in'); if (!box) return;
    var sets = loadSets(), cur = $('#wb-sv-pick') ? $('#wb-sv-pick').value : '';
    box.innerHTML = '<div class="wb-sv-row"><label class="f wb-sv-f"><span class="lbl">Name</span><input type="text" id="wb-sv-name" maxlength="40" autocomplete="off" placeholder="For example: Monday flood posts"></label>' +
      '<button type="button" class="btn ghost sm" data-act="sv-save" data-fk="svs"' + (S.sel.length ? '' : ' disabled title="Select items first"') + '>Save current selection</button></div>' +
      (sets.length ? '<div class="wb-sv-row"><label class="f wb-sv-f"><span class="lbl">Saved</span><select id="wb-sv-pick">' + sets.map(function (s) { return '<option value="' + esc(s.name) + '">' + esc(s.name) + ' (' + s.items.length + ')</option>'; }).join('') + '</select></label>' +
        '<button type="button" class="btn ghost sm" data-act="sv-restore" data-fk="svr">Restore</button><button type="button" class="btn ghost sm" data-act="sv-delete" data-fk="svd">Delete</button></div>' : '<p class="note">No saved selections yet.</p>') +
      '<p class="note" id="wb-sv-msg" role="status">' + (storeOk ? 'Saved in this browser only, up to ' + MAX_SETS + ' selections.' : 'This browser is not keeping storage, so saved selections last only until you reload the page.') + '</p>';
    if (cur && $('#wb-sv-pick')) $('#wb-sv-pick').value = cur;
  }
  function svMsg(t) { var n = $('#wb-sv-msg'); if (n) n.textContent = t; }

  /* =====================================================================================================================
     WIRING
     ===================================================================================================================== */
  function renderAll(soft) {
    if (!S.built) return;
    renderTray(); renderTools(); renderProg(); renderTable(); renderCmp();
    var b = $('#wb-find'); if (b) { /* the Add buttons follow the selection */
      Array.prototype.forEach.call(b.querySelectorAll('.wb-fr[data-id]'), function (li) {
        var id = li.getAttribute('data-id'), a = li.querySelector('.wb-add'), added = inSel(id), full = !added && S.sel.length >= MAX_SEL;
        if (!a) return;
        li.classList.toggle('in', added); a.classList.toggle('ghost', !added); a.setAttribute('aria-pressed', added); a.textContent = added ? 'Added' : 'Add'; a.disabled = full;
        a.setAttribute('aria-label', (added ? 'Remove from selection: ' : 'Add to selection: ') + clip(li.querySelector('.wb-fr-t').textContent, 80));
      });
      var all = $('#wb-addall'), room = MAX_SEL - S.sel.length; all.disabled = !Fd.items.length || room <= 0 || !Fd.items.some(function (x) { return !inSel(x.id); });
    }
    if (soft) { var rid = S.view; if (rid) { var key = (S.res[rid] || {}).state + '|' + !!(S.res[rid] || {}).result + '|' + ((S.run && S.run.items[rid]) || {}).state; if (key !== S.rkey) { S.rkey = key; paintResult(); } } }
  }
  function show(id, scroll) {
    S.view = id; S.rkey = null; renderTray(); renderTable(); loadView();
    if (scroll && window.matchMedia('(max-width: 1279px)').matches) { var n = $('#wb-res-h'); if (n && n.scrollIntoView) n.scrollIntoView({ block: 'start' }); }
  }
  function addMeta(m) {
    var r = selAdd(m);
    if (r === 'full') { setNote('warn', '<b>The selection is full.</b> It holds at most ' + MAX_SEL + ' items. Remove one to add another.'); }
    return r;
  }
  function wire() {
    var w = el.wb;
    $('#wb-q').addEventListener('input', function () { Fd.q = this.value; findSoon(); });
    $('#wb-q').addEventListener('keydown', function (e) { if (e.key === 'Escape' && this.value) { this.value = ''; Fd.q = ''; findSoon(); e.stopPropagation(); } });
    $('#wb-kind').addEventListener('click', function (e) {
      var b = e.target.closest('.vchip'); if (!b) return; Fd.kind = b.getAttribute('data-kind');
      Array.prototype.forEach.call(this.querySelectorAll('.vchip'), function (x) { x.setAttribute('aria-pressed', x === b); }); fetchFind(true);
    });
    [['platform', '#wb-platform'], ['topic', '#wb-topic'], ['lang', '#wb-lang'], ['analysed', '#wb-analysed'], ['sort', '#wb-sort']].forEach(function (p) { $(p[1]).addEventListener('change', function () { Fd[p[0]] = this.value; fetchFind(true); }); });
    $('#wb-addall').addEventListener('click', function () {
      var room = MAX_SEL - S.sel.length, add = Fd.items.filter(function (x) { return !inSel(x.id); }), n = 0;
      add.slice(0, Math.max(0, room)).forEach(function (x) { if (selAdd(metaOf(x)) === 'added') n++; });
      setNote(n < add.length ? 'warn' : 'ok', '<b>Added ' + n + ' of the ' + add.length + ' items shown that were not selected yet.</b>' + (n < add.length ? ' The selection holds at most ' + MAX_SEL + ' items.' : ''));
      live('Added ' + n + ' items to the selection.');
      if (n && !S.view) { S.view = S.sel[0].id; loadView(); }
      renderAll();
    });
    $('#wb-run').addEventListener('click', function () { if (S.sel.length) submit(S.sel.map(function (m) { return m.id; }), $('#wb-force').checked); });
    $('#wb-cmp').addEventListener('click', compareSelection);
    $('#wb-ins').addEventListener('click', function () { if (S.sel.length) { Insights.useSelection(); Shell.navigate('#evidence/insights'); } });
    $('#wb-copy').addEventListener('click', copyJson);
    $('#wb-saved').addEventListener('toggle', function () { if (this.open) renderSaved(); });

    w.addEventListener('click', function (e) {
      var b = e.target.closest('[data-act]'); if (!b) return;
      var a = b.getAttribute('data-act'), id = b.getAttribute('data-id');
      if (a === 'view') { show(id, true); }
      else if (a === 'toggle') {
        if (inSel(id)) selRemove(id); else { var f = Fd.items.filter(function (x) { return x.id === id; })[0]; if (f) { addMeta(metaOf(f)); if (!S.view) { S.view = id; loadView(); } } }
        renderAll(); if (S.view && !S.sel.length) { /* keep the preview */ }
      }
      else if (a === 'toggle-view') { if (inSel(id)) selRemove(id); else addMeta(viewMeta(id)); renderAll(); paintResult(); }
      else if (a === 'remove') { var idx = S.sel.map(function (m) { return m.id; }).indexOf(id); selRemove(id); renderAll(); paintResult(); var rows = $('#wb-tray-l') && C.rows($('#wb-tray-l')); if (rows && rows.length) C.setCursor($('#wb-tray-l'), rows[Math.min(idx, rows.length - 1)], false); live('Removed from the selection.'); }
      else if (a === 'clear') { S.sel = []; S.view = null; S.cmp = null; renderAll(); paintResult(); live('Selection cleared.'); }
      else if (a === 'cancel') { b.disabled = true; call('POST', '/api/rt/jobs/' + encodeURIComponent(S.run.id) + '/cancel', {}).then(function (j) { applyJob(j); renderAll(); live('Cancelling the run.'); poll(true); }).catch(function (er) { if (!H.failure(er)) setNote('warn', '<b>The run could not be cancelled.</b> ' + esc(errText(er))); }); }
      else if (a === 'sort') { var k = b.getAttribute('data-key'); if (S.sort.key === k) S.sort.dir = S.sort.dir === 'desc' ? 'asc' : 'desc'; else S.sort = { key: k, dir: 'desc' }; renderTable(); }
      else if (a === 'run-one') { if (!runActive()) { S.view = id; submit([id], false); } }
      else if (a === 'retry-failed') { var ids = (S.runRetry && S.runRetry.length ? S.runRetry : (S.run ? S.run.ids.filter(function (x) { return !S.res[x] || (S.res[x].state !== 'done' && S.res[x].state !== 'cached'); }) : [])).filter(inSel); if (ids.length) submit(ids, false); }
      else if (a === 'compare-one') { S.cmpFor = S.cmpFor === id ? null : id; paintResult(); }
      else if (a === 'cmp-close') { S.cmp = null; renderCmp(); renderTools(); }
      else if (a === 'find-more') { fetchFind(false); }
      else if (a === 'find-retry') { fetchFind(true); }
      else if (a === 'sv-save') {
        var name = ($('#wb-sv-name').value || '').trim().slice(0, 40); if (!name) { svMsg('Give the selection a name first.'); $('#wb-sv-name').focus(); return; }
        var sets = loadSets().filter(function (s) { return s.name !== name; });
        if (sets.length >= MAX_SETS) { svMsg('You can keep ' + MAX_SETS + ' saved selections. Delete one first.'); return; }
        sets.unshift({ name: name, saved: new Date().toISOString(), items: S.sel.map(function (m) { return { id: m.id, title: m.title, kind: m.kind, platform: m.platform, source: m.source, lang: m.lang, published_at: m.published_at }; }) });
        saveSets(sets); renderSaved(); svMsg(storeOk ? 'Saved "' + name + '" (' + S.sel.length + ' items) in this browser.' : 'Kept for this page only: this browser is not keeping storage.'); live('Saved the selection.');
      }
      else if (a === 'sv-restore') {
        var pick = $('#wb-sv-pick').value, s = loadSets().filter(function (x) { return x.name === pick; })[0]; if (!s) return;
        S.sel = []; S.cmp = null; s.items.slice(0, MAX_SEL).forEach(function (m) { if (m && typeof m.id === 'string') selAdd({ id: m.id, title: String(m.title || m.id), kind: m.kind, platform: m.platform, source: m.source, lang: m.lang, published_at: m.published_at }); });
        S.view = S.sel.length ? S.sel[0].id : null; renderAll(); loadView(); svMsg('Restored "' + pick + '" (' + S.sel.length + ' items). Items that have left the cache are reported when you run them.'); live('Restored the saved selection.');
      }
      else if (a === 'sv-delete') { var del = $('#wb-sv-pick').value; saveSets(loadSets().filter(function (x) { return x.name !== del; })); renderSaved(); svMsg('Deleted "' + del + '".'); }
    });
    /* result panel: retry buttons from Claim Check's own engine notice, expanders */
    $('#wb-result').addEventListener('click', function (e) {
      if (C.onDetailClick(e)) return;
      var b = e.target.closest('[data-act="retry"]'); if (b && S.view && !runActive()) submit([S.view], false);
    });
    /* tray keyboard: arrows/j/k/Home/End move, Enter/Space shows the result, Delete removes */
    C.rowNav($('#wb-tray'), function (row) { show(row.getAttribute('data-id'), false); });
    $('#wb-tray').addEventListener('click', function (e) { var r = e.target.closest('.row[data-id]'); if (r) { C.setCursor($('#wb-tray-l'), r, false); show(r.getAttribute('data-id'), true); } });
    $('#wb-tray').addEventListener('keydown', function (e) {
      if (e.key !== 'Delete' && e.key !== 'Backspace') return;
      var r = e.target.closest && e.target.closest('.row[data-id]'); if (!r) return; e.preventDefault();
      var id = r.getAttribute('data-id'), rows = C.rows($('#wb-tray-l')), idx = rows.indexOf(r); selRemove(id); renderAll(); paintResult();
      var rr = C.rows($('#wb-tray-l')); if (rr.length) C.setCursor($('#wb-tray-l'), rr[Math.min(idx, rr.length - 1)], true); else $('#wb-find').querySelector('input').focus();
    });
  }

  /* =====================================================================================================================
     WORKBENCH ROUTE
     ===================================================================================================================== */
  var Workbench = {
    /* "Analyse" in the Browse case file: add the item to the selection and open it here (nothing runs until Run is pressed) */
    analyse: function (id) {
      Ev.getItem(id).then(function (it) {
        S.items[id] = it; var r = selAdd(metaOf(it));
        S.view = id; S.rkey = null; S.cmpFor = null;
        if (r === 'full') S.note = { kind: 'warn', html: '<b>The selection is full.</b> It holds at most ' + MAX_SEL + ' items; this item is shown below but was not added.' };
        else S.note = null;
        Shell.navigate('#evidence/workbench');
        if (S.built) { renderAll(); renderNote(); loadView(); }
      }).catch(function (e) { if (!H.failure(e)) { S.itemErr[id] = 'The item could not be loaded.'; S.view = id; Shell.navigate('#evidence/workbench'); } });
    },
    selection: function () { return S.sel.map(function (m) { return m.id; }); }
  };
  window.Workbench = Workbench;
  Ev.register('workbench', {
    enter: function () {
      build(); S.active = true; el.wb.hidden = false;
      fillSelects();
      if (!Fd.total && !Fd.loading) fetchFind(true); else renderFind();
      renderAll(); renderNote(); renderSaved();
      if (S.view) loadView(); else paintResult();
      if (runActive()) { startTimers(); poll(true); } else if (S.run && S.run.needFull) poll(true);
    },
    leave: function () { S.active = false; stopTimers(); }
  });

  /* =====================================================================================================================
     INSIGHTS
     ===================================================================================================================== */
  var I = { scope: 'browse', seq: 0, data: null, loading: false, error: null, q: '', ph: { state: 'idle' }, built: false, pop: null };
  var Insights = { useSelection: function () { I.scope = 'selection'; I.ph = { state: 'idle' }; } };

  function brScope() {
    var b = Ev.scopes().browse, o = {};
    ['q', 'kind', 'platform', 'source', 'topic', 'lang', 'from', 'to', 'analysed'].forEach(function (k) { var v = String(b[k] == null ? '' : b[k]).trim(); if (v) o[k] = v; });
    return o;
  }
  function describeScope(sc) {
    var parts = [], f = H.facets(), topics = (f && f.by_topic && f.by_topic.top) || [];
    if (sc.ids) return fmtN(sc.ids.length) + ' selected item' + (sc.ids.length === 1 ? '' : 's');
    if (sc.q) parts.push('keyword "' + sc.q + '"');
    if (sc.kind) parts.push(sc.kind === 'news' ? 'news only' : 'social only');
    if (sc.platform) parts.push(L.platform(sc.platform));
    if (sc.source) parts.push('source ' + sc.source);
    if (sc.topic) parts.push(sc.topic === 'unclassified' ? 'unclassified topics' : 'topic ' + ((topics.filter(function (t) { return t.topic_id === sc.topic; })[0] || {}).name || sc.topic));
    if (sc.lang) parts.push(L.lang(sc.lang));
    if (sc.from || sc.to) parts.push((sc.from || 'start') + ' to ' + (sc.to || 'now'));
    if (sc.days) parts.push('last ' + (sc.days === 1 ? '24 hours' : sc.days + ' days'));
    if (sc.analysed) parts.push(sc.analysed === 'yes' ? 'analysed only' : 'not analysed yet');
    return parts.length ? parts.join(', ') : 'all items in the Realtime Database';
  }
  function scopeObj() {
    if (I.scope === 'selection') return { ids: Workbench.selection() };
    if (I.scope === 'overview') { var o = Ev.scopes().overview, s = {}; if (o.kind) s.kind = o.kind; if (o.platform) s.platform = o.platform; if (o.days) s.days = o.days; return s; }
    return brScope();
  }
  function buildIns() {
    if (I.built) return; I.built = true;
    el.ins.innerHTML = '<div class="ins" id="ins"></div><div class="sr" id="ins-live" aria-live="polite"></div>';
    el.ins.addEventListener('click', function (e) {
      var pb = e.target.closest('[data-pop]');
      if (pb) { togglePop(pb); return; }
      var b = e.target.closest('[data-act]');
      if (b) {
        var a = b.getAttribute('data-act');
        if (a === 'ins-phrase') phrase();
        else if (a === 'ins-refresh') loadIns();
        else if (a === 'ins-pop-close') closePop();
        return;
      }
      if (I.pop && !e.target.closest('.ins-pop')) closePop();
    });
    el.ins.addEventListener('change', function (e) {
      if (e.target.name === 'ins-scope') { I.scope = e.target.value; I.ph = { state: 'idle' }; loadIns(); }
    });
    el.ins.addEventListener('input', function (e) {
      if (e.target.id === 'ins-q') { I.q = e.target.value; var c = $('#ins-qc'); if (c) c.textContent = I.q.length + ' / 300'; }
    });
    document.addEventListener('keydown', function (e) { if (e.key === 'Escape' && I.pop) { var b = I.pop.btn; closePop(); if (b) b.focus(); e.stopPropagation(); } }, true);
    document.addEventListener('click', function (e) { if (I.pop && !e.target.closest('.ins-pop') && !e.target.closest('[data-pop]')) closePop(); });
  }
  function closePop() { if (I.pop) { I.pop.btn.setAttribute('aria-expanded', 'false'); if (I.pop.node.parentNode) I.pop.node.parentNode.removeChild(I.pop.node); I.pop = null; } }
  function togglePop(btn) {
    var same = I.pop && I.pop.btn === btn; closePop(); if (same) return;
    var key = btn.getAttribute('data-pop'), k = I.data && I.data.facts && I.data.facts.keys && I.data.facts.keys[key], li = btn.closest('li, .ins-phr') || btn.parentNode;
    var body;
    if (!k) body = '<b>' + esc(key) + '</b><span>No further detail is stored for this reference.</span>';
    else body = '<b>' + esc(k.label) + '</b><span class="ins-pop-n">' + esc(fmtN(k.value)) + (k.of != null ? ' of ' + esc(fmtN(k.of)) + (k.pct != null ? ' (' + esc(String(k.pct)) + '%)' : '') : '') + '</span><small>' + (k.of != null ? 'The number is the count; the denominator is the group it is divided by.' : 'A count taken directly from the database.') + '</small>';
    var node = document.createElement('div'); node.className = 'ins-pop'; node.setAttribute('role', 'dialog'); node.setAttribute('aria-label', 'Number behind this statement');
    node.innerHTML = body + '<button type="button" class="link-btn" data-act="ins-pop-close">Close</button>';
    li.style.position = 'relative'; li.appendChild(node);
    var left = Math.max(0, Math.min(btn.offsetLeft, li.clientWidth - node.offsetWidth)); node.style.left = left + 'px'; node.style.top = (btn.offsetTop + btn.offsetHeight + 6) + 'px';
    btn.setAttribute('aria-expanded', 'true'); I.pop = { btn: btn, node: node };
  }
  function rowLabel(id) {
    var rows = (I.data && I.data.facts && I.data.facts.rows) || [], i = rows.map(function (r) { return r.id; }).indexOf(id);
    return i >= 0 ? 'Row ' + (i + 1) : 'Row';
  }
  function rowText(id) { var r = ((I.data && I.data.facts && I.data.facts.rows) || []).filter(function (x) { return x.id === id; })[0]; return r ? r.text : ''; }
  function cite(c) {
    if (/^fact:/.test(c)) { var key = c.slice(5), k = I.data.facts.keys && I.data.facts.keys[key]; return '<button type="button" class="ins-cite fact" data-pop="' + esc(key) + '" aria-expanded="false" aria-haspopup="dialog" title="Show the number behind this statement">' + esc(clip(k ? k.label : key, 34)) + '</button>'; }
    var t = rowText(c);
    return '<a class="ins-cite rowc" href="#evidence/item/' + encodeURIComponent(c) + '" title="' + esc(t ? clip(t, 160) : 'Open this item') + '">' + esc(rowLabel(c)) + '<span class="sr">: open this item\'s case file</span></a>';
  }
  function renderIns() {
    var box = $('#ins'); if (!box) return;
    var d = I.data, sc = scopeObj(), nsel = Workbench.selection().length, opts = [
      ['browse', 'Browse filters', 'The filters now set in the Browse tab: ' + describeScope(brScope()) + '.', false],
      ['overview', 'Overview filters', 'The scope now set in the Overview tab: ' + describeScope(scopeFor('overview')) + '.', false],
      ['selection', 'Workbench selection', nsel ? fmtN(nsel) + ' selected item' + (nsel === 1 ? '' : 's') + '.' : 'Nothing is selected in the Workbench yet.', !nsel]];
    var h = '<fieldset class="ev-filters ins-scope"><legend class="lbl">Scope</legend>' + opts.map(function (o) {
      return '<label class="ins-opt' + (o[3] ? ' dis' : '') + '"><input type="radio" name="ins-scope" value="' + o[0] + '"' + (I.scope === o[0] ? ' checked' : '') + (o[3] ? ' disabled' : '') + '><span><b>' + esc(o[1]) + '</b><small>' + esc(o[2]) + '</small></span></label>';
    }).join('') + '</fieldset>';
    h += '<p class="ins-ground" role="note"><b>Grounded:</b> every statement cites rows or numbers from the connected database; no outside knowledge is used. ' + term('grounded insight', 'What does grounded mean?') + '</p>';
    if (I.loading && !d) h += '<p class="ev-scope" role="status"><span class="cc-spin" aria-hidden="true"></span> Reading the database for this scope...</p>';
    else if (I.error) h += '<div class="cc-note warn" role="alert"><div><b>The insights could not be built.</b> ' + esc(I.error) + '</div><button type="button" class="btn sm" data-act="ins-refresh">Try again</button></div>';
    if (d) {
      var f = d.facts || {};
      h += '<p class="ev-scope ins-line" role="status"><b>' + esc(d.scope_line) + '</b><span class="sub"> Scope: ' + esc(d.scope_desc || describeScope(sc)) + '.</span></p>';
      h += '<section class="panel ov-panel ins-panel" aria-labelledby="ins-h"><header class="ov-ph"><h2 id="ins-h" class="panel-title">Insights</h2><p class="ov-sub">Statements written from the numbers only. Select a chip to see the number behind it, or a row to open the item.</p><p class="ov-badges">' + H.BADGE_DB + '</p></header>';
      if (!d.template_insights.length) h += '<div class="ov-empty"><b>No items in this scope</b><span>There is nothing to describe. Widen the filters or choose another scope.</span></div>';
      else h += '<ol class="ins-list">' + d.template_insights.map(function (x) { return '<li class="ins-it"><span class="ins-t">' + esc(x.text) + '</span><span class="ins-cs">' + (x.cites || []).map(cite).join('') + '</span></li>'; }).join('') + '</ol>';
      h += '</section>';
      /* phrasing */
      var pz = d.phrasing || {}, off = pz.available === false, busy = I.ph.state === 'loading', empty = !f.n_items;
      h += '<section class="panel ov-panel ins-panel" aria-labelledby="ins-g-h"><header class="ov-ph"><h2 id="ins-g-h" class="panel-title">Ask Gemma to phrase this</h2><p class="ov-sub">Gemma rewrites the statements above as a short paragraph. The server checks every number and every cited row against the database facts before showing it, and refuses anything it cannot verify.</p></header>' +
        '<label class="lbl" for="ins-q">Question (optional)</label><textarea id="ins-q" class="ins-q" rows="2" maxlength="300" placeholder="For example: Which platforms drive most of these items?"' + (off || busy || empty ? ' disabled' : '') + '>' + esc(I.q) + '</textarea>' +
        '<p class="ins-qc" id="ins-qc">' + I.q.length + ' / 300</p>' +
        '<div class="ev-actions"><button type="button" class="btn" data-act="ins-phrase" data-fk="iph"' + (off || busy || empty ? ' disabled' : '') + ' aria-describedby="ins-why">' + (busy ? 'Phrasing... up to a minute' : 'Ask Gemma to phrase this') + '</button>' +
        (off ? '<button type="button" class="btn ghost" data-act="ins-refresh">Check again</button>' : '') + '</div>' +
        '<p class="note" id="ins-why">' + (off ? 'Not available right now: ' + esc(pz.reason || 'the phrasing service is not reachable') + '. The statements above do not need it. The server checks again about a minute after a failure.' : empty ? 'There are no items in this scope to phrase.' : busy ? 'Gemma is writing; the check of its answer happens before anything is shown.' : 'Optional. The statements above are always available without it.') + '</p>';
      if (busy) h += '<p class="ev-scope" role="status"><span class="cc-spin" aria-hidden="true"></span> Waiting for Gemma...</p>';
      var ph = d.phrased;
      if (ph && ph.refused) h += '<div class="cc-note plain ins-refusal" role="status"><b>Gemma\'s phrasing was not shown.</b> ' + esc(ph.reason || 'It could not be checked against the data.') + '<span class="sub">This is the safeguard working: the statements above are unaffected and still cite the database.</span></div>';
      else if (ph && ph.text) h += '<div class="ins-phr"><p class="ins-phr-h">' + ob('ob-fnp solid', 'Phrased by Gemma', 'Written by the Gemma language model through the FNP service; every number and row it cites was checked against the database facts.') + '</p><p class="ins-phr-t">' + esc(ph.text) + '</p>' +
        '<p class="ins-cs">' + (ph.citations || []).map(cite).join('') + '</p><p class="note">Checked by the server: every cited row is in the data and every number appears in the facts. Source: ' + esc(ph.source || 'FNP phrasing service') + '.</p></div>';
      h += '</section>';
      /* the numbers and the rows behind everything */
      var keys = f.keys || {}, kk = Object.keys(keys);
      if (kk.length) h += '<details class="ins-det"><summary>Numbers used (' + kk.length + ')</summary><div class="wb-tbl-wrap"><table class="wb-tbl"><caption class="sr">Every number the statements cite, with its denominator</caption><thead><tr><th scope="col">Number</th><th scope="col">Count</th><th scope="col">Out of</th><th scope="col">Share</th></tr></thead><tbody>' +
        kk.map(function (k) { var x = keys[k]; return '<tr><th scope="row" data-l="Number">' + esc(x.label) + '</th><td class="num" data-l="Count">' + esc(fmtN(x.value)) + '</td><td class="num" data-l="Out of">' + (x.of == null ? 'n/a' : esc(fmtN(x.of))) + '</td><td class="num" data-l="Share">' + (x.pct == null ? 'n/a' : esc(String(x.pct)) + '%') + '</td></tr>'; }).join('') + '</tbody></table></div></details>';
      var rows = f.rows || [];
      if (rows.length) h += '<details class="ins-det"><summary>Rows behind these statements (' + rows.length + ')</summary><p class="note">' + esc(f.sample_method || 'A sample of the items in scope') + '. Select one to open its case file.</p><ol class="ins-rows">' +
        rows.map(function (r, i) { return '<li><a href="#evidence/item/' + encodeURIComponent(r.id) + '"><b>Row ' + (i + 1) + '</b> ' + esc(clip(r.text, 200)) + '</a><small>' + esc(L.platform(r.platform)) + ' - ' + esc(H.sourceName(r.source)) + (r.published_at ? ' - ' + esc(H.ago(r.published_at)) : '') + '</small></li>'; }).join('') + '</ol></details>';
    }
    keepFocus(function () { var q = $('#ins-q'), had = document.activeElement === q, pos = had ? q.selectionStart : 0; box.innerHTML = h; if (had) { var q2 = $('#ins-q'); if (q2) { q2.focus(); q2.setSelectionRange(pos, pos); } } });
  }
  function scopeFor(which) { var o = Ev.scopes().overview, s = {}; if (which === 'overview') { if (o.kind) s.kind = o.kind; if (o.platform) s.platform = o.platform; if (o.days) s.days = o.days; } return s; }
  function loadIns() {
    var my = ++I.seq, sc = scopeObj();
    I.loading = true; I.error = null; closePop();
    if (I.scope === 'selection' && !sc.ids.length) { I.scope = 'browse'; sc = scopeObj(); }
    renderIns();
    call('POST', '/api/rt/insights', { scope: sc }).then(function (d) {
      if (my !== I.seq) return;
      I.loading = false; I.data = d;
      if (I.ph.state !== 'loading') I.ph = { state: 'idle' };
      renderIns(); var n = $('#ins-live'); if (n) n.textContent = d.scope_line;
    }).catch(function (e) {
      if (my !== I.seq) return;
      I.loading = false;
      if (H.failure(e)) return;
      I.error = errText(e); I.data = null; renderIns();
    });
  }
  function phrase() {
    if (!I.data || I.ph.state === 'loading') return;
    var my = ++I.seq, sc = scopeObj(), body = { scope: sc, phrase: true }, q = (I.q || '').trim();
    if (q) body.question = q;
    I.ph = { state: 'loading' }; renderIns();
    call('POST', '/api/rt/insights', body).then(function (d) {
      if (my !== I.seq) return;
      I.ph = { state: 'done' }; I.data = d; renderIns();
      var n = $('#ins-live'); if (n) n.textContent = d.phrased && d.phrased.refused ? 'Gemma\'s phrasing was not shown.' : d.phrased ? 'Gemma\'s phrasing is shown.' : 'Gemma phrasing is not available.';
    }).catch(function (e) {
      if (my !== I.seq) return;
      I.ph = { state: 'idle' };
      if (H.failure(e)) return;
      renderIns(); var n = $('#ins-why'); if (n) n.textContent = 'The request failed: ' + errText(e);
    });
  }
  Ev.register('insights', {
    enter: function () { buildIns(); el.ins.hidden = false; if (I.scope === 'selection' && !Workbench.selection().length) I.scope = 'browse'; loadIns(); },
    leave: function () { closePop(); I.seq++; I.loading = false; if (I.ph.state === 'loading') I.ph = { state: 'idle' }; }
  });
})();
