/* Evidence Base: the realtime posts and articles (route #evidence). Four sub-tabs: Overview (statistics), Browse (list | case file,
   like the Desk), Workbench and Insights (built in workbench.js, which plugs in through Evidence.register and the helpers in Evidence.h).
   Data comes only from /api/rt/gate, /api/rt/status, /api/rt/stats, /api/rt/items, /api/rt/item/<id> and (Workbench, Insights) /api/rt/analyse,
   /api/rt/jobs/<id> and /api/rt/insights. Everything in an item (title, text,
   source names, technique names, span text) is untrusted data: it only ever reaches the page through esc(), never as raw HTML.
   Every field is badged "From the database" (neutral) or "Inferred by FNP" (accent); an item FNP has not analysed says "Not analysed yet". */
(function () {
  'use strict';
  var $ = function (s, r) { return (r || document).querySelector(s); };
  var L = window.Labels, F = window.FNPUI, C = window.Case, esc = L.esc, fmtN = F.fmtN, term = L.term, root = document.documentElement;
  var OFFSET_MAX = 10000, PAGE = 40, MOBILE = window.matchMedia('(max-width: 959px)');

  /* ---------- small helpers ---------- */
  var clockOffset = 0; /* server time minus browser time, so "12 min ago" is right even when this computer's clock is off */
  function nowMs() { return Date.now() + clockOffset; }
  var P = window.FNPPure;
  function minutesSince(iso) { return P.minutesBetween(iso, nowMs()); }
  var dur = P.dur, pct = P.pct, n2 = P.n2, plural = P.plural, kindName = P.kindName, sourceName = P.sourceName, oneLine = P.oneLine, itemTitle = P.itemTitle;
  function inT(iso) { return P.inAt(iso, nowMs()); }
  function ago(iso) { return P.agoAt(iso, nowMs()); }
  function put(sel, html) { var el = typeof sel === 'string' ? $(sel) : sel; if (el) el.innerHTML = html; return el; }
  var BADGE_DB = '<span class="ob ob-db" title="Stored by the Realtime Database, shown unchanged">From the database</span>';
  var BADGE_FNP = '<span class="ob ob-fnp" title="Estimated by the FNP analysis pipeline, not stored in the database">Inferred by FNP</span>';
  var BADGE_FN = '<span class="ob ob-db" title="Matched against the fact-check index, which is a separate source">Fake News Database</span>';
  var ICON = {
    news: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="4" y="4.5" width="16" height="15" rx="1.5"/><path d="M8 9h8M8 12.5h8M8 16h5"/></svg>',
    social: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M5 5h14a1 1 0 0 1 1 1v9a1 1 0 0 1-1 1h-8l-4.5 3.5V16H5a1 1 0 0 1-1-1V6a1 1 0 0 1 1-1Z"/></svg>',
    chart: '<path d="M4 20V4M4 20h16"/><path d="M8 16v-4M12 16V8M16 16v-6"/>', layers: '<path d="m12 3 9 5-9 5-9-5 9-5Z"/><path d="m3 12.5 9 5 9-5"/><path d="m3 17 9 5 9-5"/>',
    trans: '<path d="M4 6h9M8.5 4v2M6 6c.5 3.5 3 6 6.5 7.5M12 6c-.5 3-3 6-7 8"/><path d="m13 20 4-9 4 9M14.5 17h5"/>', warn: '<path d="M12 3.5 21.5 20h-19L12 3.5Z"/><path d="M12 10v4.5M12 17.2v.1"/>',
    db: '<ellipse cx="12" cy="6" rx="7.5" ry="3"/><path d="M4.5 6v6c0 1.7 3.4 3 7.5 3s7.5-1.3 7.5-3V6M4.5 12v6c0 1.7 3.4 3 7.5 3s7.5-1.3 7.5-3v-6"/>',
    globe: '<circle cx="12" cy="12" r="8.5"/><path d="M3.5 12h17M12 3.5c2.6 2.4 3.8 5.3 3.8 8.5s-1.2 6.1-3.8 8.5c-2.6-2.4-3.8-5.3-3.8-8.5S9.4 5.9 12 3.5Z"/>',
    tag: '<path d="M4 4h8l8 8-8 8-8-8V4Z"/><circle cx="8.5" cy="8.5" r="1.2"/>', clock: '<circle cx="12" cy="12" r="8.5"/><path d="M12 7v5l3.2 2"/>',
    list: '<path d="M8 6h12M8 12h12M8 18h12M4 6h.1M4 12h.1M4 18h.1"/>', check: '<path d="M12 3 4.5 6v5.5c0 4.6 3 7.9 7.5 9.5 4.5-1.6 7.5-4.9 7.5-9.5V6L12 3Z"/><path d="m8.8 12 2.3 2.3 4.2-4.6"/>'
  };
  /* a bare (?) that opens the glossary definition; the label is already printed next to it */
  function q(key) { var g = L.glossaryEntry(key); return g ? '<span class="term term-bare" tabindex="0" data-term="' + esc(key) + '"><span class="term-q" aria-hidden="true">?</span><span class="sr"> Definition of ' + esc(g.label) + '</span></span>' : ''; }
  function vchip(v) { return '<span class="chip v-' + esc(v) + '">' + esc(L.verdict(v)) + '</span>'; }
  function svgIco(k) { return '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">' + (ICON[k] || '') + '</svg>'; }

  /* ---------- network: strict JSON in, typed errors out ---------- */
  function rt(path) {
    return fetch(path, { headers: { Accept: 'application/json' } }).then(function (r) {
      return r.json().catch(function () { return {}; }).then(function (j) {
        if (!r.ok) { var e = new Error((j && j.error) || ('HTTP ' + r.status)); e.status = r.status; e.code = j && j.error; e.hint = j && j.hint; e.retry = r.headers.get('Retry-After'); throw e; }
        return j;
      });
    }, function () { var e = new Error('network'); e.network = true; throw e; });
  }
  var el = {
    top: $('#ev-top'), src: $('#ev-src'), main: $('#ev-main'), notice: $('#ev-notice'), ov: $('#ev-overview'), br: $('#ev-browse'),
    wb: $('#ev-workbench'), ins: $('#ev-insights'),
    tabOv: $('#evt-overview'), tabBr: $('#evt-browse'), tabWb: $('#evt-workbench'), tabIns: $('#evt-insights')
  };
  var ext = {}; /* sub-tabs built elsewhere: name -> {enter(ctx), leave()} */
  var tab = '', started = { ov: false, br: false }, gateP = null, facets = null, facetsAt = 0, facetsP = null, tickTimer = null, sched = null, schedAt = 0;

  /* ---------- gate: the Evidence Base is closed unless the site has a login (or the administrator opened it on purpose) ---------- */
  function checkGate(force) {
    if (gateP && !force) return gateP;
    gateP = rt('/api/rt/gate').then(function (g) { return { open: !!(g && (g.open || g.auth)), auth: !!(g && g.auth), info: g }; });
    gateP.catch(function () { gateP = null; });
    return gateP;
  }
  function notice(kind, o) {
    var box = el.notice, h = '';
    if (kind === 'gate') {
      h = '<div class="ev-card-note" role="region" aria-labelledby="ev-gate-h"><span class="ev-card-ico" aria-hidden="true">' + svgIco('check') + '</span><div>' +
        '<h2 id="ev-gate-h">The Evidence Base needs a login</h2>' +
        '<p>The Evidence Base shows the live posts and articles of the Realtime Database, so it is only available behind a login. Ask the administrator to turn the login on for this site, then reload this page.</p>' +
        '<p class="note">Nothing from the Realtime Database is shown or sent to your browser until then. The Fake News Database (Overview, Desk, Claim Check) is not affected.</p>' +
        '<details class="ev-admin"><summary>For administrators</summary><p>Set <code>FNP_AUTH_USER</code> and <code>FNP_AUTH_PASS</code> in the server\'s environment and restart the server. For a deliberate, temporary test on a private machine only, <code>FNP_RT_ALLOW_OPEN=1</code> opens these routes without a login.' +
        (o && o.hint ? ' Server hint: <q>' + esc(o.hint) + '</q>.' : '') + '</p></details>' +
        '<div class="ev-actions"><button type="button" class="btn" data-act="gate-retry">Check again</button><a class="btn ghost" href="#desk">Open the Desk</a></div></div></div>';
    } else if (kind === 'limit') {
      h = '<div class="ev-card-note" role="alert"><span class="ev-card-ico" aria-hidden="true">' + svgIco('clock') + '</span><div><h2>Too many requests</h2><p>The server is limiting how fast this page can ask for data. Wait ' +
        esc(o && o.retry ? o.retry + ' seconds' : 'a moment') + ' and try again.</p><div class="ev-actions"><button type="button" class="btn" data-act="reload">Try again</button></div></div></div>';
    } else if (kind === 'auth') {
      h = '<div class="ev-card-note" role="alert"><span class="ev-card-ico" aria-hidden="true">' + svgIco('check') + '</span><div><h2>Sign in again</h2><p>The server asked for a login that this page did not have. Reload the page and enter your credentials when the browser asks.</p>' +
        '<div class="ev-actions"><button type="button" class="btn" data-act="hard-reload">Reload the page</button></div></div></div>';
    } else {
      h = '<div class="ev-card-note" role="alert"><span class="ev-card-ico" aria-hidden="true">' + svgIco('warn') + '</span><div><h2>' + esc(o && o.network ? 'The server could not be reached' : 'The Evidence Base could not be loaded') + '</h2><p>' +
        esc(o && o.network ? 'Check your connection, then try again.' : (o && o.message ? 'The server said: ' + o.message + '. ' : '') + 'Try again in a moment.') + '</p><div class="ev-actions"><button type="button" class="btn" data-act="reload">Try again</button></div></div></div>';
    }
    box.innerHTML = h; box.hidden = false;
    el.ov.hidden = true; el.br.hidden = true; el.wb.hidden = true; el.ins.hidden = true;
    if (kind === 'gate') { put(el.src, srcChip() + '<span class="ev-src-t">Closed: login required</span>'); }
  }
  function clearNotice() { el.notice.hidden = true; el.notice.innerHTML = ''; }
  /* route an API failure to the right calm state; returns true when it was handled as a page-level state */
  function failure(e) {
    if (e && e.status === 403 && e.code === 'login_required') { gateP = null; notice('gate', e); return true; }
    if (e && e.status === 401) { notice('auth'); return true; }
    if (e && e.status === 429) { notice('limit', e); return true; }
    if (e && (e.network || (e.status && e.status >= 500) || e.status === 503)) { notice('error', e); return true; }
    return false;
  }
  el.notice.addEventListener('click', function (e) {
    var b = e.target.closest('[data-act]'); if (!b) return;
    var a = b.getAttribute('data-act');
    if (a === 'gate-retry') { gateP = null; clearNotice(); enter(currentArg, true); }
    else if (a === 'reload') { clearNotice(); enter(currentArg, true); }
    else if (a === 'hard-reload') location.reload();
  });

  /* ---------- the source banner: which database, how fresh, how much has been analysed ---------- */
  function srcChip() { return '<span class="src-chip rt"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' + ICON.db + '</svg>Realtime Database</span>'; }
  function loadFacets(force) {
    if (facetsP && !force && Date.now() - facetsAt < 60000) return facetsP;
    facetsAt = Date.now();
    facetsP = rt('/api/rt/stats').then(function (d) {
      var g = Date.parse(d.generated_at); if (!isNaN(g)) clockOffset = g - Date.now();
      facets = d; renderSrc(); return d;
    });
    facetsP.catch(function () { facetsP = null; });
    loadSchedule();
    return facetsP;
  }
  /* the scheduled deep analysis (read-only status): optional, so a failure just leaves the line out */
  function loadSchedule() {
    schedAt = Date.now();
    return rt('/api/rt/schedule').then(function (s) {
      sched = s; var g = Date.parse(s.generated_at); if (!isNaN(g)) clockOffset = g - Date.now();
      renderSrc(); renderSchedule(); return s;
    }).catch(function () { return null; });
  }
  /* 'Deep analysis: last run 2.1 h ago (80 analysed), next run in 52 min' / calm off state */
  /* '12 analysed, 5 partial, 3 degraded, 1 error': the counts of one scheduled run (older rows have no partial column) */
  function cycleCounts(r) {
    var t = fmtN(r.analysed || 0) + ' analysed';
    if (r.partial) t += ', ' + fmtN(r.partial) + ' partial';
    if (r.degraded) t += ', ' + fmtN(r.degraded) + ' degraded';
    if (r.errors) t += ', ' + plural(r.errors, 'error', 'errors');
    return t;
  }
  function deepLine() {
    if (!sched) return '';
    if (!sched.enabled) return '<span class="ev-src-d" title="The server was started without the scheduled analysis, so only manual Claim Check runs add deep verdicts.">Deep analysis: scheduled analysis is off</span>';
    var lr = sched.last_run, t = 'Deep analysis: ';
    if (sched.running_now) t += 'running now' + (lr && lr.ended_at ? ', last run ' + ago(lr.ended_at) + ' (' + cycleCounts(lr) + ')' : '');
    else {
      t += lr && lr.ended_at ? 'last run ' + ago(lr.ended_at) + ' (' + cycleCounts(lr) + ')' : 'no run yet';
      t += sched.next_run_at ? ', next run ' + inT(sched.next_run_at) : sched.state === 'waiting_for_sync' ? ', waiting for the first sync' : '';
    }
    return '<span class="ev-src-d">' + esc(t) + '</span>';
  }
  /* '0.16%' for a small share, '12.5%' otherwise: a first export covers a sliver of the window and must not round to 0% */
  function fmtPct(n, d) { if (!d) return 'n/a'; var p = n / d * 100; return p > 0 && p < 0.01 ? 'under 0.01%' : (p > 0 && p < 1 ? p.toFixed(2) : p.toFixed(1)) + '%'; }
  function nextRunText(b) {
    var d = b && b.next_run_at ? Date.parse(b.next_run_at) : NaN;
    if (isNaN(d)) return '';
    return (d - nowMs()) / 60000 < -30 ? 'next run overdue (was due ' + ago(b.next_run_at) + ')' : 'next run ' + inT(b.next_run_at);
  }
  function failNote(b) {
    var n = b && b.n_failed_this_run;
    return n > 0 ? '<span class="ev-src-d ev-calm" title="Items FNP could not score in its latest run. This does not affect the other items.">' + esc(plural(n, 'item', 'items') + ' could not be scored in the latest run.') + '</span>' : '';
  }
  /* 'FNP batch: 1,000 of 628,535 scored (0.16%), last export 12 min ago, next run in 2.8 h' from meta.json via /api/rt/stats */
  function batchLine() {
    var b = facets && facets.coverage && facets.coverage.batch, at = b && b.built_at;
    if (b && b.present && b.n_in_window != null) {
      var t = 'FNP batch: ' + fmtN(b.scored) + ' of ' + fmtN(b.n_in_window) + ' scored (' + fmtPct(b.scored, b.n_in_window) + '), last export ' + (at ? ago(at) : 'time unknown'), nx = nextRunText(b);
      return '<span class="ev-src-d" title="' + esc((at ? 'Export built ' + L.dateTimeUTC(at) + '. ' : '') + fmtN(b.n_unscored_in_window == null ? 0 : b.n_unscored_in_window) + ' items in the window are not scored yet.') + '">' + esc(t + (nx ? ', ' + nx : '')) + '</span>' + failNote(b);
    }
    return '<span class="ev-src-d" title="' + esc(at ? L.dateTimeUTC(at) : '') + '">' + esc('FNP batch: ' + (at ? 'last export ' + ago(at) : 'no export yet')) + '</span>';
  }
  function renderSrc() {
    if (!facets) return;
    var fr = facets.freshness || {}, cv = facets.coverage || {}, bk = facets.by_kind || {};
    var since = fr.synced_at ? minutesSince(fr.synced_at) : null, old = fr.newest_item_at ? minutesSince(fr.newest_item_at) : null;
    var stale = since != null && since > 30;
    var h = srcChip();
    h += '<span class="ev-src-t">' + (fr.synced_at ? 'Synced ' + esc(ago(fr.synced_at)) : 'Never synced') + (fr.newest_item_at ? ', newest item ' + esc(dur(old)) + ' old' : '') + '</span>';
    if (stale) h += '<span class="ev-src-w" title="The cache is refreshed by a sync thread. If it is not running, no new items arrive.">Sync is behind</span>';
    h += '<span class="ev-src-n opt">' + esc(fmtN(facets.total)) + ' items: ' + esc(fmtN(bk.news || 0)) + ' news, ' + esc(fmtN(bk.social || 0)) + ' social</span>';
    var nB = cv.batch ? cv.batch.analysed : 0, nD = cv.deep ? cv.deep.analysed : 0, nT = cv.items_in_scope || 0;
    h += '<span class="ev-src-a" title="Items with an FNP batch (cheap pass) record, and items with a deep Claim Check verdict. An item can have both.">Batch cheap pass: ' + esc(fmtN(nB)) + ' of ' + esc(fmtN(nT)) + ' &middot; Deep verdict: ' + esc(fmtN(nD)) + ' of ' + esc(fmtN(nT)) + (cv.deep && cv.deep.partial_n ? ' (+' + esc(fmtN(cv.deep.partial_n)) + ' partial)' : '') + ' items</span>';
    h += '<span class="ev-src-r2">' + deepLine() + batchLine() + '</span>';
    put(el.src, h);
  }
  function startTick() { stopTick(); tickTimer = setInterval(function () { if (!document.hidden) { renderSrc(); renderSchedule(); if (Date.now() - schedAt > 120000) loadSchedule(); } }, 30000); }
  function stopTick() { if (tickTimer) { clearInterval(tickTimer); tickTimer = null; } }

  /* =====================================================================================================================
     OVERVIEW
     ===================================================================================================================== */
  var OV = { kind: '', platform: '', days: 7 }, ovSeq = 0, ovData = null, ovBuilt = false;
  function scopeText() {
    return (OV.kind ? (OV.kind === 'news' ? 'News only' : 'Social only') : 'News and social') + ', ' + (OV.platform ? L.platform(OV.platform) : 'all platforms') + ', last ' + (OV.days === 1 ? '24 hours' : OV.days + ' days');
  }
  function panel(id, icon, title, sub, badges, cls) {
    return '<section class="panel ov-panel ' + (cls || '') + '" aria-labelledby="evp-' + id + '-h"><header class="ov-ph has-ico"><span class="ph-ico" aria-hidden="true">' + svgIco(icon) + '</span>' +
      '<h2 id="evp-' + id + '-h" class="panel-title">' + title + '</h2><p class="ov-sub">' + sub + '</p><p class="ov-badges">' + (badges || '') + '</p></header><div id="evp-' + id + '" class="ev-body"><p class="note">Loading</p></div></section>';
  }
  function buildOverview() {
    if (ovBuilt) return; ovBuilt = true;
    var plat = '';
    el.ov.innerHTML =
      '<div class="ev-filters" role="group" aria-label="Scope of the statistics"><h2 class="sr">Scope</h2>' +
        '<div class="ev-f"><span class="lbl" id="evf-kind-l">Kind</span><div class="vchips" id="evf-kind" role="group" aria-labelledby="evf-kind-l">' +
          [['', 'All'], ['news', 'News'], ['social', 'Social']].map(function (c) { return '<button type="button" class="vchip" data-kind="' + c[0] + '" aria-pressed="' + (OV.kind === c[0]) + '">' + esc(c[1]) + '</button>'; }).join('') + '</div></div>' +
        '<label class="ev-f"><span class="lbl">Platform</span><select id="evf-platform"><option value="">All platforms</option></select></label>' +
        '<div class="ev-f"><span class="lbl" id="evf-win-l">Window</span><div class="seg" role="radiogroup" aria-labelledby="evf-win-l">' +
          [1, 3, 7, 30].map(function (d) { return '<label><input type="radio" name="evf-days" value="' + d + '"' + (OV.days === d ? ' checked' : '') + '><span>' + d + (d === 1 ? ' day' : ' days') + '</span></label>'; }).join('') + '</div></div>' +
      '</div>' +
      '<p class="ev-scope" id="ev-scope" role="status" aria-live="polite"></p>' +
      '<h2 class="sr">Key figures</h2><div class="kpis ev-kpis" id="ev-kpis" aria-label="Key figures"></div>' +
      panel('vol', 'chart', 'Volume per day', 'How many items were published each day, split into news and social. Axes start at zero and are to scale; days are UTC.', BADGE_DB, 'ev-wide') +
      '<div class="ev-grid">' +
        panel('plat', 'layers', 'Top platforms', 'Where the items come from. Select a platform to browse its items.', BADGE_DB) +
        panel('src', 'list', 'Top sources', 'The accounts, channels and sites that publish the most. Select one to browse its items.', BADGE_DB) +
        panel('topic', 'tag', 'Topics ' + term('database classification', '(database classification)'), 'The topic the database assigned. Only news is classified; everything else is shown separately as Unclassified.', BADGE_DB) +
        panel('lang', 'trans', 'Language mix', 'The language the database recorded for each item. Unknown means no language was recorded.', BADGE_DB) +
        panel('tech', 'warn', 'Propaganda techniques', 'Techniques FNP found in the wording, counted over analysed items only (the batch list when an item has one, otherwise its deep result).', BADGE_FNP) +
        panel('pprop', 'chart', 'Propaganda signal distribution', 'How the batch-analysed items spread across the ' + term('uncalibrated signal', 'p_prop signal') + ', in tenths. A ranking aid, not a share of items that are propaganda. Deep results have no p_prop, so they are not in this chart.', BADGE_FNP) +
        panel('cov', 'check', 'Analysis coverage', 'How many items FNP has analysed so far (the export fills in over several runs, so it is partial until then), counted separately for the ' + term('batch cheap pass', 'batch cheap pass') + ' and the ' + term('deep verdict', 'deep verdict') + '. The Inferred figures on this page only describe those items.', BADGE_FNP) +
        panel('howto', 'list', 'How to read this', 'What the FNP figures on this page mean, and what they do not.', BADGE_FNP, 'ev-wide') +
        panel('sched', 'clock', 'Schedule ' + term('scheduled analysis', '(scheduled analysis)'), 'When FNP last ran a deep analysis on this data and when it runs next. Nothing here can start a run.', BADGE_FNP) +
        panel('srcfresh', 'clock', 'Source freshness', 'The newest item in each source table of the local copy, how much of the store it holds, and whether anything new is arriving from that source.', BADGE_DB, 'ev-wide ev-span') +
        panel('fresh', 'clock', 'Freshness and sync', 'When the local copy of the database was last refreshed and how old the newest item is.', BADGE_DB) +
      '</div>';
    el.ov.addEventListener('click', function (e) {
      var k = e.target.closest('[data-kind]');
      if (k && k.closest('#evf-kind')) { OV.kind = k.getAttribute('data-kind'); syncOvFilters(); loadStats(); return; }
      var sfb = e.target.closest('[data-sf]');
      if (sfb) { var k2 = sfb.getAttribute('data-sf'); sfSort = sfSort.key === k2 ? { key: k2, dir: sfSort.dir === 'asc' ? 'desc' : 'asc' } : { key: k2, dir: k2 === 'name' ? 'asc' : 'desc' }; renderSourceFresh(); var again = el.ov.querySelector('[data-sf="' + k2 + '"]'); if (again) again.focus({ preventScroll: true }); return; }
      var b = e.target.closest('[data-go]'); if (b) { var f = {}; f[b.getAttribute('data-go')] = b.getAttribute('data-v'); if (OV.kind && b.getAttribute('data-go') !== 'kind') f.kind = OV.kind; browseWith(f); }
    });
    el.ov.addEventListener('change', function (e) {
      if (e.target.id === 'evf-platform') { OV.platform = e.target.value; loadStats(); }
      else if (e.target.name === 'evf-days') { OV.days = +e.target.value; loadStats(); }
    });
    window.addEventListener('resize', function () { clearTimeout(rz); rz = setTimeout(function () { if (tab === 'overview' && ovData) drawCharts(ovData); }, 150); });
  }
  var rz;
  function syncOvFilters() {
    $('#evf-kind', el.ov) && Array.prototype.forEach.call(el.ov.querySelectorAll('#evf-kind .vchip'), function (b) { b.setAttribute('aria-pressed', b.getAttribute('data-kind') === OV.kind); });
  }
  function fillPlatformSelect() {
    var s = $('#evf-platform'); if (!s || !facets) return;
    var cur = OV.platform;
    s.innerHTML = '<option value="">All platforms</option>' + (facets.by_platform || []).map(function (p) { return '<option value="' + esc(p.platform) + '">' + esc(L.platform(p.platform)) + ' (' + esc(fmtN(p.count)) + ')</option>'; }).join('');
    s.value = cur;
  }

  /* ---- KPI strip ---- */
  function kpi(label, value, sub, icon, cls) {
    return '<div class="kpi panel ' + (cls || '') + '"><span class="ico" aria-hidden="true">' + svgIco(icon) + '</span><span class="lbl">' + label + '</span><b>' + esc(value) + '</b><small>' + sub + '</small></div>';
  }
  function renderKpis(d) {
    var bk = d.by_kind || {}, news = bk.news || 0, soc = bk.social || 0, tot = d.total, cv = d.coverage || {}, lf = d.likely_fake_misleading || {};
    var langs = (d.by_language && d.by_language.top) || [], lead = langs[0];
    var h = '';
    h += kpi('Items in scope', fmtN(tot), esc(scopeText()), 'layers');
    h += kpi('News and social', tot ? pct(soc, tot, 0) + ' social' : 'n/a', tot ? esc(fmtN(news)) + ' news and ' + esc(fmtN(soc)) + ' social, of ' + esc(fmtN(tot)) + ' items' : 'No items in this scope', 'db');
    h += kpi('Sources and platforms', fmtN((d.by_source && d.by_source.distinct) || 0), 'distinct sources on ' + esc(fmtN((d.by_platform || []).length)) + ' platform' + ((d.by_platform || []).length === 1 ? '' : 's'), 'list');
    h += kpi('Languages', fmtN((d.by_language && d.by_language.distinct) || 0), lead ? 'led by ' + esc(L.lang(lead.lang)) + ' (' + esc(pct(lead.count, tot, 0)) + ' of items)' : 'No items in this scope', 'trans');
    var cb = cv.batch || { analysed: 0 }, cd = cv.deep || { analysed: 0 };
    h += kpi(term('analysed coverage', 'Analysed coverage'), tot ? pct(cv.analysed || 0, tot) : 'n/a',
      '<span class="kpi-two"><span>Batch cheap pass (FNP export): <strong>' + esc(fmtN(cb.analysed)) + ' of ' + esc(fmtN(tot)) + '</strong></span><span>Deep verdict: <strong>' + esc(fmtN(cd.analysed)) + ' of ' + esc(fmtN(tot)) + '</strong></span></span>' +
      (cb.present && cb.n_in_window != null ? '<span class="kpi-two"><span>FNP export, whole window: <strong>' + esc(fmtN(cb.scored)) + ' of ' + esc(fmtN(cb.n_in_window)) + '</strong> scored (' + esc(fmtPct(cb.scored, cb.n_in_window)) + ')</span></span>' : '') +
      (cv.analysed ? '' : 'None of the ' + esc(fmtN(tot)) + ' items has been analysed yet'), 'check');
    var ls = lf.by_source || {};
    if (lf.denominator > 0) h += kpi('Likely fake or misleading', pct(lf.count, lf.denominator), esc(fmtN(lf.count)) + ' of ' + esc(fmtN(lf.denominator)) + ' items that have a verdict (' + esc(fmtN(cv.analysed || 0)) + ' analysed; deep ' + esc(fmtN((ls.deep || {}).denominator || 0)) + ', batch ' + esc(fmtN((ls.batch || {}).denominator || 0)) + '). Not the share of all items.', 'warn', 'v-FAKE');
    else h += kpi('Likely fake or misleading', 'n/a', 'No analysed item has a verdict yet, so there is nothing to divide. Verdicts exist only for items that closely match a known fact-check.', 'warn', 'v-FAKE');
    put('#ev-kpis', h);
  }

  /* ---- charts: linear axes starting at zero, ticks at 1/2/5 steps, text and colour from CSS tokens ---- */
  function nice(max, n) { max = max || 1; var raw = max / n, mag = Math.pow(10, Math.floor(Math.log(raw) / Math.LN10)), f = raw / mag, step = (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10) * mag; return { step: step, top: Math.ceil(max / step) * step }; }
  function ticks(sc) { var a = []; for (var v = 0; v <= sc.top + sc.step / 1000; v += sc.step) a.push(v); return a; }
  function tl(v) { return v >= 10000 ? (v / 1000) + 'k' : fmtN(v); }
  function dayAdd(s, n) { var d = new Date(s + 'T00:00:00Z'); d.setUTCDate(d.getUTCDate() + n); return d.toISOString().slice(0, 10); }
  function volumeChart(days) {
    var host = $('#evp-vol'); if (!host) return;
    if (!days.length) { host.innerHTML = empty('No items in this scope', 'Nothing was published in this window, or none has been synced yet.'); return; }
    var rows = [], byDay = {}; days.forEach(function (x) { byDay[x.day] = x; });
    for (var d = days[0].day, guard = 0; d <= days[days.length - 1].day && guard < 400; d = dayAdd(d, 1), guard++) rows.push(byDay[d] || { day: d, news: 0, social: 0, total: 0 });
    var W = Math.max(300, host.clientWidth || 640), H = 250, Lm = 62, R = 10, T = 16, B = 46, pw = W - Lm - R, ph = H - T - B, n = rows.length;
    var mx = Math.max.apply(null, rows.map(function (r) { return r.total; })) || 1, sc = nice(mx, 4), y = function (v) { return T + ph - v / sc.top * ph; };
    var bw = pw / n, w = Math.min(bw * 0.7, 64), every = Math.max(1, Math.ceil(46 / bw));
    var g = '<svg viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="Items per day, news and social stacked. ' + esc(rows.map(function (r) { return L.dateLong(r.day, true) + ' ' + fmtN(r.total); }).join(', ')) + '">';
    ticks(sc).forEach(function (v) { g += '<line class="grid-line" x1="' + Lm + '" y1="' + y(v).toFixed(1) + '" x2="' + (Lm + pw) + '" y2="' + y(v).toFixed(1) + '"/><text x="' + (Lm - 6) + '" y="' + (y(v) + 4).toFixed(1) + '" text-anchor="end">' + tl(v) + '</text>'; });
    rows.forEach(function (r, i) {
      var x = Lm + i * bw + (bw - w) / 2, hN = r.news / sc.top * ph, hS = r.social / sc.top * ph;
      if (r.news) g += '<rect class="f-news" x="' + x.toFixed(1) + '" y="' + (T + ph - hN).toFixed(1) + '" width="' + w.toFixed(1) + '" height="' + hN.toFixed(1) + '"><title>' + esc(L.dateLong(r.day) + ': ' + fmtN(r.news) + ' news') + '</title></rect>';
      if (r.social) g += '<rect class="f-social" x="' + x.toFixed(1) + '" y="' + (T + ph - hN - hS).toFixed(1) + '" width="' + w.toFixed(1) + '" height="' + hS.toFixed(1) + '"><title>' + esc(L.dateLong(r.day) + ': ' + fmtN(r.social) + ' social') + '</title></rect>';
      if (w >= 26 && r.total) g += '<text class="t-strong" x="' + (x + w / 2).toFixed(1) + '" y="' + (y(r.total) - 4).toFixed(1) + '" text-anchor="middle">' + esc(fmtN(r.total)) + '</text>';
      if (i % every === 0) g += '<text x="' + (x + w / 2).toFixed(1) + '" y="' + (T + ph + 16) + '" text-anchor="middle">' + esc(L.dateLong(r.day, true)) + '</text>';
    });
    g += '<line class="axis" x1="' + Lm + '" y1="' + (T + ph) + '" x2="' + (Lm + pw) + '" y2="' + (T + ph) + '"/><line class="axis" x1="' + Lm + '" y1="' + T + '" x2="' + Lm + '" y2="' + (T + ph) + '"/>' +
      '<text class="t-axis" x="' + (Lm + pw / 2) + '" y="' + (H - 4) + '" text-anchor="middle">Day (UTC)</text><text class="t-axis" transform="translate(10 ' + (T + ph / 2) + ') rotate(-90)" text-anchor="middle">Items</text></svg>';
    var tn = rows.reduce(function (a, r) { return a + r.news; }, 0), ts = rows.reduce(function (a, r) { return a + r.social; }, 0);
    host.innerHTML = '<div class="chart">' + g + '</div><div class="legend"><span class="lg-news"><i></i>News ' + esc(fmtN(tn)) + '</span><span class="lg-social"><i></i>Social ' + esc(fmtN(ts)) + '</span></div>' +
      '<p class="ev-denom">Denominator: ' + esc(fmtN(tn + ts)) + ' items on ' + esc(fmtN(n)) + ' day' + (n === 1 ? '' : 's') + ' (' + esc(scopeText()) + '). The first and the latest day are usually partial. Days with no items appear as empty slots.</p>';
  }
  function histChart(buckets, n) {
    var host = $('#evp-pprop'); if (!host) return;
    var W = Math.max(280, host.clientWidth || 320), H = 230, Lm = 44, R = 8, T = 14, B = 46, pw = W - Lm - R, ph = H - T - B, bw = pw / buckets.length;
    var mx = Math.max.apply(null, buckets.map(function (b) { return b.count; })) || 1, sc = nice(mx, 4), y = function (v) { return T + ph - v / sc.top * ph; };
    var g = '<svg viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="Analysed items by propaganda signal, in tenths from 0 to 1">';
    ticks(sc).forEach(function (v) { g += '<line class="grid-line" x1="' + Lm + '" y1="' + y(v).toFixed(1) + '" x2="' + (Lm + pw) + '" y2="' + y(v).toFixed(1) + '"/><text x="' + (Lm - 6) + '" y="' + (y(v) + 4).toFixed(1) + '" text-anchor="end">' + tl(v) + '</text>'; });
    buckets.forEach(function (b, i) {
      var h = b.count / sc.top * ph, x = Lm + i * bw;
      g += '<rect class="f-accent" x="' + (x + 1.5).toFixed(1) + '" y="' + (T + ph - h).toFixed(1) + '" width="' + Math.max(1, bw - 3).toFixed(1) + '" height="' + h.toFixed(1) + '"><title>' + esc(b.lo.toFixed(1) + ' to ' + b.hi.toFixed(1) + ': ' + fmtN(b.count) + ' items') + '</title></rect>';
      if (b.count && bw >= 24) g += '<text x="' + (x + bw / 2).toFixed(1) + '" y="' + (T + ph - h - 4).toFixed(1) + '" text-anchor="middle">' + esc(fmtN(b.count)) + '</text>';
    });
    var step = bw < 30 ? 2 : 1;
    for (var i = 0; i <= buckets.length; i += step) g += '<text x="' + (Lm + i * bw).toFixed(1) + '" y="' + (T + ph + 16) + '" text-anchor="middle">' + (i / buckets.length).toFixed(1) + '</text>';
    g += '<line class="axis" x1="' + Lm + '" y1="' + (T + ph) + '" x2="' + (Lm + pw) + '" y2="' + (T + ph) + '"/><line class="axis" x1="' + Lm + '" y1="' + T + '" x2="' + Lm + '" y2="' + (T + ph) + '"/>' +
      '<text class="t-axis" x="' + (Lm + pw / 2) + '" y="' + (H - 4) + '" text-anchor="middle">Propaganda signal (p_prop)</text><text class="t-axis" transform="translate(11 ' + (T + ph / 2) + ') rotate(-90)" text-anchor="middle">Items</text></svg>';
    host.innerHTML = '<div class="chart">' + g + '</div><p class="ev-denom">Denominator: ' + esc(fmtN(n)) + ' analysed items that carry a signal. Uncalibrated: a high bar means a higher ranking, not a proven share.</p>';
  }
  function empty(title, text) { return '<div class="ov-empty"><b>' + esc(title) + '</b><span>' + esc(text) + '</span></div>'; }

  /* ---- bar rows (to scale against the longest bar; the value is a share of the stated denominator) ---- */
  function barRows(list, denom, denomText, o) {
    o = o || {};
    if (!list.length) return o.empty || empty('Nothing to show', 'No items in this scope.');
    var max = Math.max.apply(null, list.map(function (x) { return x.n; })) || 1;
    return '<div class="ev-bars' + (o.wide ? ' wide' : '') + '">' + list.map(function (x) {
      var inner = '<span class="brow-k" title="' + esc(x.name) + '">' + esc(x.name) + '</span><span class="bar-track"><b style="width:' + (x.n / max * 100).toFixed(1) + '%"></b></span><span class="bar-v">' + esc(fmtN(x.n)) + ' <small>' + esc(pct(x.n, denom, denom && x.n / denom < 0.1 ? 1 : 0)) + '</small></span>';
      return x.go ? '<button type="button" class="brow" data-go="' + esc(x.go[0]) + '" data-v="' + esc(x.go[1]) + '" title="Browse these items">' + inner + '</button>' : '<div class="brow static">' + inner + '</div>';
    }).join('') + '</div><p class="ev-denom">' + denomText + '</p>' + (o.after || '');
  }

  function drawCharts(d) {
    volumeChart(d.by_day || []);
    var cv = d.coverage || {};
    var pp = d.p_prop_distribution || { n: 0, buckets: [] };
    if (!pp.n) histEmpty(cv); else histChart(pp.buckets, pp.n);
  }
  function histEmpty(cv) {
    var deepOnly = cv && cv.analysed && !(cv.batch && cv.batch.analysed);
    put('#evp-pprop', deepOnly ? empty('No batch propaganda signal yet', 'The ' + fmtN(cv.analysed) + ' analysed items in scope have deep verdicts only. A deep result lists techniques but has no p_prop, so this chart fills when the FNP batch export arrives.')
      : empty('No items analysed yet by FNP', 'No items analysed yet by FNP; this fills as the analysis pipeline runs.'));
  }

  function renderOverview(d) {
    var tot = d.total, cv = d.coverage || {};
    renderKpis(d);
    drawCharts(d);
    /* platforms */
    put('#evp-plat', barRows((d.by_platform || []).slice(0, 12).map(function (p) { return { name: L.platform(p.platform), n: p.count, go: ['platform', p.platform] }; }), tot,
      'Denominator: ' + esc(fmtN(tot)) + ' items in scope across ' + esc(fmtN((d.by_platform || []).length)) + ' platform' + ((d.by_platform || []).length === 1 ? '' : 's') + '.'));
    /* sources */
    var bs = d.by_source || { top: [], distinct: 0 }, shown = bs.top.reduce(function (a, s) { return a + s.count; }, 0);
    put('#evp-src', barRows(bs.top.map(function (s) { return { name: sourceName(s.source), n: s.count, go: s.source === '(unknown)' ? null : ['source', s.source] }; }), tot,
      'Denominator: ' + esc(fmtN(tot)) + ' items. These ' + esc(fmtN(bs.top.length)) + ' of ' + esc(fmtN(bs.distinct)) + ' sources account for ' + esc(pct(shown, tot)) + ' of them.', { wide: true }));
    /* topics */
    var tp = d.by_topic || { top: [], classified: 0, unclassified: { count: 0 } }, un = tp.unclassified || { count: 0 }, news = (d.by_kind && d.by_kind.news) || 0;
    var unBlock = '<div class="ev-uncl"><div><b>Unclassified</b> ' + q('unclassified') + '<span>' + esc(fmtN(un.count)) + ' items, ' + esc(pct(un.count, tot)) + ' of ' + esc(fmtN(tot)) + '</span></div>' +
      '<span class="bar-track"><b style="width:' + (tot ? un.count / tot * 100 : 0).toFixed(1) + '%"></b></span>' +
      '<small>' + esc(tp.note || 'Social posts and news the database has not classified yet.') + '</small>' +
      (un.count ? '<button type="button" class="link-btn" data-go="topic" data-v="unclassified">Browse unclassified items</button>' : '') + '</div>';
    put('#evp-topic', (tp.top.length ? barRows(tp.top.map(function (t) { return { name: t.name, n: t.count, go: ['topic', t.topic_id] }; }), tot,
      'Denominator: ' + esc(fmtN(tot)) + ' items in scope. Classified: ' + esc(fmtN(tp.classified)) + ' (' + esc(pct(tp.classified, tot)) + '), of which ' + esc(fmtN(tp.distinct)) + ' topic' + (tp.distinct === 1 ? '' : 's') + '; ' + esc(fmtN(news)) + ' news items in scope.', { wide: true }) :
      empty('No classified items in this scope', OV.kind === 'social' ? 'The database classifies news only, so social posts are all Unclassified.' : 'None of the items in scope has been classified by the database.')) + unBlock);
    /* languages */
    var lg = d.by_language || { top: [], distinct: 0 }, lgShown = lg.top.slice(0, 10);
    put('#evp-lang', barRows(lgShown.map(function (x) { return { name: L.lang(x.lang), n: x.count, go: ['lang', x.lang] }; }), tot,
      'Denominator: ' + esc(fmtN(tot)) + ' items in scope. ' + esc(fmtN(lg.distinct)) + ' language' + (lg.distinct === 1 ? '' : 's') + ' recorded' + (lg.distinct > lgShown.length ? '; the top ' + esc(fmtN(lgShown.length)) + ' are shown' : '') + '.'));
    /* techniques (analysed only) */
    var tm = d.technique_mix || { top: [], denominator_analysed: 0 };
    if (!cv.analysed) put('#evp-tech', empty('No items analysed yet by FNP', 'No items analysed yet by FNP; this fills as the analysis pipeline runs.'));
    else put('#evp-tech', tm.top.length ? barRows(tm.top.slice(0, 10).map(function (t) { return { name: L.technique(t.technique), n: t.items }; }), tm.denominator_analysed,
      'Denominator: ' + esc(fmtN(tm.denominator_analysed)) + ' analysed items (of ' + esc(fmtN(tot)) + ' in scope): ' + esc(fmtN(tm.n_batch || 0)) + ' with a batch record (their batch technique list) and ' + esc(fmtN(tm.n_deep_only || 0)) + ' with a deep result only (its technique classes). Each bar counts items that carry the technique at least once, so bars can add up to more than 100%.', { wide: true }) :
      empty('No techniques found', 'FNP analysed ' + fmtN(cv.analysed) + ' items in scope and found no propaganda technique in them.'));
    /* coverage */
    var bkc = cv.by_kind || {}, covRows = ['news', 'social'].filter(function (k) { return bkc[k]; }).map(function (k) {
      var c = bkc[k]; return '<div class="covrow"><span class="covk">' + esc(kindName(k).replace(' article', '').replace(' post', '')) + '</span><span class="bar-track"><b style="width:' + (c.pct || 0).toFixed(1) + '%"></b></span><span class="bar-v">' + esc(c.pct == null ? 'n/a' : c.pct.toFixed(1) + '%') + ' <small>' + esc(fmtN(c.analysed)) + ' of ' + esc(fmtN(c.items)) + '</small></span></div>';
    }).join('');
    var cb = cv.batch || { analysed: 0, pct: null }, cd = cv.deep || { analysed: 0, pct: null };
    put('#evp-cov', '<dl class="kvt"><dt>' + term('batch cheap pass', 'Batch cheap pass') + ' (FNP export)</dt><dd>' + esc(fmtN(cb.analysed)) + ' of ' + esc(fmtN(tot)) + ' <span class="sub">' + esc(pct(cb.analysed, tot)) + ' of items in scope. FNP\'s fast propaganda pass; it is still working through the window, so this share grows with each run.</span></dd>' +
      (cb.present && cb.n_in_window != null ? '<dt>FNP export window</dt><dd>' + esc(fmtN(cb.scored)) + ' of ' + esc(fmtN(cb.n_in_window)) + ' scored (' + esc(fmtPct(cb.scored, cb.n_in_window)) + ') <span class="sub">' + esc(fmtN(cb.n_unscored_in_window || 0)) + ' not scored yet' + (cb.too_short_in_scope ? '; ' + esc(fmtN(cb.too_short_in_scope)) + ' scored items in this view are too short to score' : '') + '. ' + esc(nextRunText(cb) ? nextRunText(cb).charAt(0).toUpperCase() + nextRunText(cb).slice(1) + '.' : '') + '</span></dd>' : '') +
      '<dt>' + term('deep verdict', 'Deep verdict') + ' (Claim Check)</dt><dd>' + esc(fmtN(cd.analysed)) + ' of ' + esc(fmtN(tot)) + ' <span class="sub">' + esc(pct(cd.analysed, tot)) + ' of items in scope. Slow and selective: a bounded batch every few hours' + (cd.last_at ? ', last stored ' + esc(ago(cd.last_at)) : '') + '.</span></dd>' +
      ((cd.partial_n || 0) ? '<dt>Partial deep results</dt><dd>' + esc(fmtN(cd.partial_n)) + ' of ' + esc(fmtN(tot)) + ' <span class="sub">Partial = propaganda signals only, the verdict timed out. Not counted as analysed with a verdict and never in the likely fake or misleading share.</span></dd>' : '') +
      '<dt>Either, counted once</dt><dd>' + esc(fmtN(cv.analysed || 0)) + ' of ' + esc(fmtN(tot)) + ' <span class="sub">' + esc(fmtN(cv.both || 0)) + ' items have both.</span></dd>' +
      '<dt>With a propaganda signal</dt><dd>' + esc(fmtN(cv.with_p_prop || 0)) + ' of ' + esc(fmtN(cb.analysed)) + ' batch-analysed</dd>' +
      '<dt>With a verdict</dt><dd>' + esc(fmtN(cv.with_verdict || 0)) + ' of ' + esc(fmtN(cv.analysed || 0)) + ' analysed <span class="sub">A deep verdict where one exists, otherwise a batch verdict (only items that closely match a known fact-check get one). Counted once per item.</span></dd>' +
      '<dt>Batch model</dt><dd>' + (d.enrichment && d.enrichment.model_version ? esc(d.enrichment.model_version) + '<span class="sub">Export built ' + esc(d.enrichment.built_at ? L.dateTimeUTC(d.enrichment.built_at) : 'at an unknown time') + '</span>' : 'No FNP export loaded yet') + '</dd></dl>' +
      (covRows ? '<div class="covrows">' + covRows + '</div>' : '') + '<p class="ev-denom">Denominator: ' + esc(fmtN(tot)) + ' items in scope. The bars show items with either kind of analysis.' + (cv.analysed ? '' : ' Nothing has been analysed yet.') + '</p>');
    put('#evp-howto', howHTML(cv.batch || {}, d));
    renderSchedule();
    renderSourceFresh();
    renderFresh(d);
  }
  /* the caveat is FNP's own sentence from meta.json, shown verbatim (escaped); the rest explains the figures in plain words */
  function backlogText(b) {
    if (!b || !b.present || b.n_unscored_in_window == null) return '';
    if (b.estimated_runs != null) return 'At the current rate of ' + fmtN(b.runs_rate) + ' items per run, the ' + fmtN(b.n_unscored_in_window) + ' unscored items clear in about ' + fmtN(b.estimated_runs) + ' more run' + (b.estimated_runs === 1 ? '' : 's') + (b.interval_hours ? ' (one run every ' + String(b.interval_hours) + ' hours)' : '') + '.';
    return fmtN(b.n_unscored_in_window) + ' items in the window are not scored yet. The export does not say how many it scores per run, so no time estimate is shown.';
  }
  function howHTML(b, d) {
    var h = b.caveat ? '<blockquote class="ev-caveat" cite="meta.json"><p>' + esc(b.caveat) + '</p><footer>FNP, in its export notes</footer></blockquote>' : '';
    h += '<ul class="ev-howl"><li><b>Coverage is partial.</b> The batch export scores the window over several runs. ' + (b.present && b.n_in_window != null ? esc(fmtN(b.scored)) + ' of ' + esc(fmtN(b.n_in_window)) + ' items are scored so far (' + esc(fmtPct(b.scored, b.n_in_window)) + '). ' : 'No export has been loaded yet. ') + esc(backlogText(b)) + ' An unscored item is unscored, not clean.</li>' +
      '<li><b>Propaganda signal.</b> A ranking score from an encoder, not a probability that the item is propaganda. It was trained on long news articles and has not been validated on social posts.</li>' +
      '<li><b>Technique bars.</b> Each bar is the highest token probability the encoder gave that technique anywhere in the text. They are uncalibrated and do not add up.</li>' +
      '<li><b>Too short to score.</b> Very short items get no propaganda signal. They are counted separately' + (b.too_short_in_scope ? ' (' + esc(fmtN(b.too_short_in_scope)) + ' in this view)' : '') + ' and are not in the signal chart or the technique mix.</li>' +
      '<li><b>Nearest fact-check.</b> The closest known fact-check by embedding similarity. It is a lookup, not a verdict: a close topic is not the same claim, and a weak match (below 0.72) is shown for reference only.</li></ul>';
    return h;
  }
  var STOP_NOTE = { 'done': 'finished the batch', 'nothing to analyse': 'nothing left to analyse', 'server stopping': 'server restarted', 'interrupted': 'interrupted by a restart', 'engine unreachable': 'stopped early: the engine could not be reached' };
  function stopText(r) {
    if (!r) return '';
    if (STOP_NOTE[r]) return STOP_NOTE[r];
    var m = /^(\d+) of the last (\d+) replies were/.exec(r);
    if (m) return 'stopped early: ' + m[1] + ' of the last ' + m[2] + ' engine replies were busy, degraded or failed';
    m = /^time cap \((\d+) s\)/.exec(r);
    if (m) return 'stopped at the time limit (' + dur(m[1] / 60) + ')';
    if (/^engine slow/.test(r)) return 'the engine was slow (over 40 s per item), so half of the remaining items were left for the next run';
    return String(r).replace(/^./, function (c) { return c.toUpperCase(); });
  }
  function capFirst(t) { return String(t).replace(/^./, function (c) { return c.toUpperCase(); }); }
  function rateText(ro) {
    if (!ro || ro.success_rate == null) return 'No engine replies yet';
    return (ro.success_rate * 100).toFixed(0) + '% of engine replies over the last ' + plural(ro.cycles, 'run', 'runs');
  }
  function renderSchedule() {
    var box = $('#evp-sched'); if (!box) return;
    if (!sched) { box.innerHTML = '<p class="note">Loading</p>'; return; }
    var lr = sched.last_run, rows = (sched.log || []).slice(0, 6), tot = sched.totals || {}, ro = sched.rolling || {};
    var next = !sched.enabled ? 'Scheduled analysis is off on this server.' : sched.running_now ? 'Running now.' : sched.next_run_at ? 'Next run ' + inT(sched.next_run_at) + ' (' + L.dateTimeUTC(sched.next_run_at) + ').' : sched.state === 'waiting_for_sync' ? 'Waiting for the first sync to finish before the first run.' : 'Next run time not set yet.';
    var h = '<dl class="kvt"><dt>Status</dt><dd>' + esc(next) + (sched.enabled ? '<span class="sub">Every ' + esc(dur(sched.interval_s / 60)) + ', up to ' + esc(fmtN(sched.budget)) + ' items per run, one at a time. A run ends early when most engine replies are busy, degraded or failed, or when the engine is slow.</span>' : '<span class="sub">When it is on, FNP runs a bounded deep analysis on new items every few hours. It is off now, so deep verdicts come only from manual Claim Check runs in the Workbench.</span>') + '</dd>' +
      '<dt>Last run</dt><dd>' + (lr && lr.ended_at ? esc(L.dateTimeUTC(lr.ended_at)) + '<span class="sub">' + esc(ago(lr.ended_at)) + ': ' + esc(cycleCounts(lr)) + (lr.median_secs != null ? ', median ' + esc(lr.median_secs.toFixed(0)) + ' s per item' : '') + (lr.stop_reason ? '. ' + esc(capFirst(stopText(lr.stop_reason))) : '') + '</span>' : 'No scheduled run yet') + '</dd>' +
      '<dt>Engine success rate</dt><dd>' + esc(rateText(ro)) + '<span class="sub">Replies that gave a complete result, out of all replies (complete, partial or degraded, busy, failed).</span></dd>' +
      '<dt>Deep verdicts stored</dt><dd>' + esc(fmtN(tot.analysed_deep || 0)) + ' items' + (tot.partial_items ? ' <span class="sub">plus ' + esc(fmtN(tot.partial_items)) + ' with a partial result</span>' : '') + '<span class="sub">From ' + esc(fmtN(tot.cycles || 0)) + ' scheduled run' + (tot.cycles === 1 ? '' : 's') + ' and manual Workbench runs.</span></dd></dl>';
    if (rows.length) {
      h += '<div class="wb-tbl-wrap"><table class="wb-tbl" aria-label="Recent scheduled runs"><thead><tr><th scope="col">Run and why it stopped</th><th scope="col" class="num">Analysed</th><th scope="col" class="num">Partial</th><th scope="col" class="num">Degraded</th><th scope="col" class="num">Errors</th></tr></thead><tbody>' + rows.map(function (r) {
        return '<tr><th scope="row" title="' + esc(L.dateTimeUTC(r.started_at)) + '">' + esc(ago(r.started_at)) + (r.median_secs != null ? '<small> median ' + esc(r.median_secs.toFixed(0)) + ' s</small>' : '') + '<small class="ev-stopnote">' + esc(r.ended_at ? capFirst(stopText(r.stop_reason)) : 'Running') + '</small></th><td class="num" data-l="Analysed">' + esc(fmtN(r.analysed)) + (r.cached_dupes ? '<small> +' + esc(fmtN(r.cached_dupes)) + ' dup.</small>' : '') + '</td><td class="num" data-l="Partial">' + esc(fmtN(r.partial || 0)) + (r.retried ? '<small> ' + esc(fmtN(r.retried)) + ' retried</small>' : '') + '</td><td class="num" data-l="Degraded">' + esc(fmtN(r.degraded)) + '</td><td class="num" data-l="Errors">' + esc(fmtN(r.errors)) + '</td></tr>';
      }).join('') + '</tbody></table></div>';
    }
    h += '<p class="ev-denom">Counts only; the schedule never shows item text. Partial = propaganda signals only, the verdict timed out: the signals are kept, there is no verdict, and the item is retried on later runs (up to 3 times). Degraded counts the replies that timed out.</p>';
    box.innerHTML = h;
  }
  var statusData = null, sfSort = { key: 'newest', dir: 'desc' };
  var SF_SORTS = [['name', 'Platform'], ['rows', 'Rows'], ['newest', 'Newest item'], ['share', 'Share of store']];
  /* per source table: platform, rows, newest item, share of the store, a freshness bar, and two honest flags computed by the server
     (stale = healthy sync but the database itself has nothing newer than 48 h; capped = the per-table row cap was reached) */
  function renderSourceFresh() {
    var box = $('#evp-srcfresh'); if (!box) return;
    var sf = statusData && statusData.source_freshness;
    if (!sf || !sf.rows || !sf.rows.length) { box.innerHTML = '<p class="note">' + (statusData ? 'No source tables are stored yet.' : 'Source freshness is unavailable: the status could not be loaded.') + '</p>'; return; }
    var now = nowMs(), nameOf = function (r) { return L.platform(r.platform); };
    var rows = P.freshSort(sf.rows, sfSort.key, sfSort.dir, nameOf), nStale = rows.filter(function (r) { return r.stale; }).length, nCap = rows.filter(function (r) { return r.capped; }).length;
    var h = '<p class="sf-lede">One row per source table. Each bar shows how recent the newest item is, across the last 30 days. ' +
      (sf.sync_healthy ? 'The sync is healthy, so &ldquo;No new rows at the source&rdquo; means the database itself has nothing newer than ' + esc(String(sf.stale_after_hours || 48)) + ' hours; it is not a sync fault.' : 'The sync is not healthy right now, so no feed is marked as having no new rows at the source.') +
      (nStale || nCap ? ' ' + esc(plural(nStale, 'feed', 'feeds')) + ' without new rows, ' + esc(plural(nCap, 'feed', 'feeds')) + ' capped.' : '') + '</p>';
    h += '<div class="sf-sort" role="group" aria-label="Sort the source list"><span class="lbl">Sort by</span>' + SF_SORTS.map(function (c) {
      var on = sfSort.key === c[0];
      return '<button type="button" class="vchip" data-sf="' + c[0] + '" aria-pressed="' + on + '">' + esc(c[1]) + (on ? '<span aria-hidden="true"> ' + (sfSort.dir === 'asc' ? '&uarr;' : '&darr;') + '</span><span class="sr"> (' + (sfSort.dir === 'asc' ? 'ascending' : 'descending') + ')</span>' : '') + '</button>';
    }).join('') + '</div><ul class="sf-list">';
    h += rows.map(function (r) {
      var frac = P.freshFraction(r.newest_published_at, now, 30), cls = r.stale ? ' stale' : r.capped ? ' capped' : '';
      var badge = '';
      if (r.stale) badge += '<span class="sf-badge stale" title="The sync is healthy and the database has no newer rows for this source table.">No new rows at the source since ' + esc(L.dateLong(r.newest_published_at)) + '</span>';
      if (r.capped) badge += '<span class="sf-badge capped" title="The local copy stores at most ' + esc(fmtN(r.cap)) + ' rows per source table.">Capped at ' + esc(fmtN(r.cap)) + ' rows: older posts in the window are not stored</span>';
      return '<li class="sf-row' + cls + '"><div class="sf-name"><b title="Source table: ' + esc(r.source_table) + '">' + esc(L.platform(r.platform)) + '</b><span class="sf-tbl">' + esc(r.source_table) + '</span></div>' +
        '<div class="sf-bar"><span class="bar-track" role="img" aria-label="Newest item ' + esc(r.newest_published_at ? ago(r.newest_published_at) : 'unknown') + '"><b style="width:' + (frac * 100).toFixed(1) + '%"></b></span></div>' +
        '<div class="sf-new"><span class="sf-v">' + (r.newest_published_at ? esc(ago(r.newest_published_at)) : 'No items') + '</span><small>' + (r.newest_published_at ? esc(L.dateTimeUTC(r.newest_published_at)) : '') + '</small></div>' +
        '<div class="sf-n"><span class="sf-v">' + esc(fmtN(r.rows)) + '</span><small>' + esc(r.share == null ? 'n/a' : (r.share * 100 < 0.1 && r.rows ? '<0.1' : (r.share * 100).toFixed(1)) + '% of store') + '</small></div>' +
        (badge ? '<div class="sf-flags">' + badge + '</div>' : '') + '</li>';
    }).join('') + '</ul><p class="ev-denom">Denominator: ' + esc(fmtN(statusData.total_rows)) + ' items stored across ' + esc(fmtN(rows.length)) + ' source tables. This card ignores the scope filters above: it describes the whole local copy.</p>';
    box.innerHTML = h;
  }
  function renderFresh(d) {
    var f = d.freshness || {}, st = statusData, errs = (st && st.recent_errors) || [], pol = st && st.policy;
    var rows = '<dt>Newest item</dt><dd>' + (f.newest_item_at ? esc(L.dateTimeUTC(f.newest_item_at)) + '<span class="sub">' + esc(dur(minutesSince(f.newest_item_at))) + ' old</span>' : 'No items in scope') + '</dd>' +
      '<dt>Last sync</dt><dd>' + (f.synced_at ? esc(L.dateTimeUTC(f.synced_at)) + '<span class="sub">' + esc(ago(f.synced_at)) + '</span>' : 'The cache has never been synced') + '</dd>' +
      '<dt>Lag</dt><dd>' + (f.lag_minutes != null ? esc(dur(minutesSince(f.newest_item_at))) + '<span class="sub">Age of the newest item in scope. Items newer than the last sync have not been copied yet.</span>' : 'n/a') + '</dd>' +
      '<dt>Sync errors</dt><dd>' + (st ? (errs.length ? '<span class="ev-err">' + esc(plural(errs.length, 'recent error', 'recent errors')) + '</span><span class="sub">Latest ' + esc(errs[0].ts ? ago(errs[0].ts) : '') + (errs[0].source_key ? ' in ' + esc(L.platform(String(errs[0].source_key).replace(/^news:/, '').replace(/_posts$/, ''))) : '') + ': ' + esc(oneLine(errs[0].error).slice(0, 140)) + '</span>' : 'None recent') : 'Status unavailable') + '</dd>';
    if (st) rows += '<dt>Items stored</dt><dd>' + esc(fmtN(st.total_rows)) + ' across ' + esc(fmtN((st.per_source || []).length)) + ' source tables</dd>';
    put('#evp-fresh', '<dl class="kvt">' + rows + '</dl><p class="ev-denom">Scope: ' + esc(scopeText()) + '. Lag and newest item describe this scope; the last sync is for the whole cache.' + (pol && pol.social_days ? ' The cache keeps the last ' + esc(fmtN(pol.social_days)) + ' days of social posts and ' + esc(fmtN(pol.news_days)) + ' of news.' : '') + '</p>');
  }

  function loadStats() {
    var my = ++ovSeq, q = new URLSearchParams();
    if (OV.kind) q.set('kind', OV.kind); if (OV.platform) q.set('platform', OV.platform); q.set('days', OV.days);
    el.ov.setAttribute('aria-busy', 'true');
    $('#ev-scope').textContent = 'Loading statistics for ' + scopeText() + '...';
    syncOvFilters();
    Promise.all([rt('/api/rt/stats?' + q.toString()), rt('/api/rt/status').catch(function () { return null; })]).then(function (r) {
      if (my !== ovSeq) return;
      ovData = r[0]; statusData = r[1];
      var g = Date.parse(ovData.generated_at); if (!isNaN(g)) clockOffset = g - Date.now();
      el.ov.removeAttribute('aria-busy');
      $('#ev-scope').textContent = fmtN(ovData.total) + ' items in scope: ' + scopeText() + '.';
      fillPlatformSelect(); renderOverview(ovData); loadSchedule();
    }).catch(function (e) {
      if (my !== ovSeq) return;
      el.ov.removeAttribute('aria-busy');
      if (!failure(e)) { $('#ev-scope').textContent = 'Statistics could not be loaded.'; put('#ev-kpis', '<div class="ov-empty" role="alert"><b>Statistics could not be loaded</b><span>' + esc(e.message || 'Unknown error') + '</span><button type="button" class="btn ghost" data-act="reload">Try again</button></div>'); }
    });
  }
  el.ov.addEventListener('click', function (e) { if (e.target.closest('[data-act="reload"]')) loadStats(); });

  /* =====================================================================================================================
     BROWSE: list | case file
     ===================================================================================================================== */
  var DEF = { q: '', kind: '', platform: '', analysed: '', sort: '', source: '', topic: '', lang: '', from: '', to: '' };
  var B = JSON.parse(JSON.stringify(DEF)), bl = { page: 0, total: null, capped: false, loading: false, seq: 0, error: null, done: false, limitHit: false, shown: 0 };
  var sel = null, caseSeq = 0, routeCount = 0, wasShown = false, listScroll = 0, bTimer = null, itemCache = {}, brBuilt = false, bEl = null, io = null, brDirty = false;
  var SORTS = [['', 'Best match'], ['recent', 'Newest first'], ['p_prop', 'Top propaganda signal'], ['p_fake', 'Top fake probability']];

  function buildBrowse() {
    if (brBuilt) return; brBuilt = true;
    el.br.innerHTML =
      '<div class="desk evb" id="evb" data-pane="list">' +
        '<div class="desk-list" id="evb-list" tabindex="-1">' +
          '<div class="dl-head" id="evb-head"><h2 class="sr" id="evb-h">Items</h2>' +
            '<div class="dl-line"><div class="dl-q" role="search" aria-label="Search the Evidence Base"><label class="sr" for="evb-q">Search titles and text</label>' +
              '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="10.5" cy="10.5" r="6"/><path d="m15 15 5.5 5.5"/></svg>' +
              '<input type="search" id="evb-q" placeholder="Search titles and text" autocomplete="off" maxlength="300"></div>' +
              '<button type="button" class="more-btn" id="evb-more-btn" aria-haspopup="dialog" aria-expanded="false" aria-controls="evb-more">More</button></div>' +
            '<div class="vchips" id="evb-kind" role="group" aria-label="Filter by kind">' +
              [['', 'All'], ['news', 'News'], ['social', 'Social']].map(function (c) { return '<button type="button" class="vchip" data-kind="' + c[0] + '" aria-pressed="false">' + esc(c[1]) + '</button>'; }).join('') + '</div>' +
            '<div class="dl-selects">' +
              '<label class="f"><span class="sr">Platform</span><select id="evb-platform"><option value="">All platforms</option></select></label>' +
              '<label class="f"><span class="sr">Analysed</span><select id="evb-analysed"><option value="">Any state</option><option value="yes">Analysed</option><option value="no">Not analysed yet</option></select></label>' +
              '<label class="f"><span class="sr">Sort</span><select id="evb-sort">' + SORTS.map(function (s) { return '<option value="' + s[0] + '">' + esc(s[1]) + '</option>'; }).join('') + '</select></label></div>' +
            '<div class="dl-active"><span class="dl-count" id="evb-count" role="status" aria-live="polite">Loading</span><span class="dl-fchips" id="evb-active"></span></div>' +
            '<div class="more-pop" id="evb-more" role="dialog" aria-label="More filters" hidden><h2>More filters</h2><div class="more-grid">' +
              '<div class="f f-wide ta" id="evb-source-wrap"><label for="evb-source-q"><span>Source (type to search)</span></label>' +
                '<input type="search" id="evb-source-q" role="combobox" aria-autocomplete="list" aria-expanded="false" aria-controls="evb-source-list" placeholder="All sources: type a name" autocomplete="off" maxlength="80">' +
                '<ul class="ta-list" id="evb-source-list" role="listbox" aria-label="Matching sources" hidden></ul><p class="ta-note" id="evb-source-note" role="status" aria-live="polite"></p></div>' +
              '<label class="f f-wide"><span>Topic (database classification)</span><select id="evb-topic"><option value="">All topics</option></select></label>' +
              '<label class="f"><span>Language</span><select id="evb-lang"><option value="">All languages</option></select></label><span></span>' +
              '<label class="f"><span>From</span><input type="date" id="evb-from"></label><label class="f"><span>To</span><input type="date" id="evb-to"></label></div>' +
              '<div class="more-foot"><button type="button" class="btn ghost" id="evb-reset">Reset all filters</button></div></div>' +
          '</div>' +
          '<ul class="rows" id="evb-rows" role="listbox" aria-label="Evidence Base items"></ul>' +
          '<div class="rows-empty" id="evb-empty" hidden></div><div class="rows-foot" id="evb-foot" hidden></div>' +
        '</div>' +
        '<section class="desk-detail" id="evb-detail" aria-label="Case file" tabindex="-1">' +
          '<div class="guide" id="evb-guide"><div class="eyebrow lbl">Case file</div><h2>How to read an Evidence Base item</h2>' +
            '<p class="guide-lede">Select an item in the list to open its case file here. Use <kbd>j</kbd> and <kbd>k</kbd> or the arrow keys to move through the list and Enter to open. Every field says where it comes from.</p>' +
            '<ol class="guide-steps"><li><b>From the database.</b> What the Realtime Database stores: platform, source, language, time, topic, engagement. Shown as it is.</li>' +
            '<li><b>Inferred by FNP.</b> What the FNP analysis estimated: the propaganda signal, techniques, countries and, for a few items, a verdict. Items FNP has not analysed say so.</li>' +
            '<li><b>Original text.</b> The stored excerpt, with technique spans highlighted when FNP found any.</li>' +
            '<li><b>Compare.</b> Near-duplicates in the realtime data and the nearest fact-checks in the Fake News Database.</li></ol>' +
            '<p class="note">FNP signals are a triage aid, not a verdict. Read the original before acting.</p></div>' +
          '<div id="evb-body" hidden></div></section>' +
      '</div><div class="sr" id="evb-live" aria-live="polite"></div>';
    bEl = { desk: $('#evb'), scroller: $('#evb-list'), rows: $('#evb-rows'), foot: $('#evb-foot'), count: $('#evb-count'), active: $('#evb-active'), q: $('#evb-q'), kind: $('#evb-kind'), platform: $('#evb-platform'), analysed: $('#evb-analysed'),
      sort: $('#evb-sort'), more: $('#evb-more-btn'), pop: $('#evb-more'), detail: $('#evb-detail'), guide: $('#evb-guide'), body: $('#evb-body'), live: $('#evb-live'), head: $('#evb-head'), empty: $('#evb-empty') };
    wireBrowse();
  }

  function params(page) { return P.browseQuery(B, PAGE, page); }
  function moreCount() { return ['source', 'topic', 'lang', 'from', 'to'].filter(function (k) { return B[k]; }).length; }
  function optLabel(selectEl, v) { var o = selectEl && Array.prototype.filter.call(selectEl.options, function (x) { return x.value === v; })[0]; return o ? o.textContent.replace(/\s*\([\d,]+\)$/, '') : v; }
  function fchip(key, text) { return '<button type="button" class="fchip" data-clear="' + key + '"><span>' + esc(text) + '</span><b aria-hidden="true">&times;</b><span class="sr">Remove filter</span></button>'; }
  function renderActive() {
    var h = '', add = function (k, t) { h += fchip(k, t); };
    if (B.q.trim()) add('q', 'Search: ' + B.q.trim());
    if (B.platform) add('platform', L.platform(B.platform));
    if (B.analysed) add('analysed', B.analysed === 'yes' ? 'Analysed' : 'Not analysed yet');
    if (B.source) add('source', sourceName(B.source));
    if (B.topic) add('topic', B.topic === 'unclassified' ? 'Unclassified' : optLabel($('#evb-topic'), B.topic));
    if (B.lang) add('lang', L.lang(B.lang));
    if (B.from) add('from', 'From ' + L.dateLong(B.from));
    if (B.to) add('to', 'To ' + L.dateLong(B.to));
    var changed = JSON.stringify(B) !== JSON.stringify(DEF);
    var tot = bl.total == null ? 'Loading' : (bl.total === 0 ? 'No items' : (bl.capped ? '10,000+' : fmtN(bl.total)) + (bl.total === 1 ? ' item' : ' items'));
    bEl.count.textContent = tot;
    bEl.count.title = bl.capped ? 'Matches are counted up to 10,000; narrow the search to see the exact number' : '';
    bEl.active.innerHTML = h + (changed ? '<button type="button" class="fchip clear" data-clear="all"><span>Reset filters</span></button>' : '');
    var fn = $('#evb-fn'); if (fn) { var nf = ['kind', 'platform', 'analysed', 'sort'].filter(function (k) { return B[k]; }).length; fn.textContent = nf; fn.hidden = !nf; }
    var nm = moreCount();
    bEl.more.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" aria-hidden="true"><path d="M4 7h10M18 7h2M4 17h2M10 17h10"/><circle cx="16" cy="7" r="2"/><circle cx="8" cy="17" r="2"/></svg>More' + (nm ? '<span class="n" aria-hidden="true">' + nm + '</span><span class="sr"> (' + nm + ' active)</span>' : '') + '<span class="sr"> filters</span>';
  }
  function renderKindChips() { Array.prototype.forEach.call(bEl.kind.querySelectorAll('.vchip'), function (b) { b.setAttribute('aria-pressed', b.getAttribute('data-kind') === B.kind); }); }
  function ensureOption(s, v, label) { if (v && !Array.prototype.some.call(s.options, function (o) { return o.value === v; })) s.insertAdjacentHTML('beforeend', '<option value="' + esc(v) + '">' + esc(label) + '</option>'); }
  function syncControls() {
    ensureOption($('#evb-topic'), B.topic, B.topic === 'unclassified' ? 'Unclassified' : B.topic);
    ensureOption($('#evb-lang'), B.lang, L.lang(B.lang)); ensureOption(bEl.platform, B.platform, L.platform(B.platform));
    bEl.q.value = B.q; bEl.platform.value = B.platform; bEl.analysed.value = B.analysed; bEl.sort.value = B.sort;
    $('#evb-source-q').value = B.source; $('#evb-topic').value = B.topic; $('#evb-lang').value = B.lang; $('#evb-from').value = B.from; $('#evb-to').value = B.to;
    renderKindChips(); renderActive();
  }
  function fillBrowseSelects() {
    if (!facets) return;
    var fill = function (id, first, items, val, label, count) {
      var s = $(id); s.innerHTML = '<option value="">' + esc(first) + '</option>' + items.map(function (x) { return '<option value="' + esc(val(x)) + '">' + esc(label(x)) + (count ? ' (' + esc(fmtN(count(x))) + ')' : '') + '</option>'; }).join('');
    };
    fill('#evb-platform', 'All platforms', facets.by_platform || [], function (x) { return x.platform; }, function (x) { return L.platform(x.platform); }, function (x) { return x.count; });
    var tp = facets.by_topic || { top: [], unclassified: { count: 0 } };
    fill('#evb-topic', 'All topics', tp.top.concat([{ topic_id: 'unclassified', name: 'Unclassified', count: tp.unclassified.count }]), function (x) { return x.topic_id; }, function (x) { return x.name; }, function (x) { return x.count; });
    fill('#evb-lang', 'All languages', (facets.by_language && facets.by_language.top) || [], function (x) { return x.lang; }, function (x) { return L.lang(x.lang); }, function (x) { return x.count; });
  }

  /* ---- list ---- */
  function mini(label, p, cls) {
    return '<span class="mini ' + cls + '" role="img" aria-label="' + esc(label) + ' ' + n2(p) + '" title="' + esc(label) + ' ' + n2(p) + '"><b style="width:' + (Math.max(0, Math.min(1, p)) * 100).toFixed(1) + '%"></b></span>';
  }
  var PARTIAL_NOTE = 'Partial = propaganda signals only, the verdict timed out. There is no verdict for this item yet; the scheduled analysis retries it on later runs.';
  function scoreHTML(inf) {
    if (!inf) return '<span class="ev-na">Not analysed yet</span>';
    var deep = inf.deep, both = inf.batch && deep, part = !deep && inf.partial;
    var h = part && !inf.batch ? '<span class="ev-an part" title="' + esc(PARTIAL_NOTE) + '">Analysed (partial: propaganda only)</span>' : deep ? '<span class="ev-an deep" title="A stored Claim Check (deep) verdict' + (both ? '; the batch cheap pass also has a record' : '') + '">Analysed (deep)</span>' : '<span class="ev-an" title="FNP batch cheap pass (export) record">Analysed (batch)</span>';
    if (part && inf.batch) h += '<span class="ev-an part" title="' + esc(PARTIAL_NOTE) + '">+ partial deep</span>';
    if (inf.batch && inf.batch.too_short && !deep) h += '<span class="ev-short" title="This item has too little text for the FNP encoder to rate. It gets no propaganda signal, which says nothing about whether it is true.">Too short to score</span>';
    else if (inf.p_prop != null) h += '<span class="ev-sc" title="Propaganda signal from the batch pass, uncalibrated"><span class="ev-sck">Prop.</span>' + mini('Propaganda signal', inf.p_prop, 'acc') + '<span class="ev-scn">' + n2(inf.p_prop) + '</span></span>';
    if (deep && deep.p_fake_misleading != null) h += '<span class="ev-sc" title="Deep verdict: P(fake) + P(misleading)"><span class="ev-sck">Fake/mis.</span>' + mini('Fake or misleading probability', deep.p_fake_misleading, F.band(deep.p_fake_misleading)) + '<span class="ev-scn">' + n2(deep.p_fake_misleading) + '</span></span>';
    else if (inf.p_fake != null) h += '<span class="ev-sc" title="Probability that the item is fake"><span class="ev-sck">Fake</span>' + mini('Fake probability', inf.p_fake, F.band(inf.p_fake)) + '<span class="ev-scn">' + n2(inf.p_fake) + '</span></span>';
    return h;
  }
  function rowHTML(it) {
    var inf = it.inferred, v = inf && inf.verdict ? inf.verdict : 'UNRATED', k = it.kind === 'news' ? 'news' : 'social';
    return '<li class="row s-' + esc(v) + '" role="option" data-id="' + esc(it.id) + '" tabindex="-1" aria-selected="' + (it.id === sel) + '"' + (it.id === sel ? ' aria-current="true"' : '') + '>' +
      '<span class="row-title">' + esc(itemTitle(it)) + '</span>' +
      '<span class="row-meta"><span class="ev-kind" title="' + esc(kindName(it.kind)) + '">' + ICON[k] + '<span class="sr">' + esc(kindName(it.kind)) + '. </span></span><span class="ev-plat" title="Source table: ' + esc(L.platformTable(it.platform)) + '">' + esc(L.platform(it.platform)) + '</span><span aria-hidden="true">&middot;</span>' +
      '<span class="row-src">' + esc(sourceName(it.source)) + '</span><span aria-hidden="true">&middot;</span><span>' + esc(L.lang(it.lang)) + '</span><span aria-hidden="true">&middot;</span><span title="' + esc(it.published_at ? L.dateTimeUTC(it.published_at) : '') + '">' + esc(it.published_at ? ago(it.published_at) : 'undated') + '</span></span>' +
      '<span class="row-score">' + scoreHTML(inf) + '</span></li>';
  }
  function skeletonRows() { var r = ''; for (var i = 0; i < 7; i++) r += '<li class="row sk" aria-hidden="true"><span class="skel w1"></span><span class="skel w2"></span><span class="skel w3"></span></li>'; return r; }
  function setFoot(html) { bEl.foot.innerHTML = html; bEl.foot.hidden = !html; }
  function footState() {
    if (bl.error) { setFoot('<p>Could not load more items.</p><button type="button" class="btn ghost" data-act="more-retry">Try again</button>'); return; }
    if (bl.total === 0) { setFoot(''); return; }
    if (bl.done) { setFoot(bl.limitHit ? 'Showing the first ' + fmtN(OFFSET_MAX) + ' results. Narrow the filters to reach the rest.' : 'End of list. ' + fmtN(bl.shown) + ' item' + (bl.shown === 1 ? '' : 's') + ' shown.'); return; }
    setFoot('<span class="spin" aria-hidden="true"></span>Loading more items');
  }
  function fetchPage(page, replace, seq) {
    bl.loading = true; bl.error = null; bEl.rows.setAttribute('aria-busy', 'true');
    if (!bEl.rows.children.length) bEl.rows.innerHTML = skeletonRows();
    footState();
    rt('/api/rt/items?' + params(page)).then(function (d) {
      if (seq !== bl.seq) return;
      bl.page = d.page; bl.total = d.total; bl.capped = !!d.total_capped; bl.loading = false;
      if (replace) { bEl.rows.innerHTML = ''; bEl.scroller.scrollTop = 0; bl.shown = 0; }
      bl.shown += d.items.length;
      bl.limitHit = d.page * d.size > OFFSET_MAX && d.page < d.pages;
      bl.done = d.page >= d.pages || bl.limitHit || !d.items.length;
      if (!d.items.length && replace) { bEl.rows.innerHTML = ''; bEl.foot.hidden = true; showEmpty(); }
      else { hideEmpty(); bEl.rows.insertAdjacentHTML('beforeend', d.items.map(rowHTML).join('')); if (replace) C.setCursor(bEl.rows, bEl.rows.querySelector('.row'), false); }
      bEl.rows.removeAttribute('aria-busy'); renderActive(); footState(); applyHeadPad(); watch();
    }).catch(function (e) {
      if (seq !== bl.seq) return;
      bl.loading = false; bl.error = e.message; bEl.rows.removeAttribute('aria-busy');
      if (failure(e)) return;
      if (replace) { bEl.rows.innerHTML = ''; showEmpty('The list could not be loaded', e.status === 400 ? (e.message || 'These filters are not valid.') : 'Check your connection, then try again.', true); setFoot(''); } else footState();
      renderActive();
    });
  }
  function showEmpty(title, text, retry) {
    bEl.empty.innerHTML = '<b>' + esc(title || 'No items match these filters') + '</b><span>' + esc(text || 'Try removing a filter or searching for fewer words.') + '</span><button type="button" class="btn ghost" ' + (retry ? 'data-act="more-retry-all"' : 'data-clear="all"') + '>' + (retry ? 'Try again' : 'Reset filters') + '</button>';
    bEl.empty.hidden = false;
  }
  function hideEmpty() { bEl.empty.hidden = true; }
  function reload() { bl.seq++; bl.page = 0; bl.total = null; bl.capped = false; bl.done = false; bl.limitHit = false; bl.error = null; renderActive(); fetchPage(1, true, bl.seq); }
  function schedule() { clearTimeout(bTimer); bTimer = setTimeout(reload, 140); }
  function changed() { renderKindChips(); renderActive(); schedule(); }
  function loadMore() { if (bl.loading || bl.error || bl.done || bl.total === 0) return; fetchPage(bl.page + 1, false, bl.seq); }
  function nearEnd() { return !bEl.foot.hidden && bEl.scroller.offsetParent && bEl.foot.getBoundingClientRect().top < bEl.scroller.getBoundingClientRect().bottom + 700; }
  function watch() {
    if (!('IntersectionObserver' in window)) { bEl.scroller.onscroll = function () { if (nearEnd()) loadMore(); }; }
    else if (!io) io = new IntersectionObserver(function (es) { if (es[0].isIntersecting) loadMore(); }, { root: bEl.scroller, rootMargin: '0px 0px 700px 0px' });
    if (io) { io.disconnect(); if (!bEl.foot.hidden && !bl.done) io.observe(bEl.foot); }
    requestAnimationFrame(function () { if (!bl.done && nearEnd()) loadMore(); });
  }
  function applyHeadPad() { if (bEl) bEl.scroller.style.scrollPaddingTop = bEl.head.offsetHeight + 'px'; }
  function markSelected() {
    C.rows(bEl.rows).forEach(function (r) { var on = r.getAttribute('data-id') === sel; r.setAttribute('aria-selected', on); if (on) r.setAttribute('aria-current', 'true'); else r.removeAttribute('aria-current'); });
  }
  function selectedRow() { return C.rows(bEl.rows).filter(function (r) { return r.getAttribute('data-id') === sel; })[0] || null; }

  /* ---- spans: technique highlights located by their text (the cache cleans text differently from FNP, so offsets are not trusted) ---- */
  var HUE = { 'loaded language': 1, 'exaggeration / minimisation': 1, 'appeal to fear-prejudice': 2, 'flag-waving': 2, 'name calling / labeling': 3, 'bandwagon / reductio ad hitlerum': 3, 'doubt': 4, 'appeal to authority': 4,
    'black-and-white fallacy': 5, 'causal oversimplification': 5, 'whataboutism / straw men / red herring': 6, 'thought-terminating cliches': 6, 'repetition': 7, 'slogans': 7 };
  function hue(name) { return HUE[String(name).toLowerCase()] || 8; }
  function highlight(text, spans) {
    var cps = Array.from(String(text == null ? '' : text)), flat = cps.join(''), found = [], unplaced = 0;
    (spans || []).forEach(function (s) {
      var t = s && typeof s.text === 'string' ? s.text : '', at = t ? flat.indexOf(t) : -1, st;
      /* a real FNP span carries its offsets: trust them when the stored text at that place is exactly the span text, else search for the text */
      if (s && t && isFinite(s.start) && isFinite(s.end) && s.start >= 0 && s.end <= cps.length && cps.slice(s.start, s.end).join('') === t) st = s.start;
      else if (at < 0) { unplaced++; return; }
      else st = Array.from(flat.slice(0, at)).length;
      var name = L.technique(s.technique);
      found.push({ s: st, e: st + Array.from(t).length, name: name, h: hue(name) });
    });
    var cuts = [0, cps.length]; found.forEach(function (x) { cuts.push(x.s, x.e); });
    cuts = cuts.filter(function (v, i, a) { return a.indexOf(v) === i; }).sort(function (a, b) { return a - b; });
    var out = '';
    for (var i = 0; i + 1 < cuts.length; i++) {
      var a = cuts[i], b = cuts[i + 1], txt = esc(cps.slice(a, b).join('')), cov = found.filter(function (x) { return x.s <= a && x.e >= b; });
      out += cov.length ? '<mark class="cc-mk k' + cov[0].h + (cov.length > 1 ? ' multi' : '') + '" title="' + esc(cov.map(function (x) { return x.name; }).join(', ')) + '">' + txt + '</mark>' : txt;
      found.forEach(function (x) { if (x.e === b && x.s < b) out += '<span class="cc-tag k' + x.h + '">' + esc(x.name) + '</span>'; });
    }
    return { html: out, unplaced: unplaced, placed: found.length };
  }

  /* ---- case file ---- */
  function fv(val, badge) { return '<span class="fv"><span class="fv-v">' + val + '</span>' + badge + '</span>'; }
  function openLink(url) {
    var u = F.safeUrl(url);
    return u ? '<a class="case-open" href="' + esc(u) + '" target="_blank" rel="noopener noreferrer">Open original<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M14 4h6v6M20 4l-9 9M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5"/></svg><span class="sr"> (opens in a new tab)</span></a>' : '';
  }
  function engagementHTML(e) {
    if (e == null || e === '') return 'Not recorded';
    if (typeof e === 'number') return esc(fmtN(e));
    if (typeof e === 'string' && /^-?\d+$/.test(e.trim())) return esc(fmtN(+e));
    if (typeof e === 'object') { var k = Object.keys(e).slice(0, 8); return k.length ? k.map(function (x) { return esc(L.titleCase(x)) + ' ' + esc(typeof e[x] === 'number' ? fmtN(e[x]) : String(e[x])); }).join(', ') : 'Not recorded'; }
    return esc(String(e).slice(0, 120));
  }
  var UNCL = '<span class="ev-uncl-t">Unclassified</span> ' + q('unclassified');
  function dbSection(it) {
    var rows = [];
    var add = function (label, val, sub) { rows.push([label, fv(val + (sub ? '<span class="sub">' + sub + '</span>' : ''), BADGE_DB)]); };
    add('Kind', esc(kindName(it.kind)));
    add('Platform', esc(L.platform(it.platform)), 'Source table: ' + esc(it.source_table || L.platformTable(it.platform)));
    add('Source', esc(sourceName(it.source)) + (it.source ? ' <button type="button" class="link-btn inline" data-act="more-source" data-v="' + esc(it.source) + '">More from this source</button>' : ''));
    add('Language', esc(L.lang(it.lang)), it.lang && it.lang !== 'und' ? 'Recorded by the database' : 'No language was recorded');
    add('Published', it.published_at ? esc(L.dateTimeUTC(it.published_at)) : 'Unknown', it.published_at ? esc(ago(it.published_at)) : '');
    add(term('database classification', 'Topic'), it.topic && it.topic.name ? esc(it.topic.name) : UNCL);
    add('Category', it.category && it.category.name ? esc(it.category.name) : UNCL);
    add('Subtopic', it.subtopic_id ? 'Assigned' : UNCL, it.subtopic_id ? 'Code ' + esc(it.subtopic_id) : '');
    add('Story cluster', it.cluster_id != null && it.cluster_id !== '' ? 'Cluster ' + esc(it.cluster_id) : 'Not in a cluster', 'Items the database grouped as one story.');
    add(term('near-duplicate', 'Duplicate flag'), it.is_duplicate == null ? 'Not recorded' : it.is_duplicate ? 'Marked as a duplicate' : 'Not marked as a duplicate', 'The database\'s own flag; the Compare section below counts near-duplicates by text.');
    add('Engagement', engagementHTML(it.engagement), it.kind === 'social' ? 'The platform\'s own count when the post was collected.' : '');
    add('Synced here', it.synced_at ? esc(L.dateTimeUTC(it.synced_at)) : 'Unknown');
    return C.section('evb-s-db', 'From the database', BADGE_DB + ' What the ' + term('from the database', 'connected database') + ' stores about this item, shown unchanged.', C.kv(rows));
  }
  function srcName(s) { return s === 'triage' ? 'Triage export' : s === 'realtime_export' ? 'Realtime export' : s ? L.titleCase(s) : 'Unknown'; }
  function techniquesHTML(list) {
    if (!list || !list.length) return '<p class="note">FNP found no propaganda technique in this item.</p>';
    return '<ul class="ev-techs">' + list.map(function (t) {
      var name = L.technique(t.technique), g = L.glossaryEntry(name);
      return '<li class="k' + hue(name) + '"><span class="ev-tn">' + (g ? term(name, name) : esc(name)) + '</span><span class="ev-tm">found ' + esc(plural(t.count || 1, 'time', 'times')) + (t.max_confidence != null ? ', top confidence ' + esc(n2(t.max_confidence)) : '') + '</span>' +
        '<span class="ev-td">' + (g ? esc(g.definition) : 'No plain-language definition is available for this technique yet.') + '</span></li>';
    }).join('') + '</ul>';
  }
  /* ---- the 14 technique buckets for a stored FNP record: highest span confidence per merged bucket, 0 when absent (the same span_max basis as Claim Check) ---- */
  var BUCKETS = ['Loaded_Language', 'Name_Calling,Labeling', 'Repetition', 'Exaggeration,Minimisation', 'Doubt', 'Appeal_to_fear-prejudice', 'Flag-Waving', 'Causal_Oversimplification', 'Slogans',
    'Appeal_to_Authority', 'Black-and-White_Fallacy', 'Thought-terminating_Cliches', 'Whataboutism,Straw_Men,Red_Herring', 'Bandwagon,Reductio_ad_hitlerum'];
  var BUCKET_OF = {};
  function tkey(n) { return String(n == null ? '' : n).toLowerCase().replace(/[^a-z0-9]/g, ''); }
  BUCKETS.forEach(function (b) { BUCKET_OF[tkey(b)] = b; });
  ['whataboutism', 'strawmen', 'redherring'].forEach(function (k) { BUCKET_OF[k] = BUCKETS[12]; });
  ['bandwagon', 'reductioadhitlerum'].forEach(function (k) { BUCKET_OF[k] = BUCKETS[13]; });
  var SPAN_NOTE = 'highest span confidence, not a calibrated probability';
  function techClasses(techs) {
    var p = {}; BUCKETS.forEach(function (b) { p[b] = 0; });
    (techs || []).forEach(function (t) {
      var b = t && BUCKET_OF[tkey(t.technique)], c = t && Number(t.max_confidence);
      if (b && isFinite(c)) p[b] = Math.max(p[b], Math.min(1, Math.max(0, c)));
    });
    return BUCKETS.map(function (b, i) { return { technique: b, probability: p[b], basis: 'span_max', i: i }; })
      .sort(function (a, c) { return c.probability - a.probability || a.i - c.i; });
  }
  var TOO_SHORT_NOTE = 'Too short to score: this item has too little text for the FNP encoder to rate it, so there is no propaganda signal and no technique bars. That says nothing about whether it is true.';
  /* the 14 bars of a real FNP record: the encoder's max token probability per technique, sorted high to low, basis note always visible */
  function encoderBars(tp) {
    var cls = tp && Array.isArray(tp.classes) ? tp.classes : [];
    if (!cls.length) return '';
    return '<p class="cs-sub-2" style="margin-top:0">Probability by technique</p><div class="probs cc-tprobs">' + cls.map(function (c) {
      var pr = Math.max(0, Math.min(1, Number(c.probability) || 0)), name = L.technique(c.technique);
      return '<div class="prob tq k' + hue(name) + (pr === 0 ? ' zero' : '') + '"><span>' + term(name, name) + '</span><span class="meter" role="img" aria-label="' + esc(name) + ' ' + (pr * 100).toFixed(0) + ' percent"><b style="width:' + (pr * 100).toFixed(1) + '%"></b></span><b>' + (pr * 100).toFixed(0) + '%</b></div>';
    }).join('') + '</div><p class="note ev-basis">' + esc(tp.basis_note || 'encoder score (max token probability): uncalibrated, not validated on social posts') + '.</p>';
  }
  function inferredBars(inf) {
    var cc = window.ClaimCheck, b = inf && inf.batch;
    if (b && b.too_short) return '<p class="note ev-short-note">' + esc(TOO_SHORT_NOTE) + '</p>';
    if (b && b.technique_probs && b.technique_probs.classes && b.technique_probs.classes.length) return encoderBars(b.technique_probs);
    return cc && cc.classBars && inf ? cc.classBars({ classes: techClasses(inf.techniques), basis_note: SPAN_NOTE }) : '';
  }
  function verdictBars(classes) {
    if (!Array.isArray(classes) || !classes.length) return '';
    return '<div class="ev-vb"><p class="cs-sub-2">Probability by verdict class</p><div class="probs">' + classes.map(function (c) {
      var pr = Number(c.probability) || 0, sec = !!c.secondary;
      return '<div class="prob v-' + esc(c.label) + (sec ? ' sec' : '') + '"><span>' + esc(L.verdict(c.label)) + (sec ? ' (secondary tag)' : '') + '</span><span class="meter" role="img" aria-label="' + esc(L.verdict(c.label)) + ' ' + (pr * 100).toFixed(0) + ' percent"><b style="width:' + (pr * 100).toFixed(1) + '%"></b></span><b>' + (pr * 100).toFixed(0) + '%</b></div>';
    }).join('') + '</div></div>';
  }
  /* the newest stored Claim Check result of the item that carries an engine verdict (the case file loads it with the item) */
  function deepResultOf(it) {
    var a = (it.analyses || []).filter(function (x) { return x && x.result && x.result.engine && x.result.engine.verdict; })[0];
    return a ? a.result : null;
  }
  function deepSection(it) {
    var inf = it.inferred, d = inf.deep, res = deepResultOf(it), rows = [], cc = window.ClaimCheck;
    var sub = BADGE_FNP + ' A ' + term('deep verdict', 'deep verdict') + ': the full Claim Check run on this item (a ' + term('scheduled analysis', 'scheduled run') + ' or a manual Workbench run). Estimates, uncalibrated, not findings.';
    var add = function (label, val) { rows.push([label, fv(val, BADGE_FNP)]); };
    add('Verdict', vchip(d.verdict) + '<span class="sub">The engine\'s fake-or-not call. Treat it as a lead, not a ruling.</span>');
    if (d.p_fake_misleading != null) add('Fake or misleading', '<span class="ev-big">' + esc(n2(d.p_fake_misleading)) + '</span> ' + mini('Fake or misleading probability', d.p_fake_misleading, F.band(d.p_fake_misleading)) + '<span class="sub">P(fake) ' + esc(n2(d.p_fake)) + ' + P(misleading) ' + esc(n2(d.p_misleading)) + '. This is the figure behind the likely fake or misleading share.</span>');
    add('Engine', esc(d.engine_version || 'Unknown') + '<span class="sub">Stored ' + esc(d.created_at ? L.dateTimeUTC(d.created_at) : 'at an unknown time') + '</span>');
    if (d.dup_of) add('Reused result', 'Copied from a near-duplicate<span class="sub">An item with almost the same text was analysed, so the engine was not run again for this one. <a href="#evidence/item/' + encodeURIComponent(d.dup_of) + '">Open that item</a>.</span>');
    var body = C.kv(rows);
    if (res && res.step1) body += verdictBars(res.step1.classes);
    var prop = res && res.step2 && res.step2.propaganda;
    if (prop && cc && cc.classBars) body += cc.classBars(prop);
    body += '<p class="cs-sub-2">Propaganda techniques found</p>' + techniquesHTML(inf.batch ? deepTechs(res) : inf.techniques);
    body += '<p class="ev-actions"><button type="button" class="btn ghost" data-act="wb-add" data-id="' + esc(it.id) + '">Open the stored Claim Check result</button></p>';
    return C.section('evb-s-deep', 'Inferred by FNP: deep verdict, run ' + (d.created_at ? ago(d.created_at) : 'at an unknown time'), sub, body);
  }
  /* a deep run whose verdict timed out: propaganda signals only, never presented as a verdict */
  function partialSection(it) {
    var inf = it.inferred, p = inf.partial, res = (it.analyses || []).filter(function (x) { return x && x.result && x.result.partial; })[0];
    res = res ? res.result : null;
    var sub = BADGE_FNP + ' A deep run that did not finish: <b>partial = propaganda signals only, the verdict timed out.</b> It is not counted in any verdict share.';
    var rows = [['Verdict', fv('No verdict yet<span class="sub">The deep verdict timed out. Nothing here says whether the item is true or false.</span>', BADGE_FNP)],
      ['Tries', fv(esc(fmtN((p.retries || 0) + 1)) + ' so far<span class="sub">' + ((p.retries || 0) >= (p.max_retries || 3) ? 'The scheduled analysis has stopped retrying this item. Run it by hand in the Workbench to try again.' : 'The scheduled analysis retries up to ' + esc(fmtN((p.max_retries || 3) - (p.retries || 0))) + ' more time' + (((p.max_retries || 3) - (p.retries || 0)) === 1 ? '' : 's') + ', one run at a time, after the fresh items.') + (p.last_try_at ? ' Last try ' + esc(ago(p.last_try_at)) + '.' : '') + '</span>', BADGE_FNP)]];
    var cc = window.ClaimCheck, prop = res && res.step2 && res.step2.propaganda;
    var body = C.kv(rows) + (prop && cc && cc.classBars ? cc.classBars(prop) : '') + '<p class="cs-sub-2">Propaganda techniques found</p>' + techniquesHTML(inf.batch ? deepTechs(res) : inf.techniques);
    body += '<p class="ev-actions"><button type="button" class="btn ghost" data-act="wb-add" data-id="' + esc(it.id) + '">Open the stored partial result</button></p>';
    return C.section('evb-s-partial', 'Inferred by FNP: partial deep analysis, ' + (p.created_at ? ago(p.created_at) : 'at an unknown time'), sub, body);
  }
  function deepTechs(res) {
    var t = res && res.step2 && res.step2.propaganda && res.step2.propaganda.techniques, agg = {}, out = [];
    (t || []).forEach(function (x) { if (!x || !x.technique) return; var a = agg[x.technique] || (agg[x.technique] = { technique: x.technique, count: 0, max_confidence: null }); a.count++; var c = Number(x.confidence); if (isFinite(c) && (a.max_confidence == null || c > a.max_confidence)) a.max_confidence = c; });
    Object.keys(agg).forEach(function (k) { out.push(agg[k]); });
    return out.sort(function (a, b) { return b.count - a.count; });
  }
  /* countries of a real FNP record: [{iso3, name, weight, methods, terms}]; every string is data and is escaped */
  function countriesHTML(list) {
    if (!list.length) return 'None found';
    return '<ul class="ev-ctry">' + list.map(function (c) {
      var nm = c.name || c.iso3, how = (c.terms || []).length ? 'Matched on ' + c.terms.map(function (t) { return '\u201c' + t + '\u201d'; }).join(', ') + ((c.methods || []).length ? ' (' + c.methods.join(', ') + ')' : '') : ((c.methods || []).length ? 'Matched by ' + c.methods.join(', ') : '');
      return '<li><span class="ev-cn">' + esc(nm) + '</span> <span class="ev-tm">' + esc(c.iso3) + (c.weight != null ? ', weight ' + esc(n2(c.weight)) : '') + '</span>' + (how ? '<span class="ev-td">' + esc(how) + '</span>' : '') + '</li>';
    }).join('') + '</ul><span class="sub">Countries the text names, by FNP\'s place matching. Not where the post was written.</span>';
  }
  /* FNP's nearest known fact-check (embedding similarity): a lookup, not a verdict. Desk link when the URL is in the Fake News Database, else the external link */
  var RERANK_FLOOR = 0.72;
  function factcheckCard(nf) {
    var v = /^(FAKE|MISLEADING|LEGIT|INCONCLUSIVE|PROPAGANDA)$/.test(nf.label || '') ? nf.label : 'UNRATED', sim = nf.similarity, weak = sim != null && sim < RERANK_FLOOR;
    var ext = F.safeUrl(nf.url);
    var link = nf.desk_id ? '<a class="btn ghost" href="#desk/' + encodeURIComponent(nf.desk_id) + '">Open in the Desk</a>'
      : ext ? '<a class="case-open" href="' + esc(ext) + '" target="_blank" rel="noopener noreferrer">Open the fact-check<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M14 4h6v6M20 4l-9 9M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5"/></svg><span class="sr"> (opens in a new tab)</span></a>' : '';
    return '<h4 class="cs-sub-2">Nearest fact-check (FNP)</h4><div class="ev-card ev-fnf s-' + esc(v) + '"><span class="ev-fnf-top">' + vchip(v) + '<span class="ev-sim">similarity ' + esc(n2(sim)) + (nf.rerank != null ? ', rerank ' + esc(n2(nf.rerank)) : '') + '</span></span>' +
      (nf.claim ? '<span class="ev-title">' + esc(nf.claim) + '</span>' : '<span class="ev-snip">No claim text was exported for this fact-check.</span>') +
      '<span class="ev-simbar" aria-hidden="true"><b style="width:' + (sim == null ? 0 : Math.max(0, Math.min(1, sim)) * 100).toFixed(0) + '%"></b></span>' +
      '<span class="ev-snip ev-fnf-note">' + (weak ? 'Weak match: similarity is below ' + RERANK_FLOOR.toFixed(2) + ', where FNP stops reranking. Shown for reference only; a similar topic is not the same claim. The label is the fact-check\'s own rating of that claim, not of this item.' : 'The label is the fact-check\'s own rating of that claim, not a verdict on this item. Check that it is the same claim.') + '</span>' +
      (link ? '<span class="ev-fnf-act">' + link + '</span>' : '') + '</div>';
  }
  function infSection(it) {
    var inf = it.inferred, sub = BADGE_FNP + ' Estimated by FNP\'s analysis, not stored in the database. These are triage signals, not findings.';
    if (!inf) {
      return C.section('evb-s-inf', 'Inferred by FNP', sub, '<div class="ev-notan"><b>Not analysed yet</b><p>FNP has not analysed this item, neither in the batch cheap pass nor with a deep verdict. That says nothing about whether it is true or false: it is simply unanalysed. The batch pass fills in as FNP exports it, and deep verdicts are added in bounded runs.</p>' +
        '<button type="button" class="btn" data-act="wb-add" data-id="' + esc(it.id) + '">' + (it.analyses && it.analyses.length ? 'Open the stored Claim Check result' : 'Analyse in the Workbench') + '</button></div>');
    }
    var out = '';
    if (inf.deep) out += deepSection(it);
    else if (inf.partial) out += partialSection(it);
    var b = inf.batch;
    if (!b) return out;
    var rows = [];
    var add = function (label, val) { rows.push([label, fv(val, BADGE_FNP)]); };
    add(term('uncalibrated signal', 'Propaganda signal (p_prop)'), b.too_short ? 'Too short to score<span class="sub">There is too little text for the encoder to rate this item, so it has no propaganda signal. That says nothing about whether it is true.</span>' : b.p_prop == null ? 'Not computed' : '<span class="ev-big">' + esc(n2(b.p_prop)) + '</span> ' + mini('Propaganda signal', b.p_prop, 'acc') + '<span class="sub">An <b>uncalibrated signal</b>: it ranks items, it does not say that this share of such items is propaganda.</span>');
    add('Verdict', b.verdict ? vchip(b.verdict) + '<span class="sub">Only items that closely match a known fact-check get a verdict. Treat it as a lead, not a ruling.' + (inf.deep ? ' The deep verdict above is the one counted in the totals.' : '') + '</span>' : 'No verdict<span class="sub">Only items that closely match a known fact-check get a verdict; this one did not, which says nothing about whether it is true.</span>');
    if (b.p_fake != null) add('Probability fake', '<span class="ev-big">' + esc(n2(b.p_fake)) + '</span> ' + mini('Fake probability', b.p_fake, F.band(b.p_fake)));
    if (b.p_misleading != null) add('Probability misleading', '<span class="ev-big">' + esc(n2(b.p_misleading)) + '</span> ' + mini('Misleading probability', b.p_misleading, F.band(b.p_misleading)));
    if (b.n_spans != null && !b.too_short) add('Flagged wording', b.n_spans ? esc(plural(b.n_spans, 'span', 'spans')) + (b.n_strong != null ? ', ' + esc(fmtN(b.n_strong)) + ' strong' : '') + '<span class="sub">Spans are stretches of wording the encoder marked; strong ones have a high token probability.' + (b.spans && b.spans.length && b.spans.length < b.n_spans ? ' The case file highlights the ' + esc(fmtN(b.spans.length)) + ' most confident.' : '') + '</span>' : 'None');
    var c = inf.countries;
    if (Array.isArray(c)) add('Countries mentioned', countriesHTML(c));
    else {
      add('Countries mentioned', c && c.about && c.about.length ? esc(c.about.map(L.country).join(', ')) : 'None found');
      if (c && c.origin) add('Posted from', esc(L.country(c.origin)));
    }
    var nf = inf.nearest_factcheck, nfCard = nf && nf.claim !== undefined ? factcheckCard(nf) : '';
    if (!nfCard) add('Nearest fact-check', nf ? 'A close match exists in FNP\'s fact-check set' + (nf.similarity != null ? ' (similarity ' + esc(n2(nf.similarity)) + ')' : '') + '<span class="sub">Reference ' + esc(nf.id) + '. Open the Compare section below for matches you can read.</span>' : 'None above the similarity floor');
    add('Model', esc(b.model_version || 'Unknown') + '<span class="sub">' + esc(srcName(inf.export_source)) + (b.built_at ? ', built ' + esc(L.dateTimeUTC(b.built_at)) : '') + '</span>');
    var body = C.kv(rows) + inferredBars(inf) + '<p class="cs-sub-2">Propaganda techniques found</p>' + (b.too_short ? '<p class="note">Not listed: too short to score.</p>' : techniquesHTML(inf.techniques)) + nfCard;
    if (!inf.deep) body += '<p class="ev-actions"><button type="button" class="btn ghost" data-act="wb-add" data-id="' + esc(it.id) + '">' + (it.analyses && it.analyses.length ? 'Open the stored Claim Check result' : 'Run Claim Check in the Workbench') + '</button></p>';
    return out + C.section('evb-s-inf', 'Inferred by FNP: batch cheap pass' + (b.built_at ? ', built ' + ago(b.built_at) : ''), BADGE_FNP + ' The ' + term('batch cheap pass', 'batch cheap pass') + ': the fast analysis FNP runs in bulk and exports, run by run. Estimates, not findings.', body);
  }
  function textSection(it) {
    var text = String(it.text_excerpt || ''), inf = it.inferred, spans = (inf && inf.spans) || [], body;
    var ts = spans.filter(function (s) { return s.field !== 'title'; }), tooShort = inf && inf.batch && inf.batch.too_short;
    if (tooShort) { ts = []; spans = []; }
    if (!text.trim()) body = '<p class="note">No text is stored for this item.</p>';
    else if (ts.length) {
      var h = highlight(text, ts);
      body = '<div class="cc-text" aria-label="Item text with technique highlights">' + h.html + '</div>' + (h.unplaced ? '<p class="note">' + esc(plural(h.unplaced, 'flagged phrase', 'flagged phrases')) + ' could not be found in the stored excerpt (FNP read the full text) and ' + (h.unplaced === 1 ? 'is' : 'are') + ' not highlighted.</p>' : '');
      body += '<p class="note">Highlights come from FNP. Each carries its technique name as text, not only a colour. The encoder reads word pieces, so a highlight can start or end inside a word.' + (inf && inf.batch && inf.batch.n_spans > ts.length ? ' FNP flagged ' + esc(fmtN(inf.batch.n_spans)) + ' spans in all; ' + esc(fmtN(ts.length)) + (ts.length === 1 ? ' is' : ' are') + ' highlighted here (the others are in the title, are below the 12 most confident, or could not be placed in the stored text).' : '') + '</p>';
    } else body = '<div class="excerpt">' + esc(text) + '</div>' + (tooShort ? '<p class="note">' + esc(TOO_SHORT_NOTE) + '</p>' : inf ? '<p class="note">FNP flagged no phrase in this text.</p>' : '');
    return C.section('evb-s-text', 'Original text', BADGE_DB + ' The stored excerpt (up to 1,500 characters), not the full article.' + (spans.length ? ' ' + BADGE_FNP + ' for the highlights.' : ''), body);
  }
  function compareSection(it) {
    var nd = it.near_duplicates || {}, fc = it.nearest_factchecks || {}, h = '<h4 class="cs-sub-2">Near-duplicates in the realtime data</h4>';
    if (nd.count == null) h += '<p class="note">' + esc(nd.reason ? 'Not compared: ' + nd.reason + '.' : 'Not compared.') + '</p>';
    else if (!nd.count) h += '<p>No other item has almost the same text.</p>';
    else h += '<p><b>' + esc(plural(nd.count, 'other item has', 'other items have')) + '</b> almost the same text (' + term('near-duplicate', 'near-duplicate') + ', within ' + esc(nd.max_hamming) + ' bits).' + (nd.sample.length < nd.count ? ' Showing ' + esc(nd.sample.length) + '.' : '') + '</p>' +
      '<ul class="ev-dups">' + nd.sample.map(function (s) { return '<li><a href="#evidence/item/' + encodeURIComponent(s.id) + '"><span class="ev-dt">' + esc(oneLine(s.title) || 'Open item') + '</span><span class="ev-dm">' + esc(sourceName(s.source)) + (s.published_at ? ' - ' + esc(ago(s.published_at)) : '') + '</span></a></li>'; }).join('') + '</ul>';
    h += '<h4 class="cs-sub-2">Nearest fact-checks in the Fake News Database</h4>';
    if (fc.available === false) h += '<p class="note">The Fake News Database could not be searched just now. Reload the case file to try again.</p>';
    else if (!fc.matches || !fc.matches.length) h += '<p>' + esc(fc.message || (fc.state === 'no_text' ? 'There is too little text to compare.' : 'No fact-check in the Fake News Database is close enough to this item.')) + '</p><p class="note">No close match does not mean the item is true.</p>';
    else h += '<p class="note">' + esc(fc.message || 'Ranked by similarity. Select one to open its case file in the Desk.') + '</p><div class="evs">' + fc.matches.map(function (m) { return C.evCard(m, { snippet: true }); }).join('') + '</div>';
    return C.section('evb-s-cmp', 'Compare', BADGE_DB + ' ' + BADGE_FN + ' Similar items in this database, and similar claims that have already been fact-checked.', h);
  }
  function caseHTML(it) {
    var inf = it.inferred, t = itemTitle(it), tspans = inf && inf.spans ? inf.spans.filter(function (s) { return s.field === 'title'; }) : [];
    if (!oneLine(it.title) && t.length > 160) t = t.slice(0, 159).trim() + '\u2026';
    var titleHTML = tspans.length && oneLine(it.title) ? highlight(oneLine(it.title), tspans).html : esc(t);
    var left = '<span class="chip tag">' + esc(kindName(it.kind)) + '</span>' + (inf ? '<span class="ob ob-fnp solid">' + (inf.deep ? 'Analysed (deep)' : inf.batch ? 'Analysed (batch)' : 'Analysed (partial: propaganda only)') + '</span>' + (inf.partial && inf.batch ? '<span class="ob ob-fnp">+ partial deep</span>' : '') + (inf.batch && inf.deep ? '<span class="ob ob-fnp">+ batch</span>' : '') + (inf.verdict ? vchip(inf.verdict) : '') : '<span class="ob ob-db solid">Not analysed yet</span>');
    var h = '<article class="case evcase" aria-labelledby="evb-title">' + C.bar(left, openLink(it.url), 'Back to the list') + '<div class="case-in">';
    h += '<div class="case-hero ev-hero"><div class="case-id"><p class="case-kicker">' + esc([L.platform(it.platform), sourceName(it.source), L.lang(it.lang), it.published_at ? ago(it.published_at) : 'undated'].join(' - ')) + '</p>' +
      '<h2 class="case-title" id="evb-title" tabindex="-1">' + titleHTML + '</h2>' + (oneLine(it.title) && it.text_excerpt ? '<p class="case-claim clamp">' + esc(oneLine(it.text_excerpt)) + '</p>' : '') + '</div></div>';
    h += dbSection(it) + infSection(it) + textSection(it) + compareSection(it);
    return h + '</div></article>';
  }
  function pane(n) { bEl.desk.setAttribute('data-pane', n); }
  function announce(t) { bEl.live.textContent = ''; setTimeout(function () { bEl.live.textContent = t; }, 30); }
  function showBody(html) { bEl.guide.hidden = true; bEl.body.hidden = false; bEl.body.innerHTML = html; bEl.detail.scrollTop = 0; }
  function clearCase() { caseSeq++; if (!MOBILE.matches) sel = null; markSelected(); bEl.body.hidden = true; bEl.body.innerHTML = ''; bEl.guide.hidden = false; pane('list'); }
  function openCase(id, focusTitle) {
    sel = id; markSelected(); pane('case');
    var my = ++caseSeq;
    var done = function (it) {
      if (my !== caseSeq) return;
      showBody(caseHTML(it)); Shell.setTitle(itemTitle(it)); announce('Case file opened: ' + itemTitle(it));
      if (focusTitle && MOBILE.matches) { var t = $('#evb-title'); if (t) t.focus({ preventScroll: true }); }
    };
    if (itemCache[id]) { done(itemCache[id]); return; }
    showBody(C.skeleton()); bEl.detail.setAttribute('aria-busy', 'true');
    rt('/api/rt/item/' + encodeURIComponent(id)).then(function (it) { itemCache[id] = it; var ks = Object.keys(itemCache); if (ks.length > 30) delete itemCache[ks[0]]; bEl.detail.removeAttribute('aria-busy'); done(it); })
      .catch(function (e) {
        bEl.detail.removeAttribute('aria-busy');
        if (my !== caseSeq) return;
        if (failure(e)) return;
        showBody(C.error(e.status === 404 ? 'This item is not in the Evidence Base copy (it may have aged out of the window).' : 'Check your connection, then try again.', id));
      });
  }
  function goBack() { if (routeCount > 1 && history.length > 1) history.back(); else Shell.navigate('#evidence/browse', true); }
  function openRow(row) { Shell.navigate('#evidence/item/' + encodeURIComponent(row.getAttribute('data-id'))); }

  function popOpen(open) { bEl.pop.hidden = !open; bEl.more.setAttribute('aria-expanded', open ? 'true' : 'false'); if (open) { var f = $('#evb-source-q'); if (f) f.focus(); } }
  function resetAll() { B = JSON.parse(JSON.stringify(DEF)); syncControls(); schedule(); }
  /* ---- source type-ahead: GET /api/rt/sources?q=&kind=&platform=&limit=20 (distinct sources with counts); the choice sets B.source ---- */
  var taSeq = 0, taTimer = null, taItems = [], taIdx = -1;
  function taClose() { var l = $('#evb-source-list'), i = $('#evb-source-q'); if (l) l.hidden = true; if (i) { i.setAttribute('aria-expanded', 'false'); i.removeAttribute('aria-activedescendant'); } taIdx = -1; }
  function taPaint() {
    var l = $('#evb-source-list'), i = $('#evb-source-q');
    l.innerHTML = taItems.length ? taItems.map(function (x, n) { return '<li role="option" id="evb-so-' + n + '" data-v="' + esc(x.source) + '" aria-selected="' + (n === taIdx) + '"><span class="ta-n">' + esc(sourceName(x.source)) + '</span><small>' + esc(fmtN(x.count)) + '</small></li>'; }).join('') : '<li class="ta-none" aria-disabled="true">No source matches</li>';
    l.hidden = false; i.setAttribute('aria-expanded', 'true');
    if (taIdx >= 0) { i.setAttribute('aria-activedescendant', 'evb-so-' + taIdx); var on = $('#evb-so-' + taIdx); if (on && on.scrollIntoView) on.scrollIntoView({ block: 'nearest' }); } else i.removeAttribute('aria-activedescendant');
  }
  function taFetch() {
    var my = ++taSeq, txt = $('#evb-source-q').value.trim();
    if (txt === B.source) txt = '';
    rt('/api/rt/sources?' + P.sourcesQuery(txt, B.kind, B.platform, 20)).then(function (d) {
      if (my !== taSeq) return;
      taItems = (d.sources || []).filter(function (x) { return x.source !== '(unknown)'; }); taIdx = -1; taPaint();
      $('#evb-source-note').textContent = txt ? '' : 'Showing the 20 largest sources; type to search all of them.';
    }).catch(function (e) { if (my !== taSeq) return; if (!failure(e)) { taItems = []; taClose(); $('#evb-source-note').textContent = 'Sources could not be loaded.'; } });
  }
  function taPick(v) { B.source = v || ''; $('#evb-source-q').value = B.source; taClose(); $('#evb-source-note').textContent = ''; changed(); }
  function wireSourceAhead() {
    var inp = $('#evb-source-q'), list = $('#evb-source-list');
    inp.addEventListener('focus', function () { clearTimeout(taTimer); taTimer = setTimeout(taFetch, 60); });
    inp.addEventListener('input', function () { clearTimeout(taTimer); if (!inp.value.trim() && B.source) { B.source = ''; changed(); } taTimer = setTimeout(taFetch, 220); });
    inp.addEventListener('keydown', function (e) {
      if (e.key === 'ArrowDown' || e.key === 'ArrowUp') { e.preventDefault(); if (list.hidden) { taFetch(); return; } if (taItems.length) { taIdx = (taIdx + (e.key === 'ArrowDown' ? 1 : -1) + taItems.length) % taItems.length; taPaint(); } }
      else if (e.key === 'Enter') { e.preventDefault(); if (taIdx >= 0 && taItems[taIdx]) taPick(taItems[taIdx].source); else if (taItems.length === 1) taPick(taItems[0].source); }
      else if (e.key === 'Escape' && !list.hidden) { e.preventDefault(); e.stopPropagation(); taClose(); }
    });
    list.addEventListener('mousedown', function (e) { e.preventDefault(); });
    list.addEventListener('click', function (e) { var li = e.target.closest('li[data-v]'); if (li) taPick(li.getAttribute('data-v')); });
    inp.addEventListener('blur', function () { setTimeout(taClose, 120); });
  }
  /* phones: search stays visible, the kind chips and the three selects sit behind a Filters button */
  function wirePhoneFilters() {
    var line = $('.dl-line', bEl.head), btn = document.createElement('button');
    btn.type = 'button'; btn.className = 'more-btn evb-fbtn'; btn.id = 'evb-fbtn'; btn.setAttribute('aria-expanded', 'false'); btn.setAttribute('aria-controls', 'evb-fold');
    btn.innerHTML = 'Filters<span class="n" id="evb-fn" hidden></span>';
    line.insertBefore(btn, bEl.more);
    var fold = document.createElement('div'); fold.className = 'evb-fold'; fold.id = 'evb-fold';
    var sel = $('.dl-selects', bEl.head); bEl.head.insertBefore(fold, bEl.kind); fold.appendChild(bEl.kind); fold.appendChild(sel);
    btn.addEventListener('click', function () { var on = !bEl.head.classList.contains('fold-open'); bEl.head.classList.toggle('fold-open', on); btn.setAttribute('aria-expanded', on ? 'true' : 'false'); applyHeadPad(); });
  }
  function wireBrowse() {
    bEl.detail.addEventListener('click', function (e) {
      if (C.onDetailClick(e)) return;
      var b = e.target.closest('[data-act]'); if (!b) return;
      var a = b.getAttribute('data-act');
      if (a === 'back') goBack();
      else if (a === 'retry') openCase(b.getAttribute('data-id'), false);
      else if (a === 'wb-add') { if (window.Workbench) window.Workbench.analyse(b.getAttribute('data-id')); }
      else if (a === 'more-source') { B = JSON.parse(JSON.stringify(DEF)); B.source = b.getAttribute('data-v'); syncControls(); schedule(); pane('list'); if (MOBILE.matches) { Shell.navigate('#evidence/browse'); } }
    });
    bEl.rows.addEventListener('click', function (e) { var r = e.target.closest('.row[data-id]'); if (r) { C.setCursor(bEl.rows, r, false); openRow(r); } });
    C.rowNav(bEl.rows, openRow);
    bEl.kind.addEventListener('click', function (e) { var b = e.target.closest('.vchip'); if (!b) return; B.kind = b.getAttribute('data-kind'); changed(); });
    var qt = null;
    bEl.q.addEventListener('input', function () { B.q = bEl.q.value; clearTimeout(qt); qt = setTimeout(changed, 260); });
    bEl.q.addEventListener('keydown', function (e) { if (e.key === 'Escape' && bEl.q.value) { bEl.q.value = ''; B.q = ''; changed(); e.stopPropagation(); } });
    [['platform', bEl.platform], ['analysed', bEl.analysed], ['sort', bEl.sort]].forEach(function (p) { p[1].addEventListener('change', function () { B[p[0]] = p[1].value; changed(); }); });
    bEl.pop.addEventListener('input', function (e) { var id = e.target.id; if (!id || id === 'evb-source-q') return; B[id.replace('evb-', '')] = e.target.value; changed(); });
    wireSourceAhead();
    wirePhoneFilters();
    $('#evb-reset').addEventListener('click', resetAll);
    bEl.active.addEventListener('click', function (e) {
      var b = e.target.closest('[data-clear]'); if (!b) return;
      var k = b.getAttribute('data-clear'); if (k === 'all') { resetAll(); return; }
      B[k] = ''; syncControls(); schedule();
    });
    bEl.empty.addEventListener('click', function (e) {
      var b = e.target.closest('button'); if (!b) return;
      if (b.getAttribute('data-act') === 'more-retry-all') reload(); else if (b.getAttribute('data-clear') === 'all') resetAll();
    });
    bEl.foot.addEventListener('click', function (e) { if (e.target.closest('[data-act="more-retry"]')) { bl.error = null; loadMore(); } });
    bEl.more.addEventListener('click', function () { popOpen(bEl.pop.hidden); });
    document.addEventListener('click', function (e) { if (bEl && !bEl.pop.hidden && !e.target.closest('#evb-more') && !e.target.closest('#evb-more-btn')) popOpen(false); });
    bEl.pop.addEventListener('focusout', function (e) { var n = e.relatedTarget; if (n && !bEl.pop.contains(n) && n !== bEl.more) popOpen(false); });
    window.addEventListener('resize', applyHeadPad);
  }

  /* keyboard on the Browse tab: j and k from anywhere outside form fields; Escape on small screens returns to the list */
  document.addEventListener('keydown', function (e) {
    if (Shell.current().route !== 'evidence' || tab !== 'browse' || !bEl) return;
    if (e.key === 'Escape') {
      if (!bEl.pop.hidden) { popOpen(false); bEl.more.focus(); e.stopPropagation(); return; }
      if (MOBILE.matches && sel) { e.preventDefault(); goBack(); }
      return;
    }
    if ((e.key !== 'j' && e.key !== 'k') || e.ctrlKey || e.metaKey || e.altKey) return;
    var t = e.target, tag = t && t.tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || (t && t.isContentEditable) || (t && t.closest && t.closest('.desk-detail, #evb-more'))) return;
    if (t && t.closest && t.closest('#evb-rows')) return;
    e.preventDefault();
    var from = selectedRow() || C.rows(bEl.rows)[0];
    if (from) { if (!selectedRow()) C.setCursor(bEl.rows, from, true); else C.move(bEl.rows, from, e.key === 'j' ? 1 : -1); }
  });
  Shell.on(function (r) {
    routeCount++;
    if (r.route === 'evidence' && bEl && /^browse$/.test(r.arg) && wasShown && MOBILE.matches) { var row = selectedRow(); if (row) setTimeout(function () { row.focus({ preventScroll: true }); }, 0); }
    wasShown = r.route === 'evidence' && /^item\//.test(r.arg);
  });

  function browseWith(f) {
    B = JSON.parse(JSON.stringify(DEF)); Object.keys(f).forEach(function (k) { if (k in B) B[k] = f[k]; });
    brDirty = true; Shell.navigate('#evidence/browse');
  }

  /* =====================================================================================================================
     ROUTE
     ===================================================================================================================== */
  var currentArg = '';
  function setTab(t) {
    tab = t; root.setAttribute('data-evtab', t);
    [['overview', el.tabOv], ['browse', el.tabBr], ['workbench', el.tabWb], ['insights', el.tabIns]].forEach(function (p) { if (p[0] === t) p[1].setAttribute('aria-current', 'page'); else p[1].removeAttribute('aria-current'); });
    el.ov.hidden = t !== 'overview'; el.br.hidden = t !== 'browse'; el.wb.hidden = t !== 'workbench'; el.ins.hidden = t !== 'insights';
  }
  function replaceHash(h) { try { history.replaceState(null, '', h); } catch (e) { location.hash = h; } }
  function enter(arg, changedRoute) {
    currentArg = arg || '';
    var itemId = null, t = 'overview';
    if (!arg) replaceHash('#evidence/overview');
    else if (arg === 'browse') t = 'browse';
    else if (arg === 'overview') t = 'overview';
    else if (arg === 'workbench') t = 'workbench';
    else if (arg === 'insights') t = 'insights';
    else if (/^item\/./.test(arg)) { t = 'browse'; itemId = arg.slice(5); }
    else replaceHash('#evidence/overview');
    setTab(t); clearNotice();
    Shell.setTitle(t === 'browse' ? 'Browse' : t === 'workbench' ? 'Workbench' : t === 'insights' ? 'Insights' : 'Overview');
    Object.keys(ext).forEach(function (k) { if (k !== t && ext[k].leave) ext[k].leave(); });
    var gate = checkGate(false);
    gate.then(function (g) {
      if (!g.open) { notice('gate'); setTab(tab); el.ov.hidden = true; el.br.hidden = true; el.wb.hidden = true; el.ins.hidden = true; return null; }
      return loadFacets(false).then(function () {
        if (ext[tab]) { if (tab === t) ext[tab].enter({ arg: arg }); }
        else if (tab === 'overview') { buildOverview(); fillPlatformSelect(); if (!started.ov) { started.ov = true; loadStats(); } else if (!ovData) loadStats(); else { drawCharts(ovData); } }
        else {
          buildBrowse(); fillBrowseSelects();
          if (!started.br || brDirty) { started.br = true; brDirty = false; syncControls(); reload(); applyHeadPad(); watch(); }
          else { if (listScroll) bEl.scroller.scrollTop = listScroll; applyHeadPad(); markSelected(); }
          if (itemId) openCase(itemId, true); else clearCase();
        }
        startTick();
      });
    }).catch(function (e) { if (!failure(e)) notice('error', e); });
  }
  Shell.register('evidence', {
    enter: function (arg, changedRoute) { enter(arg, changedRoute); },
    leave: function () { if (bEl) { listScroll = bEl.scroller.scrollTop; popOpen(false); } stopTick(); root.removeAttribute('data-evtab'); Object.keys(ext).forEach(function (k) { if (ext[k].leave) ext[k].leave(); }); }
  });

  /* the other views link here; the unit-style check uses _highlight to prove hostile text stays inert */
  function getItem(id) {
    if (itemCache[id]) return Promise.resolve(itemCache[id]);
    return rt('/api/rt/item/' + encodeURIComponent(id)).then(function (it) { itemCache[id] = it; var ks = Object.keys(itemCache); if (ks.length > 60) delete itemCache[ks[0]]; return it; });
  }
  function scopes() { return { overview: { kind: OV.kind, platform: OV.platform, days: OV.days }, browse: JSON.parse(JSON.stringify(B)) }; }
  window.Evidence = {
    browse: browseWith, ago: ago, _highlight: highlight, _caseHTML: function (it) { return caseHTML(it); },
    register: function (name, h) { ext[name] = h; }, rt: rt, scopes: scopes, getItem: getItem,
    /* helpers for the Workbench and Insights tabs: one implementation of badges, rows and failure states */
    h: {
      rt: rt, failure: failure, notice: notice, clearNotice: clearNotice, loadFacets: loadFacets, facets: function () { return facets; }, setClock: function (iso) { var g = Date.parse(iso); if (!isNaN(g)) clockOffset = g - Date.now(); },
      nowMs: nowMs, ago: ago, dur: dur, pct: pct, n2: n2, plural: plural, put: put, esc: esc, kindName: kindName, sourceName: sourceName, oneLine: oneLine, itemTitle: itemTitle,
      BADGE_DB: BADGE_DB, BADGE_FNP: BADGE_FNP, BADGE_FN: BADGE_FN, ICON: ICON, svgIco: svgIco, q: q, vchip: vchip, mini: mini, fv: fv, openLink: openLink, highlight: highlight, srcChip: srcChip,
      techniquesHTML: techniquesHTML, dbSection: dbSection, hue: hue, optLabel: optLabel, compareSection: compareSection, inferredBars: inferredBars, techClasses: techClasses, scoreHTML: scoreHTML, itemUrl: openLink
    }
  };
})();
