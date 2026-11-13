/* Triage: a ranked "what to read first" list from two sources (route #triage, deep link #triage/<source>/<id>).
   Sources: the FNP triage export (fnp) and the Realtime Database queue (realtime, ranked by FNP's batch signals and deep verdicts).
   Data comes only from /api/rt/gate and the four read-only routes /api/triage/{meta,items,facets,item/<id>} (same gate and rate limit as the
   Evidence Base). Nothing here runs an engine call: "Analyse in the Workbench" only adds the item to the Workbench selection.
   Everything in an item (text, author, claims, rationale, technique and span text) is untrusted data: it reaches the page only through esc().
   Tiers are a ranking aid and are never presented as verdicts; while no item has been validated by a person the page says so. */
(function () {
  'use strict';
  var $ = function (s, r) { return (r || document).querySelector(s); };
  var L = window.Labels, F = window.FNPUI, C = window.Case, Ev = window.Evidence, H = Ev && Ev.h, esc = L.esc, fmtN = F.fmtN, term = L.term, root = document.documentElement;
  if (!H) return;
  var PAGE = 40, OFFSET_MAX = 10000, MOBILE = window.matchMedia('(max-width: 959px)');
  var SRC_NAME = { fnp: 'FNP triage export', realtime: 'Realtime Database queue' };
  var TIER_NAME = { high: 'High', medium: 'Medium', low: 'Low' };
  var TIER_METER = { high: 'hi', medium: 'mid', low: 'lo' };
  var RULE_TEXT = { rerank_match: 'Matches a known fact-check', strong_propaganda: 'Strong propaganda wording', deep_demoted: 'Lowered by its deep verdict', p_prop: 'Propaganda signal', cosine: 'Similar to a known fact-check', deep: 'Deep verdict', below: 'Below every line' };
  var WINDOWS = [['', 'Any time'], ['1', 'Last 24 hours'], ['7', 'Last 7 days'], ['30', 'Last 30 days'], ['90', 'Last 90 days']];
  var SORTS = [['priority', 'Priority'], ['recent', 'Newest first']];
  var SPAN_NOTE = 'highest span confidence for that technique (FNP’s span encoder): uncalibrated, not validated on social posts';
  var ICON_FUNNEL = '<path d="M4 5h16l-6.2 7.4V19l-3.6-1.8v-4.8L4 5Z"/>';

  /* ---------- helpers ---------- */
  function n2(p) { return p == null || isNaN(p) ? 'n/a' : Number(p).toFixed(2); }
  function mini(label, p, cls) { return H.mini(label, p, cls); }
  function rt(path) { return H.rt(path); }
  function put(el, html) { if (el) el.innerHTML = html; return el; }
  function oneLine(s) { return String(s == null ? '' : s).replace(/\s+/g, ' ').trim(); }
  function reachText(n) { return n == null ? 'reach not recorded' : (n >= 10000 ? (Math.round(n / 100) / 10) + 'k' : fmtN(n)) + ' reach'; }
  function tierTag(t) { return '<span class="tr-tier t-' + esc(t) + '">' + esc(TIER_NAME[t] || 'Unranked') + '</span>'; }
  function vchip(v) { return H.vchip(v); }
  function srcLabel(s) { return s === 'realtime' ? 'Realtime Database' : 'FNP triage export'; }
  function when(iso) { return iso ? L.dateTimeUTC(iso) : 'unknown'; }
  function fcLabel(l) { return /^(FAKE|MISLEADING|LEGIT|INCONCLUSIVE|PROPAGANDA)$/.test(l || '') ? l : 'UNRATED'; }

  /* ---------- state ---------- */
  var DEF = { q: '', tier: '', platform: '', lang: '', country: '', technique: '', days: '', sort: 'priority' };
  var ST = { fnp: copy(DEF), realtime: copy(DEF) };
  var cur = 'fnp', sel = null, selSrc = null, meta = null, metaAt = 0, metaP = null, facets = { fnp: null, realtime: null }, facetsAt = { fnp: 0, realtime: 0 };
  var built = false, bEl = null, el = {}, bl = { page: 0, total: null, tiers: null, loading: false, seq: 0, error: null, done: false, limitHit: false, shown: 0, available: true, reason: null };
  var caseSeq = 0, itemCache = {}, tm = null, io = null, gateP = null, routeCount = 0, wasCase = false;
  function copy(o) { return JSON.parse(JSON.stringify(o)); }
  function F_() { return ST[cur]; }

  /* ---------- gate and failure states (the same calm explanation as the Evidence Base) ---------- */
  function checkGate(force) {
    if (gateP && !force) return gateP;
    gateP = rt('/api/rt/gate').then(function (g) { return { open: !!(g && (g.open || g.auth)), auth: !!(g && g.auth) }; });
    gateP.catch(function () { gateP = null; });
    return gateP;
  }
  function notice(kind, o) {
    var h;
    if (kind === 'gate') {
      h = '<div class="ev-card-note" role="region" aria-labelledby="tr-gate-h"><span class="ev-card-ico" aria-hidden="true">' + H.svgIco('check') + '</span><div>' +
        '<h2 id="tr-gate-h">Triage needs a login</h2>' +
        '<p>Triage ranks the live posts and articles of the Realtime Database and the FNP triage export, so it is only available behind a login. Ask the administrator to turn the login on for this site, then reload this page.</p>' +
        '<p class="note">Nothing from the Realtime Database or the triage export is shown or sent to your browser until then. The Fake News Database (Overview, Desk, Claim Check) is not affected.</p>' +
        '<details class="ev-admin"><summary>For administrators</summary><p>Set <code>FNP_AUTH_USER</code> and <code>FNP_AUTH_PASS</code> in the server\'s environment and restart the server. For a deliberate, temporary test on a private machine only, <code>FNP_RT_ALLOW_OPEN=1</code> opens these routes without a login.' +
        (o && o.hint ? ' Server hint: <q>' + esc(o.hint) + '</q>.' : '') + '</p></details>' +
        '<div class="ev-actions"><button type="button" class="btn" data-act="gate-retry">Check again</button><a class="btn ghost" href="#desk">Open the Desk</a></div></div></div>';
    } else if (kind === 'limit') {
      h = '<div class="ev-card-note" role="alert"><span class="ev-card-ico" aria-hidden="true">' + H.svgIco('clock') + '</span><div><h2>Too many requests</h2><p>The server is limiting how fast this page can ask for data. Wait ' +
        esc(o && o.retry ? o.retry + ' seconds' : 'a moment') + ' and try again.</p><div class="ev-actions"><button type="button" class="btn" data-act="reload">Try again</button></div></div></div>';
    } else if (kind === 'auth') {
      h = '<div class="ev-card-note" role="alert"><span class="ev-card-ico" aria-hidden="true">' + H.svgIco('check') + '</span><div><h2>Sign in again</h2><p>The server asked for a login that this page did not have. Reload the page and enter your credentials when the browser asks.</p>' +
        '<div class="ev-actions"><button type="button" class="btn" data-act="hard-reload">Reload the page</button></div></div></div>';
    } else {
      h = '<div class="ev-card-note" role="alert"><span class="ev-card-ico" aria-hidden="true">' + H.svgIco('warn') + '</span><div><h2>' + esc(o && o.network ? 'The server could not be reached' : 'Triage could not be loaded') + '</h2><p>' +
        esc(o && o.network ? 'Check your connection, then try again.' : (o && o.message ? 'The server said: ' + o.message + '. ' : '') + 'Try again in a moment.') + '</p><div class="ev-actions"><button type="button" class="btn" data-act="reload">Try again</button></div></div></div>';
    }
    el.notice.innerHTML = h; el.notice.hidden = false; el.top.hidden = true; el.br.hidden = true;
    root.setAttribute('data-trstate', 'notice');
  }
  function clearNotice() { el.notice.hidden = true; el.notice.innerHTML = ''; el.top.hidden = false; el.br.hidden = false; root.removeAttribute('data-trstate'); }
  function failure(e) {
    if (e && e.status === 403 && e.code === 'login_required') { gateP = null; notice('gate', e); return true; }
    if (e && e.status === 401) { notice('auth'); return true; }
    if (e && e.status === 429) { notice('limit', e); return true; }
    if (e && e.network) { notice('error', e); return true; }
    return false;
  }

  /* ---------- frame ---------- */
  function build() {
    if (built) return; built = true;
    el = { top: $('#tr-top'), kpis: $('#tr-kpis'), banner: $('#tr-banner'), notice: $('#tr-notice'), br: $('#tr-browse'), seg: $('#tr-seg'), how: $('#tr-how'), src: $('#tr-src') };
    el.br.innerHTML =
      '<div class="desk evb tr" id="trb" data-pane="list">' +
        '<div class="desk-list" id="trb-list" tabindex="-1">' +
          '<div class="dl-head" id="trb-head"><h2 class="sr">Triage items</h2>' +
            '<div class="dl-line"><div class="dl-q" role="search" aria-label="Search triage items"><label class="sr" for="trb-q">Search the text</label>' +
              '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="10.5" cy="10.5" r="6"/><path d="m15 15 5.5 5.5"/></svg>' +
              '<input type="search" id="trb-q" placeholder="Search the text" autocomplete="off" maxlength="300"></div>' +
              '<button type="button" class="more-btn" id="trb-more-btn" aria-haspopup="dialog" aria-expanded="false" aria-controls="trb-more">More</button></div>' +
            '<div class="vchips" id="trb-tier" role="group" aria-label="Filter by tier"></div>' +
            '<div class="dl-selects tr-selects">' +
              '<label class="f"><span class="sr">Platform</span><select id="trb-platform"><option value="">All platforms</option></select></label>' +
              '<label class="f"><span class="sr">Time window</span><select id="trb-days">' + WINDOWS.map(function (w) { return '<option value="' + w[0] + '">' + esc(w[1]) + '</option>'; }).join('') + '</select></label>' +
              '<label class="f"><span class="sr">Sort</span><select id="trb-sort">' + SORTS.map(function (s) { return '<option value="' + s[0] + '">' + esc(s[1]) + '</option>'; }).join('') + '</select></label></div>' +
            '<div class="dl-active"><span class="dl-count" id="trb-count" role="status" aria-live="polite">Loading</span><span class="dl-fchips" id="trb-active"></span></div>' +
            '<div class="more-pop" id="trb-more" role="dialog" aria-label="More filters" hidden><h2>More filters</h2><div class="more-grid">' +
              '<label class="f f-wide"><span>Language</span><select id="trb-lang"><option value="">All languages</option></select></label>' +
              '<label class="f f-wide"><span>Country named in the text</span><select id="trb-country"><option value="">All countries</option></select></label>' +
              '<label class="f f-wide"><span>Propaganda technique</span><select id="trb-technique"><option value="">Any technique</option></select></label></div>' +
              '<p class="note tr-winnote" id="trb-winnote"></p>' +
              '<div class="more-foot"><button type="button" class="btn ghost" id="trb-reset">Reset all filters</button></div></div>' +
          '</div>' +
          '<ul class="rows" id="trb-rows" role="listbox" aria-label="Triage items"></ul>' +
          '<div class="rows-empty" id="trb-empty" hidden></div><div class="rows-foot" id="trb-foot" hidden></div>' +
        '</div>' +
        '<section class="desk-detail" id="trb-detail" aria-label="Case file" tabindex="-1">' +
          '<div class="guide" id="trb-guide"><div class="eyebrow lbl">Case file</div><h2>How to read a triage item</h2>' +
            '<p class="guide-lede">Select an item to open its case file here. Use <kbd>j</kbd> and <kbd>k</kbd> or the arrow keys to move through the list, Enter to open and Esc to go back. The list is ordered by priority: the tier first, then how strong the signals are.</p>' +
            '<ol class="guide-steps"><li><b>Why this was flagged.</b> The rule that put the item in its tier, in plain words, with the numbers behind it.</li>' +
            '<li><b>Inferred by FNP.</b> The propaganda signal, a bar for each of the 14 techniques, the highlighted wording, countries and the nearest known fact-check. All estimates, none validated.</li>' +
            '<li><b>From the database.</b> What the collected post itself says: platform, author, language, time and reach.</li>' +
            '<li><b>Next step.</b> Open the original, open the fact-check, or analyse the item in the Workbench (realtime items). Nothing runs from this tab.</li></ol>' +
            '<p class="note">A tier says what to read first. It is not a verdict, and no tier has been checked against hand-labelled posts.</p></div>' +
          '<div id="trb-body" hidden></div></section>' +
      '</div><div class="sr" id="trb-live" aria-live="polite"></div>';
    bEl = { desk: $('#trb'), scroller: $('#trb-list'), rows: $('#trb-rows'), foot: $('#trb-foot'), count: $('#trb-count'), active: $('#trb-active'), q: $('#trb-q'), tier: $('#trb-tier'),
      platform: $('#trb-platform'), days: $('#trb-days'), sort: $('#trb-sort'), lang: $('#trb-lang'), country: $('#trb-country'), technique: $('#trb-technique'), more: $('#trb-more-btn'), pop: $('#trb-more'),
      detail: $('#trb-detail'), guide: $('#trb-guide'), body: $('#trb-body'), live: $('#trb-live'), head: $('#trb-head'), empty: $('#trb-empty'), winnote: $('#trb-winnote') };
    wire();
  }

  /* ---------- KPI strip, honest banner, rules ---------- */
  function kpiTile(label, value, sub, cls) {
    return '<div class="tr-kpi ' + (cls || '') + '"><span class="lbl">' + label + '</span><b>' + esc(value) + '</b><small>' + sub + '</small></div>';
  }
  function builtText(iso) { return iso ? 'built ' + esc(H.ago(iso)) : 'build time unknown'; }
  function nextText(iso) {
    if (!iso) return '';
    var d = Date.parse(iso); if (isNaN(d)) return '';
    var m = (d - H.nowMs()) / 60000;
    return m < 1 ? 'next run any moment now' : 'next run in ' + H.dur(m);
  }
  function renderTop() {
    if (!meta) { put(el.kpis, ''); return; }
    var m = meta.sources[cur], h = '', n;
    if (!m || m.available === false) {
      put(el.kpis, '<div class="tr-kpi tr-kpi-wide"><span class="lbl">' + esc(SRC_NAME[cur]) + '</span><b>Not available</b><small>' + esc(m && m.reason ? m.reason : 'No data has been loaded for this source yet.') + '</small></div>');
      renderBanner(m); renderHow(m); return;
    }
    var tiers = m.tiers || {}, total = (tiers.high || 0) + (tiers.medium || 0) + (tiers.low || 0);
    ['high', 'medium', 'low'].forEach(function (t) {
      h += '<div class="tr-kpi k-' + t + '"><span class="lbl">' + term('tier', TIER_NAME[t] + ' tier') + '</span><b>' + esc(fmtN(tiers[t] || 0)) + '</b><small>' + esc(total ? H.pct(tiers[t] || 0, total, 0) + ' of ' + fmtN(total) + ' ranked' : 'none') + '</small></div>';
    });
    if (cur === 'realtime') {
      n = m.n_items;
      h += kpiTile('Share scored', m.share_scored == null ? 'n/a' : (m.share_scored * 100 < 1 ? (m.share_scored * 100).toFixed(2) : (m.share_scored * 100).toFixed(1)) + '%',
        esc(fmtN(m.n_queue + m.n_too_short)) + ' of ' + esc(fmtN(n)) + ' items scored by FNP; ' + esc(fmtN(m.n_unscored)) + ' not yet, ' + esc(fmtN(m.n_too_short)) + ' too short to rate. Unscored items are not ranked.');
      h += kpiTile(term('validated', 'Validated'), '0 of ' + fmtN(m.n_queue), 'items checked by a person');
      h += kpiTile('FNP batch built', m.built_at ? H.ago(m.built_at) : 'unknown', esc(nextText(m.next_run_at) || 'next run not announced') + (m.n_deep ? '; ' + esc(fmtN(m.n_deep)) + ' with a deep verdict' : ''));
    } else {
      h += kpiTile('FNP verdicts', fmtN(m.n_with_verdict), 'of ' + esc(fmtN(m.n_items)) + ' items have an FNP verdict; the rest are listed by fact-check match.');
      h += kpiTile(term('validated', 'Validated'), fmtN(m.validated) + ' of ' + fmtN(m.n_items), 'items checked by a person');
      h += kpiTile('Export built', m.built_at ? H.ago(m.built_at) : 'unknown', esc(m.data_through ? 'posts through ' + L.dateLong(m.data_through) + '. ' : '') + 'Rebuilt by FNP after each scan; no fixed schedule.');
    }
    put(el.kpis, h);
    renderBanner(m); renderHow(m);
  }
  function renderBanner(m) {
    var vals = m ? (cur === 'realtime' ? m.validated : m.validated) : 0, none = !vals;
    var h = '<span class="tr-banner-ico" aria-hidden="true">' + H.svgIco('warn') + '</span><span class="tr-banner-t">' +
      (none ? '<b>No item has been validated by a person yet.</b> ' : '<b>' + esc(fmtN(vals)) + ' item' + (vals === 1 ? ' has' : 's have') + ' been validated by a person.</b> ') +
      '<span class="tr-bn2">Tiers are a ranking aid for what to read first, not verdicts.</span></span>';
    put(el.banner, h);
  }
  function renderHow(m) {
    var rules = (m && m.rules) || {}, th = m && m.thresholds, h = '<summary>How the tiers are decided</summary><div class="tr-how-body">';
    h += '<dl class="tr-rules">';
    ['high', 'medium', 'low'].forEach(function (t) { if (rules[t]) h += '<div class="t-' + t + '"><dt>' + tierTag(t) + '</dt><dd>' + esc(rules[t]) + '</dd></div>'; });
    if (rules.excluded) h += '<div><dt><span class="tr-tier t-none">Not ranked</span></dt><dd>' + esc(rules.excluded) + '</dd></div>';
    h += '</dl>';
    if (rules.order) h += '<p class="note">' + esc(rules.order) + '</p>';
    if (cur === 'fnp') h += '<p class="note">' + esc((m && m.window_note) || '') + '</p>';
    if (th && cur === 'realtime') h += '<p class="note">Lines used: strong propaganda signal at least ' + esc(n2(th.p_prop_high)) + ', propaganda signal at least ' + esc(n2(th.p_prop_medium)) + ', fact-check similarity at least ' + esc(n2(th.cosine_medium)) +
      ', deep fake-or-misleading probability at least ' + esc(n2(th.deep_medium)) + ', lowered when a deep verdict is below ' + esc(n2(th.deep_demote)) + '. These lines are assumptions, not fitted or validated.</p>';
    var cav = (meta && meta.caveats || []).concat((m && m.caveats) || []);
    if (cav.length) h += '<ul class="tr-cav">' + cav.map(function (c) { return '<li>' + esc(c) + '</li>'; }).join('') + '</ul>';
    put(el.how, h + '</div>');
  }
  function renderSeg() {
    Array.prototype.forEach.call(el.seg.querySelectorAll('button'), function (b) { b.setAttribute('aria-pressed', b.getAttribute('data-src') === cur); });
    var m = meta && meta.sources[cur];
    var h = '<span class="src-chip rt">' + esc(srcLabel(cur)) + '</span>';
    if (m && m.available !== false && cur === 'fnp') h += '<span class="ev-src-t">' + esc(fmtN(m.n_items)) + ' items, ' + builtText(m.built_at) + '</span>';
    else if (m && m.available !== false) h += '<span class="ev-src-t">' + esc(fmtN(m.n_queue)) + ' ranked items from the local store</span>';
    put(el.src, h);
  }

  /* ---------- data ---------- */
  function loadMeta(force) {
    if (metaP && !force && Date.now() - metaAt < 45000) return metaP;
    metaAt = Date.now();
    metaP = rt('/api/triage/meta').then(function (m) { meta = m; H.setClock(m.generated_at); return m; });
    metaP.catch(function () { metaP = null; });
    return metaP;
  }
  function loadFacets(src) {
    if (facets[src] && Date.now() - facetsAt[src] < 300000) return Promise.resolve(facets[src]);
    return rt('/api/triage/facets?source=' + src).then(function (f) { facets[src] = f; facetsAt[src] = Date.now(); return f; }).catch(function () { return facets[src] || null; });
  }
  function params(page) {
    var f = F_(), p = new URLSearchParams(), set = function (k, v) { if (v) p.set(k, v); };
    p.set('source', cur); set('q', f.q.trim()); set('tier', f.tier); set('platform', f.platform); set('lang', f.lang); set('country', f.country); set('technique', f.technique); set('days', f.days);
    if (f.sort !== 'priority') p.set('sort', f.sort);
    p.set('size', PAGE); p.set('page', page);
    return p.toString();
  }

  /* ---------- filters ---------- */
  function fillSelect(s, list, allLabel, labeler, keep) {
    var v = keep, h = '<option value="">' + esc(allLabel) + '</option>';
    (list || []).forEach(function (o) { h += '<option value="' + esc(o.value) + '">' + esc(labeler(o) + ' (' + fmtN(o.count) + ')') + '</option>'; });
    if (v && !(list || []).some(function (o) { return o.value === v; })) h += '<option value="' + esc(v) + '">' + esc(labeler({ value: v })) + '</option>';
    s.innerHTML = h; s.value = v || '';
  }
  function fillFilters() {
    var f = F_(), fc = facets[cur] || {};
    fillSelect(bEl.platform, fc.platforms, 'All platforms', function (o) { return L.platform(o.value); }, f.platform);
    fillSelect(bEl.lang, fc.langs, 'All languages', function (o) { return L.lang(o.value); }, f.lang);
    fillSelect(bEl.country, fc.countries, 'All countries', function (o) { return o.name || o.value; }, f.country);
    fillSelect(bEl.technique, fc.techniques, 'Any technique', function (o) { return L.technique(o.value); }, f.technique);
    bEl.winnote.textContent = cur === 'fnp' ? 'Time windows count back from the newest post in the export, not from today.' : 'Time windows count back from now.';
  }
  function syncControls() {
    var f = F_(); bEl.q.value = f.q; bEl.days.value = f.days; bEl.sort.value = f.sort;
    fillFilters(); renderTierChips(); renderActive();
  }
  function renderTierChips() {
    var f = F_(), t = bl.tiers, tot = t ? t.high + t.medium + t.low : null;
    var h = '<button type="button" class="vchip all" data-tier="" aria-pressed="' + (!f.tier) + '">All' + (tot == null ? '' : ' <small>' + esc(fmtN(tot)) + '</small>') + '</button>';
    ['high', 'medium', 'low'].forEach(function (k) {
      h += '<button type="button" class="vchip tc-' + k + '" data-tier="' + k + '" aria-pressed="' + (f.tier === k) + '"><i aria-hidden="true"></i>' + esc(TIER_NAME[k]) + (t ? ' <small>' + esc(fmtN(t[k])) + '</small>' : '') + '</button>';
    });
    bEl.tier.innerHTML = h;
  }
  function optText(s, v) { var o = Array.prototype.filter.call(s.options, function (x) { return x.value === v; })[0]; return o ? o.textContent.replace(/\s*\([\d,]+\)$/, '') : v; }
  function fchip(key, text) { return '<button type="button" class="fchip" data-clear="' + key + '"><span>' + esc(text) + '</span><b aria-hidden="true">&times;</b><span class="sr">Remove filter</span></button>'; }
  function renderActive() {
    var f = F_(), chips = '', n = 0;
    [['q', 'Search: ' + f.q.trim(), f.q.trim()], ['platform', optText(bEl.platform, f.platform), f.platform], ['days', optText(bEl.days, f.days), f.days], ['lang', optText(bEl.lang, f.lang), f.lang],
      ['country', optText(bEl.country, f.country), f.country], ['technique', optText(bEl.technique, f.technique), f.technique]].forEach(function (c) { if (c[2]) { chips += fchip(c[0], c[1]); } });
    ['lang', 'country', 'technique'].forEach(function (k) { if (f[k]) n++; });
    bEl.more.innerHTML = 'More' + (n ? ' <span class="n">' + n + '</span>' : '');
    if (chips) chips += '<button type="button" class="fchip clear" data-clear="all">Clear all</button>';
    bEl.active.innerHTML = chips;
    var c = bl.error ? 'Could not load' : bl.total == null ? 'Loading' : fmtN(bl.total) + (bl.total === 1 ? ' item' : ' items');
    bEl.count.textContent = c;
  }

  /* ---------- list ---------- */
  function signals(it) {
    var h = '';
    if (it.p_prop != null) h += '<span class="ev-sc" title="Propaganda signal from the FNP encoder, uncalibrated"><span class="ev-sck">Prop.</span>' + mini('Propaganda signal', it.p_prop, 'acc') + '<span class="ev-scn">' + n2(it.p_prop) + '</span></span>';
    else if (it.source === 'realtime' && it.deep == null) h += '<span class="ev-short" title="No propaganda signal for this item">No signal</span>';
    var nf = it.nearest_factcheck;
    if (nf && (nf.cosine != null || nf.rerank != null)) {
      var v = fcLabel(nf.label);
      h += '<span class="tr-fc" title="Nearest known fact-check: a lookup, not a verdict on this item"><span class="ev-sck">Fact-check</span>' + (v === 'UNRATED' ? '' : '<span class="chip v-' + esc(v) + '">' + esc(L.verdict(v)) + '</span>') + '<span class="ev-sck">' + esc(n2(nf.rerank != null ? nf.rerank : nf.cosine)) + (nf.rerank != null ? ' rerank' : ' similar') + '</span></span>';
    }
    if (it.verdict) h += '<span class="tr-fc" title="FNP\'s own verdict on this post"><span class="ev-sck">FNP verdict</span>' + vchip(it.verdict) + '</span>';
    if (it.deep && it.deep.verdict) h += '<span class="ev-an deep" title="A deep Claim Check verdict exists for this item">Deep: ' + esc(L.verdict(it.deep.verdict)) + (it.deep.p_fake_misleading != null ? ' ' + esc(n2(it.deep.p_fake_misleading)) : '') + '</span>';
    return h;
  }
  function rowHTML(it) {
    var id = it.id, head = it.title ? '<span class="row-title tr-title">' + esc(it.title) + '</span>' : '';
    return '<li class="row tr-row t-' + esc(it.tier) + '" role="option" data-id="' + esc(id) + '" tabindex="-1" aria-selected="' + (id === sel && cur === selSrc) + '"' + (id === sel && cur === selSrc ? ' aria-current="true"' : '') + '>' +
      '<span class="tr-top">' + tierTag(it.tier) + '<span class="ev-plat">' + esc(L.platform(it.platform)) + '</span>' + (it.author ? '<span aria-hidden="true">&middot;</span><span class="row-src">' + esc(it.author) + '</span>' : '') +
      '<span class="tr-when" title="' + esc(it.posted_at ? L.dateTimeUTC(it.posted_at) : '') + '">' + esc(it.posted_at ? H.ago(it.posted_at) : 'undated') + '</span></span>' +
      head + '<span class="tr-snip">' + esc(it.snippet || 'No text') + '</span>' +
      '<span class="row-score tr-sig">' + signals(it) + '<span class="tr-reach">' + esc(reachText(it.reach)) + '</span>' + (it.rule && it.rule !== 'below' ? '<span class="tr-why" title="The rule that set this tier">' + esc(RULE_TEXT[it.rule] || '') + '</span>' : '') + '</span></li>';
  }
  function skeletonRows() { var r = ''; for (var i = 0; i < 7; i++) r += '<li class="row sk" aria-hidden="true"><span class="skel w1"></span><span class="skel w2"></span><span class="skel w3"></span></li>'; return r; }
  function setFoot(html) { bEl.foot.innerHTML = html; bEl.foot.hidden = !html; }
  function footState() {
    if (bl.error) { setFoot('<p>Could not load more items.</p><button type="button" class="btn ghost" data-act="more-retry">Try again</button>'); return; }
    if (bl.total === 0) { setFoot(''); return; }
    if (bl.done) { setFoot(bl.limitHit ? 'Showing the first ' + fmtN(OFFSET_MAX) + ' results. Narrow the filters to reach the rest.' : 'End of list. ' + fmtN(bl.shown) + ' item' + (bl.shown === 1 ? '' : 's') + ' shown.'); return; }
    setFoot('<span class="spin" aria-hidden="true"></span>Loading more items');
  }
  function showEmpty(title, text, retry) {
    bEl.empty.innerHTML = '<b>' + esc(title) + '</b><span>' + esc(text) + '</span>' + (retry === 'retry' ? '<button type="button" class="btn ghost" data-act="more-retry-all">Try again</button>' : retry === 'reset' ? '<button type="button" class="btn ghost" data-clear="all">Reset filters</button>' : '');
    bEl.empty.hidden = false;
  }
  function emptyFor(d) {
    if (d && d.available === false) { showEmpty('There is no triage export to show', (d.reason ? d.reason.charAt(0).toUpperCase() + d.reason.slice(1) + '. ' : '') + 'The FNP session writes it with its triage export script; this page picks it up by itself once it exists.', 'retry'); return; }
    var f = F_(), any = f.q || f.tier || f.platform || f.lang || f.country || f.technique || f.days;
    if (any) showEmpty('No items match these filters', 'Try removing a filter or searching for fewer words.', 'reset');
    else if (cur === 'realtime') showEmpty('The queue is empty', 'FNP has not scored any item in the local store yet, or the batch export is not loaded. Unscored items are never ranked. The batch export fills in over several runs.', 'retry');
    else showEmpty('The export has no items', 'The triage export loaded but lists no items.', 'retry');
  }
  function fetchPage(page, replace, seq) {
    bl.loading = true; bl.error = null; bEl.rows.setAttribute('aria-busy', 'true');
    if (!bEl.rows.children.length) bEl.rows.innerHTML = skeletonRows();
    footState();
    rt('/api/triage/items?' + params(page)).then(function (d) {
      if (seq !== bl.seq) return;
      bl.page = d.page; bl.total = d.total; bl.tiers = d.tiers; bl.loading = false; bl.available = d.available !== false;
      if (replace) { bEl.rows.innerHTML = ''; bEl.scroller.scrollTop = 0; bl.shown = 0; }
      bl.shown += d.items.length;
      bl.limitHit = d.page * d.size > OFFSET_MAX && d.page < d.pages;
      bl.done = d.page >= d.pages || bl.limitHit || !d.items.length;
      if (!d.items.length && replace) { bEl.rows.innerHTML = ''; bEl.foot.hidden = true; emptyFor(d); }
      else { bEl.empty.hidden = true; bEl.rows.insertAdjacentHTML('beforeend', d.items.map(rowHTML).join('')); if (replace) C.setCursor(bEl.rows, bEl.rows.querySelector('.row'), false); }
      bEl.rows.removeAttribute('aria-busy'); renderTierChips(); renderActive(); footState(); applyHeadPad(); watch();
    }).catch(function (e) {
      if (seq !== bl.seq) return;
      bl.loading = false; bl.error = e.message; bEl.rows.removeAttribute('aria-busy');
      if (failure(e)) return;
      if (replace) { bEl.rows.innerHTML = ''; showEmpty('The list could not be loaded', e.status === 400 ? (e.message || 'These filters are not valid.') : e.status === 503 ? (e.message || 'The data is not ready yet.') : 'Check your connection, then try again.', 'retry'); setFoot(''); } else footState();
      renderActive();
    });
  }
  function reload() { bl.seq++; bl.page = 0; bl.total = null; bl.done = false; bl.limitHit = false; bl.error = null; renderActive(); fetchPage(1, true, bl.seq); }
  function schedule() { clearTimeout(tm); tm = setTimeout(reload, 140); }
  function changed() { renderTierChips(); renderActive(); schedule(); }
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
    C.rows(bEl.rows).forEach(function (r) { var on = r.getAttribute('data-id') === sel && cur === selSrc; r.setAttribute('aria-selected', on); if (on) r.setAttribute('aria-current', 'true'); else r.removeAttribute('aria-current'); });
  }
  function selectedRow() { return C.rows(bEl.rows).filter(function (r) { return r.getAttribute('data-id') === sel; })[0] || null; }

  /* ---------- case file ---------- */
  function openLink(url) { return H.openLink(url); }
  function whySection(d) {
    var t = d.triage, it = d.item, tier = t.tier, h = '', rows = [];
    var inQueue = d.source === 'fnp' || t.in_queue;
    var sub = H.BADGE_FNP + ' ' + (inQueue ? 'The rule that put this item in the <b>' + esc(TIER_NAME[tier]) + '</b> tier, with the numbers behind it.' : 'This item is not in the ranked queue.');
    h += '<ul class="tr-why-list">' + (t.why || []).map(function (w) { return '<li>' + esc(w) + '</li>'; }).join('') + '</ul>';
    if (t.rule_text) h += '<p class="note"><b>' + (inQueue ? esc(TIER_NAME[tier]) + ' tier rule' : 'Why it is not ranked') + ':</b> ' + esc(t.rule_text) + '</p>';
    if (d.source === 'realtime' && t.signals) {
      var s = t.signals;
      rows.push(['Propaganda signal', H.fv(s.p_prop == null ? 'Not scored' : '<span class="ev-big">' + esc(n2(s.p_prop)) + '</span> ' + mini('Propaganda signal', s.p_prop, 'acc'), H.BADGE_FNP)]);
      rows.push(['Strong spans', H.fv(s.n_strong == null ? 'Not scored' : esc(fmtN(s.n_strong)), H.BADGE_FNP)]);
      rows.push(['Fact-check match', H.fv(s.rerank != null ? 'Reranked: ' + esc(n2(s.rerank)) + ' (similarity ' + esc(n2(s.cosine)) + ')' : s.cosine != null ? 'Similarity ' + esc(n2(s.cosine)) + ', not reranked' : 'None above the floor', H.BADGE_FNP)]);
      if (s.has_deep) rows.push(['Deep verdict', H.fv('Fake or misleading ' + esc(n2(s.deep_pfm)), H.BADGE_FNP)]);
      h += C.kv(rows);
    }
    h += '<p class="note tr-unval"><b>Not validated.</b> No person has checked this item or this tier. ' + (t.validated ? '' : 'Treat it as a lead and read the original before acting.') + '</p>';
    if (d.source === 'realtime') h += '<p class="ev-actions"><button type="button" class="btn" data-act="wb-add" data-id="' + esc(it.id) + '">Analyse in the Workbench</button><span class="note tr-wbnote">Adds the item to the Workbench selection. Nothing runs until you press Run there.</span></p>';
    return C.section('trg-s-why', 'Why this was flagged', sub, h);
  }
  function countriesList(list) {
    if (!list || !list.length) return 'None found';
    return '<ul class="ev-ctry">' + list.map(function (c) { return '<li><span class="ev-cn">' + esc(c.name || c.iso3) + '</span> <span class="ev-tm">' + esc(c.iso3) + '</span></li>'; }).join('') + '</ul><span class="sub">Countries the text names, by FNP\'s place matching. Not where the post was written.</span>';
  }
  function fcCard(nf) {
    if (!nf) return '';
    var v = fcLabel(nf.label), ext = F.safeUrl(nf.url), weak = nf.similarity != null && nf.similarity < 0.72;
    var link = nf.desk_id ? '<a class="btn ghost" href="#desk/' + encodeURIComponent(nf.desk_id) + '">Open in the Desk</a>'
      : ext ? '<a class="case-open" href="' + esc(ext) + '" target="_blank" rel="noopener noreferrer">Open the fact-check<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M14 4h6v6M20 4l-9 9M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5"/></svg><span class="sr"> (opens in a new tab)</span></a>' : '';
    return '<h4 class="cs-sub-2">Nearest fact-check (FNP)</h4><div class="ev-card ev-fnf s-' + esc(v) + '"><span class="ev-fnf-top">' + vchip(v) + '<span class="ev-sim">similarity ' + esc(n2(nf.similarity)) + (nf.rerank != null ? ', rerank ' + esc(n2(nf.rerank)) : '') + '</span></span>' +
      (nf.claim ? '<span class="ev-title">' + esc(nf.claim) + '</span>' : '<span class="ev-snip">No claim text was exported for this fact-check.</span>') +
      '<span class="ev-simbar" aria-hidden="true"><b style="width:' + (nf.similarity == null ? 0 : Math.max(0, Math.min(1, nf.similarity)) * 100).toFixed(0) + '%"></b></span>' +
      '<span class="ev-snip ev-fnf-note">' + (weak ? 'Weak match: similarity is below 0.72, where FNP stops reranking. A similar topic is not the same claim. ' : '') + 'The label is the fact-check\'s own rating of that claim, not a verdict on this item. Check that it is the same claim.' +
      (nf.debunk_match ? ' FNP marks this a debunk match.' : '') + '</span>' + (link ? '<span class="ev-fnf-act">' + link + '</span>' : '') + '</div>';
  }
  function fnpCase(d) {
    var it = d.item, tr = d.triage, text = it.text || '';
    var first = oneLine(text.split('\n')[0]), title = first.length > 160 ? first.slice(0, 159).trim() + '…' : first || 'No text';
    var rest = oneLine(text).slice(first.length).trim();
    var left = tierTag(it.tier) + '<span class="chip tag">Unvalidated</span><span class="ob ob-fnp solid">FNP triage export</span>';
    var h = '<article class="case evcase trcase" aria-labelledby="trg-title">' + C.bar(left, openLink(it.url), 'Back to the list') + '<div class="case-in">';
    h += '<div class="case-hero ev-hero"><div class="case-id"><p class="case-kicker">' + esc([L.platform(it.platform), it.author || 'Unknown author', L.lang(it.lang), it.posted_at ? H.ago(it.posted_at) : 'undated'].join(' - ')) + '</p>' +
      '<h2 class="case-title" id="trg-title" tabindex="-1">' + esc(title) + '</h2>' + (rest ? '<p class="case-claim clamp">' + esc(rest) + '</p>' : '') + '</div></div>';
    h += whySection(d);
    var rows = [], add = function (label, val, badge) { rows.push([label, H.fv(val, badge || H.BADGE_DB)]); };
    add('Platform', esc(L.platform(it.platform)));
    add('Author', it.author ? esc(it.author) : 'Unknown');
    add('Language', esc(L.lang(it.lang)), H.BADGE_DB);
    add('Posted', it.posted_at ? esc(L.dateTimeUTC(it.posted_at)) + '<span class="sub">' + esc(H.ago(it.posted_at)) + '</span>' : 'Unknown');
    add('Reach', it.reach == null ? 'Not recorded' : esc(fmtN(it.reach)) + '<span class="sub">Likes, comments, shares and views where the platform gives them.</span>');
    h += C.section('trg-s-db', 'From the database', H.BADGE_DB + ' What the collected post says about itself, shown unchanged.', C.kv(rows));
    rows = [];
    var v = it.verdict, add2 = function (label, val) { rows.push([label, H.fv(val, H.BADGE_FNP)]); };
    add2('Verdict', v ? vchip(v.label) + '<span class="sub">FNP\'s own call' + (v.basis ? ' (basis: ' + esc(v.basis) + (v.source ? ', ' + esc(L.titleCase(String(v.source).replace(/_/g, ' '))) : '') + ')' : '') + (v.at ? ', ' + esc(L.dateTimeUTC(v.at)) : '') + '. A lead, not a ruling.</span>' : 'No verdict<span class="sub">FNP has no verdict for this post; it is listed because its nearest fact-check is close. That says nothing about whether it is true.</span>');
    if (it.fake_score != null) add2('Fake score', '<span class="ev-big">' + esc(n2(it.fake_score)) + '</span> ' + mini('Fake or misleading score', it.fake_score, F.band(it.fake_score)) + '<span class="sub">P(fake) ' + esc(n2(it.p_fake)) + ' + P(misleading) ' + esc(n2(it.p_misleading)) + '. Propaganda is left out of this score.</span>');
    add2(term('uncalibrated signal', 'Propaganda signal (p_prop)'), it.p_prop == null ? 'Not computed<span class="sub">Posts under 12 tokens are too short to score. That says nothing about whether the post is true.</span>' : '<span class="ev-big">' + esc(n2(it.p_prop)) + '</span> ' + mini('Propaganda signal', it.p_prop, 'acc') + '<span class="sub">An <b>uncalibrated signal</b>: it ranks posts, it does not say that this share of such posts is propaganda.</span>');
    if (it.n_spans) add2('Flagged wording', esc(H.plural(it.n_spans, 'span', 'spans')) + '<span class="sub">' + (it.spans.length < it.n_spans ? 'The case file highlights the ' + esc(fmtN(it.spans.length)) + ' most confident. ' : '') + (it.spans_dropped ? esc(fmtN(it.spans_dropped)) + ' could not be placed in the text.' : '') + '</span>');
    add2('Countries mentioned', countriesList(it.countries));
    add2('Model', esc(it.model_version || 'Unknown') + '<span class="sub">' + (it.built_at ? 'built ' + esc(L.dateTimeUTC(it.built_at)) : '') + '</span>');
    var cc = window.ClaimCheck, bars = cc && cc.classBars ? cc.classBars({ classes: H.techClasses(it.techniques), basis_note: SPAN_NOTE }) : '';
    var body = C.kv(rows) + bars + '<p class="cs-sub-2">Propaganda techniques found</p>' + H.techniquesHTML(it.techniques) + fcCard(it.nearest_factcheck);
    if (v && v.rationale) body += '<h4 class="cs-sub-2">FNP\'s stated reason</h4><blockquote class="ev-caveat"><p>' + esc(v.rationale) + '</p><footer>Written by FNP\'s adjudicator; an estimate, not a finding.</footer></blockquote>';
    h += C.section('trg-s-inf', 'Inferred by FNP', H.BADGE_FNP + ' Estimated by FNP\'s analysis, not stored in the database. These are triage signals, not findings.', body);
    var spans = it.spans || [], tbody;
    if (!text.trim()) tbody = '<p class="note">No text is stored for this item.</p>';
    else if (spans.length) {
      var hl = H.highlight(text, spans);
      tbody = '<div class="cc-text" aria-label="Post text with technique highlights">' + hl.html + '</div><p class="note">Highlights come from FNP. Each carries its technique name as text, not only a colour. The encoder reads word pieces, so a highlight can start or end inside a word.' + (hl.unplaced ? ' ' + esc(H.plural(hl.unplaced, 'phrase', 'phrases')) + ' could not be placed.' : '') + '</p>';
    } else tbody = '<div class="excerpt">' + esc(text) + '</div><p class="note">FNP flagged no phrase in this text.</p>';
    h += C.section('trg-s-text', 'Original text', H.BADGE_DB + ' The first 1,200 characters of the post.' + (spans.length ? ' ' + H.BADGE_FNP + ' for the highlights.' : ''), tbody);
    return h + '</div></article>';
  }
  /* a realtime item reuses the Evidence Base case file (database facts, deep/partial/batch blocks with the 14 bars, text with highlights,
     compare); ids are renamed so the two views never share an id, and the triage "why" section goes in right after the title */
  function rtCase(d) {
    var html = Ev._caseHTML(d.item).replace(/\b(id|aria-labelledby|for|aria-controls)="evb-/g, '$1="trg-')
      .replace(/<button type="button" class="link-btn inline" data-act="more-source"[^>]*>[^<]*<\/button>/g, '');
    var tierChip = d.triage.tier ? tierTag(d.triage.tier) + '<span class="chip tag">Unvalidated</span>' : '<span class="tr-tier t-none">Not ranked</span>';
    html = html.replace('<div class="grow">', '<div class="grow">' + tierChip);
    var i = html.indexOf('<section class="cs"');
    return i < 0 ? html : html.slice(0, i) + whySection(d) + html.slice(i);
  }
  function caseTitle(d) { return d.source === 'fnp' ? oneLine((d.item.text || '').split('\n')[0]).slice(0, 80) : H.itemTitle(d.item); }
  function pane(n) { bEl.desk.setAttribute('data-pane', n); root.setAttribute('data-trpane', n); }
  function announce(t) { bEl.live.textContent = ''; setTimeout(function () { bEl.live.textContent = t; }, 30); }
  function showBody(html) { bEl.guide.hidden = true; bEl.body.hidden = false; bEl.body.innerHTML = html; bEl.detail.scrollTop = 0; }
  function clearCase() { caseSeq++; if (!MOBILE.matches) { sel = null; selSrc = null; } markSelected(); bEl.body.hidden = true; bEl.body.innerHTML = ''; bEl.guide.hidden = false; pane('list'); }
  function openCase(src, id, focusTitle) {
    sel = id; selSrc = src; markSelected(); pane('case');
    var my = ++caseSeq, key = src + '/' + id;
    var done = function (d) {
      if (my !== caseSeq) return;
      showBody(d.source === 'fnp' ? fnpCase(d) : rtCase(d)); Shell.setTitle(caseTitle(d)); announce('Case file opened: ' + caseTitle(d));
      if (focusTitle && MOBILE.matches) { var t = $('#trg-title'); if (t) t.focus({ preventScroll: true }); }
    };
    if (itemCache[key]) { done(itemCache[key]); return; }
    showBody(C.skeleton()); bEl.detail.setAttribute('aria-busy', 'true');
    rt('/api/triage/item/' + encodeURIComponent(id) + '?source=' + src).then(function (d) {
      itemCache[key] = d; var ks = Object.keys(itemCache); if (ks.length > 30) delete itemCache[ks[0]];
      bEl.detail.removeAttribute('aria-busy'); done(d);
    }).catch(function (e) {
      bEl.detail.removeAttribute('aria-busy');
      if (my !== caseSeq) return;
      if (failure(e)) return;
      showBody(C.error(e.status === 404 ? 'This item is not in the ' + SRC_NAME[src] + ' (it may have been rebuilt away or aged out of the store).' : 'Check your connection, then try again.', id));
    });
  }
  function goBack() { if (routeCount > 1 && history.length > 1) history.back(); else Shell.navigate('#triage/' + cur, true); }
  function openRow(row) { Shell.navigate('#triage/' + cur + '/' + encodeURIComponent(row.getAttribute('data-id'))); }

  /* ---------- wiring ---------- */
  function popOpen(open) { bEl.pop.hidden = !open; bEl.more.setAttribute('aria-expanded', open ? 'true' : 'false'); if (open) { var f = $('#trb-lang'); if (f) f.focus(); } }
  function resetAll() { ST[cur] = copy(DEF); syncControls(); schedule(); }
  function wire() {
    bEl.detail.addEventListener('click', function (e) {
      if (C.onDetailClick(e)) return;
      var b = e.target.closest('[data-act]'); if (!b) return;
      var a = b.getAttribute('data-act');
      if (a === 'back') goBack();
      else if (a === 'retry') openCase(cur, b.getAttribute('data-id'), false);
      else if (a === 'wb-add') { if (window.Workbench) window.Workbench.analyse(b.getAttribute('data-id')); }
    });
    bEl.rows.addEventListener('click', function (e) { var r = e.target.closest('.row[data-id]'); if (r) { C.setCursor(bEl.rows, r, false); openRow(r); } });
    C.rowNav(bEl.rows, openRow);
    bEl.tier.addEventListener('click', function (e) { var b = e.target.closest('.vchip'); if (!b) return; F_().tier = b.getAttribute('data-tier'); changed(); });
    var qt = null;
    bEl.q.addEventListener('input', function () { F_().q = bEl.q.value; clearTimeout(qt); qt = setTimeout(changed, 260); });
    bEl.q.addEventListener('keydown', function (e) { if (e.key === 'Escape' && bEl.q.value) { bEl.q.value = ''; F_().q = ''; changed(); e.stopPropagation(); } });
    [['platform', bEl.platform], ['days', bEl.days], ['sort', bEl.sort], ['lang', bEl.lang], ['country', bEl.country], ['technique', bEl.technique]].forEach(function (p) { p[1].addEventListener('change', function () { F_()[p[0]] = p[1].value; changed(); }); });
    $('#trb-reset').addEventListener('click', resetAll);
    bEl.active.addEventListener('click', function (e) {
      var b = e.target.closest('[data-clear]'); if (!b) return;
      var k = b.getAttribute('data-clear'); if (k === 'all') { resetAll(); return; }
      F_()[k] = ''; syncControls(); schedule();
    });
    bEl.empty.addEventListener('click', function (e) {
      var b = e.target.closest('button'); if (!b) return;
      if (b.getAttribute('data-act') === 'more-retry-all') { loadMeta(true).then(renderTop); reload(); } else if (b.getAttribute('data-clear') === 'all') resetAll();
    });
    bEl.foot.addEventListener('click', function (e) { if (e.target.closest('[data-act="more-retry"]')) { bl.error = null; loadMore(); } });
    bEl.more.addEventListener('click', function () { popOpen(bEl.pop.hidden); });
    document.addEventListener('click', function (e) { if (bEl && !bEl.pop.hidden && !e.target.closest('#trb-more') && !e.target.closest('#trb-more-btn')) popOpen(false); });
    bEl.pop.addEventListener('focusout', function (e) { var n = e.relatedTarget; if (n && !bEl.pop.contains(n) && n !== bEl.more) popOpen(false); });
    window.addEventListener('resize', applyHeadPad);
    el.seg.addEventListener('click', function (e) { var b = e.target.closest('button[data-src]'); if (!b || b.getAttribute('data-src') === cur) return; Shell.navigate('#triage/' + b.getAttribute('data-src')); });
    el.notice.addEventListener('click', function (e) {
      var b = e.target.closest('[data-act]'); if (!b) return;
      var a = b.getAttribute('data-act');
      if (a === 'gate-retry') { gateP = null; metaP = null; clearNotice(); enter(currentArg); }
      else if (a === 'reload') { metaP = null; clearNotice(); enter(currentArg); }
      else if (a === 'hard-reload') location.reload();
    });
  }
  /* j and k from anywhere outside form fields; Escape closes the popover, or on small screens returns to the list */
  document.addEventListener('keydown', function (e) {
    if (Shell.current().route !== 'triage' || !bEl || bEl.desk.offsetParent === null) return;
    if (e.key === 'Escape') {
      if (!bEl.pop.hidden) { popOpen(false); bEl.more.focus(); e.stopPropagation(); return; }
      if (MOBILE.matches && sel) { e.preventDefault(); goBack(); }
      return;
    }
    if ((e.key !== 'j' && e.key !== 'k') || e.ctrlKey || e.metaKey || e.altKey) return;
    var t = e.target, tag = t && t.tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || (t && t.isContentEditable) || (t && t.closest && t.closest('.desk-detail, #trb-more'))) return;
    if (t && t.closest && t.closest('#trb-rows')) return;
    e.preventDefault();
    var from = selectedRow() || C.rows(bEl.rows)[0];
    if (from) { if (!selectedRow()) C.setCursor(bEl.rows, from, true); else C.move(bEl.rows, from, e.key === 'j' ? 1 : -1); }
  });
  Shell.on(function (r) {
    routeCount++;
    if (r.route === 'triage' && bEl && !/^[a-z]+\/./.test(r.arg) && wasCase && MOBILE.matches) { var row = selectedRow(); if (row) setTimeout(function () { row.focus({ preventScroll: true }); }, 0); }
    wasCase = r.route === 'triage' && /^[a-z]+\/./.test(r.arg);
  });

  /* ---------- route ---------- */
  var currentArg = '';
  function parseArg(arg) {
    var m = /^(fnp|realtime)(?:\/(.+))?$/.exec(arg || '');
    return m ? { src: m[1], id: m[2] || null } : null;
  }
  function enter(arg) {
    currentArg = arg || '';
    var p = parseArg(arg);
    if (!p) { var fix = '#triage/' + (arg && /^realtime/i.test(arg) ? 'realtime' : cur); try { history.replaceState(null, '', fix); } catch (e) { location.hash = fix; } p = { src: /realtime$/.test(fix) ? 'realtime' : cur, id: null }; }
    build();
    var switched = p.src !== cur;
    if (switched) bl.tiers = null;
    cur = p.src;
    clearNotice(); renderSeg();
    Shell.setTitle(SRC_NAME[cur]);
    checkGate(false).then(function (g) {
      if (!g.open) { notice('gate'); return null; }
      return Promise.all([loadMeta(false), loadFacets(cur)]).then(function () {
        if (cur !== p.src) return;
        renderSeg(); renderTop(); syncControls();
        if (switched || !bl.seq || bl.src !== cur) { bl.src = cur; reload(); applyHeadPad(); watch(); }
        else { applyHeadPad(); markSelected(); }
        if (p.id) openCase(cur, p.id, true); else clearCase();
      });
    }).catch(function (e) { if (!failure(e)) notice('error', e); });
  }
  Shell.register('triage', {
    enter: function (arg) { enter(arg); },
    leave: function () { if (bEl) popOpen(false); root.removeAttribute('data-trpane'); }
  });
  window.Triage = { _rowHTML: rowHTML, _fnpCase: fnpCase, _rtCase: rtCase };
})();
