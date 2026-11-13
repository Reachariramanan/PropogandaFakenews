/* Evidence Desk shell: hash router, top bar (search, data status, help, settings), shared data cache and the (?) term tooltip.
   Views register themselves with Shell.register(route, {enter, leave}); Shell.start() runs once all scripts have loaded. */
(function () {
  'use strict';
  var $ = function (s, r) { return (r || document).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); };
  var L = window.Labels, esc = L.esc, root = document.documentElement;
  var fmtN = function (n) { return Number(n || 0).toLocaleString('en-US'); };

  /* ---------- shared data (one request each, reused by every view) ---------- */
  function getJSON(u, opts) {
    return fetch(u, opts).then(function (r) {
      return r.json().then(function (j) { if (!r.ok) throw new Error(j.error || r.status); return j; });
    });
  }
  var memo = {};
  function once(k, fn) { return memo[k] || (memo[k] = fn()); }
  function getStats() { return once('stats', function () { return getJSON('/api/stats'); }); }
  function getLabMeta() { return once('lab', function () { return getJSON('/api/proplab/meta').catch(function () { return null; }); }); }
  function getLatest() { /* publication date of the newest dated item: the anchor for "this week" */
    return once('latest', function () {
      return getJSON('/api/items?sort=date&page_size=1').then(function (d) { return d.items && d.items[0] ? d.items[0].published : null; }).catch(function () { return null; });
    });
  }

  /* ---------- router ---------- */
  var ROUTES = {
    overview: { view: 'view-overview', title: 'Overview' }, desk: { view: 'view-gallery', title: 'Desk' }, check: { view: 'view-check', title: 'Claim Check' },
    lab: { view: 'view-proplab', title: 'Propaganda Lab' }, evidence: { view: 'view-evidence', title: 'Evidence Base' }, triage: { view: 'view-triage', title: 'Triage' }, sources: { view: 'view-sources', title: 'Sources' }
  };
  var ALIAS = { gallery: 'desk', proplab: 'lab', detector: 'check' }; /* old bookmarks */
  var handlers = {}, subs = [], cur = { route: null, arg: '' }, started = false;

  function parse() {
    var h = (location.hash || '').replace(/^#\/?/, ''), i = h.indexOf('/'), r = i < 0 ? h : h.slice(0, i), a = i < 0 ? '' : h.slice(i + 1);
    try { a = decodeURIComponent(a); } catch (e) {}
    return { route: r, arg: a };
  }
  function replaceHash(h) { try { history.replaceState(null, '', h); } catch (e) { location.hash = h; } }
  function route() {
    var p = parse(), r = ALIAS[p.route] || p.route;
    if (!ROUTES[r]) { replaceHash('#overview'); r = 'overview'; p.arg = ''; }
    else if (r !== p.route) replaceHash('#' + r + (p.arg ? '/' + encodeURIComponent(p.arg) : ''));
    var changed = r !== cur.route, prev = cur.route;
    if (changed) {
      if (prev && handlers[prev] && handlers[prev].leave) handlers[prev].leave();
      Object.keys(ROUTES).forEach(function (n) { var v = $('#' + ROUTES[n].view); if (v) v.hidden = n !== r; });
      $$('[data-route]').forEach(function (a) { if (a.getAttribute('data-route') === r) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current'); });
      root.setAttribute('data-route', r);
      var live = $('#route-live'); if (live) live.textContent = ROUTES[r].title;
      closePops(null, false); closeSearch();
      window.scrollTo(0, 0);
    }
    cur = { route: r, arg: p.arg };
    setTitle(''); /* every navigation resets the tab title; a view adds its detail (claim, country) once it knows it */
    subs.forEach(function (f) { f({ route: r, arg: p.arg, prev: prev }); });
    if (handlers[r] && handlers[r].enter) handlers[r].enter(p.arg, changed);
  }
  /* page title per route: "Desk - Evidence Desk", or "<detail> - Desk - Evidence Desk" once a view knows what is open */
  function setTitle(detail) {
    var t = cur.route && ROUTES[cur.route] ? ROUTES[cur.route].title : '';
    var d = String(detail || '').replace(/\s+/g, ' ').trim(); if (d.length > 70) d = d.slice(0, 69) + '\u2026';
    document.title = (d ? d + ' - ' : '') + (t ? t + ' - ' : '') + 'Evidence Desk';
  }
  function navigate(h, replace) {
    if (location.hash === h) { route(); return; }
    if (replace) { replaceHash(h); route(); } else location.hash = h;
  }

  /* ---------- popovers: one open at a time, Escape and outside click close, focus returns to the button ---------- */
  var pops = [];
  function addPop(btn, panel, onOpen, focusSel) {
    var p = { btn: btn, panel: panel, wrap: btn.parentNode, onOpen: onOpen, focusSel: focusSel };
    pops.push(p);
    btn.addEventListener('click', function () { if (panel.hidden) openPop(p); else closePops(null, true); });
    panel.addEventListener('focusout', function (e) {
      var n = e.relatedTarget; if (n && !p.wrap.contains(n)) closePops(null, false);
    });
  }
  function openPop(p) {
    closePops(p, false); closeSearch();
    if (p.onOpen) p.onOpen();
    p.panel.hidden = false; p.btn.setAttribute('aria-expanded', 'true');
    var t = (p.focusSel && p.panel.querySelector(p.focusSel)) || p.panel;
    if (t === p.panel) p.panel.tabIndex = -1;
    t.focus({ preventScroll: true });
  }
  function closePops(except, returnFocus) {
    var any = false;
    pops.forEach(function (p) {
      if (p === except || p.panel.hidden) return;
      any = true; p.panel.hidden = true; p.btn.setAttribute('aria-expanded', 'false');
      if (returnFocus) p.btn.focus();
    });
    return any;
  }
  document.addEventListener('click', function (e) {
    pops.forEach(function (p) { if (!p.panel.hidden && !p.wrap.contains(e.target)) { p.panel.hidden = true; p.btn.setAttribute('aria-expanded', 'false'); } });
    if (!e.target.closest('.gsearch')) closeSearch();
  });

  /* ---------- settings: typeface + theme live in data attributes on <html> ---------- */
  var sPanel = $('#settings');
  var THEMES = ['system', 'light', 'dark', 'command'];
  var THEME_NAME = { system: 'System', light: 'Light', dark: 'Dark', command: 'Command Center' };
  var THEME_SHORT = { system: 'System', light: 'Light', dark: 'Dark', command: 'Command' };
  function store(k, v) { try { localStorage.setItem(k, v); } catch (e) {} }
  function curTheme() { var t = root.getAttribute('data-theme'); return THEMES.indexOf(t) < 0 ? 'command' : t; }
  function syncSettings() {
    var f = root.getAttribute('data-font') || 'default', t = curTheme();
    var rf = sPanel.querySelector('input[name="s-font"][value="' + f + '"]'), rt = sPanel.querySelector('input[name="s-theme"][value="' + t + '"]');
    if (rf) rf.checked = true; if (rt) rt.checked = true;
  }
  /* Command Center top bar: live UTC clock (runs only in that theme, only while the tab is visible) and the quick theme button */
  var clockTimer = null, pad2 = function (n) { return (n < 10 ? '0' : '') + n; };
  function tickClock() {
    var d = new Date(), t = $('#cc-time'), dt = $('#cc-date');
    if (t) t.textContent = pad2(d.getUTCHours()) + ':' + pad2(d.getUTCMinutes()) + ':' + pad2(d.getUTCSeconds());
    if (dt) dt.textContent = pad2(d.getUTCDate()) + '/' + pad2(d.getUTCMonth() + 1) + '/' + d.getUTCFullYear();
  }
  function syncClock() {
    var on = curTheme() === 'command' && !document.hidden;
    if (on && !clockTimer) { tickClock(); clockTimer = setInterval(tickClock, 1000); }
    else if (!on && clockTimer) { clearInterval(clockTimer); clockTimer = null; }
  }
  function nextTheme(t) { return THEMES[(THEMES.indexOf(t) + 1) % THEMES.length]; }
  function syncThemeBtn() {
    var t = curTheme(), b = $('#theme-quick'); if (!b) return;
    b.setAttribute('aria-label', 'Theme: ' + THEME_NAME[t] + '. Switch to ' + THEME_NAME[nextTheme(t)] + '.');
    var tt = $('#theme-quick-t'); if (tt) tt.textContent = THEME_SHORT[t];
  }
  function setTheme(v, announce) {
    if (THEMES.indexOf(v) < 0) return;
    root.setAttribute('data-theme', v); store('fnp-theme', v);
    syncSettings(); syncThemeBtn(); syncClock();
    if (announce) { var live = $('#route-live'); if (live) live.textContent = 'Theme: ' + THEME_NAME[v]; }
    try { document.dispatchEvent(new CustomEvent('fnp-theme', { detail: v })); } catch (e) {}
  }
  sPanel.addEventListener('change', function (e) {
    var el = e.target; if (!el || el.type !== 'radio') return;
    if (el.name === 's-font') { root.setAttribute('data-font', el.value); store('fnp-font', el.value); }
    if (el.name === 's-theme') setTheme(el.value, true);
  });
  $('#theme-quick').addEventListener('click', function () { setTheme(nextTheme(curTheme()), true); });
  document.addEventListener('visibilitychange', syncClock);
  addPop($('#settings-btn'), sPanel, syncSettings, 'input:checked');
  addPop($('#help-btn'), $('#help-pop'));
  addPop($('#status-btn'), $('#status-pop'));
  syncSettings(); syncThemeBtn(); syncClock();

  /* ---------- term tooltip: hover, focus or tap on any .term shows its glossary definition ---------- */
  var tip = $('#term-tip'), tipFor = null;
  function showTip(el) {
    var g = L.glossaryEntry(el.getAttribute('data-term')); if (!g) return;
    hideTip();
    tip.innerHTML = '<b>' + esc(g.label) + '</b>' + esc(g.definition); tip.hidden = false; tipFor = el;
    el.setAttribute('aria-describedby', 'term-tip');
    placeTip();
  }
  function placeTip() {
    if (!tipFor) return;
    var r = tipFor.getBoundingClientRect(), w = tip.offsetWidth, h = tip.offsetHeight;
    var left = Math.max(8, Math.min(r.left, window.innerWidth - w - 8)), top = r.bottom + 6;
    if (top + h > window.innerHeight - 8) top = Math.max(8, r.top - h - 6);
    tip.style.left = left + 'px'; tip.style.top = top + 'px';
  }
  function hideTip() { if (tipFor) { tipFor.removeAttribute('aria-describedby'); tipFor = null; } tip.hidden = true; }
  document.addEventListener('mouseover', function (e) { var t = e.target.closest && e.target.closest('.term'); if (t && t !== tipFor) showTip(t); });
  document.addEventListener('mouseout', function (e) { var t = e.target.closest && e.target.closest('.term'); if (t && !(e.relatedTarget && t.contains(e.relatedTarget)) && document.activeElement !== t) hideTip(); });
  document.addEventListener('focusin', function (e) { var t = e.target.closest && e.target.closest('.term'); if (t) showTip(t); else hideTip(); });
  document.addEventListener('focusout', function (e) { var t = e.target.closest && e.target.closest('.term'); if (t) hideTip(); });
  window.addEventListener('scroll', placeTip, true);
  window.addEventListener('resize', placeTip);

  /* ---------- global search ---------- */
  var gq = $('#gs-q'), gl = $('#gs-list'), gseq = 0, gtimer = null, gopts = [], gact = -1;
  function closeSearch() {
    gl.hidden = true; gq.setAttribute('aria-expanded', 'false'); gq.removeAttribute('aria-activedescendant'); gact = -1; gseq++; clearTimeout(gtimer);
  }
  function setActive(i) {
    var els = $$('li', gl); gact = i;
    els.forEach(function (li, j) { li.setAttribute('aria-selected', j === i); li.classList.toggle('on', j === i); });
    if (i >= 0 && els[i]) { gq.setAttribute('aria-activedescendant', els[i].id); els[i].scrollIntoView({ block: 'nearest' }); } else gq.removeAttribute('aria-activedescendant');
  }
  function renderSearch(d, q) {
    gopts = d.items || [];
    var h = gopts.map(function (it, i) {
      var v = it.verdict_norm || 'UNRATED';
      return '<li role="option" id="gs-o-' + i + '" data-id="' + esc(it.id) + '" aria-selected="false"><span class="gs-t">' + esc(it.title || it.claim || it.id) + '</span>' +
        '<span class="gs-m"><span class="chip v-' + esc(v) + '">' + esc(L.verdict(v)) + '</span>' + esc(L.site(it.site)) + (it.published ? ' - ' + esc(L.dateLong(it.published)) : '') + '</span></li>';
    }).join('');
    if (!gopts.length) h = '<li class="gs-none" role="presentation">No claims match &ldquo;' + esc(q) + '&rdquo;. Try fewer or different words.</li>';
    else h += '<li role="option" id="gs-o-all" data-all="1" aria-selected="false" class="gs-all">See all ' + esc(fmtN(d.total)) + ' result' + (d.total === 1 ? '' : 's') + ' in the Desk</li>';
    gl.innerHTML = h; gl.hidden = false; gq.setAttribute('aria-expanded', 'true'); gact = -1;
  }
  gq.addEventListener('input', function () {
    var q = gq.value.trim(); clearTimeout(gtimer);
    if (q.length < 2) { closeSearch(); return; }
    var my = ++gseq;
    gtimer = setTimeout(function () {
      getJSON('/api/items?q=' + encodeURIComponent(q) + '&page_size=5&sort=plaus').then(function (d) { if (my === gseq) renderSearch(d, q); })
        .catch(function (e) { if (my === gseq) { gl.innerHTML = '<li class="gs-none" role="presentation">Search is unavailable right now. Try again in a moment.</li>'; gl.hidden = false; gq.setAttribute('aria-expanded', 'true'); } });
    }, 200);
  });
  function pick(li) {
    if (!li) return;
    var q = gq.value.trim();
    if (li.getAttribute('data-all')) { closeSearch(); if (window.Desk) window.Desk.search(q); navigate('#desk'); return; }
    var id = li.getAttribute('data-id'); closeSearch(); gq.blur();
    if (id) navigate('#desk/' + encodeURIComponent(id));
  }
  gq.addEventListener('keydown', function (e) {
    var els = $$('li[role="option"]', gl);
    if (e.key === 'ArrowDown' && els.length) { e.preventDefault(); if (gl.hidden) { gl.hidden = false; gq.setAttribute('aria-expanded', 'true'); } setActive((gact + 1) % els.length); }
    else if (e.key === 'ArrowUp' && els.length) { e.preventDefault(); setActive((gact - 1 + els.length) % els.length); }
    else if (e.key === 'Enter') {
      e.preventDefault();
      if (gact >= 0 && els[gact]) pick(els[gact]);
      else if (gq.value.trim().length >= 2) { var q = gq.value.trim(); closeSearch(); if (window.Desk) window.Desk.search(q); navigate('#desk'); }
    }
  });
  gl.addEventListener('mousedown', function (e) { var li = e.target.closest('li[role="option"]'); if (li) { e.preventDefault(); pick(li); } });
  gq.addEventListener('focus', function () { if (gopts.length && gq.value.trim().length >= 2 && gl.innerHTML) { gl.hidden = false; gq.setAttribute('aria-expanded', 'true'); } });

  /* Escape: tooltip, then dropdown, then popovers. Runs first so the drawer's own Escape handler is not also triggered. */
  document.addEventListener('keydown', function (e) {
    if (e.key !== 'Escape') return;
    if (!tip.hidden) { hideTip(); e.stopPropagation(); return; }
    if (!gl.hidden) { closeSearch(); e.stopPropagation(); return; }
    if (closePops(null, true)) e.stopPropagation();
  }, true);

  /* ---------- freshness, data status, footer ---------- */
  function ymd(s) { var d = new Date(s); return isNaN(d) ? null : d; }
  function row(k, v) { return '<dt>' + esc(k) + '</dt><dd>' + v + '</dd>'; }
  function renderStatus(st, lab, latest) {
    var eng = (st && st.meta && st.meta.engine) || {}, built = st && st.meta ? ymd(st.meta.built) : null;
    var dts = [lab && ymd(lab.data_through), latest && ymd(latest)].filter(Boolean);
    var through = dts.length ? new Date(Math.min.apply(null, dts.map(Number))) : null;
    var fresh = built && Date.now() - built.getTime() < 72 * 3600 * 1000;
    $('#fresh-text').textContent = through ? 'Data through ' + L.dateLong(through) : built ? 'Index built ' + L.dateLong(built) : 'Data status';
    $('#status-btn').setAttribute('aria-label', 'Data status: ' + $('#fresh-text').textContent);
    $('#fresh-dot').className = 'dot ' + (st ? (fresh ? 'ok' : 'wait') : '');
    var h = '';
    if (st) {
      h += row('Fact-check index', esc(fmtN(st.total)) + ' items from ' + esc(fmtN(Object.keys(st.sites).length)) + ' sources');
      h += row('Index rebuilt', esc(built ? L.dateTimeUTC(built) : 'unknown'));
      if (latest) h += row('Newest claim published', esc(L.dateLong(latest)));
      h += row('Engine scores', eng.available ? 'Loaded for ' + esc(fmtN(eng.matched)) + ' items' : 'Pending: scores use fact-checker labels only');
      h += row('Text matching', 'TF-IDF (' + esc(st.meta.tfidf_backend || 'unknown') + ')');
    } else h += row('Fact-check index', 'Could not be loaded');
    if (lab) {
      h += row('Propaganda Lab', esc(fmtN(lab.n_posts)) + ' posts through ' + esc(L.dateLong(lab.data_through)));
      h += row('Lab rebuilt', esc(L.dateTimeUTC(lab.built_at)));
      if (!(lab.n_articles > 0)) h += row('News source', 'Offline: Lab covers social posts only');
    } else h += row('Propaganda Lab', 'Not built yet');
    $('#status-body').innerHTML = h;
    if (st) {
      $('#foot-meta').textContent = fmtN(st.total) + ' items from ' + fmtN(Object.keys(st.sites).length) + ' sources' + (built ? ' - rebuilt ' + L.dateLong(built) : '') +
        (eng.available ? ' - scores blend fact-checker labels with the verdict engine' : ' - scores from fact-checker labels only');
      $('#side-total').textContent = fmtN(st.total) + ' items';
      $('#side-note').textContent = (built ? 'Rebuilt ' + L.dateLong(built) + '. ' : '') + (eng.available ? 'Engine scores loaded.' : 'Engine scores pending.');
      $('#side-dot').className = 'dot ' + (eng.available ? 'ok' : 'wait');
    } else {
      $('#foot-meta').textContent = 'Fact-check index unavailable'; $('#side-note').textContent = 'Unavailable';
    }
  }

  function start() {
    if (started) return; started = true;
    window.addEventListener('hashchange', route);
    route();
    Promise.all([getStats().catch(function () { return null; }), getLabMeta(), getLatest()]).then(function (r) { renderStatus(r[0], r[1], r[2]); });
  }

  window.Shell = {
    register: function (r, h) { handlers[r] = h; }, on: function (f) { subs.push(f); }, start: start, navigate: navigate, current: function () { return cur; },
    closePopovers: function () { closePops(null, false); }, addPop: addPop, setTitle: setTitle, getJSON: getJSON, getStats: getStats, getLabMeta: getLabMeta, getLatest: getLatest, hideTip: hideTip
  };
})();
