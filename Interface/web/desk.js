/* Desk: filterable claim list (infinite scroll over /api/items) beside the case file. The hash #desk/<id> is the single source of
   truth for the open item, so browser Back, deep links and the mobile Back button all work from it. Filters live in memory and survive
   leaving and re-entering the view. */
(function () {
  'use strict';
  var $ = function (s, r) { return (r || document).querySelector(s); };
  var L = window.Labels, F = window.FNPUI, C = window.Case, esc = L.esc, fmtN = F.fmtN;
  var PAGE = 40, MOBILE = window.matchMedia('(max-width: 959px)');
  var VCHIPS = [['', 'All'], ['FAKE', 'Fake'], ['MISLEADING', 'Misleading'], ['LEGIT', 'Legit'], ['INCONCLUSIVE', 'Inconclusive'], ['UNRATED', 'Unrated']];
  var DEFAULTS = { q: '', verdicts: ['FAKE', 'MISLEADING'], lang: '', site: '', sort: 'plaus', country: '', tier: '', channel: '', from: '', to: '', min: 0 };
  var S = JSON.parse(JSON.stringify(DEFAULTS));
  var groups = { site: {}, lang: {} }, stats = null, ready = null;
  var list = { page: 0, pages: 1, total: null, loading: false, seq: 0, shown: 0, error: null };
  var sel = null, caseSeq = 0, routeCount = 0, wasShown = false, listScroll = 0, entered = false, timer = null, cache = {};

  var el = {
    desk: $('#desk'), scroller: $('#desk-list'), rows: $('#rows'), foot: $('#rows-foot'), count: $('#dl-count'), active: $('#dl-active'), chips: $('#vchips'),
    q: $('#f-q'), lang: $('#f-lang'), site: $('#f-site'), sort: $('#f-sort'), more: $('#more-btn'), pop: $('#more-pop'), detail: $('#desk-detail'),
    empty: $('#case-empty'), body: $('#case-body'), live: $('#desk-live'), head: $('#dl-head')
  };

  /* ---------- filter state <-> DOM ---------- */
  function codes(kind, name) { return name && groups[kind][name] ? groups[kind][name] : ''; }
  function apiParams(page) {
    var p = new URLSearchParams();
    var set = function (k, v) { if (v) p.set(k, v); };
    set('q', S.q.trim()); set('verdict', S.verdicts.join(',')); set('site', codes('site', S.site).toLowerCase()); set('lang', codes('lang', S.lang).toLowerCase());
    set('country', S.country); set('tier', S.tier); set('channel', S.channel); set('from', S.from); set('to', S.to);
    if (S.min > 0) p.set('min_plaus', S.min);
    p.set('sort', S.sort); p.set('page', page); p.set('page_size', PAGE);
    return p.toString();
  }
  function moreCount() { return ['country', 'tier', 'channel', 'from', 'to'].filter(function (k) { return S[k]; }).length + (S.min > 0 ? 1 : 0); }
  function chipCounts() {
    var vs = (stats && stats.verdicts) || {}, all = stats ? stats.total : 0;
    return function (v) { return v ? vs[v] || 0 : all; };
  }
  function compact(n) { return n >= 1000 ? (n / 1000).toFixed(n >= 100000 ? 0 : 1).replace(/\.0$/, '') + 'k' : String(n); }
  function renderChips() {
    var cnt = chipCounts();
    el.chips.innerHTML = VCHIPS.map(function (c) {
      var on = c[0] === '' ? S.verdicts.length === 0 : S.verdicts.indexOf(c[0]) >= 0;
      return '<button type="button" class="vchip ' + (c[0] ? 'v-' + c[0] : 'all') + '" data-v="' + c[0] + '" aria-pressed="' + on + '"' + (c[0] === 'UNRATED' ? ' title="Social posts no fact-checker has rated yet"' : '') + '><i aria-hidden="true"></i>' + esc(c[1]) +
        (stats ? ' <small title="' + fmtN(cnt(c[0])) + ' items">' + compact(cnt(c[0])) + '</small>' : '') + '</button>';
    }).join('');
  }
  function fchip(key, text) { return '<button type="button" class="fchip" data-clear="' + key + '"><span>' + esc(text) + '</span><b aria-hidden="true">&times;</b><span class="sr">Remove filter</span></button>'; }
  function renderActive() {
    var h = '', n = 0, add = function (k, t) { h += fchip(k, t); n++; };
    if (S.q.trim()) add('q', 'Search: ' + S.q.trim());
    if (S.lang) add('lang', S.lang);
    if (S.site) add('site', S.site);
    if (S.country) add('country', L.country(S.country));
    if (S.tier) add('tier', L.tier(S.tier));
    if (S.channel) add('channel', L.channel(S.channel));
    if (S.from) add('from', 'From ' + L.dateLong(S.from));
    if (S.to) add('to', 'To ' + L.dateLong(S.to));
    if (S.min > 0) add('min', 'Score ' + S.min.toFixed(2) + ' or more');
    var changed = JSON.stringify(S) !== JSON.stringify(DEFAULTS);
    el.count.textContent = list.total == null ? 'Loading' : fmtN(list.total) + ' claim' + (list.total === 1 ? '' : 's');
    el.active.innerHTML = h + (changed ? '<button type="button" class="fchip clear" data-clear="all"><span>Reset filters</span></button>' : '');
    var nm = moreCount(); el.more.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" aria-hidden="true"><path d="M4 7h10M18 7h2M4 17h2M10 17h10"/><circle cx="16" cy="7" r="2"/><circle cx="8" cy="17" r="2"/></svg>More' + (nm ? '<span class="n" aria-hidden="true">' + nm + '</span><span class="sr"> (' + nm + ' active)</span>' : '') + '<span class="sr"> filters</span>';
  }
  function syncControls() {
    el.q.value = S.q; el.lang.value = S.lang; el.site.value = S.site; el.sort.value = S.sort;
    $('#f-country').value = S.country; $('#f-tier').value = S.tier; $('#f-channel').value = S.channel; $('#f-from').value = S.from; $('#f-to').value = S.to;
    $('#f-min').value = S.min; $('#o-min').textContent = S.min.toFixed(2);
    renderChips(); renderActive();
  }
  function changed() { renderChips(); renderActive(); schedule(); }
  function schedule() { clearTimeout(timer); timer = setTimeout(function () { reload(); }, 120); }

  /* ---------- list ---------- */
  function rowHTML(it) {
    var v = it.verdict_norm || 'UNRATED', t = String(it.title || it.claim || it.id).replace(/\s+/g, ' ').trim();
    return '<li class="row s-' + esc(v) + '" role="option" data-id="' + esc(it.id) + '" tabindex="-1" aria-selected="' + (it.id === sel) + '"' + (it.id === sel ? ' aria-current="true"' : '') + '>' +
      '<span class="row-title">' + esc(t) + '</span>' +
      '<span class="row-meta"><span class="row-v">' + esc(L.verdict(v)) + '</span><span class="row-src">' + esc(L.site(it.site)) + '</span><span aria-hidden="true">&middot;</span><span>' + esc(L.lang(it.lang)) + '</span><span aria-hidden="true">&middot;</span><span>' + esc(it.published ? L.dateLong(it.published) : 'undated') + '</span></span>' +
      '<span class="row-score">' + F.meter(it.plaus) + '</span></li>';
  }
  function skeletonRows() { var r = ''; for (var i = 0; i < 8; i++) r += '<li class="row sk" aria-hidden="true"><span class="skel w1"></span><span class="skel w2"></span><span class="skel w3"></span></li>'; return r; }
  function setFoot(html) { el.foot.innerHTML = html; el.foot.hidden = !html; }
  function footState() {
    if (list.error) { setFoot('<p>Could not load more claims. Check your connection and try again.</p><button type="button" class="btn ghost" data-act="more-retry">Try again</button>'); return; }
    if (list.loading) { setFoot('<span class="spin" aria-hidden="true"></span>Loading more claims'); return; }
    if (list.total === 0) { setFoot(''); return; }
    if (list.page >= list.pages) { setFoot('End of list. ' + fmtN(list.total) + ' claim' + (list.total === 1 ? '' : 's') + ' shown.'); return; }
    setFoot('<span class="spin" aria-hidden="true"></span>Loading more claims'); /* sentinel: scrolled into view, the next page loads */
  }
  function fetchPage(page, replace, seq) {
    list.loading = true; list.error = null;
    el.rows.setAttribute('aria-busy', 'true');
    if (!el.rows.children.length) el.rows.innerHTML = skeletonRows();
    footState();
    F.api('/api/items?' + apiParams(page)).then(function (d) {
      if (seq !== list.seq) return;
      list.page = d.page; list.pages = d.pages; list.total = d.total; list.loading = false;
      if (replace) { el.rows.innerHTML = ''; el.scroller.scrollTop = 0; }
      if (!d.items.length && replace) {
        el.rows.innerHTML = '';
        el.foot.hidden = true;
        showEmpty();
      } else {
        hideEmpty();
        el.rows.insertAdjacentHTML('beforeend', d.items.map(rowHTML).join(''));
        if (replace) C.setCursor(el.rows, el.rows.querySelector('.row'), false);
      }
      el.rows.removeAttribute('aria-busy'); renderActive(); footState(); applyHeadPad(); watch();
    }).catch(function (e) {
      if (seq !== list.seq) return;
      list.loading = false; list.error = e.message; el.rows.removeAttribute('aria-busy');
      if (replace) { el.rows.innerHTML = ''; showEmpty('The claim list could not be loaded', 'Check your connection, then try again.', true); setFoot(''); } else footState();
      renderActive();
    });
  }
  function showEmpty(title, text, retry) {
    var box = $('#rows-empty');
    box.innerHTML = '<b>' + esc(title || 'No claims match these filters') + '</b><span>' + esc(text || 'Try removing a filter or searching for fewer words.') + '</span><button type="button" class="btn ghost" data-clear="all" ' + (retry ? 'data-act="more-retry"' : '') + '>' + (retry ? 'Try again' : 'Reset filters') + '</button>';
    box.hidden = false;
  }
  function hideEmpty() { $('#rows-empty').hidden = true; }
  function reload() { list.seq++; list.page = 0; list.pages = 1; list.total = null; renderActive(); fetchPage(1, true, list.seq); }
  function loadMore() {
    if (list.loading || list.error || list.page >= list.pages || list.total === 0) return;
    fetchPage(list.page + 1, false, list.seq);
  }
  /* load the next page when the footer nears the bottom of the scroller; checked again after each append so short pages cannot stall */
  var io = null;
  function watch() {
    if (!('IntersectionObserver' in window)) { el.scroller.onscroll = function () { if (nearEnd()) loadMore(); }; }
    else if (!io) { io = new IntersectionObserver(function (es) { if (es[0].isIntersecting) loadMore(); }, { root: el.scroller, rootMargin: '0px 0px 700px 0px' }); }
    if (io) { io.disconnect(); if (!el.foot.hidden) io.observe(el.foot); }
    requestAnimationFrame(function () { if (nearEnd()) loadMore(); });
  }
  function nearEnd() { return !el.foot.hidden && el.scroller.offsetParent && el.foot.getBoundingClientRect().top < el.scroller.getBoundingClientRect().bottom + 700; }
  function applyHeadPad() { el.scroller.style.scrollPaddingTop = el.head.offsetHeight + 'px'; }

  function markSelected() {
    C.rows(el.rows).forEach(function (r) {
      var on = r.getAttribute('data-id') === sel;
      r.setAttribute('aria-selected', on); if (on) r.setAttribute('aria-current', 'true'); else r.removeAttribute('aria-current');
    });
  }
  function selectedRow() { return C.rows(el.rows).filter(function (r) { return r.getAttribute('data-id') === sel; })[0] || null; }

  /* ---------- case file ---------- */
  function pane(name) { el.desk.setAttribute('data-pane', name); }
  function announce(t) { el.live.textContent = ''; setTimeout(function () { el.live.textContent = t; }, 30); }
  function showBody(html) { el.empty.hidden = true; el.body.hidden = false; el.body.innerHTML = html; el.detail.scrollTop = 0; }
  function clearCase() { caseSeq++; if (!MOBILE.matches) sel = null; markSelected(); /* on small screens the row that was open stays marked, so Back lands where you left */ el.body.hidden = true; el.body.innerHTML = ''; el.empty.hidden = false; pane('list'); }
  function openCase(id, focusTitle) {
    sel = id; markSelected(); pane('case');
    var my = ++caseSeq;
    var done = function (it) {
      if (my !== caseSeq) return;
      showBody(C.item(it)); Shell.setTitle(it.title || it.claim || '');
      announce('Case file opened: ' + (it.title || it.claim || ''));
      if (focusTitle && MOBILE.matches) { var t = $('#case-title'); if (t) t.focus({ preventScroll: true }); }
    };
    if (cache[id]) { done(cache[id]); return; }
    showBody(C.skeleton()); el.detail.setAttribute('aria-busy', 'true');
    F.api('/api/item/' + encodeURIComponent(id)).then(function (it) { cache[id] = it; el.detail.removeAttribute('aria-busy'); done(it); })
      .catch(function (e) { el.detail.removeAttribute('aria-busy'); if (my === caseSeq) showBody(C.error('Check your connection, then try again.', id)); });
  }
  function goBack() {
    if (routeCount > 1 && history.length > 1) history.back(); else Shell.navigate('#desk', true);
  }
  function openRow(row) { Shell.navigate('#desk/' + encodeURIComponent(row.getAttribute('data-id'))); }

  el.detail.addEventListener('click', function (e) {
    if (C.onDetailClick(e)) return;
    var b = e.target.closest('[data-act]'); if (!b) return;
    if (b.getAttribute('data-act') === 'back') goBack();
    else if (b.getAttribute('data-act') === 'retry') openCase(b.getAttribute('data-id'), false);
  });

  /* ---------- events ---------- */
  el.rows.addEventListener('click', function (e) { var r = e.target.closest('.row[data-id]'); if (r) { C.setCursor(el.rows, r, false); openRow(r); } });
  C.rowNav(el.rows, openRow);
  el.chips.addEventListener('click', function (e) {
    var b = e.target.closest('.vchip'); if (!b) return;
    var v = b.getAttribute('data-v');
    if (v === '') S.verdicts = []; else { var i = S.verdicts.indexOf(v); if (i >= 0) S.verdicts.splice(i, 1); else S.verdicts.push(v); }
    changed();
  });
  var qt = null;
  el.q.addEventListener('input', function () { S.q = el.q.value; clearTimeout(qt); qt = setTimeout(changed, 220); });
  el.q.addEventListener('keydown', function (e) { if (e.key === 'Escape' && el.q.value) { el.q.value = ''; S.q = ''; changed(); e.stopPropagation(); } });
  [['lang', el.lang], ['site', el.site], ['sort', el.sort]].forEach(function (p) { p[1].addEventListener('change', function () { S[p[0]] = p[1].value; changed(); }); });
  $('#more-pop').addEventListener('input', function (e) {
    var id = e.target.id; if (!id) return;
    if (id === 'f-min') { S.min = +e.target.value; $('#o-min').textContent = S.min.toFixed(2); } else S[id.replace('f-', '')] = e.target.value;
    changed();
  });
  $('#more-reset').addEventListener('click', function () { resetAll(); });
  function resetAll() { S = JSON.parse(JSON.stringify(DEFAULTS)); syncControls(); schedule(); }
  el.active.addEventListener('click', function (e) {
    var b = e.target.closest('[data-clear]'); if (!b) return;
    var k = b.getAttribute('data-clear');
    if (k === 'all') { resetAll(); return; }
    S[k] = k === 'min' ? 0 : ''; syncControls(); schedule();
  });
  $('#rows-empty').addEventListener('click', function (e) {
    var b = e.target.closest('button'); if (!b) return;
    if (b.getAttribute('data-act') === 'more-retry') { reload(); } else if (b.getAttribute('data-clear') === 'all') resetAll();
  });
  el.foot.addEventListener('click', function (e) { if (e.target.closest('[data-act="more-retry"]')) { list.error = null; loadMore(); } });

  /* More filters popover */
  function popOpen(open) {
    el.pop.hidden = !open; el.more.setAttribute('aria-expanded', open ? 'true' : 'false');
    if (open) { var f = $('#f-country'); if (f) f.focus(); }
  }
  el.more.addEventListener('click', function () { popOpen(el.pop.hidden); });
  document.addEventListener('click', function (e) { if (!el.pop.hidden && !e.target.closest('#more-pop') && !e.target.closest('#more-btn')) popOpen(false); });
  el.pop.addEventListener('focusout', function (e) { var n = e.relatedTarget; if (n && !el.pop.contains(n) && n !== el.more) popOpen(false); });

  /* keyboard: j/k from anywhere on the Desk (outside form fields), Escape on small screens returns to the list */
  document.addEventListener('keydown', function (e) {
    if (Shell.current().route !== 'desk') return;
    if (e.key === 'Escape') {
      if (!el.pop.hidden) { popOpen(false); el.more.focus(); e.stopPropagation(); return; }
      if (MOBILE.matches && sel) { e.preventDefault(); goBack(); }
      return;
    }
    if ((e.key !== 'j' && e.key !== 'k') || e.ctrlKey || e.metaKey || e.altKey) return;
    var t = e.target, tag = t && t.tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || (t && t.isContentEditable) || (t && t.closest && t.closest('.desk-detail, #more-pop'))) return;
    if (t && t.closest && t.closest('#rows')) return; /* the list handles its own keys */
    e.preventDefault();
    var from = selectedRow() || C.rows(el.rows)[0];
    if (from) { if (!selectedRow()) C.setCursor(el.rows, from, true); else C.move(el.rows, from, e.key === 'j' ? 1 : -1); }
  });

  /* ---------- boot ---------- */
  function groupOptions(counts, nameOf) {
    var g = {};
    Object.keys(counts || {}).forEach(function (c) { var n = nameOf(c); (g[n] = g[n] || { name: n, codes: [], n: 0 }); g[n].codes.push(c); g[n].n += counts[c]; });
    return Object.keys(g).map(function (k) { return g[k]; }).sort(function (a, b) { return b.n - a.n; });
  }
  function fillSelect(sel, gs, key) {
    gs.forEach(function (g) {
      groups[key][g.name] = g.codes.join(',');
      sel.insertAdjacentHTML('beforeend', '<option value="' + esc(g.name) + '" title="' + fmtN(g.n) + ' items">' + esc(g.name) + '</option>');
    });
  }
  function fillMore(id, counts, nameOf) {
    Object.keys(counts || {}).forEach(function (c) { $(id).insertAdjacentHTML('beforeend', '<option value="' + esc(c) + '">' + esc(nameOf(c)) + ' (' + fmtN(counts[c]) + ')</option>'); });
  }
  ready = Shell.getStats().then(function (st) {
    stats = st;
    fillSelect(el.lang, groupOptions(st.languages, L.lang), 'lang');
    fillSelect(el.site, groupOptions(st.sites, L.site), 'site');
    fillMore('#f-channel', st.channels, L.channel); fillMore('#f-country', st.countries, L.country); fillMore('#f-tier', st.source_tiers, L.tier);
    syncControls();
  }).catch(function () { stats = null; renderChips(); renderActive(); });

  Shell.on(function () { routeCount++; });
  Shell.register('desk', {
    enter: function (id) {
      var first = !entered; entered = true;
      var go = function () {
        if (first) reload();
        else if (listScroll) el.scroller.scrollTop = listScroll;
        applyHeadPad();
        if (id) openCase(id, true); else clearCase();
        if (first) watch();
        else { markSelected(); }
      };
      if (first) { renderChips(); renderActive(); ready.then(go); } else go();
    },
    leave: function () { listScroll = el.scroller.scrollTop; popOpen(false); }
  });
  window.addEventListener('resize', applyHeadPad);

  /* return focus to the row that was open when going back to the list on small screens */
  Shell.on(function (r) {
    if (r.route === 'desk' && !r.arg && wasShown && MOBILE.matches) { var row = selectedRow(); if (row) setTimeout(function () { row.focus({ preventScroll: true }); }, 0); }
    wasShown = r.route === 'desk' && !!r.arg;
  });

  /* helpers other views use (overview.js: verdict and language shortcuts, global search: Show all) */
  function setVerdicts(v) { S.verdicts = v ? String(v).split(',').filter(Boolean) : []; }
  window.Desk = {
    filter: function (kind, value) {
      ready.then(function () {
        if (kind === 'verdict') setVerdicts(value);
        if (kind === 'lang') S.lang = groups.lang[value] !== undefined ? value : '';
        syncControls(); schedule();
      });
    },
    search: function (q) { S.q = q; S.verdicts = []; ready.then(function () { syncControls(); schedule(); }); }
  };
})();
