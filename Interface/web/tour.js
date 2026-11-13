/* First-visit guided tour (7 steps) and the Glossary panel. Both open from the Help button.
   The tour is a small non-blocking card: nothing is dimmed, the page stays usable, Escape or Skip closes it, and it is shown once
   (remembered in localStorage inside try/catch; without storage it still works, it just may show again on the next visit). */
(function () {
  'use strict';
  var $ = function (s, r) { return (r || document).querySelector(s); };
  var L = window.Labels, esc = L.esc, Shell = window.Shell;
  var KEY = 'fnp-tour';
  var STEPS = [
    { route: 'desk', target: '#desk-list', side: 'right', title: 'Start with the queue',
      body: 'Every fact-checked claim is listed here, highest plausibility score first. Use the verdict chips and the language and source menus to narrow the list, More for dates and score, and j and k to move down it.' },
    { route: 'desk', target: '#desk-detail', side: 'left', title: 'Open a case file',
      body: 'Select a claim and its case file opens here: the verdict and score, why other fact-checkers agree, the nearest earlier fact-checks and the original text. Terms marked with a ? have plain-language definitions.' },
    { route: 'check', target: '#det-list', side: 'right', title: 'Check your own text',
      body: 'Paste a claim, headline or post into Claim Check, or pick an example. Step 1 says whether it looks fake, from the fact-check index and a verdict engine. Step 2 is always shown: a bar for each of the 14 propaganda techniques and the words highlighted. A new claim takes 20 to 35 seconds; repeats are instant.' },
    { route: 'lab', target: '#lab-stage', side: 'left', title: 'Explore the Propaganda Lab',
      body: 'The map shows where propaganda techniques appear most, country by country. Pick a metric above the map, open Filters for time window and source, and select a country for its narratives and posts.' },
    { route: 'evidence', hash: '#evidence/overview', target: '#ev-main', side: 'right', title: 'Browse the live posts and articles',
      body: 'The Evidence Base is a separate source: the posts and articles in the Realtime Database, not the Fake News Database. Overview counts them by day, platform, source, topic and language; Browse opens each one. Every field says whether it comes from the database or was inferred by FNP (a fast batch pass over all items, or a deep Claim Check verdict that a scheduled run adds every few hours), and items FNP has not analysed say so. The banner shows when the last deep run happened and when the next one is due. It needs a login.' },
    { route: 'triage', hash: '#triage/fnp', target: '#trb-list', side: 'right', title: 'Read the highest tier first',
      body: 'Triage ranks what to read first. Switch between the FNP triage export (posts FNP already examined) and the Realtime Database queue (stored items ranked by FNP\'s signals). Each case file starts with why the item was flagged, with the numbers. The banner above says how many items a person has validated; today that is none, so a tier is a lead, never a verdict.' },
    { route: 'evidence', hash: '#evidence/workbench', target: '#wb-find', side: 'right', title: 'Run Claim Check in the Workbench',
      body: 'Find items on the left and add up to 25 to the selection. Run Claim Check checks them one at a time, about 15 to 40 seconds each, with progress and a Cancel button. Each result shows the fake-or-not verdict and a bar for all 14 propaganda techniques. Insights then describes the selection using only numbers from the database.' }
  ];
  var tour = null, step = 0, open = false, returnTo = null, startHash = '', moved = false, autoTried = false, hlEl = null, seq = 0;

  function seen() { try { return localStorage.getItem(KEY) === 'done'; } catch (e) { return false; } }
  function remember() { try { localStorage.setItem(KEY, 'done'); } catch (e) {} }

  /* ---------- tour card ---------- */
  function build() {
    if (tour) return;
    tour = document.createElement('div');
    tour.className = 'tour'; tour.id = 'tour'; tour.hidden = true; tour.tabIndex = -1;
    tour.setAttribute('role', 'dialog'); tour.setAttribute('aria-labelledby', 'tour-h'); tour.setAttribute('aria-describedby', 'tour-b'); tour.setAttribute('aria-label', 'Guided tour');
    tour.innerHTML = '<div class="tour-top"><span class="lbl" id="tour-n"></span><span class="tour-dots" aria-hidden="true"></span></div>' +
      '<h2 id="tour-h" tabindex="-1"></h2><p class="tour-body" id="tour-b"></p>' +
      '<div class="tour-foot"><button type="button" class="btn ghost" id="tour-back">Back</button><button type="button" class="btn" id="tour-next">Next</button><button type="button" class="tour-skip" id="tour-skip">Skip tour</button></div>';
    document.body.appendChild(tour);
    $('#tour-back').addEventListener('click', function () { if (step > 0) show(step - 1); });
    $('#tour-next').addEventListener('click', function () { if (step < STEPS.length - 1) show(step + 1); else close(); });
    $('#tour-skip').addEventListener('click', function () { close(); });
  }
  function clearHl() { if (hlEl) { hlEl.classList.remove('tour-hl'); hlEl = null; } }
  function visible(el) { return el && el.getClientRects().length > 0 && getComputedStyle(el).visibility !== 'hidden'; }
  function highlight(sel, my, tries) {
    if (my !== seq || !open) return;
    var el = $(sel);
    if (visible(el)) { clearHl(); el.classList.add('tour-hl'); hlEl = el; place(); return; }
    if (tries < 15) setTimeout(function () { highlight(sel, my, tries + 1); }, 120);
  }
  /* Put the card in the viewport corner where it overlaps the highlighted element least (pure logic in FNPPure.placeCard),
     so it never covers the thing it is describing. Re-run when the target appears and when the window changes size. */
  function place() {
    if (!tour || tour.hidden || !window.FNPPure) return;
    var st = STEPS[step], t = st && $(st.target), r = t && visible(t) ? t.getBoundingClientRect() : null;
    var mobile = window.matchMedia('(max-width: 959px)').matches, cs = getComputedStyle(document.documentElement);
    var px = function (v, d) { var n = parseFloat(v); return isNaN(n) ? d : n; };
    var navW = mobile ? 0 : px(cs.getPropertyValue('--nav-w'), 0), topMin = px(cs.getPropertyValue('--top-h'), 60) + 12;
    var bnav = px(cs.getPropertyValue('--bnav-h'), 56);
    var vw = document.documentElement.clientWidth, vh = window.innerHeight;
    tour.style.left = tour.style.top = '0px'; tour.style.right = tour.style.bottom = 'auto';
    var card = tour.getBoundingClientRect();
    var pos = FNPPure.placeCard({ target: r, card: { w: card.width, h: card.height }, vw: vw, vh: vh, side: st.side, navW: navW, margin: mobile ? px(cs.getPropertyValue('--gutter'), 16) : 24, topMin: topMin, bottomMin: mobile ? bnav + 12 : 24, mobile: mobile });
    tour.style.left = pos.left + 'px'; tour.style.top = pos.top + 'px';
    if (pos.w) tour.style.width = pos.w + 'px'; else tour.style.width = '';
  }
  window.addEventListener('resize', function () { if (open) place(); });
  function show(i) {
    build();
    step = i; var st = STEPS[i], my = ++seq;
    clearHl();
    if (Shell.current().route !== st.route || (st.hash && location.hash !== st.hash)) { moved = true; Shell.navigate(st.hash || '#' + st.route, true); } /* replace, so the tour leaves no history entries */
    $('#tour-n').textContent = 'Step ' + (i + 1) + ' of ' + STEPS.length;
    $('.tour-dots', tour).innerHTML = STEPS.map(function (_, j) { return '<i' + (j === i ? ' class="on"' : '') + '></i>'; }).join('');
    $('#tour-h').textContent = st.title; $('#tour-b').textContent = st.body;
    $('#tour-back').disabled = i === 0;
    $('#tour-next').textContent = i === STEPS.length - 1 ? 'Done' : 'Next';
    tour.setAttribute('data-side', st.side);
    tour.hidden = false;
    place();
    $('#tour-h').focus({ preventScroll: true });
    highlight(st.target, my, 0);
  }
  function start(fromHelp) {
    if (open) { show(step); return; }
    closeGlossary(false);
    open = true; returnTo = fromHelp ? $('#help-btn') : document.activeElement; startHash = location.hash || '#overview'; moved = false;
    Shell.closePopovers();
    show(0);
  }
  function close() {
    if (!open) return;
    open = false; seq++; remember(); clearHl();
    if (tour) tour.hidden = true;
    if (moved) Shell.navigate(startHash, true);
    var t = returnTo && document.contains(returnTo) && returnTo !== document.body ? returnTo : $('#main');
    if (t) t.focus({ preventScroll: true });
  }

  /* ---------- glossary panel ---------- */
  var gloss = null, gScrim = null, gReturn = null, gOpen = false;
  function entries() {
    var g = L.glossary;
    return Object.keys(g).map(function (k) { return g[k]; }).sort(function (a, b) { return a.label.localeCompare(b.label, 'en', { sensitivity: 'base' }); });
  }
  function buildGloss() {
    if (gloss) return;
    gScrim = document.createElement('div'); gScrim.className = 'gloss-scrim'; gScrim.hidden = true;
    gloss = document.createElement('aside');
    gloss.className = 'gloss'; gloss.hidden = true; gloss.tabIndex = -1;
    gloss.setAttribute('role', 'dialog'); gloss.setAttribute('aria-modal', 'true'); gloss.setAttribute('aria-labelledby', 'gloss-h');
    gloss.innerHTML = '<div class="gloss-head"><div class="gloss-top"><h2 id="gloss-h" tabindex="-1">Glossary</h2><button type="button" class="btn ghost" id="gloss-x">Close</button></div>' +
      '<p class="gloss-sub">Every term marked with a ? in the workspace, in alphabetical order.</p>' +
      '<label class="sr" for="gloss-q">Filter terms</label><input type="search" id="gloss-q" class="gloss-q" placeholder="Filter terms" autocomplete="off">' +
      '<p class="gloss-count" id="gloss-n" role="status" aria-live="polite"></p></div>' +
      '<div class="gloss-body" id="gloss-body" tabindex="0" role="region" aria-label="Glossary terms"></div>';
    document.body.appendChild(gScrim); document.body.appendChild(gloss);
    gScrim.addEventListener('click', function () { closeGlossary(true); });
    $('#gloss-x').addEventListener('click', function () { closeGlossary(true); });
    $('#gloss-q').addEventListener('input', renderGloss);
    gloss.addEventListener('keydown', function (e) {
      if (e.key !== 'Tab') return;
      var f = [$('#gloss-x'), $('#gloss-q'), $('#gloss-body')], i = f.indexOf(document.activeElement);
      if (e.shiftKey && i <= 0) { e.preventDefault(); f[f.length - 1].focus(); }
      else if (!e.shiftKey && i === f.length - 1) { e.preventDefault(); f[0].focus(); }
    });
  }
  function renderGloss() {
    var q = $('#gloss-q').value.trim().toLowerCase(), all = entries();
    var rows = all.filter(function (g) { return !q || g.label.toLowerCase().indexOf(q) >= 0 || g.definition.toLowerCase().indexOf(q) >= 0; });
    $('#gloss-n').textContent = rows.length + ' of ' + all.length + ' terms';
    $('#gloss-body').innerHTML = rows.length ? '<dl class="gloss-list">' + rows.map(function (g) { return '<div><dt>' + esc(g.label) + '</dt><dd>' + esc(g.definition) + '</dd></div>'; }).join('') + '</dl>'
      : '<p class="gloss-none">No terms match &ldquo;' + esc($('#gloss-q').value.trim()) + '&rdquo;. Try a shorter word.</p>';
  }
  function openGlossary() {
    buildGloss();
    Shell.closePopovers(); Shell.hideTip();
    gReturn = $('#help-btn'); gOpen = true;
    $('#gloss-q').value = ''; renderGloss();
    gScrim.hidden = false; gloss.hidden = false;
    $('#gloss-q').focus();
  }
  function closeGlossary(restore) {
    if (!gOpen) return;
    gOpen = false; gloss.hidden = true; gScrim.hidden = true;
    if (restore && gReturn) gReturn.focus({ preventScroll: true });
  }

  /* Escape: glossary first, then the tour. Capture phase, so the views' own Escape handlers do not also run. */
  document.addEventListener('keydown', function (e) {
    if (e.key !== 'Escape') return;
    if (gOpen) { e.preventDefault(); e.stopPropagation(); closeGlossary(true); return; }
    if (open) { e.preventDefault(); e.stopPropagation(); close(); }
  }, true);

  $('#help-tour').addEventListener('click', function () { start(true); });
  $('#help-gloss').addEventListener('click', openGlossary);

  /* first visit: offer the tour once, a moment after the workspace has drawn. A deep link (a claim or country in the address) is left alone. */
  function auto() {
    if (autoTried || seen()) return; autoTried = true;
    setTimeout(function () { if (!seen() && !open && !Shell.current().arg) start(false); }, 1400);
  }
  auto();

  window.Tour = { start: function () { start(true); }, glossary: openGlossary };
})();
