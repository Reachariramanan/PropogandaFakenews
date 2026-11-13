/* Case file renderer for the Desk (stored items); Claim Check reuses its builders: the same sections in the same order for a stored item and for a pasted claim.
   Pure string builders plus the keyboard helper for row lists. No network calls, no raw site or language codes: everything goes through Labels. */
(function () {
  'use strict';
  var L = window.Labels, F = window.FNPUI, esc = L.esc, term = L.term, fmtN = F.fmtN, band = F.band;
  var ICON_BACK = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m15 5-7 7 7 7"/></svg>';
  var ICON_OUT = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M14 4h6v6M20 4l-9 9M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5"/></svg>';
  var n2 = function (p) { return p == null || isNaN(p) ? 'n/a' : Number(p).toFixed(2); };
  var BAND = { hi: 'high', mid: 'moderate', lo: 'low' };
  var CAVEAT = 'Signals describe style only. Sensational wording is common in false posts but is not proof of falsehood, and calm wording is not proof of truth.';

  /* ---------- small builders ---------- */
  function gauge(score, size, muted) {
    var cx = 120, cy = 110, r = 90, a = Math.PI * (1 - score), x = cx + r * Math.cos(a), y = cy - r * Math.sin(a);
    var tick = function (v) {
      var t = Math.PI * (1 - v), c = Math.cos(t), s = Math.sin(t);
      return '<line class="g-tick" stroke-width="2" x1="' + (cx + (r - 11) * c).toFixed(1) + '" y1="' + (cy - (r - 11) * s).toFixed(1) + '" x2="' + (cx + (r + 11) * c).toFixed(1) + '" y2="' + (cy - (r + 11) * s).toFixed(1) + '"/>';
    };
    return '<svg class="gauge' + (muted ? ' muted' : '') + '" viewBox="0 0 240 150" role="img" aria-label="Plausibility score ' + score.toFixed(2) + ' (' + BAND[band(score)] + ') of being fake or misleading">' +
      '<path class="g-track" d="M30 110 A90 90 0 0 1 210 110" fill="none" stroke-width="16"/>' +
      (score > 0 ? '<path class="g-fill ' + band(score) + '" d="M30 110 A90 90 0 0 1 ' + x.toFixed(1) + ' ' + y.toFixed(1) + '" fill="none" stroke-width="16"/>' : '') +
      tick(0.5) + tick(0.8) +
      '<text class="g-big" x="120" y="104" text-anchor="middle">' + score.toFixed(2) + '</text>' +
      '<text class="g-dim" x="30" y="132" text-anchor="middle">0</text><text class="g-dim" x="210" y="132" text-anchor="middle">1</text>' +
      '<text class="g-dim" x="120" y="146" text-anchor="middle">P(fake or misleading)</text></svg>';
  }
  function section(id, title, sub, body) {
    return '<section class="cs" aria-labelledby="' + id + '"><h3 id="' + id + '">' + esc(title) + '</h3><p class="cs-sub">' + sub + '</p>' + body + '</section>';
  }
  function kv(rows) {
    return '<dl class="kvt">' + rows.filter(Boolean).map(function (r) { return '<dt>' + r[0] + '</dt><dd>' + r[1] + '</dd>'; }).join('') + '</dl>';
  }
  function bar(left, right, backLabel) {
    return '<div class="case-bar"><button type="button" class="btn ghost case-back" data-act="back" aria-label="' + esc(backLabel || 'Back') + '">' + ICON_BACK + 'Back</button>' +
      '<div class="grow">' + (left || '') + '</div>' + (right || '') + '</div>';
  }
  function openLink(url) {
    var u = F.safeUrl(url);
    return u ? '<a class="case-open" href="' + esc(u) + '" target="_blank" rel="noopener noreferrer">Open original' + ICON_OUT + '<span class="sr"> (opens in a new tab)</span></a>' : '';
  }
  function scoreBar(p) {
    return p == null ? '' : '<span class="case-bar-score" title="Plausibility score">' + p.toFixed(2) + '<small>' + BAND[band(p)] + ' plausibility</small></span>';
  }
  function dateOf(s) { return s ? L.dateLong(s) : 'undated'; }
  function simPct(s) { return Math.max(0, Math.min(100, s * 100)).toFixed(0); }

  /* a nearest fact-check: opens its own case file */
  function evCard(m, o) {
    o = o || {};
    var v = m.verdict_norm || 'UNRATED';
    return '<a class="ev-card s-' + esc(v) + '" href="#desk/' + encodeURIComponent(m.id) + '">' +
      '<span class="ev-top">' + F.chip(v) + (m.similarity != null ? '<span class="ev-sim">similarity ' + Number(m.similarity).toFixed(2) + '</span>' : '') + '</span>' +
      '<span class="ev-title">' + esc(m.title || m.claim || m.id) + '</span>' +
      (o.snippet && (m.claim || m.summary) ? '<span class="ev-snip">' + esc(m.claim || m.summary) + '</span>' : '') +
      '<span class="ev-meta">' + esc(L.site(m.site)) + (m.published ? ' - ' + esc(dateOf(m.published)) : '') + '</span>' +
      (m.similarity != null ? '<span class="ev-simbar" aria-hidden="true"><b style="width:' + simPct(m.similarity) + '%"></b></span>' : '') + '</a>';
  }
  function tiles(cells) {
    return '<div class="sig-grid">' + cells.map(function (c) { return '<div class="sig' + (c[2] ? ' flag' : '') + '"><b>' + esc(c[1]) + '</b><span>' + c[0] + '</span></div>'; }).join('') + '</div>';
  }
  function excerpt(text, limit, key) {
    text = String(text || '').trim();
    if (!text) return '<p class="note">No text is stored for this item.</p>';
    var long = text.length > 700;
    return '<div class="excerpt' + (long ? ' clamp' : '') + '" id="ex-' + key + '">' + esc(text.slice(0, limit)) + (text.length > limit ? '...' : '') + '</div>' +
      (long ? '<button type="button" class="link-btn" data-act="expand" data-target="ex-' + key + '" aria-expanded="false" aria-controls="ex-' + key + '">Show full text</button>' : '');
  }

  /* ---------- states ---------- */
  function skeleton() {
    return bar('<span class="skel" style="width:90px;height:22px"></span>', '', 'Back to the list') +
      '<div class="case-skel" aria-hidden="true"><span class="skel big"></span><span class="skel w50"></span><span class="skel blk"></span><span class="skel"></span><span class="skel"></span><span class="skel w50"></span></div>';
  }
  function error(msg, retryId) {
    return bar('', '', 'Back to the list') + '<div class="case-err" role="alert"><h2>This case file could not be loaded</h2><p>' + esc(msg || 'Something went wrong.') + '</p>' +
      '<button type="button" class="btn" data-act="retry"' + (retryId ? ' data-id="' + esc(retryId) + '"' : '') + '>Try again</button></div>';
  }

  /* ---------- a stored item ---------- */
  function itemHTML(it, backLabel) {
    var v = it.verdict_norm || 'UNRATED', e = it.engine, p = it.plaus, h = '', k = 'k';
    var isProp = e && e.label === 'PROPAGANDA';
    var title = it.title || it.claim || it.id, claim = it.claim && it.claim.trim() !== String(it.title || '').trim() ? it.claim : '';
    if (claim && it.title && claim.slice(0, 40) === String(it.title).slice(0, 40) && claim.length > it.title.length) { title = claim; claim = ''; } /* title is a clipped copy of the claim */
    var inferred = it.verdict_basis && it.verdict_basis !== 'claimreview';

    h += '<article class="case s-' + esc(v) + '" aria-labelledby="case-title">';
    h += bar(F.chip(v) + (isProp ? '<span class="chip tag" title="Secondary tag from the engine">Propaganda tag</span>' : '') + (inferred ? '<span class="chip tag" title="' + esc(L.verdictBasis(it.verdict_basis)) + '">Inferred rating</span>' : '') + scoreBar(p), openLink(it.url), backLabel);
    h += '<div class="case-in">';
    h += '<div class="case-hero"><div class="case-gauge">' + (p == null ? '<div class="case-nosc"><div><b>No score</b>Engine scores are pending for this item.</div></div>' : gauge(p)) + '</div>' +
      '<div class="case-id"><p class="case-kicker">' + esc([L.site(it.site), L.lang(it.lang), dateOf(it.published)].join(' - ')) + '</p>' +
      '<h2 class="case-title" id="case-title" tabindex="-1">' + esc(title) + '</h2>' +
      (claim ? '<p class="case-claim"><span class="sr">Claim checked: </span>' + esc(claim) + '</p>' : '') + '</div></div>';

    /* 1. Verdict and score */
    var rows = [];
    rows.push(['Fact-checker rating',it.verdict_raw ? esc(it.verdict_raw) + '<span class="sub">Counted as ' + (L.glossaryEntry(v) ? term(v, L.verdict(v)) : esc(L.verdict(v))) + '</span>' :
      (it.kind === 'candidate' || v === 'UNRATED' ? 'Not rated by a fact-checker<span class="sub">A social post scored by the verdict engine only.</span>' : (L.glossaryEntry(v) ? term(v, L.verdict(v)) : esc(L.verdict(v))))]);
    if (it.rating_value) rows.push(['Numeric rating', esc(it.rating_value)]);
    if (it.verdict_basis) rows.push(['Rating basis', esc(L.verdictBasis(it.verdict_basis)) + (inferred ? '<span class="sub">Inferred ratings are weaker evidence, so the label score is pulled toward 0.50.</span>' : '')]);
    rows.push([term('plausibility score'), p == null ? 'n/a' : '<b>' + p.toFixed(2) + '</b> (' + BAND[band(p)] + ')<span class="sub">Probability, 0 to 1, that the claim is fake or misleading. Bands: below 0.5, 0.5 to 0.8, 0.8 and above.</span>']);
    rows.push([term('score source'), esc(L.scoreSource(it.score_source))]);
    var blend = it.score_source === 'blend' ? '50% fact-checker label score + 50% engine probability' : it.score_source === 'engine' ? 'Engine probability alone: there is no fact-checker label to blend' :
      it.score_source === 'fact-checker-label' ? 'Label score alone: engine scores are pending for this item' : 'Nothing to blend';
    rows.push([term('blend', 'Blend method'), esc(blend)]);
    rows.push([term('label score'), it.label_score == null ? 'n/a (no fact-checker rating)' : n2(it.label_score)]);
    rows.push([term('engine probability'), e ? n2(e.p_fake_or_misleading) + '<span class="sub">' + (isProp ? 'Top class is the secondary ' + term('PROPAGANDA', 'propaganda') + ' tag, unreliable on social posts. ' : 'Top class: ' + esc(L.verdict(e.label)) + '. ') + esc(e.level === 'doc' ? 'Whole-document' : e.level === 'claim' ? 'Single-claim' : L.titleCase(e.level)) + ' estimate.</span>' : 'Engine scores pending for this item']);
    rows.push([term('source rate'), it.site_fake_rate == null ? 'n/a' : n2(it.site_fake_rate)]);
    var body = kv(rows);
    if (e && e.probs) {
      var cls = Object.keys(e.probs).sort(function (a, b) { return e.probs[b] - e.probs[a]; });
      body += '<p class="cs-sub-2">Engine probability by class</p><div class="probs">' + cls.map(function (c) {
        var pr = Number(e.probs[c]) || 0, sec = c === 'PROPAGANDA';
        return '<div class="prob v-' + esc(c) + (sec ? ' sec' : '') + '"><span>' + esc(L.verdict(c)) + (sec ? ' (secondary tag)' : '') + '</span><span class="meter" role="img" aria-label="' + esc(L.verdict(c)) + ' ' + (pr * 100).toFixed(0) + ' percent"><b style="width:' + (pr * 100).toFixed(1) + '%"></b></span><b>' + (pr * 100).toFixed(0) + '%</b></div>';
      }).join('') + '</div>';
      if (e.rationale) body += '<p class="cs-sub-2">Engine reasoning</p><p class="note" style="margin-top:0">' + esc(e.rationale) + '</p>';
    }
    body += '<p class="cal"><b>' + term('calibration', 'Calibration') + '.</b> Scores are a ranking aid, not calibrated probabilities: the blend weights are fixed rather than fitted, so 0.80 does not mean that 80% of such claims are false. Read the fact-checker\'s own article before acting.</p>';
    h += section(k + '-s1', 'Verdict and score', 'What the fact-checker concluded and how the plausibility score was put together.', body);

    /* 2. Why */
    var nb = it.neighbors || [], agree = nb.filter(function (n) { return n.verdict_norm === v; }).length, n = it.cross_checker_agreement || 0;
    var vt = it.cross_checker_verdicts || {}, vkeys = Object.keys(vt);
    var why = '<p class="why-lead">' + (n ? '<b>' + fmtN(n) + '</b> distinct source' + (n === 1 ? '' : 's') + ' rated this claim or a near-identical one. ' : 'No other source rated a near-identical claim in this index. ') +
      (nb.length ? '<b>' + agree + ' of ' + nb.length + '</b> of the most similar fact-checks reached the same rating.' : 'No similar fact-check cleared the 0.35 similarity cut-off.') + '</p>';
    var wr = [[term('cross-checker agreement'), n ? fmtN(n) + ' source' + (n === 1 ? '' : 's') + (it.cross_checker_sites && it.cross_checker_sites.length ? ': ' + esc(it.cross_checker_sites.map(L.site).join(', ')) : '') : 'None']];
    if (vkeys.length) wr.push(['Ratings given', '<span class="vtally">' + vkeys.map(function (c) { return '<span class="chip v-' + esc(c) + '">' + esc(L.verdict(c)) + ' x ' + fmtN(vt[c]) + '</span>'; }).join('') + '</span>']);
    wr.push(['Similar fact-checks that agree', nb.length ? agree + ' of ' + nb.length : 'n/a']);
    h += section(k + '-s2', 'Why', 'Who else checked this claim, and whether they agree. Agreement from independent sources is stronger evidence.', why + kv(wr));

    /* 3. Evidence */
    h += section(k + '-s3', 'Evidence', 'The nearest earlier fact-checks, ranked by ' + term('similarity', 'similarity') + '. Select one to open its case file.',
      nb.length ? '<div class="evs">' + nb.map(function (m) { return evCard(m); }).join('') + '</div>' : '<p class="note">No similar fact-checks above the 0.35 similarity cut-off.</p>');

    /* 4. Original text */
    h += section(k + '-s4', 'Original text', 'An excerpt of the article or post that was checked.', excerpt(it.body || it.description || it.claim, 6000, 'it'));

    /* 5. Signals */
    var cells = [['Characters', fmtN(it.claim_len || 0), false], ['Capitals', it.caps_ratio == null ? 'n/a' : (it.caps_ratio * 100).toFixed(0) + '%', it.caps_ratio > 0.3],
      ['Exclamations', it.exclamation_ratio == null ? 'n/a' : (it.exclamation_ratio * 100).toFixed(1) + '%', it.exclamation_ratio > 0.02], [term('sensationalism'), n2(it.sensational), it.sensational >= 0.4],
      ['Age', it.age_days == null ? 'n/a' : fmtN(it.age_days) + ' d', false], ['Recency', n2(it.recency), false]];
    h += section(k + '-s5', 'Signals', 'Wording cues in the claim, such as capitals, exclamations and sensational terms. These describe style, not whether the claim is true.', tiles(cells) + '<p class="note">' + esc(CAVEAT) + '</p>');

    /* 6. Links */
    var u = F.safeUrl(it.url), who = [it.org, it.country ? L.country(it.country) : '', it.source_tier ? L.tier(it.source_tier) : ''].filter(Boolean);
    var ln = '<ul class="links-list">' + (u ? '<li><a href="' + esc(u) + '" target="_blank" rel="noopener noreferrer">' + esc(u) + '</a></li>' : '<li>No link is stored for this item.</li>') +
      '<li>Published by ' + esc(L.site(it.site)) + (who.length ? ' (' + esc(who.join(', ')) + ')' : '') + '</li>' +
      (it.claimant ? '<li>Claim first shared via ' + esc(it.claimant) + '</li>' : '') + '</ul>';
    h += section(k + '-s6', 'Links', 'Where to read the original article or post.', ln);
    return h + '</div></article>';
  }

  /* ---------- an index-only match for pasted text (kept for reference; Claim Check builds its own two-step view) ---------- */
  var STATE = {
    strong: ['Close match', 'Close match to earlier fact-checks: the score reflects how those were rated.'],
    moderate: ['Partial overlap', 'Partial overlap with earlier fact-checks: read the evidence cards before trusting the score.'],
    weak: ['Weak resemblance', 'Only a weak resemblance to earlier fact-checks: treat the score as a faint hint, not a finding.'],
    no_close_match: ['No close precedent', 'Nothing in the fact-check database resembles this text, so no plausibility score is given.']
  };
  function stateName(s) { return (STATE[s] || [L.titleCase(s || 'result')])[0]; }
  function checkHTML(en, backLabel) {
    var d = en.res, none = d.state === 'no_close_match', c = d.confidence, k = 'c';
    var st = STATE[d.state] || [L.titleCase(d.state), ''];
    var h = '<article class="case" aria-labelledby="case-title">';
    h += bar('<span class="chip tag">' + esc(st[0]) + '</span>' + (none ? '' : scoreBar(d.score)), '', backLabel);
    h += '<div class="case-in"><div class="case-hero"><div class="case-gauge">' +
      (none ? '<div class="case-nosc"><div><b>No score</b>Best similarity ' + n2(d.nearest_similarity) + ', below the 0.25 needed.</div></div>' : gauge(d.score, null, d.state === 'weak')) + '</div>' +
      '<div class="case-id"><p class="case-kicker">Your claim - checked ' + esc(new Date(en.at).toTimeString().slice(0, 5)) + '</p>' +
      '<h2 class="case-title" id="case-title" tabindex="-1">' + esc(st[0]) + '</h2>' +
      '<p class="case-note">' + esc(st[1]) + '</p>' +
      '<p class="case-claim clamp">' + esc(en.text) + '</p></div></div>';

    var rows = [[term('plausibility score'), none ? 'No score given<span class="sub">A score is only shown when at least one fact-check is similar enough. Absence of a match does not mean the claim is true.</span>' : '<b>' + d.score.toFixed(2) + '</b> (' + BAND[band(d.score)] + ')<span class="sub">Taken from the ratings of the matched fact-checks, not from your text.</span>'],
      ['Match strength', esc(st[0])],
      [term('similarity', 'Top similarity'), n2(d.nearest_similarity) + '<span class="sub">Fact-checks below ' + n2(d.thresholds && d.thresholds.no_close_match) + ' are ignored; ' + n2(d.thresholds && d.thresholds.strong) + ' or more counts as a close match.</span>']];
    if (c) {
      rows.push(['Raw weighted score', n2(d.raw_score) + '<span class="sub">Similarity-weighted average of the matched fact-checks\' scores.</span>']);
      rows.push(['Confidence factor', n2(c.shrink) + '<span class="sub">Pulls the score toward 0.50 when the best match is weak.</span>']);
      rows.push(['Fact-checks used', fmtN(c.matches_used)]);
    }
    h += section(k + '-s1', 'Result and score', 'How close the nearest fact-checks are and the score taken from them.',
      kv(rows) + '<p class="cal"><b>' + term('calibration', 'Calibration') + '.</b> The score is a ranking aid, not a calibrated probability, and it describes the matched fact-checks rather than a ruling on your text.</p>');
    h += section(k + '-s2', 'Why', 'Claim Check compares your text with every fact-check in the local database.', '<p class="why-lead">' + esc(d.message || st[1]) + '</p>');
    h += section(k + '-s3', 'Evidence', none ? 'No fact-check was close enough to show.' : 'The nearest fact-checks, ranked by ' + term('similarity', 'similarity') + '. Select one to open its case file in the Desk.',
      d.matches && d.matches.length ? '<div class="evs">' + d.matches.map(function (m) { return evCard(m, { snippet: true }); }).join('') + '</div>' : '<p class="note">No matches to show.</p>');
    h += section(k + '-s4', 'Your text', 'The text you pasted, as it was compared. It is not stored outside this browser tab.', excerpt(en.text, 6000, 'ck'));
    var s = d.signals || {}, urg = s.urgency_terms || [];
    h += section(k + '-s5', 'Signals', 'Wording cues in your text. These describe style, not whether the claim is true.', tiles([
      ['Characters', fmtN(s.chars || 0), false], ['Words', fmtN(s.words || 0), false], ['Capitals', ((s.caps_ratio || 0) * 100).toFixed(0) + '%', s.caps_ratio > 0.3],
      ['ALL-CAPS words', fmtN(s.allcaps_words || 0), s.allcaps_words >= 2], ['Exclamations', fmtN(s.exclamations || 0), s.exclamations >= 2], ['Question marks', fmtN(s.question_marks || 0), false],
      ['Urgency terms', fmtN(urg.length), urg.length > 0], [term('sensationalism'), n2(s.sensational), s.sensational >= 0.4]]) +
      (urg.length ? '<p class="note">Urgency wording found: ' + esc(urg.join(', ')) + '</p>' : '') + '<p class="note">' + esc(CAVEAT) + '</p>');
    return h + '</div></article>';
  }

  /* ---------- keyboard for a list of .row elements: j/k and arrows move the cursor, Home/End jump, Enter or Space opens ---------- */
  function rows(list) { return Array.prototype.slice.call(list.querySelectorAll('.row[data-id]')); }
  function setCursor(list, row, focus) {
    rows(list).forEach(function (r) { r.tabIndex = r === row ? 0 : -1; });
    if (row && focus) row.focus();
  }
  function move(list, from, dir) {
    var all = rows(list); if (!all.length) return;
    var i = from ? all.indexOf(from) : -1, t = dir === 'first' ? 0 : dir === 'last' ? all.length - 1 : Math.max(0, Math.min(all.length - 1, i < 0 ? 0 : i + dir));
    setCursor(list, all[t], true);
  }
  function rowNav(list, onOpen) {
    list.addEventListener('keydown', function (e) {
      var row = e.target.closest && e.target.closest('.row[data-id]'); if (!row || e.altKey || e.ctrlKey || e.metaKey) return;
      var k = e.key;
      if (k === 'ArrowDown' || k === 'j') { e.preventDefault(); move(list, row, 1); }
      else if (k === 'ArrowUp' || k === 'k') { e.preventDefault(); move(list, row, -1); }
      else if (k === 'Home') { e.preventDefault(); move(list, row, 'first'); }
      else if (k === 'End') { e.preventDefault(); move(list, row, 'last'); }
      else if (k === 'Enter' || k === ' ') { e.preventDefault(); onOpen(row); }
    });
    list.addEventListener('focusin', function (e) { var r = e.target.closest && e.target.closest('.row[data-id]'); if (r && r.tabIndex !== 0) setCursor(list, r, false); });
  }

  /* detail-column actions shared by both views: expand an excerpt */
  function onDetailClick(e) {
    var b = e.target.closest && e.target.closest('[data-act="expand"]'); if (!b) return false;
    var ex = document.getElementById(b.getAttribute('data-target')), open = b.getAttribute('aria-expanded') === 'true';
    if (ex) ex.classList.toggle('clamp', open);
    b.setAttribute('aria-expanded', open ? 'false' : 'true'); b.textContent = open ? 'Show full text' : 'Show less';
    return true;
  }

  window.Case = { gauge: gauge, evCard: evCard, tiles: tiles, section: section, kv: kv, bar: bar, skeleton: skeleton, error: error, item: itemHTML, check: checkHTML, stateName: stateName,
    rowNav: rowNav, move: move, setCursor: setCursor, rows: rows, onDetailClick: onDetailClick };
})();
