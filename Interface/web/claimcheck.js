/* Claim Check: paste a claim, then two steps. Step 1 "Is it fake?" (instant fact-check index match from POST /api/check, then the verdict engine's
   answer from POST /api/analyse, which takes 20-35 s). Step 2 "What kind of propaganda?" (a probability bar for each of the 14 technique buckets, then the highlighted
   spans), always shown whenever the engine returned it: propaganda is never gated by the step 1 verdict. The list column is "Your checks", a history kept in memory for this tab: each entry stores the whole response so
   clicking it restores the result without calling the server again. #check/<id> addresses one check so browser Back works on small screens. */
(function () {
  'use strict';
  var $ = function (s) { return document.querySelector(s); };
  var L = window.Labels, F = window.FNPUI, C = window.Case, esc = L.esc, term = L.term, fmtN = F.fmtN;
  var MOBILE = window.matchMedia('(max-width: 959px)');
  var el = {
    desk: $('#det'), rows: $('#det-rows'), form: $('#check-form'), ta: $('#check-text'), go: $('#check-go'), msg: $('#det-msg'), count: $('#check-count'),
    detail: $('#det-detail'), empty: $('#det-empty'), body: $('#det-body'), live: $('#det-live'), n: $('#det-n'), hist: $('#det-hist-empty'), scroller: $('#det-list'),
    exList: $('#ex-list'), exCap: $('#ex-cap'), exBtn: $('#ex-refresh'), exBtnT: $('#ex-refresh-t'), exSpin: $('#ex-refresh').querySelector('.spin'),
    bar: $('#cc-bar'), stepper: $('#cc-stepper'), notice: $('#cc-notice'), s1: $('#cc-s1'), s2: $('#cc-s2'), how: $('#cc-how')
  };
  var BAND = { hi: 'high', mid: 'moderate', lo: 'low' };
  var STRONG = 0.5, MAX_ENGINE = 8000;
  var entries = [], seq = 0, cur = null, busy = false, routeCount = 0, tick = null;
  var ex = { items: [], loading: false, error: false, sel: -1, started: false, refocus: false };

  /* technique -> colour slot (CSS tokens --k1..--k8); related techniques share a hue and the name is always printed */
  var HUE = { 'loaded language': 1, 'exaggeration / minimisation': 1, 'appeal to fear-prejudice': 2, 'flag-waving': 2, 'name calling / labeling': 3, 'bandwagon / reductio ad hitlerum': 3,
    'doubt': 4, 'appeal to authority': 4, 'black-and-white fallacy': 5, 'causal oversimplification': 5, 'whataboutism / straw men / red herring': 6, 'thought-terminating cliches': 6,
    'repetition': 7, 'slogans': 7 };
  function hue(name) { return HUE[String(name).toLowerCase()] || 8; }

  /* ---------- small helpers ---------- */
  function entry(id) { return entries.filter(function (e) { return e.id === id; })[0] || null; }
  function timeOf(t) { return new Date(t).toTimeString().slice(0, 5); }
  function n2(p) { return p == null || isNaN(p) ? 'n/a' : Number(p).toFixed(2); }
  function announce(t) { el.live.textContent = ''; setTimeout(function () { el.live.textContent = t; }, 30); }
  function put(node, html) { if (node.__h !== html) { node.__h = html; node.innerHTML = html; } }
  function plural(n, one, many) { return fmtN(n) + ' ' + (n === 1 ? one : many); }
  function vchip(label) { return '<span class="chip v-' + esc(label) + '">' + esc(L.verdict(label)) + '</span>'; }
  function ix(en) { return en.res ? en.res.index : en.idx; }
  function topMatch(en) { var d = ix(en); return d && d.matches && d.matches[0] || null; }
  function strongIdx(d) { return !!(d && d.matches && d.matches.length && d.score != null && (d.nearest_similarity || 0) >= STRONG); }
  function engineState(en) { return en.eng; }

  /* the step 1 answer: the server's step1 once it has arrived, otherwise a provisional one from the instant index match */
  function step1Of(en) {
    if (en.res) return en.res.step1;
    var d = en.idx;
    if (strongIdx(d)) return { source: 'fact-check', lead: 'index', label: d.matches[0].verdict_norm, plausibility: d.score, provisional: true, second_opinion: null };
    if (en.eng === 'running' || !d) return { source: en.eng === 'running' ? 'pending' : 'none', lead: null, label: null, plausibility: null, provisional: true, second_opinion: null };
    return { source: d.score != null ? 'index only' : 'none', lead: null, label: null, plausibility: d.score, provisional: true, second_opinion: null };
  }
  function step2Of(en) { return en.res ? en.res.step2 : null; }
  function techCount(en) { var s = step2Of(en); return s && s.shown && s.propaganda ? s.propaganda.techniques.length : null; }
  /* the span-only signal {level, strong_n, weak_n, max_confidence, threshold, techniques}: the server's, or (older stored results) derived from the spans */
  function signalOf(p) {
    if (p.signal && p.signal.level) return p.signal;
    var t = p.techniques || [], conf = t.map(function (x) { return x.confidence; }).filter(function (c) { return c != null; });
    var st = t.filter(function (x) { return !isWeak(x) && (typeof x.strong === 'boolean' || x.confidence != null); }).length;
    return { level: st ? 'strong' : t.length ? 'weak' : 'none', strong_n: st, weak_n: t.length - st, max_confidence: conf.length ? Math.max.apply(null, conf) : null, threshold: STRONG, techniques: [], basis: 'confidence', p_prop: null };
  }
  /* weak = not strong: the engine's own `strong` flag when it sent one, else confidence below 0.50 */
  function isWeak(t) { return typeof t.strong === 'boolean' ? !t.strong : t.confidence != null && t.confidence < STRONG; }
  function sigWords(sg) {
    if (!sg || sg.level === 'none') return 'no strong spans';
    return sg.strong_n ? sg.strong_n + ' strong' + (sg.weak_n ? ', ' + sg.weak_n + ' weak' : '') : sg.weak_n + ' weak';
  }
  function sigOf(en) { var s = step2Of(en); return s && s.shown && s.propaganda ? signalOf(s.propaganda) : null; }

  /* ---------- history ---------- */
  function scoreChip(en) {
    var s1 = step1Of(en);
    if (en.eng === 'running' && s1.plausibility == null) return '<span class="chk-pend">Checking</span>';
    if (s1.plausibility == null) return '<span class="chip num v-UNRATED" title="No score">No score</span>';
    var b = F.band(s1.plausibility);
    return '<span class="chip num v-' + (b === 'hi' ? 'FAKE' : b === 'mid' ? 'MISLEADING' : 'INCONCLUSIVE') + '" title="Plausibility score">' + s1.plausibility.toFixed(2) + '</span>';
  }
  function labelChip(en) {
    var s1 = step1Of(en);
    if (s1.label) return vchip(s1.label);
    if (en.eng === 'running') return '<span class="chip tag">Step 1 running</span>';
    return '<span class="chip tag">' + (s1.source === 'index only' ? 'Index only' : 'No verdict') + '</span>';
  }
  function rowSummary(en) {
    var s2 = step2Of(en), n = techCount(en);
    if (en.eng === 'running') return 'Checking';
    if (!s2) return en.eng === 'cancelled' ? 'Stopped' : 'Engine did not answer';
    if (s2.shown) return 'Step 2: ' + sigWords(sigOf(en));
    return 'Step 2 unavailable';
  }
  function renderRows() {
    el.hist.hidden = entries.length > 0; el.n.textContent = entries.length ? String(entries.length) : '';
    el.rows.hidden = !entries.length;
    el.rows.innerHTML = entries.map(function (en) {
      var s1 = step1Of(en);
      return '<li class="row chk' + (s1.label ? ' s-' + esc(s1.label) : '') + '" role="option" data-id="' + esc(en.id) + '" tabindex="-1" aria-selected="' + (en.id === cur) + '"' + (en.id === cur ? ' aria-current="true"' : '') + '>' +
        '<span class="row-title">' + esc(en.text.replace(/\s+/g, ' ')) + '</span>' +
        '<span class="row-line"><span class="cc-rl">' + labelChip(en) + scoreChip(en) + '</span>' +
        '<span class="row-meta"><span>' + esc(rowSummary(en)) + '</span><span aria-hidden="true">&middot;</span><span>' + esc(timeOf(en.at)) + '</span></span></span></li>';
    }).join('');
    var first = el.rows.querySelector('.row'); if (first) C.setCursor(el.rows, el.rows.querySelector('.row[aria-selected="true"]') || first, false);
  }

  /* ---------- stepper header ---------- */
  function stepStates(en) {
    var s1 = step1Of(en), s2 = step2Of(en), run = en.eng === 'running', a, b;
    if (run) a = ['st-run', 'Running']; else if (s1.source === 'none') a = ['st-na', 'Not completed']; else a = ['st-done', s1.lead ? 'Done' : 'Index match only'];
    if (run) b = ['st-wait', 'Waiting for step 1'];
    else if (s2 && s2.shown) b = ['st-done', 'Done'];
    else b = ['st-na', 'Not available'];
    return [a, b];
  }
  function stepperHTML(en) {
    var st = stepStates(en), names = ['Is it fake?', 'What propaganda?'];
    return '<ol class="cc-stepper" aria-label="Claim Check progress">' + st.map(function (s, i) {
      return '<li class="' + s[0] + '"' + (s[0] === 'st-run' ? ' aria-current="step"' : '') + '><span class="cc-n" aria-hidden="true">' + (s[0] === 'st-run' ? '<i class="cc-spin"></i>' : i + 1) + '</span>' +
        '<span class="cc-st-t"><span class="cc-k">Step ' + (i + 1) + '</span><b>' + esc(names[i]) + '</b><small>' + esc(s[1]) + '</small></span></li>';
    }).join('') + '</ol>';
  }

  /* ---------- header bar ---------- */
  function barHTML(en) {
    var s1 = step1Of(en), n = techCount(en), sg = sigOf(en);
    var chips = labelChip(en) + (n != null ? '<span class="chip tag">' + esc(!sg || sg.level === 'none' ? 'No strong spans' : sg.strong_n ? plural(sg.strong_n, 'strong span', 'strong spans') : plural(sg.weak_n, 'weak span', 'weak spans')) + '</span>' : '');
    var score = s1.plausibility == null ? '' : '<span class="case-bar-score" title="Plausibility score">' + s1.plausibility.toFixed(2) + '<small>' + BAND[F.band(s1.plausibility)] + ' plausibility</small></span>';
    return C.bar(chips + score, '', 'Back to your checks');
  }

  /* ---------- step 1 ---------- */
  function plainLine(en, s1) {
    var top = topMatch(en), d = ix(en);
    if (s1.source === 'fact-check') return 'Matches a fact-check by ' + esc(L.site(top && top.site)) + ': ' + esc(L.verdictPlain(s1.label)) + '.' + (s1.provisional && en.eng === 'running' ? ' The verdict engine is still reading it.' : '');
    if (s1.source === 'model estimate') return 'No matching fact-check; ' + term('model estimate', 'model estimate') + ': ' + esc(L.verdictPlain(s1.label)) + '.';
    if (s1.source === 'index only') return 'The verdict engine did not answer, so this is the nearest fact-check match only. Treat it as a hint.';
    if (s1.source === 'pending') {
      return (d && (d.nearest_similarity || 0) >= 0.25 ? 'Only a weak match in the fact-check index (similarity ' + n2(d.nearest_similarity) + ').' : d ? 'No matching fact-check in the index.' : 'Comparing with the fact-check index.') + ' Waiting for the verdict engine.';
    }
    return 'No matching fact-check, and the verdict engine did not answer.';
  }
  function heroHTML(en, s1) {
    var gauge = s1.plausibility != null ? C.gauge(s1.plausibility, null, s1.source === 'index only' && ix(en) && ix(en).state === 'weak')
      : '<div class="case-nosc"><div><b>' + (en.eng === 'running' ? 'Waiting' : 'No score') + '</b>' + (en.eng === 'running' ? 'for the verdict engine' : 'Nothing close enough to score.') + '</div></div>';
    var title = s1.label ? L.verdictPlain(s1.label) : s1.source === 'pending' ? 'Checking' : s1.source === 'index only' ? 'Index match only' : 'No verdict';
    return '<div class="case-hero"><div class="case-gauge">' + gauge + '</div><div class="case-id"><p class="case-kicker">' + esc(en.kicker || 'Your claim - checked ' + timeOf(en.at)) + '</p>' +
      '<h2 class="case-title" id="case-title" tabindex="-1">' + esc(title) + '</h2><p class="case-note">' + plainLine(en, s1) + '</p>' +
      '<p class="case-claim clamp"><span class="sr">Claim checked: </span>' + esc(en.text) + '</p></div></div>';
  }
  function progressHTML() {
    return '<div class="cc-prog"><span class="cc-spin big" aria-hidden="true"></span><p><b>Running the verdict engine...</b> <span class="cc-el" data-el></span><span class="cc-usual">, usually 20-35 s</span></p>' +
      '<button type="button" class="btn ghost sm" data-act="cancel">Cancel</button></div>';
  }
  var ENG_MSG = {
    unavailable: ['The verdict engine is not reachable', 'It may be restarting. The fact-check index result above is still valid. Try again in a minute.'],
    timeout: ['The verdict engine took too long', 'It did not answer within 60 seconds. The index result above is still valid. A retry is often faster because the engine may have finished in the background.'],
    busy: ['The verdict engine is busy', 'Other checks are ahead of this one. The index result above is still valid.'],
    error: ['The verdict engine returned an error', 'The index result above is still valid. Try again in a moment.'],
    degraded: ['The deep verdict timed out', 'Showing the fast result and propaganda techniques only. Retry for the full verdict.'],
    degraded_unavailable: ['The deep verdict was unavailable', 'Showing the fast result and propaganda techniques only. Retry for the full verdict.'],
    degraded_no_claims: ['No checkable claim found', 'There was no factual claim to check, so step 1 has no verdict. Step 2 below still reads the wording.'],
    cancelled: ['You stopped the engine check', 'The server may still finish it, in which case a retry is instant. The index result above is still valid.'],
    failed: ['The check could not be completed', 'The server did not answer properly. Check your connection and try again.'],
    ratelimit: ['Too many checks in a minute', 'Claim Check allows 10 engine checks a minute. Wait a moment, then retry.']
  };
  function retryLabel(en) {
    var left = Math.ceil((en.retryAt - Date.now()) / 1000);
    return left > 0 ? 'Retry in ' + left + ' s' : 'Retry';
  }
  function engNoticeHTML(en) {
    var why = (en.eng === 'degraded' || en.eng === 'ok') && en.res && en.res.engine ? en.res.engine.degraded_why : null;
    if (en.eng === 'ok' && why !== 'no_claims') return '';
    var m = ENG_MSG[why === 'no_claims' ? 'degraded_no_claims' : why === 'unavailable' ? 'degraded_unavailable' : en.eng]; if (!m) return '';
    var reason = en.eng === 'error' && en.res && en.res.engine && en.res.engine.reason && en.eng === en.res.engine.state ? '<span class="sub">Detail: ' + esc(en.res.engine.reason) + '</span>' : '';
    var wait = en.retryAt > Date.now();
    return '<div class="cc-note' + (en.eng === 'degraded' ? ' warn' : '') + '" role="status"><div><b>' + esc(m[0]) + '.</b> ' + esc(m[1]) + reason + '</div>' +
      (why === 'no_claims' ? '' : '<button type="button" class="btn sm" data-act="retry" data-retry' + (wait ? ' disabled' : '') + '>' + esc(retryLabel(en)) + '</button>') + '</div>';
  }
  /* verdict class bars: step1.classes = [{label, probability, secondary}] (FAKE, MISLEADING, LEGIT, INCONCLUSIVE, then the raw PROPAGANDA as a secondary tag);
     older stored results without it fall back to the engine's own probability map */
  function probsHTML(probs, classes, note) {
    var list = Array.isArray(classes) && classes.length ? classes.map(function (c) { return { c: c.label, p: c.probability, sec: !!c.secondary }; })
      : Object.keys(probs || {}).sort(function (a, b) { return probs[b] - probs[a]; }).map(function (c) { return { c: c, p: probs[c], sec: c === 'PROPAGANDA' }; });
    if (!list.length) return '';
    return '<p class="cs-sub-2">Probability by verdict class</p><div class="probs">' + list.map(function (x) {
      var c = x.c, pr = Number(x.p) || 0, sec = x.sec;
      return '<div class="prob v-' + esc(c) + (sec ? ' sec' : '') + '"><span>' + esc(L.verdict(c)) + (sec ? ' (secondary tag)' : '') + '</span><span class="meter" role="img" aria-label="' + esc(L.verdict(c)) + ' ' + (pr * 100).toFixed(0) + ' percent"><b style="width:' + (pr * 100).toFixed(1) + '%"></b></span><b>' + (pr * 100).toFixed(0) + '%</b></div>';
    }).join('') + '</div>' + (note ? '<p class="note" style="margin:var(--s-2) 0 0">' + esc(note) + '</p>' : '');
  }
  function engineHTML(en, s1) {
    var r = en.res, e = r && r.engine; if (!e || e.state !== 'ok' || !e.verdict) return '';
    var v = e.verdict, so = s1.second_opinion, lead = s1.lead === 'engine';
    var head = lead ? term('model estimate', 'Model estimate') + ' from the verdict engine' : term('second opinion', 'Second opinion') + ' from the verdict engine';
    var agree = so ? '<span class="chip tag">' + (so.agrees ? 'Agrees with the fact-check' : 'Differs from the fact-check') + '</span>' : '';
    var plaus = lead ? s1.plausibility : so && so.plausibility;
    var rows = [['Decided by', esc(L.titleCase(v.decider || 'unknown'))], [term('plausibility score', 'Engine plausibility'), n2(plaus)]];
    if (typeof v.plausibly_fake === 'boolean') rows.push(['Plausibly fake tier', v.plausibly_fake ? 'Yes' : 'No']);
    var ly = e.laya && e.laya.label ? e.laya : null, lyTop = null;
    if (ly) {
      var lp = ly.probs || {}, lk = Object.keys(lp).sort(function (a, b) { return lp[b] - lp[a]; })[0];
      lyTop = lk != null ? lp[lk] : ly.confidence;
      rows.push([term('fine-tuned laya check', 'Fine-tuned Laya check'), esc(L.verdict(ly.label)) + (lyTop != null ? ' (' + Math.round(Number(lyTop) * 100) + '%)' : '') + ' <small>(details in step 2)</small>']);
    }
    var h = '<div class="cc-eng"><p class="cc-eng-h"><b>' + head + '</b></p><p class="cc-eng-v">' + vchip(lead ? (s1.label || v.label) : ((so && so.label) || v.label)) + (v.label === 'PROPAGANDA' ? '<span class="chip tag" title="The engine\'s top class was propaganda-style wording; see step 2">' + esc(L.verdict('PROPAGANDA')) + ' (secondary tag)</span>' : '') + agree + '</p>' + probsHTML(v.probs, s1.classes, s1.classes_note);
    if (v.rationale) h += '<p class="cs-sub-2">Engine reasoning</p><p class="note" style="margin-top:0">' + esc(v.rationale) + '</p>';
    h += '<p class="cs-sub-2">Technical detail</p>' + C.kv(rows);
    if (ly) h += '<p class="note" style="margin:var(--s-2) 0 0">Laya is reported separately: the calibrated stacker is the decider, so Laya\'s answer does not change step 1.</p>';
    h += '<p class="cal"><b>' + term('calibration', 'Calibration') + '.</b> Scores are a ranking aid, not calibrated probabilities: 0.80 does not mean that 80% of such claims are false. Read the fact-checker\'s own article before acting.</p></div>';
    return h;
  }
  function evidenceHTML(en) {
    var d = ix(en); if (!d) return '';
    var m = d.matches || [];
    return '<p class="cs-sub-2">Nearest fact-checks</p>' + (m.length ? '<div class="evs">' + m.map(function (x) { return C.evCard(x, { snippet: true }); }).join('') + '</div>'
      : '<p class="note">No fact-check was close enough to show (best ' + term('similarity', 'similarity') + ' ' + (d.nearest_similarity == null ? 'n/a' : Number(d.nearest_similarity).toFixed(3)) + ', below the ' + n2(d.thresholds && d.thresholds.no_close_match) + ' needed). That does not mean the claim is true.</p>');
  }
  function s1HTML(en) {
    var s1 = step1Of(en), body = heroHTML(en, s1);
    if (en.eng === 'running') body += progressHTML(); else body += engNoticeHTML(en);
    body += evidenceHTML(en) + engineHTML(en, s1);
    return C.section('cc-h1', 'Step 1: Is it fake?', 'Whether the claim looks fake or misleading, from the nearest fact-checks and the verdict engine.', body);
  }

  /* ---------- step 2 ---------- */
  function cpIndex(text, i) { return Array.from(text.slice(0, i)).length; }
  /* offsets from the engine count characters (code points); mark up the text with a mark per span and the technique name after each span */
  function highlightHTML(text, techs) {
    var cps = Array.from(text), spans = [], unplaced = 0;
    techs.forEach(function (t) {
      var s = t.start, e = t.end;
      if (!(s >= 0 && e > s && e <= cps.length) || cps.slice(s, e).join('') !== t.text) {
        var at = t.text ? text.indexOf(t.text) : -1;
        if (at < 0) { unplaced++; return; }
        s = cpIndex(text, at); e = s + Array.from(t.text).length;
      }
      spans.push({ s: s, e: e, name: L.technique(t.technique), h: hue(L.technique(t.technique)), weak: isWeak(t) });
    });
    var cuts = [0, cps.length]; spans.forEach(function (x) { cuts.push(x.s, x.e); });
    cuts = cuts.filter(function (v, i, a) { return a.indexOf(v) === i; }).sort(function (a, b) { return a - b; });
    var out = '';
    for (var i = 0; i + 1 < cuts.length; i++) {
      var a = cuts[i], b = cuts[i + 1], txt = esc(cps.slice(a, b).join('')), cov = spans.filter(function (x) { return x.s <= a && x.e >= b; });
      var wk = cov.length && cov.every(function (x) { return x.weak; });
      out += cov.length ? '<mark class="cc-mk k' + cov[0].h + (cov.length > 1 ? ' multi' : '') + (wk ? ' weak' : '') + '" title="' + esc(cov.map(function (x) { return x.name + (x.weak ? ' (weak)' : ''); }).join(', ')) + '">' + txt + '</mark>' : txt;
      spans.forEach(function (x) { if (x.e === b && x.s < b) out += '<span class="cc-tag k' + x.h + (x.weak ? ' weak' : '') + '">' + esc(x.name) + (x.weak ? ' (weak)' : '') + '</span>'; });
    }
    return { html: out, unplaced: unplaced, weak: spans.some(function (x) { return x.weak; }) };
  }
  function bucketsHTML(techs) {
    var by = {};
    techs.forEach(function (t) { var n = L.technique(t.technique), b = by[n] || (by[n] = { name: n, n: 0, max: null, weak: true }); b.n++; if (!isWeak(t)) b.weak = false; if (t.confidence != null && (b.max == null || t.confidence > b.max)) b.max = t.confidence; });
    var list = Object.keys(by).map(function (k) { return by[k]; }).sort(function (a, b) { return b.n - a.n || (b.max || 0) - (a.max || 0); });
    return '<ul class="cc-buckets">' + list.map(function (b) {
      var g = L.glossaryEntry(b.name);
      var wk = b.weak;
      return '<li class="cc-bk k' + hue(b.name) + (wk ? ' weak' : '') + '"><span class="cc-bk-top"><span class="cc-bk-name">' + term(b.name, b.name) + '</span><span class="cc-bk-n">' + esc(plural(b.n, 'span', 'spans')) + ', max confidence ' + esc(n2(b.max)) + (wk ? ' (weak)' : '') + '</span></span>' +
        (g ? '<span class="cc-bk-def">' + esc(g.definition) + '</span>' : '') + '</li>';
    }).join('') + '</ul>';
  }
  /* the span-only summary line: one chip + one sentence + the rule. Laya is NOT merged into it. */
  function signalHTML(p) {
    var sg = signalOf(p), one = p.one_sided_score != null ? p.one_sided_score >= 0.5 : p.one_sided === true, line, chip;
    if (sg.level === 'strong') {
      var nm = (sg.techniques || []).map(function (t) { return L.technique(t); }), shown = nm.slice(0, 3).join(', ') + (nm.length > 3 ? ' and ' + (nm.length - 3) + ' more' : '');
      chip = '<span class="chip v-MISLEADING">STRONG</span>'; line = 'Strong signal: ' + plural(sg.strong_n, 'span', 'spans') + (shown ? ' (' + shown + ')' : '') + '.';
    } else if (sg.level === 'weak') {
      chip = '<span class="chip v-INCONCLUSIVE">WEAK</span>'; line = 'Weak signal only: strongest span ' + n2(sg.max_confidence) + '.';
    } else { chip = '<span class="chip tag">NONE</span>'; line = 'No strong spans found.'; }
    return '<div class="cc-sig lv-' + esc(sg.level) + '"><p class="cc-sec-row"><b>' + term('propaganda signal', 'Propaganda signal') + '</b>' + chip + (one ? '<span class="chip tag">' + term('one-sided', 'One-sided') + '</span>' : '') + '</p>' +
      '<p class="cc-sig-line">' + esc(line) + '</p><p class="note cc-sig-rule">Rule: ' + (sg.basis === 'engine' ? 'a span is strong when the tagger rates it at or above its class threshold and 0.50' : 'a span is strong at a tagger confidence of ' + esc(Number(sg.threshold || STRONG).toFixed(2)) + ' or more') + '. Not calibrated, not validated on social media.' +
      (sg.p_prop != null ? ' Engine score ' + esc(n2(sg.p_prop)) + ' (an uncalibrated score, not a probability).' : '') + ' Laya is shown below and is not merged into this line.</p>' +
      (p.one_sided_score != null ? '<p class="note cc-sig-rule">Framing (' + term('one-sided', 'one-sided') + ') score ' + esc(n2(p.one_sided_score)) + ': Gemma\'s framing judgement, unvalidated. 0 does not prove the text is balanced.</p>' : '') + '</div>';
  }
  var LAYA_NONE = {
    not_run_no_claims: 'Laya did not run: no checkable factual claim was extracted, and Laya reads claims only.',
    timeout: 'Not available: the deep verdict timed out.',
    not_returned: 'Not returned by this engine version.'
  };
  function layaHTML(lv) {
    var head = '<div class="cc-sec cc-lay"><p class="cs-sub-2" style="margin-top:0">' + term('secondary signal', 'Secondary signal') + '</p>' +
      '<p class="cc-sec-row"><b>' + term('fine-tuned laya check', 'Fine-tuned Laya check') + '</b><span class="chip tag">Secondary, not validated</span></p>';
    if (!lv) return head + '<p class="cc-na">Details not stored for this older result.</p></div>';
    var cav = ['Laya reads a structured summary (the claim, signals and spans), not the raw words.',
      'That summary already contains the span tagger\'s spans, so Laya is not an independent second opinion.',
      'It never decides step 1 and is not validated on social media.'];
    var cavH = '<ul class="cc-cav cc-lay-cav">' + cav.map(function (c) { return '<li>' + esc(c) + '</li>'; }).join('') + '</ul>';
    if (lv.state !== 'ok') return head + '<p class="cc-na">' + esc(LAYA_NONE[lv.state] || LAYA_NONE.not_returned) + '</p>' + cavH + '</div>';
    var probs = Array.isArray(lv.probs) ? lv.probs : [], conf = lv.confidence;
    var rowA = '<div class="cc-lay-r"><span class="cc-lay-k">Laya\'s own verdict (5 classes)</span><div class="cc-lay-v">' +
      (lv.label ? '<p class="cc-sec-row" style="margin:0">' + vchip(lv.label) + (conf != null ? '<b class="num">' + Math.round(Number(conf) * 100) + '%</b>' : '') + '</p>' : '<span class="cc-na">Not returned</span>') +
      (probs.length ? '<div class="probs cc-lprobs">' + probs.map(function (x) {
        var pr = Math.max(0, Math.min(1, Number(x.probability) || 0));
        return '<div class="prob v-' + esc(x.label) + (x.secondary ? ' sec hl' : '') + '"><span>' + esc(L.verdict(x.label)) + '</span><span class="meter" role="img" aria-label="' + esc(L.verdict(x.label)) + ' ' + (pr * 100).toFixed(0) + ' percent"><b style="width:' + (pr * 100).toFixed(1) + '%"></b></span><b>' + (pr * 100).toFixed(0) + '%</b></div>';
      }).join('') + '</div>' : '') + '<p class="note cc-lay-cap">Laya\'s own class, not validated.</p></div></div>';
    var hv = lv.propaganda_head, pp = null;
    probs.forEach(function (x) { if (x.label === 'PROPAGANDA') pp = x.probability; });
    var rowB = '<div class="cc-lay-r"><span class="cc-lay-k">' + term('laya propaganda head', 'Propaganda yes/no head') + ' (raw output)</span><div class="cc-lay-v">' +
      (hv != null ? '<p class="cc-sec-row" style="margin:0"><b class="num cc-lay-n">' + n2(hv) + '</b><span class="meter cc-lmeter" role="img" aria-label="Propaganda head ' + n2(hv) + ' out of 1"><b style="width:' + (hv * 100).toFixed(1) + '%"></b></span></p>' : '<span class="cc-na">Not returned by this engine version.</span>') +
      (pp != null ? '<p class="note cc-lay-cap">Next to it, Laya\'s own PROPAGANDA class: ' + Math.round(pp * 100) + '% (Laya\'s own class, not validated).</p>' : '') +
      '<p class="note cc-lay-cap">Raw output, not a verdict, not validated. It appears to separate article-style training data rather than propaganda wording (on a test it read 1.0 on news-article states and 0.0 on others), so a high or low value says little about this text. Not used for the summary line or step 1.</p></div></div>';
    var rowC = '<div class="cc-lay-r"><span class="cc-lay-k">Compared with the span tagger</span><div class="cc-lay-v"><span class="chip tag cc-cmp ' + esc(lv.compared || 'not_compared') + '">' + esc(lv.compared_note || 'Not compared') + '</span></div></div>';
    return head + '<div class="cc-lay-grid">' + rowA + rowB + rowC + '</div>' + cavH + '</div>';
  }
  /* one bar per merged technique bucket: step2.propaganda.classes = [{technique, probability, basis}], all 14, sorted descending, explicit 0s */
  function classBarsHTML(p) {
    var cls = Array.isArray(p.classes) ? p.classes : [];
    if (!cls.length) return '';
    return '<p class="cs-sub-2" style="margin-top:0">Probability by technique</p><div class="probs cc-tprobs">' + cls.map(function (c) {
      var pr = Math.max(0, Math.min(1, Number(c.probability) || 0)), name = L.technique(c.technique);
      return '<div class="prob tq k' + hue(name) + (pr === 0 ? ' zero' : '') + '"><span>' + term(name, name) + '</span><span class="meter" role="img" aria-label="' + esc(name) + ' ' + (pr * 100).toFixed(0) + ' percent"><b style="width:' + (pr * 100).toFixed(1) + '%"></b></span><b>' + (pr > 0 && pr < 0.005 ? '&lt;1' : (pr * 100).toFixed(0)) + '%</b></div>';
    }).join('') + '</div><p class="note" style="margin:var(--s-2) 0 var(--s-3)">' + esc(p.basis === 'engine' ? (p.basis_note || 'encoder score = max token probability, uncalibrated, not validated on social posts') + '. Small bars on short texts are weak signals only.' : 'Each bar is the strongest span the tagger reported for that technique. 0% means no span was reported; the tagger may still see weak evidence it does not return.') + '</p>';
  }
  function s2HTML(en) {
    var s2 = step2Of(en);
    if (!s2 || !s2.shown || !s2.propaganda) return '';
    var p = s2.propaganda, techs = p.techniques || [], text = en.text.trim().slice(0, MAX_ENGINE).trim();
    var body = signalHTML(p) + classBarsHTML(p);
    var unplaced = Number(p.dropped) || 0, words = p.text_words != null ? p.text_words : text.split(/\s+/).filter(Boolean).length;
    if (techs.length) {
      var hl = highlightHTML(text, techs); unplaced += hl.unplaced;
      body += '<p class="cs-sub-2" style="margin-top:0">Your claim, with the wording marked</p><div class="cc-text">' + hl.html + '</div>';
      if (hl.weak) body += '<p class="note cc-legend">Solid underline = strong span. Dotted underline and "(weak)" = a span the tagger reported but did not rate strong (under 0.50 or under its own class threshold).</p>';
      body += '<div class="cc-found"><p class="cs-sub-2">Techniques found</p></div>' + bucketsHTML(techs);
    } else {
      body += '<div class="cc-empty"><b>No strong propaganda spans found</b><span>' + (p.too_short === true ? 'The text is too short for the span tagger (under 12 tokens), so it can say little; this is not evidence the wording is neutral.' : p.too_short == null && words < 60 ? 'The span tagger was trained on long news articles and reports little on short texts (' + esc(plural(words, 'word', 'words')) + ' here), so this is not evidence the wording is neutral.' : 'The tagger found no wording that fits any of the 14 techniques.') + ' It says nothing about whether the claim is true.</span></div>';
    }
    if (unplaced > 0) body += '<p class="note">' + esc(plural(unplaced, 'span', 'spans')) + ' could not be located in the text, so ' + (unplaced === 1 ? 'it is' : 'they are') + ' not highlighted.</p>';
    body += layaHTML(s2.laya);
    return C.section('cc-h2', 'Step 2: What kind of propaganda?', 'Shown for every claim, whether or not step 1 finds it fake. It describes how the text is worded, not whether it is true.', body);
  }

  /* ---------- how we decided ---------- */
  var ENGINE_NAME = { predict_v0: 'Interim engine (predict_v0)', check_v1: 'Claim Check service (check_v1)' };
  function versions(mv) {
    if (!mv) return 'Not reported';
    if (typeof mv === 'string') return esc(mv);
    var parts = Array.isArray(mv) ? mv.map(String) : Object.keys(mv).map(function (k) { return L.titleCase(k) + ' ' + (typeof mv[k] === 'object' ? JSON.stringify(mv[k]) : mv[k]); });
    return parts.length ? parts.map(esc).join('<br>') : 'Not reported';
  }
  function howHTML(en) {
    var r = en.res; if (!r) return '';
    var e = r.engine || {}, ok = e.state === 'ok' || e.state === 'degraded', s1 = r.step1, rows = [];
    rows.push(['Step 1 source', esc(s1.source === 'fact-check' ? 'Fact-check index (the nearest fact-check leads)' : s1.source === 'model estimate' ? 'Verdict engine (no close fact-check)' : 'Fact-check index only')]);
    rows.push(['Verdict engine', ok ? esc(ENGINE_NAME[e.engine] || L.titleCase(e.engine || 'unknown')) : 'Did not answer: ' + esc(L.titleCase(e.state || 'unknown'))]);
    if (ok) {
      rows.push(['Models and versions', versions(e.model_versions)]);
      rows.push(['Engine time', e.secs == null ? 'n/a' : esc(Number(e.secs).toFixed(1)) + ' s']);
      rows.push(['Waited in queue', e.queue_wait_s == null ? 'n/a' : esc(Number(e.queue_wait_s).toFixed(1)) + ' s' + (e.queue_position ? ' (position ' + esc(e.queue_position) + ')' : '')]);
      rows.push(['Served from cache', e.cached ? 'Yes: a repeat of an earlier check' : 'No: the engine read this text now']);
    }
    rows.push(['Text analysed', esc(fmtN(r.engine_chars)) + ' of ' + esc(fmtN(r.text_chars)) + ' characters']);
    var cav = ['The propaganda label alone is unreliable: it follows text length more than persuasion, so it is a secondary tag and never decides step 1.',
      'The ' + term('fine-tuned laya check', 'fine-tuned Laya check') + ' reads a structured summary of the text (which already includes the span tagger\'s spans), not the raw words, and its propaganda head appears to track article-style training data rather than wording. Texts over about 60 words go through automatic claim extraction first.',
      'Accuracy on real social-media data has not been measured. Treat every verdict as a lead to verify, not a ruling.',
      'Fact-check matching compares words, not meaning, so a paraphrase or another language may find no match.'];
    return '<details class="cc-how"' + (en.howOpen ? ' open' : '') + '><summary>How we decided</summary><div class="cc-how-in">' + C.kv(rows) +
      '<p class="cs-sub-2">Caveats</p><ul class="cc-cav">' + cav.map(function (c) { return '<li>' + c + '</li>'; }).join('') + '</ul></div></details>';
  }

  /* ---------- paint the current entry (slots update only when their HTML changed, so the live stepper only announces real changes) ---------- */
  function paint(en) {
    var act = document.activeElement, keep = act && el.body.contains(act) ? act.getAttribute('data-act') : null, top = el.detail.scrollTop;
    var r = en.res;
    put(el.bar, barHTML(en));
    put(el.stepper, stepperHTML(en));
    put(el.notice, r && r.truncated ? '<p class="cc-note plain">Only the first ' + fmtN(MAX_ENGINE) + ' characters were analysed</p>' : '');
    put(el.s1, s1HTML(en)); put(el.s2, s2HTML(en)); put(el.how, howHTML(en));
    el.detail.scrollTop = top;
    if (keep) { var again = el.body.querySelector('[data-act="' + keep + '"]'); if (again && document.activeElement !== again) again.focus({ preventScroll: true }); }
    syncTick();
  }
  function syncTick() {
    var en = entry(cur); if (!en) return;
    var s = el.body.querySelectorAll('[data-el]'), secs = Math.max(0, Math.round((Date.now() - (en.t0 || Date.now())) / 1000));
    Array.prototype.forEach.call(s, function (n) { n.textContent = secs + ' s'; });
    var rb = el.body.querySelector('[data-retry]');
    if (rb) { var wait = en.retryAt > Date.now(); rb.disabled = wait; rb.textContent = retryLabel(en); }
  }
  function onTick() {
    var any = entries.some(function (e) { return e.ctrl; }), en = entry(cur);
    if (en) {
      if (en.retryAt > Date.now()) any = true;
      syncTick();
      if (en.ctrl) {
        var secs = Math.round((Date.now() - en.t0) / 1000), mark = Math.floor(secs / 15);
        if (mark > 0 && mark > (en.mark || 0)) { en.mark = mark; announce('Still running the verdict engine, ' + secs + ' seconds so far.'); }
      } else if (en.retryAt && en.retryAt <= Date.now() && !en.retryShown) { en.retryShown = true; paint(en); }
    }
    if (!any) { clearInterval(tick); tick = null; }
  }
  function startTick() { if (!tick) tick = setInterval(onTick, 1000); }
  function refreshView(en) { renderRows(); if (cur === en.id) { paint(en); Shell.setTitle(en.text); } }

  function showEntry(en, focus) {
    cur = en.id; el.desk.setAttribute('data-pane', 'case');
    el.empty.hidden = true; el.body.hidden = false;
    paint(en); el.detail.scrollTop = 0;
    Shell.setTitle(en.text);
    renderRows();
    if (focus && MOBILE.matches) { var t = $('#det-body #case-title'); if (t) t.focus({ preventScroll: true }); }
  }
  function clearView() { cur = null; el.body.hidden = true; el.empty.hidden = false; el.desk.setAttribute('data-pane', 'list'); renderRows(); }

  /* ---------- running a check ---------- */
  function setBusy() {
    busy = entries.some(function (e) { return e.ctrl; });
    el.go.disabled = busy; el.go.textContent = busy ? 'Checking' : 'Check claim';
  }
  function startEngine(en) {
    var ctrl = new AbortController();
    en.ctrl = ctrl; en.t0 = Date.now(); en.mark = 0; en.retryAt = 0; en.retryShown = false; en.eng = 'running';
    setBusy(); startTick(); refreshView(en);
    var payload = { text: en.text };
    fetch('/api/analyse', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload), signal: ctrl.signal })
      .then(function (r) {
        return r.json().then(function (j) { return { status: r.status, j: j || {}, ra: parseInt(r.headers.get('Retry-After'), 10) }; }, function () { return { status: r.status, j: {}, ra: NaN }; });
      })
      .then(function (o) {
        if (en.ctrl !== ctrl) return;
        if (o.status === 200 && o.j.engine && o.j.step1) {
          var st = o.j.engine.state;
          en.res = o.j; en.eng = st;
          if (st === 'busy') en.retryAt = Date.now() + (Number(o.j.engine.retry_after_s) || o.ra || 10) * 1000;
        } else if (o.status === 429) {
          en.eng = 'ratelimit';
          en.retryAt = Date.now() + (o.ra || Number(o.j.retry_after_s) || 20) * 1000;
        } else en.eng = 'failed';
      })
      .catch(function (e) {
        if (en.ctrl !== ctrl || (e && e.name === 'AbortError')) return;
        en.eng = 'failed';
      })
      .then(function () {
        if (en.ctrl !== ctrl) return;
        en.ctrl = null; setBusy(); if (en.retryAt > Date.now()) startTick();
        refreshView(en);
      });
  }
  function cancel(en) {
    if (!en.ctrl) return;
    en.ctrl.abort(); en.ctrl = null;
    en.eng = 'cancelled';
    setBusy(); refreshView(en);
    announce('Stopped. The fact-check index result is still shown.');
  }
  function run(text) {
    text = String(text || '').trim();
    if (text.length < 3) { el.msg.textContent = 'Enter at least a few words to check.'; el.ta.focus(); return; }
    if (busy) { el.msg.textContent = 'A check is still running. Wait for it, or cancel it first.'; return; }
    el.msg.textContent = '';
    var en = { id: 'c' + (++seq), text: text, at: Date.now(), idx: null, res: null, eng: 'running', ctrl: null, retryAt: 0 };
    entries.unshift(en);
    Shell.navigate('#check/' + en.id);
    /* instant half of step 1: the index match answers in milliseconds while the engine reads the text */
    fetch('/api/check', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text: en.text }) })
      .then(function (r) { return r.ok ? r.json() : null; }).catch(function () { return null; })
      .then(function (d) { if (d && !en.res) { en.idx = d; refreshView(en); } });
    startEngine(en);
  }

  /* ---------- examples ---------- */
  function renderExamples() {
    el.exCap.innerHTML = 'Examples come from the fact-check index, so an exact match is expected (' + term('similarity', 'similarity') + ' about 1.0). Paste your own text to test a claim the index has not seen.';
    el.exBtn.disabled = ex.loading; el.exSpin.hidden = !ex.loading; el.exBtn.querySelector('svg').style.display = ex.loading ? 'none' : '';
    el.exBtnT.textContent = ex.loading ? 'Loading examples' : 'Refresh examples';
    el.exList.setAttribute('aria-busy', String(ex.loading));
    if (ex.error && !ex.loading) {
      el.exList.innerHTML = '<p class="ex-err" role="alert">Examples could not be loaded. <button type="button" class="link-btn" data-ex-retry>Try again</button></p>'; return;
    }
    if (ex.loading && !ex.items.length) { el.exList.innerHTML = '<div class="ex-sk" aria-hidden="true"><span class="skel"></span><span class="skel"></span><span class="skel"></span><span class="skel"></span></div>'; return; }
    el.exList.style.opacity = ex.loading ? '.5' : '';
    el.exList.innerHTML = ex.items.map(function (it, i) {
      return '<button type="button" class="ex-btn" data-i="' + i + '" aria-pressed="' + (i === ex.sel) + '"><span>' + esc(it.claim) + '</span><small>' + esc(L.site(it.site)) + ' rated it ' + esc(L.verdict(it.verdict_norm).toLowerCase()) +
        (i === ex.sel ? ' - select again to check' : '') + '</small></button>';
    }).join('');
  }
  function loadExamples(fromRefresh) {
    if (ex.loading) return;
    ex.loading = true; ex.error = false; ex.sel = -1; ex.refocus = !!fromRefresh; renderExamples();
    F.api('/api/examples?n=4').then(function (d) { ex.items = (d.items || []).filter(function (x) { return x && x.claim; }); ex.error = !ex.items.length; })
      .catch(function () { ex.error = true; })
      .then(function () {
        ex.loading = false; renderExamples();
        if (ex.refocus) { el.exBtn.focus({ preventScroll: true }); ex.refocus = false; }
        if (fromRefresh) announce(ex.error ? 'Examples could not be loaded.' : 'Showing ' + ex.items.length + ' different examples.');
      });
  }

  /* ---------- events ---------- */
  el.form.addEventListener('submit', function (e) { e.preventDefault(); run(el.ta.value); });
  el.ta.addEventListener('keydown', function (e) { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); run(el.ta.value); } });
  el.ta.addEventListener('input', function () {
    el.count.textContent = el.ta.value.length + ' / 5000'; if (el.msg.textContent) el.msg.textContent = '';
    if (ex.sel >= 0 && ex.items[ex.sel] && ex.items[ex.sel].claim !== el.ta.value) { ex.sel = -1; renderExamples(); }
  });
  el.rows.addEventListener('click', function (e) { var r = e.target.closest('.row[data-id]'); if (r) { C.setCursor(el.rows, r, false); Shell.navigate('#check/' + r.getAttribute('data-id')); } });
  C.rowNav(el.rows, function (r) { Shell.navigate('#check/' + r.getAttribute('data-id')); });
  el.exList.addEventListener('click', function (e) {
    if (e.target.closest('[data-ex-retry]')) { loadExamples(true); return; }
    var b = e.target.closest('.ex-btn'); if (!b) return;
    var i = Number(b.getAttribute('data-i')), it = ex.items[i]; if (!it) return;
    if (ex.sel === i && el.ta.value === it.claim) { run(it.claim); return; }
    ex.sel = i; el.ta.value = it.claim; el.ta.dispatchEvent(new Event('input')); ex.sel = i; renderExamples();
    var again = el.exList.querySelector('.ex-btn[data-i="' + i + '"]'); if (again) again.focus({ preventScroll: true });
  });
  el.exBtn.addEventListener('click', function () { loadExamples(true); });
  el.detail.addEventListener('click', function (e) {
    if (C.onDetailClick(e)) return;
    var b = e.target.closest('[data-act]'); if (!b) return;
    var act = b.getAttribute('data-act'), en = entry(cur);
    if (act === 'back') { if (routeCount > 1 && history.length > 1) history.back(); else Shell.navigate('#check', true); }
    else if (en && act === 'cancel') cancel(en);
    else if (en && act === 'retry' && !busy) startEngine(en);
  });
  el.detail.addEventListener('toggle', function (e) { var d = e.target; if (d && d.classList && d.classList.contains('cc-how')) { var en = entry(cur); if (en) { en.howOpen = d.open; } } }, true);
  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape' && Shell.current().route === 'check' && MOBILE.matches && cur && !e.defaultPrevented) { e.preventDefault(); if (routeCount > 1 && history.length > 1) history.back(); else Shell.navigate('#check', true); }
  });

  /* the Workbench (workbench.js) shows a stored /api/analyse-shaped result in this same two-step layout */
  window.ClaimCheck = {
    classBars: classBarsHTML,
    renderResult: function (res, o) {
      o = o || {};
      var en = { id: 'wb', text: String(o.text || ''), at: o.at || Date.now(), idx: res.index, res: res, eng: res.engine && res.engine.state, kicker: o.kicker || '', howOpen: !!o.howOpen, retryAt: 0 };
      return { stepper: stepperHTML(en), s1: s1HTML(en), s2: s2HTML(en), how: howHTML(en), label: step1Of(en).label, plaus: step1Of(en).plausibility, techs: techCount(en) };
    }
  };
  Shell.on(function () { routeCount++; });
  Shell.register('check', {
    enter: function (arg) {
      if (!ex.started) { ex.started = true; loadExamples(false); }
      var en = arg && entry(arg);
      if (en) showEntry(en, true);
      else { if (arg) Shell.navigate('#check', true); clearView(); }
    }
  });
  renderExamples(); renderRows();
})();
