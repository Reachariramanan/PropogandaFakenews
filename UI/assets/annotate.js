'use strict';
/* FNP annotation UI. All offsets are CODE POINTS into item.text (== Python str indices), never UTF-16 units.
   Scraped text is untrusted: every node is built with textContent / createElement, never innerHTML. */

const $ = (s, r = document) => r.querySelector(s);
const ls = {get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
            set(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* private mode */ } }};

function h(tag, attrs, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === false || v == null) continue;
    if (k === 'class') e.className = v;
    else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
    else if (k === 'style') e.style.cssText = v;
    else e.setAttribute(k, v === true ? '' : v);
  }
  for (const c of kids.flat()) if (c != null && c !== false) e.append(c.nodeType ? c : document.createTextNode(String(c)));
  return e;
}

// ------------------------------------------------------------------ state + api
const S = {user: null, token: ls.get('annot_token') || '', tax: null, data: null, layer: 'claim', pending: null, progress: null, idleSec: 120,
           editing: null, reselect: null, showOther: false, focus: null, lastTick: Date.now(), lastInput: 0, activeMs: 0, view: 'annotate', days: 30};

async function api(path, opt = {}) {
  const r = await fetch('/api/annot/' + path, {
    method: opt.method || 'GET',
    headers: {'Content-Type': 'application/json', ...(S.token ? {Authorization: 'Bearer ' + S.token} : {})},
    body: opt.body ? JSON.stringify(opt.body) : undefined});
  let j = null; try { j = await r.json(); } catch (e) { /* non-json */ }
  if (!r.ok) {
    const err = new Error((j && j.error) || ('HTTP ' + r.status)); err.status = r.status; err.code = j && j.code;
    if (r.status === 401 && path !== 'auth/login' && S.user) { signedOut('Your session ended: please sign in again.'); }
    if (r.status === 403 && err.code === 'must_change_password') { passwordView(true); }
    throw err;
  }
  return j;
}

function toast(msg, err) {
  const t = $('#toast'); t.textContent = msg; t.className = 'toast' + (err ? ' err' : '');
  clearTimeout(toast.t); toast.t = setTimeout(() => t.classList.add('hidden'), err ? 5000 : 2200);
}
const guard = fn => async (...a) => { try { return await fn(...a); } catch (e) { if (e.status !== 401 && e.code !== 'must_change_password') toast(e.message, true); } };

// active time: the item timer counts only while the tab is visible and the annotator interacted in the last 30 s;
// the heartbeat tells the server (every 30 s) whether there was input within the idle window, which is what counts as work time
for (const ev of ['mousemove', 'keydown', 'mousedown', 'scroll', 'touchstart']) addEventListener(ev, () => {
  const now = Date.now(); S.lastInput = now;
  if (document.visibilityState === 'visible') S.activeMs += Math.min(now - S.lastTick, 30000); S.lastTick = now; }, {passive: true});
setInterval(() => {
  if (!S.user || S.user.must_change_password) return;
  const active = document.visibilityState === 'visible' && Date.now() - S.lastInput < S.idleSec * 1000;
  api('heartbeat', {method: 'POST', body: {active}}).then(() => { if (active) tickProgress(); }).catch(() => {});
}, 30000);

// replaceChildren() would print the string "null" for an absent child: drop those here
const setView = (...kids) => $('#view').replaceChildren(...kids.flat().filter(x => x != null && x !== false));

// ------------------------------------------------------------------ formatting helpers
const fmtMin = m => m == null ? '–' : m >= 60 ? `${Math.floor(m / 60)} h ${String(Math.round(m % 60)).padStart(2, '0')} m` : `${Math.round(m)} m`;
const fmtSec = s => fmtMin(Math.round((s || 0) / 60));
const fmtTs = x => x ? new Date(x).toLocaleString([], {dateStyle: 'medium', timeStyle: 'short'}) : '–';
const fmtDay = x => x ? new Date(x + 'T00:00:00').toLocaleDateString([], {weekday: 'short', day: 'numeric', month: 'short'}) : '';
const ago = x => { if (!x) return 'never'; const s = (Date.now() - new Date(x)) / 1000;
  return s < 90 ? 'just now' : s < 3600 ? Math.round(s / 60) + ' min ago' : s < 86400 ? Math.round(s / 3600) + ' h ago' : Math.round(s / 86400) + ' d ago'; };
const num = n => n == null ? '–' : Number(n).toLocaleString();
const bar = (pct, cls) => h('div', {class: 'bar ' + (cls || '')}, h('i', {style: `width:${Math.max(0, Math.min(100, pct || 0))}%`}));
const stat = (label, value, sub) => h('div', {class: 'card stat'}, h('div', {class: 'muted small'}, label), h('h2', {}, value), sub ? h('div', {class: 'small muted'}, sub) : null);
const tbl = (cols, rows, onrow) => h('table', {}, h('thead', {}, h('tr', {}, cols.map(c => h('th', {class: c.num ? 'num' : ''}, c.h)))),
  h('tbody', {}, rows.map(r => h('tr', {class: onrow ? 'click' : '', onclick: onrow ? () => onrow(r) : null},
    cols.map(c => h('td', {class: c.num ? 'num' : ''}, c.f ? c.f(r) : (r[c.k] == null ? '' : String(r[c.k]))))))));

// ------------------------------------------------------------------ colours
const hue = (i, n) => Math.round(i * 360 / n);
const colorOf = (layer, code) => {
  const list = layer === 'claim' ? S.tax.claim_types : S.tax.techniques;
  const i = Math.max(0, list.findIndex(x => x.code === code));
  const dark = document.documentElement.dataset.theme === 'dark';
  return `hsl(${hue(i, list.length)} ${dark ? 60 : 65}% ${dark ? 62 : 46}%)`;
};
const tint = c => c.replace('hsl(', 'hsla(').replace(')', ' / .22)');
const typeLabel = (layer, code) => ((layer === 'claim' ? S.tax.claim_types : S.tax.techniques).find(x => x.code === code) || {label: code}).label;

// ------------------------------------------------------------------ offsets
const cps = () => S.data.item._cps || (S.data.item._cps = Array.from(S.data.item.text));
const cpSlice = (a, b) => cps().slice(a, b).join('');
function trimSpan(s, e) {
  const c = cps();
  while (s < e && /\s/.test(c[s])) s++;
  while (e > s && /\s/.test(c[e - 1])) e--;
  return [s, e];
}
/** Map a DOM (node, offset) inside #text to a code-point offset. */
function cpOffset(node, off) {
  const root = $('#text');
  if (node === root) {                       // element-level boundary (e.g. triple click): child index
    const kids = root.childNodes;
    if (off >= kids.length) return cps().length;
    return +kids[off].dataset.s;
  }
  const seg = (node.nodeType === 3 ? node.parentElement : node).closest('.seg');
  if (!seg || !root.contains(seg)) return null;
  if (node.nodeType === 3) return +seg.dataset.s + Array.from(node.data.slice(0, off)).length;
  return off === 0 ? +seg.dataset.s : +seg.dataset.e;
}

// ------------------------------------------------------------------ text rendering
function layerSpans(layer) { return layer === 'claim' ? S.data.claims : S.data.propaganda; }
function spanColor(layer, sp) { return colorOf(layer, layer === 'claim' ? sp.claim_type : sp.techniques[0]); }

function renderText() {
  const root = $('#text'); if (!root || !S.data) return;
  root.replaceChildren();
  const L = S.layer, other = L === 'claim' ? 'propaganda' : 'claim';
  const items = [];  // {s,e,kind,color,id}
  for (const sp of layerSpans(L)) items.push({s: sp.start, e: sp.end, kind: 'own', color: spanColor(L, sp), id: sp.span_id});
  for (const sg of S.data.suggestions.filter(x => x.layer === L)) items.push({s: sg.start, e: sg.end, kind: 'sugg', color: null, id: sg.suggestion_id});
  if (S.showOther) for (const sp of layerSpans(other)) items.push({s: sp.start, e: sp.end, kind: 'other', color: spanColor(other, sp), id: sp.span_id});
  if (S.pending) items.push({s: S.pending.start, e: S.pending.end, kind: 'focus', color: null});
  if (S.focus) items.push({s: S.focus.start, e: S.focus.end, kind: 'focus', color: null});
  const n = cps().length;
  const cuts = new Set([0, n]); for (const it of items) { cuts.add(it.s); cuts.add(it.e); }
  const pts = [...cuts].filter(x => x >= 0 && x <= n).sort((a, b) => a - b);
  for (let i = 0; i < pts.length - 1; i++) {
    const a = pts[i], b = pts[i + 1];
    const cov = items.filter(it => it.s <= a && it.e >= b);
    const seg = h('span', {class: 'seg', 'data-s': a, 'data-e': b}, cpSlice(a, b));
    const own = cov.filter(c => c.kind === 'own'), oth = cov.filter(c => c.kind === 'other');
    const shadows = [];
    own.slice(0, 4).forEach((c, k) => shadows.push(`inset 0 ${-(2 + 3 * k)}px 0 ${c.color}`));
    if (own.length) { seg.style.background = tint(own[0].color); seg.classList.add('hasspan'); seg.dataset.ids = own.map(c => c.id).join(','); }
    else if (oth.length) { seg.style.background = tint(oth[0].color); seg.classList.add('other'); }
    if (shadows.length) seg.style.boxShadow = shadows.join(',');
    if (cov.some(c => c.kind === 'sugg')) seg.classList.add('sugg');
    if (cov.some(c => c.kind === 'focus')) seg.classList.add('focus');
    root.append(seg);
  }
}

// ------------------------------------------------------------------ popover (create / edit a span)
function closePop() { const p = $('#pop'); p.classList.add('hidden'); p.replaceChildren(); S.pending = null; S.editing = null; S.reselect = null; renderText(); }

function placePop(rect) {
  const p = $('#pop'); p.classList.remove('hidden');
  const w = p.offsetWidth, hgt = p.offsetHeight;
  let x = Math.min(Math.max(8, rect.left), innerWidth - w - 8);
  let y = rect.bottom + 8; if (y + hgt > innerHeight - 8) y = Math.max(8, rect.top - hgt - 8);
  p.style.left = x + 'px'; p.style.top = y + 'px';
}

function chipSet(options, selected, multi, onChange, hotkeys) {
  const wrap = h('div', {class: 'chips'});
  const draw = () => { wrap.replaceChildren(...options.map((o, i) => h('span', {
    class: 'chip' + (selected.has(o.code) ? ' on' : ''), title: o.description || '', tabindex: 0,
    onclick: () => { if (multi) { selected.has(o.code) ? selected.delete(o.code) : selected.add(o.code); } else { selected.clear(); selected.add(o.code); } draw(); onChange(); }},
    hotkeys && i < 9 ? h('kbd', {}, i + 1) : null, o.label))); };
  draw(); wrap.redraw = draw; return wrap;
}

/** form = {layer, start, end, id?(edit), init?: {...}, suggestion_id?} */
function openPop(form, rect) {
  S.pending = form.id ? null : {start: form.start, end: form.end};
  S.focus = null; renderText();
  const p = $('#pop'); p.replaceChildren();
  const L = form.layer, init = form.init || {};
  const sel = new Set(L === 'claim' ? (init.claim_type ? [init.claim_type] : []) : (init.techniques || []));
  let cw = init.check_worthy || null;
  const taNorm = h('textarea', {rows: 2, placeholder: 'Optional: self-contained restatement of the claim'}, init.normalised_claim || '');
  const inSpk = h('input', {type: 'text', placeholder: 'Optional: who made the claim', value: init.speaker || ''});
  const cwBox = h('div', {class: 'chips'});
  const drawCw = () => cwBox.replaceChildren(...[['yes', 'Check-worthy'], ['no', 'Not check-worthy'], ['unsure', 'Unsure']].map(([v, l]) =>
    h('span', {class: 'chip' + (cw === v ? ' on' : ''), onclick: () => { cw = v; drawCw(); validate(); }}, h('kbd', {}, v[0]), l)));
  const addBtn = h('button', {class: 'primary', onclick: guard(save)}, form.id ? 'Update' : 'Add');
  const chips = chipSet(L === 'claim' ? S.tax.claim_types : S.tax.techniques, sel, L === 'propaganda', () => validate(), L === 'claim');
  function validate() { addBtn.disabled = !(sel.size && (L === 'propaganda' || cw)); }
  async function save() {
    const [s, e] = [form.start, form.end];
    const body = L === 'claim'
      ? {start: s, end: e, claim_type: [...sel][0], check_worthy: cw, normalised_claim: taNorm.value, speaker: inSpk.value, suggestion_id: form.suggestion_id}
      : {start: s, end: e, techniques: [...sel], suggestion_id: form.suggestion_id};
    const base = L === 'claim' ? 'claims' : 'propaganda';
    if (form.id) await api(`${base}/${form.id}`, {method: 'PUT', body}); else await api(`item/${S.data.item.item_id}/${base}`, {method: 'POST', body});
    closePop(); await reload(); toast(form.id ? 'Updated' : 'Added');
  }
  p.append(h('div', {},
    form.siblings && form.siblings.length > 1 ? h('div', {class: 'tabs'}, form.siblings.map(sb => h('button', {class: sb.span_id === form.id ? 'primary' : '', onclick: () => editSpan(L, sb.span_id, rect)},
      `${sb.start}–${sb.end}`))) : null,
    h('div', {class: 'sel'}, cpSlice(form.start, form.end)),
    h('h4', {}, L === 'claim' ? 'Claim type (form of the claim)' : 'Propaganda technique(s)'), chips,
    L === 'claim' ? [h('h4', {}, 'Check-worthiness (separate from type)'), cwBox, h('h4', {}, 'Normalised claim'), taNorm, h('h4', {}, 'Speaker'), inSpk] : null,
    h('div', {class: 'btns'},
      form.id ? h('button', {class: 'warn', onclick: guard(async () => { await api(`${L === 'claim' ? 'claims' : 'propaganda'}/${form.id}`, {method: 'DELETE'}); closePop(); await reload(); toast('Deleted'); })}, 'Delete') : null,
      form.id ? h('button', {onclick: () => { S.reselect = form; $('#pop').classList.add('hidden'); toast('Now select the new range in the text'); }}, 'Re-select range') : null,
      h('button', {onclick: closePop}, 'Cancel'), addBtn)));
  drawCw(); validate();
  placePop(rect || {left: 40, bottom: 120, top: 120});
  S.popKeys = ev => {
    if (ev.target.matches('textarea,input')) { if (ev.key === 'Escape') closePop(); return; }
    if (ev.key === 'Escape') return closePop();
    if (ev.key === 'Enter' && !addBtn.disabled) { ev.preventDefault(); return guard(save)(); }
    if (L === 'claim') {
      if (/^[1-9]$/.test(ev.key) && S.tax.claim_types[+ev.key - 1]) { sel.clear(); sel.add(S.tax.claim_types[+ev.key - 1].code); chips.redraw(); validate(); }
      if ('ynu'.includes(ev.key) && ev.key.length === 1) { cw = {y: 'yes', n: 'no', u: 'unsure'}[ev.key]; drawCw(); validate(); }
    }
  };
}

function editSpan(layer, id, rect) {
  const sp = layerSpans(layer).find(x => x.span_id === id); if (!sp) return;
  S.layer = layer; S.focus = null;
  const covering = layerSpans(layer).filter(x => x.start < sp.end && x.end > sp.start);
  openPop({layer, id, start: sp.start, end: sp.end, siblings: covering, init: sp}, rect || $('#text').getBoundingClientRect());
}

// ------------------------------------------------------------------ selection handling
function onTextMouseUp(ev) {
  if (!S.data || S.view !== 'annotate' || $('#pop').contains(ev.target)) return;
  const sel = getSelection();
  if (!sel || sel.rangeCount === 0) return;
  if (sel.isCollapsed) {                       // click: edit the span under the cursor
    const seg = ev.target.closest && ev.target.closest('#text .seg.hasspan');
    if (seg) { const ids = seg.dataset.ids.split(',').map(Number); editSpan(S.layer, ids[ids.length - 1], seg.getBoundingClientRect()); }
    return;
  }
  const r = sel.getRangeAt(0);
  let a = cpOffset(r.startContainer, r.startOffset), b = cpOffset(r.endContainer, r.endOffset);
  if (a == null || b == null) return;
  [a, b] = trimSpan(Math.min(a, b), Math.max(a, b));
  if (b <= a) return;
  const rect = r.getBoundingClientRect(); sel.removeAllRanges();
  if (S.reselect) {                            // re-select the range of an existing span
    const f = S.reselect; S.reselect = null; f.start = a; f.end = b; S.editing = f;
    openPop(f, rect); return;
  }
  if (S.data.assignment && S.data.assignment.status === 'skipped') return toast('Re-open this item to edit it', true);
  openPop({layer: S.layer, start: a, end: b}, rect);
}

// ------------------------------------------------------------------ sidebar
function layerStatus(l) {
  const r = S.data.results[l]; const n = layerSpans(l).length;
  return r === 'annotated' ? `✓ ${n} span${n === 1 ? '' : 's'}` : r === 'none' ? '✓ none' : n ? `${n} span${n === 1 ? '' : 's'} (not submitted)` : 'not started';
}

function acceptSuggestion(sg, review, rect) {
  if (sg.layer === 'propaganda' && !review) {
    return guard(async () => { await api(`item/${S.data.item.item_id}/propaganda`, {method: 'POST', body: {start: sg.start, end: sg.end, techniques: [sg.label], suggestion_id: sg.suggestion_id}}); await reload(); })();
  }
  const init = sg.layer === 'claim'
    ? {claim_type: sg.label, normalised_claim: (sg.payload || {}).normalised_claim || '', speaker: (sg.payload || {}).speaker || ''}
    : {techniques: [sg.label]};
  S.layer = sg.layer;
  openPop({layer: sg.layer, start: sg.start, end: sg.end, init, suggestion_id: sg.suggestion_id}, rect || $('#text').getBoundingClientRect());
}

function sidebar() {
  const L = S.layer, d = S.data;
  const spans = layerSpans(L);
  const sg = d.suggestions.filter(x => x.layer === L);
  const noneLabel = L === 'claim' ? 'No claims in this text' : 'No propaganda in this text';
  return h('div', {class: 'side sticky'},
    h('div', {class: 'layers'}, ['claim', 'propaganda'].map(l => h('button', {class: S.layer === l ? 'on' : '', onclick: () => { closePop(); S.layer = l; S.focus = null; render(); }},
      l === 'claim' ? 'Claims' : 'Propaganda', h('span', {class: 'st'}, layerStatus(l))))),
    d.adjudication ? adjudicationCard(L, d) : null,
    h('div', {class: 'card'},
      h('b', {}, L === 'claim' ? 'Your claims' : 'Your propaganda spans'),
      spans.length ? spans.map(sp => h('div', {class: 'spanrow', onmouseenter: () => { S.focus = {start: sp.start, end: sp.end}; renderText(); }, onmouseleave: () => { S.focus = null; renderText(); }},
        h('i', {class: 'dot', style: `background:${spanColor(L, sp)}`}),
        h('div', {class: 'tx'}, h('q', {}, sp.text),
          L === 'claim' ? [h('span', {class: 'tag'}, typeLabel('claim', sp.claim_type)), h('span', {class: 'tag'}, sp.check_worthy === 'yes' ? 'check-worthy' : sp.check_worthy === 'no' ? 'not check-worthy' : 'unsure'),
                           sp.normalised_claim ? h('div', {class: 'small muted'}, sp.normalised_claim) : null]
                         : sp.techniques.map(t => h('span', {class: 'tag'}, typeLabel('propaganda', t)))),
        h('div', {class: 'acts'}, h('button', {onclick: () => editSpan(L, sp.span_id)}, 'edit')))) : h('div', {class: 'hint'}, L === 'claim' ? 'Select text and choose a claim type.' : 'Select text and choose one or more techniques.')),
    sg.length ? h('div', {class: 'card'}, h('b', {}, 'Model suggestions'), h('div', {class: 'hint'}, 'Corrections welcome: accept, edit, or reject.'),
      sg.map(s => h('div', {class: 'spanrow sg', onmouseenter: () => { S.focus = {start: s.start, end: s.end}; renderText(); }, onmouseleave: () => { S.focus = null; renderText(); }},
        h('div', {class: 'tx'}, h('q', {}, s.text), h('span', {class: 'tag'}, typeLabel(L, s.label)), s.confidence != null ? h('span', {class: 'small muted'}, ' ' + s.confidence.toFixed(2)) : null),
        h('div', {class: 'acts'},
          h('button', {onclick: ev => acceptSuggestion(s, L === 'claim', ev.target.getBoundingClientRect())}, L === 'claim' ? 'review' : 'accept'),
          h('button', {onclick: guard(async () => { await api(`item/${d.item.item_id}/suggestions/${s.suggestion_id}/reject`, {method: 'POST', body: {}}); await reload(); })}, 'reject'))))) : null,
    h('div', {class: 'card'},
      h('label', {class: 'small'}, h('input', {type: 'checkbox', checked: S.showOther, onchange: ev => { S.showOther = ev.target.checked; renderText(); }}), ' show the other layer (read-only)'),
      h('div', {class: 'actions'},
        h('button', {class: 'primary', disabled: !spans.length, onclick: guard(() => submit('annotated'))}, `Submit ${L === 'claim' ? 'claims' : 'propaganda'}`),
        h('button', {disabled: spans.length > 0, title: spans.length ? 'delete your spans first' : '', onclick: guard(() => submit('none'))}, noneLabel)),
      d.item.is_blind ? h('div', {class: 'hint'}, 'No model suggestions are shown for this item.') : null),
    h('div', {class: 'actions'},
      h('button', {onclick: guard(skip), disabled: d.assignment && d.assignment.status === 'skipped'}, 'Skip item…'),
      h('button', {onclick: () => { S.view = 'mine'; route(); }}, 'My items')));
}

async function submit(outcome) {
  const layer = S.layer;
  const res = await api(`item/${S.data.item.item_id}/submit`, {method: 'POST', body: {layer, outcome, active_ms: S.activeMs}});
  S.activeMs = 0;
  await reload(); tickProgress();
  if (res.batch_completed) toast(`Batch ${res.batch_completed} complete: the next batch has been added. Press N for the next item`);
  else if (res.item_complete) toast('Item complete: press N for the next one');
  else { const other = layer === 'claim' ? 'propaganda' : 'claim'; if (!S.data.results[other]) { S.layer = other; render(); toast(`Saved. Now the ${other} layer`); } }
}

const SKIP_REASONS = ['Not enough text / unreadable', 'Language I cannot annotate', 'Spam or non-content', 'Offensive or disturbing', 'Other'];

// Radio list of reasons; a free-text box appears only when "Other" is chosen. Resolves to the reason string, or null if cancelled.
function askSkipReason() {
  return new Promise(resolve => {
    const name = 'skipreason';
    const note = h('textarea', {rows: 3, maxlength: 300, placeholder: 'Describe the reason (required)', style: 'width:100%;margin-top:8px'});
    const noteWrap = h('div', {class: 'hidden'}, note);
    const err = h('div', {class: 'err'});
    const radios = SKIP_REASONS.map(r => h('input', {type: 'radio', name, value: r}));
    const sync = () => { const other = radios.find(x => x.checked)?.value === 'Other'; noteWrap.classList.toggle('hidden', !other); if (other) note.focus(); err.textContent = ''; };
    radios.forEach(x => x.addEventListener('change', sync));
    const done = v => { document.removeEventListener('keydown', onKey, true); ov.remove(); resolve(v); };
    const ok = () => {
      const sel = radios.find(x => x.checked);
      if (!sel) { err.textContent = 'Pick a reason.'; return; }
      if (sel.value === 'Other') { const t = note.value.trim(); if (!t) { err.textContent = 'Please describe the reason.'; note.focus(); return; } return done('Other: ' + t); }
      done(sel.value);
    };
    const onKey = ev => { if (ev.key === 'Escape') { ev.stopPropagation(); done(null); } else if (ev.key === 'Enter' && ev.target !== note) { ev.preventDefault(); ev.stopPropagation(); ok(); } };
    document.addEventListener('keydown', onKey, true);
    const ov = h('div', {class: 'modal', role: 'dialog', 'aria-label': 'Reason for skipping', onmousedown: ev => { if (ev.target === ov) done(null); }},
      h('div', {class: 'modal-box'},
        h('h3', {style: 'margin:0 0 8px'}, 'Reason for skipping'),
        ...SKIP_REASONS.map((r, i) => h('label', {class: 'opt'}, radios[i], ' ', r)),
        noteWrap, err,
        h('div', {class: 'btns'}, h('button', {onclick: () => done(null)}, 'Cancel'), h('button', {class: 'primary', onclick: ok}, 'Skip item'))));
    document.body.append(ov); radios[0].focus();
  });
}

async function skip() {
  const reason = await askSkipReason();
  if (!reason) return;
  await api(`item/${S.data.item.item_id}/skip`, {method: 'POST', body: {reason}});
  toast('Skipped'); await nextItem();
}

// ------------------------------------------------------------------ views
async function reload() { S.data = await api('item/' + S.data.item.item_id); render(); }

async function nextItem() {
  closePop();
  const r = await api('next');
  if (!r.item_id) { S.data = null; S.view = 'annotate'; renderEmpty(); return; }
  S.data = r.item; S.layer = 'claim'; S.activeMs = 0; S.focus = null; tickProgress();
  if (S.data.results.claim && !S.data.results.propaganda) S.layer = 'propaganda';
  render();
}

function renderEmpty() {
  if (S.view !== 'annotate') return;
  setView(h('div', {class: 'card'}, h('h2', {}, 'No items available for you right now'),
    h('p', {class: 'muted'}, 'Everything fetched has been annotated, skipped by you, or reserved by another annotator. Ask an admin to fill the item pool (Admin > Data).'),
    h('button', {class: 'primary', onclick: guard(nextItem)}, 'Check again')));
}

function render() {
  if (S.view !== 'annotate') return;                       // a slow response must not paint over another tab
  const d = S.data; if (!d) return renderEmpty();
  const it = d.item, done = d.results.claim && d.results.propaganda;
  const m = h('div', {class: 'meta'}, h('span', {class: 'badge'}, it.kind + ' · ' + it.platform), h('span', {class: 'badge'}, it.lang_group + (it.lang && it.lang !== it.lang_group ? ' (' + it.lang + ')' : '')),
    h('span', {class: 'muted'}, it.period), it.author ? h('span', {class: 'muted'}, '· ' + it.author) : null,
    it.url && /^https?:\/\//.test(it.url) ? h('a', {href: it.url, target: '_blank', rel: 'noopener noreferrer'}, 'source ↗') : null,
    h('span', {class: 'muted'}, `· #${it.item_id}`));
  const legend = h('div', {class: 'legend'}, (S.layer === 'claim' ? S.tax.claim_types : S.tax.techniques).map(x => h('span', {title: x.description}, h('i', {class: 'dot', style: `background:${colorOf(S.layer, x.code)}`}), x.label)));
  const left = h('div', {},
    done ? h('div', {class: 'banner'}, h('span', {}, 'Both layers submitted for this item.'), h('button', {class: 'primary', onclick: guard(nextItem)}, 'Next item (N) →')) : null,
    d.assignment && d.assignment.status === 'skipped' ? h('div', {class: 'banner'}, h('span', {}, 'Skipped: ' + (d.assignment.skip_reason || '')), h('button', {onclick: guard(async () => { await api(`item/${it.item_id}/reopen`, {method: 'POST', body: {}}); await reload(); })}, 'Re-open')) : null,
    h('div', {class: 'card'}, m, h('div', {id: 'text'}), legend,
      h('div', {class: 'hint'}, 'Select text with the mouse to annotate it · click a highlight to edit/delete · overlapping spans are allowed · keys: C claims, P propaganda' + (done ? ', N next' : ''))));
  setView(h('div', {class: 'ann'}, left, sidebar()));
  renderText();
}

async function viewMine() {
  const r = await api('mine');
  setView(h('div', {class: 'card'}, h('h2', {}, 'My items'), h('table', {},
    h('thead', {}, h('tr', {}, ['#', 'status', 'source', 'lang', 'claims', 'prop.', 'text'].map(x => h('th', {}, x)))),
    h('tbody', {}, r.items.map(i => h('tr', {class: 'click', onclick: guard(async () => { S.data = await api('item/' + i.item_id); S.view = 'annotate'; S.layer = 'claim'; setTabs(); render(); })},
      h('td', {}, i.item_id), h('td', {}, h('span', {class: 'badge ' + (i.status === 'done' ? 'done' : i.status === 'skipped' ? 'stalled' : 'running')}, i.status)),
      h('td', {}, i.platform), h('td', {}, i.lang_group), h('td', {class: 'num'}, i.n_claims), h('td', {class: 'num'}, i.n_prop), h('td', {class: 'small'}, i.preview)))))));
}

// ------------------------------------------------------------------ annotator: my progress
function drawProgress() {
  const el = $('#prog'); if (!el) return;
  const p = S.progress; if (!S.user || !p) { el.replaceChildren(); return; }
  const b = p.batch;
  el.replaceChildren(...[
    b ? h('span', {class: 'chip-s', title: 'Items finished in your current batch; the next batch is added when it is complete'}, `Batch ${b.seq}: ${b.finished}/${b.size}`) : null,
    h('span', {class: 'chip-s', title: 'Active working time today'}, `today ${fmtMin(p.minutes_today)}`)].filter(Boolean));
}
async function tickProgress() { try { S.progress = await api('progress'); drawProgress(); } catch (e) { /* header chip only */ } }

async function viewMe() {
  const p = await api('progress'); S.progress = p; drawProgress();
  const b = p.batch;
  setView(
    h('div', {class: 'grid g4'},
      stat('Current batch', b ? `${b.finished} / ${b.size}` : 'none yet', b ? `batch #${b.seq}` : 'one is built when you open Annotate'),
      stat('Finished today', p.done_today, `${fmtMin(p.minutes_today)} active`),
      stat('Finished in total', num(p.done), `${p.skipped} skipped · ${p.batches_completed} batches complete`),
      stat('Total work time', fmtMin(p.minutes_total))),
    b ? h('div', {class: 'card'}, h('b', {}, `Batch ${b.seq}`), bar(100 * b.finished / b.size),
      h('div', {class: 'small muted'}, `${b.size - b.finished} items left. When all are finished the next batch is added automatically.`)) : null,
    h('div', {class: 'card'}, h('b', {}, 'Team progress'), bar(100 * p.team_done / p.target_items),
      h('div', {class: 'small muted'}, `${num(p.team_done)} of ${num(p.target_items)} items fully annotated (claims and propaganda) by the whole team.`)));
}

// ------------------------------------------------------------------ admin: team monitoring
function periodSelect(onchange) {
  const sel = h('select', {onchange: () => { S.days = +sel.value; onchange(); }},
    [[1, 'last 24 hours'], [7, 'last 7 days'], [30, 'last 30 days'], [90, 'last 90 days'], [0, 'all time']].map(([v, l]) => h('option', {value: v, selected: S.days === v}, l)));
  return sel;
}

async function viewTeam() {
  const t = await api('admin/team?days=' + S.days), T = t.totals;
  const dot = r => h('span', {class: 'pdot ' + (r.online ? 'on' : r.active ? '' : 'off'), title: r.online ? 'online now' : r.active ? 'last seen ' + ago(r.last_seen_at) : 'disabled'});
  setView(
    h('div', {class: 'row', style: 'justify-content:space-between;margin-bottom:10px'}, h('h2', {style: 'margin:0'}, 'Team'),
      h('div', {class: 'row'}, 'period', periodSelect(viewTeam), h('button', {onclick: viewTeam}, 'refresh'), h('span', {class: 'small muted'}, 'days are in ' + t.tz))),
    h('div', {class: 'grid g4'}, stat('Online now', T.online), stat('Active work time', fmtMin(T.active_minutes), `${T.logins} logins`),
      stat('Items finished', num(T.items_done), `${T.skipped} skipped`), stat('Spans created', num(T.claim_spans + T.propaganda_spans), `${num(T.claim_spans)} claim · ${num(T.propaganda_spans)} propaganda`)),
    h('div', {class: 'card', style: 'overflow-x:auto'}, tbl([
      {h: 'annotator', f: r => h('span', {}, dot(r), ' ', h('b', {}, r.username), r.role !== 'annotator' ? h('span', {class: 'tag'}, r.role) : null, r.locked ? h('span', {class: 'tag'}, 'locked') : null, !r.active ? h('span', {class: 'tag'}, 'disabled') : null)},
      {h: 'last login', f: r => h('span', {title: fmtTs(r.last_login_at)}, ago(r.last_login_at))},
      {h: 'logins', k: 'logins', num: 1},
      {h: 'logged in', f: r => fmtSec(r.logged_in_seconds), num: 1},
      {h: 'active work', f: r => fmtMin(r.active_minutes), num: 1},
      {h: 'today', f: r => `${fmtMin(r.active_minutes_today)} · ${r.items_done_today} items`, num: 1},
      {h: 'items done', k: 'items_done', num: 1},
      {h: 'skipped', k: 'skipped', num: 1},
      {h: 'claim spans', k: 'claim_spans', num: 1},
      {h: 'prop. spans', k: 'propaganda_spans', num: 1},
      {h: 'none labels', k: 'none_labels', num: 1},
      {h: 'items / active h', f: r => r.items_per_active_hour == null ? '–' : r.items_per_active_hour, num: 1},
      {h: 'batch', f: r => r.batch ? `#${r.batch.seq}: ${r.batch.finished}/${r.batch.size}` : '–', num: 1},
      {h: 'batches done', k: 'batches_completed', num: 1},
    ], t.annotators, r => { S.userId = r.annotator_id; S.view = 'user'; route(); })),
    h('div', {class: 'hint'}, 'Active work = minutes in which the annotator gave input (or saved a label); logged in = from sign-in to sign-out or last activity. Click a row for sessions, daily history and label mix.'));
}

async function viewUser() {
  const id = S.userId, d = await api(`admin/user/${id}?days=${S.days}`), a = d.annotator;
  const secret = h('div', {id: 'secret'});
  const uq = (await api('admin/quality/' + id)).row;
  const qcard = h('div', {class: 'card'}, h('h3', {}, 'Quality'), uq ? h('div', {class: 'row'},
    h('span', {}, 'median ', h('b', {}, uq.median_sec == null ? '–' : Math.round(uq.median_sec) + ' s'), ' per item'),
    h('span', {}, '"none" ', h('b', {}, `${pct0(uq.claim_none_rate)} / ${pct0(uq.propaganda_none_rate)}`)),
    h('span', {}, 'accepts suggestions ', h('b', {}, pct0(uq.suggestion_accept_rate))),
    h('span', {}, 'gold F1 ', h('b', {}, uq.gold ? `${f2(uq.gold.claim.f1)} / ${f2(uq.gold.propaganda.f1)}` : '–'), uq.gold ? ` (n=${uq.gold.n_items})` : ''),
    h('span', {}, 'partner F1 ', h('b', {}, uq.agreement ? `${f2(uq.agreement.claim_f1)} / ${f2(uq.agreement.propaganda_f1)}` : '–'), uq.agreement ? ` (n=${uq.agreement.n_items})` : ''),
    uq.flags.map(x => h('span', {class: 'tag warn'}, x))) : h('div', {class: 'hint'}, 'No finished work yet.'));
  const act = (label, body, after) => h('button', {onclick: guard(async () => {
    const r = await api('admin/users/' + id, {method: 'POST', body});
    if (r.password) secret.replaceChildren(h('div', {class: 'secret'}, h('div', {class: 'small muted'}, 'Temporary password (shown once; the user must change it at next login):'), h('code', {}, r.password)));
    toast('Done' + (r.released_items != null ? `: ${r.released_items} unopened items returned to the pool` : '')); viewUser(); })}, label);
  const npw = h('input', {type: 'password', autocomplete: 'new-password', placeholder: 'min 8 characters', style: 'width:170px'});
  const mustCh = h('input', {type: 'checkbox', checked: true});
  const bs = h('input', {type: 'number', min: 1, max: 5000, value: a.batch_size || '', placeholder: 'default', style: 'width:90px'});
  const maxMin = Math.max(1, ...d.daily.map(x => x.active_minutes || 0));
  setView(
    h('div', {class: 'row', style: 'justify-content:space-between;margin-bottom:10px'},
      h('h2', {style: 'margin:0'}, h('a', {class: 'click', onclick: () => { S.view = 'team'; route(); }}, 'Team'), ' › ', a.username, ' ', h('span', {class: 'tag'}, a.role), !a.active ? h('span', {class: 'tag'}, 'disabled') : null),
      h('div', {class: 'row'}, 'period', periodSelect(viewUser))),
    h('div', {class: 'card'}, h('div', {class: 'row'},
      h('span', {class: 'muted small'}, 'new password'), npw, showPw(npw), h('label', {class: 'small'}, mustCh, ' must change at next sign-in'),
      h('button', {class: 'primary', onclick: guard(async () => {
        if (!npw.value) { toast('Type the new password first, or use "Generate password"'); npw.focus(); return; }
        await api('admin/users/' + id, {method: 'POST', body: {new_password: npw.value, must_change: mustCh.checked}});
        npw.value = ''; toast('Password set for ' + a.username + '; their open sessions were signed out'); viewUser(); })}, 'Set password'),
      act('Generate password', {reset_password: true}),
      a.active ? act('Disable account', {active: false}) : act('Enable account', {active: true}),
      a.locked_until && new Date(a.locked_until) > new Date() ? act('Unlock', {unlock: true}) : null,
      h('button', {onclick: guard(async () => { const r = await api(`admin/users/${id}/release`, {method: 'POST', body: {}}); toast(`${r.released_items} unopened items returned to the pool`); viewUser(); })}, 'Release unopened items'),
      h('span', {class: 'muted small'}, 'batch size'), bs, h('button', {onclick: guard(async () => { await api('admin/users/' + id, {method: 'POST', body: {batch_size: bs.value ? +bs.value : null}}); toast('Saved'); viewUser(); })}, 'Set')),
      secret,
      h('div', {class: 'small muted', style: 'margin-top:6px'}, `Created ${fmtTs(a.created_at)} · last login ${fmtTs(a.last_login_at)}${a.must_change_password ? ' · has not yet replaced the temporary password' : ''}`)),
    qcard,
    h('div', {class: 'grid g2'},
      h('div', {class: 'card'}, h('h3', {}, 'Daily activity'), d.daily.length ? tbl([
        {h: 'day', f: r => fmtDay(r.day)},
        {h: 'active work', f: r => h('span', {class: 'inl'}, h('i', {style: `width:${Math.round(100 * (r.active_minutes || 0) / maxMin)}%`}), fmtMin(r.active_minutes || 0))},
        {h: 'items', f: r => r.items_done || 0, num: 1}, {h: 'claim', f: r => r.claim_spans || 0, num: 1}, {h: 'prop.', f: r => r.propaganda_spans || 0, num: 1}, {h: 'logins', f: r => r.logins || 0, num: 1},
      ], d.daily) : h('div', {class: 'hint'}, 'No activity in this period.')),
      h('div', {class: 'card'}, h('h3', {}, 'Batches'), d.batches.length ? tbl([
        {h: '#', k: 'seq', num: 1}, {h: 'done', k: 'done', num: 1}, {h: 'skipped', k: 'skipped', num: 1}, {h: 'size', k: 'size', num: 1},
        {h: 'started', f: r => fmtTs(r.created_at)}, {h: 'completed', f: r => r.completed_at ? fmtTs(r.completed_at) : 'open'},
      ], d.batches) : h('div', {class: 'hint'}, 'No batch yet.'),
      d.skip_reasons.length ? [h('h3', {}, 'Skip reasons'), tbl([{h: 'reason', k: 'skip_reason'}, {h: 'n', k: 'n', num: 1}], d.skip_reasons)] : null)),
    h('div', {class: 'card', style: 'overflow-x:auto'}, h('h3', {}, 'Sessions'), tbl([
      {h: 'signed in', f: r => fmtTs(r.created_at)}, {h: 'until', f: r => r.ended_at ? fmtTs(r.ended_at) : r.online ? 'online now' : 'last seen ' + fmtTs(r.last_seen_at)},
      {h: 'length', f: r => fmtSec(r.seconds), num: 1}, {h: 'active work', f: r => fmtMin(r.active_minutes), num: 1},
      {h: 'ended by', f: r => r.end_reason || (r.online ? '' : 'timed out')}, {h: 'address', k: 'ip'}, {h: 'browser', f: r => h('span', {class: 'small'}, r.ua || '')},
    ], d.sessions)),
    h('div', {class: 'grid g2'},
      h('div', {class: 'card'}, h('h3', {}, 'Claim types labelled (all time)'), tbl([{h: 'type', f: r => typeLabel('claim', r.code)}, {h: 'spans', k: 'n', num: 1}], d.claim_types),
        h('h3', {}, 'Techniques labelled (all time)'), tbl([{h: 'technique', f: r => typeLabel('propaganda', r.code)}, {h: 'spans', k: 'n', num: 1}], d.techniques)),
      h('div', {class: 'card'}, h('h3', {}, 'Recent sign-in events'), tbl([{h: 'when', f: r => fmtTs(r.ts)}, {h: 'event', k: 'outcome'}, {h: 'address', k: 'ip'}, {h: 'detail', k: 'detail'}], d.login_events))));
}

// ------------------------------------------------------------------ admin: coverage vs the 50k target
async function viewCoverage() {
  const c = await api('admin/coverage');
  const pct = 100 * c.done_items / c.target_items;
  const classes = layer => h('div', {class: 'card'},
    h('h3', {}, layer === 'claim' ? 'Claim extraction' : 'Propaganda spans + techniques'),
    h('div', {class: 'small muted'}, `${num(c.tasks[layer].annotated_items)} items with labels · ${num(c.tasks[layer].none_items)} explicit none (${c.tasks[layer].none_pct_of_finished}% of finished; target ${Math.round(100 * c.target_neg / c.target_items)}%)`),
    tbl([{h: 'class', f: r => typeLabel(layer, r.code)},
         {h: 'progress', f: r => r.target ? bar(r.pct, r.pct >= 100 ? 'full' : '') : h('span', {class: 'small muted'}, 'catch-all: no target')},
         {h: 'items', f: r => r.target ? `${num(r.done)} / ${num(r.target)}` : num(r.done), num: 1},
         {h: 'steered to', f: r => r.target ? h('span', {title: 'the target is capped at what the models say the pool can supply; the remaining imbalance is corrected by class weights in the export'}, num(r.feasible)) : '', num: 1},
         {h: '%', f: r => r.pct == null ? '' : r.pct, num: 1},
         {h: 'reserved (predicted)', f: r => r.in_flight_predicted, num: 1}], c.tasks[layer].classes));
  const preTotal = c.pre_annotated.n || 1;
  const share = (rows, key) => tbl([{h: key, k: key}, {h: 'finished', k: 'done', num: 1}, {h: 'in DB', k: 'items', num: 1}, {h: 'share of finished', f: r => (100 * r.share_done).toFixed(1) + '%', num: 1}], rows);
  setView(
    h('div', {class: 'row', style: 'justify-content:space-between;margin-bottom:10px'}, h('h2', {style: 'margin:0'}, 'Coverage'), h('button', {onclick: viewCoverage}, 'refresh')),
    h('div', {class: 'card'}, h('b', {}, `${num(c.done_items)} of ${num(c.target_items)} items fully annotated`), bar(pct),
      h('div', {class: 'small muted'}, c.items_per_active_hour_7d ? `Last 7 days: ${c.items_per_active_hour_7d} items per active hour → about ${num(c.annotator_hours_remaining)} annotator-hours to go.` : 'Not enough tracked work time yet for an estimate.',
        ' Target = fully annotated items per task; each item gets both layers.')),
    h('div', {class: 'grid g4'},
      ...c.pool.map(p => stat(`Pool · ${p.lang_group}`, num(p.unassigned), `unassigned of ${num(p.items)} items`)),
      stat('Pre-annotated pool', `${Math.round(100 * c.pre_annotated.enc / preTotal)}%`, `encoder ${num(c.pre_annotated.enc)} · Gemma ${num(c.pre_annotated.gemma)} of ${num(c.pre_annotated.n)}`)),
    c.pre_annotated.enc < 0.5 * c.pre_annotated.n ? h('div', {class: 'banner'}, h('span', {}, 'Less than half of the pool is pre-annotated. Class-balanced selection can only steer by predictions: run "Pre-annotate" (Admin > Data).')) : null,
    h('div', {class: 'hint'}, 'Imbalance is accepted: "steered to" is the part of each target the pool can plausibly supply; rare classes beyond it are handled by class weights in the export (class_weights.json). ',
      `Selection so far: ${Object.entries(c.selection || {}).map(([k, v]) => k + ' ' + num(v)).join(' · ') || 'none'}. `,
      `Excluded from the pool: ${Object.entries(c.excluded_by_reason || {}).map(([k, v]) => `${k.replace(/_/g, ' ')} ${num(v)}`).join(' · ') || 'none'}.`),
    h('div', {class: 'grid g2'}, classes('claim'), classes('propaganda')),
    h('div', {class: 'grid g2'},
      h('div', {class: 'card'}, h('h3', {}, 'By source'), share(c.by_platform, 'platform')),
      h('div', {class: 'card'}, h('h3', {}, 'By language'), share(c.by_lang, 'lang_group'), h('h3', {}, 'By month (published)'), share(c.by_period, 'period'))));
}

// ------------------------------------------------------------------ admin: users
async function viewUsers() {
  const t = await api('admin/team?days=0');
  const un = h('input', {type: 'text', placeholder: 'username, e.g. priya.s', style: 'width:200px'});
  const role = h('select', {}, ['annotator', 'adjudicator', 'admin'].map(r => h('option', {value: r}, r)));
  const pw = h('input', {type: 'password', placeholder: 'leave empty to generate', style: 'width:200px', autocomplete: 'new-password'});
  const out = h('div', {});
  setView(
    h('div', {class: 'card'}, h('h3', {}, 'Add a user'), h('div', {class: 'row'}, un, role, pw,
      showPw(pw),
      h('button', {class: 'primary', onclick: guard(async () => {
        const r = await api('admin/users', {method: 'POST', body: {username: un.value.trim(), role: role.value, password: pw.value || null}});
        out.replaceChildren(h('div', {class: 'secret'}, h('div', {}, `Created ${r.annotator.username}. Send them these credentials; they must change the password at first sign-in.`),
          h('div', {}, 'username: ', h('code', {}, r.annotator.username)), h('div', {}, 'password: ', h('code', {}, r.password || '(the one you typed)'))));
        un.value = ''; pw.value = ''; viewUsers.keep = out.firstChild; viewUsers();
      })}, 'Create')), out),
    h('div', {class: 'card', style: 'overflow-x:auto'}, h('h3', {}, 'Everyone'), tbl([
      {h: 'username', f: r => h('b', {}, r.username)},
      {h: 'role', f: r => h('select', {onchange: guard(async ev => { await api('admin/users/' + r.annotator_id, {method: 'POST', body: {role: ev.target.value}}); toast('Role updated'); }), onclick: ev => ev.stopPropagation()},
        ['annotator', 'adjudicator', 'admin'].map(x => h('option', {value: x, selected: x === r.role}, x)))},
      {h: 'status', f: r => !r.active ? 'disabled' : r.locked ? 'locked' : r.online ? 'online' : 'ok'},
      {h: 'last login', f: r => fmtTs(r.last_login_at)},
      {h: 'items done', k: 'items_done_total', num: 1},
      {h: '', f: r => h('button', {onclick: ev => { ev.stopPropagation(); S.userId = r.annotator_id; S.view = 'user'; route(); }}, 'details')},
    ], t.annotators)));
  if (viewUsers.keep) { $('#view').prepend(h('div', {class: 'card'}, viewUsers.keep)); viewUsers.keep = null; }
}

// ------------------------------------------------------------------ admin: data (ingest, pool, pre-annotation)
async function viewData() {
  const [s, j] = await Promise.all([api('stats'), api('admin/job')]);
  const tb = (cols, rows) => tbl(cols.map(c => ({h: c[1], k: c[0]})), rows);
  const run = (path, body, msg) => guard(async () => { await api(path, {method: 'POST', body}); toast(msg); viewData(); });
  const mode = h('select', {}, h('option', {value: 'newest'}, 'newest (watermark)'), h('option', {value: 'backfill'}, 'random historical backfill'));
  const per = h('input', {type: 'number', value: 100, min: 1, max: 500, style: 'width:80px'});
  const since = h('input', {type: 'date'}), until = h('input', {type: 'date'});
  const total = h('input', {type: 'number', value: Math.round((S.progress ? S.progress.target_items : 50000) * 1.2), min: 100, step: 1000, style: 'width:110px'});
  const lim = h('input', {type: 'number', value: 500, min: 1, max: 5000, style: 'width:80px'});
  const cEnc = h('input', {type: 'checkbox', checked: true}), cGem = h('input', {type: 'checkbox'});
  const state = j.running ? 'running' : j.error ? 'error' : j.result ? 'done' : null;
  const job = j.kind ? h('div', {class: 'card'}, h('b', {}, `Background job: ${j.kind} `), state ? h('span', {class: 'badge ' + state}, state) : null,
    j.progress ? h('div', {class: 'small muted'}, 'latest round: ' + JSON.stringify(j.progress)) : null,
    j.error ? h('pre', {class: 'small'}, j.error)
      : Array.isArray(j.result) ? tbl(['platform', 'fetched', 'dup', 'filtered', 'inserted', 'error'].map(k => ({h: k, k})), j.result)
      : j.result ? h('pre', {class: 'small'}, JSON.stringify(j.result, null, 1).slice(0, 1200)) : null) : null;
  setView(
    h('div', {class: 'grid g4'}, ['queued', 'in_progress', 'done', 'excluded'].map(k => stat(k.replace('_', ' '), num(s.items_by_status[k] || 0)))),
    h('div', {class: 'card'}, h('h3', {}, 'Fill the pool (English + Hindi only)'),
      h('div', {class: 'row'}, 'grow to', total, 'items', h('button', {class: 'primary', onclick: run('admin/fill', {total: +total.value, per_source: 250}, 'Pool fill started')}, 'Fill pool'),
        h('span', {class: 'small muted'}, 'One newest-first round, then random historical rounds across all sources; runs in the background.'))),
    h('div', {class: 'card'}, h('h3', {}, 'Pre-annotate (steers class-balanced batches; shown to annotators except on blind items)'),
      h('div', {class: 'row'}, 'next', lim, 'unseen items', h('label', {class: 'small'}, cEnc, ' technique encoder (CPU)'), h('label', {class: 'small'}, cGem, ' Gemma claims (slow, shared endpoint)'),
        h('button', {class: 'primary', onclick: run('preannotate', {limit: +lim.value, encoder: cEnc.checked, gemma: cGem.checked}, 'Pre-annotation started')}, 'Pre-annotate'))),
    h('div', {class: 'card'}, h('h3', {}, 'One-off ingest'), h('div', {class: 'row'}, mode, 'per source', per, 'from', since, 'to', until,
      h('button', {onclick: guard(async () => { await api('ingest', {method: 'POST', body: {mode: mode.value, per_source: +per.value, since: since.value, until: until.value}}); toast('Ingest started'); viewData(); })}, 'Fetch items'),
      h('button', {onclick: viewData}, 'refresh'))),
    job,
    h('div', {class: 'grid g2'},
      h('div', {class: 'card'}, h('h3', {}, 'By platform'), tb([['platform', 'platform'], ['items', 'items'], ['done', 'done']], s.by_platform)),
      h('div', {class: 'card'}, h('h3', {}, 'By language'), tb([['lang_group', 'lang'], ['items', 'items'], ['done', 'done']], s.by_lang),
        h('h3', {}, 'Suggestion feedback'), tb([['layer', 'layer'], ['accepted', 'accepted'], ['rejected', 'rejected']], s.suggestion_feedback))),
    h('div', {class: 'card'}, h('h3', {}, 'Ingest watermarks (source scraped_at)'), tb([['platform', 'source'], ['newest_ts', 'newest'], ['last_run_at', 'last run'], ['rows_seen', 'rows seen']], s.watermarks)));
  if (j.running) { clearTimeout(viewData.t); viewData.t = setTimeout(() => { if (S.view === 'data') viewData().catch(() => {}); }, 4000); }
}

// ------------------------------------------------------------------ adjudication panel (shown only to the adjudicator)
function adjudicationCard(L, d) {
  const A = d.adjudication, both = d.results.claim && d.results.propaganda;
  return h('div', {class: 'card'},
    h('b', {}, 'Adjudication: the annotators\' labels'), h('div', {class: 'hint'}, 'Copy one annotator\'s spans as your starting point, edit them, then submit both layers. Your labels replace theirs in the export.'),
    A.annotators.map(p => { const r = p[L]; return h('div', {style: 'margin-top:8px'},
      h('div', {class: 'row'}, h('b', {}, p.username), r ? h('span', {class: 'tag'}, r.outcome === 'none' ? (L === 'claim' ? 'no claims' : 'no propaganda') : `${r.spans.length} span${r.spans.length === 1 ? '' : 's'}`) : h('span', {class: 'tag'}, 'not labelled'),
        r ? h('button', {onclick: guard(async () => { await api('admin/adjudicate/copy', {method: 'POST', body: {item_id: d.item.item_id, from_annotator_id: p.annotator_id, layer: L}}); await reload(); toast('Copied: now edit and submit'); })}, 'copy as mine') : null),
      r ? r.spans.map(sp => h('div', {class: 'spanrow', onmouseenter: () => { S.focus = {start: sp.start, end: sp.end}; renderText(); }, onmouseleave: () => { S.focus = null; renderText(); }},
        h('div', {class: 'tx'}, h('q', {}, sp.text), L === 'claim' ? h('span', {class: 'tag'}, typeLabel('claim', sp.claim_type)) : sp.techniques.map(t => h('span', {class: 'tag'}, typeLabel('propaganda', t))))))
        : null); }),
    both ? h('div', {class: 'row', style: 'margin-top:10px'},
      A.is_gold ? h('span', {class: 'tag'}, 'this item is gold') : h('button', {onclick: guard(async () => { await api('admin/gold', {method: 'POST', body: {item_id: d.item.item_id}}); toast('Saved as a gold item: it will appear, blind, in future batches'); await reload(); })}, 'Use as gold (calibration)'),
      h('span', {class: 'small muted'}, 'Gold items are shown blind to other annotators inside their batches and scored on the Quality tab; their labels never enter the export.')) : null);
}

// ------------------------------------------------------------------ admin: quality
const pct0 = x => x == null ? '–' : Math.round(100 * x) + '%';
const f2 = x => x == null ? '–' : Number(x).toFixed(2);
async function viewQuality() {
  const q = await api('admin/quality'), T = q.thresholds;
  const goldCell = r => r.gold ? h('span', {title: `claims P ${f2(r.gold.claim.precision)} R ${f2(r.gold.claim.recall)} · propaganda P ${f2(r.gold.propaganda.precision)} R ${f2(r.gold.propaganda.recall)} · claim-type accuracy ${pct0(r.gold.claim.claim_type_accuracy)}`},
    `${f2(r.gold.claim.f1)} / ${f2(r.gold.propaganda.f1)} `, h('span', {class: 'small muted'}, `(n=${r.gold.n_items})`)) : h('span', {class: 'muted small'}, 'no gold yet');
  const agrCell = r => r.agreement ? h('span', {}, `${f2(r.agreement.claim_f1)} / ${f2(r.agreement.propaganda_f1)} `, h('span', {class: 'small muted'}, `(n=${r.agreement.n_items})`)) : h('span', {class: 'muted small'}, 'no doubles yet');
  setView(
    h('div', {class: 'row', style: 'justify-content:space-between;margin-bottom:10px'}, h('h2', {style: 'margin:0'}, 'Annotator quality'), h('button', {onclick: viewQuality}, 'refresh')),
    h('div', {class: 'hint'}, 'Gold = span-F1 against adjudicated reference items served blind inside normal batches (claims / propaganda). Agreement = span-F1 against the partner on double-annotated items. ',
      `Flags are prompts to look at someone's work, not verdicts: gold below F1 ${T.ann_gold_min_f1} (needs ${T.ann_gold_min_items}+ gold items), agreement below ${T.ann_q_min_agree_f1}, a 'no claims' share more than ${Math.round(100 * T.ann_q_none_dev)} points from the team median (${pct0(q.team.claim_none_rate_median)}), or a median under ${T.ann_q_fast_sec} s per item (after ${T.ann_q_min_items} items).`),
    q.rows.length ? h('div', {class: 'card', style: 'overflow-x:auto'}, tbl([
      {h: 'annotator', f: r => h('span', {}, h('b', {}, r.username), r.role !== 'annotator' ? h('span', {class: 'tag'}, r.role) : null, !r.active ? h('span', {class: 'tag'}, 'disabled') : null)},
      {h: 'items', k: 'items_done', num: 1},
      {h: 'median s/item', f: r => r.median_sec == null ? '–' : Math.round(r.median_sec), num: 1},
      {h: '"none" claims / prop.', f: r => `${pct0(r.claim_none_rate)} / ${pct0(r.propaganda_none_rate)}`, num: 1},
      {h: 'accepts suggestions', f: r => pct0(r.suggestion_accept_rate), num: 1},
      {h: 'gold F1 (claims / prop.)', f: goldCell, num: 1},
      {h: 'partner F1 (claims / prop.)', f: agrCell, num: 1},
      {h: 'flags', f: r => r.flags.length ? r.flags.map(x => h('span', {class: 'tag warn', title: x}, x)) : h('span', {class: 'muted small'}, 'none')},
    ], q.rows, r => { S.userId = r.annotator_id; S.view = 'user'; route(); })) : h('div', {class: 'card'}, h('div', {class: 'hint'}, 'No finished work yet.')));
}

// ------------------------------------------------------------------ admin / adjudicator: disputes and gold
async function viewDisputes() {
  const [d, g] = await Promise.all([api('admin/disputes'), api('admin/gold')]);
  const open = guard(async id => { await api('admin/adjudicate', {method: 'POST', body: {item_id: id}}); S.data = await api('item/' + id); S.view = 'annotate'; S.layer = 'claim'; setTabs(); render(); });
  const stateTag = r => h('span', {class: 'badge ' + (r.state === 'adjudicated' ? 'done' : r.state === 'in_progress' ? 'running' : 'stalled')}, r.state.replace('_', ' ') + (r.adjudicator ? ' · ' + r.adjudicator : ''));
  setView(
    h('div', {class: 'row', style: 'justify-content:space-between;margin-bottom:10px'}, h('h2', {style: 'margin:0'}, 'Disputes'), h('button', {onclick: viewDisputes}, 'refresh')),
    h('div', {class: 'hint'}, `Double-annotated items where the two annotators disagree (one marked something the other did not, or different techniques). They stay out of the training export until resolved: open one, copy a side or relabel, and submit both layers. ${d.counts.pending} waiting · ${d.counts.in_progress} in progress · ${d.counts.adjudicated} resolved.`),
    h('div', {class: 'card', style: 'overflow-x:auto'}, d.items.length ? tbl([
      {h: '#', k: 'item_id', num: 1}, {h: 'state', f: stateTag}, {h: 'source', k: 'platform'},
      {h: 'disputed in', f: r => Object.entries(r.layers).map(([l, v]) => h('span', {class: 'tag'}, `${l} (${v.unmatched ?? '?'} unmatched)`))},
      {h: 'annotators', f: r => r.annotators.join(', ')}, {h: 'text', f: r => h('span', {class: 'small'}, r.preview)},
      {h: '', f: r => h('span', {class: 'inl'}, r.state === 'adjudicated' ? (r.gold ? h('span', {class: 'tag'}, 'gold') : null) : null,
        h('button', {class: r.state === 'pending' ? 'primary' : '', onclick: () => open(r.item_id)}, r.state === 'adjudicated' ? 'Open' : 'Adjudicate'))},
    ], d.items) : h('div', {class: 'hint'}, 'No disputes yet. They appear once double-annotated items are finished and disagree.')),
    h('div', {class: 'card', style: 'overflow-x:auto'}, h('h3', {}, 'Gold items'),
      h('div', {class: 'hint'}, 'Adjudicated items served blind to annotators (about ' + Math.round(100 * 0.03) + '% of each batch once any exist). Mark an item as gold from its adjudication screen.'),
      g.items.length ? tbl([{h: '#', k: 'item_id', num: 1}, {h: 'by', k: 'created_by'}, {h: 'served', k: 'served', num: 1}, {h: 'answered', k: 'answered', num: 1},
        {h: 'text', f: r => h('span', {class: 'small'}, r.preview)},
        {h: '', f: r => r.active ? h('button', {onclick: guard(async () => { await api('admin/gold', {method: 'POST', body: {item_id: r.item_id, remove: true}}); viewDisputes(); })}, 'retire') : h('span', {class: 'tag'}, 'retired')}], g.items)
        : h('div', {class: 'hint'}, 'No gold items yet.')));
}

// ------------------------------------------------------------------ shell
function setTabs() {
  const admin = S.user && S.user.role === 'admin';
  const tabs = S.user ? [['annotate', 'Annotate'], ['mine', 'My items'], ['me', 'My progress'],
    ...(admin ? [['team', 'Team'], ['quality', 'Quality'], ['disputes', 'Disputes'], ['coverage', 'Coverage'], ['users', 'Users'], ['data', 'Data']]
      : S.user.role === 'adjudicator' ? [['disputes', 'Disputes']] : [])] : [];
  const on = S.view === 'user' ? 'team' : S.view;
  $('#tabs').replaceChildren(...tabs.map(([k, n]) => h('a', {class: on === k ? 'on' : '', onclick: () => { closePop(); S.view = k; route(); }}, n)));
}
async function route() {
  setTabs();
  if (!S.user) return;
  try {
    const v = {mine: viewMine, me: viewMe, team: viewTeam, user: viewUser, coverage: viewCoverage, users: viewUsers, data: viewData, quality: viewQuality, disputes: viewDisputes}[S.view];
    if (v) return await v();
    if (!S.data) return await nextItem();
    render();
  } catch (e) { if (e.status !== 401 && e.code !== 'must_change_password') toast(e.message, true); }
}

function signedOut(msg) {
  S.user = null; S.token = ''; S.data = null; S.progress = null; S.view = 'annotate'; ls.set('annot_token', '');
  $('#who').textContent = ''; $('#prog').replaceChildren(); $('#logout').classList.add('hidden'); $('#chpw').classList.add('hidden');
  closePop(); setTabs(); loginView(msg);
}

function enter(r) {
  S.token = r.token || S.token; ls.set('annot_token', S.token);
  S.user = r.annotator; S.tax = r.taxonomy; S.progress = r.progress; S.idleSec = r.idle_seconds || 120; S.data = null; S.view = 'annotate';
  $('#who').textContent = S.user.username + (S.user.role === 'admin' ? ' (admin)' : '');
  if (S.user.role === 'adjudicator') S.view = 'disputes';
  if (S.user.role === 'admin') S.view = 'team';          // admins start on the monitoring page (no batch is built for them until they annotate)
  $('#logout').classList.remove('hidden'); $('#chpw').classList.remove('hidden'); drawProgress();
  if (S.user.must_change_password) return passwordView(true);
  route();
}

// "Show password" checkbox that flips the given password inputs between hidden and visible
function showPw(...inputs) {
  const cb = h('input', {type: 'checkbox', id: 'showpw'});
  cb.addEventListener('change', () => inputs.forEach(i => { i.type = cb.checked ? 'text' : 'password'; }));
  return h('label', {class: 'small muted showpw'}, cb, ' Show password');
}

function loginView(msg) {
  const u = h('input', {type: 'text', placeholder: 'username', autocomplete: 'username', value: ls.get('annot_user') || '', autofocus: true});
  const p = h('input', {type: 'password', placeholder: 'password', autocomplete: 'current-password'});
  const err = h('p', {class: 'err'}, msg || '');
  const go = async () => {
    err.textContent = '';
    try { const r = await api('auth/login', {method: 'POST', body: {username: u.value.trim(), password: p.value}}); ls.set('annot_user', r.annotator.username); p.value = ''; enter(r); }
    catch (e) { err.textContent = e.message; }
  };
  setView(h('div', {class: 'card login'}, h('h2', {}, 'Sign in to annotate'), err,
    h('label', {class: 'small muted'}, 'Username'), u, h('label', {class: 'small muted'}, 'Password'), p, showPw(p),
    h('button', {class: 'primary', onclick: go}, 'Sign in'),
    h('p', {class: 'hint'}, 'No account or forgot your password? Ask your admin. Your name is recorded with every label, and your sign-in and active working time are tracked.')));
  [u, p].forEach(x => x.addEventListener('keydown', ev => { if (ev.key === 'Enter') go(); }));
  (u.value ? p : u).focus();
}

function passwordView(forced) {
  const o = h('input', {type: 'password', placeholder: forced ? 'temporary password you were given' : 'current password', autocomplete: 'current-password'});
  const n = h('input', {type: 'password', placeholder: 'new password (at least 8 characters)', autocomplete: 'new-password'});
  const n2 = h('input', {type: 'password', placeholder: 'new password again', autocomplete: 'new-password'});
  const err = h('p', {class: 'err'});
  const go = async () => {
    err.textContent = '';
    if (n.value !== n2.value) { err.textContent = 'the two new passwords differ'; return; }
    try { await api('auth/password', {method: 'POST', body: {old: o.value, new: n.value}}); S.user.must_change_password = false; toast('Password changed'); S.view = 'annotate'; route(); }
    catch (e) { err.textContent = e.message; }
  };
  $('#tabs').replaceChildren();
  setView(h('div', {class: 'card login'}, h('h2', {}, forced ? 'Choose a new password' : 'Change password'),
    forced ? h('p', {class: 'muted'}, 'You are signed in with a temporary password. Set your own to continue.') : null, err,
    o, n, n2, showPw(o, n, n2), h('div', {class: 'row'}, h('button', {class: 'primary', onclick: go}, 'Change password'), forced ? null : h('button', {onclick: () => { S.view = 'annotate'; route(); }}, 'Cancel'))));
  [o, n, n2].forEach(x => x.addEventListener('keydown', ev => { if (ev.key === 'Enter') go(); }));
  o.focus();
}

document.addEventListener('mouseup', onTextMouseUp);
addEventListener('keydown', ev => {
  if (ev.key === 'Escape' && S.reselect) { S.reselect = null; closePop(); toast('Cancelled'); return; }
  if ($('#pop') && !$('#pop').classList.contains('hidden') && S.popKeys) return S.popKeys(ev);
  if (ev.target.matches('input,textarea,select') || ev.ctrlKey || ev.metaKey || ev.altKey || S.view !== 'annotate' || !S.data) return;
  if (ev.key === 'c') { S.layer = 'claim'; render(); } else if (ev.key === 'p') { S.layer = 'propaganda'; render(); }
  else if (ev.key === 'n' && S.data.results.claim && S.data.results.propaganda) guard(nextItem)();
});
document.addEventListener('mousedown', ev => { const p = $('#pop'); if (!p.classList.contains('hidden') && !p.contains(ev.target) && !ev.target.closest('#text')) closePop(); });
$('#theme').addEventListener('click', () => {
  const r = document.documentElement; r.dataset.theme = r.dataset.theme === 'dark' ? 'light' : 'dark'; ls.set('annot_theme', r.dataset.theme); if (S.data) render(); });
document.documentElement.dataset.theme = ls.get('annot_theme') || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
$('#logout').addEventListener('click', guard(async () => { try { await api('auth/logout', {method: 'POST', body: {}}); } catch (e) { /* already out */ } signedOut('Signed out.'); }));
$('#chpw').addEventListener('click', () => passwordView(false));
loginView();
if (S.token) {                                  // resume the session stored in this browser, if it is still valid
  (async () => {
    try { enter(await api('me')); }
    catch (e) { S.user = null; loginView(e.status === 401 ? 'Please sign in.' : null); }
  })();
}
