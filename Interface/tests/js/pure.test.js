'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');
const P = require(path.join(__dirname, '..', '..', 'web', 'pure.js'));

test('esc escapes the five HTML characters and null', () => {
  assert.equal(P.esc('<img src=x onerror="a()"> & \'q\''), '&lt;img src=x onerror=&quot;a()&quot;&gt; &amp; &#39;q&#39;');
  assert.equal(P.esc(null), ''); assert.equal(P.esc(undefined), ''); assert.equal(P.esc(0), '0');
});

test('fmtN, pct, n2, plural', () => {
  assert.equal(P.fmtN(1234567), '1,234,567'); assert.equal(P.fmtN(null), '0');
  assert.equal(P.pct(1, 3), '33.3%'); assert.equal(P.pct(0, 5), '0%'); assert.equal(P.pct(1, 0), 'n/a'); assert.equal(P.pct(1, 3, 0), '33%');
  assert.equal(P.n2(0.456), '0.46'); assert.equal(P.n2(null), 'n/a'); assert.equal(P.n2(NaN), 'n/a');
  assert.equal(P.plural(1, 'item', 'items'), '1 item'); assert.equal(P.plural(1500, 'item', 'items'), '1,500 items');
});

test('dur picks minutes, hours, days', () => {
  assert.equal(P.dur(null), 'an unknown time'); assert.equal(P.dur(0.4), 'under a minute'); assert.equal(P.dur(12.4), '12 min');
  assert.equal(P.dur(90), '1.5 h'); assert.equal(P.dur(120), '2 h'); assert.equal(P.dur(60 * 30), '30 h'); assert.equal(P.dur(60 * 72), '3 days');
});

test('time ago and in, with an explicit clock', () => {
  const now = Date.parse('2026-11-12T12:00:00Z');
  assert.equal(P.agoAt('2026-11-12T11:48:00Z', now), '12 min ago'); assert.equal(P.agoAt('2026-11-12T11:59:50Z', now), 'just now');
  assert.equal(P.agoAt('2026-11-09T12:00:00Z', now), '3 days ago'); assert.equal(P.agoAt('nonsense', now), 'unknown time');
  assert.equal(P.agoAt('2026-11-13T00:00:00Z', now), 'just now'); /* future clamps to zero */
  assert.equal(P.inAt('2026-11-12T12:52:00Z', now), 'in 52 min'); assert.equal(P.inAt('2026-11-12T11:00:00Z', now), 'any moment now');
  assert.equal(P.secs(45), '45 s'); assert.equal(P.secs(125), '2 min 05 s');
});

test('item text helpers', () => {
  assert.equal(P.kindName('news'), 'News article'); assert.equal(P.kindName('social'), 'Social post'); assert.equal(P.kindName('video_clip'), 'Video Clip');
  assert.equal(P.sourceName(''), 'Unknown source'); assert.equal(P.sourceName('(unknown)'), 'Unknown source'); assert.equal(P.sourceName('a.com'), 'a.com');
  assert.equal(P.itemTitle({ title: '  Big\n  news ' }), 'Big news');
  assert.equal(P.itemTitle({ title: '', text_excerpt: 'body  text' }), 'body text');
  assert.equal(P.itemTitle({ title: '', kind: 'social' }), 'Social post without text');
  assert.equal(P.clip('abcdefghij', 5), 'abcd…'); assert.equal(P.clip('abc', 5), 'abc');
});

test('query strings leave out empty values and encode the rest', () => {
  assert.equal(P.buildQuery([['a', ''], ['b', 'x y'], ['c', null], ['d', 0]]), 'b=x%20y&d=0');
  const B = { q: '  flood & rain ', kind: 'news', platform: '', analysed: 'yes', source: 'a.com', topic: '', lang: 'hi', from: '', to: '', sort: '' };
  assert.equal(P.browseQuery(B, 40, 2), 'q=flood%20%26%20rain&kind=news&analysed=yes&source=a.com&lang=hi&size=40&page=2');
  assert.equal(P.sourcesQuery(' ab ', 'social', '', 20), 'q=ab&kind=social&limit=20');
  assert.equal(P.sourcesQuery('', '', '', 0), 'limit=20');
});

test('selection: add, duplicate, full, remove', () => {
  const sel = [];
  assert.equal(P.selAdd(sel, { id: 'a' }, 2), 'added'); assert.equal(P.selAdd(sel, { id: 'a' }, 2), 'dup');
  assert.equal(P.selAdd(sel, { id: 'b' }, 2), 'added'); assert.equal(P.selAdd(sel, { id: 'c' }, 2), 'full');
  assert.equal(P.selHas(sel, 'b'), true); P.selRemove(sel, 'a'); assert.deepEqual(sel.map(x => x.id), ['b']); P.selRemove(sel, 'zzz'); assert.equal(sel.length, 1);
});

test('freshSort and freshFraction', () => {
  const rows = [
    { source_table: 'a_posts', platform: 'reddit', rows: 10, share: 0.1, newest_published_at: '2026-11-01T00:00:00Z' },
    { source_table: 'b_posts', platform: 'tiktok', rows: 30, share: 0.3, newest_published_at: '2026-11-10T00:00:00Z' },
    { source_table: 'c_posts', platform: 'discord', rows: 20, share: null, newest_published_at: null }];
  const ids = (k, d) => P.freshSort(rows, k, d, r => r.platform).map(r => r.source_table[0]).join('');
  assert.equal(ids('rows', 'desc'), 'bca'); assert.equal(ids('rows', 'asc'), 'acb');
  assert.equal(ids('newest', 'desc'), 'bac'); assert.equal(ids('name', 'asc'), 'cab'); assert.equal(ids('share', 'desc'), 'bac');
  assert.equal(rows[0].source_table, 'a_posts'); /* input not mutated */
  const now = Date.parse('2026-11-12T00:00:00Z');
  assert.equal(P.freshFraction('2026-11-12T00:00:00Z', now), 1); assert.equal(P.freshFraction('2026-10-01T00:00:00Z', now), 0);
  assert.ok(Math.abs(P.freshFraction('2026-10-28T00:00:00Z', now) - 0.5) < 1e-9); assert.equal(P.freshFraction('bad', now), 0);
});

test('placeCard keeps the tour card off the highlighted element', () => {
  const base = { card: { w: 344, h: 260 }, vw: 1440, vh: 900, navW: 220, margin: 24, topMin: 72, bottomMin: 24, mobile: false };
  /* Workbench Find column on the left: the card goes to the right-hand corner, not over the column */
  const find = { left: 244, top: 130, right: 744, bottom: 880 };
  const p = P.placeCard(Object.assign({}, base, { target: find, side: 'right' }));
  assert.equal(p.overlap, 0); assert.ok(p.left >= 744);
  /* a target on the right pushes the preferred right-hand card to the left */
  const right = { left: 900, top: 100, right: 1400, bottom: 880 };
  const q = P.placeCard(Object.assign({}, base, { target: right, side: 'right' }));
  assert.equal(q.overlap, 0); assert.ok(q.left + 344 <= 900);
  /* no target: the preferred corner */
  assert.deepEqual(P.placeCard(Object.assign({}, base, { target: null, side: 'right' })), { left: 1440 - 24 - 344, top: 900 - 24 - 260, overlap: 0 });
  assert.equal(P.placeCard(Object.assign({}, base, { target: null, side: 'left' })).left, 244);
  /* a target that fills the screen: the least-overlapping corner is still chosen and stays inside the viewport */
  const all = P.placeCard(Object.assign({}, base, { target: { left: 0, top: 0, right: 1440, bottom: 900 }, side: 'right' }));
  assert.ok(all.left >= 0 && all.top >= 0 && all.left + 344 <= 1440 && all.top + 260 <= 900);
  /* phone: full-width card docked at the bottom, flips to the top when the target is down there */
  const mob = { card: { w: 300, h: 200 }, vw: 390, vh: 800, margin: 16, topMin: 68, bottomMin: 68, mobile: true };
  const m1 = P.placeCard(Object.assign({}, mob, { target: { left: 0, top: 70, right: 390, bottom: 400 } }));
  assert.equal(m1.w, 358); assert.equal(m1.top, 800 - 68 - 200);
  const m2 = P.placeCard(Object.assign({}, mob, { target: { left: 0, top: 500, right: 390, bottom: 740 } }));
  assert.equal(m2.top, 68);
});
