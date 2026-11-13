/* Pure helpers shared by the Evidence Base, Workbench and the tour: no DOM, no network, no globals read.
   They are loaded in the page (window.FNPPure) and in node for the unit tests (module.exports), which is the only reason
   they live in their own file. Everything that handles untrusted text goes through esc(). */
(function (root, factory) {
  'use strict';
  var api = factory();
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (root) root.FNPPure = api;
})(typeof window !== 'undefined' ? window : null, function () {
  'use strict';

  function esc(s) { return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]; }); }
  function fmtN(n) { return Number(n || 0).toLocaleString('en-US'); }
  function oneLine(s) { return String(s == null ? '' : s).replace(/\s+/g, ' ').trim(); }
  function clip(s, n) { s = oneLine(s); return s.length > n ? s.slice(0, n - 1).trim() + '…' : s; }

  /* ---- time ---- */
  function minutesBetween(iso, nowMs) { var d = Date.parse(iso); return isNaN(d) ? null : Math.max(0, (nowMs - d) / 60000); }
  function dur(min) {
    if (min == null) return 'an unknown time';
    if (min < 1) return 'under a minute';
    if (min < 60) return Math.round(min) + ' min';
    var h = min / 60;
    if (h < 48) return (h < 10 ? h.toFixed(1).replace(/\.0$/, '') : Math.round(h)) + ' h';
    return Math.round(h / 24) + ' days';
  }
  function agoAt(iso, nowMs) { var m = minutesBetween(iso, nowMs); return m == null ? 'unknown time' : m < 1 ? 'just now' : dur(m) + ' ago'; }
  function inAt(iso, nowMs) { var d = Date.parse(iso); if (isNaN(d)) return 'at an unknown time'; var m = (d - nowMs) / 60000; return m < 1 ? 'any moment now' : 'in ' + dur(m); }
  function secs(n) { n = Math.max(0, Math.round(n)); return n < 60 ? n + ' s' : Math.floor(n / 60) + ' min ' + (n % 60 < 10 ? '0' : '') + (n % 60) + ' s'; }

  /* ---- numbers ---- */
  function pct(n, d, dp) { if (d && !n) return '0%'; return d ? (n / d * 100).toFixed(dp == null ? 1 : dp) + '%' : 'n/a'; }
  function n2(p) { return p == null || isNaN(p) ? 'n/a' : Number(p).toFixed(2); }
  function plural(n, one, many) { return fmtN(n) + ' ' + (n === 1 ? one : many); }

  /* ---- item text ---- */
  function titleCase(s) { return String(s).replace(/[_\-.]+/g, ' ').trim().replace(/\b[a-z]/g, function (c) { return c.toUpperCase(); }); }
  function kindName(k) { return k === 'news' ? 'News article' : k === 'social' ? 'Social post' : titleCase(k || 'Item'); }
  function sourceName(s) { return s == null || String(s).trim() === '' || s === '(unknown)' ? 'Unknown source' : String(s); }
  function itemTitle(it) {
    var t = oneLine(it.title);
    if (t) return t;
    var s = oneLine(it.snippet != null ? it.snippet : it.text_excerpt);
    return s || (kindName(it.kind) + ' without text');
  }

  /* ---- query strings: empty values are left out, order is stable ---- */
  function buildQuery(pairs) {
    var out = [];
    pairs.forEach(function (p) { if (p[1] !== '' && p[1] != null && p[1] !== false) out.push(encodeURIComponent(p[0]) + '=' + encodeURIComponent(String(p[1]))); });
    return out.join('&');
  }
  var BROWSE_KEYS = ['q', 'kind', 'platform', 'analysed', 'source', 'topic', 'lang', 'from', 'to', 'sort'];
  function browseQuery(B, size, page) {
    var pairs = BROWSE_KEYS.map(function (k) { return [k, k === 'q' ? String(B.q || '').trim() : B[k]]; });
    pairs.push(['size', size], ['page', page]);
    return buildQuery(pairs);
  }
  function sourcesQuery(text, kind, platform, limit) { return buildQuery([['q', String(text || '').trim()], ['kind', kind], ['platform', platform], ['limit', limit || 20]]); }

  /* ---- selection (the Workbench cart): mutates the array it is given, returns what happened ---- */
  function selHas(sel, id) { return sel.some(function (x) { return x.id === id; }); }
  function selAdd(sel, m, max) { if (selHas(sel, m.id)) return 'dup'; if (sel.length >= max) return 'full'; sel.push(m); return 'added'; }
  function selRemove(sel, id) { for (var i = sel.length - 1; i >= 0; i--) if (sel[i].id === id) sel.splice(i, 1); return sel; }

  /* ---- source freshness card ---- */
  var DAY_MS = 86400000;
  /* rows from /api/rt/status source_freshness: sorted by key ('rows', 'newest', 'share', 'name'), bar = how recent the newest item is within 30 days */
  function freshSort(rows, key, dir, nameOf) {
    var sign = dir === 'asc' ? 1 : -1;
    var val = function (r) {
      if (key === 'rows') return r.rows || 0;
      if (key === 'share') return r.share == null ? -1 : r.share;
      if (key === 'newest') { var t = Date.parse(r.newest_published_at); return isNaN(t) ? -Infinity : t; }
      return String(nameOf ? nameOf(r) : r.platform).toLowerCase();
    };
    return rows.slice().sort(function (a, b) {
      var x = val(a), y = val(b);
      if (x < y) return -1 * sign; if (x > y) return 1 * sign;
      return String(a.source_table).localeCompare(String(b.source_table));
    });
  }
  function freshFraction(newestIso, nowMs, windowDays) {
    var d = Date.parse(newestIso); if (isNaN(d)) return 0;
    var age = Math.max(0, nowMs - d), w = (windowDays || 30) * DAY_MS;
    return Math.max(0, Math.min(1, 1 - age / w));
  }

  /* ---- tour card placement: choose the viewport corner whose card overlaps the highlighted element least ---- */
  function overlapArea(a, b) {
    var w = Math.min(a.right, b.right) - Math.max(a.left, b.left), h = Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top);
    return w > 0 && h > 0 ? w * h : 0;
  }
  /* o: {target:{left,top,right,bottom}|null, card:{w,h}, vw, vh, side:'left'|'right', navW, margin, topMin, bottomMin, mobile}
     Returns {left, top, w?}: the card's top-left corner in viewport pixels (mobile: full width, docked top or bottom). */
  function placeCard(o) {
    var m = o.margin == null ? 24 : o.margin, cw = o.card.w, ch = o.card.h, topMin = o.topMin == null ? m : o.topMin, botMin = o.bottomMin == null ? m : o.bottomMin;
    var view = { left: 0, top: 0, right: o.vw, bottom: o.vh };
    var tgt = o.target ? { left: Math.max(o.target.left, 0), top: Math.max(o.target.top, 0), right: Math.min(o.target.right, o.vw), bottom: Math.min(o.target.bottom, o.vh) } : null;
    var cands;
    if (o.mobile) {
      var wMob = o.vw - 2 * m;
      cands = [{ left: m, top: o.vh - botMin - ch, w: wMob }, { left: m, top: topMin, w: wMob }];
    } else {
      var L = (o.navW || 0) + m, R = o.vw - m - cw, B = o.vh - botMin - ch;
      var bl = { left: L, top: B }, br = { left: R, top: B }, tl = { left: L, top: topMin }, tr = { left: R, top: topMin };
      cands = o.side === 'left' ? [bl, br, tl, tr] : [br, bl, tr, tl];
    }
    var best = null, bestA = Infinity;
    cands.forEach(function (c) {
      var rect = { left: c.left, top: c.top, right: c.left + (c.w || cw), bottom: c.top + ch };
      var a = tgt ? overlapArea(rect, tgt) : 0;
      if (a < bestA) { best = c; bestA = a; }
    });
    var out = { left: Math.max(0, Math.round(best.left)), top: Math.max(0, Math.round(best.top)), overlap: bestA };
    if (best.w) out.w = best.w;
    void view;
    return out;
  }

  return { esc: esc, fmtN: fmtN, oneLine: oneLine, clip: clip, minutesBetween: minutesBetween, dur: dur, agoAt: agoAt, inAt: inAt, secs: secs,
    pct: pct, n2: n2, plural: plural, titleCase: titleCase, kindName: kindName, sourceName: sourceName, itemTitle: itemTitle,
    buildQuery: buildQuery, browseQuery: browseQuery, sourcesQuery: sourcesQuery, selHas: selHas, selAdd: selAdd, selRemove: selRemove,
    freshSort: freshSort, freshFraction: freshFraction, overlapArea: overlapArea, placeCard: placeCard, STALE_HOURS: 48 };
});
