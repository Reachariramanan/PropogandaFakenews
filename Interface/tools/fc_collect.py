#!/usr/bin/env python3
"""Collect fact-checks from authoritative fact-checkers (last N years) into one JSONL.

usage: fc_collect.py SITE [SITE ...] | all   [--years 3] [--limit N] [--workers 4] [--delay 0.4]
                                             [--out Interface/data/factchecks_3y.jsonl] [--download-media DIR]
       fc_collect.py --list

Adapters (SITES below):
  telegram  public t.me/s/<channel> preview (PIB Fact Check)
  wp        WordPress REST API (/wp-json/wp/v2/posts, _embed=1)
  web       URL discovery (sitemaps, listing pages, Common Crawl index) + article page parse
            (schema.org ClaimReview / NewsArticle JSON-LD, OG meta, DOM)

One row per fact-check; resumable (ids already in --out are skipped). Re-running is safe.
Dates: `published` is the fact-check article date, `claim_date` is when the claim/event happened or first
appeared (`claim_date_basis` = claimreview | text | none; text is a regex heuristic, see find_event_dates).
"""
import re, sys, json, time, html, argparse, hashlib, threading, urllib.parse
from datetime import datetime, timezone, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import requests
from lxml import html as LH
from dateutil import parser as dparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fc_scrape import norm, jsonld, S, types  # verdict normaliser + JSON-LD walker shared with the older scraper

DATA = Path(__file__).resolve().parent.parent / 'data'
UA = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36',
      'Accept-Language': 'en,hi;q=0.8'}
SESSION = requests.Session(); SESSION.headers.update(UA)
_lock = threading.Lock(); _last = {}

def get(url, delay=0.4, timeout=40, tries=3, **kw):
    host = urllib.parse.urlsplit(url).netloc
    err = None
    for i in range(tries):
        with _lock:
            wait = _last.get(host, 0) + delay - time.time(); _last[host] = time.time() + max(wait, 0)
        if wait > 0: time.sleep(wait)
        try:
            r = SESSION.get(url, timeout=timeout, **kw)
            if r.status_code in (429, 500, 502, 503, 504): time.sleep(5 * (i + 1)); err = requests.HTTPError(f'HTTP {r.status_code} {url}'); continue
            if r.status_code >= 400: raise requests.HTTPError(f'HTTP {r.status_code} {url}')
            return r
        except requests.HTTPError: raise
        except Exception as e: err = e; time.sleep(1.5 * (i + 1))
    raise err

# ---------------------------------------------------------------- text / date helpers
def clean(s): return re.sub(r'\s+', ' ', html.unescape(s or '').replace('\xa0', ' ')).strip()
def text_of(node): return clean(node.text_content()) if node is not None else ''
def iso(d):
    try:
        dt = dparse.parse(d) if isinstance(d, str) else d
        return dt.isoformat()
    except Exception: return ''
def to_utc(s):
    try:
        d = dparse.parse(s)
        if d.tzinfo is None: d = d.replace(tzinfo=timezone.utc)
        return d.astimezone(timezone.utc)
    except Exception: return None

MON = r'(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?'
DATE_RES = [
    re.compile(r'\b(\d{1,2})(?:st|nd|rd|th)?\s+' + MON + r',?\s+(\d{4})\b', re.I),            # 12 March 2021
    re.compile(r'\b' + MON + r'\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b', re.I),            # March 12, 2021
    re.compile(r'\b' + MON + r',?\s+(\d{4})\b', re.I),                                         # March 2021
    re.compile(r'\b(\d{1,2})[/.-](\d{1,2})[/.-](20\d{2})\b'),                                  # 12/03/2021 (day first, Indian style)
    re.compile(r'\b(\d{1,2})(?:st|nd|rd|th)?\s+' + MON + r'(?!\s*,?\s*\d{4})\b', re.I),          # 27 October (year inferred from the fact-check date)
    re.compile(r'\b' + MON + r'\s+(\d{1,2})(?:st|nd|rd|th)?\b(?!\s*,?\s*\d{4})', re.I),          # October 27
]
CTX = re.compile(r'(video|image|photo|picture|clip|footage|post|claim|tweet|visual|screenshot|shared|posted|uploaded|recorded|filmed|'
                 r'taken|dated|dating|published|aired|since|originally|first|old|from|on)', re.I)

STRONG = re.compile(r'(shared|posted|uploaded|recorded|filmed|taken|dated|dates back|dating back|since|originally|first (?:appeared|surfaced|published|posted)|old (?:video|image|photo|clip)|footage|clip)', re.I)

def find_event_dates(text, published=None, max_chars=6000):
    """Dates mentioned near claim words, strictly before the fact-check's own date. Heuristic, so each hit
    keeps its snippet. Returns list of dict(date, snippet) ordered by position."""
    text = text[:max_chars]; pub = to_utc(published) if published else None
    seen, out, taken = set(), [], []  # taken: spans of day-precision hits, so 'March 2019' inside '19 March 2019' is not a second hit
    for k, rx in enumerate(DATE_RES):
        for m in rx.finditer(text):
            if k in (2, 4, 5) and any(a <= m.start() < b for a, b in taken): continue
            g = m.groups()
            try:
                if k == 0: d = datetime(int(g[2]), _mon(g[1]), int(g[0]))
                elif k == 1: d = datetime(int(g[2]), _mon(g[0]), int(g[1]))
                elif k == 2: d = datetime(int(g[1]), _mon(g[0]), 1)
                elif k == 3: d = datetime(int(g[2]), int(g[1]), int(g[0]))
                else:  # no year: latest such date not after the fact-check
                    if not pub: continue
                    day, mon = (int(g[0]), _mon(g[1])) if k == 4 else (int(g[1]), _mon(g[0]))
                    d = datetime(pub.year, mon, day)
                    if d.replace(tzinfo=timezone.utc) > pub + timedelta(days=1): d = d.replace(year=pub.year - 1)
            except Exception: continue
            if d.year < 2005: continue
            if pub and d.replace(tzinfo=timezone.utc) > pub + timedelta(days=1): continue
            pre = text[max(0, m.start() - 90):m.start()]
            if not CTX.search(pre): continue
            key = (d.date().isoformat(), k == 2)
            if key in seen: continue
            seen.add(key)
            if k in (0, 1, 3): taken.append((m.start(), m.end()))
            out.append(dict(date=d.date().isoformat(), precision='month' if k == 2 else 'day', year_inferred=k > 3, pos=m.start(), strong=bool(STRONG.search(pre) or STRONG.search(text[m.end():m.end() + 45])),
                            snippet=text[max(0, m.start() - 90):m.end() + 40].strip()))
    return sorted(out, key=lambda x: x['pos'])[:6]

def _mon(s): return ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'].index(s[:3].lower()) + 1

# ---------------------------------------------------------------- media
SOCIAL = re.compile(r'(twitter\.com|x\.com|facebook\.com|fb\.watch|instagram\.com|youtube\.com|youtu\.be|t\.me|whatsapp\.com|tiktok\.com|threads\.net|reddit\.com|koo|sharechat)', re.I)
VID_EXT = re.compile(r'\.(mp4|webm|m3u8|mov)(\?|$)', re.I)

def best_src(img):
    ss = img.get('srcset') or img.get('data-srcset') or ''
    if ss:
        cands = [c.strip().split() for c in ss.split(',') if c.strip()]
        try: return max(cands, key=lambda c: float(re.sub(r'\D', '', c[1]) or 0) if len(c) > 1 else 0)[0]
        except Exception: pass
    for a in ('data-src', 'data-lazy-src', 'data-original', 'src'):
        v = img.get(a)
        if v and not v.startswith('data:'): return v

def extract_media(node, base):
    imgs, vids, social, links = [], [], [], []
    if node is None: return imgs, vids, social, links
    seen = set()
    def add(lst, d):
        if d['url'] and d['url'] not in seen: seen.add(d['url']); lst.append(d)
    for im in node.iter('img'):
        u = best_src(im)
        if not u: continue
        u = urllib.parse.urljoin(base, u)
        if re.search(r'(logo|avatar|emoji|spinner|pixel|gravatar|/icons?/)', u, re.I): continue
        cap = ''
        p = im.getparent()
        while p is not None and p.tag not in ('figure', 'body'): p = p.getparent()
        if p is not None and p.tag == 'figure':
            c = p.find('.//figcaption'); cap = text_of(c)
        add(imgs, dict(url=u, alt=clean(im.get('alt')), caption=cap))
    for v in node.iter('video'):
        srcs = [v.get('src')] + [s.get('src') for s in v.iter('source')]
        for s in srcs:
            if s: add(vids, dict(url=urllib.parse.urljoin(base, s), kind='file', poster=urllib.parse.urljoin(base, v.get('poster') or '') if v.get('poster') else ''))
    for f in node.iter('iframe'):
        s = f.get('src') or f.get('data-src')
        if s:
            s = urllib.parse.urljoin(base, s)
            add(vids if re.search(r'(youtube|youtu\.be|vimeo|facebook\.com/plugins/video|dailymotion)', s) else social, dict(url=s, kind='embed'))
    for b in node.iter('blockquote'):
        cls = b.get('class') or ''
        if re.search(r'(twitter-tweet|instagram-media|tiktok-embed|fb-xfbml|bluesky-embed)', cls):
            for a in b.iter('a'):
                h = a.get('href') or ''
                if re.search(r'/status/|/p/|/reel/|/video|/post', h): add(social, dict(url=h, kind=cls.split()[0], text=text_of(b)[:240]))
    for a in node.iter('a'):
        h = a.get('href') or ''
        if not h.startswith('http'): continue
        if VID_EXT.search(h): add(vids, dict(url=h, kind='file', poster='')); continue
        if SOCIAL.search(h) and re.search(r'/status/|/p/|/reel|/watch|/video|/posts?/|youtu', h): add(social, dict(url=h, kind='link', text=text_of(a)[:120]))
        else: add(links, dict(url=h, text=text_of(a)[:120]))
    return imgs, vids, social, links

CONTENT_XP = ["//*[contains(@class,'lhs-area')]", "//*[contains(@class,'entry-content')]", "//*[contains(@class,'post-content')]", "//*[contains(@class,'article-content')]",
              "//*[contains(@class,'story-element')]/..", "//*[contains(@class,'details-content')]", "//*[contains(@class,'single-content')]",
              "//*[contains(@class,'td-post-content')]", "//*[@itemprop='articleBody']", "//article", "//main"]
STRIP_XP = ".//script|.//style|.//noscript|.//aside|.//nav|.//footer|.//form|.//*[contains(@class,'related')]|.//*[contains(@class,'share')]|.//*[contains(@class,'newsletter')]|.//*[contains(@class,'comment')]|.//*[contains(@class,'advert')]"

def content_node(doc):
    best, blen = None, 0
    for xp in CONTENT_XP:
        for n in doc.xpath(xp):
            L = len(text_of(n))
            if L > blen * 1.15 and L > 200: best, blen = n, L
        if best is not None and blen > 800: break
    return best

def body_text(node):
    if node is None: return ''
    for n in node.xpath(STRIP_XP):
        if n.getparent() is not None: n.getparent().remove(n)
    ps = [text_of(p) for p in node.iter('p', 'h2', 'h3', 'li', 'blockquote')]
    ps = [p for p in dict.fromkeys(ps) if len(p) > 25]
    return '\n\n'.join(ps)

# ---------------------------------------------------------------- record building
DEBUNK_TITLE = re.compile(r"(^|[?.:!] ?)(no|nope)[,.!]|\b(did not|does not|do not|is not|was not|are not|has not|never|falsely|false claim|unrelated|fabricated|not real|not true|"
                          r"old video|old image|old photo|unconnected|fake|hoax|misleading|digitally altered|ai-generated|edited|doctored|baseless)\b", re.I)

def vnorm(c):
    n = norm(c)
    if n == 'INCONCLUSIVE' and re.search(r'miscaption|mixture|outdated|out of date|unproven|labeled satire|legend|correct attribution|needs context', c or '', re.I): return 'MISLEADING'
    return n

def verdict_from(*cands):
    for c in cands:
        n = vnorm(c)
        if c and n != 'INCONCLUSIVE': return c, n
    last = next((c for c in reversed(cands) if c), '')
    if last and DEBUNK_TITLE.search(last): return last, 'FAKE'  # title reads as a debunk; weak, flagged by verdict_basis
    return (cands[0] if cands and cands[0] else ''), 'INCONCLUSIVE'

def make_row(site, cfg, url, **f):
    pub = f.get('published', '')
    body = f.get('body', '')
    ev = f.get('claim_date') or ''; basis = 'claimreview' if ev else 'none'
    cands = find_event_dates(f.get('claim', '') + '\n' + (f.get('description', '') or '') + '\n' + body, pub)
    pubd = iso(pub)[:10]
    strong = sorted([c for c in cands if c['strong'] and c['date'] != pubd], key=lambda c: (c['precision'] != 'day', c['year_inferred'], c['pos']))  # explicit full dates first
    if not ev and strong: ev, basis = strong[0]['date'], 'text'  # weak-context hits stay in claim_date_candidates only
    lang = f.get('lang') or cfg.get('lang', '')
    if lang in ('', 'en'):  # script-based override for multilingual feeds (Hindi/Urdu/Telugu...)
        for code, rx in (('hi', r'[\u0900-\u097F]'), ('ur', r'[\u0600-\u06FF]'), ('te', r'[\u0C00-\u0C7F]'), ('ta', r'[\u0B80-\u0BFF]'), ('bn', r'[\u0980-\u09FF]'),
                         ('ml', r'[\u0D00-\u0D7F]'), ('kn', r'[\u0C80-\u0CFF]'), ('gu', r'[\u0A80-\u0AFF]'), ('pa', r'[\u0A00-\u0A7F]'), ('mr', r'[\u0900-\u097F]')):
            if len(re.findall(rx, f.get('title', '') + f.get('claim', ''))) > 8: lang = code; break
    row = dict(id=f"{site}:{f.get('native_id') or hashlib.sha1(url.encode()).hexdigest()[:16]}", site=site, org=cfg.get('org', site), country=cfg.get('country', ''),
               source_tier=cfg.get('tier', ''), url=url, lang=lang[:8], title=f.get('title', ''), claim=f.get('claim', ''),
               claim_basis=f.get('claim_basis', ''), verdict_raw=f.get('verdict_raw', ''), verdict_norm=f.get('verdict_norm', 'INCONCLUSIVE'),
               verdict_basis=f.get('verdict_basis', ''), rating_value=f.get('rating_value', ''), published=pub, modified=f.get('modified', ''),
               claim_date=ev, claim_date_basis=basis, claim_date_candidates=cands, claimant=f.get('claimant', ''), claim_urls=f.get('claim_urls', []),
               author=f.get('author', ''), categories=f.get('categories', []), tags=f.get('tags', []), description=f.get('description', ''), body=body,
               images=f.get('images', []), videos=f.get('videos', []), social_embeds=f.get('social_embeds', []), links=f.get('links', [])[:60],
               lead_image=f.get('lead_image', ''), scraped_at=datetime.now(timezone.utc).isoformat(timespec='seconds'))
    return row

def parse_page(site, cfg, url, h):
    doc = LH.fromstring(h); ld = jsonld(h)
    cr = next((x for x in ld if 'ClaimReview' in types(x)), None)
    na = next((x for x in ld if any(t in types(x) for t in ('NewsArticle', 'Article', 'ReportageNewsArticle', 'BlogPosting'))), {})
    if cfg.get('require_claimreview', True) and not cr: return None
    def meta(k):
        v = doc.xpath(f'//meta[@property="{k}" or @name="{k}"]/@content'); return clean(v[0]) if v else ''
    rr = (cr or {}).get('reviewRating') or {}
    if isinstance(rr, list): rr = rr[0] if rr else {}
    vraw = S(rr.get('alternateName')) or S(rr.get('ratingExplanation'))
    if not vraw and isinstance(rr.get('ratingValue'), str) and not rr['ratingValue'].isdigit(): vraw = rr['ratingValue']
    title = clean(re.sub(r'\s*[|\-–]\s*[^|\-–]{2,30}$', '', meta('og:title') or S((cr or {}).get('name')) or text_of(doc.find('.//title'))))
    item = (cr or {}).get('itemReviewed') or {}
    if isinstance(item, list): item = item[0] if item else {}
    first = item.get('firstAppearance') if isinstance(item.get('firstAppearance'), dict) else {}
    apps = item.get('appearance') or []
    if isinstance(apps, dict): apps = [apps]
    cdate = S(item.get('datePublished')) or S(first.get('datePublished')) or next((S(a.get('datePublished')) for a in apps if isinstance(a, dict) and a.get('datePublished')), '')
    ia = item.get('author') if isinstance(item.get('author'), dict) else {}
    claim_urls = [u for u in [S(first.get('url'))] + [S(a.get('url')) for a in apps if isinstance(a, dict)] + [S(ia.get('sameAs')), S(ia.get('url'))] if u]
    pubd = iso(S((cr or {}).get('datePublished')) or S(na.get('datePublished')) or meta('article:published_time'))
    if cdate and pubd and iso(cdate)[:10] == pubd[:10]: cdate = ''  # some sites copy the review date into itemReviewed.datePublished
    node = content_node(doc)
    imgs, vids, soc, links = extract_media(node, url)
    body = body_text(node)
    lead = meta('og:image') or S(na.get('image'))
    if isinstance(na.get('image'), dict): lead = na['image'].get('url') or lead
    for v in (na.get('video') if isinstance(na.get('video'), list) else [na.get('video')] if na.get('video') else []):
        if isinstance(v, dict) and v.get('contentUrl') or isinstance(v, dict) and v.get('embedUrl'):
            vids.insert(0, dict(url=v.get('contentUrl') or v.get('embedUrl'), kind='jsonld', poster=S(v.get('thumbnailUrl'))))
    if lead and lead not in [i['url'] for i in imgs]: imgs.insert(0, dict(url=lead, alt='', caption='', role='lead'))
    claim = clean(S((cr or {}).get('claimReviewed')))
    if len(claim) < 25: claim = title  # some feeds truncate claimReviewed
    vr, vn = verdict_from(vraw, *([] if vraw else [title]))
    kw = na.get('keywords') or meta('keywords')
    tags = [t.strip() for t in (kw if isinstance(kw, list) else str(kw or '').split(',')) if t.strip()][:25] or [clean(x) for x in doc.xpath('//meta[@property="article:tag"]/@content')]
    lang = S((cr or {}).get('inLanguage')) or S(na.get('inLanguage')) or (doc.get('lang') or '')
    auth = S((cr or {}).get('author')) if isinstance((cr or {}).get('author'), dict) and (cr or {}).get('author', {}).get('@type') == 'Person' else ''
    auth = S(na.get('author')) or auth or meta('author')
    return make_row(site, cfg, url, title=title, claim=claim, claim_basis='claimreview' if len(clean(S((cr or {}).get('claimReviewed')))) >= 25 else 'title', verdict_raw=vr, verdict_norm=vn,
                    verdict_basis='claimreview' if vraw else 'heuristic', rating_value=str(rr.get('ratingValue', '')),
                    published=pubd,
                    modified=iso(S(na.get('dateModified')) or meta('article:modified_time')), claim_date=iso(cdate) if cdate else '', claimant=clean(S(item.get('author'))),
                    claim_urls=claim_urls, author=auth, categories=[S(na.get('articleSection'))] if na.get('articleSection') else [], tags=tags,
                    description=meta('og:description') or meta('description'), body=body, images=imgs, videos=vids, social_embeds=soc, links=links,
                    lead_image=lead, lang=lang[:8], native_id=native_id(url))

def native_id(url):
    m = re.search(r'-(\d{4,})/?$', urllib.parse.urlsplit(url).path)
    return m.group(1) if m else None

# ---------------------------------------------------------------- discovery
def sitemap_entries(u, sub_ok=None, depth=0):
    """-> list of (url, lastmod|None). sub_ok(sub_url, sub_lastmod) -> bool prunes sub-sitemaps."""
    x = get(u, delay=0.3).text
    items = re.findall(r'<(url|sitemap)>(.*?)</\1>', x, re.S)
    out = []
    for kind, blk in items:
        loc = re.search(r'<loc>\s*(.*?)\s*</loc>', blk, re.S); lm = re.search(r'<lastmod>\s*(.*?)\s*</lastmod>', blk)
        if not loc: continue
        loc, lm = html.unescape(loc.group(1)), lm.group(1) if lm else None
        if kind == 'sitemap':
            if depth < 3 and (sub_ok is None or sub_ok(loc, lm)):
                try: out += sitemap_entries(loc, sub_ok, depth + 1)
                except Exception as e: print('  sitemap fail', loc, e, flush=True)
        else: out.append((loc, lm))
    return out

def cc_urls(pattern, since):
    """URL discovery through Common Crawl indexes (every crawl since `since`). Only URLs, content is fetched live."""
    ids = [c['id'] for c in get('https://index.commoncrawl.org/collinfo.json', delay=0).json()
           if (to_utc(c.get('to') or '') or datetime.now(timezone.utc)) >= since]
    out = {}
    for cid in ids:
        try:
            r = get(f'https://index.commoncrawl.org/{cid}-index', delay=1, timeout=300, params=dict(url=pattern, output='json', fl='url,status', filter='=status:200'), stream=True)
            for line in r.iter_lines():
                if line:
                    u = json.loads(line)['url']; u = u.split('?')[0].split('#')[0].rstrip('/')
                    out[u] = None
            print(f'  CC {cid}: {len(out)} urls so far', flush=True)
        except Exception as e: print(f'  CC {cid} fail: {e}', flush=True)
    return list(out.items())

def listing_urls(base, link_re, since, max_pages=400, pages_param='/page/{n}', date_re=r'data-datestring="([^"]+)"'):
    """Paginated archive pages, newest first. Stops at since or when a page yields nothing new."""
    seen, out, p = set(), [], 1
    while p <= max_pages:
        try: h = get(base + (pages_param.format(n=p) if p > 1 else ''), delay=0.6).text
        except Exception as e: print(f'  listing {p} stop: {e}', flush=True); break
        hrefs = [urllib.parse.urljoin(base, x) for x in dict.fromkeys(re.findall(link_re, h))]
        new = [u for u in hrefs if u not in seen]
        if not new: break
        seen.update(new); out += [(u, None) for u in new]
        ds = [to_utc(d) for d in re.findall(date_re, h)]
        if ds and max(ds) < since: break
        p += 1
    print(f'  listing {base}: {len(out)} urls in {p} pages', flush=True)
    return out

def id_bisect(entries, fetch_date, since):
    """URLs end in a monotonic numeric id (BOOM, NewsMeter). Drop ids older than `since` by bisecting on fetched dates."""
    idd = sorted(((int(native_id(u)), u) for u, _ in entries if native_id(u)), key=lambda t: t[0])
    if len(idd) < 50: return entries
    lo, hi = 0, len(idd) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        d = None
        for j in range(mid, min(mid + 6, len(idd))):  # skip pages that fail or lack a date
            d = fetch_date(idd[j][1])
            if d: break
        if d is None or d >= since: hi = mid
        else: lo = mid + 1
    cut = max(0, lo - 20)
    print(f'  id bisect: keeping ids >= {idd[cut][0]} ({len(idd) - cut}/{len(idd)} urls)', flush=True)
    return [(u, None) for _, u in idd[cut:]] + [e for e in entries if not native_id(e[0])]

# ---------------------------------------------------------------- site registry
def _snopes_sub(since):
    def ok(u, lm):
        m = re.search(r'articles-(\d{4})-(\d{2})', u)
        return bool(m) and (int(m.group(1)), int(m.group(2))) >= (since.year, since.month)
    return ok
def _lastmod_sub(since):
    return lambda u, lm: lm is None or (to_utc(lm) or since) >= since

SITES = {
    'pib_factcheck': dict(kind='telegram', channel='PIB_FactCheck', org='PIB Fact Check (Govt of India)', country='IN', tier='government', lang='en'),
    'boomlive': dict(kind='web', org='BOOM', country='IN', tier='ifcn', url_re=r'/fact-check/', ids=True, snowball=True,
                     discover=lambda s: [e for c in ('', '/world', '/sports', '/social', '/politics', '/entertainment', '/business')
                                         for e in listing_urls('https://www.boomlive.in/fact-check' + c, r'href="(/fact-check/[^"?#/]+-\d+)"', s, max_pages=60)]
                     + sitemap_entries('https://www.boomlive.in/news-sitemap-daily.xml') + cc_urls('www.boomlive.in/fact-check/*', s)),
    'newsmeter': dict(kind='web', org='NewsMeter', country='IN', tier='ifcn', url_re=r'/fact-check/', ids=True, snowball=True,
                      discover=lambda s: listing_urls('https://newsmeter.in/fact-check', r'href="(/fact-check/[^"?#]+)"', s, max_pages=60)
                      + cc_urls('newsmeter.in/fact-check/*', s)),
    'quint_webqoof': dict(kind='web', org='The Quint WebQoof', country='IN', tier='ifcn', url_re=r'/webqoof/',
                          discover=lambda s: quint_urls(s)),
    'vishvasnews': dict(kind='web', org='Vishvas News', country='IN', tier='ifcn', url_re=r'/fact-check|/english/|/hindi/|/',
                        discover=lambda s: sitemap_entries('https://www.vishvasnews.com/sitemap_index.xml', sub_ok=lambda u, lm: 'post-sitemap' in u and _lastmod_sub(s)(u, lm))),
    'snopes': dict(kind='web', org='Snopes', country='US', tier='ifcn', url_re=r'/fact-check/',
                   discover=lambda s: sitemap_entries('https://media.snopes.com/sitemaps/sitemap-index.xml', sub_ok=_snopes_sub(s))),
    'fullfact': dict(kind='web', org='Full Fact', country='GB', tier='ifcn', url_re=r'fullfact\.org/[^/]+/[^/]+/$',
                     discover=lambda s: sitemap_entries('https://fullfact.org/sitemap.xml')),
    'altnews': dict(kind='wp', host='www.altnews.in', org='Alt News', country='IN', tier='ifcn', cat_re=r'fact check', lang='en'),
    'factly': dict(kind='wp', host='factly.in', org='Factly', country='IN', tier='ifcn', cat_re=r'fake news|fact check|misleading|false|deepfake', lang='en'),
    'dfrac': dict(kind='wp', host='dfrac.org', org='DFRAC', country='IN', tier='independent', cat_re=r'fact check|fake|misleading', lang='en'),
    'factcheck_org': dict(kind='wp', host='www.factcheck.org', org='FactCheck.org', country='US', tier='ifcn', cat_re=r'factcheck posts|ask factcheck|scicheck|viral|en español', lang='en'),
}

def quint_urls(since):
    """Quintype keeps one sitemap per day (sitemap-daily-YYYY-MM-DD.xml); keep the /webqoof/ (fact-check desk) URLs."""
    days, d, today = [], since.date(), datetime.now(timezone.utc).date()
    while d <= today: days.append(d); d += timedelta(days=1)
    def one(day):
        try:
            x = get(f'https://www.thequint.com/sitemap/sitemap-daily-{day}.xml', delay=0.25).text
            return [(html.unescape(u), None) for u in re.findall(r'<loc>\s*([^<]*/webqoof/[^<]*?)\s*</loc>', x)]
        except Exception: return []
    out = []
    with ThreadPoolExecutor(4) as ex:
        for i, r in enumerate(ex.map(one, days), 1):
            out += r
            if i % 200 == 0: print(f'  quint sitemaps {i}/{len(days)}: {len(out)} webqoof urls', flush=True)
    return out

# ---------------------------------------------------------------- adapters
def run_web(site, cfg, since, args, done, emit):
    entries = cfg['discover'](since)
    entries = [(re.sub(r'#.*', '', u).split('?')[0], lm) for u, lm in entries]
    entries = [(u, lm) for u, lm in dict(entries).items() if re.search(cfg['url_re'], u) and not re.search(r'\.(jpg|png|webp|xml|pdf|gif)$', u)]
    entries = [(u, lm) for u, lm in entries if not lm or (to_utc(lm) or since) >= since]
    def fetch_date(u):
        try:
            h = get(u, delay=args.delay).text; m = re.search(r'"datePublished"\s*:\s*"([^"]+)"', h)
            return to_utc(m.group(1)) if m else None
        except Exception: return None
    if cfg.get('ids'): entries = id_bisect(entries, fetch_date, since)
    todo = [u for u in dict.fromkeys(u for u, _ in entries) if u not in done.get('urls', set())]
    print(f'{site}: {len(entries)} candidate urls, {len(todo)} to fetch', flush=True)
    todo = todo[:args.limit] if args.limit else todo
    stats = dict(ok=0, old=0, no_cr=0, err=0)
    ids = [int(native_id(u)) for u, _ in entries if native_id(u)]
    min_id = min(ids) if ids and cfg.get('ids') else 0
    host = urllib.parse.urlsplit(entries[0][0]).netloc if entries else ''
    seen = set(todo) | done.get('urls', set())
    def work(u):
        try:
            h = get(u, delay=args.delay).text
            links = []
            if cfg.get('snowball'):  # follow in-site article links on the page (related / recent lists) to find URLs the index missed
                for x in re.findall(r'href=["\']([^"\'#?]+)', h):
                    x = urllib.parse.urljoin(u, x).rstrip('/')
                    if urllib.parse.urlsplit(x).netloc == host and re.search(cfg['url_re'], x) and (native_id(x) or '0').isdigit() and int(native_id(x) or 0) >= min_id: links.append(x)
            r = parse_page(site, cfg, u, h)
            if r is None: return 'no_cr', links
            d = to_utc(r['published'])
            if d and d < since: return 'old', links
            return r, links
        except Exception as e: return 'err', []
    frontier, i = todo, 0
    with ThreadPoolExecutor(args.workers) as ex:
        while frontier:
            nxt = []
            for r, links in ex.map(work, frontier):
                i += 1
                if isinstance(r, dict): emit(r); stats['ok'] += 1
                else: stats[r] += 1
                for l in links:
                    if l not in seen: seen.add(l); nxt.append(l)
                if i % 200 == 0: print(f'  {site} {i} fetched (+{len(nxt)} queued) {stats}', flush=True)
            frontier = [] if args.limit else nxt
    print(f'DONE {site}: {stats}', flush=True)

def run_wp(site, cfg, since, args, done, emit):
    api = f"https://{cfg['host']}/wp-json/wp/v2/posts"
    page, n, skipped = 1, 0, 0
    while True:
        try:
            r = get(api, delay=args.delay, params=dict(per_page=50, page=page, _embed=1, orderby='date', order='desc', after=since.strftime('%Y-%m-%dT%H:%M:%S')))
        except requests.HTTPError as e:
            if '400' in str(e): break  # past last page
            raise
        try: posts = r.json()
        except ValueError: break
        if not isinstance(posts, list) or not posts: break
        for p in posts:
            url = p['link']
            if url in done.get('urls', set()): continue
            em = p.get('_embedded', {}); terms = [t for g in em.get('wp:term', []) for t in g]
            cats = [t['name'] for t in terms if t.get('taxonomy') not in ('post_tag', 'author')]  # category + custom taxonomies (Alt News keeps 'Fact Check' in an_content_type)
            tags = [t['name'] for t in terms if t.get('taxonomy') == 'post_tag']
            title = clean(LH.fromstring('<p>' + p['title']['rendered'] + '</p>').text_content()) if p['title']['rendered'] else ''
            if not re.search(cfg['cat_re'], ' '.join(cats + tags) + ' ' + url + ' ' + title, re.I): skipped += 1; continue
            frag = LH.fromstring('<div>' + p['content']['rendered'] + '</div>')
            imgs, vids, soc, links = extract_media(frag, url)
            body = body_text(frag)
            fm = (em.get('wp:featuredmedia') or [{}])[0]; lead = fm.get('source_url') or (p.get('yoast_head_json', {}).get('og_image') or [{}])[0].get('url', '')
            if lead and lead not in [i['url'] for i in imgs]: imgs.insert(0, dict(url=lead, alt=clean(fm.get('alt_text', '')), caption='', role='lead'))
            vr, vn = verdict_from(*[t for t in cats + tags if vnorm(t) != 'INCONCLUSIVE'], title)
            auth = ', '.join(a.get('name') or '' for a in em.get('author', []) if a.get('name'))
            r = make_row(site, cfg, url, native_id=str(p['id']), title=title, claim=title, claim_basis='title', verdict_raw=vr, verdict_norm=vn,
                         verdict_basis='category/tag/title heuristic', published=iso(p['date_gmt'] + '+00:00'), modified=iso(p['modified_gmt'] + '+00:00'),
                         author=auth, categories=cats, tags=tags, description=clean(LH.fromstring('<p>' + (p['excerpt']['rendered'] or '') + '</p>').text_content()),
                         body=body, images=imgs, videos=vids, social_embeds=soc, links=links, lead_image=lead,
                         lang=next((t for t in cats + tags if t.lower() in ('hindi', 'telugu', 'urdu', 'bengali', 'tamil', 'marathi', 'malayalam', 'kannada', 'gujarati', 'punjabi', 'odia')), cfg.get('lang', '')).lower())
            emit(r); n += 1
            if args.limit and n >= args.limit: print(f'DONE {site}: limit {n}'); return
        print(f'  {site} page {page}/{r.headers.get("X-WP-TotalPages") if hasattr(r, "headers") else "?"}: {n} rows, {skipped} non-factcheck skipped', flush=True)
        page += 1
    print(f'DONE {site}: {n} rows, {skipped} skipped as non-fact-check', flush=True)

def run_telegram(site, cfg, since, args, done, emit):
    ch = cfg['channel']; before, n = None, 0
    while True:
        r = get(f'https://t.me/s/{ch}', delay=0.8, params={'before': before} if before else None)
        doc = LH.fromstring(r.text); msgs = doc.xpath('//div[contains(@class,"tgme_widget_message ")][@data-post]')
        if not msgs: break
        oldest = None
        for m in reversed(msgs):  # newest first
            post = m.get('data-post'); mid = int(post.split('/')[1]); oldest = mid
            tm = m.xpath('.//a[contains(@class,"tgme_widget_message_date")]/time/@datetime')
            pub = tm[0] if tm else ''; d = to_utc(pub)
            if d and d < since: print(f'DONE {site}: reached {pub}, {n} rows'); return
            url = f'https://t.me/{post}'
            if url in done.get('urls', set()): continue
            tn = m.xpath('.//div[contains(@class,"tgme_widget_message_text")]')
            if not tn: continue
            for br in tn[0].xpath('.//br'): br.tail = '\n' + (br.tail or '')
            text = '\n'.join(clean(l) for l in tn[0].text_content().split('\n') if clean(l))
            imgs = [dict(url=u, alt='', caption='') for u in re.findall(r"background-image:url\('([^']+)'\)", ' '.join(m.xpath('.//a[contains(@class,"tgme_widget_message_photo_wrap")]/@style')))]
            vids = [dict(url=u, kind='file', poster='') for u in m.xpath('.//video/@src')] + [dict(url=u, kind='thumb', poster='') for u in m.xpath('.//i[contains(@class,"tgme_widget_message_video_thumb")]/@style')]
            vids = [v for v in vids if v['kind'] == 'file'] or ([dict(url=url, kind='telegram-video', poster='')] if m.xpath('.//*[contains(@class,"message_video")]') else [])
            links = [a.get('href') for a in tn[0].xpath('.//a[@href]') if a.get('href', '').startswith('http') and not a.get('href', '').startswith('https://t.me/s/')]
            lp = m.xpath('.//a[contains(@class,"tgme_widget_message_link_preview")]/@href')
            if lp: links.append(lp[0])
            links = list(dict.fromkeys(links))
            soc = [dict(url=u, kind='link', text='') for u in links if SOCIAL.search(u) and 't.me/PIB_FactCheck' not in u]
            tags = re.findall(r'#(\w+)', text)
            vraw, vn = verdict_from(*[('#' + t) for t in tags if vnorm(t) != 'INCONCLUSIVE'], text)
            lines = [l for l in text.split('\n') if not l.startswith('#')]
            claim = next((l for l in lines if len(l) > 60 and not re.match(r'^\W*(?:alert|fake|deepfake|#)', l, re.I)), lines[0] if lines else text)
            claim = re.sub(r'^\W*(?:Claim\s*[:\-]\s*)', '', claim); real = re.search(r'(?:Reality|Fact|Truth)\s*[:\-]\s*(.+)', text)
            views = (m.xpath('.//span[contains(@class,"tgme_widget_message_views")]/text()') or [''])[0]
            row = make_row(site, cfg, url, native_id=str(mid), title=claim[:140], claim=claim, claim_basis='first line of post', verdict_raw=vraw, verdict_norm=vn,
                           verdict_basis='hashtag/text', published=iso(pub), body=text, description=real.group(1) if real else '', tags=tags,
                           images=imgs, videos=vids, social_embeds=soc, links=links, lead_image=imgs[0]['url'] if imgs else '', author='PIB Fact Check', lang='hi' if len(re.findall(r'[ऀ-ॿ]', text)) > len(text) * 0.3 else 'en')
            row['views'] = views; emit(row); n += 1
            if args.limit and n >= args.limit: print(f'DONE {site}: limit {n}'); return
        if oldest is None or oldest <= 1: break
        before = oldest
        print(f'  {site} before={before}: {n} rows', flush=True)
    print(f'DONE {site}: {n} rows')

RUN = dict(telegram=run_telegram, wp=run_wp, web=run_web)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('sites', nargs='*'); ap.add_argument('--list', action='store_true')
    ap.add_argument('--years', type=float, default=3); ap.add_argument('--limit', type=int, default=0); ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--delay', type=float, default=0.4); ap.add_argument('--out', default=str(DATA / 'factchecks_3y.jsonl'))
    ap.add_argument('--download-media', default='', help='directory to download lead/body images into (off by default)')
    a = ap.parse_args()
    if a.list or not a.sites: print('\n'.join(f'{k:15} {v["kind"]:9} {v["org"]}' for k, v in SITES.items())); return
    sites = list(SITES) if a.sites == ['all'] else a.sites
    since = datetime.now(timezone.utc) - timedelta(days=365.25 * a.years)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    seen_ids, seen_urls = set(), set()
    if out.exists():
        for l in open(out, encoding='utf-8'):
            try: r = json.loads(l); seen_ids.add(r['id']); seen_urls.add(r['url'])
            except Exception: pass
    wl = threading.Lock(); fh = open(out, 'a', encoding='utf-8')
    def emit(r):
        if a.download_media: download_media(r, Path(a.download_media))
        with wl:
            if r['id'] in seen_ids: return
            seen_ids.add(r['id']); fh.write(json.dumps(r, ensure_ascii=False) + '\n'); fh.flush()
    for s in sites:
        cfg = SITES[s]; print(f'== {s} since {since.date()}', flush=True)
        try: RUN[cfg['kind']](s, cfg, since, a, dict(urls=seen_urls), emit)
        except Exception as e: print(f'FAILED {s}: {type(e).__name__}: {e}', flush=True)

def download_media(r, root, max_per_row=8):
    d = root / r['site']; d.mkdir(parents=True, exist_ok=True)
    for im in r['images'][:max_per_row]:
        u = im['url']; ext = '.' + (re.search(r'\.(jpe?g|png|webp|gif)(?:$|\?)', u, re.I) or [0, 'jpg'])[1]
        f = d / (hashlib.sha1(u.encode()).hexdigest()[:16] + ext)
        if not f.exists():
            try: f.write_bytes(get(u, delay=0.1, timeout=30).content)
            except Exception: continue
        im['local'] = str(f)

if __name__ == '__main__': main()
