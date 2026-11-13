#!/usr/bin/env python3
"""Generic fact-check scraper. Reads ClaimReview JSON-LD from article pages.
usage: fc_scrape.py SITE --sitemap URL [--sitemap URL ...] [--pattern REGEX] [--limit N] [--workers 3] [--delay 0.5]
Writes Interface/factchecks/<SITE>.jsonl. Resumable (skips URLs already in the file)."""
import re, sys, json, time, html, argparse, gzip, urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
OUT = Path(__file__).resolve().parent.parent / 'factchecks'
UA = {'User-Agent': 'Mozilla/5.0'}
def get(u, timeout=40):
    for i in range(3):
        try:
            b = urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=timeout).read()
            if b[:2] == b'\x1f\x8b': b = gzip.decompress(b)
            return b.decode('utf-8', 'replace')
        except Exception as e: err = e; time.sleep(1.5)
    raise err
def sitemap_urls(u, depth=0):
    x = get(u); locs = re.findall(r'<loc>\s*(.*?)\s*</loc>', x, re.S)
    if '<sitemapindex' in x and depth < 3:
        out = []
        for s in locs:
            try: out += sitemap_urls(html.unescape(s), depth + 1)
            except Exception: pass
            time.sleep(0.3)
        return out
    return [html.unescape(l) for l in locs]
NORM = [('FAKE', r'\b(false|fake|fabricated|hoax|scam|fraud|not true|no evidence|baseless|incorrect|untrue|pants on fire|wrong|manipulated|deepfake|digitally altered|ai[- ]generated|satire|doctored|old video|unrelated|photoshop)'),
        ('MISLEADING', r'(misleading|mostly false|half[- ]true|partly|partially|missing context|out of context|exaggerat|mixture|unproven|misattributed|distort|lacks context|selective)'),
        ('LEGIT', r'\b(true|correct|accurate|real|genuine|authentic|verified|mostly true|legit)\b'),
        ]
def norm(v):
    s = (v or '').lower()
    if not s: return 'INCONCLUSIVE'
    if re.search(NORM[1][1], s): return 'MISLEADING'
    if re.search(r'\b(not|n.t)\s+(true|accurate|correct|real|genuine)', s): return 'FAKE'
    if re.search(NORM[0][1], s): return 'FAKE'
    if re.search(NORM[2][1], s): return 'LEGIT'
    return 'INCONCLUSIVE'
def jsonld(h):
    out = []
    for m in re.finditer(r'<script[^>]*application/ld\+json[^>]*>(.*?)</script>', h, re.S | re.I):
        try: d = json.loads(m.group(1), strict=False)
        except Exception: continue
        stack = d if isinstance(d, list) else [d]
        while stack:
            x = stack.pop()
            if isinstance(x, dict):
                if '@graph' in x: stack += x['@graph']
                out.append(x)
            elif isinstance(x, list): stack += x
    return out
def S(v):
    if isinstance(v, list): v = ' '.join(S(i) for i in v if i)
    elif isinstance(v, dict): v = v.get('name') or v.get('@value') or ''
    return str(v or '')
def types(x):
    t = x.get('@type', []); return t if isinstance(t, list) else [t]
def paras(h):
    m = re.search(r'<(?:article|div)[^>]+class="[^"]*(?:entry-content|post-content|article-content|story|details-content|td-post-content|single-content|content-area)[^"]*"[^>]*>(.*)', h, re.S)
    seg = m.group(1) if m else h
    seg = re.sub(r'<(script|style|noscript|aside|nav|footer|form)[^>]*>.*?</\1>', '', seg, flags=re.S | re.I)
    ps = [re.sub(r'\s+', ' ', html.unescape(re.sub(r'<[^>]+>', '', p)).replace('\xa0', ' ')).strip() for p in re.findall(r'<p[^>]*>(.*?)</p>', seg, re.S)]
    return [p for p in ps if len(p) > 40]
def meta(h, k):
    m = re.search(r'<meta[^>]+(?:property|name)=["\']%s["\'][^>]+content=["\']([^"\']*)' % re.escape(k), h, re.I) or re.search(r'<meta[^>]+content=["\']([^"\']*)["\'][^>]+(?:property|name)=["\']%s["\']' % re.escape(k), h, re.I)
    return html.unescape(m.group(1)).strip() if m else ''
def parse(site, u):
    h = get(u); ld = jsonld(h)
    cr = next((x for x in ld if 'ClaimReview' in types(x)), None)
    na = next((x for x in ld if any(t in types(x) for t in ('NewsArticle', 'Article', 'ReportageNewsArticle', 'BlogPosting'))), {})
    if not cr: return None
    rr = cr.get('reviewRating') or {}
    if isinstance(rr, list): rr = rr[0] if rr else {}
    verdict = S(rr.get('alternateName')) or S(rr.get('ratingExplanation'))
    if not verdict and isinstance(rr.get('ratingValue'), str) and not rr['ratingValue'].isdigit(): verdict = rr['ratingValue']
    claim = S(cr.get('claimReviewed'))
    body = na.get('articleBody') or ''
    ps = paras(h)
    title = html.unescape(re.sub(r'\s*[|\-–]\s*[^|\-–]{2,30}$', '', meta(h, 'og:title') or cr.get('name') or '')).strip()
    lang = cr.get('inLanguage') or na.get('inLanguage') or (re.search(r'<html[^>]+lang="([^"]+)"', h) or [0, ''])[1]
    if isinstance(lang, dict): lang = lang.get('name', '')
    return dict(id=f'{site}:{re.sub(r"[^a-z0-9]+","-",u.lower().split("://",1)[1])[:120]}', site=site, url=u, claim=claim.strip(), verdict_raw=verdict.strip(),
        verdict_norm=norm(verdict), rating_value=str(rr.get('ratingValue', '')), published=(cr.get('datePublished') or na.get('datePublished') or ''),
        lang=str(lang)[:8], title=title, body='\n\n'.join(ps) if len(' '.join(ps)) > len(body) else body, description=meta(h, 'og:description') or meta(h, 'description'))
if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('site'); ap.add_argument('--sitemap', action='append', default=[]); ap.add_argument('--urls-file')
    ap.add_argument('--pattern', default='.'); ap.add_argument('--limit', type=int, default=2000); ap.add_argument('--workers', type=int, default=3); ap.add_argument('--delay', type=float, default=0.5)
    a = ap.parse_args(); OUT.mkdir(exist_ok=True); f = OUT / f'{a.site}.jsonl'
    urls = []
    for s in a.sitemap: urls += sitemap_urls(s)
    if a.urls_file: urls += [l.strip() for l in open(a.urls_file) if l.strip()]
    urls = [u for u in dict.fromkeys(urls) if re.search(a.pattern, u) and not re.search(r'\.(jpg|png|webp|xml|pdf)$', u)]
    done = set(json.loads(l)['url'] for l in open(f)) if f.exists() else set()
    todo = [u for u in urls if u not in done][:a.limit]
    print(f'{a.site}: {len(urls)} candidate urls, {len(done)} done, fetching {len(todo)}', flush=True)
    n = skip = err = 0
    def work(u):
        time.sleep(a.delay)
        try: return parse(a.site, u)
        except Exception as e: return 'ERR'
    with ThreadPoolExecutor(a.workers) as ex, open(f, 'a', encoding='utf-8') as out:
        for r in ex.map(work, todo):
            if r == 'ERR': err += 1
            elif r is None: skip += 1
            else: out.write(json.dumps(r, ensure_ascii=False) + '\n'); out.flush(); n += 1
            if (n + skip + err) % 100 == 0: print(f'  ok={n} no_claimreview={skip} err={err}', flush=True)
    print(f'DONE {a.site}: ok={n} no_claimreview={skip} err={err}')
