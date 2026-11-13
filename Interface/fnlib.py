"""Shared helpers: tokenizer, TF-IDF index (scikit-learn or pure python), text signals."""
import math
import re
from collections import Counter, defaultdict

try:  # optional accelerator
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    HAVE_SKLEARN = True
except Exception:  # pragma: no cover
    HAVE_SKLEARN = False

STOP = set("""a an and are as at be been but by for from has have he her his i in is it its of on or she that the
their them they this to was were will with you your we our not no do does did so if than then there these those
who whom which what when where why how also just about into over after before said says say shared sharing
video viral claim claims claimed social media post posted users user""".split())

_TOKEN_RE = re.compile(r"[\wऀ-෿]+", re.UNICODE)


def tok(text):
    """Unigrams + bigrams over lowercase word tokens (Indic scripts kept intact)."""
    words = [w for w in _TOKEN_RE.findall((text or "").lower())
             if len(w) > 1 and w not in STOP and not w.isdigit()]
    out = list(words)
    out += [a + "_" + b for a, b in zip(words, words[1:])]
    return out


class TfidfIndex:
    """Cosine retrieval over short texts. Backend: scikit-learn if present, else pure python."""

    def __init__(self):
        self.backend = "sklearn" if HAVE_SKLEARN else "python"
        self.n = 0

    # ---- fitting
    def fit(self, texts):
        self.n = len(texts)
        if self.backend == "sklearn":
            self.vec = TfidfVectorizer(tokenizer=tok, lowercase=False, token_pattern=None,
                                       sublinear_tf=True, norm="l2", dtype=np.float32)
            self.X = self.vec.fit_transform(texts).tocsr() if texts else None
            return self
        docs = [Counter(tok(t)) for t in texts]
        df = Counter()
        for d in docs:
            df.update(d.keys())
        n = max(len(docs), 1)
        self.idf = {t: math.log((1 + n) / (1 + c)) + 1.0 for t, c in df.items()}
        self.vecs = [self._weigh(d) for d in docs]
        self.inv = defaultdict(list)
        for i, v in enumerate(self.vecs):
            for t, w in v.items():
                self.inv[t].append((i, w))
        return self

    def _weigh(self, counts):
        v = {t: (1 + math.log(c)) * self.idf[t] for t, c in counts.items() if t in self.idf}
        norm = math.sqrt(sum(w * w for w in v.values()))
        return {t: w / norm for t, w in v.items()} if norm else {}

    # ---- querying
    def query(self, text, k=5):
        """Return [(doc_index, cosine)] best first."""
        if not self.n:
            return []
        if self.backend == "sklearn":
            q = self.vec.transform([text])
            if q.nnz == 0:
                return []
            s = (self.X @ q.T).toarray().ravel()
            idx = np.argsort(-s)[:k]
            return [(int(i), float(s[i])) for i in idx if s[i] > 0]
        q = self._weigh(Counter(tok(text)))
        acc = defaultdict(float)
        for t, w in q.items():
            for i, dw in self.inv.get(t, ()):
                acc[i] += w * dw
        return sorted(acc.items(), key=lambda x: -x[1])[:k]

    def neighbors(self, k=5, thr=0.3):
        """For every indexed doc: [(other_index, cosine)] excluding itself, cosine >= thr."""
        res = []
        if self.backend == "sklearn" and self.X is not None:
            for start in range(0, self.n, 800):
                S = (self.X[start:start + 800] @ self.X.T).toarray()
                for r in range(S.shape[0]):
                    S[r, start + r] = 0
                    top = np.argsort(-S[r])[:k]
                    res.append([(int(j), float(S[r, j])) for j in top if S[r, j] >= thr])
            return res
        for i in range(self.n):
            acc = defaultdict(float)
            for t, w in self.vecs[i].items():
                for j, dw in self.inv[t]:
                    if j != i:
                        acc[j] += w * dw
            top = sorted(((j, s) for j, s in acc.items() if s >= thr), key=lambda x: -x[1])[:k]
            res.append(top)
        return res


URGENCY = ["share", "forward", "urgent", "breaking", "shocking", "exposed", "banned", "must watch",
           "viral", "alert", "before it is deleted", "wake up", "100%", "they don't want you to know",
           "secret", "conspiracy", "betrayal", "hidden truth"]


def signals(text):
    """Linguistic signals of a text; values only, no verdict."""
    t = (text or "").strip()
    letters = [c for c in t if c.isalpha()]
    upper = [c for c in letters if c.isupper()]
    words = t.split()
    caps_words = [w for w in words if len(re.sub(r"\W", "", w)) >= 3 and re.sub(r"\W", "", w).isupper()
                  and re.sub(r"\W", "", w).isascii()]
    ex = t.count("!")
    low = t.lower()
    hits = sorted({u for u in URGENCY if u in low})
    caps_ratio = len(upper) / len(letters) if letters else 0.0
    ex_ratio = ex / max(len(t), 1)
    sens = min(1.0, 1.4 * min(caps_ratio, 0.5) + min(ex, 4) * 0.08 + len(hits) * 0.12
               + (0.1 if len(caps_words) >= 2 else 0))
    return {
        "chars": len(t), "words": len(words),
        "caps_ratio": round(caps_ratio, 4),
        "allcaps_words": len(caps_words),
        "exclamations": ex,
        "exclamation_ratio": round(ex_ratio, 4),
        "question_marks": t.count("?"),
        "urgency_terms": hits,
        "hashtags": len(re.findall(r"#\w+", t)),
        "urls": len(re.findall(r"https?://\S+", t)),
        "sensational": round(sens, 3),
    }


def clean_json(v):
    """Replace NaN/inf with None recursively so output is strict JSON that browsers can parse."""
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, dict):
        return {k: clean_json(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean_json(x) for x in v]
    return v
