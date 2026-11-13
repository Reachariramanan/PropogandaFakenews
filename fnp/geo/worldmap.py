"""World map geometry for the Propaganda Lab: Natural Earth admin-0 (India point-of-view
worldview, 10m, public domain) -> simplified -> Equal Earth projection -> one SVG path per ISO3.

Built once; the UI draws plain SVG paths (no JS map library, no runtime network).
  build_world_paths(out_path) -> {"viewBox": "...", "countries": {ISO3: {"name", "d"}}}
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

from shapely.geometry import shape
from shapely.ops import unary_union

from fnp.config import DATA_DIR

SRC = DATA_DIR / "geo" / "src" / "ne_10m_admin_0_countries_ind.geojson"
TOL_DEG = 0.12          # Douglas-Peucker tolerance; keeps the file ~200-300 KB
MIN_AREA_DEG2 = 0.08    # drop tiny islands (they still exist as features with a larger part)
SCALE = 160.0

# Equal Earth (Savric, Patterson, Jenny 2018)
A1, A2, A3, A4 = 1.340264, -0.081106, 0.000893, 0.003796
M = math.sqrt(3) / 2


def equal_earth(lon: float, lat: float) -> tuple[float, float]:
    t = math.asin(M * math.sin(math.radians(lat)))
    t2 = t * t
    t6 = t2 * t2 * t2
    x = math.radians(lon) * math.cos(t) / (M * (A1 + 3 * A2 * t2 + t6 * (7 * A3 + 9 * A4 * t2)))
    y = t * (A1 + A2 * t2 + t6 * (A3 + A4 * t2))
    return x * SCALE, -y * SCALE


def iso3_of(props: dict) -> str:
    for k in ("ISO_A3_EH", "ISO_A3", "ADM0_A3"):
        v = props.get(k)
        if v and v != "-99":
            return v
    return props["ADM0_A3"]


def _ring(coords) -> str:
    pts = [equal_earth(x, y) for x, y in coords]
    return "M" + "L".join(f"{x:.1f},{y:.1f}" for x, y in pts) + "Z"


def build_world_paths(out_path: Path) -> dict:
    feats = json.loads(SRC.read_text())["features"]
    geoms, names = defaultdict(list), {}
    for f in feats:
        p = f["properties"]
        if p.get("ADM0_A3") == "ATA":          # Antarctica: no posts, wastes map height
            continue
        k = iso3_of(p)
        geoms[k].append(shape(f["geometry"]))
        names.setdefault(k, p.get("NAME") or k)
    out, xs, ys = {}, [], []
    for k, gs in geoms.items():
        g = unary_union(gs).simplify(TOL_DEG, preserve_topology=True)
        polys = list(getattr(g, "geoms", [g]))
        big = [q for q in polys if q.area >= MIN_AREA_DEG2] or [max(polys, key=lambda q: q.area)]
        d = "".join(_ring(q.exterior.coords) for q in big if not q.is_empty)
        if not d:
            continue
        out[k] = {"name": names[k], "d": d}
        for q in big:
            for x, y in q.exterior.coords:
                px, py = equal_earth(x, y)
                xs.append(px); ys.append(py)
    pad = 4
    vb = f"{min(xs)-pad:.0f} {min(ys)-pad:.0f} {max(xs)-min(xs)+2*pad:.0f} {max(ys)-min(ys)+2*pad:.0f}"
    res = {"viewBox": vb, "projection": "Equal Earth", "source": "Natural Earth 10m admin-0, India worldview (public domain)",
           "countries": out}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, ensure_ascii=False, separators=(",", ":")))
    return res
