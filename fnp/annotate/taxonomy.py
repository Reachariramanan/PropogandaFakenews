"""Seed taxonomies. Claim types are the configured form-of-claim set (not subject domain);
techniques are the 14 merged PTC classes the Laya span encoder is trained on. Seeding only ADDS
missing codes, so later edits made in the DB (descriptions, active flags) are preserved."""
from __future__ import annotations

from ..eval.ptc_scorer import CLASSES14, MERGE14
from ..llm.prompts import TECHNIQUE_DEFS

CLAIM_TYPES = [
    ("DIRECT_STATEMENT", "Direct statement", "The author asserts the claim in their own voice."),
    ("REPORTED_OR_REQUOTED_STATEMENT", "Reported / requoted", "Someone else's statement is quoted or reported (\"X said ...\", forwarded text)."),
    ("ALLEGATION_OR_ACCUSATION", "Allegation / accusation", "Accuses a person or group of wrongdoing, with or without evidence."),
    ("PREDICTION_OR_FORECAST", "Prediction / forecast", "A claim about a future event or outcome."),
    ("QUESTION_WITH_IMPLICIT_CLAIM", "Question with implicit claim", "A rhetorical or leading question that presupposes a factual claim."),
    ("DENIAL_OR_COUNTERCLAIM", "Denial / counterclaim", "Denies or contradicts another claim."),
    ("PARAPHRASED_STATEMENT", "Paraphrased statement", "A restatement or summary of a claim made elsewhere, not a verbatim quote."),
    ("OTHER", "Other", "A claim that fits none of the forms above."),
    ("UNCLEAR", "Unclear", "Cannot tell the form of the claim from the text."),
]


def technique_rows():
    members: dict[str, list[str]] = {}
    for k in TECHNIQUE_DEFS:
        members.setdefault(MERGE14.get(k, k), []).append(k)
    for i, code in enumerate(CLASSES14):
        mem = members.get(code, [code])
        desc = " / ".join(TECHNIQUE_DEFS[m] for m in mem)
        yield code, code.replace("_", " ").replace(",", " / "), desc, mem, i


def seed(cur) -> dict:
    n = {"claim_types": 0, "techniques": 0}
    for i, (c, l, d) in enumerate(CLAIM_TYPES):
        cur.execute("INSERT INTO taxonomy_claim_type (code,label,description,sort_order) VALUES (%s,%s,%s,%s) "
                    "ON CONFLICT (code) DO NOTHING", (c, l, d, i))
        n["claim_types"] += cur.rowcount
    for code, label, desc, mem, i in technique_rows():
        cur.execute("INSERT INTO taxonomy_technique (code,label,description,ptc18_members,sort_order) "
                    "VALUES (%s,%s,%s,%s,%s) ON CONFLICT (code) DO NOTHING", (code, label, desc, mem, i))
        n["techniques"] += cur.rowcount
    return n
