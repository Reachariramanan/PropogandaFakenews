"""Update Fake_News_Propaganda_Detection.pptx: fix defects, add method/maths slides, notes."""
import copy, re, sys
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION

SRC, OUT = sys.argv[1], sys.argv[2]
prs = Presentation(SRC)
S = list(prs.slides)
T5 = S[4]  # methods slide: source of card shape styles
shp = list(T5.shapes)


def _fill(sh):
    try:
        return str(sh.fill.fore_color.rgb)
    except Exception:
        return None


auto = [s for s in shp if s.shape_type == 1]
TPL = {"bg": auto[0],
       "card": next(s for s in auto if _fill(s) == "FFFFFF"),
       "bar": next(s for s in auto if s.height < Inches(0.12)),
       "pill": next(s for s in auto if _fill(s) == "E6EFFA"),
       "band": next(s for s in auto if _fill(s) == "142239")}

NAVY, BODY, MUTED = "142239", "364254", "69778B"
COL = {"blue": ("457CBE", "E6EFFA"), "purple": ("8063B0", "F0EAF9"),
       "green": ("289D61", "E2F5EB"), "orange": ("DD9E2F", "FCF4E0"),
       "red": ("C24848", "FBE9E9")}
FONT = "Aptos"


def clone(slide, key, x, y, w, h, fill=None):
    el = copy.deepcopy(TPL[key]._element)
    slide.shapes._spTree.append(el)
    sh = slide.shapes[-1]
    sh.left, sh.top, sh.width, sh.height = Inches(x), Inches(y), Inches(w), Inches(h)
    if fill:
        sh.fill.solid(); sh.fill.fore_color.rgb = RGBColor.from_string(fill)
    return sh


def tb(slide, x, y, w, h, paras, size=10, color=BODY, bold=False, italic=False,
       align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, space_after=0):
    """paras: str | list of paragraphs; a paragraph is str or list of (text, {opts}) runs."""
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.04); tf.margin_top = tf.margin_bottom = Inches(0.02)
    if isinstance(paras, str):
        paras = [paras]
    for i, p in enumerate(paras):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = align; para.space_after = Pt(space_after)
        runs = [(p, {})] if isinstance(p, str) else p
        for text, o in runs:
            r = para.add_run(); r.text = text
            f = r.font; f.name = FONT; f.size = Pt(o.get("size", size))
            f.bold = o.get("bold", bold); f.italic = o.get("italic", italic)
            f.color.rgb = RGBColor.from_string(o.get("color", color))
    return box


def frame(slide, kicker, title, sub, page):
    clone(slide, "bg", 0, 0, 13.333, 7.5)
    tb(slide, 0.65, 0.35, 11.8, 0.28, kicker, 10, COL["green"][0], bold=True)
    tb(slide, 0.65, 0.65, 12.0, 0.55, title, 26, NAVY, bold=True)
    tb(slide, 0.67, 1.22, 12.0, 0.4, sub, 12, MUTED)
    tb(slide, 0.65, 7.08, 6.0, 0.2, "FAKE NEWS & PROPAGANDA DETECTION", 8, MUTED, bold=True)
    tb(slide, 12.25, 7.06, 0.5, 0.22, str(page), 9, MUTED, align=PP_ALIGN.RIGHT)


def method_card(slide, x, y, w, h, c, name, q, does, inp, out):
    dark, light = COL[c]
    clone(slide, "card", x, y, w, h)
    clone(slide, "bar", x, y, w, 0.09, dark)
    tb(slide, x + .12, y + .15, w - .24, .3, name, 13, NAVY, bold=True)
    tb(slide, x + .12, y + .45, w - .24, .28, q, 10.5, dark, bold=True, italic=True)
    tb(slide, x + .12, y + .76, w - .24, .52, does, 10, BODY)
    tb(slide, x + .12, y + 1.28, w - .24, .38, [[("IN  ", {"bold": True, "color": dark}), (inp, {})]], 9.5, BODY)
    clone(slide, "pill", x + .1, y + h - .45, w - .2, .34, light)
    tb(slide, x + .14, y + h - .44, w - .28, .32, [[("OUT  ", {}), (out, {})]], 9.5, dark, bold=True,
       anchor=MSO_ANCHOR.MIDDLE)


def band(slide, y, label, text, h=0.62):
    clone(slide, "band", 0.65, y, 12.05, h)
    tb(slide, 0.88, y + .05, 1.9, h - .1, label, 11, "91DDB3", bold=True, anchor=MSO_ANCHOR.MIDDLE)
    tb(slide, 2.8, y + .05, 9.7, h - .1, text, 10.5, "FFFFFF", bold=True, anchor=MSO_ANCHOR.MIDDLE)


def new_slide():
    return prs.slides.add_slide(prs.slide_layouts[6])


def grid6(slide, cards):
    xs, ys, w, h = (0.65, 4.725, 8.8), (1.75, 3.98), 3.9, 2.1
    for k, c in enumerate(cards):
        method_card(slide, xs[k % 3], ys[k // 3], w, h, *c)


# ---------- new slide A: core maths 1 ----------
A = new_slide()
frame(A, "04 / THE CORE MATH (1 OF 2): WEIGHING THE EVIDENCE",
      "How strong is the case for or against a claim?",
      "Input to every method: the evidence list for one claim. Each answers one question and outputs a number the next step uses.", 5)
grid6(A, [
    ("blue", "Source reliability", "How much should we trust this source?",
     "Start from a trust level set by source type, then raise or lower it with every past hit or miss.",
     "source type + history of verified / debunked claims", "trust score (0–1) per source"),
    ("blue", "Truth discovery", "When sources disagree, who is usually right?",
     "Alternately estimate which claims are true and which sources agree with the truth, until stable.",
     "table of sources × claims with each stance", "learned accuracy per source"),
    ("green", "Weight of evidence", "Adding it all up, is the claim true?",
     "Start at 50/50. Each item pushes up (support) or down (refute), scaled by trust and independence.",
     "evidence items: stance, trust, independence, relevance", "P(claim is true)"),
    ("purple", "Dempster–Shafer", "What do we know, and what don't we?",
     "Split belief into “true”, “false” and “can't tell”. Conflicting sources add to “can't tell”.",
     "the same evidence items", "belief, plausibility, uncertainty"),
    ("orange", "Meta-analysis", "Do the sources agree? How sure can we be?",
     "Pool sources like clinical studies; widen the error band when they disagree with each other.",
     "evidence strength from each independent source", "confidence interval + disagreement"),
    ("orange", "Information gain", "Did the evidence actually tell us anything?",
     "Measure how far the evidence moved belief from the starting guess. Near zero = nothing learned.",
     "belief before and after the evidence", "bits learned (low → lean INCONCLUSIVE)"),
])
band(A, 6.25, "RESULT PER CLAIM",
     "P(true) with an honest error band, an uncertainty score and how much we learned. No single method decides.", 0.6)
A.notes_slide.notes_text_frame.text = (
    "This is the heart of the maths, explained without formulas. All six methods look at the same evidence list for one claim.\n"
    "Blue = who to trust: reliability starts from the source type (official record > news > anonymous post) and is updated by track record; "
    "truth discovery (Dawid–Skene, TruthFinder) learns source accuracy and claim truth together.\n"
    "Green = the main score: weight of evidence adds up log-odds, so ten weak items or one strong item can move the needle.\n"
    "Purple = Dempster–Shafer keeps 'don't know' as its own bucket, so conflict raises uncertainty instead of forcing a side (Yager rule).\n"
    "Orange = meta-analysis gives an error band (DerSimonian–Laird random effects); information gain says whether we learned anything.\n"
    "Key message: the output is not just a probability but a probability with an honest uncertainty.")

# ---------- new slide B: core maths 2 ----------
B = new_slide()
frame(B, "05 / THE CORE MATH (2 OF 2): CONSISTENCY, TIMING AND SPREAD",
      "Does the story hang together, and did it spread naturally?",
      "These methods look beyond a single claim: at the text's persuasion style, the story's other claims, dates and how it spread.", 6)
grid6(B, [
    ("purple", "Propaganda score", "How likely is the text using propaganda?",
     "Each tagged span is one chance of propaganda, weighted by how reliably that technique is detected.",
     "tagged spans + technique type (18 PTC techniques)", "P(propaganda), density, one-sidedness"),
    ("green", "Belief propagation", "Do the story's claims hang together?",
     "Pass beliefs along support / contradict links: if B is false and A implies B, A is pulled down.",
     "claims of one story + links between them", "consistency-adjusted P(true) per claim"),
    ("blue", "Centrality (PageRank)", "Which claims carry the story?",
     "Rank claims by how many important, independent links point to them (copies removed first).",
     "claim–evidence graph after de-duplication", "importance per claim"),
    ("red", "Timeline check", "Is the chronology even possible?",
     "Each date clash (e.g. a “new” video first seen years earlier) raises the chance of recycled content.",
     "dates in the claim vs dates in the evidence", "P(timeline inconsistent) → MISLEADING"),
    ("orange", "Coordination (tensor)", "Are accounts posting in lock-step?",
     "Find groups of accounts that push the same claims in the same time windows.",
     "cube of accounts × claims × time", "coordination score"),
    ("orange", "Spread signals (HMM)", "Did it spread unnaturally?",
     "Detect burst phases and linked episodes of the same narrative (details on the next slide).",
     "post times, platforms, authors, near-duplicates", "burst, spread, diversity, duplicate rate"),
])
band(B, 6.25, "ALL TOGETHER",
     "30 numbers per claim (its “feature vector”) go to the deciders. Signals add weight; none alone proves a claim fake.", 0.6)
B.notes_slide.notes_text_frame.text = (
    "Second half of the maths. Slide 5 judged one claim against its evidence; this slide adds context.\n"
    "Propaganda score: a noisy-OR — the text is propaganda if at least one tagged span is real; each span counts by how precise that technique detector is. "
    "We also report density per 100 words and one-sidedness. Propaganda is separate from truth: a true claim can be pushed with propaganda.\n"
    "Belief propagation: claims in one story constrain each other. Centrality: which claims matter most for the post verdict.\n"
    "Timeline: a small Bayesian network flags impossible chronology — the classic 'old video, new caption' case, which is MISLEADING.\n"
    "Coordination and spread: tensor decomposition and the HMM from the EDA work find lock-step posting and unnatural bursts. These raise risk but never prove falsity on their own.\n"
    "Everything ends up as 30 numbers per claim.")

# ---------- new slide C: deciders ----------
C = new_slide()
frame(C, "06 / DECIDING",
      "Three deciders vote; a small trained model learns to decide",
      "Input: the 30 numbers + a short evidence summary. Output: a probability for each of the 5 labels.", 8)
dec = [
    ("blue", "1  Rule head", "the 30 numbers",
     "Transparent if-then rules: very low P(true) + strong independent refutation → FAKE; high P(true) + low propaganda → LEGIT; context clash → MISLEADING.",
     "label + margin; fully explainable"),
    ("purple", "2  Gemma adjudicator", "evidence summary only, never the raw post",
     "The LLM reasons over the evidence table and writes a rationale. It never sees the post's wording, so style cannot sway it.",
     "label probabilities + written reason"),
    ("green", "3  Stacker", "30 numbers + rule vote + Gemma vote",
     "A simple learned model (logistic regression) that learns from ~1,800 gold-labelled claims how much to trust each signal, then calibrates.",
     "calibrated probability per label"),
    ("orange", "4  Laya (trained by us)", "the same evidence, written as text",
     "A 322M-parameter decision model. Step 1: imitate gold labels + Gemma. Step 2: reinforcement learning rewards honest, calibrated probabilities.",
     "label probabilities; fast, 3.4 GB GPU"),
]
for k, (c, name, inp, what, out) in enumerate(dec):
    x, w, y, h = 0.65 + k * 3.2, 2.85, 1.8, 3.4
    dark, light = COL[c]
    clone(C, "card", x, y, w, h); clone(C, "bar", x, y, w, .09, dark)
    tb(C, x + .12, y + .17, w - .24, .32, name, 13, NAVY, bold=True)
    tb(C, x + .12, y + .58, w - .24, .5, [[("INPUT  ", {}), (inp, {})]], 10, dark, bold=True)
    tb(C, x + .12, y + 1.15, w - .24, 1.6, [[("WHAT IT DOES  ", {"bold": True, "color": NAVY}), (what, {})]], 11.5, BODY)
    clone(C, "pill", x + .1, y + h - .62, w - .2, .5, light)
    tb(C, x + .16, y + h - .6, w - .32, .46, [[("OUTPUT  ", {}), (out, {})]], 9.5, dark, bold=True, anchor=MSO_ANCHOR.MIDDLE)
band(C, 5.55, "RESULT", "Laya beat Gemma as the decider (macro-F1 0.61 vs 0.38, far better calibrated). "
     "So Laya decides; Gemma stays the reader.", 0.62)
tb(C, 0.65, 6.35, 12.05, 0.3, "Reinforcement learning (RLCD): the model is rewarded with a proper scoring rule, so it scores best only by reporting its true confidence; lazy “inconclusive” answers are penalised.",
   9.5, MUTED, italic=True)
C.notes_slide.notes_text_frame.text = (
    "Four deciders, from most transparent to most learned.\n"
    "1. Rule head: readable rules over the numbers; good for explaining, weak on accuracy (0.38).\n"
    "2. Gemma adjudicator: important design choice — it sees only the structured evidence, never the raw post, so emotive writing cannot sway it.\n"
    "3. Stacker: a multinomial logistic regression that learns how much to trust each signal; it is the best calibrated (ECE 0.055).\n"
    "4. Laya: a small decision model we fine-tuned. First supervised (imitate gold labels plus Gemma's soft labels), then reinforcement learning: "
    "the reward is a proper scoring rule, meaning the model gets the most reward only when its stated probability matches reality. "
    "We also penalise answering INCONCLUSIVE when the evidence is strong.\n"
    "Outcome: Laya passed our adoption gate, so it takes over deciding; Gemma remains the reader (claims, spans, stance).")

# ---------- new slide D: abstain ----------
D = new_slide()
frame(D, "07 / KNOWING WHEN NOT TO ANSWER",
      "The system abstains instead of guessing",
      "A wrong confident verdict is worse than “inconclusive”. Three methods decide whether to answer, and who should answer.", 9)
ab = [
    ("green", "Calibration", "Is “80% sure” really right 80% of the time?",
     "Rescales the model's confidence on held-out labelled claims (temperature scaling) so probabilities can be trusted.",
     "model probabilities + held-out labelled claims", "trustworthy probabilities"),
    ("blue", "Conformal prediction", "Which labels can we not rule out?",
     "Keeps every label that cannot be ruled out at a chosen error rate (e.g. 10%). One label left → answer; several → INCONCLUSIVE.",
     "calibrated probabilities + chosen error rate", "a label set with a coverage guarantee"),
    ("orange", "Routing bandit (RL)", "Who should decide this claim?",
     "Learns by trial and reward (accuracy minus cost) when fast Laya is enough, and when to escalate to Gemma or a human analyst.",
     "difficulty signals + past outcomes", "Laya / Gemma / human analyst"),
]
for k, (c, name, q, does, inp, out) in enumerate(ab):
    x, w, y, h = 0.65 + k * 4.1, 3.75, 1.8, 2.8
    dark, light = COL[c]
    clone(D, "card", x, y, w, h); clone(D, "bar", x, y, w, .09, dark)
    tb(D, x + .14, y + .17, w - .28, .32, name, 14, NAVY, bold=True)
    tb(D, x + .14, y + .52, w - .28, .3, q, 10.5, dark, bold=True, italic=True)
    tb(D, x + .14, y + .9, w - .28, .95, does, 10.5, BODY)
    tb(D, x + .14, y + 1.62, w - .28, .45, [[("INPUT  ", {"bold": True, "color": dark}), (inp, {})]], 10, BODY)
    clone(D, "pill", x + .1, y + h - .6, w - .2, .46, light)
    tb(D, x + .16, y + h - .58, w - .32, .42, [[("OUTPUT  ", {}), (out, {})]], 10, dark, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    if k < 2:
        tb(D, x + w + .05, y + 1.3, .3, .45, "→", 19, MUTED, align=PP_ALIGN.CENTER)
tb(D, 0.65, 4.95, 12, 0.3, "HOW THE ABSTAIN RULE READS", 10, COL["green"][0], bold=True)
chips = [("green", [("Label set {FAKE}", {"bold": True}), ("  →  answer FAKE", {})]),
         ("orange", [("Set {FAKE, MISLEADING}", {"bold": True}), ("  →  INCONCLUSIVE, send to analyst", {})]),
         ("blue", [("High uncertainty or < 1 independent source", {"bold": True}), ("  →  INCONCLUSIVE", {})])]
for k, (c, runs) in enumerate(chips):
    x, w = 0.65 + k * 4.1, 3.75
    clone(D, "pill", x, 5.3, w, .75, COL[c][1])
    tb(D, x + .15, 5.33, w - .3, .69, [runs], 10.5, COL[c][0], anchor=MSO_ANCHOR.MIDDLE)
tb(D, 0.65, 6.3, 12.05, 0.45, "Trade-off: answering fewer claims makes the answers more accurate. At the strictest setting the system answers 26% of claims "
   "with macro-F1 0.73; the error rate is a policy choice for the analysts.", 9.5, MUTED, italic=True)
D.notes_slide.notes_text_frame.text = (
    "This slide is about honesty. Calibration makes the probabilities mean what they say. "
    "Conformal prediction then turns probabilities into a set of labels we cannot rule out at a chosen error rate; "
    "if more than one label survives, the system says INCONCLUSIVE rather than guessing. This comes with a statistical coverage guarantee.\n"
    "The routing bandit is the second RL component: a contextual bandit (Thompson sampling / LinUCB) that learns, from outcomes, "
    "when the cheap model is enough and when to pay for Gemma or a human analyst.\n"
    "The error rate (alpha) is a policy knob: stricter = fewer but more accurate answers.")

# ---------- new slide E: worked example ----------
E = new_slide()
frame(E, "09 / WORKED EXAMPLE",
      "One post, end to end",
      "Illustrative example built to show the flow; the numbers are indicative, not a real case.", 11)
P, G, O, R, BL = COL["purple"][0], COL["green"][0], COL["orange"][0], COL["red"][0], COL["blue"][0]
# column 1: the post
clone(E, "card", 0.65, 1.8, 3.7, 4.95); clone(E, "bar", 0.65, 1.8, 3.7, .09, P)
tb(E, 0.79, 1.97, 3.4, .3, "1  THE POST", 12, NAVY, bold=True)
tb(E, 0.79, 2.4, 3.4, 1.7, [[("“", {}), ("Shocking!", {"color": P, "bold": True}),
                            (" Video shows Army trucks carrying ballot boxes in Patna last night. Media is hiding it. ", {}),
                            ("Share before they delete it!", {"color": P, "bold": True}), ("”", {})]], 12, NAVY, italic=True)
tb(E, 0.79, 3.75, 3.4, .28, "PROPAGANDA SPANS FOUND", 9.5, P, bold=True)
tb(E, 0.79, 4.05, 3.4, .8, ["“Shocking!” → Loaded language", "“Share before they delete it!” → Appeal to fear"], 10, BODY, space_after=3)
clone(E, "pill", 0.75, 4.95, 3.5, .6, COL["purple"][1])
tb(E, 0.82, 4.97, 3.36, .56, "P(propaganda) = 0.70, one-sided", 10.5, P, bold=True, anchor=MSO_ANCHOR.MIDDLE)
tb(E, 4.38, 3.9, .3, .45, "→", 19, MUTED, align=PP_ALIGN.CENTER)
# column 2: claims
clone(E, "card", 4.7, 1.8, 4.25, 4.95); clone(E, "bar", 4.7, 1.8, 4.25, .09, G)
tb(E, 4.84, 1.97, 3.95, .3, "2  CLAIMS AND EVIDENCE", 12, NAVY, bold=True)
tb(E, 4.84, 2.38, 3.95, .5, [[("Claim A  ", {"bold": True, "color": NAVY}), ("Army trucks carried ballot boxes in Patna last night", {})]], 10.5, BODY)
tb(E, 4.84, 2.9, 3.95, 1.25, ["• Fact-check match: same video, 2019 flood relief in Assam → context mismatch",
                             "• Timeline check: video seen 5 years earlier",
                             "• 3 copies of the debunk → counted as 1 source"], 9.5, BODY, space_after=2)
clone(E, "pill", 4.8, 4.2, 4.05, .42, COL["orange"][1])
tb(E, 4.86, 4.21, 3.93, .4, "→ MISLEADING   P(fake or misleading) = 0.78", 10, O, bold=True, anchor=MSO_ANCHOR.MIDDLE)
tb(E, 4.84, 4.85, 3.95, .5, [[("Claim B  ", {"bold": True, "color": NAVY}), ("Media is hiding it", {})]], 10.5, BODY)
tb(E, 4.84, 5.2, 3.95, .55, "• Opinion, low check-worthiness, no evidence either way", 9.5, BODY)
clone(E, "pill", 4.8, 6.08, 4.05, .42, COL["blue"][1])
tb(E, 4.86, 6.09, 3.93, .4, "→ INCONCLUSIVE (not used for the verdict)", 10, BL, bold=True, anchor=MSO_ANCHOR.MIDDLE)
tb(E, 8.98, 3.9, .3, .45, "→", 19, MUTED, align=PP_ALIGN.CENTER)
# column 3: rollup
clone(E, "card", 9.3, 1.8, 3.4, 4.95); clone(E, "bar", 9.3, 1.8, 3.4, .09, O)
tb(E, 9.44, 1.97, 3.1, .3, "3  POST VERDICT", 12, NAVY, bold=True)
rules = [("1  Any key claim FAKE → FAKE", False), ("2  Any claim false / out of context → MISLEADING", True),
         ("3  Strong propaganda → PROPAGANDA", False), ("4  All claims supported → LEGIT", False),
         ("5  Otherwise → INCONCLUSIVE", False)]
for k, (r, hit) in enumerate(rules):
    y = 2.38 + k * 0.5
    if hit:
        clone(E, "pill", 9.4, y - .03, 3.2, .46, COL["orange"][1])
    tb(E, 9.48, y, 3.05, .42, r, 9.5, O if hit else MUTED, bold=hit, anchor=MSO_ANCHOR.MIDDLE)
tb(E, 9.44, 4.95, 3.1, .45, "MISLEADING", 22, O, bold=True)
tb(E, 9.44, 5.45, 3.1, 1.2, ["+ secondary tag: PROPAGANDA", "+ flag: plausibly fake", "+ evidence trail for the analyst"], 10, BODY, space_after=2)
E.notes_slide.notes_text_frame.text = (
    "Walk through one post left to right.\n"
    "Step 1: Gemma tags two propaganda phrases; the propaganda score is 0.70. This alone does not make it fake.\n"
    "Step 2: two claims are extracted. Claim A matches an existing fact-check: the same video, but from 2019 Assam floods. "
    "The stance is 'context mismatch', the timeline check fires, and three copies of the debunk are collapsed into one independent source. "
    "Result: MISLEADING with P(fake or misleading) 0.78 — the 'plausibly fake' zone. Claim B is opinion and is not check-worthy.\n"
    "Step 3: the post rule ladder is applied in order; rule 2 fires, so the post is MISLEADING, with PROPAGANDA kept as a secondary tag.\n"
    "Stress that this is an illustrative example, not a real case.")

# ---------- new slide F: results ----------
F = new_slide()
frame(F, "10 / RESULTS SO FAR",
      "Maths + LLM beats the LLM alone",
      "Macro-F1 on 945 held-out test claims (LIAR, X-Fact, BOOM, PolitiFact, PTC). Macro-F1 averages accuracy over the 5 classes; 1.0 is perfect.", 12)
cd = CategoryChartData()
cats = ["Laya (trained decider)", "Stacker (maths + Gemma)", "Rule head (maths only)", "Gemma on structured evidence", "Gemma on raw text"]
vals = [0.61, 0.55, 0.38, 0.38, 0.25]
cd.categories = cats; cd.add_series("Macro-F1", vals)
gf = F.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(0.65), Inches(1.8), Inches(7.3), Inches(4.6), cd)
ch = gf.chart
ch.has_legend = False; ch.has_title = True
ch.chart_title.text_frame.text = "Macro-F1 by decider (higher is better)"
tp = ch.chart_title.text_frame.paragraphs[0].runs[0].font
tp.size = Pt(12); tp.bold = True; tp.name = FONT; tp.color.rgb = RGBColor.from_string(NAVY)
pl = ch.plots[0]; pl.gap_width = 60; pl.has_data_labels = True
dl = pl.data_labels; dl.number_format = "0.00"; dl.number_format_is_linked = False
dl.position = XL_LABEL_POSITION.OUTSIDE_END; dl.font.size = Pt(11); dl.font.bold = True; dl.font.name = FONT
dl.font.color.rgb = RGBColor.from_string(NAVY)
ser = pl.series[0]
for i, v in enumerate(vals):
    pt = ser.points[i]; pt.format.fill.solid()
    pt.format.fill.fore_color.rgb = RGBColor.from_string(G if i == 0 else ("457CBE" if i == 1 else "A9B4C2"))
va = ch.value_axis; va.maximum_scale = 0.8; va.minimum_scale = 0; va.has_major_gridlines = True
va.major_gridlines.format.line.color.rgb = RGBColor.from_string("E3E8EF")
va.tick_labels.font.size = Pt(10); va.tick_labels.font.name = FONT; va.tick_labels.font.color.rgb = RGBColor.from_string(MUTED)
va.format.line.fill.background()
ca = ch.category_axis; ca.tick_labels.font.size = Pt(10.5); ca.tick_labels.font.name = FONT
ca.tick_labels.font.color.rgb = RGBColor.from_string(BODY); ca.reverse_order = True
ca.reverse_order = False
ca.format.line.color.rgb = RGBColor.from_string("C9D1DC")
# stat callouts
stats = [("0.73", G, "macro-F1 when the system may abstain on unclear claims (it answers 26%)"),
         ("0.055", BL, "calibration error of the stacker: its confidence can be trusted"),
         ("3.4 GB", O, "GPU memory to train and run Laya, next to the shared LLM")]
for k, (num, c, lab) in enumerate(stats):
    y = 1.8 + k * 1.05
    tb(F, 8.3, y, 1.75, .7, num, 30, c, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    tb(F, 10.05, y + .02, 2.65, .7, lab, 10, BODY, anchor=MSO_ANCHOR.MIDDLE)
clone(F, "card", 8.3, 4.95, 4.4, 1.6, "FCF4E0")
tb(F, 8.45, 5.02, 4.1, .28, "HONEST CAVEATS", 10, O, bold=True)
tb(F, 8.45, 5.32, 4.1, 1.45, ["• PROPAGANDA is only tested on long news articles; on short social posts it over-fires.",
                             "• MISLEADING is the hardest class (recall 0.14).",
                             "• Fact-checks about a similar but different event can be counted as refutation."],
   9.5, BODY, space_after=2)
F.notes_slide.notes_text_frame.text = (
    "Main result: the hybrid wins. Gemma reading the raw post is the weakest decider (0.25). Giving Gemma the structured evidence helps (0.38). "
    "Adding the maths features and a learned stacker reaches 0.55 with excellent calibration, and the Laya model we trained reaches 0.61.\n"
    "When the system is allowed to abstain, the claims it does answer reach 0.73.\n"
    "Be upfront about the caveats: the PROPAGANDA class is perfect on the test set only because all propaganda examples are long articles — "
    "a format shortcut, not real skill; MISLEADING is hard; and evidence about look-alike events is the main error source, with a fix planned (a 'same claim' check).")

# ---------- fixes on existing slides ----------
def set_runs(shape, size=None, color=None, bold=None, font=FONT):
    for p in shape.text_frame.paragraphs:
        for r in p.runs:
            if font: r.font.name = font
            if size: r.font.size = Pt(size)
            if color: r.font.color.rgb = RGBColor.from_string(color)
            if bold is not None: r.font.bold = bold


def set_text(shape, text):
    rs = [r for p in shape.text_frame.paragraphs for r in p.runs]
    rs[0].text = text
    for r in rs[1:]:
        r.text = ""


def fix_header(slide, kicker, page):
    for sh in slide.shapes:
        if not sh.has_text_frame:
            continue
        t = sh.text_frame.text.strip()
        if sh.top < Inches(0.5) and sh.left < Inches(1) and t and t.upper() == t:
            set_text(sh, kicker); set_runs(sh, 10, COL["green"][0], True)
        elif re.fullmatch(r"\d+", t) and sh.left > Inches(11.5):
            set_text(sh, str(page)); set_runs(sh, 9, MUTED, False)


o1, o2, o3, o4, o5, o6, o7 = S
fix_header(o2, "01 / THE PURPOSE", 2)
fix_header(o3, "02 / HOW IT WORKS", 3)
fix_header(o5, "03 / METHODS: STEPS 1–4", 4)
fix_header(o7, "08 / VERDICT OUTPUT", 10)
fix_header(o6, "11 / WHY IT IS MORE THAN A CLASSIFIER", 13)
for sh in o3.shapes:  # stage names wrap mid-word at 11pt
    if sh.has_text_frame and sh.text_frame.text in ("Ingest", "Claims", "Propaganda", "Evidence", "Independence",
                                                    "Reliability", "Fusion", "Verdict"):
        set_runs(sh, 8.5 if sh.text_frame.text == "Independence" else 9.5)
for sh in o5.shapes:
    if sh.has_text_frame and sh.text_frame.text.startswith("CORE MATH, IN PLAIN ENGLISH"):
        set_text(sh, "CORE MATH, IN PLAIN ENGLISH: weigh support vs contradiction, discount copies, combine uncertainty "
                     "and consistency (next two slides), then calibrate.")
for sh in o7.shapes:  # summary text was navy on the navy band
    if sh.has_text_frame and sh.text_frame.text.startswith("Primary verdict"):
        set_runs(sh, color="FFFFFF")
for sh in o6.shapes:  # title wrapped onto the subtitle
    if sh.has_text_frame and sh.text_frame.text.startswith("AI reads."):
        set_runs(sh, 23)
bul = [sh for sh in o6.shapes if sh.has_text_frame and sh.text_frame.text.startswith("•")]
for sh in bul:  # third bullet ran into the pill below
    row = round((sh.top / 914400 - 3.35) / 0.65)
    sh.top = Inches(3.28 + row * 0.56); sh.height = Inches(0.52); set_runs(sh, 10.5)

notes = [
    (o1, "Title. One line pitch: from a claim to an evidence-backed, explainable assessment. The system combines an LLM (Gemma) that reads text with mathematical models that weigh evidence."),
    (o2, "Two separate questions. Truth: is the claim supported by evidence? Persuasion: how is the message trying to persuade? "
        "A true claim can be pushed with propaganda, and a calm message can be false, so we score both dimensions separately."),
    (o3, "The eight stages, L0 to L7. The LLM does reading (L1–L3); maths does weighing (L4–L6); L7 decides and may abstain. "
        "Emphasise de-duplication and abstention: they are what make it more than a classifier."),
    (o5, "Steps 1–4 in input → what we do → output form. Claims are quoted verbatim from the text so the LLM cannot invent them. "
        "Evidence comes from a knowledge base of ~32,000 already fact-checked claims; Gemma labels each match support / refute / context mismatch / neutral. "
        "Independence: ten reposts of one tweet count as one source (effective number of sources). Step 5, fusion, is unpacked on the next two slides."),
    (o4, "Spread over time. A hidden Markov model labels each period as baseline, emerging burst, coordinated spread or decay; event chains link episodes of the same narrative across platforms. "
        "Outputs are risk features (burst score, coordination, cross-platform spread, source diversity). Important: they indicate coordination risk and never by themselves prove a claim is fake."),
    (o7, "The five labels and what each means. FAKE beats PROPAGANDA as the main label; the other signals are kept as secondary tags. "
        "'Plausibly fake' = probability of fake or misleading between 0.5 and 0.8. The output always carries the evidence trail."),
    (o6, "Closing message: AI reads, evidence is checked, maths combines. The LLM alone was the weakest decider; the hybrid with calibration and abstention is both more accurate and more honest. "
        "Human review stays in the loop for high-impact, novel or disputed cases."),
]
for s, n in notes:
    s.notes_slide.notes_text_frame.text = n

# ---------- reorder ----------
order = [o1, o2, o3, o5, A, B, o4, C, D, o7, E, F, o6]
lst = prs.slides._sldIdLst
sid = [s.slide_id for s in order]
by_id = {int(el.get("id")): el for el in lst}
for el in list(lst):
    lst.remove(el)
for i in sid:
    lst.append(by_id[i])
prs.save(OUT)
print("saved", OUT, len(order), "slides")
