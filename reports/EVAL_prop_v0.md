# Propaganda tagger baseline v0 (Gemma, units_v1)

Scorer: `fnp/eval/ptc_scorer.py` (SemEval-2020 T11 style, fractional char overlap, 14 merged classes). Splits: `data/splits/ptc_v1.json`. Spans come from the stored run, so every split is scored only on the articles that run covers.

| split | docs scored / in split | SI P | SI R | SI F1 | span+technique P | R | F1 |
|---|---|---|---|---|---|---|---|
| train | 226/260 | 0.475 | 0.246 | 0.325 | 0.246 | 0.131 | 0.171 |
| dev | 22/26 | 0.545 | 0.227 | 0.320 | 0.295 | 0.122 | 0.173 |
| test | 75/85 | 0.435 | 0.206 | 0.280 | 0.233 | 0.112 | 0.152 |

## Per technique (test split)

| technique | gold spans | pred spans | P | R | F1 |
|---|---|---|---|---|---|
| Loaded_Language | 606 | 279 | 0.263 | 0.194 | 0.223 |
| Name_Calling,Labeling | 304 | 142 | 0.261 | 0.148 | 0.189 |
| Repetition | 223 | 6 | 0.000 | 0.000 | 0.000 |
| Exaggeration,Minimisation | 127 | 47 | 0.191 | 0.086 | 0.118 |
| Doubt | 122 | 37 | 0.175 | 0.045 | 0.071 |
| Appeal_to_fear-prejudice | 51 | 29 | 0.108 | 0.044 | 0.063 |
| Flag-Waving | 57 | 2 | 0.284 | 0.035 | 0.062 |
| Causal_Oversimplification | 53 | 14 | 0.214 | 0.052 | 0.084 |
| Slogans | 38 | 10 | 0.198 | 0.053 | 0.083 |
| Appeal_to_Authority | 38 | 0 | 0.000 | 0.000 | 0.000 |
| Black-and-White_Fallacy | 33 | 6 | 0.167 | 0.030 | 0.051 |
| Thought-terminating_Cliches | 12 | 3 | 0.000 | 0.000 | 0.000 |
| Whataboutism,Straw_Men,Red_Herring | 23 | 15 | 0.130 | 0.067 | 0.088 |
| Bandwagon,Reductio_ad_hitlerum | 14 | 5 | 0.200 | 0.054 | 0.085 |

Notes: `train` rows are not training data for Gemma (it is prompted, not trained); all three splits are valid evaluation slices. Test is the headline number.
