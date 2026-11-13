# Gemma tagger variants on PTC dev (26 articles)

| variant | SI P | SI R | SI F1 | span+tech P | R | F1 |
|---|---|---|---|---|---|---|
| chunk_v1 | 0.475 | 0.476 | 0.476 | 0.209 | 0.240 | 0.223 |
| chunk_v2 | 0.401 | 0.645 | 0.495 | 0.174 | 0.376 | 0.238 |

| technique | gold | chunk_v1 F1 (pred) | chunk_v2 F1 (pred) |
|---|---|---|---|
| Loaded_Language | 167 | 0.269 (263) | 0.266 (522) |
| Name_Calling,Labeling | 113 | 0.251 (110) | 0.295 (223) |
| Repetition | 26 | 0.049 (7) | 0.033 (17) |
| Exaggeration,Minimisation | 36 | 0.207 (39) | 0.154 (114) |
| Doubt | 53 | 0.118 (20) | 0.166 (111) |
| Appeal_to_fear-prejudice | 37 | 0.177 (27) | 0.243 (54) |
| Flag-Waving | 19 | 0.322 (5) | 0.272 (19) |
| Causal_Oversimplification | 23 | 0.109 (5) | 0.269 (38) |
| Slogans | 15 | 0.144 (10) | 0.291 (16) |
| Appeal_to_Authority | 9 | 0.000 (2) | 0.032 (14) |
| Black-and-White_Fallacy | 6 | 0.000 (12) | 0.065 (21) |
| Thought-terminating_Cliches | 8 | 0.000 (3) | 0.000 (6) |
| Whataboutism,Straw_Men,Red_Herring | 5 | 0.000 (11) | 0.000 (9) |
| Bandwagon,Reductio_ad_hitlerum | 7 | 0.000 (0) | 0.091 (12) |
