# No fertility refund: speculative-decoding acceptance across 21 languages (data and code)

High-fertility languages need more tokens for the same text. One hope is that speculative decoding gives some of that cost back: continuation tokens inside a word are easy to predict, so a drafter should accept more of them, and more tokens per verify step would offset the extra tokens.

This deposit tests that hope for two built-in self-drafters, on 200 FLORES-200 sentences in each of 21 languages:
- Qwen3.5-9B with its MTP head;
- Gemma 4 E4B with its assistant model.

The design, gates and kill rules are in `DESIGN.md`. They were stated before the runs, but that cannot be verified independently; see "Not claimed" below.

## Result

**Verdict: KILLED.** Every gate passed in both families, and both kill rules fired in both families, on the point estimates the rules use.

| | Qwen3.5-9B MTP | Gemma 4 E4B assistant |
|---|---|---|
| G1: English step-1 acceptance, correct wiring | 0.945 | 0.830 |
| Preflight agreement / G2 / G3 | 159/160 · 0.995 · 21 of 21 | 159/160 · 0.996 · 21 of 21 |
| F1: partial Spearman ρ(AL₃, T_L/T_en \| nll/char); fires at ≤ 0.10 | −0.033 (item CI [−0.129, −0.010]; language CI [−0.53, 0.52]) | −0.656 (item CI [−0.688, −0.621]; language CI [−0.80, −0.40]) |
| F2′: median acc1(cont) − acc1(word_start); fires at < 0.20 | 0.172 (item CI [0.164, 0.184]; language CI [0.152, 0.224]) | 0.049 (item CI [0.040, 0.066]; language CI [0.014, 0.137]) |
| AL at K = 3: English / other 20 languages | 3.16 / 1.89–2.81 | 2.93 / 1.26–2.19 |
| Refund fraction R_L(3), where defined (14 languages each) | −0.86 to −0.10 | −1.65 to −0.66 |

- **Acceptance does not rise with fertility in either family.** It falls clearly in Gemma, and is flat in Qwen.
- **The refund fraction is negative wherever it is defined.** That is because every other language has a lower AL than English, not because of a slope with fertility. High-fertility languages need more verify passes than their token count alone implies: a surcharge, not a refund.

**Why (exploratory, post hoc; `results/mechanism/`).** Each language's step-1 acceptance gap to English is split in two (a two-fold Oaxaca split). One part comes from a different mix of token tags, weighted by English acceptance rates. The other comes from lower acceptance within the same tag, weighted by the language's own tag shares.

- **The mix part is small and positive.** Every other language has more continuation tokens than English (continuation share 0.19–0.82 against 0.08 for Qwen; 0.16–0.87 against 0.06 for Gemma), and continuation tokens are easier to draft.
  - It averages +0.026 (Qwen) and +0.055 (Gemma) over the 20 languages.
  - It reflects the contrast with English rather than a fertility gradient: its Spearman correlation with T_L/T_en is 0.01 for Qwen and 0.36 for Gemma.
- **The within-tag part is much larger and negative:** −0.195 (Qwen) and −0.426 (Gemma), with the CI below zero in all 20 languages.
  - Acceptance of `word_start`, `cont` and `other` tokens is lower than English in every non-English language, in both families.
  - `special` tokens are accepted at a rate of 0.99 or more in most languages, though not all; the lowest is Gemma's Thai at 0.85.
  - English has no `frag` tokens, so `frag` uses the language's own rate. Its whole effect is therefore counted in the mix part.
- **The drafters' disadvantage persists at deeper drafts.** Acceptance of draft 2, given that draft 1 was accepted, is lower outside English: 0.46–0.74 against 0.85 for Qwen, and 0.43–0.66 against 0.83 for Gemma. The cumulative gap therefore grows with depth. The per-depth gap does not grow for Gemma.
- **One exception.** Qwen's Amharic tokens are 35% byte fragments, and under the `frag` convention above its mix part is negative (−0.040).

## Files

**Licences, by path:**

| path | contents | licence |
|---|---|---|
| `data/` | run records | CC BY-SA 4.0 |
| `results/` | the mechanism outputs | CC BY-SA 4.0 |
| `*.md`, `CITATION.cff`, `.zenodo.json` | documentation | CC BY-SA 4.0 |
| `code/`, `reproduce.sh` | code | MIT (`LICENSE-CODE`) |

**Data:**
- `data/run2/{qwen,gemma}/<lang>.jsonl`: one record per item. The fields are:
  - `y`, the output token ids, and `hyp`, the output text;
  - `ref`, the FLORES-200 reference;
  - `acc`, the acceptance matrix, five bits per anchor, decoded by `fert_lib.decode_acc`;
  - `iters`, verify steps per K, and `tags`, the token tags;
  - `valid`, plus the G2 counts and `ref_nll`.
- `data/run2/{qwen,gemma}/{g1,preflight,meta,DONE}.json`: gate results and run metadata.
- `data/run2/verdict.json` and `data/run2/REPORT.md`: the verdict as computed during the run.
- `data/run1_g1fail/qwen/`: the earlier run that stopped at the wiring gate (see `DESIGN.md`, A5).

**Code:**
- `code/probe/`: `fert_probe.py`, the runner; `fert_lib.py`, the emulation, tagging and statistics; `verdict.py`, the gates and kill rules.
- `code/mechanism/analyze_mechanism.py`, with outputs in `results/mechanism/`: the exploratory decomposition.
- `reproduce.sh`: recomputes the verdict and the decomposition from the data and checks them against the shipped files.

**Code versions.**
- `fert_lib.py` is byte-identical to the version that produced run 2.
- `fert_probe.py` and `verdict.py` are later versions. Their changes since run 2 add provenance and robustness checks: run fingerprints, recorded model revisions, sample-size and design checks, refusal to resume a different run, and an exact value-ceiling bound. Measurement is unchanged. The decision rules are unchanged except that G1 now also fails if any of its 20 items is unusable; run 2 used all 20.
- In the code, "preregistered" means the decision rules stated before the runs, in the sense qualified below.

## Reproduce

```
./reproduce.sh
```

- **Requirements:** Python ≥ 3.10 with numpy, scipy, sacrebleu, torch and transformers. `fert_lib.py` imports torch at module level; no GPU or model weights are needed.
- **What it checks:**
  - the checksums;
  - a fresh `verdict.py` run, written to an empty directory, against the shipped verdict: the verdict, its reason and every per-family field, recursively;
  - the G1 and preflight gates, recomputed from their stored counts;
  - the decomposition against `results/mechanism/mechanism.json`.
- **Runtime:** a few minutes.

Regenerating the data needs one A100 and the model revisions in `DESIGN.md`. `fert_probe.py --help` lists the arguments.

## Not claimed

- **The design's timing is not independently verifiable.** No public timestamp of the design predates the runs, and this repository's history begins after them. Read the decision rules as stated in advance by the author, not as externally preregistered.
- **One gate amendment came after data.** Amendment A5 changed the Qwen wiring ablation after the first run failed that gate, and it made the gate easier to pass.
- **The verdict rests on Qwen's point estimates.** KILLED needs one rule to fire in both families. Gemma's F1 and F2′ fire with their language-bootstrap intervals clear of the thresholds. Qwen's do not: the F1 interval is [−0.53, 0.52] and the F2′ interval is [0.152, 0.224]. Point estimates were the stated rule, but this is where the result is weakest.
- **Run 2's provenance is documented, not recorded in its files.** Run 2 predates two things the current code records: model revisions and a code/data fingerprint. `verdict.py` exempts exactly these two directories, identified by a hash of their files, and reports a note instead of a deviation. That hash proves the files are the shipped ones; it does not prove which code or model revisions produced them. `spec.target_rev` in `meta.json` is the pinned constant, not a recorded download.
- **Scope.**
  - Greedy decoding only.
  - Verify-pass counts, not wall-clock latency.
  - One translation prompt, and 200 sentences per language.
  - Two drafters, both built into their targets.
- **The mechanism split is exploratory.** It was chosen after seeing the verdict, and it depends on the five token tags, on English as the reference, and on the `frag` convention above.
- **No result about tokenizers in general.** The fertility axis is the one these two tokenizers happen to produce: T_L/T_en is 0.90–4.72 for Qwen and 1.03–2.28 for Gemma.

## Attribution and licence

- **FLORES-200.** The `ref` fields are FLORES-200 devtest sentences, unmodified. They come from NLLB Team et al., "No Language Left Behind: Scaling Human-Centered Machine Translation" (arXiv:2207.04672, 2022), and are licensed CC BY-SA 4.0.
- **Model outputs.** These were generated with Qwen3.5-9B and Gemma 4 E4B, both released under Apache-2.0, and are released here as part of the dataset under CC BY-SA 4.0.
- **Licence files.** `LICENSE` has the full CC BY-SA 4.0 text, which covers everything except the code. `LICENSE-CODE` is the MIT License, for `code/` and `reproduce.sh`.
