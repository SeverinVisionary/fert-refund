# No fertility refund: speculative-decoding acceptance across 21 languages (data and code)

High-fertility languages need more tokens for the same text. One hope is that speculative decoding gives some of that cost back: continuation tokens inside a word are easy to predict, so a drafter should accept more of them, and more tokens per verify step would offset the extra tokens.

This deposit tests that hope for two built-in self-drafters, on 200 FLORES-200 sentences in each of 21 languages:
- Qwen3.5-9B with its MTP head;
- Gemma 4 E4B with its assistant model.

A follow-up measures the same question in a real serving engine (vLLM): wall-clock latency, sampling, and two trained EAGLE-3 heads. See "Serving in vLLM" below.

The paper that reports these results is `paper/fert-refund-paper.pdf` (v1.4). It carries its own AI-use statement.

**Cite as:** Yang, H. (2026). *No fertility refund: speculative-decoding acceptance across 21 languages (data and code)*, version 1.4. Zenodo. https://doi.org/10.5281/zenodo.23355847 (see `CITATION.cff`). All versions: https://doi.org/10.5281/zenodo.23331652.

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

- Acceptance does not rise with fertility in either family: It falls clearly in Gemma, and is flat in Qwen.
- The refund fraction is negative wherever it is defined: That is because every other language has a lower AL than English, not because of a slope with fertility. High-fertility languages need more verify passes than their token count alone implies: a surcharge, not a refund.

**Why (exploratory, post hoc; `results/mechanism/`).** Each language's step-1 acceptance gap to English is split in two (a two-fold Oaxaca split). One part comes from a different mix of token tags, weighted by English acceptance rates. The other comes from lower acceptance within the same tag, weighted by the language's own tag shares.

- The mix part is small and positive: Every other language has more continuation tokens than English (continuation share 0.19–0.82 against 0.08 for Qwen; 0.16–0.87 against 0.06 for Gemma), and continuation tokens are easier to draft.
  - It averages +0.026 (Qwen) and +0.055 (Gemma) over the 20 languages.
  - It reflects the contrast with English rather than a fertility gradient: its Spearman correlation with T_L/T_en is 0.01 for Qwen and 0.36 for Gemma.
- The within-tag part is much larger and negative: −0.195 (Qwen) and −0.426 (Gemma), with the CI below zero in all 20 languages.
  - Acceptance of `word_start`, `cont` and `other` tokens is lower than English in every non-English language, in both families.
  - `special` tokens are accepted at a rate of 0.99 or more in most languages, though not all; the lowest is Gemma's Thai at 0.85.
  - English has no `frag` tokens, so `frag` uses the language's own rate. Its whole effect is therefore counted in the mix part.
- The drafters' disadvantage persists at deeper drafts: Acceptance of draft 2, given that draft 1 was accepted, is lower outside English: 0.46–0.74 against 0.85 for Qwen, and 0.43–0.66 against 0.83 for Gemma. The cumulative gap therefore grows with depth. The per-depth gap does not grow for Gemma.
- One exception: Qwen's Amharic tokens are 35% byte fragments, and under the `frag` convention above its mix part is negative (−0.040).

## Serving in vLLM (descriptive follow-up)

The probe counts verify passes under greedy decoding. The follow-up measures the same drafters in vLLM 0.31.0 on one A100 40GB, adding two EAGLE-3 heads for Qwen3-8B: a general-purpose head (`eagle_rh`) and a multilingual one (`eagle_patel`). It has no kill rules. Its design and stated expectations are in `DESIGN.md` (serving extension), and its full tables are in `results/serve/SERVE_REPORT.md`.

- Calibration: vLLM's English AL at K = 3 is within 4% of the probe's emulated AL: 3.38 against 3.26 for Qwen, and 3.12 against 3.01 for Gemma. The EAGLE heads reach 2.46 (`eagle_rh`) and 1.54 (`eagle_patel`).
- Latency at batch 1, K = 3: The expectation held for both built-in drafters, Qwen and Gemma. For each of them, every non-English language gets a smaller speedup than English, with its interval below English's, and its latency premium over English grows under speculation. This does not hold for the multilingual EAGLE-3 head (below).

| speedup S_L(3) | eng | zho | rus | fin | hin | amh |
|---|---|---|---|---|---|---|
| Qwen3.5-9B MTP | 1.94 | 1.60 | 1.73 | 1.52 | 1.65 | 1.41 |
| Gemma 4 E4B assistant | 2.38 | 1.54 | 1.86 | 1.23 | 1.78 | 1.30 |
| Qwen3-8B + `eagle_rh` | 1.97 | 1.03 | 0.99 | 1.01 | 1.08 | 0.92 |
| Qwen3-8B + `eagle_patel` | 1.31 | 0.92 | 0.95 | 0.93 | 1.64 | 1.25 |

  For example, Gemma's Finnish takes 1.55× English's latency without speculation and 3.02× with it. The general EAGLE-3 head gives almost no speedup outside English (0.92–1.08). With the multilingual head, Hindi is the one language faster than English: 1.64 against 1.31.
- Sampling, all 21 languages, K = 3: The Spearman ρ of AL with T_L/T_en:

| ρ(AL, T_L/T_en) | T = 0 | T = 0.7 | T = 1.0 |
|---|---|---|---|
| Qwen3.5-9B MTP | −0.21 | −0.38 | −0.48 |
| Gemma 4 E4B assistant | −0.68 | −0.67 | −0.67 |
| Qwen3-8B + `eagle_rh` | −0.61 | −0.65 | −0.72 |
| Qwen3-8B + `eagle_patel` | 0.02 | −0.02 | −0.02 |

  - For both built-in drafters, no refund appears under sampling: ρ is negative at every temperature, and English has the highest AL of the 21 languages at each one.
  - The general EAGLE-3 head behaves like the built-in drafters.
  - The multilingual head shows no trend over all 21 languages (ρ ≈ 0), and it is weak in English: its English AL is 1.70, against 2.42 for the general head. Gujarati (2.26) and Hindi (1.87) exceed its English AL at T = 0. At T = 0 it beats the general head in only 4 of 21 languages, mainly Gujarati (+1.20) and Hindi (+0.65).
  - That ρ ≈ 0 depends on degenerate outputs (corrected in v1.1; v1.0 called this head flat in fertility). Its target, Qwen3-8B, produces capped or looping output for many sampled items in a few languages, for example 83 of 100 Amharic items at T = 0. Without every language whose sampled items are more than 10% invalid, the multilingual head's ρ is +0.42 at T = 0 (16 languages), +0.25 at T = 0.7 and +0.15 at T = 1.0. That threshold gives the largest values, though. Across exclusion thresholds of 5–50%, its ρ ranges from +0.08 to +0.42 at T = 0, and from −0.13 to +0.15 at T = 1.0 (`serve_extra.json`, `sensitivity_A2_threshold_sweep`). So the multilingual head is at most weakly positive, and its sign depends on the threshold. Qwen, Gemma and the general head stay negative at every threshold (Qwen and Gemma have 35 and 3 invalid items of 6,300). This check is post hoc.

**Post-hoc additions (v1.1)** are in `results/serve/serve_extra.json`, from `code/serve/analyze_serve_extra.py`. They cover three outputs the serving design specified (`DESIGN.md`, serving extension) but the analysis code that ran did not emit, plus the sensitivity check above:
- Spearman ρ of the K = 3 speedup with T_L/T_en over the six latency languages: Qwen −0.54, Gemma −0.71, `eagle_rh` −0.49, `eagle_patel` +0.49 (n = 6).
- vLLM's greedy calibration outputs equal the probe's Hugging Face greedy outputs on 20 of 20 English items, for both Qwen and Gemma. `serve_extra.json` stores the fraction (1.0); the 20 items are `data/serve/<arm>_K3_calib.jsonl` against `data/run2/<arm>/eng_Latn.jsonl`.
- The per-language AL difference between the two EAGLE-3 heads at T = 0.

## Files

**Licences, by path:**

| path | contents | licence |
|---|---|---|
| `data/` | run records | CC BY-SA 4.0 |
| `results/` | the mechanism and serving outputs | CC BY-SA 4.0 |
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
- `data/serve/`: the vLLM records, per arm and configuration, and the engine versions (`logs/vllm_version.txt`; `logs/env.txt` gives torch, transformers and whether CUDA was available).
  - `<arm>_K<k>_latency.jsonl`: one record per request, with latency, output tokens, drafts and accepted tokens.
  - `<arm>_K3_sampling.jsonl` and `<arm>_K3_calib.jsonl`: the sampling and calibration records.
  - `<arm>_K3_calib.json`: the gate results. `STATUS.json`, `RUN_INFO.json` (card and hours) and `models.json` (revisions) describe the run.

**Code:**
- `code/probe/`: `fert_probe.py`, the runner; `fert_lib.py`, the emulation, tagging and statistics; `verdict.py`, the gates and kill rules.
- `code/mechanism/analyze_mechanism.py`, with outputs in `results/mechanism/`: the exploratory decomposition.
- `code/serve/`: `fert_serve.py`, the vLLM runner, and `analyze_serve.py`, the serving analyses, with outputs in `results/serve/`. Both are byte-identical to the versions that produced `data/serve/`. `analyze_serve_extra.py` (v1.1) was written after the data, for the post-hoc additions.
- `paper/fert-refund-paper.pdf`: the paper (v1.4).
- `reproduce.sh`: recomputes the verdict and the decomposition from the data and checks them against the shipped files.

**Code versions.**
- `fert_lib.py` is byte-identical to the version that produced run 2.
- `fert_probe.py` and `verdict.py` are later versions. Their changes since run 2 add provenance and robustness checks: run fingerprints, recorded model revisions, sample-size and design checks, refusal to resume a different run, and an exact value-ceiling bound. Measurement is unchanged. The decision rules are unchanged except that G1 now also fails if any of its 20 items is unusable; run 2 used all 20.
- In the code, "preregistered" means the decision rules stated before the runs, in the sense qualified below.

## Reproduce

```
./reproduce.sh
```

- Requirements: Python ≥ 3.10 with numpy, scipy, sacrebleu, torch and transformers. `fert_lib.py` imports torch at module level; no GPU or model weights are needed.
- What it checks:
  - the checksums;
  - a fresh `verdict.py` run, written to an empty directory, against the shipped verdict: the verdict, its reason and every per-family field, recursively;
  - the G1 and preflight gates, recomputed from their stored counts;
  - the decomposition against `results/mechanism/mechanism.json`;
  - a fresh `analyze_serve.py` run on a copy of `data/serve/`, which must reproduce `results/serve/` byte for byte;
  - each serving calibration gate, recomputed from its raw records;
  - a fresh `analyze_serve_extra.py` run, which must reproduce `results/serve/serve_extra.json` byte for byte.
- Runtime: a few minutes.

Regenerating the data needs one A100 and the model revisions in `DESIGN.md`; `fert_probe.py --help` and `fert_serve.py --help` list the arguments. The serving runner also needs vLLM 0.31.0.

## Not claimed

- The design's timing is not independently verifiable: No public timestamp of the design predates the runs, and this repository's history begins after them. Read the decision rules as stated in advance by the author, not as externally preregistered.
- One gate amendment came after data: Amendment A5 changed the Qwen wiring ablation after the first run failed that gate, and it made the gate easier to pass.
- The verdict rests on Qwen's point estimates: KILLED needs one rule to fire in both families. Gemma's F1 and F2′ fire with their language-bootstrap intervals clear of the thresholds. Qwen's do not: the F1 interval is [−0.53, 0.52] and the F2′ interval is [0.152, 0.224]. Point estimates were the stated rule, but this is where the result is weakest.
- Run 2's provenance is documented, not recorded in its files: Run 2 predates two things the current code records: model revisions and a code/data fingerprint. `verdict.py` exempts exactly these two directories, identified by a hash of their files, and reports a note instead of a deviation. That hash proves the files are the shipped ones; it does not prove which code or model revisions produced them. `spec.target_rev` in `meta.json` is the pinned constant, not a recorded download.
- Scope:
  - The verdict uses greedy decoding and verify-pass counts. The serving follow-up adds sampling and wall-clock latency, but has no decision rules.
  - One translation prompt, and 200 sentences per language (100 for sampling, and 30 for latency in six languages).
  - Two built-in drafters, plus two EAGLE-3 heads for one third target. One engine version, one GPU type, and batch size 1 for latency.
- The serving numbers are one run:
  - A first serving run on the previous day finished, but its outputs were lost before they were saved. The shipped data are from the second run, of the same runner and analysis code.
  - The lost run's printed summary was close but not identical (it is not shipped, so this cannot be checked): its Qwen ρ at T = 0.7 was −0.35, against −0.38 here. Sampling under batching is not bit-reproducible across runs.
- Speculative greedy output is often not identical to plain greedy output in vLLM: The losslessness column of `SERVE_REPORT.md` gives the fraction of identical outputs. It is 0.97 for English in the built-in drafters, and mostly lower elsewhere: down to 0.50 for Qwen's Amharic at K = 1, and 1 of 3 on the EAGLE Amharic items. The speedups are latency ratios over the same valid inputs, not speedups at identical output, and output quality was not compared.
- Sampling AL includes invalid items: It is aggregated from vLLM's per-language counters, so the validity rule cannot be applied to it. This matters only for the Qwen3-8B EAGLE arms (419 and 420 of 6,300 items invalid, per arm); see the sensitivity check above.
- Some cells are thin: Only 3 of the 30 Amharic latency items are valid in both Qwen3-8B's baseline and its EAGLE runs, so the EAGLE Amharic latency figures rest on 3 items.
- The mechanism split is exploratory: It was chosen after seeing the verdict, and it depends on the five token tags, on English as the reference, and on the `frag` convention above.
- No result about tokenizers in general: The fertility axis is the one these two tokenizers happen to produce: T_L/T_en is 0.90–4.72 for Qwen and 1.03–2.28 for Gemma.

## Attribution and licence

- FLORES-200: The `ref` fields are FLORES-200 devtest sentences, unmodified. They come from NLLB Team et al., "No Language Left Behind: Scaling Human-Centered Machine Translation" (arXiv:2207.04672, 2022), and are licensed CC BY-SA 4.0.
- Model outputs: These were generated with Qwen3.5-9B and Gemma 4 E4B, both released under Apache-2.0, and are released here as part of the dataset under CC BY-SA 4.0.
- Licence files: `LICENSE` has the full CC BY-SA 4.0 text, which covers everything except the code. `LICENSE-CODE` is the MIT License, for `code/` and `reproduce.sh`.

## Versions

- 1.4: a corrected paper. It now defines the token ratio as measured on the models' own translations, gives the refund fraction, F2′ language set and validity rules, states that Qwen's result rests on point estimates, cites FLORES-200 and related work it missed, and fixes the figures (Figure 2 hid one negative bar). Data, code and results are unchanged.
- 1.3 (doi:10.5281/zenodo.23354898): adds the paper, `paper/fert-refund-paper.pdf`. Data, code and results are unchanged from 1.2.
- 1.2 (doi:10.5281/zenodo.23348249): adds the exclusion-threshold sweep, and states that the multilingual head's positive ρ depends on the threshold. v1.1 reported only the 10% threshold, which gives the largest value.
- 1.1 (doi:10.5281/zenodo.23344127): corrects the description of the multilingual EAGLE-3 head, which v1.0 called flat in fertility. Adds the post-hoc analyses above, with their code and a reproduction step. `DESIGN.md` now includes two reported-only items of the serving design that v1.0's restatement omitted. Data, and the code that produced them, are unchanged.
- 1.0: first release, doi:10.5281/zenodo.23331653.
