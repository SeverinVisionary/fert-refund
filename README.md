# No fertility refund: speculative-decoding acceptance across 21 languages (data and code)

High-fertility languages need more tokens for the same text. One hope is that speculative decoding gives some of that cost back: continuation tokens inside a word are easy to predict, so a drafter should accept more of them, and more tokens per verify step would offset the extra tokens.

This deposit tests that hope for two built-in self-drafters, on 200 FLORES-200 sentences in each of 21 languages:
- Qwen3.5-9B with its MTP head;
- Gemma 4 E4B with its assistant model.

The design, gates and kill rules are in `DESIGN.md`.

## Result

**Verdict: KILLED.** Every gate passed in both families, and both kill rules fired in both families.

| | Qwen3.5-9B MTP | Gemma 4 E4B assistant |
|---|---|---|
| G1: English step-1 acceptance, correct wiring | 0.945 | 0.830 |
| Preflight agreement / G2 / G3 | 159/160 · 0.995 · 21 of 21 | 159/160 · 0.996 · 21 of 21 |
| F1: partial Spearman ρ(AL₃, T_L/T_en \| nll/char); fires at ≤ 0.10 | −0.033 (item CI [−0.129, −0.010]) | −0.656 (item CI [−0.688, −0.621]) |
| F2′: median acc1(cont) − acc1(word_start); fires at < 0.20 | 0.172 (item CI [0.164, 0.184]) | 0.049 (item CI [0.040, 0.066]) |
| AL at K = 3: English / other 20 languages | 3.16 / 1.89–2.81 | 2.93 / 1.26–2.19 |
| Refund fraction R_L(3), where defined (14 languages each) | −0.86 to −0.10 | −1.65 to −0.66 |

The refund fraction is negative everywhere. High-fertility languages need more verify passes than their token count alone implies, not fewer: there is a surcharge, not a refund.

**Why (exploratory, post hoc; `results/mechanism/`).** The step-1 acceptance gap to English is split two ways: the share due to a different mix of token types, and the share due to lower acceptance within the same type.
- **The mix effect is small but positive.** High-fertility languages do have more continuation tokens, and those are easier to draft: +0.026 for Qwen, +0.055 for Gemma, averaged over 20 languages.
- **The within-type loss is much larger.** The drafters are worse on tokens of every type in every non-English language: −0.195 for Qwen and −0.426 for Gemma, with the CI below zero in all 20 languages.
- **The loss compounds with draft depth.**
- **One exception.** Qwen's Amharic tokens are 35% byte fragments, and there the mix effect is negative.

## Files

- `data/run2/{qwen,gemma}/<lang>.jsonl`: one record per item. The fields are:
  - `y`, the output token ids, and `hyp`, the output text;
  - `ref`, the FLORES reference;
  - `acc`, the acceptance matrix, five bits per anchor, decoded by `fert_lib.decode_acc`;
  - `iters`, verify steps per K, and `tags`, the token tags;
  - `valid`, plus the G2 counts and `ref_nll`.
- `data/run2/{qwen,gemma}/{g1,preflight,meta,DONE}.json`: gate results and run metadata.
- `data/run2/verdict.json` and `data/run2/REPORT.md`: the verdict as computed during the run.
- `data/run1_g1fail/qwen/`: the earlier run that stopped at the wiring gate (see `DESIGN.md`, A5).
- `code/probe/`:
  - `fert_probe.py`, the runner;
  - `fert_lib.py`, the emulation, tagging and statistics;
  - `verdict.py`, the gates and kill rules;
  - `test_fert.py` and `make_tiny.py`, the tests and the tiny random models they use.
- `code/mechanism/analyze_mechanism.py` and `results/mechanism/`: the exploratory decomposition.
- `reproduce.sh`: recomputes the verdict and the decomposition from the data and checks them against the shipped files.

## Reproduce

```
./reproduce.sh
```

- **Requirements:** Python ≥ 3.10 with numpy, scipy, sacrebleu, torch and transformers. `fert_lib.py` imports torch at module level; no GPU or model weights are needed.
- **What it checks:** the checksums, then that a fresh `verdict.py` run gives the shipped verdict and F1/F2′ statistics, and that the decomposition matches `results/mechanism/mechanism.json`.
- **Runtime:** a few minutes.

Regenerating the data needs one A100 and the pinned model revisions. `fert_probe.py --help` lists the arguments.

## Not claimed

- **Timing of the design is not externally verifiable.** It was fixed before the runs in a private repository. One amendment (A5, the Qwen wiring ablation) came after the first run's gate failure.
- **The Qwen F1 kill rests on its point estimate.** That was the stated rule. Qwen's language-bootstrap interval is wide ([−0.53, 0.52]). Gemma's F1, and F2′ in both families, do not depend on it.
- **Scope.**
  - Greedy decoding only.
  - Verify-pass counts, not wall-clock latency.
  - One translation prompt, and 200 sentences per language.
  - Two drafters, both built into their targets; trained external drafters such as EAGLE heads are not covered.
- **The mechanism split is exploratory.** It was chosen after seeing the verdict, and it depends on the four token tags and on English as the reference weights.
- **No result about tokenizers in general.** The fertility axis is the one these two tokenizers happen to produce: T_L/T_en is 0.90–4.72 for Qwen and 1.03–2.28 for Gemma.

## Licence

- Data and documentation: CC BY-SA 4.0. The `ref` fields are FLORES-200 text (CC BY-SA 4.0), and the model outputs come from Apache-2.0 models.
- Code: also available under the MIT License (`LICENSE`).
- See `LICENSE-DOCS`.
