# Design of the probe

The design, gates and kill rules below were written down before the GPU runs, and frozen as constants in the versions of `fert_probe.py` and `verdict.py` used for those runs. The shipped versions are later; README.md lists what changed. No public timestamp of the design predates the runs, so read the decision rules as stated in advance by the author, not as externally preregistered. Five amendments were made; they are listed at the end, and one of them (A5) came after data.

## Question

With a built-in self-drafter, does speculative acceptance length per verify step rise with tokenizer fertility across languages? And is any extra acceptance concentrated in word-continuation tokens?

## Models

Both run in bf16 on one A100, with transformers 5.19.0 and SDPA attention.

| Family | Target @ revision | Drafter | Drafting semantics reproduced |
|---|---|---|---|
| `qwen` | `Qwen/Qwen3.5-9B` @ `c202236235762e1c871ad0ccb60c8ee5ba337b9a` | its built-in 1-layer MTP head, applied recursively | vLLM's Qwen3.5 MTP proposer |
| `gemma` | `google/gemma-4-E4B-it` @ `ee0ef6023621cff504d758262d4e04895a5af4a2` | `google/gemma-4-E4B-it-assistant` @ `8d0031ea8c2109e2b1e86bb9368a4539b537f80a` | Hugging Face's single-position multi-token candidate generator |

**Qwen.**
- The MTP consumes `fc([pre_fc_norm_embedding(embed(ids[p+1])); pre_fc_norm_hidden(h_p)])` at position p, where h_p is the target's post-final-norm hidden state.
- That goes through one full-attention decoder layer, then `norm`, then the shared `lm_head`.
- Chain step j ≥ 2 at anchor t feeds the drafted token and the MTP's normed output at position t+j−1. It attends to the step-1 entries at positions ≤ t and to the chain's own entries.

**Gemma.**
- Every step queries at the constant position t+1, the Hugging Face convention.
- Each step attends only to the K/V exported by target layers 22 (sliding) and 23 (full), at positions ≤ t. The sliding layers keep the last 513 positions.
- The input is `[scaled target embed(token); hidden]`. The hidden is the target's last hidden state at step 1 and the assistant's `post_projection` output afterwards.
- Drafts come from the centroid-masked logits.

## Data and generation

- **Data.** FLORES-200 devtest, sentences 0–199, in 21 languages: eng_Latn, amh_Ethi, mya_Mymr, khm_Khmr, guj_Gujr, kat_Geor, tel_Telu, ben_Beng, hin_Deva, tam_Taml, tha_Thai, fin_Latn, tur_Latn, swh_Latn, rus_Cyrl, arb_Arab, jpn_Jpan, zho_Hans, kor_Hang, fra_Latn and deu_Latn.
- **Source file.** `flores200_dataset.tar.gz`, sha256 `b8b0b76783024b85797e5cc75064eb83fc5288b41e9654dabc7be6ae944011f6`.
- **Prompt.** "Translate the following English text into {Language}. Output only the translation, nothing else.", followed by a blank line and the English sentence. English outputs are translated from fra_Latn with the matching prompt. Every prompt goes through the model's own chat template with thinking disabled.
- **Generation.** The target greedy-generates its own translation, in left-padded batches of 50, with at most 400 new tokens, up to and including its first stop token. That output y, of length n, is the text that acceptance is measured on.

## Measurement

The emulation is exact for greedy lossless speculation.

1. One teacher-forced target forward pass over prompt + y.
2. Batched drafter emulation at every anchor, to depth 5. Draft j at anchor t counts as accepted if it equals token t+1+j.
3. Simulated chain decoding for K = 1..5. FP = 1 + the number of verify steps, and AL = (n−1) / the number of verify steps.

**Per-language quantities.** All use items valid in both L and English.
- T_L/T_en = Σn_L / Σn_en.
- AL_L(K) = Σ(n−1) / Σ(verify steps).
- R_L(K) = 1 − ln(FP_L/FP_en) / ln(T_L/T_en), defined only when T_L/T_en ≥ 1.3.
- Step-1 acceptance by the tag of the predicted token:
  - `word_start`: the token completes the first code point of a word, where a word is a maximal run of L, M or N characters;
  - `cont`: the token completes only later code points;
  - `frag`: the token completes no code point;
  - `other` and `special` cover the rest.
- `ref_nll_char`: the target's teacher-forced NLL of the FLORES reference, in nats per character. This is the competence covariate.
- chrF, reported only.

**Validity.**
- An item is invalid if it hit the 400-token cap, or if its token 4-gram repeat fraction is above 0.3 (for outputs of at least 8 tokens).
- A language is included if at least 100 of its 200 items are valid in both L and English.

## Pre-run checks and gates

Any gate failure makes the verdict INVALID.

- **G1, wiring.** English step-1 acceptance on the first 20 items must be ≥ 0.50, and it must beat every deliberate mis-wiring by ≥ 0.05.
  - Qwen mis-wirings: swapped concatenation order, and an off-by-one hidden state.
  - Gemma mis-wirings: swapped order, and an unscaled embedding.
  - For Gemma, vLLM's position convention would replace Hugging Face's if it were ≥ 0.02 better. Each mis-wiring is scored under the convention selected.
- **Preflight.** Batched drafts must agree with a sequential reference (the unmodified Hugging Face layer, or the assistant call) on ≥ 97% of checked positions.
- **G2.** The target's teacher-forced argmax must equal the realized token at ≥ 98% of output positions.
- **G3.** At least 16 languages must be included, English among them.

## Kill rules

Point estimates decide.

- **F1.** The partial Spearman ρ between AL_L(3) and T_L/T_en, controlling for ref_nll_char across the included languages, is ≤ 0.10.
- **F2′.** Over the included whitespace-delimited languages with at least 200 tokens in both the `word_start` and `cont` tags, the median of acc1(cont) − acc1(word_start) is < 0.20. It needs at least 5 such languages.
- **Verdict.**
  - INVALID if any gate fails in either family.
  - Otherwise KILLED if F1 fires in both families, or F2′ fires in both families.
  - Otherwise SURVIVES.

**CIs.** Item-bootstrap 95% CIs (1000 resamples, seed 0) are reported, with a language bootstrap as a secondary interval. Neither decides anything.

## Amendments

- **A1 (before data).** Inclusion counts items valid in both L and English, rather than items valid in L. This makes inclusion stricter.
- **A2 (before data).** The paired item bootstrap became the primary CI. It is reported only.
- **A3 (before data).** Gemma's mis-wirings are scored under the selected position convention. This makes G1 stricter.
- **A4 (before data).** In the value ceiling, which is reported only, English AL uses L's matched items.
- **A5 (after data).** Run 1 stopped at Qwen's G1.
  - The original hidden-state ablation fed the pre-final-norm hidden state. The correct wiring beat it by only 0.012 (0.945 against 0.933), short of the 0.05 margin, because the MTP re-normalizes its input.
  - It was replaced by the off-by-one hidden state, and run 2 was made with that.
  - Only English G1 rates had been seen, and the amendment makes G1 easier to pass.
  - Run 1's gate files are in `data/run1_g1fail/`.

# Serving extension (vLLM)

This is a descriptive follow-up to the probe. It was written down before its runs, in the serving preregistration that the docstrings of `fert_serve.py` and `analyze_serve.py` cite as PREREG_SERVE.md. That file is not shipped; its design is restated here in full. As with the probe, no public timestamp predates the runs. It has no kill rules and does not reopen the verdict above. It answers three gaps in the probe: forward passes are not milliseconds, the decoding was greedy only, and both drafters were built into their targets.

## Engine and drafters

- **Engine.** vLLM 0.31.0, bf16, on one A100 40GB (`data/serve/RUN_INFO.json` records the card).
  - Settings: `generation_config="vllm"`, `watermarking=False`, `enable_prefix_caching=False`, `max_model_len=2048`, `gpu_memory_utilization=0.85`, engine `seed=0`, at most 400 new tokens.
  - Prompts, items and stop tokens are the probe's.

| Arm | Target @ revision | Drafter @ revision | `speculative_config` |
|---|---|---|---|
| `qwen` | `Qwen/Qwen3.5-9B` @ `c2022362` | built-in MTP head | `method: mtp` |
| `gemma` | `google/gemma-4-E4B-it` @ `ee0ef602` | `google/gemma-4-E4B-it-assistant` @ `8d0031ea` | `method: mtp` |
| `eagle_rh` | `Qwen/Qwen3-8B` @ `b968826d` | `RedHatAI/Qwen3-8B-speculator.eagle3` @ `08610ffa` (general-purpose) | `method: eagle3` |
| `eagle_patel` | `Qwen/Qwen3-8B` @ `b968826d` | `SwitchXDDD/multilingual-eagle3-qwen3-8b` @ `72072c42` (multilingual) | `method: eagle3` |

Full revisions are in `data/serve/models.json`. AL is `1 + accepted / drafts`, from vLLM's speculative-decoding counters; it is the same quantity as the probe's emulated AL.

## Stages

1. **Calibration gate**, per drafter, before its other runs: greedy, K = 3, English items 0–19.
   - `qwen` and `gemma` must come within 10% (relative) of the probe's emulated AL on the same items (3.2576 and 3.0124).
   - `eagle_*` have no probe reference and must reach AL ≥ 1.3.
   - A failed gate skips that drafter's other runs. The run plan enforces this: each drafter's first configuration runs the gate, and its exit code decides the rest. `fert_serve.py --stages` can omit the gate when called by hand.
2. **Latency**, batch size 1, after 3 warm-up requests.
   - Six languages: eng_Latn, zho_Hans, rus_Cyrl, fin_Latn, hin_Deva and amh_Ethi; items 0–29 each.
   - `qwen` and `gemma` at K ∈ {none, 1, 3, 5}. `eagle_*` at K = 3, against one shared Qwen3-8B no-speculation baseline.
3. **Sampling**, batched within a language, at K = 3: all 21 languages, items 0–99, T ∈ {0, 0.7, 1.0} (top-p 1, no top-k), seed 1000 + item.

Item validity is the probe's rule: under the 400-token cap, and a 4-gram repeat fraction ≤ 0.3.

## Analyses

- **A1, latency.** Over valid items matched across configurations:
  - speedup S_L(K) = Σ latency(none) / Σ latency(K);
  - premium P_L(K) = mean latency_L(K) / mean latency_en(K).
  - Stated expectation: S_L(3) < S_en(3) for every non-English language in `qwen` and `gemma`, and P_L(3) > P_L(none).
- **A2, sampling.** Per arm and T, the Spearman ρ of AL_L with T_L/T_en over the 21 languages. T_L is the arm's own output tokens at T = 0, over items valid in both L and English. Stated expectation: ρ ≤ 0.10 at every T for `qwen` and `gemma`.
- **A3, trained EAGLE-3 heads.** The same ρ at T = 0 for `eagle_rh` and `eagle_patel`, and the per-language AL difference between the two heads. No direction was stated.
- **Reported only.** Greedy losslessness: the fraction of latency items whose speculative output equals the no-speculation output.
- **CIs.** Item-bootstrap 95% intervals (2000 resamples, seed 0) for S_L(K) and the latency-stage AL. Sampling AL is a per-language aggregate of vLLM's counters and has no item-level interval.
