"""Tests for the FERT-REFUND probe (CPU, fp32, tiny random models + the real tokenizers).

Usage: python test_fert.py <cfg_root> [test_name ...]
  cfg_root holds qwen35/, gemma4/, gemma4a/ with the released config.json, tokenizer files and
  chat template (no weights).
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fert_lib as fl  # noqa: E402
import make_tiny as mt  # noqa: E402

CFG: Path = Path(".")
torch.set_num_threads(4)


def _cfg(name):
    return json.loads((CFG / name / "config.json").read_text())


def _tok(name):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(CFG / name))


def _qwen(vocab, seed=0, layers=4):
    qc = mt.tiny_qwen_config(_cfg("qwen35"), vocab=vocab, layers=layers)
    qc["text_config"]["eos_token_id"] = None
    m = mt.build_qwen(qc, seed=seed)
    m.config._attn_implementation = "sdpa"
    tc = m.config.text_config
    mtp = fl.QwenMTP(tc)
    sd = mt.random_mtp_state(qc["text_config"], seed=seed + 1)
    mtp.load_state_dict({k[4:]: v for k, v in sd.items()})
    return m, mtp.eval(), qc, sd


def _gemma(vocab, seed=0, window=6):
    gc = mt.tiny_gemma_config(_cfg("gemma4"), vocab=vocab, sliding_window=window)
    gc["text_config"]["eos_token_id"] = None
    gc["eos_token_id"] = None
    m = mt.build_gemma(gc, seed=seed)
    ac = mt.tiny_assistant_config(_cfg("gemma4a"), gc["text_config"], vocab=vocab)
    a = mt.build_assistant(ac, seed=seed + 2)
    return m, a


def _rand_ids(V, N, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, V, (N,), generator=g)


# ----------------------------------------------------------------------------- pure logic

def test_simulate_bruteforce():
    """simulate() against a token-by-token reference of greedy chain speculation."""
    rng = np.random.default_rng(0)
    for trial in range(3000):
        P = int(rng.integers(1, 6))
        n = int(rng.integers(1, 30))
        N = P + n
        T = max(n - 1, 0)
        acc = rng.random((T, fl.K_MAX)) < rng.uniform(0.1, 0.95)
        for a in range(T):  # drafts past the end cannot match
            t = P - 1 + a
            for j in range(1, fl.K_MAX + 1):
                if t + 1 + j > N - 1:
                    acc[a, j - 1] = False
        for K in range(1, fl.K_MAX + 1):
            # token-level: each verify step checks drafts against the realized text one by one
            produced, steps = 1, 0  # prefill emits ids[P]
            while produced < n:
                t = P - 2 + produced  # the target has produced ids[t+1]
                steps += 1
                new = 0
                for j in range(1, K + 1):
                    if P + produced + new > N - 1 or not acc[t - (P - 1), j - 1]:
                        break
                    new += 1
                produced += new + (1 if produced + new < n else 0)
            assert fl.simulate(acc, P, N, K) == steps, (trial, K)
            if n >= 2:
                assert (n - 1) / max(steps, 1) <= K + 1 + 1e-9


def test_compose_wiring():
    assert fl.compose_wiring("ok", "swap") == "swap"
    assert fl.compose_wiring("pos_t", "unscaled") == "pos_t+unscaled"
    for m in fl.FAMILIES["gemma"]["miswirings"]:
        for p in ("ok", "pos_t"):
            assert fl.compose_wiring(p, m) in fl.FAMILIES["gemma"]["wirings"]
    assert fl._flags("pos_t+swap") == {"pos_t", "swap"}


def test_acceptance_matrix_and_codec():
    ids = torch.tensor([5, 6, 7, 8, 9, 10, 11])
    P = 3  # anchors t = 2..4
    # draft j at anchor t predicts ids[t+1+j]; targets past ids[-1] never match
    drafts = torch.tensor([[9, 10, 0, 0, 0], [10, 0, 0, 0, 0], [11, 11, 11, 11, 11]])
    acc = fl.acceptance_matrix(drafts, ids, P)
    exp = np.array([[1, 1, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0]], bool)
    assert (acc == exp).all(), acc
    assert (fl.decode_acc(fl.encode_acc(acc)) == acc).all()


def test_rep4_and_partial_spearman():
    assert fl.rep4_fraction([1, 2, 3]) == 0.0
    assert fl.rep4_fraction(list(range(20))) == 0.0
    assert abs(fl.rep4_fraction([1, 2, 3, 4] * 5) - (1 - 4 / 17)) < 1e-12
    rng = np.random.default_rng(1)
    z = rng.normal(size=40)
    x = z + rng.normal(size=40)
    y = z + rng.normal(size=40)
    r = fl.spearman_partial(x, y, z)
    assert -0.5 < r < 0.5
    assert fl.spearman_partial(x, x, z) > 0.999


# ----------------------------------------------------------------------------- tokenizers and prompts

SAMPLES = {
    "eng_Latn": "The quick brown fox, jumping over 12 lazy dogs!",
    "hin_Deva": "भारत एक विशाल देश है। क्षत्रिय",
    "amh_Ethi": "ኢትዮጵያ ትልቅ አገር ናት።",
    "tha_Thai": "ประเทศไทยเป็นประเทศใหญ่",
    "jpn_Jpan": "日本は大きな国です。",
    "mya_Mymr": "မြန်မာနိုင်ငံသည်",
    "kat_Geor": "საქართველო დიდი ქვეყანაა",
    "arb_Arab": "مصر بلد كبير",
}


def test_token_bytes_and_tags_real():
    for name in ("qwen35", "gemma4"):
        tok = _tok(name)
        tb = fl.TokenBytes(tok)
        for lang, s in SAMPLES.items():
            ids = tok(" " + s, add_special_tokens=False)["input_ids"]
            data = b"".join(tb(i) for i in ids)
            assert data.decode("utf-8") == tok.decode(ids, clean_up_tokenization_spaces=False), (name, lang)
            tags = fl.tag_tokens(ids, tb)
            assert len(tags) == len(ids) and "special" not in tags
            # one word_start per word: count words independently
            words = 0
            prev = False
            for ch in data.decode("utf-8"):
                w = __import__("unicodedata").category(ch)[0] in "LMN" or (ch in fl.ZW and prev)
                words += w and not prev
                prev = w
            assert tags.count("word_start") <= words
            if lang == "eng_Latn":
                assert tags.count("word_start") == words, (name, tags)
        # special tokens are tagged special and contribute no bytes
        st = fl.stop_ids(tok, "qwen" if name == "qwen35" else "gemma")
        assert fl.tag_tokens(st, tb) == ["special"] * len(st)
        # a long word split into pieces: first piece word_start, rest cont/frag
        ids = tok(" antidisestablishmentarianism", add_special_tokens=False)["input_ids"]
        tags = fl.tag_tokens(ids, tb)
        assert len(ids) > 1 and tags[0] == "word_start" and set(tags[1:]) <= {"cont", "frag"}, tags


def test_frag_tag():
    class FakeTB:
        def __call__(self, i):
            return {0: "क".encode()[:2], 1: "क".encode()[2:], 2: b" ", 3: b"a"}[i]
    assert fl.tag_tokens([2, 0, 1, 3], FakeTB()) == ["other", "frag", "word_start", "cont"]
    # ZWNJ inside a word is a word character
    class TB2:
        def __call__(self, i):
            return {0: "क".encode(), 1: "‌".encode(), 2: "ष".encode()}[i]
    assert fl.tag_tokens([0, 1, 2], TB2()) == ["word_start", "cont", "cont"]


def test_prompts_real():
    flores = {"eng_Latn": ["Hello there."], "fra_Latn": ["Bonjour."], "amh_Ethi": ["ሰላም"]}
    fl.LANG_NAME.setdefault("amh_Ethi", "Amharic")
    m, ref = fl.make_prompt("amh_Ethi", 0, flores)
    assert m.startswith("Translate the following English text into Amharic.") and m.endswith("Hello there.")
    assert ref == "ሰላም"
    m, ref = fl.make_prompt("eng_Latn", 0, flores)
    assert "French text into English" in m and m.endswith("Bonjour.") and ref == "Hello there."
    tq, tg = _tok("qwen35"), _tok("gemma4")
    iq = fl.chat_prompt_ids(tq, m)
    assert tq.decode(iq).endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"), tq.decode(iq)
    ig = fl.chat_prompt_ids(tg, m)
    assert ig[0] == tg.bos_token_id and ig.count(tg.bos_token_id) == 1
    assert tg.decode(ig).endswith("<|turn>model\n"), tg.decode(ig)
    assert fl.stop_ids(tq, "qwen") == [248046, 248044]
    assert fl.stop_ids(tg, "gemma") == [1, 106, 50]


def test_flores_loader(tmp: Path):
    d = tmp / "flores"
    d.mkdir()
    for lang in ("eng_Latn", "fra_Latn", "amh_Ethi"):
        (d / f"{lang}.devtest").write_text("\n".join(f"{lang}-{i}" for i in range(1012)) + "\n", "utf-8")
    f = fl.load_flores(d, ["amh_Ethi"], 3)
    assert f["amh_Ethi"] == ["amh_Ethi-0", "amh_Ethi-1", "amh_Ethi-2"] and len(f["fra_Latn"]) == 3


# ----------------------------------------------------------------------------- Qwen MTP

def test_qwen_target_paths():
    m, _, _, _ = _qwen(64)
    ids = _rand_ids(64, 17, 3)
    tf = fl.qwen_target(m, ids)
    full = m(input_ids=ids[None]).logits[0]
    assert torch.equal(full.argmax(-1), tf["argmax"])
    assert torch.allclose(m.lm_head(tf["h"]), full, atol=1e-5)
    assert torch.allclose(m.model.language_model.norm(tf["h_pre"]), tf["h"], atol=1e-6)
    hs = fl.qwen_hidden(tf, "shift")  # off-by-one ablation: position p sees h_{p-1}
    assert torch.equal(hs[1:], tf["h"][:-1]) and torch.equal(hs[0], tf["h"][0])


def test_qwen_batched_vs_reference():
    m, mtp, _, _ = _qwen(64)
    for seed in range(3):
        ids = _rand_ids(64, 26, 10 + seed)
        P = 5 + seed
        tf = fl.qwen_target(m, ids)
        for wiring in fl.FAMILIES["qwen"]["wirings"]:
            d, _ = fl.item_drafts("qwen", m, mtp, ids, P, wiring, tf)
            h = fl.qwen_hidden(tf, wiring)
            for a in range(d.shape[0]):
                t = P - 1 + a
                ref = fl.qwen_reference_drafts(mtp, m, ids, h, t, fl.K_MAX,
                                               "swap" if wiring == "swap" else "ok")
                assert d[a].tolist() == ref, (seed, wiring, t, d[a].tolist(), ref)


def test_qwen_mtp_loader(tmp: Path):
    qc = mt.tiny_qwen_config(_cfg("qwen35"), vocab=64)
    qc["text_config"]["eos_token_id"] = None
    m = mt.build_qwen(qc)
    sd = mt.random_mtp_state(qc["text_config"], seed=7)
    mt.save_qwen(m, sd, tmp / "q", None)
    mtp = fl.load_qwen_mtp(tmp / "q", m.config.text_config, torch.float32, "cpu")
    got = {"mtp." + k: v for k, v in mtp.state_dict().items()}
    assert set(got) == set(sd) and all(torch.equal(got[k], sd[k]) for k in sd)
    # the saved target must still load without the mtp tensors interfering
    from transformers import Qwen3_5ForConditionalGeneration
    m2 = Qwen3_5ForConditionalGeneration.from_pretrained(tmp / "q", dtype=torch.float32)
    ids = _rand_ids(64, 9, 1)
    assert torch.allclose(m2(input_ids=ids[None]).logits, m(input_ids=ids[None]).logits, atol=1e-6)
    # a missing tensor must fail loudly
    idx = json.loads((tmp / "q/model.safetensors.index.json").read_text())
    del idx["weight_map"]["mtp.norm.weight"]
    (tmp / "q/model.safetensors.index.json").write_text(json.dumps(idx))
    try:
        fl.load_qwen_mtp(tmp / "q", m.config.text_config, torch.float32, "cpu")
    except ValueError:
        pass
    else:
        raise AssertionError("loader accepted an index that disagrees with the shards")


# ----------------------------------------------------------------------------- Gemma assistant

def test_gemma_batched_vs_hf():
    m, a = _gemma(64, window=6)
    for seed in range(3):
        ids = _rand_ids(64, 30, 20 + seed)
        P = 4 + seed
        tf = fl.gemma_target(m, ids)
        assert tf["kv"]["sliding_attention"][0].shape[2] == ids.shape[0]  # full-length sliding K/V
        for wiring in fl.FAMILIES["gemma"]["wirings"]:
            d, _ = fl.item_drafts("gemma", m, a, ids, P, wiring, tf)
            for k in range(d.shape[0]):
                t = P - 1 + k
                ref = fl.gemma_reference_drafts(a, m, ids, tf["h"], tf["kv"], t, fl.K_MAX, wiring)
                assert d[k].tolist() == ref, (seed, wiring, t, d[k].tolist(), ref)


def test_gemma_window_matters():
    """The sliding mask must bind: with a large window the drafts change for far anchors."""
    m, a = _gemma(64, window=6)
    ids = _rand_ids(64, 30, 5)
    tf = fl.gemma_target(m, ids)
    d6 = fl.gemma_drafts(a, m, ids, tf["h"], tf["kv"], 3)
    a.config.get_text_config().sliding_window = 10_000
    for layer in a.model.layers:
        if hasattr(layer.self_attn, "sliding_window") and layer.self_attn.sliding_window:
            layer.self_attn.sliding_window = 10_000
    dbig = fl.gemma_drafts(a, m, ids, tf["h"], tf["kv"], 3)
    assert not torch.equal(d6, dbig)


def test_gemma_hidden_is_post_norm():
    m, _ = _gemma(64)
    ids = _rand_ids(64, 12, 2)
    tf = fl.gemma_target(m, ids)
    lh = m.model.language_model(input_ids=ids[None]).last_hidden_state[0]
    assert torch.allclose(tf["h"], lh, atol=1e-6)


# ----------------------------------------------------------------------------- generation

def _greedy_manual(model, family, prompt, max_new, stops):
    ids = list(prompt)
    for _ in range(max_new):
        x = torch.tensor(ids)
        tf = fl.qwen_target(model, x) if family == "qwen" else fl.gemma_target(model, x)
        nxt = int(tf["argmax"][-1])
        ids.append(nxt)
        if nxt in stops:
            return ids[len(prompt):], False
    return ids[len(prompt):], True


def test_generation_batched_equals_unbatched():
    for family in ("qwen", "gemma"):
        m = _qwen(64)[0] if family == "qwen" else _gemma(64)[0]
        stops = [3, 7]
        prompts = [_rand_ids(64, L, 30 + L).tolist() for L in (5, 9, 13, 7)]
        prompts = [[x for x in p if x not in stops] or [1] for p in prompts]
        b = fl.generate_batch(m, prompts, 12, stops, pad_id=0, device="cpu")
        for p, (y, capped) in zip(prompts, b):
            y1, c1 = fl.generate_batch(m, [p], 12, stops, pad_id=0, device="cpu")[0]
            ym, cm = _greedy_manual(m, family, p, 12, stops)
            assert (y, capped) == (y1, c1) == (ym, cm), (family, y, y1, ym)
            # self-consistency (G2): teacher-forced argmax reproduces the realized text
            x = torch.tensor(p + y)
            tf = fl.qwen_target(m, x) if family == "qwen" else fl.gemma_target(m, x)
            assert tf["argmax"][len(p) - 1:-1].tolist() == y


# ----------------------------------------------------------------------------- end to end

def _spec_loop(family, m, dr, prompt, n, K):
    """Real greedy chain speculation, no emulation: drafts are fed back as drafted."""
    ids = list(prompt)
    P = len(prompt)
    x = torch.tensor(ids)
    tf = fl.qwen_target(m, x) if family == "qwen" else fl.gemma_target(m, x)
    ids.append(int(tf["argmax"][-1]))  # prefill
    steps = 0
    accepted_hist = []
    while len(ids) - P < n:
        t = len(ids) - 2
        x = torch.tensor(ids)
        tf = fl.qwen_target(m, x) if family == "qwen" else fl.gemma_target(m, x)
        if family == "qwen":
            d = fl.qwen_reference_drafts(dr, m, x, tf["h"], t, K, feed="drafted")
        else:
            d = fl.gemma_reference_drafts(dr, m, x, tf["h"], tf["kv"], t, K, feed="drafted")
        cand = torch.tensor(ids + d)
        tv = fl.qwen_target(m, cand) if family == "qwen" else fl.gemma_target(m, cand)
        steps += 1
        A = 0
        while A < K and d[A] == int(tv["argmax"][t + 1 + A]):
            A += 1
        new = d[:A] + [int(tv["argmax"][t + 1 + A])]
        new = new[: n - (len(ids) - P)]
        accepted_hist.append(A)
        ids += new
    return ids, steps, accepted_hist


def test_end_to_end_spec_loop():
    """Emulation + simulate() == a real speculative decoding loop, step for step."""
    for family in ("qwen", "gemma"):
        V = 4 if family == "qwen" else 8
        if family == "qwen":
            m, dr, _, _ = _qwen(V, seed=4)
        else:
            m, dr = _gemma(V, seed=4, window=5)
        big = 0
        for seed in range(6):
            prompt = _rand_ids(V, 4 + seed, 50 + seed).tolist()
            n = 22
            ref_ids, _, _ = _spec_loop(family, m, dr, prompt, n, 1)
            greedy, _ = _greedy_manual(m, family, prompt, n, stops=[])
            assert ref_ids[len(prompt):] == greedy, family  # lossless
            ids = torch.tensor(ref_ids)
            d, _ = fl.item_drafts(family, m, dr, ids, len(prompt))
            acc = fl.acceptance_matrix(d, ids, len(prompt))
            for K in range(1, fl.K_MAX + 1):
                out, steps, hist = _spec_loop(family, m, dr, prompt, n, K)
                assert out == ref_ids
                assert fl.simulate(acc, len(prompt), len(ref_ids), K) == steps, (family, seed, K, hist)
                big += sum(a >= 2 for a in hist)
        assert big > 0, f"{family}: no multi-token acceptance occurred, test is vacuous"


# ----------------------------------------------------------------------------- verdict

def _synthetic_family(d: Path, n_items: int, langs: dict, seed: int = 0):
    """langs: name -> (valid item indices, tokens-per-item scale)."""
    rng = np.random.default_rng(seed)
    d.mkdir(parents=True)
    (d / "DONE.json").write_text(json.dumps({"n_items": n_items, "langs": list(langs)}))
    (d / "g1.json").write_text(json.dumps({"pass": True, "rate": {}, "primary_wiring": "ok", "reasons": []}))
    (d / "preflight.json").write_text(json.dumps({"agree": 1, "total": 1, "rate": 1.0, "pass": True}))
    for L, (valid, scale) in langs.items():
        with open(d / f"{L}.jsonl", "w") as f:
            for i in range(n_items):
                n = int(rng.integers(10, 30) * scale)
                acc = rng.random((n - 1, fl.K_MAX)) < 0.6
                tags = [str(rng.choice(["word_start", "cont", "other"])) for _ in range(n - 1)] + ["special"]
                rec = {"i": i, "n": n, "valid": i in valid, "iters": {}, "acc": fl.encode_acc(acc),
                       "tags": tags, "ref_nll": float(rng.uniform(5, 50)), "ref_chars": 40, "hyp": "a b",
                       "ref": "a b", "g2_match": n, "g2_total": n}
                for K in range(1, fl.K_MAX + 1):
                    rec["iters"][str(K)] = fl.simulate(acc, 3, 3 + n, K)
                f.write(json.dumps(rec) + "\n")


def test_verdict_matched_inclusion_and_item_bootstrap(tmp: Path):
    import verdict as vd
    langs = {"eng_Latn": (set(range(30)), 1.0), "hin_Deva": (set(range(20)), 2.0),
             "fin_Latn": (set(range(20)), 1.4), "tur_Latn": (set(range(20)), 1.6),
             "rus_Cyrl": (set(range(20)), 1.2),
             "amh_Ethi": (set(range(20, 40)), 3.0)}  # 20 valid (enough alone), only 10 matched with English
    _synthetic_family(tmp / "qwen", 40, langs)
    out = vd.family_eval(tmp / "qwen", 40, 2, list(langs))
    assert out["valid_counts"]["amh_Ethi"] == 20 and out["matched_counts"]["amh_Ethi"] == 10
    assert "amh_Ethi" not in out["included"]
    assert out["valid_counts"]["eng_Latn"] == 30 and len(out["included"]) == 5
    # item arrays reproduce the point statistics exactly at the identity resample
    for L in out["included"]:
        rs = vd.read_jsonl(tmp / "qwen" / f"{L}.jsonl")
        en = vd.read_jsonl(tmp / "qwen" / "eng_Latn.jsonl")
        a = vd.item_arrays(rs, en, list(range(out["matched_counts"][L])))
        s = out["langs"][L]
        assert abs(a["nm1"].sum() / a["it3"].sum() - s["AL"]["3"]) < 1e-12
        assert abs(a["nL"].sum() / a["nE"].sum() - s["T_ratio"]) < 1e-12
        assert abs(a["nll"].sum() / a["chars"].sum() - s["ref_nll_char"]) < 1e-12
        ta = s["acc1_by_tag"]
        assert (a["ws_a"].sum(), a["ws_c"].sum(), a["ct_a"].sum(), a["ct_c"].sum()) == \
            (ta["word_start"][0], ta["word_start"][1], ta["cont"][0], ta["cont"][1])
    # the value ceiling uses English AL over each language's matched items, not all valid English items
    en = vd.read_jsonl(tmp / "qwen" / "eng_Latn.jsonl")
    for K in (1, 3):
        k = str(K)
        sub = [r for r in en if r["i"] < 20]
        al_sub = sum(r["n"] - 1 for r in sub) / sum(r["iters"][k] for r in sub)
        c = out["ceiling"][k]["hin_Deva"]
        assert abs(c["bound"] - out["langs"]["hin_Deva"]["T_ratio"] * al_sub / (K + 1)) < 1e-12
        assert out["langs"]["eng_Latn"]["n_items"] == 30 and out["langs"]["hin_Deva"]["n_items"] == 20
        # exact bound keeps the one prefill per matched item: FP_L >= m + sum(n_L - 1)/(K+1)
        hi_ = [r for r in vd.read_jsonl(tmp / "qwen" / "hin_Deva.jsonl") if r["i"] < 20]
        fp_en = sum(1 + r["iters"][k] for r in sub)
        exact = (20 + sum(r["n"] - 1 for r in hi_) / (K + 1)) / fp_en
        assert abs(c["bound_exact"] - exact) < 1e-12 and c["holds_exact"]
    lo, hi = out["F1"]["ci95"]
    assert out["complete"] and out["incomplete_langs"] == []
    # a run smaller than the evaluated sample must not count as complete
    (tmp / "qwen" / "DONE.json").write_text(json.dumps({"n_items": 20, "langs": list(langs)}))
    assert not vd.family_eval(tmp / "qwen", 40, 2, list(langs))["complete"]
    (tmp / "qwen" / "DONE.json").write_text(json.dumps({"n_items": 40, "langs": list(langs)}))
    assert vd.family_eval(tmp / "qwen", 40, 2, list(langs))["complete"]
    assert not vd.family_eval(tmp / "qwen", 41, 2, list(langs))["complete"]  # files hold 40 items, not 41
    # the required set is the preregistered one by default, not the subset the run recorded in DONE.json
    assert not vd.family_eval(tmp / "qwen", 40, 2)["complete"]
    assert -1 <= lo <= hi <= 1 and out["F1"]["ci95_language"] is not None


# ----------------------------------------------------------------------------- runner

def test_verdict_deviations(tmp: Path):
    """Any non-preregistered setting, in the verdict call or a family's recorded run, is listed (verdict PARTIAL)."""
    import verdict as vd
    import hashlib
    ok = argparse.Namespace(**vd.PREREG_VERDICT)
    here = Path(__file__).resolve().parent
    code = {n: hashlib.sha256((here / n).read_bytes()).hexdigest() for n in ("fert_lib.py", "fert_probe.py")}
    for f in fl.FAMILIES:
        (tmp / f).mkdir()
        (tmp / f / "fingerprint.json").write_text(json.dumps({"code": code, "flores": vd.PREREG_FLORES_MANIFEST}))
        (tmp / f / "meta.json").write_text(json.dumps({"args": dict(vd.PREREG_RUN, out="x", device="cuda"),
                                                       "gpu": "NVIDIA A100-SXM4-80GB",
                                                       "revisions": vd.PREREG_REVISIONS[f], **vd.PREREG_LIB}))
    assert vd.deviations(ok, tmp) == []
    meta = json.loads((tmp / "qwen" / "meta.json").read_text())
    for change, expect in [({"transformers": "5.20.0"}, "qwen: transformers '5.20.0' != '5.19.0'"),
                           ({"gpu": "NVIDIA H100"}, "qwen: ran on 'cuda' / 'NVIDIA H100', not one A100"),
                           ({"args": dict(meta["args"], device="cpu"), "gpu": None}, "qwen: ran on 'cpu'"),
                           ({"revisions": {"target": None, "drafter": None}}, "qwen: target revision None"),
                           ({"revisions": {"target": "f" * 40, "drafter": None}}, "qwen: target revision 'fff")]:
        (tmp / "qwen" / "meta.json").write_text(json.dumps({**meta, **change}))
        assert len(vd.deviations(ok, tmp)) == 1 and vd.deviations(ok, tmp)[0].startswith(expect)
    (tmp / "qwen" / "meta.json").write_text(json.dumps(meta))
    for change, expect in [({"code": dict(code, **{"fert_lib.py": "0" * 64})}, "qwen: fert_lib.py differs"),
                           ({"flores": "1" * 64}, "qwen: FLORES directory is not the canonical")]:
        (tmp / "qwen" / "fingerprint.json").write_text(json.dumps({"code": code, "flores": vd.PREREG_FLORES_MANIFEST,
                                                                   **change}))
        assert len(vd.deviations(ok, tmp)) == 1 and vd.deviations(ok, tmp)[0].startswith(expect)
    (tmp / "qwen" / "fingerprint.json").unlink()
    assert vd.deviations(ok, tmp) == ["qwen: no fingerprint.json"]
    legacy = {k: v for k, v in meta.items() if k != "revisions"}  # unattested pre-field run: a deviation
    (tmp / "qwen" / "meta.json").write_text(json.dumps(legacy))
    assert vd.deviations(ok, tmp) == ["qwen: model revisions not recorded", "qwen: no fingerprint.json"]
    import shutil
    import zipfile
    shutil.rmtree(tmp / "qwen")
    with zipfile.ZipFile(here.parent / "results_run2" / "fert_probe_results.zip") as z:
        z.extractall(tmp / "run2", [n for n in z.namelist() if n.startswith("qwen/")])
    shutil.move(tmp / "run2" / "qwen", tmp / "qwen")
    notes = []
    assert vd.deviations(ok, tmp, notes) == [] and len(notes) == 1  # the attested run 2 directory: a note only
    with open(tmp / "qwen" / "eng_Latn.jsonl", "a") as fh:  # attestation covers the measurements, not just meta
        fh.write("{}\n")
    assert vd.deviations(ok, tmp) == ["qwen: model revisions not recorded", "qwen: no fingerprint.json"]
    shutil.rmtree(tmp / "qwen")
    (tmp / "qwen").mkdir()
    (tmp / "qwen" / "meta.json").write_text(json.dumps(meta))
    (tmp / "qwen" / "fingerprint.json").write_text(json.dumps({"code": code, "flores": vd.PREREG_FLORES_MANIFEST}))
    assert len(vd.deviations(argparse.Namespace(**dict(vd.PREREG_VERDICT, n_items=100)), tmp)) == 1
    (tmp / "gemma" / "meta.json").write_text(json.dumps({"args": dict(vd.PREREG_RUN, max_new=300, no_cap_filter=True,
                                                                       device="cuda"),
                                                         "gpu": "NVIDIA A100-SXM4-80GB",
                                                         "revisions": vd.PREREG_REVISIONS["gemma"], **vd.PREREG_LIB}))
    assert sorted(vd.deviations(ok, tmp)) == ["gemma: --max-new 300 != 400", "gemma: --no-cap-filter True != False"]
    (tmp / "qwen" / "meta.json").unlink()
    assert "qwen: no meta.json" in vd.deviations(ok, tmp)


def test_g1_requires_full_sample(tmp: Path):
    """G1 fails when any of the first n English items is unusable, even if the rest would pass."""
    import fert_probe as fpm
    saved = (fpm.generate, fl.make_prompt, fl.chat_prompt_ids)
    try:
        fl.make_prompt = lambda lang, i, flores: ("m", "r")
        fl.chat_prompt_ids = lambda tok, msg: [1, 2]
        fpm.generate = lambda *a, **k: [([5], False)] * 3  # every item has < 2 tokens
        args = argparse.Namespace(g1_items=3, n_items=10, device="cpu")
        res, items = fpm.g1_stage("qwen", None, None, None, None, args, [0], 0, tmp)
    finally:
        fpm.generate, fl.make_prompt, fl.chat_prompt_ids = saved
    assert not res["pass"] and items == [] and res["n_used"] == 0
    assert any("only 0 of 3 G1 items usable" in r for r in res["reasons"])


def test_outdir_fingerprint(tmp: Path):
    """A reused --out with other code or settings is refused before any model loads; a finished run is a no-op."""
    import subprocess
    here = Path(__file__).resolve().parent
    out = tmp / "out"

    def run(*extra):
        cmd = [sys.executable, str(here / "fert_probe.py"), "--family", "qwen", "--target", str(tmp / "nomodel"),
               "--flores", str(tmp / "noflores"), "--out", str(out), "--langs", "eng_Latn,fra_Latn",
               "--n-items", "2", "--device", "cpu", *extra]
        return subprocess.run(cmd, capture_output=True, text=True, cwd=here).returncode

    assert run() != 0  # fresh dir: fingerprint written, then the missing model fails the load
    fp = json.loads((out / "qwen" / "fingerprint.json").read_text())
    assert set(fp["code"]) == {"fert_lib.py", "fert_probe.py"} and "out" not in fp["args"]
    import fert_probe as fpm
    m = tmp / "m"
    (m / ".cache").mkdir(parents=True)
    (m / "config.json").write_text("{}")
    (m / ".cache" / "w.metadata").write_text("rev-a")
    h0 = fpm.dir_manifest(m)
    (m / ".cache" / "w.metadata").write_text("rev-b")  # another revision into the same local_dir
    assert fpm.dir_manifest(m) != h0 and fpm.dir_manifest(None) is None
    (m / ".cache" / "huggingface" / "download").mkdir(parents=True)
    (m / ".cache" / "huggingface" / "download" / "config.json.metadata").write_text("a" * 40 + "\netag\n1.0\n")
    assert fpm.model_revision(m) == "a" * 40 and fpm.model_revision(tmp) is None
    (m / ".cache" / "huggingface" / "download" / "model.safetensors.metadata").write_text("c" * 40 + "\n")
    assert fpm.model_revision(m).startswith("mixed:")  # pinned config with weights from another revision
    (m / ".cache" / "huggingface" / "download" / "model.safetensors.metadata").unlink()
    snap = tmp / "hub" / "snapshots" / ("b" * 40)
    snap.mkdir(parents=True)
    assert fpm.model_revision(snap) == "b" * 40 and fpm.model_revision(None) is None
    import os
    big = m / "model.safetensors"
    big.write_bytes(b"a" * (2 << 20))
    os.utime(big, ns=(1, 10**18))
    h1 = fpm.dir_manifest(m)
    big.write_bytes(b"b" * (2 << 20))  # same path and size, different contents
    os.utime(big, ns=(1, 10**18 + 1))
    assert fpm.dir_manifest(m) != h1
    assert run("--max-new", "300") == 4  # changed setting
    for L in ("eng_Latn", "fra_Latn"):
        (out / "qwen" / f"{L}.jsonl").write_text('{"i": 0}\n{"i": 1}\n')
    (out / "qwen" / "DONE.json").write_text(json.dumps({"n_items": 2, "langs": ["eng_Latn", "fra_Latn"]}))
    assert run() == 0  # same fingerprint and complete: exits before loading the (missing) model
    (tmp / "noflores").mkdir()
    (tmp / "noflores" / "eng_Latn.dev").write_text("replaced text\n")
    assert run() == 4  # FLORES replaced in place at the same path
    (tmp / "noflores" / "eng_Latn.dev").unlink()
    (tmp / "noflores").rmdir()
    fp["code"]["fert_lib.py"] = "0" * 64
    (out / "qwen" / "fingerprint.json").write_text(json.dumps(fp))
    assert run() == 4  # changed probe code
    (out / "qwen" / "fingerprint.json").unlink()
    assert run() == 4  # old files without a fingerprint


def main():
    global CFG
    CFG = Path(sys.argv[1])
    only = set(sys.argv[2:])
    tests = [(k, v) for k, v in globals().items() if k.startswith("test_") and callable(v)]
    fails = 0
    for name, fn in tests:
        if only and name not in only:
            continue
        t0 = time.time()
        try:
            if fn.__code__.co_argcount:
                with tempfile.TemporaryDirectory() as d:
                    fn(Path(d))
            else:
                fn()
            print(f"PASS {name} ({time.time() - t0:.1f}s)", flush=True)
        except Exception:
            fails += 1
            print(f"FAIL {name}\n{traceback.format_exc()}", flush=True)
    print(f"{fails} failed" if fails else "ALL TESTS PASSED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
