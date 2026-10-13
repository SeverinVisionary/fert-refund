"""FERT-REFUND serving extension: vLLM calibration, latency and sampling arms (PREREG_SERVE.md).

Runs inside the vLLM 0.31.0 venv, one engine per (arm, K) per process:
    python fert_serve.py --arm qwen --k 3 --stages calib,latency,sampling --models models.json \
        --flores <devtest dir> --out <dir>
Outputs in --out: <arm>_K<k>_{calib,latency,sampling}.jsonl and <arm>_K<k>_calib.json.
Exit code 3 = calibration gate failed (the notebook then skips the arm's other configurations).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parent.parent / "probe"))  # fert_lib in the repo
import fert_lib as fl  # noqa: E402

QWEN3_STOPS = ["<|im_end|>", "<|endoftext|>"]
ARMS = {
    "qwen": {"target": "qwen35", "drafter": None, "method": "mtp", "stops": fl.FAMILIES["qwen"]["stop_tokens"],
             "ref_al3": 3.2576, "mm": {"image": 0, "video": 0}},
    "gemma": {"target": "gemma4", "drafter": "gemma4a", "method": "mtp", "stops": fl.FAMILIES["gemma"]["stop_tokens"],
              "ref_al3": 3.0124, "mm": {"image": 0, "audio": 0}},
    "qwen3_8b": {"target": "qwen3_8b", "drafter": None, "method": None, "stops": QWEN3_STOPS, "mm": None},
    "eagle_rh": {"target": "qwen3_8b", "drafter": "eagle_rh", "method": "eagle3", "stops": QWEN3_STOPS, "mm": None},
    "eagle_patel": {"target": "qwen3_8b", "drafter": "eagle_patel", "method": "eagle3", "stops": QWEN3_STOPS,
                    "mm": None},
}
REPOS = {  # pinned revisions (PREREG_SERVE.md)
    "qwen35": ("Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a"),
    "gemma4": ("google/gemma-4-E4B-it", "ee0ef6023621cff504d758262d4e04895a5af4a2"),
    "gemma4a": ("google/gemma-4-E4B-it-assistant", "8d0031ea8c2109e2b1e86bb9368a4539b537f80a"),
    "qwen3_8b": ("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218"),
    "eagle_rh": ("RedHatAI/Qwen3-8B-speculator.eagle3", "08610ffa01dd9f16731fe8f627b85905b6aa51c4"),
    "eagle_patel": ("SwitchXDDD/multilingual-eagle3-qwen3-8b", "72072c4207ab5ee0aa0d65660107a7d52e46c80a"),
}
LAT_LANGS = ["eng_Latn", "zho_Hans", "rus_Cyrl", "fin_Latn", "hin_Deva", "amh_Ethi"]
CALIB_ITEMS, LAT_ITEMS, SAMPLE_ITEMS, WARMUP = 20, 30, 100, 3
TEMPS = [0.0, 0.7, 1.0]
MAX_NEW, MAX_LEN = 400, 2048
CALIB_TOL, EAGLE_MIN_AL = 0.10, 1.3


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def make_engine(arm: str, k: int, paths: dict):
    """The vLLM engine for one configuration (tests replace this with a fake)."""
    from vllm import LLM
    a = ARMS[arm]
    spec = None
    if k > 0:
        spec = {"method": a["method"], "num_speculative_tokens": k}
        if a["drafter"]:
            spec["model"] = paths[a["drafter"]]
    kw = dict(model=paths[a["target"]], dtype="bfloat16", max_model_len=MAX_LEN, gpu_memory_utilization=0.85,
              seed=0, disable_log_stats=False, enable_prefix_caching=False, generation_config="vllm",
              speculative_config=spec)
    if a["mm"]:  # text only: no vision/audio encoders profiled
        kw["limit_mm_per_prompt"] = a["mm"]
    return LLM(**kw)


def sampling_params(t: float, seed: int, stops: list[int]):
    from vllm import SamplingParams
    return SamplingParams(temperature=t, top_p=1.0, max_tokens=MAX_NEW, seed=seed, stop_token_ids=stops,
                          watermarking=False)


def spec_counts(llm) -> dict:
    c = {"drafts": 0, "draft_tokens": 0, "accepted": 0}
    names = {"vllm:spec_decode_num_drafts": "drafts", "vllm:spec_decode_num_draft_tokens": "draft_tokens",
             "vllm:spec_decode_num_accepted_tokens": "accepted"}
    for m in llm.get_metrics():
        if m.name in names:
            c[names[m.name]] += int(m.value)
    return c


def delta(a: dict, b: dict) -> dict:
    return {k: b[k] - a[k] for k in a}


def stop_token_ids(tok, names) -> list[int]:
    out = []
    for s in names:
        i = tok.convert_tokens_to_ids(s)
        if i is None or i == tok.unk_token_id or tok.convert_ids_to_tokens(i) != s:
            raise ValueError(f"stop token {s!r} is not a single token in this tokenizer")
        out.append(int(i))
    return out


def prompt_ids(tok, lang: str, i: int, flores) -> list[int]:
    return fl.chat_prompt_ids(tok, fl.make_prompt(lang, i, flores)[0])


def one(llm, ids, sp):
    """One request at a time: (latency s, output token ids, finish reason, spec-count delta)."""
    c0 = spec_counts(llm)
    t0 = time.perf_counter()
    o = llm.generate([{"prompt_token_ids": ids}], sp, use_tqdm=False)[0].outputs[0]
    dt = time.perf_counter() - t0
    return dt, list(o.token_ids), o.finish_reason, delta(c0, spec_counts(llm))


def al(c: dict) -> float:
    return 1 + c["accepted"] / c["drafts"] if c["drafts"] else float("nan")


def write_jsonl(path: Path, recs):
    path.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--stages", default="calib,latency,sampling")
    ap.add_argument("--models", required=True, help="json {repo key: local path}")
    ap.add_argument("--flores", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lat-items", type=int, default=LAT_ITEMS)
    ap.add_argument("--sample-items", type=int, default=SAMPLE_ITEMS)
    ap.add_argument("--langs", default="all")
    args = ap.parse_args(argv)
    stages = args.stages.split(",")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.arm}_K{args.k}"
    if (out / f"{tag}.done").exists():
        log(tag, "already complete, skipping")
        return 0
    paths = json.loads(Path(args.models).read_text())
    langs = fl.LANGS if args.langs == "all" else args.langs.split(",")
    flores = fl.load_flores(args.flores, sorted(set(langs) | set(LAT_LANGS)), 200)
    t_load = time.time()
    llm = make_engine(args.arm, args.k, paths)
    tok = llm.get_tokenizer()
    stops = stop_token_ids(tok, ARMS[args.arm]["stops"])
    log(tag, f"engine ready in {time.time() - t_load:.0f}s; stops {stops}")
    greedy = sampling_params(0.0, 0, stops)

    if "calib" in stages:
        recs, tot = [], {"drafts": 0, "draft_tokens": 0, "accepted": 0}
        for i in range(CALIB_ITEMS):
            dt, y, fin, c = one(llm, prompt_ids(tok, "eng_Latn", i, flores), greedy)
            recs.append({"lang": "eng_Latn", "i": i, "lat": dt, "y": y, "finish": fin, **c})
            tot = {k: tot[k] + c[k] for k in tot}
        write_jsonl(out / f"{tag}_calib.jsonl", recs)
        a = ARMS[args.arm]
        got = al(tot)
        if "ref_al3" in a:
            rel = abs(got - a["ref_al3"]) / a["ref_al3"]
            ok = args.k == 3 and rel <= CALIB_TOL
            res = {"arm": args.arm, "k": args.k, "al": got, "ref": a["ref_al3"], "rel_diff": rel, "pass": ok}
        else:
            ok = args.k == 3 and got >= EAGLE_MIN_AL
            res = {"arm": args.arm, "k": args.k, "al": got, "min": EAGLE_MIN_AL, "pass": ok}
        (out / f"{tag}_calib.json").write_text(json.dumps(res, indent=1))
        log(tag, "CALIBRATION", json.dumps(res))
        if not ok:
            return 3

    if "latency" in stages:
        for i in range(WARMUP):
            one(llm, prompt_ids(tok, "eng_Latn", 200 - WARMUP + i, flores), greedy)
        recs = []
        for lang in LAT_LANGS:
            t0 = time.time()
            for i in range(args.lat_items):
                dt, y, fin, c = one(llm, prompt_ids(tok, lang, i, flores), greedy)
                recs.append({"lang": lang, "i": i, "lat": dt, "y": y, "finish": fin, **c})
            sub = [r for r in recs if r["lang"] == lang]
            log(tag, f"latency {lang}: {sum(r['lat'] for r in sub):.1f}s total, "
                     f"AL {al({k: sum(r[k] for r in sub) for k in ('drafts', 'draft_tokens', 'accepted')}):.3f}, "
                     f"wall {time.time() - t0:.0f}s")
        write_jsonl(out / f"{tag}_latency.jsonl", recs)

    if "sampling" in stages:
        recs = []
        for t in TEMPS:
            for lang in langs:
                ids = [prompt_ids(tok, lang, i, flores) for i in range(args.sample_items)]
                sps = [sampling_params(t, 1000 + i, stops) for i in range(args.sample_items)]
                c0 = spec_counts(llm)
                t0 = time.perf_counter()
                outs = llm.generate([{"prompt_token_ids": p} for p in ids], sps, use_tqdm=False)
                dt = time.perf_counter() - t0
                c = delta(c0, spec_counts(llm))
                recs.append({"temp": t, "lang": lang, "wall": dt, **c,
                             "items": [{"i": i, "y": list(o.outputs[0].token_ids), "finish": o.outputs[0].finish_reason}
                                       for i, o in enumerate(outs)]})
                log(tag, f"sampling T={t} {lang}: AL {al(c):.3f} wall {dt:.1f}s")
        write_jsonl(out / f"{tag}_sampling.jsonl", recs)

    (out / f"{tag}.done").write_text(json.dumps({"stages": stages, "seconds_incl_load": time.time() - t_load}))
    log(tag, "done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
