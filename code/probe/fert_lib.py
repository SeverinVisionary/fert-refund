"""FERT-REFUND probe library: data, prompts, token tagging, drafter emulation, chain simulation.

Notation (one item). ids = prompt (P tokens) + realized output y (n tokens, the last one is the
stop token), N = P + n. The target's hidden state h_p at position p predicts ids[p+1]. At a verify
step the target has produced ids[t+1] from position t ("anchor t"); draft j (j = 1..K) predicts
ids[t+1+j]. Anchors run over t = P-1 .. N-3. The drafts are emulated teacher-forced: chain inputs are
the realized tokens, which equal the drafts whenever all earlier drafts were accepted, the only case
in which a later draft matters. For greedy, lossless speculation this gives the exact accept counts.

Qwen3.5 MTP (vLLM qwen3_5_mtp.py + llm_base_proposer.py): at position p the MTP consumes
fc([pre_fc_norm_embedding(embed(ids[p+1])); pre_fc_norm_hidden(h_p)]), runs one full-attention
decoder layer, then `norm` and the shared lm_head. Its own KV cache holds those step-1 entries for
every position. Chain step j >= 2 at anchor t feeds (draft_{j-1}, normed MTP output of step j-1) at
position t+j-1 and attends to step-1 entries at positions <= t plus the chain's own entries.

Gemma 4 assistant (HF SinglePositionMultiTokenCandidateGenerator): every draft step queries at the
constant position t+1 (vLLM uses t; both are evaluated, see fert_probe.G1), attends only to the K/V
exported by the target's last non-shared sliding and full layers at positions <= t, and consumes
[scaled target embed(token); hidden], hidden = target last hidden h_t at step 1 and the assistant's
post_projection output afterwards. Logits come from the centroid-masked embedding.
"""
from __future__ import annotations

import json
import math
import unicodedata
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

K_MAX = 5

LANG_NAME = {
    "eng_Latn": "English", "amh_Ethi": "Amharic", "mya_Mymr": "Burmese", "khm_Khmr": "Khmer",
    "guj_Gujr": "Gujarati", "kat_Geor": "Georgian", "tel_Telu": "Telugu", "ben_Beng": "Bengali",
    "hin_Deva": "Hindi", "tam_Taml": "Tamil", "tha_Thai": "Thai", "fin_Latn": "Finnish",
    "tur_Latn": "Turkish", "swh_Latn": "Swahili", "rus_Cyrl": "Russian", "arb_Arab": "Arabic",
    "jpn_Jpan": "Japanese", "zho_Hans": "Simplified Chinese", "kor_Hang": "Korean",
    "fra_Latn": "French", "deu_Latn": "German",
}
LANGS = list(LANG_NAME)  # 21 languages, English first
NON_WHITESPACE = {"tha_Thai", "mya_Mymr", "khm_Khmr", "jpn_Jpan", "zho_Hans"}

FAMILIES = {
    "qwen": {
        "target": "Qwen/Qwen3.5-9B", "target_rev": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        "stop_tokens": ["<|im_end|>", "<|endoftext|>"],
        # prenorm is reported only: pre_fc_norm_hidden re-normalizes, so it differs from "ok" only by the
        # final norm's per-dimension scale (run 1: 0.933 vs 0.945). shift (h_{p-1}) is the off-by-one bug.
        "wirings": ["ok", "swap", "prenorm", "shift"], "miswirings": ["swap", "shift"],
    },
    "gemma": {
        "target": "google/gemma-4-E4B-it", "target_rev": "ee0ef6023621cff504d758262d4e04895a5af4a2",
        "drafter": "google/gemma-4-E4B-it-assistant",
        "drafter_rev": "8d0031ea8c2109e2b1e86bb9368a4539b537f80a",
        "stop_tokens": ["<eos>", "<turn|>", "<|tool_response>"],
        # "+"-joined flags compose: each mis-wiring is also scored under the vLLM position convention
        "wirings": ["ok", "pos_t", "swap", "unscaled", "pos_t+swap", "pos_t+unscaled"],
        "miswirings": ["swap", "unscaled"],
    },
}

FLORES_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
FLORES_SHA256 = "b8b0b76783024b85797e5cc75064eb83fc5288b41e9654dabc7be6ae944011f6"


# ----------------------------------------------------------------------------- data and prompts

def load_flores(devtest_dir: str | Path, langs, n_items: int) -> dict[str, list[str]]:
    root = Path(devtest_dir)
    out = {}
    for lang in sorted(set(langs) | {"eng_Latn", "fra_Latn"}):
        lines = (root / f"{lang}.devtest").read_text(encoding="utf-8").split("\n")
        if lines and lines[-1] == "":
            lines = lines[:-1]
        if len(lines) != 1012:
            raise ValueError(f"{lang}.devtest has {len(lines)} lines, expected 1012")
        out[lang] = lines[:n_items]
    return out


def make_prompt(lang: str, i: int, flores: dict) -> tuple[str, str]:
    """(user message, reference translation). English outputs are translated from French."""
    if lang == "eng_Latn":
        src, src_name, ref = flores["fra_Latn"][i], "French", flores["eng_Latn"][i]
    else:
        src, src_name, ref = flores["eng_Latn"][i], "English", flores[lang][i]
    msg = (f"Translate the following {src_name} text into {LANG_NAME[lang]}. "
           f"Output only the translation, nothing else.\n\n{src}")
    return msg, ref


def chat_prompt_ids(tok, msg: str) -> list[int]:
    s = tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False,
                                add_generation_prompt=True, enable_thinking=False)
    return tok(s, add_special_tokens=False)["input_ids"]


def stop_ids(tok, family: str) -> list[int]:
    out = []
    for s in FAMILIES[family]["stop_tokens"]:
        i = tok.convert_tokens_to_ids(s)
        if i is None or i == tok.unk_token_id or tok.convert_ids_to_tokens(i) != s:
            raise ValueError(f"stop token {s!r} is not a single token in this tokenizer")
        out.append(int(i))
    return out


# ----------------------------------------------------------------------------- token bytes and tags

def _gpt2_byte_decoder() -> dict[str, int]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + \
        list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


class TokenBytes:
    """Maps a token id to the exact bytes it contributes to the decoded text (None = special)."""

    def __init__(self, tok):
        self.tok = tok
        self.added = {int(i): t for i, t in tok.added_tokens_decoder.items()}
        vocab = tok.get_vocab()
        self.sp = any(k.startswith("▁") for k in list(vocab)[:5000]) or "<0x41>" in vocab
        self.bd = None if self.sp else _gpt2_byte_decoder()
        self.cache: dict[int, bytes | None] = {}

    def __call__(self, i: int) -> bytes | None:
        i = int(i)
        if i in self.cache:
            return self.cache[i]
        if i in self.added:
            t = self.added[i]
            b = None if t.special else t.content.encode("utf-8")
        else:
            s = self.tok.convert_ids_to_tokens(i)
            if self.sp:
                if len(s) == 6 and s.startswith("<0x") and s.endswith(">"):
                    b = bytes([int(s[3:5], 16)])
                else:
                    b = s.replace("▁", " ").encode("utf-8")
            else:
                b = bytes(self.bd[ch] for ch in s)
        self.cache[i] = b
        return b


ZW = {"‌", "‍"}


def _utf8_codepoints(data: bytes) -> list[tuple[str, int]]:
    """(char, index of its last byte); invalid bytes become U+FFFD, one per byte."""
    out, i, n = [], 0, len(data)
    while i < n:
        b = data[i]
        ln = 1 if b < 0x80 else 2 if 0xC2 <= b <= 0xDF else 3 if 0xE0 <= b <= 0xEF else \
            4 if 0xF0 <= b <= 0xF4 else 0
        if ln == 0 or i + ln > n:
            out.append(("�", i))
            i += 1
            continue
        try:
            ch = data[i:i + ln].decode("utf-8")
        except UnicodeDecodeError:
            out.append(("�", i))
            i += 1
            continue
        out.append((ch, i + ln - 1))
        i += ln
    return out


TAGS = ("word_start", "cont", "frag", "other", "special")


def tag_tokens(ids, tb: TokenBytes) -> list[str]:
    """Tag each token by the code points it completes (holds the last byte of).

    word_start: completes the first code point of a word (a maximal run of L*/M*/N* characters,
    with ZWJ/ZWNJ counted as word characters after a word character); cont: completes only later
    code points of a word; other: completes code points but no word character; frag: completes no
    code point; special: a special token.
    """
    spans, data = [], bytearray()
    for i in ids:
        b = tb(i)
        if b is None:
            spans.append(None)
        else:
            spans.append((len(data), len(data) + len(b)))
            data += b
    cps = _utf8_codepoints(bytes(data))
    owner = np.full(len(data), -1, dtype=np.int64)
    for k, sp in enumerate(spans):
        if sp is not None:
            owner[sp[0]:sp[1]] = k
    first = [False] * len(ids)
    wordc = [False] * len(ids)
    anyc = [False] * len(ids)
    prev_word = False
    for ch, last in cps:
        cat = unicodedata.category(ch)
        is_word = cat[0] in "LMN" or (ch in ZW and prev_word)
        k = int(owner[last])
        anyc[k] = True
        if is_word:
            wordc[k] = True
            if not prev_word:
                first[k] = True
        prev_word = is_word
    tags = []
    for k, sp in enumerate(spans):
        if sp is None:
            tags.append("special")
        elif not anyc[k]:
            tags.append("frag")
        elif first[k]:
            tags.append("word_start")
        elif wordc[k]:
            tags.append("cont")
        else:
            tags.append("other")
    return tags


# ----------------------------------------------------------------------------- validity and simulation

def rep4_fraction(y) -> float:
    if len(y) < 8:
        return 0.0
    grams = [tuple(y[i:i + 4]) for i in range(len(y) - 3)]
    return 1.0 - len(set(grams)) / len(grams)


def simulate(acc: np.ndarray, P: int, N: int, K: int) -> int:
    """Verify steps after prefill for greedy chain speculation at depth K.

    acc[a, j-1] is True iff draft j at anchor t = P-1+a equals ids[t+1+j]. Returns the number of
    verify forwards; the prefill is one more forward.
    """
    t, it = P - 1, 0
    while t + 1 < N - 1:
        it += 1
        a = t - (P - 1)
        A = 0
        while A < K and acc[a, A]:
            A += 1
        A = min(A, N - 2 - t)
        t += A + 1
    return it


def acceptance_matrix(drafts: torch.Tensor, ids: torch.Tensor, P: int) -> np.ndarray:
    """drafts[a, j-1] = draft j at anchor t = P-1+a. Returns the bool accept matrix."""
    N = ids.shape[0]
    T, K = drafts.shape
    t = torch.arange(P - 1, P - 1 + T)
    acc = torch.zeros(T, K, dtype=torch.bool)
    ids_c = ids.cpu()
    d = drafts.cpu()
    for j in range(1, K + 1):
        tgt = t + 1 + j
        ok = tgt <= N - 1
        acc[ok, j - 1] = d[ok, j - 1] == ids_c[tgt[ok]]
    return acc.numpy()


# ----------------------------------------------------------------------------- generation and target

@torch.no_grad()
def generate_batch(model, prompts: list[list[int]], max_new: int, stops: list[int], pad_id: int,
                   device) -> list[tuple[list[int], bool]]:
    """Greedy, left-padded. Returns (y up to and including the first stop token, capped)."""
    from transformers import GenerationConfig
    L = max(len(p) for p in prompts)
    inp = torch.full((len(prompts), L), pad_id, dtype=torch.long)
    att = torch.zeros((len(prompts), L), dtype=torch.long)
    for r, p in enumerate(prompts):
        inp[r, L - len(p):] = torch.tensor(p)
        att[r, L - len(p):] = 1
    gc = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=max_new, eos_token_id=stops,
                          pad_token_id=pad_id, use_cache=True)
    out = model.generate(input_ids=inp.to(device), attention_mask=att.to(device), generation_config=gc)
    res = []
    sset = set(stops)
    for r in range(len(prompts)):
        g = out[r, L:].tolist()
        y, capped = [], True
        for x in g:
            y.append(x)
            if x in sset:
                capped = False
                break
        res.append((y, capped))
    return res


def _argmax_chunked(head, h: torch.Tensor, chunk: int = 256) -> torch.Tensor:
    return torch.cat([head(h[i:i + chunk]).argmax(-1) for i in range(0, h.shape[0], chunk)])


@torch.no_grad()
def qwen_target(model, ids: torch.Tensor) -> dict:
    """Unpadded teacher-forced target forward. h = post-final-norm (what vLLM hands the MTP)."""
    lm = model.model.language_model
    cap = {}
    hook = lm.norm.register_forward_hook(lambda m, i, o: cap.update(pre=i[0]))
    try:
        out = lm(input_ids=ids[None], position_ids=torch.arange(ids.shape[0], device=ids.device)[None],
                 use_cache=False)
    finally:
        hook.remove()
    h = out.last_hidden_state[0]
    return {"h": h, "h_pre": cap["pre"][0], "argmax": _argmax_chunked(model.lm_head, h)}


@torch.no_grad()
def gemma_target(model, ids: torch.Tensor) -> dict:
    out = model(input_ids=ids[None], output_hidden_states=True, return_shared_kv_states=True,
                use_cache=False)
    kv = {k: (v[0], v[1]) for k, v in out.shared_kv_states.items()}
    return {"h": out.hidden_states[-1][0], "kv": kv, "argmax": out.logits[0].argmax(-1)}


@torch.no_grad()
def ref_nll(model, family: str, prompt: list[int], ref_ids: list[int], device) -> float:
    """Sum NLL (nats) of the reference tokens after the generation prompt."""
    ids = torch.tensor(prompt + ref_ids, device=device)
    P = len(prompt)
    if family == "qwen":
        lm = model.model.language_model
        h = lm(input_ids=ids[None], position_ids=torch.arange(ids.shape[0], device=device)[None],
               use_cache=False).last_hidden_state[0]
        logits = model.lm_head(h[P - 1:-1])
    else:
        logits = model(input_ids=ids[None], use_cache=False).logits[0, P - 1:-1]
    lp = torch.log_softmax(logits.float(), -1)
    return float(-lp.gather(-1, ids[P:, None]).sum())


# ----------------------------------------------------------------------------- Qwen3.5 MTP

class QwenMTP(nn.Module):
    def __init__(self, text_cfg):
        super().__init__()
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5RMSNorm
        H, eps = text_cfg.hidden_size, text_cfg.rms_norm_eps
        self.fc = nn.Linear(2 * H, H, bias=False)
        self.pre_fc_norm_embedding = Qwen3_5RMSNorm(H, eps=eps)
        self.pre_fc_norm_hidden = Qwen3_5RMSNorm(H, eps=eps)
        full_idx = text_cfg.layer_types.index("full_attention")
        self.layers = nn.ModuleList([Qwen3_5DecoderLayer(text_cfg, full_idx)])
        self.norm = Qwen3_5RMSNorm(H, eps=eps)

    def fuse(self, e, h, wiring: str):
        ne, nh = self.pre_fc_norm_embedding(e), self.pre_fc_norm_hidden(h)
        return self.fc(torch.cat([nh, ne] if wiring == "swap" else [ne, nh], dim=-1))


def load_qwen_mtp(model_dir: str | Path, text_cfg, dtype, device) -> QwenMTP:
    """Load the mtp.* tensors from the checkpoint shards; every key must map, every param must load."""
    from safetensors import safe_open
    model_dir = Path(model_dir)
    idx = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    keys = sorted(k for k in idx if k.startswith("mtp."))
    if not keys:
        raise ValueError("checkpoint has no mtp.* tensors")
    sd = {}
    for f in sorted({idx[k] for k in keys}):
        with safe_open(str(model_dir / f), framework="pt") as fh:
            for k in fh.keys():
                if k.startswith("mtp."):
                    sd[k[len("mtp."):]] = fh.get_tensor(k)
    if sorted("mtp." + k for k in sd) != keys:
        raise ValueError("mtp tensors in the shards differ from the index")
    mtp = QwenMTP(text_cfg)
    missing, unexpected = mtp.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise ValueError(f"MTP load mismatch: missing={missing} unexpected={unexpected}")
    return mtp.to(device=device, dtype=dtype).eval()


def _qkv(layer, x, cos, sin):
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
    sa = layer.self_attn
    hd = sa.head_dim
    B, T, _ = x.shape
    a = layer.input_layernorm(x)
    q, gate = torch.chunk(sa.q_proj(a).view(B, T, -1, hd * 2), 2, dim=-1)
    gate = gate.reshape(B, T, -1)
    q = sa.q_norm(q).transpose(1, 2)
    k = sa.k_norm(sa.k_proj(a).view(B, T, -1, hd)).transpose(1, 2)
    v = sa.v_proj(a).view(B, T, -1, hd).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    return q, k, v, gate


def _finish(layer, x, attn, gate):
    sa = layer.self_attn
    x = x + sa.o_proj(attn * torch.sigmoid(gate))
    return x + layer.mlp(layer.post_attention_layernorm(x))


def _rep(x, n):
    return x.repeat_interleave(n, dim=1) if n > 1 else x


@torch.no_grad()
def qwen_drafts(mtp: QwenMTP, model, ids: torch.Tensor, h: torch.Tensor, P: int, K: int = K_MAX,
                wiring: str = "ok", chunk: int = 128) -> torch.Tensor:
    """Batched teacher-forced MTP chain. Returns drafts [T, K], T = N-2-(P-1) anchors."""
    lm = model.model.language_model
    embed, rotary, head = lm.embed_tokens, lm.rotary_emb, model.lm_head
    layer = mtp.layers[0]
    sa = layer.self_attn
    grp = sa.num_key_value_groups
    N = ids.shape[0]
    dev = ids.device
    # step 1 over every position p = 0..N-2: (ids[p+1], h_p) at position p, causal
    x = mtp.fuse(embed(ids[1:])[None], h[:-1][None], wiring)
    pos = torch.arange(N - 1, device=dev)
    cos, sin = rotary(x, pos[None])
    q, k1, v1, gate = _qkv(layer, x, cos, sin)
    att = F.scaled_dot_product_attention(q.float(), _rep(k1, grp).float(), _rep(v1, grp).float(),
                                         is_causal=True, scale=sa.scaling).to(x.dtype)
    g1 = mtp.norm(_finish(layer, x, att.transpose(1, 2).reshape(1, N - 1, -1), gate))[0]
    anchors = torch.arange(P - 1, N - 2, device=dev)
    T = anchors.shape[0]
    drafts = torch.zeros(T, K, dtype=torch.long, device=dev)
    if T == 0:
        return drafts
    k1r, v1r = _rep(k1, grp).float(), _rep(v1, grp).float()
    for c0 in range(0, T, chunk):
        t = anchors[c0:c0 + chunk]
        Tc = t.shape[0]
        g = g1[t]
        drafts[c0:c0 + Tc, 0] = _argmax_chunked(head, g)
        ck, cv = [], []
        keymask = torch.arange(N - 1, device=dev)[None, :] <= t[:, None]  # [Tc, N-1]
        for j in range(2, K + 1):
            tok = ids[torch.clamp(t + j, max=N - 1)]
            xj = mtp.fuse(embed(tok)[None], g[None], wiring)
            cos, sin = rotary(xj, (t + j - 1)[None])
            qj, kj, vj, gj = _qkv(layer, xj, cos, sin)
            ck.append(_rep(kj, grp).float())
            cv.append(_rep(vj, grp).float())
            qf = qj.float()
            s1 = torch.einsum("bhtd,bhnd->bhtn", qf, k1r) * sa.scaling
            s1 = s1.masked_fill(~keymask[None, None], float("-inf"))
            sc = torch.stack([(qf * kk).sum(-1) for kk in ck], dim=-1) * sa.scaling  # [1,H,Tc,j-1]
            p = torch.softmax(torch.cat([s1, sc], dim=-1), dim=-1)
            o = torch.einsum("bhtn,bhnd->bhtd", p[..., :N - 1], v1r)
            for i, vv in enumerate(cv):
                o = o + p[..., N - 1 + i, None] * vv
            o = o.to(xj.dtype).transpose(1, 2).reshape(1, Tc, -1)
            g = mtp.norm(_finish(layer, xj, o, gj))[0]
            drafts[c0:c0 + Tc, j - 1] = _argmax_chunked(head, g)
    return drafts


@torch.no_grad()
def qwen_reference_drafts(mtp: QwenMTP, model, ids: torch.Tensor, h: torch.Tensor, t: int, K: int,
                          wiring: str = "ok", feed: str = "realized") -> list[int]:
    """Sequential reference for one anchor, built on the unmodified HF decoder layer.

    The MTP input sequence is the step-1 pairs for positions 0..t followed by the chain entries,
    run causally from scratch at every step (no cache): exactly what the vLLM KV cache holds.
    feed='realized' uses ids[t+j] as chain input tokens (teacher-forced), 'drafted' the drafts.
    """
    lm = model.model.language_model
    embed, rotary, head = lm.embed_tokens, lm.rotary_emb, model.lm_head
    layer = mtp.layers[0]
    toks = [int(x) for x in ids[1:t + 2]]
    hids = [h[p] for p in range(t + 1)]
    out = []
    for j in range(1, K + 1):
        L = len(toks)
        x = mtp.fuse(embed(torch.tensor(toks, device=ids.device))[None], torch.stack(hids)[None], wiring)
        pe = rotary(x, torch.arange(L, device=ids.device)[None])
        y = layer(x, position_embeddings=pe, attention_mask=None)
        g = mtp.norm(y)[0, -1]
        d = int(head(g).argmax())
        out.append(d)
        nxt = d if feed == "drafted" else int(ids[min(t + j + 1, ids.shape[0] - 1)])
        toks.append(nxt)
        hids.append(g)
    return out


# ----------------------------------------------------------------------------- Gemma 4 assistant

def _flags(wiring: str) -> set[str]:
    return set(wiring.split("+"))


def compose_wiring(primary: str, miswiring: str) -> str:
    """The mis-wiring evaluated under the primary's position convention."""
    return miswiring if primary == "ok" else f"{primary}+{miswiring}"


def _gemma_embed(model, tok: torch.Tensor, wiring: str):
    emb = model.get_input_embeddings()
    if "unscaled" in _flags(wiring):
        return F.embedding(tok, emb.weight)
    return emb(tok)


@torch.no_grad()
def gemma_drafts(asst, model, ids: torch.Tensor, h: torch.Tensor, kv: dict, P: int, K: int = K_MAX,
                 wiring: str = "ok", chunk: int = 64) -> torch.Tensor:
    """Batched teacher-forced assistant chain: every anchor is one query row in a single sequence.

    Each query row attends only to the target's shared K/V (the assistant has no K/V of its own),
    so rows are independent and an explicit per-row mask reproduces the per-anchor forward.
    """
    N = ids.shape[0]
    dev = ids.device
    W = asst.config.get_text_config().sliding_window
    anchors = torch.arange(P - 1, N - 2, device=dev)
    T = anchors.shape[0]
    drafts = torch.zeros(T, K, dtype=torch.long, device=dev)
    L = kv["full_attention"][0].shape[2]
    kpos = torch.arange(L, device=dev)[None, :]
    for c0 in range(0, T, chunk):
        t = anchors[c0:c0 + chunk]
        Tc = t.shape[0]
        full = kpos <= t[:, None]
        # HF's flipped bidirectional window keeps the last W+1 target positions (vLLM keeps W);
        # it binds only once t >= W, and the probe matches HF.
        slide = full & (kpos >= t[:, None] - W)
        masks = {"full_attention": full[None, None], "sliding_attention": slide[None, None]}
        pos = (t if "pos_t" in _flags(wiring) else t + 1)[None]
        tok, hid = ids[t + 1], h[t]
        for j in range(1, K + 1):
            e = _gemma_embed(model, tok, wiring)
            x = torch.cat([hid, e] if "swap" in _flags(wiring) else [e, hid], dim=-1)[None]
            x = asst.pre_projection(x)
            out = asst.model(input_ids=None, inputs_embeds=x, attention_mask=masks, position_ids=pos,
                             shared_kv_states=kv, use_cache=False)
            last = out.last_hidden_state
            hid = asst.post_projection(last)[0]
            if asst.config.use_ordered_embeddings:
                logits = asst.masked_embedding(last, asst.lm_head.weight)
            else:
                logits = asst.lm_head(last)
            drafts[c0:c0 + Tc, j - 1] = logits[0].argmax(-1)
            tok = ids[torch.clamp(t + 1 + j, max=N - 1)]
    return drafts


@torch.no_grad()
def gemma_reference_drafts(asst, model, ids: torch.Tensor, h: torch.Tensor, kv: dict, t: int, K: int,
                           wiring: str = "ok", feed: str = "realized") -> list[int]:
    """The HF candidate generator's own per-anchor call, verbatim in its inputs."""
    kvt = {k: (v[0][:, :, :t + 1], v[1][:, :, :t + 1]) for k, v in kv.items()}
    last_tok = ids[t + 1:t + 2][None]
    last_h = h[t:t + 1][None]
    pos = torch.tensor([[t if "pos_t" in _flags(wiring) else t + 1]], device=ids.device)
    att = torch.ones(1, t + 2, dtype=torch.long, device=ids.device)
    out = []
    for j in range(1, K + 1):
        e = _gemma_embed(model, last_tok, wiring)
        x = torch.cat([last_h, e] if "swap" in _flags(wiring) else [e, last_h], dim=-1)
        o = asst(inputs_embeds=x, attention_mask=att, position_ids=pos, shared_kv_states=kvt, use_cache=False)
        d = int(o.logits[0, -1].argmax())
        out.append(d)
        last_h = o.last_hidden_state
        nxt = d if feed == "drafted" else int(ids[min(t + 1 + j, ids.shape[0] - 1)])
        last_tok = torch.tensor([[nxt]], device=ids.device)
    return out


# ----------------------------------------------------------------------------- per-item driver

def qwen_hidden(tf: dict, wiring: str) -> torch.Tensor:
    """The target hidden the MTP consumes under a wiring: post-norm h_p, pre-norm, or h_{p-1}."""
    if wiring == "prenorm":
        return tf["h_pre"]
    if wiring == "shift":
        return torch.cat([tf["h"][:1], tf["h"][:-1]])
    return tf["h"]


def item_drafts(family: str, model, drafter, ids: torch.Tensor, P: int, wiring: str = "ok",
                tf: dict | None = None) -> tuple[torch.Tensor, dict]:
    if tf is None:
        tf = qwen_target(model, ids) if family == "qwen" else gemma_target(model, ids)
    if family == "qwen":
        d = qwen_drafts(drafter, model, ids, qwen_hidden(tf, wiring), P, K_MAX, "swap" if wiring == "swap" else "ok")
    else:
        d = gemma_drafts(drafter, model, ids, tf["h"], tf["kv"], P, K_MAX, wiring)
    return d, tf


def step1_acceptance(acc: np.ndarray) -> tuple[int, int]:
    return int(acc[:, 0].sum()), int(acc.shape[0])


def encode_acc(acc: np.ndarray) -> list[int]:
    w = (1 << np.arange(acc.shape[1]))[None, :]
    return [int(x) for x in (acc.astype(np.int64) * w).sum(1)]


def decode_acc(bits: list[int], K: int = K_MAX) -> np.ndarray:
    b = np.asarray(bits, dtype=np.int64)[:, None]
    return ((b >> np.arange(K)[None, :]) & 1).astype(bool) if len(bits) else np.zeros((0, K), bool)


def spearman_partial(x, y, z) -> float:
    from scipy.stats import rankdata
    rx, ry, rz = (rankdata(np.asarray(v, dtype=float)) for v in (x, y, z))
    rxy, rxz, ryz = (float(np.corrcoef(a, b)[0, 1]) for a, b in ((rx, ry), (rx, rz), (ry, rz)))
    den = math.sqrt(max((1 - rxz ** 2) * (1 - ryz ** 2), 0.0))
    return float("nan") if den == 0 else (rxy - rxz * ryz) / den
