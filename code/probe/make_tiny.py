"""Tiny random-weight stand-ins for the probe's models, used by the tests and by SMOKE runs.

Each tiny model keeps the real architecture class, the real config flags and the real
tokenizer/chat template, and shrinks only the dimensions. The Qwen stand-in also gets
`mtp.*` tensors in its own safetensors shard, listed in the index json exactly as in the
released checkpoint, so the MTP loader is exercised on the same file layout.

Usage: python make_tiny.py <real_cfg_root> <out_root>
  real_cfg_root holds qwen35/, gemma4/, gemma4a/ (config.json + tokenizer files)
"""
from __future__ import annotations

import copy
import json
import shutil
import sys
from pathlib import Path

import torch

TOK_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
             "special_tokens_map.json", "generation_config.json")


def _copy_tok(src: Path, dst: Path):
    for f in TOK_FILES:
        if (src / f).exists():
            shutil.copy(src / f, dst / f)


def tiny_qwen_config(real: dict, vocab: int | None = None, layers: int = 4) -> dict:
    c = copy.deepcopy(real)
    t = c["text_config"]
    t.update(hidden_size=64, intermediate_size=96, num_hidden_layers=layers, num_attention_heads=4,
             num_key_value_heads=2, head_dim=16, linear_key_head_dim=8, linear_value_head_dim=8,
             linear_num_key_heads=2, linear_num_value_heads=4, full_attention_interval=4)
    t["layer_types"] = ["full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(layers)]
    # partial_rotary_factor 0.25 of head_dim 16 -> 4 rotary dims; mrope_section must sum to 2
    t["rope_parameters"] = dict(t["rope_parameters"], mrope_section=[1, 1, 0])
    if vocab is not None:
        t["vocab_size"] = vocab
    v = c["vision_config"]
    v.update(depth=1, hidden_size=32, intermediate_size=32, num_heads=2, out_hidden_size=64,
             num_position_embeddings=64)
    return c


def tiny_gemma_config(real: dict, vocab: int | None = None, sliding_window: int = 6) -> dict:
    c = copy.deepcopy(real)
    t = c["text_config"]
    lt = ["sliding_attention", "sliding_attention", "full_attention"] * 2
    t.update(hidden_size=64, intermediate_size=96, num_hidden_layers=len(lt), num_attention_heads=4,
             num_key_value_heads=2, head_dim=16, global_head_dim=32, hidden_size_per_layer_input=8,
             num_kv_shared_layers=2, sliding_window=sliding_window, layer_types=lt)
    if vocab is not None:
        t["vocab_size"] = vocab
        t["vocab_size_per_layer_input"] = vocab
    v = c["vision_config"]
    v.update(hidden_size=32, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
             num_key_value_heads=2, head_dim=16, global_head_dim=16, position_embedding_size=64)
    a = c["audio_config"]
    a.update(hidden_size=32, num_hidden_layers=1, num_attention_heads=2, output_proj_dims=64,
             subsampling_conv_channels=[8, 4])
    return c


def tiny_assistant_config(real: dict, target_text: dict, vocab: int | None = None) -> dict:
    c = copy.deepcopy(real)
    t = c["text_config"]
    t.update(hidden_size=32, intermediate_size=48, num_attention_heads=4,
             num_key_value_heads=target_text["num_key_value_heads"],
             head_dim=target_text["head_dim"], global_head_dim=target_text["global_head_dim"],
             sliding_window=target_text["sliding_window"])
    c["backbone_hidden_size"] = target_text["hidden_size"]
    v = vocab if vocab is not None else t["vocab_size"]
    t["vocab_size"] = v
    c["num_centroids"] = 8 if v <= 64 else 2048
    c["centroid_intermediate_top_k"] = 2 if v <= 64 else 4
    return c


def _randomize(model, seed: int, std: float = 0.2):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dtype.is_floating_point:
                p.copy_(torch.randn(p.shape, generator=g) * std)


def build_qwen(cfg: dict, seed: int = 0, dtype=torch.float32):
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
    m = Qwen3_5ForConditionalGeneration(Qwen3_5Config(**cfg)).to(dtype).eval()
    _randomize(m, seed)
    return m


def random_mtp_state(text_cfg: dict, seed: int = 1, std: float = 0.2) -> dict[str, torch.Tensor]:
    H, I = text_cfg["hidden_size"], text_cfg["intermediate_size"]
    nh, nkv, hd = text_cfg["num_attention_heads"], text_cfg["num_key_value_heads"], text_cfg["head_dim"]
    shapes = {
        "mtp.fc.weight": (H, 2 * H),
        "mtp.layers.0.self_attn.q_proj.weight": (nh * hd * 2, H),
        "mtp.layers.0.self_attn.k_proj.weight": (nkv * hd, H),
        "mtp.layers.0.self_attn.v_proj.weight": (nkv * hd, H),
        "mtp.layers.0.self_attn.o_proj.weight": (H, nh * hd),
        "mtp.layers.0.self_attn.q_norm.weight": (hd,),
        "mtp.layers.0.self_attn.k_norm.weight": (hd,),
        "mtp.layers.0.mlp.gate_proj.weight": (I, H),
        "mtp.layers.0.mlp.up_proj.weight": (I, H),
        "mtp.layers.0.mlp.down_proj.weight": (H, I),
        "mtp.layers.0.input_layernorm.weight": (H,),
        "mtp.layers.0.post_attention_layernorm.weight": (H,),
        "mtp.norm.weight": (H,),
        "mtp.pre_fc_norm_embedding.weight": (H,),
        "mtp.pre_fc_norm_hidden.weight": (H,),
    }
    g = torch.Generator().manual_seed(seed)
    return {k: torch.randn(s, generator=g) * std for k, s in shapes.items()}


def build_gemma(cfg: dict, seed: int = 0, dtype=torch.float32):
    from transformers import Gemma4Config, Gemma4ForConditionalGeneration
    m = Gemma4ForConditionalGeneration(Gemma4Config(**cfg)).to(dtype).eval()
    _randomize(m, seed)
    return m


def build_assistant(cfg: dict, seed: int = 2, dtype=torch.float32):
    from transformers import Gemma4AssistantConfig, Gemma4AssistantForCausalLM
    m = Gemma4AssistantForCausalLM(Gemma4AssistantConfig(**cfg)).to(dtype).eval()
    _randomize(m, seed)
    g = torch.Generator().manual_seed(seed + 100)
    v = m.masked_embedding.vocab_size
    with torch.no_grad():
        m.masked_embedding.token_ordering.copy_(torch.randperm(v, generator=g))
    return m


def save_qwen(model, mtp_state: dict, out: Path, tok_src: Path | None):
    from safetensors.torch import save_file
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, max_shard_size="5GB")
    save_file({k: v.contiguous() for k, v in mtp_state.items()}, str(out / "model-mtp.safetensors"))
    idx_path = out / "model.safetensors.index.json"
    if idx_path.exists():
        idx = json.loads(idx_path.read_text())
    else:  # single-file save: write an index so the layout matches the released checkpoint
        from safetensors import safe_open
        with safe_open(str(out / "model.safetensors"), "pt") as f:
            idx = {"metadata": {}, "weight_map": {k: "model.safetensors" for k in f.keys()}}
    for k in mtp_state:
        idx["weight_map"][k] = "model-mtp.safetensors"
    idx_path.write_text(json.dumps(idx, indent=1))
    if tok_src is not None:
        _copy_tok(tok_src, out)


def save_hf(model, out: Path, tok_src: Path | None):
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    if tok_src is not None:
        _copy_tok(tok_src, out)


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    qreal = json.loads((src / "qwen35/config.json").read_text())
    qc = tiny_qwen_config(qreal)
    save_qwen(build_qwen(qc), random_mtp_state(qc["text_config"]), dst / "qwen", src / "qwen35")
    greal = json.loads((src / "gemma4/config.json").read_text())
    gc = tiny_gemma_config(greal)
    save_hf(build_gemma(gc), dst / "gemma", src / "gemma4")
    areal = json.loads((src / "gemma4a/config.json").read_text())
    ac = tiny_assistant_config(areal, gc["text_config"])
    save_hf(build_assistant(ac), dst / "gemma_assistant", None)
    print("tiny models written to", dst)


if __name__ == "__main__":
    main()
