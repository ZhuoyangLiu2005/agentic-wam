"""Precompute Cosmos-Reason1 text embeddings for a set of instructions and cache them so the
training dataset can load `prompt_embs` without holding the 8.3B encoder in the training graph.

Each cache entry: {sha256(DEFAULT_PROMPT.format(task=instr))}.cosmos_reason1.pt -> {"prompt_embs": [L,100352] bf16}
padded/truncated to a fixed L (default 128; instructions are short so real tokens fit well under it).

Usage (on a GPU node with HF internet for the Cosmos-Reason1 CONFIG only; weights come from the ckpt):
  python scripts/precompute_cosmos_text_embeds.py \
    --instructions <file, one per line> --out-dir <cache dir> \
    --ckpt <checkpoint-200000> --max-len 128 [--limit N]

"""
import argparse, glob, hashlib, os, sys, torch

DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"


def _text_encoder_state(path):
    """Cosmos-Reason1 weights from `path`: the public nvidia/Cosmos-Reason1-7B directory, or a Cosmos video
    checkpoint that embeds it under `text_encoder.text_encoder.*`. Returns (state_dict, config source)."""
    from safetensors.torch import load_file
    sd = {}
    for s in sorted(glob.glob(os.path.join(path, "model-*-of-*.safetensors"))):
        sd.update(load_file(s))
    if not sd:
        raise FileNotFoundError(f"no model-*-of-*.safetensors under {path} (paths.text_encoder)")
    pfx = "text_encoder.text_encoder."
    if any(k.startswith(pfx) for k in sd):
        sd = {k[len(pfx):]: v for k, v in sd.items() if k.startswith(pfx)}
    has_cfg = os.path.isfile(os.path.join(path, "config.json")) and os.path.isfile(os.path.join(path, "tokenizer.json"))
    return sd, (path if has_cfg else "nvidia/Cosmos-Reason1-7B")


def build_text_encoder(path, device, dtype=torch.bfloat16):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoConfig, AutoProcessor
    from accelerate import init_empty_weights
    sd, src = _text_encoder_state(path)
    cfg = AutoConfig.from_pretrained(src)
    with init_empty_weights():
        qwen = Qwen2_5_VLForConditionalGeneration(cfg)
    def _remap(k):  # the weights use the older flat Qwen2.5-VL naming; transformers>=4.49 nests under model.*
        if k.startswith("visual."):
            return "model." + k
        if k.startswith("model."):
            return "model.language_model." + k[len("model."):]
        return k
    qwen.load_state_dict({_remap(k): v for k, v in sd.items()}, strict=False, assign=True)
    _meta = [n for n, pr in qwen.named_parameters() if pr.is_meta]
    assert not _meta, f"{len(_meta)} params still on meta after remap: {_meta[:3]}"
    qwen = qwen.to(device, dtype).eval()
    proc = AutoProcessor.from_pretrained(src)
    return qwen, proc


@torch.no_grad()
def encode(qwen, proc, prompt, max_len, device):
    conv = [
        {"role": "system", "content": [{"type": "text",
         "text": "You are a helpful assistant who will provide prompts to an image generator."}]},
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]
    ids = proc.apply_chat_template(conv, tokenize=True, add_generation_prompt=False,
                                   max_length=max_len, truncation=True, padding="max_length")
    ids = torch.as_tensor(ids, dtype=torch.long, device=device)
    if ids.ndim == 2:
        ids = ids[0]
    ids = ids[:max_len].unsqueeze(0)
    hs = qwen(ids, output_hidden_states=True).hidden_states
    emb = torch.cat([(h - h.mean(-1, keepdim=True)) / (h.std(-1, keepdim=True) + 1e-8)
                     for h in hs[1:]], dim=-1)          # [1, L, 100352]
    return emb[0].to(torch.bfloat16).cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instructions", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ckpt", default=None, help="Cosmos-Reason1-7B directory (default: paths.text_encoder in configs/paths/default.yaml)")
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.ckpt is None:
        from agenticwam.utils.paths import default_path
        a.ckpt = default_path("text_encoder")

    os.makedirs(a.out_dir, exist_ok=True)
    raw = [l.strip().strip('"') for l in open(a.instructions) if l.strip()]
    uniq = sorted(set(raw))
    if a.limit:
        uniq = uniq[:a.limit]
    print(f"{len(raw)} lines -> {len(uniq)} unique instructions; max_len={a.max_len}")
    qwen, proc = build_text_encoder(a.ckpt, a.device)
    print("text encoder ready")
    done = skip = 0
    for i, instr in enumerate(uniq):
        prompt = DEFAULT_PROMPT.format(task=instr)
        h = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        out = os.path.join(a.out_dir, f"{h}.cosmos_reason1.pt")
        if os.path.exists(out):
            skip += 1
            continue
        emb = encode(qwen, proc, prompt, a.max_len, a.device)
        torch.save({"prompt_embs": emb}, out)
        done += 1
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(uniq)} (wrote {done}, skipped {skip})")
    print(f"DONE: wrote {done}, skipped {skip}, cache={a.out_dir}")


if __name__ == "__main__":
    main()
