"""Implementation of exporting LLaMA PyTorch model to TinyChatEngine format.

Usage:
   python llama_exporter.py <path of hugging face model checkpoint> <output dir>

Example commandline:
   python tools/llama_exporter.py --model models/llama2-chat/hf7B --output models/LLaMA_7B_2_chat
"""
import argparse
import math
import os
import struct

import torch
from transformers import LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, LlamaModel, LlamaAttention
from transformers.models.llama.configuration_llama import LlamaConfig

@torch.no_grad()
def _export_model(model, prefix):

    outpath = prefix
    os.makedirs(outpath, exist_ok=True)
    with open(os.path.join(f"{outpath}", "lm_head.bin"), "wb") as f:
        f.write(model.lm_head._parameters["weight"].cpu().float().numpy().tobytes())
    _export_llama_model(model.model, os.path.join(f"{outpath}", "decoder"))


def _export_embed_tokens(embed_tokens, prefix):
    outpath = prefix
    os.makedirs(outpath, exist_ok=True)
    with open(os.path.join(f"{outpath}", "weight.bin"), "wb") as f:
        f.write(embed_tokens.weight.cpu().float().numpy().tobytes())


def _export_llama_model(model, prefix):
    outpath = prefix
    os.makedirs(outpath, exist_ok=True)

    _export_embed_tokens(model.embed_tokens, os.path.join(outpath, "embed_tokens"))
    _export_LlamaRMSNorm(model.norm, os.path.join(outpath, "norm"))
    for idx, layer in enumerate(model.layers):
        _export_llama_layer(layer, os.path.join(outpath, f"layer{idx}"))


def _export_LlamaRMSNorm(op, prefix):
    outpath = prefix
    os.makedirs(outpath, exist_ok=True)
    with open(os.path.join(f"{outpath}", "weight.bin"), "wb") as f:
        f.write(op.weight.cpu().float().numpy().tobytes())


def _export_llama_layer(layer, prefix):
    outpath = prefix
    os.makedirs(outpath, exist_ok=True)
    _export_attention_params(layer.self_attn, os.path.join(outpath, "self_attn"))
    _export_LlamaRMSNorm(layer.input_layernorm, os.path.join(outpath, "input_layernorm"))
    _export_LlamaRMSNorm(
        layer.post_attention_layernorm,
        os.path.join(outpath, "post_attention_layernorm"),
    )
    _export_linearfp(layer.mlp.gate_proj, os.path.join(outpath, "gate_proj"))
    _export_linearfp(layer.mlp.down_proj, os.path.join(outpath, "down_proj"))
    _export_linearfp(layer.mlp.up_proj, os.path.join(outpath, "up_proj"))


def _export_linearfp(op, prefix):
    outpath = prefix
    os.makedirs(outpath, exist_ok=True)
    with open(os.path.join(f"{outpath}", "weight.bin"), "wb") as f:
        f.write(op._parameters["weight"].cpu().float().numpy().tobytes())


def _export_rotaryEmbedding(rotary_emb: LlamaRotaryEmbedding, prefix):
    max_seq_len = rotary_emb.max_seq_len_cached
    dtype = torch.torch.float32
    t = torch.arange(max_seq_len, device="cpu", dtype=torch.float32)

    freqs = torch.outer(t, rotary_emb.original_inv_freq)
    # Different from paper, but it uses a different permutation in order to obtain the same calculation
    emb = torch.cat((freqs, freqs), dim=-1)
    cos_cached = emb.cos().to(dtype)
    sin_cached = emb.sin().to(dtype)

    outpath = prefix
    os.makedirs(outpath, exist_ok=True)
    with open(os.path.join(f"{outpath}", "cos_cached.bin"), "wb") as f:
        f.write(cos_cached.cpu().float().numpy().tobytes())
    with open(os.path.join(f"{outpath}", "sin_cached.bin"), "wb") as f:
        f.write(sin_cached.cpu().float().numpy().tobytes())


def _export_BMM_F32T(alpha, prefix):
    outpath = prefix
    os.makedirs(outpath, exist_ok=True)
    with open(os.path.join(f"{outpath}", "alpha.bin"), "wb") as f:
        f.write(struct.pack("f", alpha))


def _export_attention_params(attn: LlamaAttention, prefix: str):
    outpath = prefix
    os.makedirs(outpath, exist_ok=True)
    _export_linearfp(attn.k_proj, os.path.join(outpath, "k_proj"))
    _export_linearfp(attn.v_proj, os.path.join(outpath, "v_proj"))
    _export_linearfp(attn.q_proj, os.path.join(outpath, "q_proj"))
    _export_linearfp(attn.o_proj, os.path.join(outpath, "o_proj"))
    qk_bmm_alpha = 1 / math.sqrt(attn.head_dim)
    _export_BMM_F32T(qk_bmm_alpha, os.path.join(outpath, "qk_bmm"))

    rotary_emb: LlamaRotaryEmbedding = LlamaRotaryEmbedding(attn.config)
    _export_rotaryEmbedding(rotary_emb, os.path.join(outpath, "rotary_emb"))


def main():
    """Export a LLaMA model to TinyChatEngine format."""
    parser = argparse.ArgumentParser(description="export LLaMA pytorch model to TinyChatEngine format.")
    parser.add_argument("--hf_path", type=str, help="Path to huggingface model hub", default=None)
    parser.add_argument("--model", type=str, help="Path of the LLaMA torch model")
    parser.add_argument("--output", type=str, help="Output directory of the exported model")

    args = parser.parse_args()

    if args.hf_path is None:
        if not os.path.exists(args.model):
            print(f"The model path '{args.model}' does not exist.")
            return

        if not os.path.exists(args.output):
            print(f"The output path '{args.output}' does not exist. Creating a new directory...")
            os.makedirs(args.output, exist_ok=True)

        print("Loading model...")
        if args.model.endswith(".pt"):
            if args.model.split("/")[-1].lower().startswith("llama-2"):
                if args.model.split("-")[2].lower() == "7b":
                    print("Loading LLaMA 7B model...")
                    model = LlamaForCausalLM.from_pretrained("decapoda-research/llama-7b-hf", torch_dtype=torch.float16)
                elif args.model.split("-")[2].lower() == "13b":
                    print("Loading LLaMA 13B model...")
                    model = LlamaForCausalLM.from_pretrained("decapoda-research/llama-13b-hf", torch_dtype=torch.float16)
            elif args.model.split("/")[-1].lower().startswith("codellama"):
                if args.model.split("-")[1].lower() == "7b":
                    print("Loading CodaLLaMA 7B model...")
                    model = LlamaForCausalLM.from_pretrained("codellama/CodeLlama-7b-Instruct-hf", torch_dtype=torch.float16)
                elif args.model.split("-")[1].lower() == "13b":
                    print("Loading CodaLLaMA 13B model...")
                    model = LlamaForCausalLM.from_pretrained("codellama/CodeLlama-13b-Instruct-hf", torch_dtype=torch.float16)
            else:
                print("Model not supported.")
                return
            
            model.load_state_dict(torch.load(args.model))
        else:
            model = LlamaForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16)
    else:
        model: LlamaModel = LlamaForCausalLM.from_pretrained(args.hf_path, torch_dtype=torch.float)

    config: LlamaConfig = model.config
    params = {
        "batch": None,
        "num_heads": config.num_attention_heads,
        "num_kv_heads": config.num_key_value_heads if hasattr(config, "num_key_value_heads") else config.num_attention_heads,
        "num_layers": config.num_hidden_layers,
        "max_sqlen": config.max_position_embeddings,
        "embed_dim": config.hidden_size,
        "hidden_dim": config.intermediate_size,
        "vocsize": config.vocab_size,
        "padding_idx": config.pad_token_id,
        "rms_norm_eps": config.rms_norm_eps
    }
    print(params)
    print("Start exporting LLaMA model...")
    _export_model(model, args.output)
    print("Finished exporting LLaMA model.")


if __name__ == "__main__":
    main()
