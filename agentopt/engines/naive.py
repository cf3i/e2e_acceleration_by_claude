"""Stage 00 engine: plain HF transformers, one request at a time.

Semantically equivalent to `model.generate(do_sample=False)`: prefill computes only the last logit,
then a Python loop decodes one token per forward with an HF DynamicCache and a host sync per token
(argmax -> int) for the stop check. Every turn re-prefills the whole conversation from scratch.
"""
import torch
from transformers import AutoTokenizer, Qwen3_5ForCausalLM

from .. import MODEL_PATH


class NaiveEngine:
    def __init__(self, model_path: str = MODEL_PATH, device: str = "cuda", dtype=torch.bfloat16):
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.model = Qwen3_5ForCausalLM.from_pretrained(model_path, dtype=dtype).to(device).eval()
        self.stop_ids = {self.tok.convert_tokens_to_ids("<|im_end|>"), self.tok.convert_tokens_to_ids("<|endoftext|>")}

    @torch.inference_mode()
    def generate(self, input_ids: list[int], max_new_tokens: int, timer):
        """Returns (generated token ids without the stop token, stats dict)."""
        ids = torch.tensor([input_ids], device=self.device)
        with timer("prefill") as t_pre:
            out = self.model(input_ids=ids, use_cache=True, logits_to_keep=1)
            past = out.past_key_values
            next_id = int(out.logits[0, -1].argmax())
        gen = [next_id]
        with timer("decode") as t_dec:
            while next_id not in self.stop_ids and len(gen) < max_new_tokens:
                out = self.model(
                    input_ids=torch.tensor([[next_id]], device=self.device), past_key_values=past, use_cache=True
                )
                past = out.past_key_values
                next_id = int(out.logits[0, -1].argmax())
                gen.append(next_id)
        stopped = gen[-1] in self.stop_ids
        if stopped:
            gen = gen[:-1]
        return gen, {"prefill_s": t_pre.elapsed, "decode_s": t_dec.elapsed, "stopped": stopped}
