import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import lightning as L
import torch

# support running without installing as a package

from generate import generate
from lib.lit_llama import Tokenizer, LLaMA
from lib.lit_llama.lora import lora
from lib.lit_llama.utils import EmptyInitOnDevice, lazy_load, llama_model_lookup
from lib.scripts.prepare_alpaca import generate_prompt


class robert:

    def __init__(self,
                 finetuned_path,
                 pretrained_path,
                 tokenizer_path,
                 quantize: Optional[str] = None,
                 max_new_tokens: int = 100,
                 top_k: int = 200,
                 temperature: float = 0.8):
        '''Inits Robert'''
        # In the init, we want to load the model.
        assert finetuned_path.is_file()
        assert pretrained_path.is_file()
        assert tokenizer_path.is_file()

        fabric = L.Fabric(devices=1)
        dtype = torch.bfloat16 if fabric.device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32

        print("Loading model ...", file=sys.stderr)
        t0 = time.time()
        with lazy_load(pretrained_path) as pretrained_checkpoint, lazy_load(finetuned_path) as finetuned_checkpoint:
            name = llama_model_lookup(pretrained_checkpoint)

            with EmptyInitOnDevice(
                    device=fabric.device, dtype=dtype, quantization_mode=quantize
            ):
                self.model = LLaMA.from_name(name)

            # 1. Load the pretrained weights
            self.model.load_state_dict(pretrained_checkpoint, strict=False)
            # 2. Load the fine-tuned adapter weights
            self.model.load_state_dict(finetuned_checkpoint, strict=False)

        self.model.eval()
        self.model = fabric.setup_module(self.model)

        self.tokenizer = Tokenizer(tokenizer_path)
        print(f"Time to load model: {time.time() - t0:.02f} seconds.", file=sys.stderr)

    def get_response(self, prompt):
        '''Takes in a prompt and returns and answer from robert'''
        sample = {"instruction": prompt, "input": ""} # If we want input, add it here
        prompt = generate_prompt(sample)
        encoded = self.tokenizer.encode(prompt, bos=True, eos=False, device=self.model.device)
        prompt_length = encoded.size(0)

        t0 = time.perf_counter()
        y = generate(
            self.model,
            idx=encoded,
            max_seq_length=max_new_tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_id=self.tokenizer.eos_id
        )
        t = time.perf_counter() - t0

        output = self.tokenizer.decode(y)
        output = output.split("### Response:")[1].strip()
        return output


def test():
    robert = robert()
    print(robert.get_response("Hi, how are you?"))


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(test)
