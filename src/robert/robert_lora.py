import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import lightning as L
import torch

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate import generate
from lit_llama import Tokenizer, LLaMA
from lit_llama.lora import lora
from lit_llama.utils import EmptyInitOnDevice, lazy_load, llama_model_lookup
# from scripts.prepare_alpaca import generate_prompt


lora_r = 8
lora_alpha = 16
lora_dropout = 0.05


class robert:

    def __init__(self,
                 finetuned_path: Path,
                 pretrained_path: Path,
                 tokenizer_path: Path,
                 quantize: Optional[str] = None,
                 dtype: str = "float32",
                 max_new_tokens: int = 100,
                 top_k: int = 200,
                 temperature: float = 0.8):
        '''Inits Robert'''
        # In the init, we want to load the model.
        assert finetuned_path.is_file()
        assert pretrained_path.is_file()
        assert tokenizer_path.is_file()

        self.max_new_tokens = max_new_tokens
        self.top_k = top_k
        self.temperature = temperature
        self.context = []

        if quantize is not None:
            raise NotImplementedError("Quantization in LoRA is not supported yet")

        fabric = L.Fabric(devices=1)
        dt = getattr(torch, dtype, None)
        if not isinstance(dt, torch.dtype):
            raise ValueError(f"{dtype} is not a valid dtype.")
        dtype = dt

        print("Loading model ...", file=sys.stderr)
        t0 = time.time()

        with lazy_load(pretrained_path) as pretrained_checkpoint, lazy_load(finetuned_path) as finetuned_checkpoint:
            name = llama_model_lookup(pretrained_checkpoint)

            with EmptyInitOnDevice(
                    device=fabric.device, dtype=dtype, quantization_mode=quantize
            ), lora(r=lora_r, alpha=lora_alpha, dropout=lora_dropout, enabled=True):
                self.model = LLaMA.from_name(name)

                # 1. Load the pretrained weights
                self.model.load_state_dict(pretrained_checkpoint, strict=False)
                # 2. Load the fine-tuned lora weights
                self.model.load_state_dict(finetuned_checkpoint, strict=False)

        print(f"Time to load model: {time.time() - t0:.02f} seconds.", file=sys.stderr)

        self.model.eval()
        self.model = fabric.setup_module(self.model)

        self.tokenizer = Tokenizer(tokenizer_path)
        torch.set_float32_matmul_precision("high")

    def generate_prompt(self, message):
        if message["input"]:
            return (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately continues the inputs. If you don't have an answer, excuse yourself.\n\n"
                f"### Instruction:\n{message['instruction']}\n\n### Input:\nDialog so far:\n{message['input']}\n\n### Response:"
            )
        return (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{message['instruction']}\n\n### Response:"
        )

    def get_response(self, message):
        '''Takes in a prompt and returns and answer from robert'''
        # We use the optional input field to store the existing chat
        # and context. We take the last X entries to the context.
        inp = '\n'.join(self.context[-4:])
        sample = {"instruction": message, "input": inp}
        prompt = self.generate_prompt(sample)
        #print("===================== PROMPT =====================")
        #print(prompt)
        #print("===================== END =====================\n")
        encoded = self.tokenizer.encode(prompt, bos=True, eos=False, device=self.model.device)

        t0 = time.perf_counter()
        y = generate(
            self.model,
            idx=encoded,
            max_seq_length=self.max_new_tokens,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            eos_id=self.tokenizer.eos_id
        )
        t = time.perf_counter() - t0

        output = self.tokenizer.decode(y)
        output = output.split("### Response:")[1].strip()
        # Sometimes the output contains a dialog prefix we dont want.
        if output.startswith("Rob:"):
            output = output[len("Rob:"):].strip()
        # Add the output to the context and also the prompt of the user
        self.context.append("Student: " + message)
        self.context.append("Rob: " + output)
        return output


def test(finetuned_path: Path = Path("/storage/projects/R.O.B.E.R.T/robert-models/robert_6k_para_chat/lit-llama-lora-finetuned.pth"),
         pretrained_path: Path = Path("/storage/projects/R.O.B.E.R.T/lit-llama-weights/7B/lit-llama.pth"),
         tokenizer_path: Path = Path("/storage/projects/R.O.B.E.R.T/lit-llama-weights/tokenizer.model"),
         quantize: Optional[str] = None,
         dtype: str = "float32",
         max_new_tokens: int = 100,
         top_k: int = 200,
         temperature: float = 0.8):
    my_robert = robert(finetuned_path,
                       pretrained_path,
                       tokenizer_path,
                       quantize,
                       dtype,
                       max_new_tokens,
                       top_k,
                       temperature)
<<<<<<< HEAD
    print(my_robert.get_response("Hi, how are you?") + "\n\n")
    print(my_robert.get_response("I'm confused. Where are we?") + "\n\n")
    print(my_robert.get_response("Tell me something about this place.") + "\n\n")
    print(my_robert.get_response("Could you tell me more?") + "\n\n")
    print(my_robert.get_response("Is there someone I could talk to?") + "\n\n")
=======
    #print(my_robert.get_response("Hi, how are you?") + "\n\n")
    #print(my_robert.get_response("I'm confused. Where are we?") + "\n\n")
    #print(my_robert.get_response("Tell me something about this place.") + "\n\n")
    #print(my_robert.get_response("Could you tell me more?") + "\n\n")
    #print(my_robert.get_response("Is there someone I could talk to?") + "\n\n")
    res1 = my_robert.get_response("Hi, who are you?")
    print(res1)
    res2 = my_robert.get_response(res1)
    print(res2)
    res3 = my_robert.get_response(res2)
    print(res3)
    res4 = my_robert.get_response(res3)
>>>>>>> f74edef1d37154b2476550874556312d1868ec97

if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(test)
