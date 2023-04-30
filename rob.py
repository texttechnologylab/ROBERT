# DialoGPT
# Currently using https://huggingface.co/microsoft/DialoGPT-medium?text=Hey+my+name+is+Thomas%21+How+are+you%3F
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class rob:
    '''The class representing the rob ai'''
    def init(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/DialoGPT-medium",
            padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/DialoGPT-medium")
        self.step = 0
        self.input_history = []

    def chat(self, message):
        '''Chat with rob by passing in a message'''
        encoded = self.tokenizer.encode(message +
                                        self.tokenizer.eos_token, return_tensors='pt')
        self.input_history = torch.cat([self.chat_history, encoded],
                                       dim=-1) if self.step > 0 else encoded
        self.chat_history = self.model.generate(self.input_history, max_length=1000,
                                                pad_token_id=self.tokenizer.eos_token_id)
        answer = self.tokenizer.decode(
            self.chat_history[:, self.input_history.shape[-1]:][0],
            skip_special_tokens=True)
        self.step += 1
        return answer
