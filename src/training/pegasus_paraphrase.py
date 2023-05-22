# See: https://huggingface.co/tuner007/pegasus_paraphrase
import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

model_name = 'tuner007/pegasus_paraphrase'
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'


class paraphraser:

    def init(self):
        self.tokenizer = PegasusTokenizer.from_pretrained(model_name)
        self.model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

    def get_response(self, input_text, num_return_sequences, num_beams):
        batch = self.tokenizer([input_text],
                               truncation=True,
                               padding='longest',
                               max_length=60,
                               return_tensors="pt").to(torch_device)
        # num_beams is referring to Beam Search Tree. It searches the best
        # output. The number of beams determine the depth.
        translated = self.model.generate(**batch,
                                         max_length=60,
                                         num_beams=num_beams,
                                         num_return_sequences=num_return_sequences,
                                         temperature=1.5)
        text_list = self.tokenizer.batch_decode(translated, skip_special_tokens=True)
        return text_list


if __name__ == "__main__":
    num_beams = 10
    num_return_sequences = 6
    context = "Room A14 is at the end of the hallway."
    paraphraser = paraphraser()
    paraphraser.init()
    print(paraphraser.get_response(context, num_return_sequences, num_beams))
