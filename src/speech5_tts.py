# From https://huggingface.co/microsoft/speecht5_tts
# !pip install git+https://github.com/huggingface/transformers sentencepiece datasets
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf
from datasets import load_dataset
import base64
import io


class speech5_tts:
    fs = 44100

    def init(self):
        '''Sets up all the transformers and models'''
        self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        self.model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
        self.embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        self.speaker_embeddings = torch.tensor(self.embeddings_dataset[7306]["xvector"]).unsqueeze(0)

    def text_to_speech(self, text):
        inputs = self.processor(text=text, return_tensors="pt")
        speech = self.model.generate_speech(inputs["input_ids"],
                                            self.speaker_embeddings,
                                            vocoder=self.vocoder)
        sf.write("speech.wav", speech.numpy(), samplerate=16000)
        return (base64.b64encode(open("speech.wav", "rb").read())).decode("utf-8")

        # Below code does not work. The base64 is corrupted it seems. Otherwise
        # this would be preffered because it doesnt write to disc
        # We dont want to write the speech to IO. Just store it in mem
        file_format = "WAV"
        memory_file = io.BytesIO()
        memory_file.name = "output.wav"
        sf.write(memory_file, speech.numpy(), self.fs, format=file_format)

        memory_file.seek(0)

        # Read from mem, to base64 and return it
        temp_data, temp_sr = sf.read(memory_file)
        return (base64.b64encode(temp_data)).decode("utf-8")
