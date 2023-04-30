# See: https://github.com/openai/whisper
import whisper


class whisper_ai:
    def init(self):
        self.model = whisper.load_model("base")

    def speech_to_text(self):
        # load audio and pad/trim it to fit 30 seconds
        audio = whisper.load_audio("input.wav")
        audio = whisper.pad_or_trim(audio)

        # make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(audio).to(self.model.device)

        # detect the spoken language
        _, probs = self.model.detect_language(mel)
        print(f"Detected language: {max(probs, key=probs.get)}")

        # decode the audio
        options = whisper.DecodingOptions(fp16=False)
        result = whisper.decode(self.model, mel, options)

        # print the recognized text
        return result.text


# jax 70x schneller
# text to speech vllt rausnehmen für Latenz Minimierung
# Andere Sprachmodellgrößen
# Bessere PCs