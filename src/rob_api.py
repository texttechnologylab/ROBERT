# The api for the chatbot Rob
import json
from flask import Flask, jsonify, request, Response
from rob import rob
from speech5_tts import speech5_tts
from whisper_ai import whisper_ai
import os

# Init the Flask api
app = Flask(__name__)
rob = rob()
speech5 = speech5_tts()
whisper_ai = whisper_ai()


@app.route('/rob/chat', methods=['POST'])
def chat():
    try:
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "input.wav")
        request.files['input.wav'].save(save_path)
        # json_data = request.get_json(force=True)
        # message = json_data['Message']
        # print(message)

        # Get whisper to extract the speech from the wav
        message = whisper_ai.speech_to_text()
        print("Transcribed message: " + message)

        # lets ask rob the message
        answer = rob.chat(message)
        print("Robs answer: " + answer)

        # Text to speech the answer
        speech_base64 = speech5.text_to_speech(answer)

        # Build the response model
        result = {
            "answer": answer,
            "speech_base64": speech_base64
        }
        # return jsonify(result)
        r = Response(response=json.dumps(result), status=200, mimetype="application/json")
        r.headers["Content-Type"] = "text/json; charset=utf-8"
        return r
    except Exception as ex:
        print(ex)
        return 'ERROR'


if __name__ == "__main__":
    print("Init rob...")
    rob.init()
    print("Done!")

    print("Init speech5_tts...")
    speech5.init()
    print("Done!")

    print("Init whisper")
    whisper_ai.init()
    print("Done!")
    # print("Asking rob: Your name is Rob.")
    # print("Rob: " + rob.chat("Your name is Rob."))
    print("Started the Rob api server!")
    app.run()
