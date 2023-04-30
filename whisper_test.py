#Note: You can send either a URL to an audio file
#OR a file from your computer to the API
import requests
url = "https://transcribe.whisperapi.com"
headers = {
'Authorization': 'Bearer T9N91KB7FI16D551IS3D2A49WHXFP3VP'
}
file = {'file': open('2.mp3', 'rb')}
data = {
  "fileType": "mp3", #default is wav
  "diarization": "false",
  #Note: setting this to be true will slow down results.
  #Fewer file types will be accepted when diarization=true
  "numSpeakers": "1",
  #if using diarization, you can inform the model how many speakers you have
  #if no value set, the model figures out numSpeakers automatically!
  # "url": "URL_OF_STORED_AUDIO_FILE", #can't have both a url and file sent!
  "initialPrompt": "",
  #pass in an initial prompt to teach the model phrases
  #Note: initial prompt can negatively affect results, so only use if necessary!
  "language": "de", #if this isn't set, the model will auto detect language,
  "task": "transcribe", #default is transcribe. Other option is "translate"
  #translate will translate speech from language to english
  #"callbackURL": "" #optional-input callback/webhook url to send results to
  #callbackURL must begin with https:// and CANNOT include www. to be valid
  #callbackURL example: https://example.com
  #callbackURL will be called at same time server returns response
}
response = requests.post(url, headers=headers, files=file, data=data)
print(response.text)