import assemblyai as aai
from credentials import API_KEY

aai.settings.api_key = API_KEY

audio_url = "test.dat.unknown"
config = aai.TranscriptionConfig(language_code="tr", speaker_labels=True)
transcriber = aai.Transcriber()

transcript = transcriber.transcribe(
    audio_url,
    config=config
    )
for utterance in transcript.utterances:
    with open("transcription.txt", "a") as file:
        file.write(f"Konuşmacı {utterance.speaker}: {utterance.text} \n")
