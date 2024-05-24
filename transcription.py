import assemblyai as aai

aai.settings.api_key = "3307a9a03f4944dc894b4ebea8c6edfa"

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