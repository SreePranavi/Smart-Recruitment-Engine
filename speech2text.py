import speech_recognition as sr
from transformers import BartForConditionalGeneration, BartTokenizer


class Speech2TextEngine:

    def __init__(self, audio_file):
        self.recognizer = sr.Recognizer()
        self.audio_file = audio_file

    def conversion(self):
        
        # Reading Audio file as source - listening to the audio file and store in audio_text variable

        with sr.AudioFile(self.audio_file) as source:
            audio_text = self.recognizer.listen(source)

        try:
            # using google speech recognition
            transcript = self.recognizer.recognize_google(audio_text)
            print('Converting audio transcripts into text')
            print(transcript)
            return transcript
        except Exception as e:
            print(e)

class TextSummarizationEngine:

    def __init__(self, transcript):
        self.transcript = transcript
        self.model_name="bart-large-cnn"

    def load_model(self):

        # Initializing the tokenizer & model

        self.tokenizer = BartTokenizer.from_pretrained(self.model_name)
        self.model = BartForConditionalGeneration.from_pretrained(self.model_name, forced_bos_token_id=0)

    def summarize(self):
        input_text = self.transcript

        # Tokenize and Generate Questions
        batch = self.tokenizer(input_text, return_tensors="pt")
        generated_ids = self.model.generate(batch['input_ids'])

        # Decode and Print Questions
        summary = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        print("Generated Summary:", summary)


conversion_engine= Speech2TextEngine('harvard.wav')
transcript = conversion_engine.conversion()

summarization_engine = TextSummarizationEngine(transcript)
summarization_engine.load_model()
summarization_engine.summarize()