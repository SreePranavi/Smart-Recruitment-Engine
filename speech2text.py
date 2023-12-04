import speech_recognition as sr
from transformers import BartForConditionalGeneration, BartTokenizer

class Speech2TextEngine:

    def __init__(self):
        self.recognizer = sr.Recognizer()

    def conversion(self, audio_file):
        
        # Reading Audio file as source - listening to the audio file and store in audio_text variable

        with sr.AudioFile(audio_file) as source:
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

    def __init__(self):
        self.model_name="bart-large-cnn"

    def load_model(self):

        # Initializing the tokenizer & model

        self.tokenizer = BartTokenizer.from_pretrained(self.model_name)
        self.model = BartForConditionalGeneration.from_pretrained(self.model_name, forced_bos_token_id=0)

    def summarize(self, transcript):
        input_text = transcript

        # Tokenize and Generate Questions
        batch = self.tokenizer(input_text, return_tensors="pt")
        generated_ids = self.model.generate(batch['input_ids'])

        # Decode and Print Questions
        summary = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        print("Generated Summary:", summary)
