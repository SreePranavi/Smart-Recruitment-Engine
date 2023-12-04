from transformers import T5ForConditionalGeneration, T5Tokenizer

class QuestionGeneratorEngine:

    model_name="t5-base-e2e-qg"

    def load_model(self):

        # Initializing the tokenizer & model

        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)

    def test(self):
        input_text = "Harry Potter is a series of seven fantasy novels written by British author J. K. Rowling. The novels chronicle the lives of a young wizard, Harry Potter, and his friends Hermione Granger and Ron Weasley, all of whom are students at Hogwarts School of Witchcraft and Wizardry. The main story arc concerns Harry's struggle against Lord Voldemort, a dark wizard who intends to become immortal, overthrow the wizard governing body known as the Ministry of Magic and subjugate all wizards and Muggles(non-magical people).</s>"

        # Tokenize and Generate Questions
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids
        questions = self.model.generate(input_ids)

        # Decode and Print Questions
        decoded_questions = self.tokenizer.decode(questions[0], skip_special_tokens=True)
        print("Generated Question:", decoded_questions)


engine = QuestionGeneratorEngine()
engine.load_model()
engine.test()
