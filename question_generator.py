from transformers import T5ForConditionalGeneration, T5Tokenizer


class QuestionGeneratorEngine:
    model_name = "t5-base-e2e-qg"

    def load_model(self):
        # Initializing the tokenizer & model

        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)

    def generate_questions(self, context):
        # Tokenize and Generate Questions
        input_ids = self.tokenizer(context, return_tensors="pt").input_ids
        questions = self.model.generate(input_ids)

        # Decode and Print Questions
        decoded_questions = self.tokenizer.decode(
            questions[0], skip_special_tokens=True
        )
        print("Generated Question:", decoded_questions)

        return decoded_questions
