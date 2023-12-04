from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

corpus = {
    "SDE1_Tooling": "Python , Javascript , Kubernetes",
    "SDE1_MARLEY": "Python , Javascript , PHP , Kubernetes",
}


class CosineSimilarityEngine:
    def __init__(self):
        self.training_vector = []
        self.training_matrix = []

        self.similarity_map = {}

    def vectorization(self, data):
        # TF-IDF vectorizer

        vectorizer = TfidfVectorizer()
        fitted_vector = vectorizer.fit(data)
        reference_vector = vectorizer.transform(data)

        return fitted_vector, reference_vector

    def train_model(self):
        self.roles = list(corpus.keys())
        self.skills = list(corpus.values())

        self.training_vector, self.training_matrix = self.vectorization(self.skills)

    def cosine_similarity_engine(self, data):
        transformed_data = self.training_vector.transform(data)
        cosine_similarity_matrix = cosine_similarity(
            transformed_data, self.training_matrix
        )

        return cosine_similarity_matrix

    def result(self, data):
        similarity_matrix = self.cosine_similarity_engine(data)

        for i in range(len(similarity_matrix[0])):
            self.similarity_map[self.roles[i]] = similarity_matrix[0][i]

        keys = list(self.similarity_map.keys())
        values = list(self.similarity_map.values())
        sorted_value_index = np.argsort(values)[::-1]
        sorted_dict = {keys[i]: values[i] for i in sorted_value_index}
        return sorted_dict
