from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class WordVectorSimilarity(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def calculate_similarities(self, against_vector, candidate_vectors):
        pass

    def sort_similarities(self, against_vector, candidate_vectors):
        sim_series = self.calculate_similarities(
            against_vector, candidate_vectors)
        return sim_series.sort_values(ascending=False).index


class CosineWordVectorSimilarity(WordVectorSimilarity):

    def cosine_similarity(self, v1, array):

        return pd.Series(np.dot(array, v1) / (np.linalg.norm(v1) * np.linalg.norm(array, axis=1)))

    def calculate_similarities(self, against_vector, candidate_vectors):
        candidate_df = pd.DataFrame(candidate_vectors).T
        return self.cosine_similarity(against_vector, candidate_df)


class EuclidWordVectorSimilarity(WordVectorSimilarity):

    def euclidian_similarity(self, v1, array):
        return - ((array - v1)**2).sum(axis=1)

    def calculate_similarities(self, against_vector, candidate_vectors):
        candidate_df = pd.DataFrame(candidate_vectors).T

        return self.euclidian_similarity(against_vector, candidate_df)


if __name__ == '__main__':
    e_w_s = EuclidWordVectorSimilarity()
    c_w_s = CosineWordVectorSimilarity()
    test_arr = np.random.normal(2, 1, size=(3, 1000))
    test_vec = np.array([2, 2, 2])

    print(e_w_s.calculate_similarities(test_vec, test_arr))
    print(c_w_s.calculate_similarities(test_vec, test_arr))
    idx = c_w_s.sort_similarities(test_vec, test_arr)
    print(test_arr[:, idx[0:5]])
