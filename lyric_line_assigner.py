from wordvector_similarity import CosineWordVectorSimilarity, EuclidWordVectorSimilarity
from lyric_weigher import ConeLyricWeigher, EqualLyricWeigher
from topic_selector import MaxMaxSelector, MeanMaxSelector
import numpy as np
import pandas as pd


class LyricLineAssigner:

    def __init__(self, weighing_type='eq', similarity_metric='cosine', topic_selector_type='max_max',
                 weighing_obj=None, similarity_obj=None, topic_selector=None):
        if similarity_obj:
            self.similarity = similarity_obj
        elif similarity_metric == 'cosine':
            self.similarity = CosineWordVectorSimilarity()
        elif similarity_metric == 'euclid':
            self.similarity = EuclidWordVectorSimilarity()

        if weighing_obj:
            self.weight = weighing_obj
        elif weighing_type == 'eq':
            self.weight = EqualLyricWeigher()
        elif weighing_type == 'cone':
            self.weight = ConeLyricWeigher()
        elif weighing_type == 'first':
            self.weight = EqualLyricWeigher(0)
        elif weighing_type == 'last':
            self.weight = EqualLyricWeigher(-1)

        if topic_selector:
            self.topic_selector = topic_selector
        elif topic_selector_type == 'max_max':
            self.topic_selector = MaxMaxSelector()
        elif topic_selector_type == 'mean_max':
            self.topic_selector = MeanMaxSelector()

    def assign_line(self, line, candidate_vectors, n=1):
        weights = self.weight.weigh_lyrics(line)
        similarities = pd.DataFrame([self.similarity.calculate_similarities(
            word, candidate_vectors) for word in line])

        weighted_similarities = weights * similarities.T
        return self.topic_selector.return_selections(weighted_similarities, n)


if __name__ == '__main__':
    lla = LyricLineAssigner()
    test_arr = np.random.normal(2, 1, size=(3, 1000))
    test_vec = [np.array([2, 2, 2]), np.array([3, 4, 3])]
    print(lla.assign_line(test_vec, test_arr, None))

    lla2 = LyricLineAssigner(weighing_type='cone', similarity_metric='euclid')
    print(lla2.assign_line(test_vec, test_arr, 4))
