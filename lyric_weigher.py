import numpy as np
import pandas as pd
from abc import ABC, abstractmethod


class LyricWeigher(ABC):

    def __init__(self, idx_range=None):
        self.idx_range = idx_range

    @abstractmethod
    def weigh_subset(self, lyrics, *args, **kwargs):
        pass

    def weigh_lyrics(self, lyrics):
        ret = np.zeros(len(lyrics))
        ret[self.idx_range] = self.weigh_subset(lyrics)
        return ret


class EqualLyricWeigher(LyricWeigher):

    def weigh_subset(self, lyrics):

        if not isinstance(lyrics, np.ndarray):
            lyrics = np.array(lyrics)

        subset = lyrics[self.idx_range] if self.idx_range else lyrics

        return np.array([1/len(subset)] * len(subset))


class ConeLyricWeigher(LyricWeigher):

    def __init__(self, idx_range=None, concavity=1):

        super().__init__(idx_range)
        self.concavity = concavity

    def weigh_subset(self, lyrics):
        if not isinstance(lyrics, np.ndarray):
            lyrics = np.array(lyrics)

        subset = lyrics[self.idx_range] if self.idx_range else lyrics
        n_tokens = len(subset)

        weights = np.array([max([(n_tokens - i)/n_tokens, (i + 1)/n_tokens])
                            for i in range(n_tokens)])
        weights = weights ** self.concavity
        weights /= weights.sum()

        return weights


if __name__ == '__main__':
    test_lyrics = ['believe', 'possibility', 'finally', 'happy']

    e_l_w = EqualLyricWeigher()
    c_l_w = ConeLyricWeigher()

    print(e_l_w.weigh_lyrics(test_lyrics))
    print(e_l_w.weigh_lyrics(test_lyrics))

    print(c_l_w.weigh_lyrics(test_lyrics))
    print(c_l_w.weigh_lyrics(test_lyrics))
