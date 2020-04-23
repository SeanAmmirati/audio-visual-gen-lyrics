from abc import ABC, abstractmethod


class TopicSelector(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def agg_strategy(self, df):
        pass

    @abstractmethod
    def selection_strategy(self, series):
        pass

    def return_selections(self, df, n=1):
        agg = self.agg_strategy(df)
        selections = self.selection_strategy(agg)
        return selections.index[0:n].tolist()


class MaxMaxSelector(TopicSelector):

    def agg_strategy(self, df):
        return df.max(axis=1)

    def selection_strategy(self, series):
        return series.sort_values(ascending=False)


class MeanMaxSelector(TopicSelector):

    def agg_strategy(self, df):
        return df.mean(axis=1)

    def selection_strategy(self, series):
        return series.sort_values(ascending=False)
