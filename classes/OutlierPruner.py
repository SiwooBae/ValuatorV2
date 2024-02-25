import pandas as pd


class OutlierPruner:
    def __init__(self):
        self.column_quantiles = []

    def fit(self, dataframe, exclude_columns, q=0.99):
        for col in dataframe.columns:
            if col not in exclude_columns:
                if dataframe[col].min() >= 0:
                    lower_bound = -1
                    upper_bound = dataframe[col].quantile(q)
                else:
                    lower_bound = dataframe[col].quantile(1 - q)
                    upper_bound = dataframe[col].quantile(q)
                self.column_quantiles.append((col, lower_bound, upper_bound))

    def transform(self, dataframe, thresh=0):
        counter = 0
        m_total = pd.Series(0, index=dataframe.index, name="total mask")
        for col, lower, upper in self.column_quantiles:
            m_total += ((dataframe[col] >= lower) & (dataframe[col] <= upper))
            counter += 1
        mask = m_total >= counter - thresh
        return dataframe[mask]
