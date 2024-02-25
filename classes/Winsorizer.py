import pandas as pd
import numpy as np


class Winsorizer:
    def __init__(self):
        self.column_quantiles = []

    def fit(self, dataframe, exclude_columns, q=0.99):
        """
        Compute the quantiles for each column to determine the winsorization limits.
        """
        for col in dataframe.columns:
            if col not in exclude_columns:
                if dataframe[col].min() >= 0:
                    lower_bound = dataframe[col].min()
                    upper_bound = dataframe[col].quantile(q)
                else:
                    lower_bound = dataframe[col].quantile(1 - q)
                    upper_bound = dataframe[col].quantile(q)
                self.column_quantiles.append((col, lower_bound, upper_bound))

    def transform(self, dataframe):
        """
        Apply winsorization to the dataframe based on the computed quantiles.
        """
        for col, lower, upper in self.column_quantiles:
            dataframe[col] = np.clip(dataframe[col], lower, upper)
        return dataframe
