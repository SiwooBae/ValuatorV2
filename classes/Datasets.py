import pandas as pd
import numpy as np
import datetime
import sqlite3
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

ABS_PATH = 'C:/Users/Siwoo/PycharmProjects/ValuatorV2'


class Winsorizer:
    def __init__(self):
        self.column_quantiles = []
        self.is_fitted = False

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
        self.is_fitted = True

    def transform(self, dataframe):
        if self.is_fitted:
            for col, lower, upper in self.column_quantiles:
                dataframe.loc[:, col] = np.clip(dataframe[col], lower, upper)
            return dataframe
        else:
            raise Exception("cannot transform before fitting.")


class CompanyProfileDataset:
    def __init__(self):
        with sqlite3.connect(ABS_PATH + '/database/valuator.db') as conn:
            self.profiles = pd.read_sql("""
            SELECT 
                Symbol,
                companyName,
                description,
                industry, 
                sector, 
                country, 
                IPOdate, 
                exchangeShortName, 
                isActivelyTrading
            FROM profiles
            WHERE
                isFund = 0 AND isEtf = 0
            GROUP BY companyName
            ORDER BY Symbol ASC
            """, conn, parse_dates='IPOdate')

        self.profiles = self.profiles.dropna(how='any', axis=0)
        self.embeddings = pd.read_csv(ABS_PATH + '/database/embeddings_combined.csv')
        self.embedding_cols = [str(i) for i in range(1024)]

        self.preprocessing_steps = []
        self.scaler = StandardScaler()

    def train_test_split(self, test_ratio):
        train, test = train_test_split(self.embeddings, test_size=test_ratio)
        self.embeddings = {'train': train, 'test': test}
        self.preprocessing_steps.append("train_test_split")

    def fit_standard_scaler(self):
        if "train_test_split" in self.preprocessing_steps:
            self.scaler.fit(self.embeddings['train'][self.embedding_cols])
        else:
            self.scaler.fit(self.embeddings[self.embedding_cols])
        self.preprocessing_steps.append("fit_standard_scaler")

    def apply_standard_scaler(self):
        if "train_test_split" in self.preprocessing_steps:
            for key, value in self.embeddings.items():
                self.embeddings[key][self.embedding_cols] = self.scaler.transform(value[self.embedding_cols])
        else:
            self.embeddings[self.embedding_cols] = self.scaler.transform(self.embeddings[self.embedding_cols])
        self.preprocessing_steps.append("apply_standard_scaler")

    def transform_standard_scaler(self, X):
        return self.scaler.transform(X)


class FundamentalsDataset:
    def __init__(self):
        with sqlite3.connect(ABS_PATH + '/database/valuator.db') as conn:
            self.df = pd.read_sql("""
            SELECT * FROM bic
            WHERE symbol in
                (SELECT symbol FROM profiles
                WHERE country = "US"
                AND isFund = 0
                AND isEtf = 0
                AND currency = "USD"
                AND (exchangeShortName = "NASDAQ" OR exchangeShortName = "NYSE"))
                AND symbol NOT LIKE '%W'
            ORDER BY symbol, date ASC
            """, con=conn, parse_dates=['fillingDate', 'date'])

            macro_df = pd.read_sql("""
            SELECT * FROM macro_US
            """, con=conn, parse_dates='date')

            cursor = conn.execute("""SELECT * FROM balance""")
            self.balance_columns = [description[0] for description in cursor.description]

            cursor = conn.execute("""SELECT * FROM income""")
            self.income_columns = [description[0] for description in cursor.description]

            cursor = conn.execute("""SELECT * FROM cashflow""")
            self.cashflow_columns = [description[0] for description in cursor.description]

        CPI_base = macro_df[macro_df['date'] == macro_df['date'].max()]['CPI'].values
        macro_df['CPI'] = macro_df['CPI'] / CPI_base

        self.df = self.df.merge(macro_df, how="left", on="date")
        self.df = self.df.set_index(['symbol', 'fillingDate'])
        self.df = self.df.drop(columns=['date', 'reportedCurrency'])

        self.winsorizer = Winsorizer()

    def drop_redundant_features(self):
        balance_ignored_features = ['goodwillAndIntangibleAssets',
                                    'totalLiabilitiesAndStockholdersEquity',
                                    'netDebt',
                                    'cashAndShortTermInvestments']

        income_ignored_features = ['grossProfitRatio',
                                   'costAndExpenses',
                                   'EBITDARatio',
                                   'operatingIncomeRatio',
                                   'incomeBeforeTaxRatio',
                                   'netIncomeRatio',
                                   'EPSDiluted',
                                   'weightedAverageShsOut',
                                   'weightedAverageShsOutDil',
                                   'interestIncome']

        cashflow_ignored_features = []  # nothing yet

        macro_ignored_features = ['realGDP',
                                  "consumerSentiment",
                                  "smoothedUSRecessionProbabilities"]

        if self.CPI_adjusted:
            macro_ignored_features += ['CPI']

        total = balance_ignored_features + income_ignored_features + cashflow_ignored_features + macro_ignored_features
        self.df = self.df.drop(columns=total)
        other_dropped_columns = []
        for column in self.df.columns:
            if "other" in str(column).lower() or "1" in str(column).lower():
                other_dropped_columns.append(column)
        self.df = self.df.drop(columns=other_dropped_columns)

    def adjust_price_w_CPI(self):
        self.CPI_adjusted = True
        self.df['federalFunds'] = self.df['federalFunds'] * self.df['CPI']
        self.df = self.df.divide(self.df['CPI'], axis=0)

    def reorder_columns(self):
        """
        Reorders a DataFrame based on a given list of column names.

        :param df: pandas DataFrame
        :param column_order: List of column names to order the DataFrame by
        :return: Reordered DataFrame
        """
        # Remove duplicates while preserving order
        order = self.balance_columns + self.income_columns + self.cashflow_columns
        seen = set()
        unique_column_order = [col for col in order if col not in seen and not seen.add(col)]

        # Filter out columns that are not in the DataFrame
        valid_columns = [col for col in unique_column_order if col in self.df.columns]

        extra_columns = [col for col in self.df.columns if col not in valid_columns]

        # Reorder the DataFrame
        self.df = self.df[valid_columns + extra_columns]

    def apply_TTM(self):
        self.df = self.df.groupby(level=0).rolling(4, min_periods=4).mean().droplevel(0)

    def apply_asset_constraint(self):
        asset_current_mask = np.abs(
            (self.df['totalCurrentAssets'] + self.df['totalNonCurrentAssets']) / self.df['totalAssets']) < 1.001
        asset_total_mask = np.abs(
            (self.df['totalStockholdersEquity'] + self.df['totalLiabilities']) / self.df['totalAssets']) < 1.001
        mask = np.logical_and(asset_total_mask, asset_current_mask)
        self.df = self.df[mask]

    def apply_nonnegative_constraint(self):
        nonnegative_constraint_columns = []
        for column in self.df.columns:
            if self.df[column].min() < 0 and np.sum(self.df[column] < 0) < 0.01 * len(self.df):
                nonnegative_constraint_columns.append(column)

        for column in nonnegative_constraint_columns:
            self.df = self.df[self.df[column] >= 0]

    def divide_by_total_assets(self):
        self.df = self.df[self.df['totalAssets'] > 0]
        unaffected_cols = ['federalFunds', 'totalAssets']
        affected_cols = [col for col in self.df.columns if col not in unaffected_cols]
        self.df[affected_cols] = self.df[affected_cols].div(self.df['totalAssets'], axis=0)
        self.df['totalAssets'] = np.log(self.df['totalAssets'])

    def dropna(self):
        self.df.dropna(axis=0, how='any')

    def get_train_valid(self, val_split_point, train_valid_gap, winsorize=True, q=0.99):
        train_mask = self.df.index.get_level_values('fillingDate') < (
                val_split_point - datetime.timedelta(days=train_valid_gap))
        valid_mask = self.df.index.get_level_values('fillingDate') > val_split_point
        self.train = self.df.loc[train_mask]
        self.valid = self.df.loc[valid_mask]

        if winsorize:
            self.winsorizer.fit(self.train, exclude_columns=['federalFunds'], q=q)
            self.train = self.winsorizer.transform(self.train)
            self.valid = self.winsorizer.transform(self.valid)

        return self.train, self.valid