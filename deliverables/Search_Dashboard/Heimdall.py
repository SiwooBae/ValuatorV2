import numpy as np
import pandas as pd
import sys
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import QuantileTransformer
from angle_emb import AnglE
import datetime

sys.path.append('../../')
from classes.Datasets import CompanyProfileDataset, FundamentalsDataset


class Heimdall:
    def __init__(self, activate_embedder=True):
        self.prof = CompanyProfileDataset()
        self.prof.fit_standard_scaler()
        self.prof.apply_standard_scaler()
        self.fund = FundamentalsDataset()

        self.fund.reorder_columns()
        self.fund.adjust_price_w_CPI()
        self.fund.drop_redundant_features()
        self.fund.dropna()
        # self.price = PriceDataset()
        self.most_recent = self.fund.df.reset_index().sort_values(by=['symbol', 'fillingDate']).drop_duplicates(
            subset='symbol', keep='last')
        self.most_recent = self.most_recent[
            self.most_recent['fillingDate'] >= self.most_recent['fillingDate'].max() - datetime.timedelta(days=120)]
        self.most_recent = self.most_recent.set_index(keys=['symbol', 'fillingDate'])

        self.quantile_transformer = QuantileTransformer()
        self.quantile_transformer.fit(self.most_recent)
        # self.embedder = SentenceTransformer('all-mpnet-base-v2')
        if activate_embedder:
            self.embedder = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').cuda()
        else:
            self.embedder = None

    def semantic_search(self, string, country_filter=None, top=10, dim_reducer=None):
        my_idea = self.prof.transform_standard_scaler(self.embedder.encode(string))

        if country_filter is not None:
            filtered_df = self.prof.embeddings[self.prof.embeddings['country'] == country_filter]
        else:
            filtered_df = self.prof.embeddings

        if dim_reducer is not None:
            my_idea = dim_reducer.transform(my_idea)
            search_database = dim_reducer.transform(filtered_df[self.prof.embedding_cols])
        else:
            search_database = filtered_df[self.prof.embedding_cols]

        search_result = cosine_similarity(my_idea, search_database)

        top_indices = np.argsort(search_result[0])[::-1][:top]

        similar_companies = []
        for index in top_indices:
            similar_companies.append({
                "score": search_result[0][index],
                "symbol": filtered_df['Symbol'].iloc[index],
                "company": filtered_df['companyName'].iloc[index]})

        return similar_companies

    def find_peers_of(self, symbol, country=None, top=10, dim_reducer=None):
        query = self.prof.profiles[self.prof.profiles['Symbol'] == symbol]['description'].tolist()
        query = query[0]
        return self.semantic_search(query, country, top=top, dim_reducer=dim_reducer)

    def show_description(self, symbol):
        return self.prof.profiles[self.prof.profiles['Symbol'] == symbol]['description'].tolist()[0]

    def fetch_fundamentals(self, symbols: list, weights: list = None):
        # Assert that symbols and weights have the same length if weights are provided
        if weights is not None:
            assert len(symbols) == len(weights), "Symbols and weights must have the same length"
        else:
            weights = [1 / len(symbols)] * len(symbols)

        # Create a DataFrame for symbols and weights
        weights_df = pd.DataFrame({'symbol': symbols, 'weight': weights})

        # Filter the DataFrame for the given symbols
        selected_df = self.most_recent[self.most_recent.index.levels[0].isin(symbols)]

        # Merge with the weights DataFrame to assign weights
        result_df = selected_df.merge(weights_df, on='symbol')

        return result_df

    def calculate_asset_structure(self, fetched_df: pd.DataFrame):
        averaged_df = (fetched_df.iloc[:, 1:].mul(fetched_df['weight'], axis=1)).sum(axis=0, numeric_only=True)
        averaged_df.drop(columns=self.fund.income_columns + self.fund.cashflow_columns)
        current_assets = ['cashAndCashEquivalents',
                          'shortTermInvestments',
                          'netReceivables',
                          'inventory',
                          'totalCurrentAssets']

        noncurrent_assets = ['propertyPlantEquipmentNet',
                             'goodwill',
                             'intangibleAssets',
                             'longTermInvestments',
                             'taxAssets',
                             'totalNonCurrentAssets']

        current_liabilities = ['accountPayables',
                               'shortTermDebt',
                               'taxPayables',
                               'deferredRevenue',
                               'totalCurrentLiabilities']

        noncurrent_liabilities = ['longTermDebt',
                                  'deferredRevenueNonCurrent',
                                  'deferredTaxLiabilitiesNonCurrent',
                                  'totalNonCurrentLiabilities']

        averaged_df[current_assets] = averaged_df[current_assets] / averaged_df['totalAssets']
        averaged_df[noncurrent_assets] = averaged_df[noncurrent_assets] / averaged_df['totalAssets']
        averaged_df[current_liabilities] = averaged_df[current_liabilities] / averaged_df['totalLiabilities']
        averaged_df[noncurrent_liabilities] = averaged_df[noncurrent_liabilities] / averaged_df['totalLiabilities']
        averaged_df['debtRatio'] = averaged_df['totalLiabilities'] / averaged_df['totalAssets']

        return averaged_df

    def transform_to_quantile(self, query_fundamental_df: pd.DataFrame):
        return self.quantile_transformer.transform(query_fundamental_df)

        # ret = current_assets_pct.to_dict() | noncurrent_assets_pct.to_dict()
