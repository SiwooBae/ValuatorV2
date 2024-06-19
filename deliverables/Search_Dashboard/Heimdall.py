import numpy as np
import pandas as pd
import sys
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import QuantileTransformer
from angle_emb import AnglE
import datetime
import torch
import re
import torch.nn as nn

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
            self.industry_predictor = torch.jit.load('./industry_predictor.pt')
            self.sector_predictor = torch.jit.load('./sector_predictor.pt')
        else:
            self.embedder = None
            self.industry_predictor = None
            self.sector_predictor = None

    def vectorize_text(self, string):
        return self.prof.transform_standard_scaler(self.embedder.encode(string))

    def semantic_search(self, vector, country_filter=None, top=10, dim_reducer=None):
        my_idea = vector

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
        query = self.show_description(symbol)
        query = self.vectorize_text(query)
        return self.semantic_search(query, country, top=top, dim_reducer=dim_reducer)

    def expression_to_vector(self, expression):
        # Regex to find patterns like "0.7<I want to cure cancer>"
        # tokens = re.findall(r'([+-]?\s*\d*\.\d+)\s*<([^>]+)>', expression)
        tokens = re.findall(r'([+-]?\s*\d*\.?\d*)\s*<([^>]+)>', expression)
        result_vector = np.zeros((1, 1024))  # Assuming vectors are 300-dimensional

        for token in tokens:
            sign_multiplier, term = token
            sign_multiplier = sign_multiplier.replace(' ', '')  # Remove spaces

            # Determine the sign and absolute value of the multiplier
            if sign_multiplier == '':
                sign = 1
                multiplier = 1
            else:
                sign = -1 if '-' in sign_multiplier else 1
                multiplier = float(sign_multiplier.replace('+', '').replace('-', ''))

            # Convert term to vector and apply multiplier
            if self.prof.profiles['Symbol'].str.contains(term).any():  # if it is a symbol
                desc = self.show_description(term)  # fetch its description
                term_vector = self.vectorize_text(desc)  # and vectorize
            else:
                term_vector = self.vectorize_text(term)
            scaled_vector = term_vector * multiplier * sign

            # Add or subtract the vector
            result_vector += scaled_vector

        return result_vector

    def show_description(self, symbol):
        return self.prof.profiles[self.prof.profiles['Symbol'] == symbol]['description'].tolist()[0]

    def predict_sector_proba(self, vector):
        return torch.softmax(self.sector_predictor(torch.Tensor(vector).to('cuda')), dim=-1).detach().cpu().numpy()

    def predict_industry_proba(self, vector):
        torch.softmax(self.industry_predictor(torch.Tensor(vector).to('cuda')), dim=-1).detach().cpu().numpy()

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
