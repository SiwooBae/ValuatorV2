import numpy as np
import pandas as pd
import sys
from sklearn.metrics.pairwise import cosine_similarity
from angle_emb import AnglE

sys.path.append('../../')
from classes.Datasets import CompanyProfileDataset


class Heimdall:
    def __init__(self):
        self.prof = CompanyProfileDataset()
        self.prof.fit_standard_scaler()
        self.prof.apply_standard_scaler()
        # self.fund = FundamentalsDataset()
        # self.price = PriceDataset()

        # self.embedder = SentenceTransformer('all-mpnet-base-v2')
        self.embedder = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').cuda()

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
