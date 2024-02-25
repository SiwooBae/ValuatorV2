import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap
import plotly.express as px
import sqlite3
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

ABS_PATH = 'C:/Users/Siwoo/PycharmProjects/ValuatorV2'


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
