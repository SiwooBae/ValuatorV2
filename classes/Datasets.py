import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap
import plotly.express as px
import sqlite3


class CompanyProfileDataset:
    def __init__(self):
        with sqlite3.connect('../database/valuator.db') as conn:
            description_df = pd.read_sql("""
            SELECT Symbol, companyName, description, industry, sector, country, IPOdate
            FROM profile_v2
            WHERE
                isFund = 0
                AND isEtf = 0
            GROUP BY companyName
            ORDER BY symbol ASC
            """, conn)
