from database.updaters.Updater import Updater
import sqlite3
import datetime
import pandas as pd
from tqdm import tqdm


conn = sqlite3.connect('../valuator.db')
u = Updater()

MIN_DATE = datetime.date(year=2020, month=1, day=1).year  # starts picking data from 2020
MAX_DATE = datetime.date.today()  # ends this year


def update_news(news_type, insert_chunk_size=100):
    # Check the type of news to update and set the appropriate data retrieval function.
    if news_type == 'general_news':
        get_news = lambda pg: pd.DataFrame.from_dict(u.get_general_news(page=pg))
    elif news_type == 'stock_news':
        get_news = lambda pg: pd.DataFrame.from_dict(u.get_stock_news(page=pg))
    else:
        # Raise an exception if an invalid news type is provided.
        raise Exception("invalid news type")

    existing_keys = pd.read_sql(sql=f"""
        SELECT publishedDate, url FROM {news_type}
        """, con=conn, parse_dates=['publishedDate'])
    # Determine the most recent date of news already in the database.
    most_recent_date = existing_keys['publishedDate'].max()

    # Initialize a list to accumulate news articles in chunks.
    chunk = []
    prev_chunk_keys = None
    for i in tqdm(range(0, 5000)):
        # Append news articles to the chunk.
        chunk.append(get_news(i))
        # Every `insert_chunk_size` iterations, process and insert the chunk into the database.
        if i % insert_chunk_size == 0:
            df = pd.concat(chunk)
            # Filter out articles that are already in the database.
            df = df.drop_duplicates(subset=['url'])
            df = df[~df['url'].isin(existing_keys['url'])]
            if prev_chunk_keys is not None:
                df = df[~df['url'].isin(prev_chunk_keys)]
            # Insert the new articles into the database.
            if not df.empty and df is not None:
                df.to_sql(news_type, con=conn, if_exists='append', index=False)

            # Clear the chunk to start collecting the next set of articles.
            chunk = []
            prev_chunk_keys = df['url']


update_news('stock_news', insert_chunk_size=100)
