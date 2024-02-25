from database.updaters.Updater import Updater
import sqlite3
import datetime
import pandas as pd

MIN_date = datetime.date(year=2000, month=1, day=1)
today = datetime.date.today()  # ends this year

conn = sqlite3.connect('../valuator.db')
u = Updater()

table_exist = conn.execute("""
SELECT name FROM sqlite_master WHERE type='table' AND name="macro_US"
""").fetchall()

# put the name of macro indicator you wish to add or update
MACRO_NAME = ["CPI", "realGDP", "federalFunds", "consumerSentiment", "smoothedUSRecessionProbabilities"]

df = pd.DataFrame(pd.date_range(start=MIN_date, end=today), columns=['date'])

for name in MACRO_NAME:
    indicator = u.get_macro_indicator(indicator=name)
    indicator = pd.json_normalize(indicator).rename(columns={"value": name})
    indicator['date'] = pd.to_datetime(indicator['date'])
    df = df.merge(indicator, how='left', on='date')
df = df.fillna(method='ffill')

if not table_exist:
    print("Table not found! creating new table...")
    df.to_sql("macro_US", conn, if_exists="replace", index=False)

else:
    macro_df = pd.read_sql("""
    SELECT * FROM macro_US
    """, con=conn)
    macro_df['date'] = pd.to_datetime(macro_df['date'])

    for name in MACRO_NAME:
        if name in macro_df.columns:
            macro_df = macro_df.drop(columns=name)
    macro_df = macro_df.merge(df, how='outer', on='date')
    macro_df = macro_df.fillna(method='ffill')
    macro_df.to_sql("macro_US", conn, if_exists="replace", index=False)
