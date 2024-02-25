from database.updaters.Updater import Updater
import sqlite3
import datetime
import pandas as pd

MIN_date = datetime.date(year=2000, month=1, day=1)
today = datetime.date.today()  # ends this year

conn = sqlite3.connect('../valuator.db')
u = Updater()

table_exist = conn.execute("""
SELECT name FROM sqlite_master WHERE type='table' AND name="treasury"
""").fetchall()

# put the name of macro indicator you wish to add or update
df = pd.DataFrame(pd.date_range(start=MIN_date, end=today), columns=['date'])

url = "https://raw.githubusercontent.com/epogrebnyak/data-ust/master/ust.csv"
treasury = pd.read_csv(url, parse_dates=["date"])

# treasury = u.get_treasury(MIN_date, today)
# treasury = pd.json_normalize(treasury)
df = df.merge(treasury, how='left', on='date').fillna(method='ffill')

if not table_exist:
    print("Table not found! creating new table...")
    df.to_sql("treasury", conn, if_exists="replace", index=False)
else:
    existing_df = pd.read_sql("""
    SELECT * FROM macro_US
    """, con=conn)
    existing_df['date'] = pd.to_datetime(existing_df['date'])
    existing_df = pd.concat([existing_df, df], axis=0)
    existing_df = existing_df.drop_duplicates(subset='date')
    existing_df.to_sql("macro_US", conn, if_exists="replace", index=False)
