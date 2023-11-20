# Use this script to make income statement up-to-date. It should automatically detect
# the latest date and import accordingly.
# CANNOT BE USED WHEN THERE IS NO ENTRY IN THE TABLE.
# By Siwoo Bae

from database.Updater import Updater
import sqlite3
import datetime
import pandas as pd

MIN_YEAR = datetime.date(year=2000, month=1, day=1).year # starts picking data from 2000
MAX_YEAR = datetime.date.today().year # ends this year

conn = sqlite3.connect('valuator.db')
u = Updater()

income_latest = conn.execute(""" SELECT MAX(date(income.date)) FROM income """).fetchall()[0][0]
latest_year = datetime.datetime.strptime(income_latest, '%Y-%M-%d').year
already_existing_keys = pd.read_sql("""SELECT date, symbol FROM income""", conn)

print("updating income statements from", latest_year, "to", MAX_YEAR)

for i in range(latest_year, MAX_YEAR + 1):
    new_income = u.get_income_statement_bulk(i, period='quarter')

    if not new_income.set_index(['date', 'symbol']).index.is_unique:
        print("duplicated entry detected for year " + str(i) + ". Picking the latest...")
        new_income = new_income.drop_duplicates(subset=['date', 'symbol'], keep='last').reset_index(drop=True)

    new_income_key = new_income[['date', 'symbol']]
    merged = new_income_key.merge(already_existing_keys, on=['date', 'symbol'], how='left', indicator=True)
    mask = merged['_merge'] == 'left_only'  # finds data whose keys don't exist in current keys

    new_income = new_income[mask]

    if not new_income.empty:
        new_income.to_sql('income', conn, if_exists="append", index=False)
        already_existing_keys = already_existing_keys.merge(new_income_key, on=['date', 'symbol'], how='outer')
        print("processed", i, "year. Added", new_income.count(), "new rows to the table.")
    else:
        print("processed", i, "year. Nothing was added to the table.")

conn.close()
