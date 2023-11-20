# Use this script to make balance sheet up-to-date. It should automatically detect
# the latest date and import accordingly.
# CANNOT BE USED WHEN THERE IS NO ENTRY IN THE TABLE.
# by Siwoo Bae

from database.Updater import Updater
import sqlite3
import datetime
import pandas as pd

MIN_YEAR = datetime.date(year=2000, month=1, day=1).year
MAX_YEAR = datetime.date.today().year

conn = sqlite3.connect('valuator.db')
u = Updater()

balance_latest = conn.execute(""" SELECT MAX(date(balance.date)) FROM balance """).fetchall()[0][0]
latest_year = datetime.datetime.strptime(balance_latest, '%Y-%M-%d').year
already_existing_keys = pd.read_sql("""SELECT date, symbol FROM balance""", conn)

print("updating balance sheets from", latest_year, "to", MAX_YEAR)

for i in range(latest_year, MAX_YEAR + 1):
    new_balance = u.get_balance_sheet_bulk(i, period='quarter')

    if not new_balance.set_index(['date', 'symbol']).index.is_unique:
        print("duplicated entry detected for year " + str(i) + ". Picking the latest...")
        new_balance = new_balance.drop_duplicates(subset=['date', 'symbol'], keep='last').reset_index(drop=True)

    new_balance_key = new_balance[['date', 'symbol']]
    merged = new_balance_key.merge(already_existing_keys, on=['date', 'symbol'], how='left', indicator=True)
    mask = merged['_merge'] == 'left_only'  # finds data whose keys don't exist in current keys

    new_balance = new_balance[mask]

    if not new_balance.empty:
        new_balance.to_sql('balance', conn, if_exists="append", index=False)
        already_existing_keys = already_existing_keys.merge(new_balance_key, on=['date', 'symbol'], how='outer')
        print("processed", i, "year. Added", len(new_balance.index), "new rows to the table.")
    else:
        print("processed", i, "year. Nothing was added to the table.")

conn.close()
