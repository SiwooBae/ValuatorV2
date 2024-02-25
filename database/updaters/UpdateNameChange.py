from database.updaters.Updater import Updater
import sqlite3
import datetime
import pandas as pd

conn = sqlite3.connect('../valuator.db')
u = Updater()

name_change = u.get_symbol_changes()
df = pd.json_normalize(name_change)

df.to_sql('symbol_changes', conn, if_exists='replace', index=False)

conn.close()