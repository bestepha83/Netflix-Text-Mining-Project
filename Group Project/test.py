import sqlite3
import pandas as pd

conn = sqlite3.connect('Netflix.db')
cur = conn.cursor()

query_threads_sql = "SELECT * FROM threads;"
query_comments_sql = "SELECT * FROM comments;"

threads_db = pd.read_sql(query_threads_sql, conn)
comments_db = pd.read_sql(query_comments_sql, conn)
print(threads_db)
print(comments_db)