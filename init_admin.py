# init_admin.py
import sqlite3
from werkzeug.security import generate_password_hash

DB = "insmedic.db"
ADMIN_USER = "admin"
ADMIN_PASS = "change_me_please"   # 建立後可在後台再改

conn = sqlite3.connect(DB)
cur = conn.cursor()
cur.execute("SELECT 1 FROM users WHERE username=?", (ADMIN_USER,))
if cur.fetchone():
    print("Admin already exists.")
else:
    cur.execute(
        "INSERT INTO users (username, password_hash, role) VALUES (?, ?, 'admin')",
        (ADMIN_USER, generate_password_hash(ADMIN_PASS))
    )
    conn.commit()
    print("Admin user created:", ADMIN_USER)
conn.close()
