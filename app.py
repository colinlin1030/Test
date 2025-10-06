from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import os, json
from functools import wraps

app = Flask(__name__)
app.secret_key = "change_this_to_a_random_secure_key"  # 正式環境請改

PRICE_FILE = "prices.json"

# ---- helpers ----
def load_prices():
    if not os.path.exists(PRICE_FILE):
        data = [{"id": i, "name": f"產品{i}", "price": 50000} for i in range(1, 11)]
        with open(PRICE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    with open(PRICE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_prices(rows):
    with open(PRICE_FILE, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)
    return wrapped

# ---- routes ----
@app.route("/")
def index():
    # 首頁：只有一個「登入」按鈕，導向 /login
    return render_template("index.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        if email and password:
            session["user"] = email
            nxt = request.args.get("next") or url_for("checkout")
            return redirect(nxt)
        error = "請輸入帳號與密碼"
    return render_template("login.html", error=error)

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

@app.route("/checkout")
@login_required
def checkout():
    # 主結算畫面（main.html 會透過 fetch('/prices') 取得單價）
    return render_template("main.html")

@app.route("/prices", methods=["GET", "POST"])
@login_required
def prices():
    if request.method == "GET":
        return jsonify(load_prices())

    # POST 由管理頁送進來
    rows = []
    for i in range(1, 11):
        name = request.form.get(f"name_{i}") or f"產品{i}"
        price_str = request.form.get(f"price_{i}") or "50000"
        try:
            price = int(price_str)
        except ValueError:
            price = 50000
        rows.append({"id": i, "name": name, "price": price})
    save_prices(rows)
    return redirect(url_for("prices_admin"))

@app.route("/prices/admin")
@login_required
def prices_admin():
    return render_template("prices_admin.html", rows=load_prices())

# 健康檢查
@app.route("/ping")
def ping():
    return "ok"

if __name__ == "__main__":
    app.run(debug=True)

@app.route("/settlement")
def settlement():
    # 如果你要限制登入才能看，也可以加上 @login_required
    return render_template("settlement.html")
