from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import os, json
from functools import wraps

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "change_this_to_a_random_secure_key")  # 正式環境請改

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
            # 未登入 → 去 login.html（GET /login 會回模板）
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)
    return wrapped

# ---- routes ----
@app.route("/", methods=["GET"])
@login_required
def index():
    # 新版首頁：左上 logo、右上使用者與 로그아웃、中央 A/B/C/D
    username = session.get("user")
    return render_template("index.html", username=username)

@app.route("/login", methods=["GET", "POST"])
def login():
    """
    GET: 回 login.html（你的前端會用 AJAX 送出）
    POST: 支援 JSON 或 form。成功 → 200 JSON；失敗 → 401 JSON
    """
    if request.method == "GET":
        return render_template("login.html")

    # 嘗試讀 JSON；若不是 JSON 再讀 form
    data = request.get_json(silent=True) or {}
    username = (data.get("username") or request.form.get("username") or "").strip()
    password = (data.get("password") or request.form.get("password") or "")

    # 這裡先簡單驗證（你可改成查 DB/帳密驗證）
    if username and password:
        session["user"] = username
        # 讓前端決定導去哪（你的 login.html 會導回 index.html）
        return jsonify({"ok": True, "username": username}), 200

    return jsonify({"ok": False, "error": "INVALID_CREDENTIALS"}), 401

@app.route("/logout", methods=["GET"])
def logout():
    session.pop("user", None)
    # 按照你的需求：登出回 login.html
    return redirect(url_for("login"))

# main.html（C 按鈕導向的頁面）
@app.route("/main", methods=["GET"])
@login_required
def main_page():
    return render_template("main.html")

# 舊的 checkout 保留（若不再使用可移除或 redirect 到 /main）
@app.route("/checkout", methods=["GET"])
@login_required
def checkout():
    return render_template("main.html")

@app.route("/prices", methods=["GET", "POST"])
@login_required
def prices():
    if request.method == "GET":
        return jsonify(load_prices())

    # POST：由管理頁送進來
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

@app.route("/prices/admin", methods=["GET"])
@login_required
def prices_admin():
    return render_template("prices_admin.html", rows=load_prices())

@app.route("/settlement", methods=["GET"])
@login_required
def settlement():
    return render_template("settlement.html")

# 健康檢查
@app.route("/ping")
def ping():
    return "ok"

if __name__ == "__main__":
    # 在 Render/雲端上可改為 debug=False
    app.run(debug=True)
