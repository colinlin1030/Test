from flask import Flask, render_template, request, redirect, url_for, session, jsonify, abort
import os, sqlite3
from functools import wraps
from contextlib import closing
import json


# --------------------
# 基本設定
# --------------------
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "change_this_to_a_random_secure_key")  # 正式環境請改
DB_PATH = os.getenv("DATABASE_PATH", "insmedic.db")

# --------------------
# DB helpers (SQLite)
# --------------------
def get_db():
    """
    取得 SQLite 連線；開啟外鍵；回傳 Row（可用欄位名存取）
    每次使用後請關閉（這裡用 with closing(...) 或在 execute()/query() 內自動關）
    """
    conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def query_all(sql, params=()):
    with closing(get_db()) as conn, closing(conn.cursor()) as cur:
        cur.execute(sql, params)
        return cur.fetchall()

def query_one(sql, params=()):
    with closing(get_db()) as conn, closing(conn.cursor()) as cur:
        cur.execute(sql, params)
        return cur.fetchone()

def execute(sql, params=()):
    with closing(get_db()) as conn, closing(conn.cursor()) as cur:
        cur.execute(sql, params)
        conn.commit()
        return cur.lastrowid

def executemany(sql, seq_of_params):
    with closing(get_db()) as conn, closing(conn.cursor()) as cur:
        cur.executemany(sql, seq_of_params)
        conn.commit()

# --------------------
# Auth decorator
# --------------------
def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)
    return wrapped

# --------------------
# Routes
# --------------------
@app.route("/", methods=["GET"])
@login_required
def index():
    username = session.get("user")
    return render_template("index.html", username=username)

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return render_template("login.html")

    data = request.get_json(silent=True) or {}
    username = (data.get("username") or request.form.get("username") or "").strip()
    password = (data.get("password") or request.form.get("password") or "")

    # 這裡先用簡單驗證；未來可改成查 DB 的 user 表
    if username and password:
        session["user"] = username
        return jsonify({"ok": True, "username": username}), 200

    return jsonify({"ok": False, "error": "INVALID_CREDENTIALS"}), 401

@app.route("/logout", methods=["GET"])
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

# 你的頁面
@app.route("/main", methods=["GET"])
@login_required
def main_page():
    return render_template("main.html")

@app.route("/checkout", methods=["GET"])
@login_required
def checkout():
    return render_template("main.html")

@app.route("/settlement", methods=["GET"])
@login_required
def settlement():
    return render_template("settlement.html")

# --------------------
# Prices：改為讀 DB 的 Product / ProductVariant / (可選)VariantPriceOverride
# --------------------
def _fetch_effective_prices(location_id=None, event_id=None):
    """
    取得「有效售價」列表：
    - 以 product_variant.base_price 為基礎
    - 若有符合時間區間的 variant_price_override，則覆蓋（可依 location_id 或 event_id）
    - 回傳欄位：variant_id, sku, product_name, variant_attrs, effective_price, currency
    """
    # 若你已建立 VIEW variant_effective_price，可改成直接查 VIEW；這裡寫成可攜式 SQL
    sql = """
    SELECT
      pv.id               AS variant_id,
      pv.sku              AS sku,
      p.name              AS product_name,
      COALESCE(pv.attributes, '') AS variant_attrs,
      COALESCE(ov.price, pv.base_price) AS effective_price,
      COALESCE(ov.currency, pv.currency) AS currency
    FROM product_variant pv
    JOIN product p ON p.id = pv.product_id
    LEFT JOIN variant_price_override ov
      ON ov.variant_id = pv.id
     AND (ov.valid_from <= CURRENT_TIMESTAMP)
     AND (ov.valid_to   IS NULL OR ov.valid_to >= CURRENT_TIMESTAMP)
     AND (
            (? IS NOT NULL AND ov.location_id = ?)
         OR (? IS NOT NULL AND ov.event_id = ?)
         OR (? IS NULL AND ? IS NULL AND ov.location_id IS NULL AND ov.event_id IS NULL)
     )
    WHERE pv.status = 'active' AND p.status = 'active'
    ORDER BY p.name, pv.sku;
    """
    # 為了簡化綁參，重複帶入（SQLite 不支援同名參數自動重用）
    params = (
        location_id, location_id,
        event_id, event_id,
        location_id, event_id
    )
    rows = query_all(sql, params)

    # 將 JSON 文字的 attributes（若有）解成 dict；失敗就給空字串
    out = []
    import json
    for r in rows:
        attrs = r["variant_attrs"]
        try:
            attrs_obj = json.loads(attrs) if attrs else {}
        except Exception:
            attrs_obj = {}
        out.append({
            "variant_id": r["variant_id"],
            "sku": r["sku"],
            "product_name": r["product_name"],
            "attributes": attrs_obj,
            "price": float(r["effective_price"] or 0),
            "currency": r["currency"],
        })
    return out

@app.route("/prices", methods=["GET", "POST"])
@login_required
def prices():
    """
    GET：
      可帶 ?location_id= 或 ?event_id=
      回傳 variant 的有效售價（考慮 override）
    POST：
      更新 base price（管理用途）
      - 支援 JSON：{"items":[{"variant_id":1,"price":12345.0}, ...]}
      - 或 form：variant_id, price（多筆可重複傳）
    """
    if request.method == "GET":
        # 讓你在賽會頁面可帶 location_id / event_id 看臨時價
        location_id = request.args.get("location_id", type=int)
        event_id = request.args.get("event_id", type=int)
        data = _fetch_effective_prices(location_id=location_id, event_id=event_id)
        return jsonify(data)

    # POST：更新 base_price
    payload = request.get_json(silent=True) or {}
    items = payload.get("items")

    # 若不是 JSON，就從 form 讀單筆或多筆
    if not items:
        form_variant_id = request.form.getlist("variant_id")
        form_price = request.form.getlist("price")
        if form_variant_id and form_price and len(form_variant_id) == len(form_price):
            items = []
            for vid, p in zip(form_variant_id, form_price):
                try:
                    items.append({"variant_id": int(vid), "price": float(p)})
                except Exception:
                    continue

    if not items:
        return jsonify({"ok": False, "error": "NO_ITEMS"}), 400

    # 執行更新
    params = [(float(it["price"]), int(it["variant_id"])) for it in items if "variant_id" in it and "price" in it]
    if not params:
        return jsonify({"ok": False, "error": "INVALID_ITEMS"}), 400

    executemany("UPDATE product_variant SET base_price = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?;", params)
    return jsonify({"ok": True, "updated": len(params)}), 200

@app.route("/prices/admin", methods=["GET"])
@login_required
def prices_admin():
    # 直接載入目前的有效售價（不帶 location/event → 顯示 base 或全域 override）
    rows = _fetch_effective_prices()
    return render_template("prices_admin.html", rows=rows)

# --------------------
# （可選）Inventory API：給之後 main.html 有需要時調用
# --------------------
@app.route("/inventory", methods=["GET"])
@login_required
def inventory():
    """
    依地點查庫存：?location_id=1
    不帶參數時回傳每個 variant 在各地點的結存彙總
    """
    location_id = request.args.get("location_id", type=int)
    if location_id:
        sql = """
        SELECT i.variant_id, pv.sku, p.name AS product_name,
               i.qty_on_hand, i.qty_reserved, i.location_id
        FROM inventory i
        JOIN product_variant pv ON pv.id = i.variant_id
        JOIN product p ON p.id = pv.product_id
        WHERE i.location_id = ?
        ORDER BY p.name, pv.sku;
        """
        rows = query_all(sql, (location_id,))
    else:
        sql = """
        SELECT i.location_id, l.name AS location_name,
               i.variant_id, pv.sku, p.name AS product_name,
               i.qty_on_hand, i.qty_reserved
        FROM inventory i
        JOIN location l ON l.id = i.location_id
        JOIN product_variant pv ON pv.id = i.variant_id
        JOIN product p ON p.id = pv.product_id
        ORDER BY l.name, p.name, pv.sku;
        """
        rows = query_all(sql)

    data = [{
        "location_id": r["location_id"],
        "location_name": r.get("location_name"),
        "variant_id": r["variant_id"],
        "sku": r["sku"],
        "product_name": r["product_name"],
        "qty_on_hand": float(r["qty_on_hand"] or 0),
        "qty_reserved": float(r["qty_reserved"] or 0),
    } for r in rows]
    return jsonify(data)

# --------------------
# 健康檢查
# --------------------
@app.route("/ping")
def ping():
    # 順便檢查 DB 是否能連線
    try:
        _ = query_one("SELECT 1 AS ok;")
        return "ok"
    except Exception as e:
        return f"db-error: {e}", 500

# --------------------
# 啟動
# --------------------
if __name__ == "__main__":
    # 在 Render/雲端可改為 debug=False
    # 若 DB 檔不存在，提醒一下（避免 .tables 爆 "file is not a database"）
    if not os.path.exists(DB_PATH):
        print(f"[WARN] Database file not found: {DB_PATH}")
        print("請先建立：sqlite3 insmedic.db \".read schema.sql\"")
    app.run(debug=True)

@app.route("/products", methods=["POST"])
@login_required
def create_product_and_variant():
    """
    接收 JSON：
    {
      "name": "...",            # 產品名稱（必填）
      "brand": "INSMEDIC",      # 選填
      "sku": "OLY-BLK-5M",      # 變體 SKU（必填、唯一）
      "attributes": {...},      # 選填（dict 會存成 JSON TEXT）
      "base_price": 50000,      # 必填
      "currency": "KRW"         # 預設 KRW
    }
    建立 product 與 product_variant；若同名產品已存在，可直接重用（此處採「同名新建」策略以簡化）
    """
    payload = request.get_json(silent=True) or {}
    name = (payload.get("name") or "").strip()
    brand = (payload.get("brand") or "").strip() or None
    sku = (payload.get("sku") or "").strip()
    attributes = payload.get("attributes")
    base_price = payload.get("base_price")
    currency = (payload.get("currency") or "KRW").strip() or "KRW"

    if not name or not sku or base_price is None:
        return jsonify({"ok": False, "error": "REQUIRED_FIELDS"}), 400

    # 屬性需為 JSON 可序列化
    attrs_text = None
    if attributes is not None:
        try:
            attrs_text = json.dumps(attributes, ensure_ascii=False)
        except Exception:
            return jsonify({"ok": False, "error": "ATTRIBUTES_NOT_JSON"}), 400

    # 建立 product
    product_id = execute(
        "INSERT INTO product (name, brand, status) VALUES (?, ?, 'active');",
        (name, brand)
    )

    # 建立 variant
    try:
        variant_id = execute(
            """
            INSERT INTO product_variant
              (product_id, sku, attributes, base_price, currency, status)
            VALUES
              (?, ?, ?, ?, ?, 'active');
            """,
            (product_id, sku, attrs_text, float(base_price), currency)
        )
    except Exception as e:
        # 例如 SKU 重複
        return jsonify({"ok": False, "error": f"VARIANT_INSERT_FAIL: {e}"}), 400

    return jsonify({"ok": True, "product_id": product_id, "variant_id": variant_id}), 200
