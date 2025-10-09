from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import os, sqlite3
from functools import wraps
from contextlib import closing
import json
from werkzeug.security import check_password_hash, generate_password_hash

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
# 審計紀錄（可選）
# --------------------
def log_action(actor_username, action, target_username=None):
    """
    將關鍵動作寫入 audit_logs（若表不存在會忽略錯誤）
    """
    try:
        execute(
            "INSERT INTO audit_logs (actor_username, action, target_username) VALUES (?, ?, ?);",
            (actor_username, action, target_username)
        )
    except Exception:
        pass

# --------------------
# Auth / RBAC decorators
# --------------------
def login_required(role=None):
    """
    - 登入保護
    - 若指定 role（'admin' / 'user'），則驗證 session 中的角色
    """
    def outer(view):
        @wraps(view)
        def wrapped(*args, **kwargs):
            if "username" not in session:
                return redirect(url_for("login", next=request.path))
            if role and session.get("role") != role:
                return render_template("error.html", msg="權限不足"), 403
            return view(*args, **kwargs)
        return wrapped
    return outer

# --------------------
# Routes
# --------------------
@app.route("/", methods=["GET"])
@login_required()
def index():
    username = session.get("username")
    return render_template("index.html", username=username)

@app.route("/login", methods=["GET", "POST"])
def login():
    """
    - GET: 顯示登入頁
    - POST: 支援 JSON 或 form，驗證 users 表（username, password_hash, role, is_active）
    期望 users 表：
      id, username UNIQUE, password_hash, role IN ('admin','user'), is_active INT
    """
    if request.method == "GET":
        return render_template("login.html")

    data = request.get_json(silent=True) or {}
    username = (data.get("username") or request.form.get("username") or "").strip()
    password = (data.get("password") or request.form.get("password") or "")

    if not username or not password:
        return jsonify({"ok": False, "error": "INVALID_CREDENTIALS"}), 401

    row = query_one("SELECT username, password_hash, role, is_active FROM users WHERE username=?;", (username,))
    if not row or not row["is_active"]:
        return jsonify({"ok": False, "error": "INVALID_CREDENTIALS"}), 401

    if not check_password_hash(row["password_hash"], password):
        return jsonify({"ok": False, "error": "INVALID_CREDENTIALS"}), 401

    # 設定 session
    session["username"] = row["username"]
    session["role"] = row["role"]

    # 寫審計（可選）
    log_action(row["username"], "login")

    # 回傳 JSON（前端會依 role 導頁）
    return jsonify({"ok": True, "username": row["username"], "role": row["role"]}), 200

@app.route("/logout", methods=["GET"])
def logout():
    actor = session.get("username")
    session.pop("username", None)
    session.pop("role", None)
    if actor:
        log_action(actor, "logout")
    return redirect(url_for("login"))

# 頁面
@app.route("/main", methods=["GET"])
@login_required()
def main_page():
    return render_template("main.html")

@app.route("/checkout", methods=["GET"])
@login_required()
def checkout():
    return render_template("main.html")

@app.route("/settlement", methods=["GET"])
@login_required()
def settlement():
    return render_template("settlement.html")

# --------------------
# Prices：讀取 Product / ProductVariant / (可選)VariantPriceOverride
# --------------------
def _fetch_effective_prices(location_id=None, event_id=None):
    """
    取得「有效售價」列表：
    - 以 product_variant.base_price 為基礎
    - 若有符合時間區間的 variant_price_override，則覆蓋
    回傳欄位：variant_id, sku, product_name, attributes, price, currency
    """
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
    params = (
        location_id, location_id,
        event_id, event_id,
        location_id, event_id
    )
    rows = query_all(sql, params)

    out = []
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
@login_required()
def prices():
    """
    GET：依 location_id / event_id 回傳有效售價
    POST：更新 base_price（限管理員）
    """
    if request.method == "GET":
        location_id = request.args.get("location_id", type=int)
        event_id = request.args.get("event_id", type=int)
        data = _fetch_effective_prices(location_id=location_id, event_id=event_id)
        return jsonify(data)

    # POST：更新 base_price（限管理員）
    if session.get("role") != "admin":
        return jsonify({"ok": False, "error": "FORBIDDEN"}), 403

    payload = request.get_json(silent=True) or {}
    items = payload.get("items")

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

    params = [(float(it["price"]), int(it["variant_id"])) for it in items if "variant_id" in it and "price" in it]
    if not params:
        return jsonify({"ok": False, "error": "INVALID_ITEMS"}), 400

    executemany("UPDATE product_variant SET base_price = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?;", params)
    log_action(session["username"], "update_prices", target_username=None)
    return jsonify({"ok": True, "updated": len(params)}), 200

# --------------------
# 管理員使用者管理頁（去除重複定義）
# --------------------
@app.route("/admin")
@login_required(role="admin")
def admin_users():
    conn = get_db()
    users = conn.execute(
        "SELECT id, username, role, is_active, created_at FROM users ORDER BY id"
    ).fetchall()
    conn.close()
    return render_template("admin_users.html", users=users)

# 管理員重設任一使用者密碼（不需舊密碼）
@app.route("/admin/reset_password", methods=["POST"])
@login_required(role="admin")
def admin_reset_password():
    uid = request.form.get("id", type=int)
    new = request.form.get("new") or ""
    if not uid or len(new) < 8:
        return jsonify({"ok": False, "msg": "id와 새 비밀번호(8자 이상)가 필요합니다."}), 400

    row = query_one("SELECT username FROM users WHERE id=?", (uid,))
    if not row:
        return jsonify({"ok": False, "msg": "사용자를 찾을 수 없습니다."}), 404

    execute("UPDATE users SET password_hash=? WHERE id=?",
            (generate_password_hash(new), uid))
    log_action(session["username"], "admin_reset_password", row["username"])
    return jsonify({"ok": True})


@app.route("/admin/create_user", methods=["POST"])
@login_required(role="admin")
def create_user():
    username = (request.form.get("username") or "").strip()
    password = request.form.get("password") or ""
    role = request.form.get("role") or "user"

    if not username or not password or role not in ("user", "admin"):
        return jsonify({"ok": False, "msg": "자료가 불완전합니다."}), 400

    try:
        execute(
            "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?);",
            (username, generate_password_hash(password), role)
        )
        log_action(session["username"], "create_user", username)
        return jsonify({"ok": True})
    except sqlite3.IntegrityError:
        return jsonify({"ok": False, "msg": "이미 존재하는 아이디입니다."}), 400

@app.route("/admin/toggle_active", methods=["POST"])
@login_required(role="admin")
def toggle_active():
    uid = request.form.get("id", type=int)
    if not uid:
        return jsonify({"ok": False, "msg": "id 필요"}), 400

    row = query_one("SELECT username, is_active FROM users WHERE id=?;", (uid,))
    if not row:
        return jsonify({"ok": False, "msg": "사용자를 찾을 수 없습니다."}), 404

    new_val = 0 if row["is_active"] else 1
    execute("UPDATE users SET is_active=? WHERE id=?;", (new_val, uid))
    log_action(session["username"], "toggle_active", row["username"])
    return jsonify({"ok": True, "is_active": new_val})

@app.route("/admin/change_role", methods=["POST"])
@login_required(role="admin")
def change_role():
    uid = request.form.get("id", type=int)
    role = request.form.get("role")
    if not uid or role not in ("user", "admin"):
        return jsonify({"ok": False, "msg": "역할이 올바르지 않습니다."}), 400

    row = query_one("SELECT username FROM users WHERE id=?;", (uid,))
    if not row:
        return jsonify({"ok": False, "msg": "사용자를 찾을 수 없습니다."}), 404

    execute("UPDATE users SET role=? WHERE id=?;", (role, uid))
    log_action(session["username"], "change_role", row["username"])
    return jsonify({"ok": True})

@app.route("/prices/admin", methods=["GET"])
@login_required(role="admin")
def prices_admin():
    rows = _fetch_effective_prices()
    return render_template("prices_admin.html", rows=rows)

# --------------------
# （可選）Inventory API：給之後 main.html 有需要時調用
# --------------------
@app.route("/inventory", methods=["GET"])
@login_required()
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

    data = []
    for r in rows:
        keys = r.keys()
        location_name = r["location_name"] if "location_name" in keys else None
        data.append({
            "location_id": r["location_id"],
            "location_name": location_name,
            "variant_id": r["variant_id"],
            "sku": r["sku"],
            "product_name": r["product_name"],
            "qty_on_hand": float(r["qty_on_hand"] or 0),
            "qty_reserved": float(r["qty_reserved"] or 0),
        })
    return jsonify(data)

# --------------------
# 健康檢查
# --------------------
@app.route("/ping")
def ping():
    try:
        _ = query_one("SELECT 1 AS ok;")
        return "ok"
    except Exception as e:
        return f"db-error: {e}", 500

# --------------------
# 產品＆變體建立（限管理員）
# --------------------
@app.route("/products", methods=["POST"])
@login_required(role="admin")
def create_product_and_variant():
    """
    接收 JSON：
    {
      "name": "...",
      "brand": "INSMEDIC",
      "sku": "OLY-BLK-5M",
      "attributes": {...},
      "base_price": 50000,
      "currency": "KRW"
    }
    建立 product 與 product_variant
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

    attrs_text = None
    if attributes is not None:
        try:
            attrs_text = json.dumps(attributes, ensure_ascii=False)
        except Exception:
            return jsonify({"ok": False, "error": "ATTRIBUTES_NOT_JSON"}), 400

    product_id = execute(
        "INSERT INTO product (name, brand, status) VALUES (?, ?, 'active');",
        (name, brand)
    )

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
        return jsonify({"ok": False, "error": f"VARIANT_INSERT_FAIL: {e}"}), 400

    log_action(session["username"], "create_variant", target_username=None)
    return jsonify({"ok": True, "product_id": product_id, "variant_id": variant_id}), 200

# --------------------
# 啟動
# --------------------
if __name__ == "__main__":
    if not os.path.exists(DB_PATH):
        print(f"[WARN] Database file not found: {DB_PATH}")
        print('請先建立：sqlite3 insmedic.db ".read schema.sql"')
    app.run(debug=True)
