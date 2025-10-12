from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import os, sqlite3, re, uuid, tempfile, json
from functools import wraps
from contextlib import closing
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename
from datetime import datetime, date, timedelta

# 第三方
from ultralytics import YOLO
import holidays
import numpy as np
import cv2

try:
    import easyocr
except Exception:
    easyocr = None  # 延後在 get_ocr_reader() 檢查，給出清楚錯誤

# --------------------
# 基本設定
# --------------------
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "change_this_to_a_random_secure_key")  # 正式環境請改
DB_PATH = os.getenv("DATABASE_PATH", "insmedic.db")

# YOLO 模型（啟動時載入一次）
YOLO_WEIGHTS = os.getenv("YOLO_WEIGHTS", "models/best.pt")
try:
    model = YOLO(YOLO_WEIGHTS)
except Exception as _e:
    model = None
    print(f"[WARN] 無法載入 YOLO 權重 {YOLO_WEIGHTS}: {_e}")

# 上傳限制/格式
MAX_CONTENT_LENGTH_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH_MB * 1024 * 1024
ALLOWED_IMAGE_EXT = {"jpg", "jpeg", "png", "webp", "bmp"}

# 共同尺寸表（公司順序）
SIZE_LIST = [210,220,230,235,240,245,250,255,260,265,270,275,280,285,290,300,310]

# --------------------
# DB helpers (SQLite)
# --------------------
def get_db():
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
# Schema / Migration（僅保留你原有需求）
# --------------------
def run_bootstrap():
    with get_db() as conn:
        cur = conn.cursor()

        cur.execute("""
        CREATE TABLE IF NOT EXISTS leave_types (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          code TEXT UNIQUE NOT NULL,
          name TEXT NOT NULL,
          annual_allowance REAL NOT NULL DEFAULT 0
        );
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS leave_balances (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          username TEXT NOT NULL,
          leave_type_id INTEGER NOT NULL,
          year INTEGER NOT NULL,
          used REAL NOT NULL DEFAULT 0,
          earned REAL NOT NULL DEFAULT 0,
          UNIQUE(username, leave_type_id, year),
          FOREIGN KEY(leave_type_id) REFERENCES leave_types(id) ON DELETE CASCADE
        );
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS business_trips (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          username TEXT NOT NULL,
          country TEXT NOT NULL,
          region TEXT,
          city TEXT,
          start_date TEXT NOT NULL,
          end_date TEXT NOT NULL,
          days REAL NOT NULL,
          purpose TEXT,
          status TEXT NOT NULL CHECK (status IN ('pending','approved','rejected','cancelled')) DEFAULT 'pending',
          reviewer_username TEXT,
          reviewed_at TEXT,
          created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        """)

        cur.execute("SELECT id, code FROM leave_types WHERE code='COMP';")
        if not cur.fetchone():
            cur.execute("INSERT INTO leave_types(code, name, annual_allowance) VALUES ('COMP', '補休 / Comp Time', 0);")

        conn.commit()

run_bootstrap()

# --------------------
# 審計（容錯）
# --------------------
def log_action(actor_username, action, target_username=None):
    try:
        execute(
            "INSERT INTO audit_logs (actor_username, action, target_username) VALUES (?, ?, ?);",
            (actor_username, action, target_username)
        )
    except Exception:
        pass

# --------------------
# Auth / RBAC
# --------------------
def login_required(role=None):
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
# 出差與補休
# --------------------
def calc_inclusive_days(start_str: str, end_str: str) -> int:
    s = datetime.strptime(start_str, "%Y-%m-%d").date()
    e = datetime.strptime(end_str, "%Y-%m-%d").date()
    if e < s:
        return -1
    return (e - s).days + 1

def get_or_create_leave_type(code: str, name: str = None, allowance: float = 0.0):
    row = query_one("SELECT id FROM leave_types WHERE code=?;", (code,))
    if row:
        return row["id"]
    name = name or code
    return execute("INSERT INTO leave_types(code, name, annual_allowance) VALUES (?,?,?);",
                   (code, name, float(allowance)))

def get_or_create_balance(username: str, leave_type_id: int, year: int):
    row = query_one("""
      SELECT * FROM leave_balances WHERE username=? AND leave_type_id=? AND year=?;
    """, (username, leave_type_id, year))
    if row:
        return row
    execute("""
      INSERT OR IGNORE INTO leave_balances(username, leave_type_id, year, used, earned)
      VALUES (?,?,?,?,0);
    """, (username, leave_type_id, year, 0))
    return query_one("""
      SELECT * FROM leave_balances WHERE username=? AND leave_type_id=? AND year=?;
    """, (username, leave_type_id, year))

def compute_trip_comp_days(country_code: str, start: date, end: date):
    years = list(range(start.year, end.year + 1))
    try:
        hcal = holidays.country_holidays(country_code, years=years)
    except Exception:
        hcal = set()

    comp = 0
    take_dates = []
    cur = start
    while cur <= end:
        is_weekend = (cur.weekday() >= 5)
        is_nat = (cur in hcal)
        if is_weekend or is_nat:
            comp += 1
            take_dates.append(cur.isoformat())
        cur += timedelta(days=1)
    return comp, take_dates

# --------------------
# 價格
# --------------------
def _fetch_effective_prices(location_id=None, event_id=None):
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

# --------------------
# 真模型（OCR + 顏色）
# --------------------
_OCR_READER = None  # 延遲初始化單例

def get_ocr_reader():
    """
    第一次用到時才建立 EasyOCR Reader。
    語言碼：'ch_tra'（繁中）、'ch_sim'（簡中）、'en'、'ko'。
    """
    global _OCR_READER
    if _OCR_READER is None:
        if easyocr is None:
            raise RuntimeError("EasyOCR 未安裝。請先 `pip install easyocr`")
        _OCR_READER = easyocr.Reader(['ch_tra', 'ch_sim', 'en', 'ko'], gpu=False)
    return _OCR_READER

def _ocr_texts_from_img(img_bgr):
    """回傳 (大寫合併字串, 大寫片段清單)；輸入為 BGR 影像（np.ndarray）。"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    reader = get_ocr_reader()
    results = reader.readtext(gray)
    texts = [t[1] for t in results]
    blob = " ".join(texts).upper()
    return blob, [t.upper() for t in texts]

def _guess_color_by_hsv(img_bgr):
    """OCR 抓不到顏色時，用 HSV 粗判 Black / Silver / Orange。"""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mean_s = float(np.mean(s))
    mean_v = float(np.mean(v))

    if mean_v < 60:
        return "Black", 0.6
    if mean_s < 40 and mean_v > 170:
        return "Silver", 0.65
    mask_orange = cv2.inRange(hsv, (10, 40, 40), (25, 255, 255))
    ratio_orange = float(np.count_nonzero(mask_orange)) / (img_bgr.shape[0]*img_bgr.shape[1] + 1e-6)
    if ratio_orange > 0.08:
        return "Orange", 0.6
    if mean_s < 50:
        return "Silver", 0.5
    return "Black", 0.5

def _pick_size_from_text(blob):
    for sz in SIZE_LIST:
        if f"{sz}" in blob:
            return str(sz), 0.9
    cand = [int(x) for x in re.findall(r"\b(\d{2,3})\b", blob) if x.isdigit()]
    if cand:
        nearest = min(SIZE_LIST, key=lambda s: min(abs(s - c) for c in cand))
        return str(nearest), 0.6
    return "", 0.0

def infer_crop(crop_bgr):
    """
    僅針對單一裁切區做品牌/顏色/尺碼推論。
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return {"brand":"", "color":"", "size":"", "confidence":0.0}

    # 文字
    try:
        blob, parts = _ocr_texts_from_img(crop_bgr)
    except Exception as e:
        msg = str(e)
        if "Chinese_tra" in msg:
            msg += " ；請改用 ['ch_tra','en'] 或移除不支援的語言碼。"
        print("[ocr error]", msg)
        blob, parts = "", []

    # 品牌
    brand = ""
    if "INSMEDIC" in blob or re.search(r"\bINS\b", blob):
        brand = "INSMEDIC"
    elif re.search(r"\bAIR\b", blob):
        brand = "AIR"

    # 顏色（文字）
    color = ""
    if re.search(r"\bBLACK|BLK|BK\b", blob):
        color = "Black"
    elif re.search(r"\bORANGE|ORG\b", blob):
        color = "Orange"
    elif re.search(r"\bSILVER|SIL\b", blob):
        color = "Silver"

    # 尺碼
    size, size_conf = _pick_size_from_text(blob)

    # 若顏色抓不到 → HSV 粗判
    color_conf_extra = 0.0
    if not color:
        color, color_conf_extra = _guess_color_by_hsv(crop_bgr)

    # 信心分數
    conf = 0.35
    if brand: conf += 0.3
    if color: conf += 0.25
    if size:  conf += min(0.25, size_conf)
    conf += color_conf_extra * 0.3
    conf = max(0.0, min(0.98, conf))

    return {
        "brand": brand,
        "color": color,
        "size": size,
        "confidence": float(round(conf, 3))
    }

def infer_image(save_path: str):
    """
    後備：針對整張圖做一次推論。
    """
    data = np.fromfile(save_path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        img = cv2.imread(save_path)
    if img is None:
        return {"brand":"", "color":"", "size":"", "confidence":0.0}
    return infer_crop(img)

# === 簡易多目標候選區（無 YOLO 的備援） ===
def find_candidate_regions(img_path, max_regions=12):
    data = np.fromfile(img_path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        img = cv2.imread(img_path)
    if img is None:
        return [], None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(gray, 60, 160)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = img.shape[:2]
    areas = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area < (H * W * 0.005):   # 過小略過
            continue
        ar = w / (h + 1e-6)
        if not (0.6 <= ar <= 4.0):  # 過窄/過高略過
            continue
        areas.append((area, (x, y, x + w, y + h)))

    areas.sort(reverse=True, key=lambda t: t[0])
    boxes = [b for _, b in areas[:max_regions]]
    return boxes, img

# === YOLO 取得候選框（優先） ===
def yolo_candidate_regions(img_path, conf_thres=0.3, max_regions=20, class_filter=None):
    """
    使用 YOLO 權重先找可能的鞋盒標籤/LOGO 區域。
    - class_filter: 若你的 best.pt 有特定類別代表「標籤」，可傳入類別 id 集合過濾；None = 全部保留。
    回傳：(boxes, img_bgr)
    boxes：[(x1,y1,x2,y2), ...]（int）
    """
    if model is None:
        return [], None

    # 以檔案路徑直接推論（避免大圖多次 decode）
    try:
        results = model.predict(source=img_path, conf=conf_thres, verbose=False)
    except Exception as e:
        print("[yolo error]", e)
        return [], None

    if not results:
        return [], None

    # 讀圖（供後續裁切）
    data = np.fromfile(img_path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        img = cv2.imread(img_path)
    if img is None:
        return [], None

    res = results[0]
    if not hasattr(res, "boxes") or res.boxes is None:
        return [], img

    xyxy = res.boxes.xyxy.cpu().numpy() if hasattr(res.boxes, "xyxy") else np.zeros((0,4))
    confs = res.boxes.conf.cpu().numpy() if hasattr(res.boxes, "conf") else np.zeros((0,))
    clses = res.boxes.cls.cpu().numpy().astype(int) if hasattr(res.boxes, "cls") else np.zeros((0,), dtype=int)

    H, W = img.shape[:2]
    boxes = []
    for i, b in enumerate(xyxy):
        if i < len(confs) and confs[i] < conf_thres:
            continue
        if class_filter is not None and i < len(clses) and clses[i] not in class_filter:
            continue
        x1, y1, x2, y2 = [int(max(0, v)) for v in b]
        x1, y1 = min(x1, W-1), min(y1, H-1)
        x2, y2 = min(x2, W-1), min(y2, H-1)
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append((x1, y1, x2, y2))

    # 依框面積大到小排序，保留前 max_regions
    boxes = sorted(boxes, key=lambda bb: (bb[2]-bb[0])*(bb[3]-bb[1]), reverse=True)[:max_regions]
    return boxes, img

def _allowed_image(filename: str):
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext in ALLOWED_IMAGE_EXT

# --------------------
# Routes（頁面）
# --------------------
@app.route("/", methods=["GET"])
@login_required()
def index():
    username = session.get("username")
    return render_template("index.html", username=username)

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return render_template("login.html")

    is_json = request.is_json or "application/json" in (request.headers.get("Content-Type") or "")
    data = request.get_json(silent=True) or {}
    username = (data.get("username") or request.form.get("username") or "").strip()
    password = (data.get("password") or request.form.get("password") or "")
    next_url = request.args.get("next") or request.form.get("next") or url_for("index")

    if not username or not password:
        if is_json:
            return jsonify({"ok": False, "error": "INVALID_CREDENTIALS"}), 401
        return render_template("login.html", error="帳號或密碼不正確"), 401

    row = query_one("SELECT username, password_hash, role, is_active FROM users WHERE username=?;", (username,))
    if (not row) or (not row["is_active"]) or (not check_password_hash(row["password_hash"], password)):
        if is_json:
            return jsonify({"ok": False, "error": "INVALID_CREDENTIALS"}), 401
        return render_template("login.html", error="帳號或密碼不正確"), 401

    session["username"] = row["username"]
    session["role"] = row["role"]
    log_action(row["username"], "login")

    if is_json:
        return jsonify({"ok": True, "username": row["username"], "role": row["role"]}), 200
    else:
        try:
            if not next_url.startswith("/"):
                next_url = url_for("index")
        except Exception:
            next_url = url_for("index")
        return redirect(next_url)

@app.route("/logout", methods=["GET"])
def logout():
    actor = session.get("username")
    session.pop("username", None)
    session.pop("role", None)
    if actor:
        log_action(actor, "logout")
    return redirect(url_for("login"))

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

# === 新增：掃描頁 ===
@app.route("/scan", methods=["GET"])
@login_required()
def scan_page():
    return render_template("scan.html")

# --------------------
# Prices API
# --------------------
@app.route("/prices", methods=["GET", "POST"])
@login_required()
def prices():
    if request.method == "GET":
        location_id = request.args.get("location_id", type=int)
        event_id = request.args.get("event_id", type=int)
        data = _fetch_effective_prices(location_id=location_id, event_id=event_id)
        return jsonify(data)

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
# Admin Users
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

    execute("UPDATE users SET password_hash=? WHERE id=?", (generate_password_hash(new), uid))
    log_action(session["username"], "admin_reset_password", row["username"])
    return jsonify({"ok": True})

@app.route("/admin/create_user", methods=["POST"])
@login_required(role="admin")
def create_user():
    username = (request.form.get("username") or "").strip()
    password = request.form.get("password") or ""
    role = request.form.get("role") or "user"

    if not username or not password or role not in ("user", "admin"):
        return jsonify({"ok": False, "msg": "資料不完整"}), 400

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
# （可選）Inventory API
# --------------------
@app.route("/inventory", methods=["GET"])
@login_required()
def inventory():
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
# 出差頁/管理/事件
# --------------------
@app.route("/trips", methods=["GET", "POST"])
@login_required()
def trips_page_or_create():
    if request.method == "GET":
        return render_template("trips.html")

    username = session.get("username")
    country = (request.form.get("country") or "KR").upper()
    region = (request.form.get("region") or "").strip() or None
    city = (request.form.get("city") or "").strip() or None
    purpose = (request.form.get("purpose") or "").strip() or None
    start_date = request.form.get("start_date")
    end_date = request.form.get("end_date")

    days = calc_inclusive_days(start_date, end_date)
    if days <= 0:
        return jsonify({"ok": False, "msg": "日期區間不正確"}), 400

    execute("""
        INSERT INTO business_trips(username, country, region, city, start_date, end_date, days, purpose, status)
        VALUES (?,?,?,?,?,?,?,?,'pending');
    """, (username, country, region, city, start_date, end_date, float(days), purpose))
    log_action(username, "create_trip")
    return jsonify({"ok": True})

@app.route("/manage/trips", methods=["GET"])
@login_required(role="admin")
def manage_trips():
    rows = query_all("SELECT * FROM business_trips ORDER BY created_at DESC;")
    return render_template("manage_trips.html", rows=rows)

@app.route("/manage/trips/<int:trip_id>/<action>", methods=["POST"])
@login_required(role="admin")
def review_trip(trip_id, action):
    if action not in ("approve", "reject", "cancel"):
        return jsonify({"ok": False, "msg": "非法操作"}), 400

    trip = query_one("SELECT * FROM business_trips WHERE id=?;", (trip_id,))
    if not trip:
        return jsonify({"ok": False, "msg": "找不到申請"}), 404

    reviewer = session.get("username")
    new_status = {"approve": "approved", "reject": "rejected", "cancel": "cancelled"}[action]
    execute("""
      UPDATE business_trips SET status=?, reviewer_username=?, reviewed_at=datetime('now') WHERE id=?;
    """, (new_status, reviewer, trip_id))

    if new_status == "approved":
        s = datetime.strptime(trip["start_date"], "%Y-%m-%d").date()
        e = datetime.strptime(trip["end_date"], "%Y-%m-%d").date()
        comp_days, _ = compute_trip_comp_days(trip["country"], s, e)
        comp_id = get_or_create_leave_type("COMP", "補休 / Comp Time", 0)
        bal = get_or_create_balance(trip["username"], comp_id, s.year)
        execute("UPDATE leave_balances SET earned = earned + ? WHERE id=?;", (float(comp_days), bal["id"]))
        log_action(reviewer, "approve_trip_and_credit_comp", trip["username"])

    return jsonify({"ok": True})

@app.route("/api/trips", methods=["GET"])
@login_required()
def api_trips():
    mine = request.args.get("mine") == "1"
    username = session.get("username")
    if mine:
        rows = query_all("SELECT * FROM business_trips WHERE username=? ORDER BY start_date ASC;", (username,))
    else:
        rows = query_all("SELECT * FROM business_trips ORDER BY start_date ASC;")
    events = []
    for r in rows:
        end_plus = (datetime.strptime(r["end_date"], "%Y-%m-%d") + timedelta(days=1)).date().isoformat()
        title = f"{r['username']} 出差（{r['city'] or r['region'] or r['country']}）"
        events.append({
            "id": r["id"],
            "title": title,
            "start": r["start_date"],
            "end": end_plus,
            "allDay": True,
            "extendedProps": {
                "status": r["status"],
                "country": r["country"],
                "city": r["city"],
                "region": r["region"],
                "purpose": r["purpose"]
            }
        })
    return jsonify(events)

@app.route("/api/holidays", methods=["GET"])
@login_required()
def api_holidays():
    code = (request.args.get("country") or "KR").upper()
    year = int(request.args.get("year") or date.today().year)
    try:
        hcal = holidays.country_holidays(code, years=[year])
        items = [{"title": "國定假日", "start": d.isoformat()} for d in hcal.keys()]
    except Exception:
        items = []
    return jsonify(items)

@app.route("/api/comp_balance", methods=["GET"])
@login_required()
def api_comp_balance():
    username = request.args.get("username") or session.get("username")
    year = int(request.args.get("year") or date.today().year)
    comp_id = get_or_create_leave_type("COMP", "補休 / Comp Time", 0)
    row = get_or_create_balance(username, comp_id, year)
    lt = query_one("SELECT annual_allowance FROM leave_types WHERE id=?;", (comp_id,))
    annual = float(lt["annual_allowance"] or 0)
    used = float(row["used"] or 0)
    earned = float(row["earned"] or 0)
    remaining = annual + earned - used
    return jsonify({"username": username, "year": year, "annual": annual, "earned": earned, "used": used, "remaining": remaining})

# --------------------
# 掃描 API（YOLO → OCR 主管線）
# --------------------
@app.route("/api/scan", methods=["POST"])
@login_required()
def api_scan():
    files = request.files.getlist("images")
    if not files:
        return jsonify({"items": []})

    items = []
    import base64

    with tempfile.TemporaryDirectory(prefix="scan_") as tmpdir:
        for f in files:
            if not f or not f.filename or not _allowed_image(f.filename):
                continue

            sname = secure_filename(f.filename)
            uid = uuid.uuid4().hex
            save_path = os.path.join(tmpdir, f"{uid}_{sname}")
            try:
                f.save(save_path)
            except Exception as e:
                print("[save error]", e)
                continue

            # === 1) 優先：YOLO 找候選框 ===
            try:
                # 如你的 best.pt 有特定類別對應「標籤」，可設定環境變數 YOLO_LABEL_CLASSES="0,3"
                cls_filter_env = os.getenv("YOLO_LABEL_CLASSES", "").strip()
                cls_filter = None
                if cls_filter_env:
                    cls_filter = set(int(x) for x in cls_filter_env.split(",") if x.strip().isdigit())

                yolo_boxes, full = yolo_candidate_regions(save_path, conf_thres=0.30, max_regions=20, class_filter=cls_filter)
            except Exception as e:
                print("[yolo pipeline error]", e)
                yolo_boxes, full = [], None

            # === 2) 備援：輪廓法 ===
            if (not yolo_boxes) or (full is None):
                try:
                    boxes, full = find_candidate_regions(save_path)
                except Exception as e:
                    print("[fallback find_candidate_regions error]", e)
                    boxes, full = [], None
            else:
                boxes = yolo_boxes

            # === 3) 如果還是沒有候選框 → 整張圖分析一次 ===
            if not boxes or full is None:
                try:
                    pred = infer_image(save_path)
                except Exception as e:
                    print("[infer_image error]", e)
                    pred = {"brand": "", "color": "", "size": "", "confidence": 0.0}

                items.append({
                    "filename": sname,
                    "det_index": 0,
                    "brand": pred.get("brand", ""),
                    "color": pred.get("color", ""),
                    "size": pred.get("size", ""),
                    "confidence": pred.get("confidence", 0),
                    "crop_b64": None
                })
                continue

            # === 4) 對每個候選框做 OCR + 顏色/尺碼 ===
            for i, b in enumerate(boxes):
                x1, y1, x2, y2 = b
                crop = full[y1:y2, x1:x2].copy()

                ok, buf = cv2.imencode(".png", crop)
                crop_b64 = base64.b64encode(buf.tobytes()).decode("ascii") if ok else None

                try:
                    pred = infer_crop(crop)  # ✅ 針對裁切區推論
                except Exception as e:
                    print("[infer_crop error]", e)
                    pred = {"brand": "", "color": "", "size": "", "confidence": 0.0}

                items.append({
                    "filename": sname,
                    "det_index": i,
                    "brand": pred.get("brand", ""),
                    "color": pred.get("color", ""),
                    "size": pred.get("size", ""),
                    "confidence": pred.get("confidence", 0),
                    "crop_b64": crop_b64
                })

    return jsonify({"items": items})

@app.route("/api/scan/commit", methods=["POST"])
@login_required()
def api_scan_commit():
    payload = request.get_json(silent=True) or {}
    items = payload.get("items") or []

    # 這裡僅回傳統計；之後可改為寫入你的庫存表
    summary = {}
    for it in items:
        brand = str(it.get("brand") or "").upper()
        color = str(it.get("color") or "").capitalize()
        size = str(it.get("size") or "")
        key = (brand, color, size)
        summary[key] = summary.get(key, 0) + 1

    total = sum(summary.values())
    return jsonify({
        "status": "ok",
        "saved": total,
        "summary": [
            {"brand": k[0], "color": k[1], "size": k[2], "count": v}
            for k, v in summary.items()
        ]
    })

# --------------------
# 產品＆變體建立（原有）
# --------------------
@app.route("/products", methods=["POST"])
@login_required(role="admin")
def create_product_and_variant():
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
# 啟動
# --------------------
if __name__ == "__main__":
    if not os.path.exists(DB_PATH):
        print(f"[WARN] Database file not found: {DB_PATH}")
        print('請先建立：sqlite3 insmedic.db ".read schema.sql"')
    app.run(debug=True)
