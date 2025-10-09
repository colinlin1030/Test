-- 建議：每次連線都開啟 FK 檢查
PRAGMA foreign_keys = ON;

-- 共用：自動時間戳（SQLite 沒有觸發器就用 DEFAULT CURRENT_TIMESTAMP）
-- 布林值用 INTEGER(0/1)，JSON 用 TEXT 存放序列化字串

-- 1) Product
CREATE TABLE product (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  spu_code TEXT,
  name TEXT NOT NULL,
  brand TEXT,
  category_id INTEGER,
  description TEXT,
  status TEXT NOT NULL DEFAULT 'active',
  created_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP),
  updated_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP)
);

-- 2) ProductVariant
CREATE TABLE product_variant (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  product_id INTEGER NOT NULL REFERENCES product(id),
  sku TEXT NOT NULL UNIQUE,
  barcode TEXT,
  attributes TEXT,           -- JSON 字串
  base_price NUMERIC NOT NULL DEFAULT 0,
  currency TEXT NOT NULL DEFAULT 'KRW',
  tax_rate NUMERIC DEFAULT 0,
  weight_g NUMERIC,
  length_mm NUMERIC,
  width_mm NUMERIC,
  height_mm NUMERIC,
  status TEXT NOT NULL DEFAULT 'active',
  created_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP),
  updated_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP)
);

-- 4) OnsiteEvent（先建，供 Location 用）
CREATE TABLE onsite_event (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  start_date TEXT,  -- YYYY-MM-DD
  end_date TEXT,    -- YYYY-MM-DD
  venue TEXT,
  city TEXT,
  country TEXT,
  organizer TEXT,
  notes TEXT,
  created_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP),
  updated_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP)
);

-- 3) Location
CREATE TABLE location (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  type TEXT NOT NULL CHECK (type IN ('warehouse','store','onsite_event')),
  name TEXT NOT NULL,
  address TEXT,
  event_id INTEGER REFERENCES onsite_event(id),
  is_active INTEGER NOT NULL DEFAULT 1,  -- 1=true, 0=false
  created_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP),
  updated_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP)
);

-- 5) Inventory（快照）
CREATE TABLE inventory (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  variant_id INTEGER NOT NULL REFERENCES product_variant(id),
  location_id INTEGER NOT NULL REFERENCES location(id),
  qty_on_hand NUMERIC NOT NULL DEFAULT 0,
  qty_reserved NUMERIC NOT NULL DEFAULT 0,
  updated_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP),
  UNIQUE (variant_id, location_id)
);

-- 6) StockMovement（流水）
CREATE TABLE stock_movement (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  variant_id INTEGER NOT NULL REFERENCES product_variant(id),
  from_location_id INTEGER REFERENCES location(id),
  to_location_id INTEGER REFERENCES location(id),
  qty_delta NUMERIC NOT NULL,
  movement_type TEXT NOT NULL CHECK (movement_type IN (
    'sale','return','receive','purchase_return','adjust','transfer_out','transfer_in','reserve','unreserve'
  )),
  reason TEXT,
  reference_type TEXT,   -- 'order','transfer','adjustment','purchase','other'
  reference_id INTEGER,
  unit_cost NUMERIC,
  currency TEXT,
  occurred_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP),
  created_by TEXT,
  created_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP)
);

CREATE INDEX idx_sm_variant_time ON stock_movement(variant_id, occurred_at);
CREATE INDEX idx_sm_to_loc_time ON stock_movement(to_location_id, occurred_at);
CREATE INDEX idx_sm_from_loc_time ON stock_movement(from_location_id, occurred_at);

-- 7) Customer & Address
CREATE TABLE customer (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  type TEXT NOT NULL CHECK (type IN ('individual','organization')),
  name TEXT NOT NULL,
  contact_person TEXT,
  email TEXT,
  phone TEXT,
  tax_id TEXT,
  default_billing_address_id INTEGER,
  default_shipping_address_id INTEGER,
  notes TEXT,
  created_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP),
  updated_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP)
);

CREATE TABLE customer_address (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  customer_id INTEGER NOT NULL REFERENCES customer(id) ON DELETE CASCADE,
  type TEXT NOT NULL CHECK (type IN ('billing','shipping')),
  receiver_name TEXT,
  phone TEXT,
  address_line1 TEXT,
  address_line2 TEXT,
  city TEXT,
  state TEXT,
  postal_code TEXT,
  country TEXT,
  is_default INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP),
  updated_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP)
);

-- 8) Order / OrderLine
CREATE TABLE "order" (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  order_no TEXT NOT NULL UNIQUE,
  customer_id INTEGER REFERENCES customer(id),
  location_id INTEGER REFERENCES location(id),           -- 成交地點（倉/店/賽會攤位）
  event_id INTEGER REFERENCES onsite_event(id),          -- 若屬賽會
  channel TEXT CHECK (channel IN ('pos','web','wholesale','onsite')),
  status TEXT NOT NULL CHECK (status IN ('draft','reserved','confirmed','fulfilled','canceled','returned')),
  order_date TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP),
  due_date TEXT,
  currency TEXT NOT NULL DEFAULT 'KRW',
  subtotal NUMERIC NOT NULL DEFAULT 0,
  discount_total NUMERIC NOT NULL DEFAULT 0,
  tax_total NUMERIC NOT NULL DEFAULT 0,
  shipping_fee NUMERIC NOT NULL DEFAULT 0,
  grand_total NUMERIC NOT NULL DEFAULT 0,
  notes TEXT,
  created_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP),
  updated_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP)
);

CREATE INDEX idx_order_status_date ON "order"(status, order_date);

CREATE TABLE order_line (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  order_id INTEGER NOT NULL REFERENCES "order"(id) ON DELETE CASCADE,
  variant_id INTEGER NOT NULL REFERENCES product_variant(id),
  qty NUMERIC NOT NULL,
  unit_price NUMERIC NOT NULL,
  discount_amount NUMERIC NOT NULL DEFAULT 0,
  tax_rate NUMERIC DEFAULT 0,
  tax_amount NUMERIC NOT NULL DEFAULT 0,
  line_total NUMERIC NOT NULL DEFAULT 0,
  reserved_movement_id INTEGER REFERENCES stock_movement(id),
  fulfilled_movement_id INTEGER REFERENCES stock_movement(id),
  created_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP),
  updated_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP)
);

-- 9) Payment（允許一單多次收款）
CREATE TABLE payment (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  order_id INTEGER NOT NULL REFERENCES "order"(id) ON DELETE CASCADE,
  method TEXT NOT NULL CHECK (method IN ('card','bank','cash','other')),
  provider TEXT,
  status TEXT NOT NULL CHECK (status IN ('authorized','captured','refunded','failed','void')),
  amount NUMERIC NOT NULL,
  currency TEXT NOT NULL DEFAULT 'KRW',
  fx_rate NUMERIC,
  txn_id TEXT,
  paid_at TEXT,
  payer_name TEXT,
  notes TEXT,
  created_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP),
  updated_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP)
);

-- （可選）若要一筆款分配到多張訂單，改用：
-- CREATE TABLE payment_allocation (
--   id INTEGER PRIMARY KEY AUTOINCREMENT,
--   payment_id INTEGER NOT NULL REFERENCES payment(id) ON DELETE CASCADE,
--   order_id INTEGER NOT NULL REFERENCES "order"(id) ON DELETE CASCADE,
--   amount NUMERIC NOT NULL
-- );

-- 10) VariantPriceOverride（地點/賽會臨時價）
CREATE TABLE variant_price_override (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  variant_id INTEGER NOT NULL REFERENCES product_variant(id),
  location_id INTEGER REFERENCES location(id),
  event_id INTEGER REFERENCES onsite_event(id),
  price NUMERIC NOT NULL,
  currency TEXT NOT NULL DEFAULT 'KRW',
  valid_from TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP),
  valid_to TEXT,
  CHECK (location_id IS NOT NULL OR event_id IS NOT NULL)
);

-- 快速查詢：依賽會地點覆蓋價，沒有就回 base_price（示範 VIEW）
CREATE VIEW variant_effective_price AS
SELECT
  v.id AS variant_id,
  COALESCE(ov.price, v.base_price) AS effective_price,
  COALESCE(ov.currency, v.currency) AS currency,
  ov.location_id,
  ov.event_id
FROM product_variant v
LEFT JOIN variant_price_override ov
  ON ov.variant_id = v.id
  AND (ov.valid_from <= CURRENT_TIMESTAMP)
  AND (ov.valid_to IS NULL OR ov.valid_to >= CURRENT_TIMESTAMP);

CREATE TABLE IF NOT EXISTS users (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  username TEXT NOT NULL UNIQUE,
  password_hash TEXT NOT NULL,
  role TEXT NOT NULL CHECK(role IN ('admin','user')),
  is_active INTEGER NOT NULL DEFAULT 1,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 審計紀錄（可選）登入/權限變更/新增使用者等
CREATE TABLE IF NOT EXISTS audit_logs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  actor_username TEXT,
  action TEXT NOT NULL,
  target_username TEXT,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);