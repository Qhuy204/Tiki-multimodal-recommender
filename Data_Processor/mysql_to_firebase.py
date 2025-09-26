import mysql.connector
import firebase_admin
from firebase_admin import credentials, firestore
import json
from decimal import Decimal
import datetime

def normalize_value(value):
    if isinstance(value, Decimal):
        # Nếu là số nguyên thì ép về int
        if value == value.to_integral_value():
            return int(value)
        # Ngược lại ép về float
        return float(value)
    if isinstance(value, datetime.date) and not isinstance(value, datetime.datetime):
        return datetime.datetime.combine(value, datetime.time())  # convert date -> datetime
    if isinstance(value, datetime.datetime):
        return value  # Firestore sẽ lưu thành Timestamp
    return value

def normalize_dict(d):
    return {k: normalize_value(v) for k, v in d.items()}

def default_serializer(o):
    if isinstance(o, (datetime.datetime, datetime.date)):
        return o.isoformat()  # convert datetime -> "YYYY-MM-DDTHH:MM:SS"
    if isinstance(o, Decimal):
        return float(o)       # hoặc str(o) nếu muốn giữ chính xác tuyệt đối
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

# ======================
# Cấu hình MySQL
# ======================
mysql_config = {
    "host": "localhost",
    "user": "root",
    "password": "Qhuy204",
    "database": "computerstore"
}

# ======================
# Kết nối Firebase
# ======================
cred = credentials.Certificate(r"D:\VSCode\DA3\serviceAccountKey.json")  # file json tải từ Firebase console
firebase_admin.initialize_app(cred)
db = firestore.client()

# ======================
# Hàm export & import
# ======================
def export_and_import():
    conn = mysql.connector.connect(**mysql_config)
    cursor = conn.cursor(dictionary=True)

    # lấy danh sách bảng
    cursor.execute("SHOW FULL TABLES WHERE Table_type = 'BASE TABLE'")
    tables = [row[f"Tables_in_{mysql_config['database']}"] for row in cursor.fetchall()]

    total_docs = 0

    for table in tables:
        print(f"\n📤 Đang xử lý bảng: {table}")
        cursor.execute(f"SELECT * FROM {table}")
        rows = cursor.fetchall()

        # Export ra file JSON (backup tùy chọn)
        with open(f"{table}.json", "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2, default=default_serializer)

        # Import vào Firestore
        for row in rows:
            # Nếu có cột id/xxx_id thì dùng làm docId
            doc_id = None
            for key in row.keys():
                if key.lower().endswith("id"):
                    doc_id = str(row[key])
                    break

            if doc_id:
                db.collection(table).document(doc_id).set(normalize_dict(row))

            else:
                db.collection(table).add(row)

        print(f"✅ Đã import {len(rows)} dòng từ bảng {table} vào Firestore")
        total_docs += len(rows)

    cursor.close()
    conn.close()

    print(f"\n🎉 Hoàn tất chuyển {len(tables)} bảng, tổng cộng {total_docs} dòng MySQL → Firestore")



# ======================
# Main
# ======================
if __name__ == "__main__":
    export_and_import()
