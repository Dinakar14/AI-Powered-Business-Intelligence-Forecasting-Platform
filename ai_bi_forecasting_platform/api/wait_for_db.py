import time
from sqlalchemy import create_engine

DATABASE_URL = "mysql+pymysql://root:root@db:3306/bi_platform"

for i in range(10):
    try:
        engine = create_engine(DATABASE_URL)
        conn = engine.connect()
        conn.close()
        print("✅ MySQL is ready")
        break
    except:
        print("⏳ Waiting for MySQL...")
        time.sleep(3)
else:
    raise Exception("❌ MySQL not available")
