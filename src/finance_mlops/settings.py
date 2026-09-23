import os

DB_URL = os.environ.get("DATABASE_URL", "postgresql://finance:finance@localhost:5432/finance_mlops")
