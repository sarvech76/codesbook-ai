import sqlite3

class Database:
    def __init__(self):
        self.DB_FILE = "codesBook.db"
        self.code_table = "codeSnippets"
        self.embeddings_table = "codeEmbeddings"

        conn = sqlite3.connect(self.DB_FILE)
        cursor = conn.cursor()

        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.code_table} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code TEXT,
                description TEXT,
                keywords TEXT
            )
        """)
        cursor.execute(f""" 
            CREATE TABLE IF NOT EXISTS {self.embeddings_table} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code_id INTEGER,
                page_content TEXT,
                embeddings BLOB
            )
        """)
        conn.commit()
        conn.close()
