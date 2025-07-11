import re
import numpy as np
import sqlite3
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from sklearn.metrics.pairwise import cosine_similarity
from model import GeminiModel, ModelGeneration
import pickle

class TextEmbeddings:
    def __init__(self):
        self.model = GeminiModel().configureModel()
        self.modelGeneration = ModelGeneration()
        self.embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.db_file = "codesBook.db"

    def normalize_code(self, code):
        return re.sub(r"\s+", " ", code.strip())

    def calculate_cosine_similarity(self, vec1, vec2):
        vec1 = np.array(vec1).reshape(1, -1)
        vec2 = np.array(vec2).reshape(1, -1)
        return cosine_similarity(vec1, vec2)[0][0]

    def add_and_index(self, code_text, description, keywords, vectorstore=None):
        normalized_code = self.normalize_code(code_text)
        combined = f"{description}. Keywords: {', '.join(keywords)}. Code: {normalized_code}"
        embedding = self.embedding_model.embed_query(combined)

        # Save to SQLite
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO codeSnippets (code, description, keywords) VALUES (?, ?, ?)
        """, (code_text, description, ','.join(keywords)))

        code_id = cursor.lastrowid
        cursor.execute("""
            INSERT INTO codeEmbeddings (code_id, page_content, embeddings) VALUES (?, ?, ?)
        """, (code_id, combined, pickle.dumps(embedding)))
        conn.commit()
        conn.close()

        if vectorstore:
            doc = Document(page_content=combined)
            vectorstore.add_documents([doc])
        else:
            doc = Document(page_content=combined)
            vectorstore = FAISS.from_documents([doc], self.embedding_model)
        return vectorstore

    def load_vectorstore_from_sqlite(self):
        docs = []
        embeddings = []

        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        cursor.execute("SELECT page_content, embeddings FROM codeEmbeddings")
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return None

        for page_content, embedding_blob in rows:
            docs.append(Document(page_content=page_content))
            embeddings.append(pickle.loads(embedding_blob))

        embeddings_np = np.array(embeddings).astype(np.float32)
        
        dummy_doc = Document(page_content="dummy")
        vectorstore = FAISS.from_documents([dummy_doc], self.embedding_model)
        
        vectorstore.docstore._dict.clear()
        vectorstore.index_to_docstore_id.clear()
        
        for i, doc in enumerate(docs):
            doc_id = str(i)
            vectorstore.docstore.add({doc_id: doc})
            vectorstore.index_to_docstore_id[i] = doc_id
        
        vectorstore.index.reset()
        vectorstore.index.add(embeddings_np)
        
        return vectorstore

    def search_similar_code(self, query, threshold=0.80, vectorstore=None):
        if not vectorstore:
            vectorstore = self.load_vectorstore_from_sqlite()
            if not vectorstore:
                explanation = self.modelGeneration.generate_explanation(self.model, query)
                return {
                    "similarity_score": round(0, 2),
                    "gemini_explanation": explanation
                }, None

        query_embedding = self.embedding_model.embed_query(query)

        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        cursor.execute("SELECT page_content, embeddings FROM codeEmbeddings")
        rows = cursor.fetchall()
        conn.close()

        best_doc = None
        best_score = -1

        for page_content, embedding_blob in rows:
            try:
                doc_embedding = pickle.loads(embedding_blob)
                sim_score = self.calculate_cosine_similarity(query_embedding, doc_embedding)
                if sim_score > best_score:
                    best_score = sim_score
                    best_doc = page_content
            except Exception as e:
                print("Error during similarity check:", e)

        if best_score >= threshold:
            explanation = self.modelGeneration.generate_explanation(self.model, query, best_doc)
            return {
                "matched_code": best_doc,
                "similarity_score": round(best_score, 2),
                "gemini_explanation": explanation
            }, None
        else:
            explanation = self.modelGeneration.generate_explanation(self.model, query)
            return {
                "similarity_score": round(best_score, 2),
                "gemini_explanation": explanation
            }, None