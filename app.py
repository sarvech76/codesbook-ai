from flask import Flask, request, jsonify
from model import GeminiModel, ModelGeneration
from codeEmbddings import TextEmbeddings
from database import Database

app = Flask(__name__)

# Initialize components
try:
    testEmbeddings = TextEmbeddings()
    model = GeminiModel().configureModel()
    modelGeneration = ModelGeneration()
    db = Database()
    
    # Load existing vectorstore
    vectorstore = testEmbeddings.load_vectorstore_from_sqlite()
    print("Application initialized successfully")
    
except Exception as e:
    print(f"Error initializing application: {e}")
    testEmbeddings = None
    model = None
    modelGeneration = None
    db = None
    vectorstore = None

@app.route("/add_code", methods=["POST"])
def add_code():
    global vectorstore
    try:
        if not testEmbeddings:
            return jsonify({"error": "Application not properly initialized"}), 500
            
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        code = data.get("code")
        description = data.get("description")
        keywords = data.get("keywords")

        if not all([code, description, keywords]):
            return jsonify({"error": "Missing required fields: code, description, keywords"}), 400

        if not isinstance(keywords, list):
            return jsonify({"error": "Keywords must be a list"}), 400

        vectorstore = testEmbeddings.add_and_index(code, description, keywords, vectorstore)
        return jsonify({"message": "Code snippet saved and indexed successfully."}), 200

    except Exception as e:
        print("Error in /add_code:", e)
        return jsonify({"error": str(e)}), 500

@app.route("/search", methods=["POST"])
def search():
    global vectorstore
    try:
        if not testEmbeddings:
            return jsonify({"error": "Application not properly initialized"}), 500
            
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        query = data.get("query")
        threshold = float(data.get("threshold", 0.80))

        if not query:
            return jsonify({"error": "Missing required field: query"}), 400

        if not isinstance(query, str) or len(query.strip()) == 0:
            return jsonify({"error": "Query must be a non-empty string"}), 400

        result, error = testEmbeddings.search_similar_code(query, threshold, vectorstore)

        if error:
            return jsonify({"error": error}), 404

        return jsonify(result), 200

    except Exception as e:
        print("Error in /search:", e)
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "message": "Code embeddings service is running"}), 200

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(port=8081)