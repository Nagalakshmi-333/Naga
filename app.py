# from flask import Flask, render_template, request, jsonify
# import pdfplumber
# import io
# import re
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from transformers import pipeline

# try:
#     import faiss
# except ImportError:
#     faiss = None

# app = Flask(__name__)

# # Load models once
# embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
# qa_pipe = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# # In-memory storage
# corpus = []
# embeddings = None
# index = None


# def clean_text(text):
#     if not text:
#         return ""
#     text = text.replace("\x00", " ")
#     text = re.sub(r"\s+", " ", text)
#     return text.strip()


# def extract_text_from_pdf(file_bytes):
#     pages = []
#     with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
#         for i, page in enumerate(pdf.pages, start=1):
#             txt = page.extract_text() or ""
#             pages.append((i, clean_text(txt)))
#     return pages


# def chunk_text(text, chunk_size=900, overlap=200):
#     words = text.split()
#     chunks = []
#     start = 0
#     while start < len(words):
#         end = min(start + chunk_size, len(words))
#         chunks.append(" ".join(words[start:end]))
#         if end == len(words):
#             break
#         start = max(end - overlap, 0)
#     return chunks


# def embed_texts(texts):
#     return np.array(embedder.encode(texts, normalize_embeddings=True), dtype="float32")


# def build_faiss_index(embs):
#     dim = embs.shape[1]
#     index = faiss.IndexFlatIP(dim)
#     index.add(embs)
#     return index


# @app.route("/")
# def index_page():
#     return render_template("index.html")


# @app.route("/upload", methods=["POST"])
# def upload():
#     global corpus, embeddings, index
#     files = request.files.getlist("files")
#     corpus = []
#     for f in files:
#         raw = f.read()
#         pages = extract_text_from_pdf(raw)
#         for pno, ptxt in pages:
#             if not ptxt.strip():
#                 continue
#             for ch in chunk_text(ptxt):
#                 corpus.append({"content": ch, "source_file": f.filename, "page": pno})
#     if not corpus:
#         return jsonify({"status": "error", "message": "No text found in PDF."})
#     texts = [c["content"] for c in corpus]
#     embeddings = embed_texts(texts)
#     if faiss:
#         index = build_faiss_index(embeddings)
#     else:
#         index = None
#     return jsonify({"status": "success", "chunks": len(corpus)})


# @app.route("/ask", methods=["POST"])
# def ask():
#     global corpus, embeddings, index
#     data = request.json
#     query = data.get("query", "")
#     if not corpus or index is None:
#         return jsonify({"answer": "No documents indexed yet."})
#     q_vec = embed_texts([query])
#     scores, idxs = index.search(q_vec, 5)
#     retrieved = [corpus[int(i)] for i in idxs[0] if i >= 0]
#     context = "\n".join([r["content"] for r in retrieved])
#     if not context.strip():
#         return jsonify({"answer": "No relevant content found."})
#     result = qa_pipe(question=query, context=context)
#     answer = result.get("answer", "No clear answer found.")
    
#     cites = []
#     for r in retrieved[:3]:
#         if "source_file" in r and "page" in r:
#             cites.append({"file": r["source_file"], "page": r["page"]})

#     return jsonify({"answer": answer, "citations": cites})


# if __name__ == "__main__":
#     app.config["DEBUG"] = True
#     app.run()
              