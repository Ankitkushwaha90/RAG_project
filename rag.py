"""
RAG minimal working example (indexing + retrieval + generation)

Features:
- Embed documents with sentence-transformers
- Index with FAISS (fallback to sklearn)
- Generate answers using Hugging Face transformers (FLAN-T5)
- CLI demo and a tiny Flask API endpoint

Usage:
    pip install -r requirements.txt
    python rag.py --build-corpus sample_docs/      # builds index
    python rag.py --ask "What is RAG?"            # ask question using built index

Author: ChatGPT (example)
"""

import os
import argparse
import json
from typing import List, Tuple
import numpy as np

# Embeddings
from sentence_transformers import SentenceTransformer

# Generation
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Indexing: try faiss, fallback to sklearn
try:
    import faiss
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False
    from sklearn.neighbors import NearestNeighbors

# Simple Flask API (optional)
from flask import Flask, request, jsonify

# ---------- Configuration ----------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # small, fast, accurate for semantic search
GEN_MODEL = "google/flan-t5-small"      # small and CPU-friendly; swap to larger if GPU available
INDEX_DIR = "rag_index"                 # where vectors + docs are stored
DOCS_JSON = os.path.join(INDEX_DIR, "documents.json")
VECTORS_NPY = os.path.join(INDEX_DIR, "vectors.npy")
FAISS_INDEX_FILE = os.path.join(INDEX_DIR, "faiss.index")

# RAG hyperparams
K = 5  # number of documents to retrieve


# ---------- Document Store ----------
class DocumentStore:
    def __init__(self):
        self.docs: List[dict] = []   # each doc: {"id": str, "text": str, "meta": {...}}
        self.embeddings = None

    def add_documents(self, texts: List[str], metas: List[dict] = None):
        if metas is None:
            metas = [{} for _ in texts]
        start_id = len(self.docs)
        for i, (t, m) in enumerate(zip(texts, metas)):
            self.docs.append({"id": str(start_id + i), "text": t, "meta": m})

    def save(self):
        os.makedirs(INDEX_DIR, exist_ok=True)
        with open(DOCS_JSON, "w", encoding="utf-8") as f:
            json.dump(self.docs, f, ensure_ascii=False, indent=2)
        if self.embeddings is not None:
            np.save(VECTORS_NPY, self.embeddings)

    def load(self):
        if os.path.exists(DOCS_JSON):
            with open(DOCS_JSON, "r", encoding="utf-8") as f:
                self.docs = json.load(f)
        if os.path.exists(VECTORS_NPY):
            self.embeddings = np.load(VECTORS_NPY)


# ---------- Indexer ----------
class VectorIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self._use_faiss = _HAS_FAISS
        if self._use_faiss:
            # We'll use IndexFlatIP with normalized vectors for cosine similarity
            self.index = faiss.IndexFlatIP(dim)
        else:
            self._nn = None  # created during fit
            self.index = None

    def fit(self, vectors: np.ndarray):
        # vectors assumed shape (n, dim)
        if self._use_faiss:
            # normalize vectors for cosine similarity using inner product
            faiss.normalize_L2(vectors)
            self.index.add(vectors.astype(np.float32))
            # save index
            faiss.write_index(self.index, FAISS_INDEX_FILE)
        else:
            # sklearn NearestNeighbors with cosine metric
            self._nn = NearestNeighbors(n_neighbors=min(10, len(vectors)), metric="cosine")
            self._nn.fit(vectors)
            # store vectors in memory for later
            self.index = vectors

    def query(self, qvec: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Returns list of (doc_idx, score) sorted by highest similarity
        Score is cosine similarity (higher is better). For sklearn fallback, we convert distance.
        """
        if self._use_faiss:
            q = qvec.copy().astype(np.float32)
            faiss.normalize_L2(q)
            distances, indices = self.index.search(q, top_k)
            # distances are inner product (cosine because we normalized)
            results = []
            for idx, dist in zip(indices[0], distances[0]):
                if idx == -1:
                    continue
                results.append((int(idx), float(dist)))
            return results
        else:
            # sklearn returns distances; convert to similarity = 1 - distance
            distances, indices = self._nn.kneighbors(qvec, n_neighbors=min(top_k, len(self.index)))
            res = []
            for idx, d in zip(indices[0], distances[0]):
                sim = 1.0 - float(d)
                res.append((int(idx), sim))
            return res


# ---------- RAG System ----------
class RAG:
    def __init__(self, embedding_model_name=EMBEDDING_MODEL, gen_model_name=GEN_MODEL, device=-1):
        # device: -1 = cpu, otherwise GPU device id (int)
        print("Loading embedding model:", embedding_model_name)
        self.embedder = SentenceTransformer(embedding_model_name, device="cpu" if device == -1 else f"cuda:{device}")

        print("Loading generator model:", gen_model_name)
        # For seq2seq models like FLAN-T5
        self.tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
        self.gen_model = AutoModelForSeq2SeqLM.from_pretrained(gen_model_name)
        # Use pipeline for simplicity
        self.generator = pipeline("text2text-generation", model=self.gen_model, tokenizer=self.tokenizer, device=0 if device != -1 else -1)

        # placeholders
        self.store = DocumentStore()
        self.index = None

    def build_index(self, documents: List[str], metas: List[dict] = None):
        self.store.add_documents(documents, metas)
        # embed
        print(f"Embedding {len(documents)} documents...")
        vectors = self.embedder.encode(documents, convert_to_numpy=True, show_progress_bar=True)
        self.store.embeddings = vectors.astype(np.float32)
        # build index
        dim = vectors.shape[1]
        self.index = VectorIndex(dim)
        self.index.fit(self.store.embeddings)
        # save docs & vectors
        self.store.save()
        print("Index built and saved to disk.")

    def load_index(self):
        self.store.load()
        if self.store.embeddings is None:
            raise RuntimeError("No embeddings found on disk. Build the index first.")
        dim = self.store.embeddings.shape[1]
        self.index = VectorIndex(dim)
        self.index.fit(self.store.embeddings)
        print("Index loaded from disk. Documents:", len(self.store.docs))

    def retrieve(self, query: str, k: int = K):
        qvec = self.embedder.encode([query], convert_to_numpy=True)
        results = self.index.query(qvec, top_k=k)
        docs = []
        for idx, score in results:
            doc = self.store.docs[idx]
            docs.append({"id": doc["id"], "text": doc["text"], "score": score, "meta": doc.get("meta", {})})
        return docs

    def generate(self, question: str, retrieved_docs: List[dict], max_length=256, temperature=0.1):
        # Combine retrieved docs into a context string (short stacking)
        # You may want to do more sophisticated chunking / prompt engineering
        context_texts = "\n\n---\n\n".join([f"[{d['id']}] {d['text']}" for d in retrieved_docs])
        prompt = (
            "You are an assistant that answers questions using the provided context.\n"
            "If the answer is not contained in the context, say 'I don't know.'\n\n"
            "Context:\n"
            f"{context_texts}\n\n"
            "Question:\n"
            f"{question}\n\nAnswer:"
        )
        # Generate
        out = self.generator(prompt, max_length=max_length, do_sample=False, temperature=temperature, num_return_sequences=1)
        answer = out[0]["generated_text"].strip()
        return {"answer": answer, "prompt": prompt, "retrieved": retrieved_docs}

# ---------- Utilities ----------
def read_text_files_from_dir(dir_path: str) -> List[str]:
    texts = []
    for fname in sorted(os.listdir(dir_path)):
        fp = os.path.join(dir_path, fname)
        if os.path.isfile(fp) and fname.lower().endswith((".txt", ".md")):
            with open(fp, "r", encoding="utf-8") as f:
                texts.append(f.read())
    return texts


# ---------- CLI & Flask API ----------
app = Flask(__name__)
rag_system: RAG = None  # will be set in main

@app.route("/api/ask", methods=["POST"])
def api_ask():
    payload = request.json
    if not payload or "question" not in payload:
        return jsonify({"error": "Send JSON with 'question' key."}), 400
    question = payload["question"]
    k = payload.get("k", K)
    retrieved = rag_system.retrieve(question, k=k)
    gen = rag_system.generate(question, retrieved, max_length=256)
    return jsonify(gen)


def main():
    parser = argparse.ArgumentParser(description="Simple RAG example")
    parser.add_argument("--build-corpus", type=str, help="Directory of .txt/.md files to build index from")
    parser.add_argument("--ask", type=str, help="Ask a question against the built index")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Flask host")
    parser.add_argument("--port", type=int, default=5000, help="Flask port")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU if available (careful)")
    args = parser.parse_args()

    global rag_system
    device = 0 if args.use_gpu else -1
    rag_system = RAG(device=device)

    if args.build_corpus:
        docs = read_text_files_from_dir(args.build_corpus)
        if not docs:
            print("No .txt or .md files found in", args.build_corpus)
            return
        rag_system.build_index(docs)
        print("Built index from", len(docs), "documents.")
        return

    # load index and run interactive ask / API
    try:
        rag_system.load_index()
    except Exception as e:
        print("Failed to load index:", e)
        print("If you haven't built the index run: python rag.py --build-corpus ./sample_docs/")
        return

    if args.ask:
        question = args.ask
        retrieved = rag_system.retrieve(question, k=K)
        print("Retrieved docs (id, score):")
        for d in retrieved:
            preview = d['text'][:160].replace('\n', ' ')
            print(f"- id={d['id']} score={d['score']:.4f} preview={preview}")

        res = rag_system.generate(question, retrieved)
        print("\n=== ANSWER ===")
        print(res["answer"])
        return

    # otherwise run API
    print(f"Starting Flask API on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
