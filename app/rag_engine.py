from sentence_transformers import SentenceTransformer
import faiss
import fitz  # PyMuPDF

class PDFRAGEngine:
    def __init__(self, pdf_path):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.pdf_path = pdf_path
        self.text_chunks = self._load_pdf()
        self.index, self.chunk_mapping = self._build_index()

    def _load_pdf(self):
        doc = fitz.open(self.pdf_path)
        text_chunks = []
        for page in doc:
            text = page.get_text()
            text_chunks.extend(text.split('. '))
        return text_chunks

    def _build_index(self):
        vectors = self.embedder.encode(self.text_chunks)
        index = faiss.IndexFlatL2(vectors.shape[1])
        index.add(vectors)
        return index, self.text_chunks

    def query(self, question):
        q_vec = self.embedder.encode([question])
        D, I = self.index.search(q_vec, k=3)
        relevant_texts = [self.chunk_mapping[i] for i in I[0]]
        return " ".join(relevant_texts)
