from pathlib import Path
from bs4 import BeautifulSoup

from services.llm import Llm

SOURCES_DIR = Path(__file__).parent.parent / "sources"
VOICE_SUFFIXES = {".mp3", ".wav", ".m4a"}


class Rag:
    system_message = (
        "Answer the user using only the related chunks below. "
        "If the chunks do not contain the answer, say you do not know.\n\n"
        "Related chunks:\n"
    )

    def __init__(self):
        self.llm = Llm()
        self.stored = []

    def gather_info(self):
        texts = []

        for path in sorted(SOURCES_DIR.iterdir()):
            if not path.is_file():
                continue

            suffix = path.suffix.lower()
            if suffix in VOICE_SUFFIXES:
                texts.append(self.llm.stt(str(path)))
            elif suffix in {".html", ".htm"}:
                soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
                texts.append(soup.get_text(separator="\n", strip=True))

        return texts

    def split(self, texts, size=1000, overlap=200):
        if isinstance(texts, str):
            texts = [texts]
        if overlap >= size:
            raise ValueError("overlap must be smaller than size")

        chunks = []
        step = size - overlap
        for text in texts:
            cleaned = text.strip()
            if not cleaned:
                continue

            start = 0
            while start < len(cleaned):
                chunk = cleaned[start : start + size].strip()
                if chunk:
                    chunks.append(chunk)
                start += step

        return chunks

    def embed_chunks(self, chunks):
        embedded = []

        for chunk in chunks:
            embedded.append(
                {
                    "chunk_text": chunk,
                    "chunk_embedding": self.llm.build_embedding(chunk),
                }
            )

        return embedded

    def store(self, embedded):
        self.stored = list(embedded)

    def embed_query(self, query):
        return self.llm.build_embedding(query)
    
    def find_similar(self, query_embedding, top_k=5, threshold=0.3):
        query_norm = sum(x * x for x in query_embedding) ** 0.5
        if query_norm == 0:
            return []

        scored = []
        for item in self.stored:
            chunk_embedding = item["chunk_embedding"]
            chunk_norm = sum(x * x for x in chunk_embedding) ** 0.5
            if chunk_norm == 0:
                continue

            dot = sum(q * c for q, c in zip(query_embedding, chunk_embedding))
            score = dot / (query_norm * chunk_norm)
            if score >= threshold:
                scored.append({"chunk_text": item["chunk_text"], "score": score})

        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:top_k]

    def build_messages(self, user_question, related_chunks):
        chunk_texts = []
        for chunk in related_chunks:
            if isinstance(chunk, dict):
                chunk_texts.append(chunk["chunk_text"])
            else:
                chunk_texts.append(chunk)

        system_content = self.system_message + "\n\n".join(chunk_texts)
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_question},
        ]

    def generate_response(self, messages):
        return self.llm.generate_response(messages)