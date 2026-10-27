import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).parent.parent / ".env")


class Llm:
    def __init__(
        self,
        embedding_model: str = "text-embedding-3-large",
        chat_model: str = "gpt-6-astra",
        stt_model: str = "whisper-1",
    ) -> None:
        self.embedding_dimension = 3072
        self.embedding_model = embedding_model
        self.chat_model = chat_model
        self.stt_model = stt_model
        api_key = os.getenv("OPEN_AI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("OPEN_AI_API_KEY is not set")
        self._client = OpenAI(api_key=api_key)

    def build_embedding(self, text: str) -> list[float]:
        cleaned = text.strip()
        if not cleaned:
            raise ValueError("text is empty")

        response = self._client.embeddings.create(
            model=self.embedding_model,
            input=cleaned,
            dimensions=self.embedding_dimension,
        )
        embedding = response.data[0].embedding
        if len(embedding) != self.embedding_dimension:
            raise ValueError(
                f"expected {self.embedding_dimension} dimensions, got {len(embedding)}"
            )
        return embedding

    def generate_response(self, messages: list[dict]) -> str:
        if not messages:
            raise ValueError("messages are empty")

        response = self._client.chat.completions.create(
            model=self.chat_model,
            messages=messages,
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("model returned an empty response")
        return content

    def stt(self, audio_path: str) -> str:
        if not audio_path.strip():
            raise ValueError("audio path is empty")

        with open(audio_path, "rb") as audio_file:
            response = self._client.audio.transcriptions.create(
                model=self.stt_model,
                file=audio_file,
            )

        transcript = response.text.strip()
        if not transcript:
            raise ValueError("model returned an empty transcript")
        return transcript
