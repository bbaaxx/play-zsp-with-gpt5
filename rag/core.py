from __future__ import annotations

import re
import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from dotenv import load_dotenv

# Load .env file with override=True to take precedence over existing env vars
load_dotenv(override=True)

from .embeddings import EmbeddingProvider
from .vector_store import VectorStore, create_vector_store
from .llm_providers import LLMManager


WHATSAPP_PATTERNS = [
    # Ej clásico 24h sin segundos: "[12/10/2023, 21:15] Juan: ¿Salimos mañana?"
    re.compile(r"^\[(\d{1,2}/\d{1,2}/\d{2,4}),\s*(\d{1,2}:\d{2})\]\s([^:]+):\s(.*)$"),
    # Variante con guion 24h: "12/10/23, 21:15 - Juan: ¿Salimos mañana?"
    re.compile(r"^(\d{1,2}/\d{1,2}/\d{2,4}),\s*(\d{1,2}:\d{2})\s*-\s*([^:]+):\s(.*)$"),
]

# Captura genérica con contenido de hora flexible dentro de corchetes, p. ej. "[26/05/25, 3:18:25 p.m.]"
GENERIC_BRACKETED = re.compile(
    r"^\[(\d{1,2}/\d{1,2}/\d{2,4}),\s*([^\]]+)\]\s([^:]+):\s(.*)$"
)


@dataclass
class ChatMessage:
    chat_id: str
    timestamp: datetime
    sender: str
    text: str
    line_no: int


def _normalize_whitespace(s: str) -> str:
    # Reemplaza NBSP y NNBSP por espacios, y colapsa dobles espacios
    return (
        s.replace("\u00A0", " ")
        .replace("\u202F", " ")
        .replace("\u2009", " ")
        .strip()
    )


def _parse_time_component(date_str: str, time_part: str) -> datetime:
    # Normaliza y detecta am/pm en español (a.m./p.m.) con o sin espacios
    clean = _normalize_whitespace(time_part).lower()
    clean = clean.replace(".", "")  # "a.m." -> "am"
    is_am = " am" in f" {clean} " or clean.endswith("am")
    is_pm = " pm" in f" {clean} " or clean.endswith("pm")
    # Extrae la porción hh:mm[:ss]
    # Remove am/pm tokens
    clean_time = clean.replace("am", "").replace("pm", "").strip()
    parts = clean_time.split(":")
    hour = int(parts[0]) if parts and parts[0].isdigit() else 0
    minute = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    second = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0

    # Convierte a 24h si hay indicador am/pm
    if is_pm and hour < 12:
        hour += 12
    if is_am and hour == 12:
        hour = 0

    # Normaliza fecha dd/mm/yy(yy)
    day, month, year = date_str.split("/")
    if len(year) == 2:
        year = "20" + year
    return datetime(
        year=int(year), month=int(month), day=int(day), hour=hour, minute=minute, second=second
    )


def _parse_line(line: str, line_no: int, chat_id: str) -> Optional[ChatMessage]:
    line = line.strip("\n\r")
    # Limpia caracteres invisibles comunes en exportes
    line = line.replace("\ufeff", "").replace("\u200e", "")
    line = _normalize_whitespace(line)
    if not line:
        return None
    # Ignorar mensajes de sistema comunes
    if (
        "cifrados" in line
        or "Messages and calls are end-to-end encrypted" in line
        or "joined using this group's invite link" in line
        or "image omitted" in line.lower()
        or "video omitted" in line.lower()
        or "gif omitted" in line.lower()
        or "audio omitted" in line.lower()
        or "document omitted" in line.lower()
        or "this message was deleted" in line.lower()
        or "you deleted this message" in line.lower()
    ):
        return None
    for pat in WHATSAPP_PATTERNS:
        m = pat.match(line)
        if m:
            date_str, time_str, sender, text = m.groups()
            # Normalizar fecha (dd/mm/yyyy or dd/mm/yy)
            day, month, year = date_str.split("/")
            if len(year) == 2:
                year = "20" + year
            dt = datetime.strptime(
                f"{year}-{month.zfill(2)}-{day.zfill(2)} {time_str}", "%Y-%m-%d %H:%M"
            )
            sender = sender.strip().lstrip("~").strip()
            return ChatMessage(
                chat_id=chat_id,
                timestamp=dt,
                sender=sender,
                text=text.strip(),
                line_no=line_no,
            )
    # Intento genérico con hora flexible (posible a.m./p.m. y segundos)
    mg = GENERIC_BRACKETED.match(line)
    if mg:
        date_str, time_part, sender, text = mg.groups()
        try:
            dt = _parse_time_component(date_str, time_part)
            sender = sender.strip().lstrip("~").strip()
            return ChatMessage(
                chat_id=chat_id,
                timestamp=dt,
                sender=sender,
                text=text.strip(),
                line_no=line_no,
            )
        except Exception:
            return None
    return None


def parse_whatsapp_txt(content: str, chat_id: Optional[str] = None) -> List[ChatMessage]:
    if chat_id is None:
        chat_id = hashlib.sha1(content.encode("utf-8")).hexdigest()[:12]
    messages: List[ChatMessage] = []
    for idx, raw in enumerate(content.splitlines(), start=1):
        msg = _parse_line(raw, idx, chat_id)
        if msg is not None and msg.text:
            messages.append(msg)
    return messages


@dataclass
class Chunk:
    chunk_id: str
    chat_id: str
    start_ts: datetime
    end_ts: datetime
    participants: List[str]
    line_span: Tuple[int, int]
    text_window: str


def format_message_for_window(msg: ChatMessage) -> str:
    return f"[{msg.timestamp.strftime('%Y-%m-%d %H:%M')}] {msg.sender}: {msg.text}"


def chunk_messages(messages: List[ChatMessage], window_size: int = 30, window_overlap: int = 10) -> List[Chunk]:
    chunks: List[Chunk] = []
    if not messages:
        return chunks
    start = 0
    while start < len(messages):
        end = min(len(messages), start + window_size)
        window = messages[start:end]
        text_window = "\n".join(format_message_for_window(m) for m in window)
        participants = sorted(list({m.sender for m in window}))
        chunk_id = hashlib.sha1(f"{messages[0].chat_id}-{start}-{end}".encode("utf-8")).hexdigest()[:12]
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                chat_id=messages[0].chat_id,
                start_ts=window[0].timestamp,
                end_ts=window[-1].timestamp,
                participants=participants,
                line_span=(window[0].line_no, window[-1].line_no),
                text_window=text_window,
            )
        )
        if end == len(messages):
            break
        start = end - window_overlap
        if start <= 0:
            start = end
    return chunks


class RAGPipeline:
    def __init__(
        self, vector_backend: str = "faiss", **vector_store_kwargs
    ) -> None:
        self.embedder = EmbeddingProvider()
        self.llm_manager = LLMManager()
        self.vector_store: Optional[VectorStore] = None
        self.chunks: List[Chunk] = []
        self.vector_backend = vector_backend
        self.vector_store_kwargs = vector_store_kwargs

    def index_messages(self, messages: List[ChatMessage]) -> None:
        chunks = chunk_messages(messages)
        self.chunks = chunks
        texts = [c.text_window for c in chunks]
        if not texts:
            self.vector_store = None
            return
        embeddings = self.embedder.embed_texts(texts)
        vs = create_vector_store(
            dim=embeddings.shape[1],
            backend=self.vector_backend,
            **self.vector_store_kwargs,
        )
        ids = [c.chunk_id for c in chunks]
        metas: List[Dict[str, Any]] = []
        for c in chunks:
            metas.append(
                {
                    "chunk_id": c.chunk_id,
                    "chat_id": c.chat_id,
                    "start_ts": c.start_ts.isoformat(timespec="minutes"),
                    "end_ts": c.end_ts.isoformat(timespec="minutes"),
                    "participants": c.participants,
                    "line_span": c.line_span,
                    "text_window": c.text_window,
                }
            )
        vs.add(ids, embeddings, metas)
        self.vector_store = vs

    def format_context(self, metas: List[Dict[str, Any]]) -> str:
        lines: List[str] = []
        for m in metas:
            snippet_lines = m.get("text_window", "").splitlines()[:5]
            lines.extend(snippet_lines)
        return "\n".join(lines)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        use_mmr: bool = True,
        fetch_k: int = 25,
        lambda_: float = 0.5,
        senders: Optional[List[str]] = None,
        date_from_iso: Optional[str] = None,
        date_to_iso: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if self.vector_store is None:
            return []
        query_emb = self.embedder.embed_texts([query])
        if use_mmr:
            _, metas_list = self.vector_store.search_mmr(
                query_emb, top_k=top_k, fetch_k=fetch_k, lambda_=lambda_,
                senders=senders, date_from_iso=date_from_iso, date_to_iso=date_to_iso
            )
        else:
            _, metas_list = self.vector_store.search(query_emb, top_k=top_k)
        return metas_list[0] if metas_list else []

    def generate_answer(
        self,
        context_snippets: str,
        question: str,
        temperature: float = 0.2,
        max_tokens: int = 800
    ) -> str:
        """Generate an answer using the configured LLM providers."""
        user_prompt = build_user_prompt(context_snippets, question)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        
        return self.llm_manager.generate_response(
            messages, temperature=temperature, max_tokens=max_tokens
        )


def build_user_prompt(context_snippets: str, question: str) -> str:
    return (
        "Contexto recuperado (fragmentos del chat):\n"
        f"{context_snippets}\n\n"
        f"Pregunta: {question}\n\n"
        "Instrucciones: Responde en español, cita 1–3 fragmentos relevantes con referencia [remitente — fecha]."
    )


SYSTEM_PROMPT = (
    "Eres un asistente en español. Responde de forma breve, correcta y sin inventar. "
    "Usa solo la evidencia del chat. Cuando cites, incluye remitente y fecha (p. ej., [Juan — 2023-10-12 21:15]). "
    "Si falta evidencia, dilo explícitamente."
)


