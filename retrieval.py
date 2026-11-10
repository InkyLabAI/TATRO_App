"""Local Qwen embeddings and the ranked consecutive-segment search policy."""
import gc
import io
from progress import report
import numpy as np

from artifacts import atomic_write, fingerprint, read_json, write_json

EMBEDDING_MODELS = [f"Qwen/Qwen3-Embedding-{size}" for size in ("0.6B", "4B", "8B")]
DIMENSIONS = 1024
QUERY_PROMPT = "Instruct: Given a question, retrieve passages from a video transcript that answer the question.\nQuery: "


def embed_texts(texts, model_name, *, query=False):
    """Inference is entirely local; Hugging Face may download weights on first use."""
    from sentence_transformers import SentenceTransformer
    import torch

    if not texts or any(not text.strip() for text in texts):
        raise ValueError("Cannot embed empty text.")
    report(f"Loading embedding weights: {model_name}")
    model = SentenceTransformer(
        model_name, device="cuda:0" if torch.cuda.is_available() else "cpu",
        tokenizer_kwargs={"padding_side": "left"},
        model_kwargs={"torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32},
    )
    try:
        # Fail explicitly instead of silently truncating a passage.
        prefix = QUERY_PROMPT if query else ""
        if any(len(model.tokenizer.encode(prefix + text)) > model.max_seq_length for text in texts):
            raise ValueError("A passage exceeds the embedding model context length.")
        report("Embedding question" if query else "Embedding transcript chunks", 0, len(texts))
        batches = []
        for offset in range(0, len(texts), 2):
            batches.append(model.encode(texts[offset:offset + 2], prompt=prefix, batch_size=2,
                                       convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=False))
            report("Questions embedded" if query else "Chunks embedded", min(offset + 2, len(texts)), len(texts))
        vectors = np.concatenate(batches, axis=0)
        if vectors.shape[1] < DIMENSIONS:
            raise ValueError("The selected model cannot produce 1,024 dimensions.")
        # Matryoshka truncation must happen BEFORE normalization.
        return normalize(vectors[:, :DIMENSIONS])
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def normalize(vectors):
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError("Invalid embedding matrix.")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if (norms == 0).any():
        raise ValueError("Embedding contains a zero vector.")
    return vectors / norms


def ensure_index(artifacts, records, model_name, *, force=False, embedder=embed_texts):
    if not records:
        raise ValueError("No transcribed speech is available for search.")
    metadata = {"model": model_name, "dimensions": DIMENSIONS,
                "records_hash": fingerprint(records), "ids": [row["id"] for row in records]}
    if not force and artifacts.embeddings.exists() and artifacts.index_metadata.exists():
        try:
            vectors = np.load(artifacts.embeddings, allow_pickle=False)
            if read_json(artifacts.index_metadata) == metadata and vectors.shape == (len(records), DIMENSIONS):
                report("Reusing saved embeddings")
                return normalize(vectors)
        except (ValueError, OSError):
            pass
    vectors = embedder([row["source_text"] for row in records], model_name)
    if vectors.shape != (len(records), DIMENSIONS):
        raise ValueError("Embedding rows or dimensions do not match the transcript.")
    vectors = normalize(vectors)
    stream = io.BytesIO()
    np.save(stream, vectors, allow_pickle=False)
    report("Saving embeddings and index metadata")
    atomic_write(artifacts.embeddings, stream.getvalue())
    write_json(artifacts.index_metadata, metadata)
    return vectors


def select_segment(ranked_indices, one_hole=True):
    """Anchor at best hit, add preceding top-five hit, then extend with one-hole bridging."""
    if not ranked_indices:
        return []
    anchor = ranked_indices[0]
    selected = [anchor]
    if anchor - 1 in ranked_indices[:5]:
        selected.insert(0, anchor - 1)
    hits = set(ranked_indices)
    current = anchor + 1
    while current in hits or (one_hole and current + 1 in hits):
        selected.append(current)
        if current not in hits:
            selected.append(current + 1)
            current += 2
        else:
            current += 1
    return selected


def search(records, vectors, question_vector, top_k=5, one_hole=True):
    if not records or top_k < 1:
        raise ValueError("Search requires indexed chunks and a positive top K.")
    vectors = normalize(vectors)
    question_vector = normalize(np.asarray(question_vector).reshape(1, -1))[0]
    if vectors.shape != (len(records), question_vector.size):
        raise ValueError("Question and transcript embeddings do not match.")
    scores = vectors @ question_vector
    ranked = np.argsort(-scores, kind="stable")[:top_k].tolist()
    selected = select_segment(ranked, one_hole)
    start = min(records[i]["start"] for i in selected)
    end = max(records[i]["end"] for i in selected)
    return {"start": start, "end": end, "duration": end - start,
            "score": float(scores[ranked[0]]), "chunk_ids": [records[i]["id"] for i in selected]}


def timestamp(seconds):
    seconds = max(0, int(seconds))
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"
