"""Stable, video-relative artifact paths and atomic writes."""
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data if isinstance(data, bytes) else data.encode("utf-8"))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path, data):
    atomic_write(path, json.dumps(data, ensure_ascii=False, indent=2))


def fingerprint(data):
    return hashlib.sha256(json.dumps(data, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class Artifacts:
    video: Path

    def __post_init__(self):
        object.__setattr__(self, "video", Path(self.video).resolve())

    def path(self, suffix):
        return self.directory / (self.video.stem + suffix)

    @property
    def directory(self):
        """Dedicated results folder beside the original MP4."""
        return self.video.parent / self.video.stem

    @property
    def raw(self):
        return self.path("_transcript.json")

    @property
    def merged(self):
        return self.path("_merged_chunks.json")

    @property
    def transcript(self):
        return self.path("_transcript_dataset.json")

    @property
    def embeddings(self):
        return self.path("_embeddings.npy")

    @property
    def index_metadata(self):
        return self.path("_embeddings.meta.json")

    def summary(self, language, extension="pdf"):
        return self.path(f"_summary_{language}.{extension}")
