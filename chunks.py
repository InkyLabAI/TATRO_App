"""Speaker-safe sliding windows on the original video timeline."""
from copy import deepcopy
from itertools import groupby


def merge_chunks(chunks, window=3, overlap=2):
    """Full windows advance by one; short speaker runs form one partial window.

    A five-chunk run produces [1,2,3], [2,3,4], [3,4,5]. No redundant
    trailing partial windows are emitted, and speaker boundaries are never crossed.
    """
    if not 0 <= overlap < window:
        raise ValueError("Overlap must be smaller than the window.")
    merged, speakers = [], {}
    for speaker, group in groupby(chunks, key=lambda chunk: chunk["speaker"]):
        run = list(group)
        speakers.setdefault(speaker, len(speakers) + 1)
        starts = list(range(0, max(1, len(run) - window + 1), window - overlap))
        if len(run) > window and starts[-1] + window < len(run):
            starts.append(len(run) - window)
        for start in starts:
            parts = run[start:start + window]
            item = deepcopy(parts[0])
            item.update(
                id=len(merged) + 1,
                speaker_number=speakers[speaker],
                initial_chunk_ids=[part["id"] for part in parts],
                start=min(part["start"] for part in parts),
                end=max(part["end"] for part in parts),
                transcription=" ".join(part["transcription"].strip() for part in parts).strip(),
                boundaries=[boundary for part in parts for boundary in part.get("boundaries", [])],
            )
            item["duration"] = item["end"] - item["start"]
            item["speech_duration"] = sum(part.get("duration", part["end"] - part["start"]) for part in parts)
            item["text"] = item["transcription"]
            merged.append(item)
    return merged
