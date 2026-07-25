# TATRO Application

**TATRO** is an offline AI-powered audio processing application that transcribes, translates, summarizes and rertrieve video segment based on user input query. 

## Features

* 🎙 **Automatic speech transcription**
* 🌍 **Translation** between user-selected input and output languages
* ✨ **AI-powered transcript correction** before translation (Optional)
* 👥 **Speaker diarization** with speaker-labeled transcripts
* 📄 **Translated PDF** generation
* 🔍 **Semantic retrieval** to locate the most relevant video segments from natural language queries
* 📝 **Automatic summarization**

## AI Models

* **Speech Recognition:** Whisper
* **Transcript Correction:** Qwen 3.5 9B
* **Translation:** Qwen 3.5 9B

## Supported Languages

* Input and output languages are fully configurable by the user.
* Translation quality varies depending on the selected language pair and the underlying language model.

## Roadmap

### Version 1

* ✅ Speech transcription
* ✅ Transcript correction
* ✅ Translation
* ✅ Speaker-labeled PDF export

### Version 2

* 🚧 Semantic retrieval from translated transcripts
* 🚧 Automatic video summarization
* 🚧 Additional search and analysis capabilities


# Project Structure

```
.
├── app.py
├── main.py
├── transcription.py
├── translation.py
├── requirements.txt
└── README.md
```

---

# Files

## `transcription.py`

### Input

- MP4 video
- Spoken language
- Merge count (4)

### What it does

- Extracts audio
- Performs speaker diarization using Pyannote
- Create audio chunk based on results of diarization: remove silence, one speaker, maximum length = 30 sec
- Transcribes each chunk with Whisper
- Correct the transcription (optional)
- Merges consecutive chunks belong to the same user (max 4)

### Output

`<video_name>_transcript.json`

Each entry contains:

- Speaker label
- Speaker number
- Start time
- End time
- Duration
- Raw transcription

---

## `translation.py`

### Input

- Transcript JSON

### Output

#### `<video_name>_translated.json`

Contains:

- Speaker
- Start / End
- Duration
- Raw transcription
- Translation

This file is intended for: Embedding, Semantic search and Retrieval

#### `<video_name>_translation_final.pdf`
- A clean, readable, and user-friendly translated text.
---

## `main.py`

Pipeline orchestrator.

### Input

- MP4 file
- Spoken language
- Merge count

Runs:

1. `transcribe_video()`
2. `translate_transcript()`

### CLI Example

```bash
python main.py --input "/path/video.mp4" --language en --merge 3
```

---

# `app.py`

The main Dash web application that provides an interactive interface for uploading videos, generating transcriptions, translating content, creating summaries, and interacting with translated documents.

## Features

* **Upload MP4 videos** for processing.
* **Select the source video and target translation language.**
* **Configure the merge count** for transcription and translation processing.
* **Provide a Hugging Face access token** to enable AI-powered features.
* **Monitor live progress updates** throughout the processing pipeline.
* **Skip transcription by uploading an existing transcript**, then proceed directly to translation.
* **Preview the translated PDF**, or upload your own PDF for translation and review.
* **Generate AI-powered summaries** from the translated PDF and preview the results.
* **Chat with the translated document** using the integrated chatbot to ask questions and explore the content interactively.


# Installation

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Additional requirements:

- FFmpeg
- Hugging Face token
- Access to:
  - `pyannote/speaker-diarization-3.1`

---

# Running

## Web Interface (Recommended)

```bash
export HF_TOKEN=your_token_here
python app.py
```

Windows PowerShell:

```powershell
$env:HF_TOKEN="your_token_here"
python app.py
```

Then:

- Open the generated URL
- Upload a video
- Select language
- Click **Run**
- Download or preview the PDF

---

## Command Line

```bash
python main.py \
    --input "/path/video.mp4" \
    --language ar (arabic)
    --merge 3
```

More options:

```bash
python main.py --help
```

Running without arguments starts interactive mode.
