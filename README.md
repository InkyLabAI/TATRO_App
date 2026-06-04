# ATRA

ATRA is a fully offline application designed to transcribe Hebrew video and audio content, translate the transcription into Arabic, and retrieve relevant video segments based on Arabic-language queries.

The entire pipeline runs locally using lightweight language models and can operate on a GPU with as little as 8 GB of VRAM.

---

## Features

* Fully offline processing
* Hebrew speech-to-text transcription
* Speaker diarization
* Hebrew-to-Arabic translation
* Arabic semantic search over transcriptions
* Relevant video segment retrieval
* JSON and PDF export
* Lightweight deployment (8 GB GPU compatible)

---

## Project Overview

The application consists of the following components:

### Core Modules

* **ASR.py**
  Performs Hebrew video/audio transcription, including speaker diarization, audio chunking, and export of results in JSON and PDF formats.

* **Translation.py**
  Translates the Hebrew transcription into Arabic and saves the translated output in JSON and PDF formats.

* **InferenceQuery.py**
  Retrieves the most relevant video segment based on an Arabic-language query.

* **app.py**
  Provides a user-friendly interface that allows users to:

  * Load a video
  * Generate a transcription
  * Translate the transcription
  * Ask questions in Arabic
  * Receive both a textual answer and the corresponding relevant video segment

### Evaluation and Benchmarking

* **ASR_Benchmark**
  Evaluation of multiple Automatic Speech Recognition (ASR) models across different datasets to identify the most effective Hebrew transcription model. See the README inside the folder for details.

* **Translation_Benchmark**
  Evaluation of various offline Hebrew-to-Arabic translation models using multiple datasets. See the README inside the folder for details.

* **Embedding_Evaluation**
  Benchmarking of different embedding models and retrieval strategies for semantic search and segment retrieval.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/mhmddirany/Video-Transcription-Hebrew-ASR-Pipeline.git
cd Video-Transcription-Hebrew-ASR-Pipeline
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Your Hugging Face Token

Add your Hugging Face access token as required by the models used in the project.

---

## Usage

### Run the Application

```bash
python app.py
```

### Steps

1. Select the input video file.
2. Specify the output directory (optional).
3. Wait for the transcription and translation processes to complete. Processing time depends on the audio duration and hardware configuration.
4. Enter your question in Arabic using the chat interface.
5. Receive:

   * The relevant text response
   * The corresponding video segment containing the answer

---

## System Requirements

### Recommended

* NVIDIA GPU with 8 GB VRAM
* Python 3.10+
* CUDA-compatible environment
