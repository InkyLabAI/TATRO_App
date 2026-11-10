# TATRO: local video transcription and search

TATRO transcribes videos, translates transcripts, generates summaries, and answers users’ questions by returning the relevant video segment with its start and end timestamps. All AI processing runs locally, helping users keep their content private.

The application helps university students review recorded lectures, online courses, and webinars; enables podcast listeners to generate concise summaries and revisit key discussions; and supports journalists in transcribing interviews and locating relevant statements. Researchers and professionals can also use TATRO to explore recorded talks, summarize meetings, and find a precise information within a long lecture.

## Models and setup

The supported deployment requires an NVIDIA GPU and Linux or WSL2 (x86_64). Use Python 3.11 or 3.12, FFmpeg, and CUDA-enabled PyTorch. Complete the GPU prerequisites below before installing the dependencies:

```bash
python -m pip install --no-build-isolation -r requirements.txt
export HF_TOKEN="your_hugging_face_token"
python app.py
```

Open http://127.0.0.1:8050. Accept the Hugging Face access terms for `pyannote/speaker-diarization-3.1` and its required segmentation model. Model weights download on first use, then inference runs on this computer. There are no embedding or LLM inference API calls. For strictly disconnected operation, download all weights first and set `HF_HUB_OFFLINE=1`.

The application offers:

- **Embedding:** `Qwen/Qwen3-Embedding-0.6B` (default), `4B`, or `8B`. Every stored vector has **1,024 dimensions**. Larger outputs are truncated to 1,024 and then L2-normalized.
- **Correction, translation and summaries:** `Qwen/Qwen3.5-9B` (default), `Qwen/Qwen3-4B`, `Qwen/Qwen3-8B`, or `Qwen/Qwen2.5-7B-Instruct`. A custom Hugging Face model ID or local folder can override this selector; it must support Transformers chat generation.
- Large models require substantial RAM/VRAM. Models are released between stages; small embedding batches reduce memory pressure. No automatic quantization or quality-changing model substitution is performed.

PDF export uses Arial on Windows or DejaVu Sans on Linux, with Arabic shaping bidirectional layout. Set `TATRO_PDF_FONT` to a Unicode TTF font if necessary. Chinese output uses ReportLab's Chinese CID font.

## One dependency file

`requirements.txt` contains the complete project environment: app packages, transcription and embedding libraries, LLMs, optimized CUDA kernels (`flash-linear-attention` and `causal-conv1d`), Whisper CUDA runtime libraries, PDF generation, and testing tools. There are no separate GPU or development requirement files.

Before installing it:

1. Install a compatible NVIDIA driver and a CUDA-enabled, matching PyTorch/torchaudio pair using the [official PyTorch installer](https://pytorch.org/get-started/locally/).
2. Install the CUDA Toolkit and a C++ compiler. `CUDA_HOME/bin/nvcc` must exist and the toolkit must be compatible with your PyTorch CUDA build. The runtime libraries included in PyTorch do not provide this compiler.
3. Install the Python build helpers first, because pip may prepare `causal-conv1d` metadata before installing the other entries in the requirements file.

```bash
python -m pip install --upgrade pip packaging setuptools wheel ninja
nvcc --version
python -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available()"
python -m pip install --no-build-isolation -r requirements.txt
python -m pip check
```

The file uses version ranges rather than a tested lockfile for every GPU/Python combination. FFmpeg, Tk, fonts, NVIDIA drivers, the CUDA Toolkit, and the C++ compiler are system prerequisites installed outside pip. GPU kernels follow the [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention/blob/main/INSTALL.md) and [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d) installation requirements.

## Linux and WSL2 setup (VS Code)

Use an Ubuntu/Debian environment with Python 3.11 or 3.12 for broad compatibility with the audio packages. In VS Code on Windows, install the WSL extension, connect to your WSL2 distribution, and open a WSL terminal. Do not activate the Windows `.venv` inside Linux.

On native Linux, install a compatible NVIDIA driver. On WSL2, install/update the NVIDIA **Windows** driver with WSL support; do not install a Linux display driver inside WSL. Check visibility from your Linux terminal:

```bash
nvidia-smi
sudo apt update
sudo apt install -y python3-venv python3-pip python3-tk ffmpeg fonts-dejavu-core build-essential

# WSL path for this project; on native Linux use your checkout's path.
cd /mnt/c/Users/ahhac/Desktop/Ar_Heb_ASR_Translation/TATRO_App
python3 -m venv .venv-wsl
source .venv-wsl/bin/activate
python -m pip install --upgrade pip packaging setuptools wheel ninja
```

Before installing the requirements, install CUDA-enabled **torch and torchaudio together**, using the command from the [official PyTorch Linux/Pip/CUDA selector](https://pytorch.org/get-started/locally/). Choose a supported build for your NVIDIA driver and GPU, rather than a CPU-only build. Then install the app:

```bash
python -m pip install --no-build-isolation -r requirements.txt
python -m pip check
```

Faster-Whisper uses CTranslate2, which also needs its own compatible CUDA libraries. Its current GPU instructions require CUDA 12 cuBLAS and cuDNN 9. The corresponding CUDA 12 libraries are included in `requirements.txt`. Expose them to CTranslate2 before starting the app:

```bash
# cuBLAS and cuDNN are already included in requirements.txt.
export LD_LIBRARY_PATH="$(python -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))')${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Set that library path **before starting Python**, in each new terminal that needs these pip-provided libraries. If your PyTorch build uses another CUDA generation, follow the [Faster-Whisper GPU installation instructions](https://github.com/SYSTRAN/faster-whisper#gpu) to provide compatible CTranslate2 libraries; do not assume PyTorch GPU detection alone verifies Whisper's runtime.

Verify CUDA in the same environment that will launch TATRO:

```bash
python - <<'PY'
import torch
import ctranslate2
print("PyTorch:", torch.__version__)
print("PyTorch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("CTranslate2 CUDA devices:", ctranslate2.get_cuda_device_count())
assert torch.cuda.is_available(), "CUDA is unavailable in this Python environment"
print("GPU:", torch.cuda.get_device_name(0))
x = torch.ones(4, device="cuda")
print("CUDA operation:", (x * 2).cpu().tolist())
print("Whisper compute types:", ctranslate2.get_supported_compute_types("cuda"))
PY
```

Start the app:

```bash
export HF_TOKEN="your_hugging_face_token"
python app.py
```

Open http://localhost:8050 in your browser. You can also enter the Hugging Face token in the app. Inside the app, use **Linux paths**, such as `/mnt/c/Users/ahhac/Videos/lecture.mp4` in WSL or `/home/yourname/Videos/lecture.mp4` on native Linux.

For later sessions, activate the existing environment, restore `LD_LIBRARY_PATH` if using the pip-provided CUDA libraries above, and run `python app.py` again. Keep downloaded model caches in the Linux filesystem for better performance when possible.

### GPU behavior across all stages

| Stage | GPU execution |
|---|---|
| Speaker diarization | Pyannote on CUDA |
| Whisper transcription | CUDA, float16 |
| Correction, transcript/question translation, summarization and summary translation | Selected LLM on `cuda:0`; bfloat16 when supported, otherwise float16 |
| Passage and question embeddings | Selected embedding model on `cuda:0`, float16 |

CPU-only operation is not a supported deployment. Complete the CUDA verification above before starting model tasks.

Embeddings remain normalized float32 arrays with **1,024 dimensions** on disk. FFmpeg audio extraction, JSON/NPY storage, PDF generation, and the small cosine-similarity calculation run on CPU.

CUDA is selected automatically using `torch.cuda.is_available()`. The LLM is explicitly placed on the GPU instead of silently offloading layers to CPU. Models are released between stages. If a selected model does not fit in VRAM, select a smaller LLM/embedding model or free GPU memory; the app reports the failure instead of silently switching to CPU. The default 9B LLM requires substantial GPU memory.

To choose one of several GPUs, set `CUDA_VISIBLE_DEVICES` before launching; the selected physical GPU becomes logical `cuda:0`:

```bash
CUDA_VISIBLE_DEVICES=1 python app.py
```

Use `watch -n 1 nvidia-smi` in another terminal during processing. If `nvidia-smi` works but PyTorch reports CUDA unavailable, check that VS Code is using `.venv-wsl`, that its PyTorch build includes CUDA, and that the driver supports that build. Missing `libcublas`/`libcudnn` errors usually indicate the separate Whisper libraries or library path need attention. GPU detection is not a substitute for a successful first inference run.

WSL driver guidance: [NVIDIA CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).

## Using the app

1. Use the single **Choose your video** picker: click **Browse / open** to open the native file-selection window, navigate to your MP4, and select it. On WSL, this opens the Windows picker and converts the path automatically. On native Linux, it opens a Tk desktop dialog (requires `python3-tk` and a graphical desktop). You can also paste the complete MP4 path into the same field and press Enter. The original video is used directly, and all results are saved in a subfolder named after the video, next to the original MP4 (without copying or moving the video). There is no video upload copy or separate output-folder setting.
2. Select the video language, translation language, your question/summary language, and local models.
3. Click **Transcribe & prepare search**. If `<video>_transcript.json` already exists, its raw chunks are reused: audio extraction, speaker detection, and Whisper are skipped. Otherwise these stages run once. The pipeline then merges same-speaker text, optionally corrects it, saves the transcript/PDF, and creates embeddings. Correction is enabled by default.
4. Ask a question and press Send or Enter. A different question language is translated into the saved video language using the selected LLM. Same-language questions skip translation. Only the extracted translation text is embedded. Search moves the video to the best matching passage and reports its start and end times.
5. Click **Generate summary** to summarize the corrected transcript when available (otherwise the uncorrected transcript), translate the summary into the selected summary language, save it as text and PDF, and show the PDF. Long transcripts are processed in sections and consolidated; prompts explicitly remove overlap repetition and preserve essential facts. Generation errors or output limits are reported instead of silently saving truncated text.

Correction is not required for summaries. The saved transcript dataset is used first (corrected when available); if it does not exist, the raw `<video>_transcript.json` is used directly. Summarization happens in the video language, followed by translation of the summary only when its requested language differs. The translated full transcript is never the input to summarization.

Choose **Transcription + translation + summary** to run all stages with one click. Its summary language is the translation target: when source and target match, the summary stays in the source language and no translation is performed; otherwise the source-language summary is translated into the target language. The generated summary PDF appears automatically. In the other modes, the separate summary button uses the chat/summary language selector.

Raw transcription reuse checks the video-named results folder first, then the legacy file beside the MP4. A legacy raw transcript is copied into the results folder when processing resumes. Invalid files or a conflicting saved language produce an error rather than silently repeating transcription. To deliberately transcribe again, rename or remove the raw transcript cache before running processing. No diarization token is required when reusing a raw transcript (model downloads may still need authentication). Search locates the most relevant passage; it does not establish that the video definitively answers the question.

The native picker opens on the computer running TATRO. Use the app locally; when connecting to a remote server, paste a path available on that server instead.

## Transcription PDF preview

Transcription also creates `<video>_corrected_transcription.pdf`, with speaker labels and text only, in the video's results folder. Neither transcription nor translation PDFs include segment timestamps; timing remains in the JSON data for video search. The document panel shows this PDF when **Transcription only** is selected or when the saved video language matches the translation language. Matching languages skip the translation model entirely. An explicit **Generate translation** action displays the requested translation for different languages.

Opening an older saved transcript creates a missing transcription PDF without repeating transcription or correction. Existing PDFs are reused when their transcript fingerprint matches. If correction was disabled, the file is named `<video>_transcription.pdf` and labelled uncorrected instead.

## Live task progress

The **Task progress** panel updates every second while processing runs. It displays the active stage, elapsed time, and a progress bar for counted steps, such as `5/25` chunks transcribed, corrected, translated, or embedded. Model loading, audio extraction, speaker detection, summary sections, PDF creation, saved-result reuse, completion, and failures appear in the activity log (latest first).

Counts refer to the current stage, not the complete pipeline. Weight loading and other steps without a known total use an activity indicator instead of a guessed percentage. The log keeps the most recent 500 events for your browser session and clears when the server restarts; it is not a persistent log file.

Start with `python app.py`: the server is explicitly threaded so progress polling continues while the model callback is busy. Progress is stored in memory in one server process. A future deployment with multiple server processes needs a shared progress store (for example Redis) before enabling multiple workers.

## Merge and retrieval rules

Within each uninterrupted speaker run, merge three consecutive original chunks with two overlapping chunks (stride one). Five chunks become `[1,2,3]`, `[2,3,4]`, `[3,4,5]`. A run of one or two chunks produces one shorter window; there are no redundant trailing partial windows. Never cross a speaker change. Keep original start/end timestamps, speech boundaries, speaker number, and contributing original chunk IDs. Empty transcriptions are excluded from the search dataset.

The search policy follows the supplied example: rank cosine similarity descending (equivalent to cosine distance ascending), take the top five, anchor at the strongest match, include the preceding top-five hit if present, then extend forward with one-hole bridging. The returned interval covers the earliest start and latest end of those windows on the original video timeline.

## Saved results and reuse

For `/home/you/Videos/interview.mp4`, all generated files live in `/home/you/Videos/interview/`. The folder is created automatically when results are first written. Existing results in that folder are reused; regeneration buttons overwrite the corresponding files there. Older results stored directly beside the MP4 can be moved into this folder to reuse them.

```text
Videos/
├── interview.mp4
└── interview/
    ├── interview_merged_chunks.json
    ├── interview_transcript_dataset.json
    ├── interview_embeddings.npy
    └── interview_summary_ar.pdf
```

The complete output list:

| File | Content |
|---|---|
| `interview_audio.wav` | Extracted audio |
| `interview_transcript.json` | Original Whisper chunks |
| `interview_merged_chunks.json` | Raw 3/2 windows, before correction, with IDs, speaker, text, source IDs and times |
| `interview_transcript_dataset.json` | Corrected window text (or explicitly marked uncorrected text), language and timestamps |
| `interview_corrected_transcription.txt` | Corrected text; uncorrected runs use `_transcription.txt` |
| `interview_embeddings.npy` | Float32 matrix, one row per dataset record, shape `(N, 1024)` |
| `interview_embeddings.meta.json` | Model, dimensions, record IDs and transcript fingerprint |
| `interview_translated_ar.json` | Translation into the selected language (Arabic example) |
| `interview_ar_translation.pdf` | Translation PDF |
| `interview_summary_ar.txt` / `.pdf` | Summary in the selected user language |
| `*.meta.json` | Transcript identity used to detect stale summaries/translations |

Loading a video never runs models. Saved PDFs are shown when present and current. Search reuses an index whose model, transcript and dimensions match; it rebuilds a missing, corrupt or incompatible index. Changing the embedding selector therefore cannot mix vector spaces. A bare legacy NPY file without model metadata is rebuilt because compatibility cannot be verified safely.

**Transcribe** reuses the raw transcript and rewrites downstream merged/corrected text, PDFs and embeddings. **Translate**, **Rebuild embeddings**, and **Generate summary** overwrite their corresponding outputs. **Show / reuse summary** uses the saved result, generating it only when missing or stale. Saved translations can be viewed without regeneration. Importing a transcript requires the dataset schema (`id`, `speaker_number`, `source_language`, `source_text`, `start`, `end`); `corrected: true` identifies corrected data for summary generation.

## Command line

```powershell
python main.py --input "C:/Videos/interview.mp4" --language he --target ar --summary
python main.py --input "C:/Videos/interview.mp4" --language he --embedding-model Qwen/Qwen3-Embedding-4B --llm-model Qwen/Qwen3.5-9B --force
```

The CLI reuses existing results by default; `--force` regenerates downstream results while still reusing a saved raw transcript. Outputs always use the video-named subfolder beside the input file.

## Code organization

- `app.py`: Dash controls, callbacks, registered media/PDF serving and browser seeking.
- `workflow.py`: Operations, model locking, cache reuse and regeneration.
- `transcription.py`: Audio chunks, diarization, Whisper and ordered transcription pipeline.
- `chunks.py`: Pure same-speaker sliding-window merging.
- `retrieval.py`: Local embeddings, index checks and ranked segment selection.
- `local_llm.py`: Model loading, clean translation, correction and hierarchical summaries.
- `artifacts.py`: File naming, fingerprints and atomic writes.
- `pdf_export.py`: Paginated multilingual PDFs.
- `languages.py`: Shared language catalog.
- `translation.py`: Compatibility entry point for dataframe callers.
- `main.py`: CLI.

## Verification

```powershell
python -m pip install --no-build-isolation -r requirements.txt
python -m pytest -q
```
