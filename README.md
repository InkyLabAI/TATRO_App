# TATRO: Local Video Assistant

Transcribe, translate, summarize, and ask questions about your videos. TATRO returns the relevant video segment with start/end timestamps and jumps to it automatically. Transcripts and summaries are available as PDFs.

Built for recorded courses, webinars, podcasts, and interviews. AI processing runs locally, without embedding or LLM inference APIs. Model weights download on first use.

## Requirements

- **Linux or WSL2 (x86_64), an NVIDIA GPU, and Python 3.11 or 3.12.** CPU-only use is not supported.
- A compatible NVIDIA driver, CUDA-enabled PyTorch/torchaudio, and the **CUDA Toolkit** (`nvcc`) for building GPU extensions.
- Enough GPU memory for your selected models. The default 9B LLM needs substantial VRAM; select a smaller model if it does not fit. The app does not automatically quantize models.

On WSL2, use the NVIDIA **Windows driver** with WSL support; do not install a Linux display driver inside WSL. In VS Code, open the project through the WSL extension and run the commands below in a **Linux terminal**.

## Install

Run these commands from the project folder. Ensure `python3 --version` is 3.11 or 3.12, or use the matching Python executable to create the environment.

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip python3-tk ffmpeg fonts-dejavu-core build-essential
python3 -m venv .venv-wsl
source .venv-wsl/bin/activate
python -m pip install --upgrade pip packaging setuptools wheel ninja
```

Install matching CUDA-enabled **torch and torchaudio** builds using the [PyTorch installer](https://pytorch.org/get-started/locally/). Install a compatible [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads); `CUDA_HOME` must point to its installation directory and `nvcc` must be on your PATH. Then check GPU access and install the app:

```bash
nvidia-smi
nvcc --version
python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0))"
python -m pip install --no-build-isolation -r requirements.txt
python -m pip check
```

For speaker detection, accept the Hugging Face access terms for [speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and [segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0). Create a read token and enter it under **Models & access**, or set `HF_TOKEN` before launching.

## Run

From the project folder, run this in each new Linux terminal. The library path makes the CUDA libraries installed by `requirements.txt` available to Whisper.

```bash
source .venv-wsl/bin/activate
export LD_LIBRARY_PATH="$(python -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))')${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python app.py
```

Open **http://127.0.0.1:8050**. Keep the terminal running; press `Ctrl+C` to stop the app.

## Use TATRO

1. Click **Browse video** and select an MP4, or paste its full path and press Enter. In WSL, pasted paths use `/mnt/c/...`; on Linux, use `/home/...`.
2. Choose the video, translation, and chat/summary languages. Under **Models & access**, select the embedding and language models. Qwen3-Embedding-0.6B uses the least memory; all embeddings have **1,024 dimensions**.
3. Choose **Transcription only**, **Transcription + translation**, or **Transcription + translation + summary**. Enable **Correct transcription** if wanted, then click **Transcribe & prepare search**. Follow the stages and chunk counts in **Task progress**.
4. Ask a question under the video and press **Send**. TATRO translates the question when needed, finds a relevant passage, and moves playback to its start.
5. Read the transcript/translation and summary PDFs under **Your documents**. Use **Generate summary** for a separate summary, or **Open saved** to view existing documents.

Summaries use the corrected transcript when available, otherwise the uncorrected transcript. TATRO summarizes in the video language, then translates the summary if needed. The combined transcription/translation/summary mode uses the **translation language**; the separate summary button uses the **chat/summary language**.

When video and translation languages match, translation is skipped and the transcript PDF is shown. Transcript and translation PDFs contain speaker labels and text, without timestamps; timing stays in the JSON files for search. Generated text and retrieved passages should be checked against the video when accuracy matters.

## Saved results

For `Videos/interview.mp4`, results are saved automatically in **`Videos/interview/`**. The original video stays in place. This folder contains raw and merged transcript JSON, corrected text, embeddings (`.npy`), translations, and summary text/PDFs.

- Opening a video reuses its saved results.
- **Transcribe & prepare search** reuses `<video>_transcript.json` when present, skips transcription, and regenerates downstream results. To transcribe again from scratch, rename or remove that raw transcript file (including any legacy copy beside the MP4).
- Generate/rebuild buttons replace the corresponding saved results. Changing the embedding model rebuilds an incompatible search index when needed.

## Common setup problems

- **Missing `nvcc`:** install the CUDA Toolkit and check `CUDA_HOME`. PyTorch's bundled runtime does not include the CUDA compiler.
- **Missing `libcublas` or `libcudnn`:** run the `LD_LIBRARY_PATH` command above before starting Python. See [Whisper GPU setup](https://github.com/SYSTRAN/faster-whisper#gpu).
- **Out of GPU memory:** choose smaller models and close other GPU-heavy applications. GPU detection alone does not guarantee that a model fits.
- **File picker unavailable:** paste the MP4 path. Native Linux dialogs need a graphical desktop; the picker opens on the computer running TATRO.
