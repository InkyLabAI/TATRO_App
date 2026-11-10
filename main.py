"""Command-line entry point; outputs live in a video-named folder beside the input video."""
import argparse
import os
from pathlib import Path

from transcript_data import raw_transcript_path
from artifacts import Artifacts
from languages import LANGUAGES
from local_llm import LLM_MODELS
from retrieval import EMBEDDING_MODELS
from workflow import index_video, transcribe, translate_video, summary_video


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--language', required=True, choices=[v['code'] for v in LANGUAGES.values()])
    parser.add_argument('--target', choices=[v['code'] for v in LANGUAGES.values()])
    parser.add_argument('--llm-model', default=LLM_MODELS[0], help='Hugging Face model ID or local folder')
    parser.add_argument('--embedding-model', choices=EMBEDDING_MODELS, default=EMBEDDING_MODELS[0])
    parser.add_argument('--hf-token', default=os.environ.get('HF_TOKEN'))
    parser.add_argument('--no-correction', action='store_true')
    parser.add_argument('--summary', action='store_true')
    parser.add_argument('--force', action='store_true', help='Regenerate existing results')
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error('Input file does not exist.')
    languages = {v['code']: v for v in LANGUAGES.values()}
    source = languages[args.language]
    target = languages[args.target or args.language]
    if args.force or not Artifacts(args.input).transcript.exists():
        if not args.hf_token and raw_transcript_path(Artifacts(args.input)) is None:
            parser.error('Set HF_TOKEN or supply --hf-token for diarization.')
        config = {'whisper_model': 'large-v3', 'diarization_model': 'pyannote/speaker-diarization-3.1',
                  'qwen_model': args.llm_model, 'embedding_model': args.embedding_model,
                  'max_chunk_duration': 30, 'use_qwen_correction': not args.no_correction}
        transcribe(args.input, source, args.hf_token, config)
    else:
        index_video(args.input, args.embedding_model)
    if args.target:
        print(translate_video(args.input, target, args.llm_model, force=args.force))
    if args.summary:
        print(summary_video(args.input, target, args.llm_model, force=args.force))
    print(f'Results saved in {Artifacts(args.input).directory}')


if __name__ == '__main__':
    main()
