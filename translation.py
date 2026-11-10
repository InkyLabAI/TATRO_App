"""Compatibility entry point for translating a transcript dataframe."""
from pathlib import Path

from artifacts import Artifacts, write_json
from workflow import translate_video


def run_translation(dataset_df, input_path, output_dir, source_selected, target_selected, config):
    import pandas as pd
    # Older callers pass just a stem and a separate folder.
    input_path = Path(input_path)
    if not input_path.is_absolute():
        input_path = Path(output_dir) / input_path
    artifacts = Artifacts(input_path)
    write_json(artifacts.transcript, dataset_df.to_dict(orient="records"))
    pdf = translate_video(input_path, target_selected, config["qwen_model"], force=True)
    translated = artifacts.path(f"_translated_{target_selected['code']}.json")
    return pd.read_json(translated), translated, pdf
