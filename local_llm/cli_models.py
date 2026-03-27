from __future__ import annotations

import argparse

from huggingface_hub import hf_hub_download

from .cli_common import add_common_arguments, handle_list_models_flag, load_cli_settings
from .settings import ModelConfig


def download_model(model: ModelConfig) -> str:
    if not model.repo_id or not model.filename or not model.local_dir:
        raise ValueError(
            f"Model '{model.alias}' is missing repo_id/filename/local_dir in settings.json."
        )
    return hf_hub_download(
        repo_id=model.repo_id,
        filename=model.filename,
        local_dir=str(model.local_dir),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download GGUF models from settings.json")
    add_common_arguments(parser)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all configured models",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        config = load_cli_settings(args.config)
        if handle_list_models_flag(config, args.list_models):
            return 0

        if args.all:
            for alias in config.model_aliases():
                model = config.models[alias]
                print(f"Downloading {alias}...")
                print(download_model(model))
            return 0

        model = config.get_model(args.model)
        model_path = download_model(model)
        print(model_path)
        return 0
    except Exception as exc:
        print(f"Error: {exc}")
        return 1
