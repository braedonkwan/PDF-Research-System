from __future__ import annotations

import argparse

from .settings import AppConfig, load_settings


def add_common_arguments(
    parser: argparse.ArgumentParser, *, include_model: bool = True
) -> None:
    parser.add_argument(
        "--config",
        default=None,
        help="Path to settings JSON (defaults to LOCAL_LLM_CONFIG or settings.json)",
    )
    if include_model:
        parser.add_argument(
            "--model",
            default=None,
            help="Model alias from settings.json (defaults to default_model)",
        )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List configured model aliases and exit",
    )


def load_cli_settings(config_path: str | None) -> AppConfig:
    return load_settings(config_path)


def print_model_aliases(config: AppConfig) -> None:
    print("Available model aliases:")
    for line in config.formatted_model_lines():
        print(line)


def handle_list_models_flag(config: AppConfig, list_models: bool) -> bool:
    if not list_models:
        return False
    print_model_aliases(config)
    return True
