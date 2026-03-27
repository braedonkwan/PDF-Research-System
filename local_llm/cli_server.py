from __future__ import annotations

import argparse
import subprocess

from .cli_common import add_common_arguments, handle_list_models_flag, load_cli_settings
from .llama_server import build_server_command, validate_server_paths
from .model_overrides import apply_model_server_overrides
from .settings import AppConfig


def run_server(config: AppConfig, model_alias: str | None, dry_run: bool = False) -> int:
    model = config.get_model(model_alias)
    server = apply_model_server_overrides(config.server, model)
    validate_server_paths(server, model)
    cmd, warnings = build_server_command(server, model)

    print(f"Starting server with model alias: {model.alias} ({model.api_name})")
    for warning in warnings:
        print(f"Note: {warning}")
    print("Command:")
    print(" ".join(cmd))
    if dry_run:
        return 0

    process = subprocess.Popen(cmd)
    try:
        return process.wait()
    except KeyboardInterrupt:
        process.terminate()
        try:
            return process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            return process.wait()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Start llama-server using settings.json")
    add_common_arguments(parser)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved llama-server command and exit",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        config = load_cli_settings(args.config)
        if handle_list_models_flag(config, args.list_models):
            return 0
        return run_server(
            config=config,
            model_alias=args.model,
            dry_run=args.dry_run,
        )
    except Exception as exc:
        print(f"Error: {exc}")
        return 1
