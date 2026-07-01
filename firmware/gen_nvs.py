#!/usr/bin/env python3
"""Generate NVS image with websocket config, preserving existing wifi/board settings.

Secrets (WiFi credentials, websocket token, board UUID) are **not** hardcoded
here. They are read from an untracked env file next to this script
(``nvs_config.env``, gitignored) or from the process environment. Copy
``nvs_config.env.sample`` to ``nvs_config.env`` and fill in real values before
running.
"""
import csv
import os
import subprocess
import sys

_HERE = os.path.dirname(__file__)


def _load_env_file(path):
    """Parse a minimal KEY=VALUE env file (``#`` comments, optional quotes)."""
    cfg = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                cfg[key.strip()] = value.strip().strip('"').strip("'")
    return cfg


_env_file = _load_env_file(os.path.join(_HERE, "nvs_config.env"))


def _cfg(key, default=None, *, required=False):
    """Look up a config value: process env first, then env file, then default."""
    value = os.environ.get(key, _env_file.get(key, default))
    if required and not value:
        sys.exit(
            f"ERROR: {key} is not set. Copy nvs_config.env.sample to "
            f"nvs_config.env and fill it in (or export {key})."
        )
    return value


# Board / wifi / display settings (secrets sourced from env, not tracked)
BOARD_UUID = _cfg("BOARD_UUID", required=True)
WIFI_SSID = _cfg("WIFI_SSID", required=True)
WIFI_PASS = _cfg("WIFI_PASS", required=True)
DISPLAY_THEME = _cfg("DISPLAY_THEME", "light")

# Websocket settings
WS_URL = _cfg("WS_URL", required=True)
WS_TOKEN = _cfg("WS_TOKEN", required=True)

rows = [
    ["key",       "type",      "encoding", "value"],
    # board namespace
    ["board",     "namespace", "",         ""],
    ["uuid",      "data",      "string",   BOARD_UUID],
    # wifi namespace
    ["wifi",      "namespace", "",         ""],
    ["ssid",      "data",      "string",   WIFI_SSID],
    ["password",  "data",      "string",   WIFI_PASS],
    # display namespace
    ["display",   "namespace", "",         ""],
    ["theme",     "data",      "string",   DISPLAY_THEME],
    # websocket namespace
    # Use local_url / local_token (OTA-safe keys not overwritten by api.tenclass.net)
    ["websocket",    "namespace", "",         ""],
    ["local_url",    "data",      "string",   WS_URL],
    ["local_token",  "data",      "string",   WS_TOKEN],
    ["token",        "data",      "string",   WS_TOKEN],
]

csv_path = os.path.join(_HERE, "nvs_config.csv")
out_path = os.path.join(_HERE, "nvs_new.bin")

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(rows)

nvs_gen = os.path.join(
    os.environ.get("IDF_PATH", os.path.expanduser("~/esp/esp-idf")),
    "components", "nvs_flash", "nvs_partition_generator", "nvs_partition_gen.py"
)

result = subprocess.run(
    [sys.executable, nvs_gen, "generate", csv_path, out_path, "0x4000"],
    capture_output=True, text=True
)
print(result.stdout)
if result.returncode != 0:
    print("ERROR:", result.stderr, file=sys.stderr)
    sys.exit(1)
print(f"Generated: {out_path}")
