#!/usr/bin/env python3
"""Generate NVS image with websocket config, preserving existing wifi/board settings."""
import subprocess, sys, os, csv, tempfile

# Existing values read from NVS dump
BOARD_UUID   = "a497a7e7-38d0-476c-acb8-ea7f3a784d51"
WIFI_SSID    = "WARPSTAR-2770FF-G"
WIFI_PASS    = "***WIFI_PASS_REMOVED***"
DISPLAY_THEME = "light"

# New websocket settings
WS_URL   = "ws://192.168.10.104:8765"
WS_TOKEN = "stackchan-secret"

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

csv_path = os.path.join(os.path.dirname(__file__), "nvs_config.csv")
out_path = os.path.join(os.path.dirname(__file__), "nvs_new.bin")

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(rows)

nvs_gen = os.path.join(
    os.environ.get("IDF_PATH", r"C:\esp\v5.5.4\esp-idf"),
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
