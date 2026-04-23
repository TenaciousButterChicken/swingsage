"""Tests for the VTrack → GSPro OpenConnect bridge.

Unit tests cover the JSON → `shots` row mapping (including heartbeat and
missing-ClubData cases). Integration tests spin up the real asyncio bridge
plus a mock GSPro on ephemeral ports, exercise the relay round-trip, and
confirm the offline fallback synthesizes a canned ack.

Async bodies are wrapped in `asyncio.run()` so we don't need pytest-asyncio.
"""

from __future__ import annotations

import asyncio
import json
import socket
import sqlite3
import time
from dataclasses import replace
from pathlib import Path

import pytest

from capture.config import Config, load_config
from capture.vtrack_openconnect import (
    _parse_openconnect_shot,
    latest_ball_flight,
    make_server,
)


# ─── Helpers ──────────────────────────────────────────────────────────
def _free_port() -> int:
    """Return a currently-free localhost TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _cfg(db_path: Path, listen_port: int, gspro_port: int) -> Config:
    """Build a Config override for a test — same shape as load_config()."""
    base = load_config()
    return replace(
        base,
        db_path=db_path,
        openconnect_enabled=True,
        openconnect_host="127.0.0.1",
        openconnect_port=listen_port,
        openconnect_gspro_host="127.0.0.1",
        openconnect_gspro_port=gspro_port,
        ball_shot_max_age_sec=120,
    )


def _canonical_shot() -> dict:
    """A realistic OpenConnect v1 shot payload (7-iron, slight draw)."""
    return {
        "DeviceID": "VTrack",
        "Units": "Yards",
        "ShotNumber": 7,
        "APIversion": "1",
        "BallData": {
            "Speed": 120.0,       # mph
            "SpinAxis": -2.5,     # deg (negative = draw for right-hander)
            "TotalSpin": 6500,
            "BackSpin": 6450,
            "SideSpin": -280,
            "HLA": -1.2,          # deg
            "VLA": 18.5,          # deg
            "CarryDistance": 175.0,  # yards
        },
        "ClubData": {
            "Speed": 85.0,        # mph
            "AngleOfAttack": -3.0,
            "FaceToTarget": -1.5,
            "Lie": 0.0,
            "Loft": 34.0,
            "Path": 0.5,
            "SpeedAtImpact": 85.0,
            "VerticalFaceImpact": 1.0,
            "HorizontalFaceImpact": -0.5,
            "ClosureRate": 0.0,
        },
        "ShotDataOptions": {
            "ContainsBallData": True,
            "ContainsClubData": True,
            "LaunchMonitorIsReady": True,
            "LaunchMonitorBallDetected": True,
            "IsHeartBeat": False,
        },
    }


# ─── Unit tests: parsing ──────────────────────────────────────────────
def test_parse_canonical_shot_converts_to_si() -> None:
    raw = _canonical_shot()
    row = _parse_openconnect_shot(raw, json.dumps(raw))

    assert row is not None
    # mph → m/s
    assert row["ball_speed_mps"] == pytest.approx(120.0 * 0.44704)
    assert row["club_speed_mps"] == pytest.approx(85.0 * 0.44704)
    # yards → m
    assert row["carry_distance_m"] == pytest.approx(175.0 * 0.9144)
    # Degrees passed through
    assert row["launch_angle_deg"] == pytest.approx(18.5)
    assert row["launch_direction_deg"] == pytest.approx(-1.2)
    assert row["spin_axis_deg"] == pytest.approx(-2.5)
    # Club delivery
    assert row["club_path_deg"] == pytest.approx(0.5)
    assert row["face_angle_deg"] == pytest.approx(-1.5)
    assert row["attack_angle_deg"] == pytest.approx(-3.0)
    # Smash factor = ball_mps / club_mps (same as mph/mph since the constant cancels)
    assert row["smash_factor"] == pytest.approx(120.0 / 85.0)
    # Identifiers
    assert row["vtrack_shot_id"].startswith("oc-VTrack-7-")
    assert row["raw_json"] == json.dumps(raw)


def test_parse_heartbeat_without_ball_data_returns_none() -> None:
    msg = {"ShotDataOptions": {"IsHeartBeat": True}}
    assert _parse_openconnect_shot(msg, json.dumps(msg)) is None


def test_parse_heartbeat_with_ball_data_is_still_captured() -> None:
    """VTrackToolKit sets IsHeartBeat=true even on real shot frames.
    Presence of BallData must override the heartbeat flag.
    """
    raw = _canonical_shot()
    raw["ShotDataOptions"]["IsHeartBeat"] = True
    row = _parse_openconnect_shot(raw, json.dumps(raw))
    assert row is not None
    assert row["ball_speed_mps"] == pytest.approx(120.0 * 0.44704)


def test_parse_no_ball_data_returns_none() -> None:
    msg = {
        "DeviceID": "VTrack",
        "ShotDataOptions": {"ContainsBallData": False, "IsHeartBeat": False},
    }
    assert _parse_openconnect_shot(msg, json.dumps(msg)) is None


def test_parse_without_club_data_leaves_club_fields_null() -> None:
    raw = _canonical_shot()
    raw.pop("ClubData")
    raw["ShotDataOptions"]["ContainsClubData"] = False

    row = _parse_openconnect_shot(raw, json.dumps(raw))
    assert row is not None
    assert row["club_speed_mps"] is None
    assert row["club_path_deg"] is None
    assert row["face_angle_deg"] is None
    assert row["attack_angle_deg"] is None
    assert row["smash_factor"] is None
    # Ball still populated
    assert row["ball_speed_mps"] == pytest.approx(120.0 * 0.44704)


# ─── latest_ball_flight ───────────────────────────────────────────────
def test_latest_ball_flight_returns_none_when_empty(tmp_db: Path) -> None:
    assert latest_ball_flight(tmp_db, 120) is None


def test_latest_ball_flight_returns_freshest_row(tmp_db: Path) -> None:
    raw = _canonical_shot()
    row = _parse_openconnect_shot(raw, json.dumps(raw))
    assert row is not None
    from capture.vtrack_watcher import insert_shot

    with sqlite3.connect(tmp_db) as conn:
        insert_shot(conn, row)

    fetched = latest_ball_flight(tmp_db, 120)
    assert fetched is not None
    assert fetched["ball_speed_mps"] == pytest.approx(120.0 * 0.44704)
    assert fetched["carry_distance_m"] == pytest.approx(175.0 * 0.9144)


def test_latest_ball_flight_ignores_stale_rows(tmp_db: Path) -> None:
    raw = _canonical_shot()
    row = _parse_openconnect_shot(raw, json.dumps(raw))
    assert row is not None
    from capture.vtrack_watcher import insert_shot

    with sqlite3.connect(tmp_db) as conn:
        insert_shot(conn, row)
        # Force ingested_at back in time so the staleness guard triggers.
        conn.execute(
            "UPDATE shots SET ingested_at = datetime('now', '-10 minutes')"
        )
        conn.commit()

    assert latest_ball_flight(tmp_db, max_age_sec=120) is None


# ─── Integration: full bridge round-trip ──────────────────────────────
async def _close_quietly(writer: asyncio.StreamWriter) -> None:
    try:
        writer.close()
        await asyncio.wait_for(writer.wait_closed(), timeout=1.0)
    except Exception:
        pass


async def _shutdown_server(server: asyncio.Server) -> None:
    try:
        server.close()
        await asyncio.wait_for(server.wait_closed(), timeout=1.0)
    except Exception:
        pass


async def _run_bridge_relay(tmp_db: Path) -> tuple[list[bytes], bytes]:
    """Drive one shot through the bridge with a mock GSPro listening.

    Returns (frames GSPro received, response the fake ToolKit received).
    """
    listen_port = _free_port()
    gspro_port = _free_port()
    cfg = _cfg(tmp_db, listen_port, gspro_port)

    gspro_received: list[bytes] = []

    async def _mock_gspro(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            while True:
                frame = await reader.readline()
                if not frame:
                    return
                gspro_received.append(frame)
                writer.write(
                    json.dumps({"Code": 200, "Message": "OK"}).encode("utf-8") + b"\n"
                )
                await writer.drain()
        finally:
            await _close_quietly(writer)

    gspro_server = await asyncio.start_server(_mock_gspro, "127.0.0.1", gspro_port)
    bridge_server = await make_server(cfg)

    reader, writer = await asyncio.open_connection("127.0.0.1", listen_port)
    try:
        payload = json.dumps(_canonical_shot()).encode("utf-8") + b"\n"
        writer.write(payload)
        await writer.drain()
        response = await asyncio.wait_for(reader.readline(), timeout=2.0)
    finally:
        await _close_quietly(writer)
        await _shutdown_server(bridge_server)
        await _shutdown_server(gspro_server)

    return gspro_received, response


def test_bridge_relays_shot_and_persists_row(tmp_db: Path) -> None:
    gspro_received, response = asyncio.run(_run_bridge_relay(tmp_db))

    assert len(gspro_received) == 1
    relayed = json.loads(gspro_received[0])
    assert relayed["BallData"]["Speed"] == 120.0  # unchanged, units intact on the wire

    parsed_response = json.loads(response)
    assert parsed_response["Code"] == 200

    # Shot must be in the db.
    with sqlite3.connect(tmp_db) as conn:
        row = conn.execute(
            "SELECT ball_speed_mps, carry_distance_m FROM shots"
        ).fetchone()
    assert row is not None
    assert row[0] == pytest.approx(120.0 * 0.44704)
    assert row[1] == pytest.approx(175.0 * 0.9144)


async def _run_bridge_offline(tmp_db: Path) -> bytes:
    listen_port = _free_port()
    gspro_port = _free_port()  # Nothing will listen here.
    cfg = _cfg(tmp_db, listen_port, gspro_port)

    bridge_server = await make_server(cfg)

    reader, writer = await asyncio.open_connection("127.0.0.1", listen_port)
    try:
        payload = json.dumps(_canonical_shot()).encode("utf-8") + b"\n"
        writer.write(payload)
        await writer.drain()
        # Windows delivers this small write to the local client ~2 s after
        # the server's drain completes (asyncio StreamReader polling delay
        # on loopback, unrelated to Nagle — confirmed across Proactor and
        # Selector event loops). A 5 s cap covers the delay comfortably
        # without masking real regressions.
        response = await asyncio.wait_for(reader.readline(), timeout=5.0)
    finally:
        await _close_quietly(writer)
        await _shutdown_server(bridge_server)

    return response


def test_bridge_captures_when_gspro_offline(tmp_db: Path) -> None:
    response = asyncio.run(_run_bridge_offline(tmp_db))

    parsed = json.loads(response)
    assert parsed["Code"] == 200
    assert "SwingSage-captured" in parsed["Message"]

    # Shot still persisted.
    with sqlite3.connect(tmp_db) as conn:
        count = conn.execute("SELECT COUNT(*) FROM shots").fetchone()[0]
    assert count == 1
