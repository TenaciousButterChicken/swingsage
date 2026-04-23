"""VTrack → GSPro OpenConnect v1 bridge.

VTrackToolKit v2.0.x (Win32) streams shots straight to GSPro on localhost:921
via the GSPro OpenConnect v1 protocol — newline-delimited JSON, no auth. We
bind that port, capture each shot into SQLite, and relay raw bytes to GSPro
on a relocated port so sim play continues untouched. Two concurrent pumps
shuttle messages in both directions per connection; first one to exit cancels
the other.

If GSPro is offline, the bridge still captures shots and returns a canned
Code:200 ack so ToolKit stays happy.

OpenConnect v1 reference: https://gsprogolf.com/GSProConnectV1.html.
Units on the wire: ball/club speeds in mph, distances in yards. We convert
to SI (m/s, m) on ingest so the `shots` table stays self-consistent.
"""

from __future__ import annotations

import asyncio
import json
import logging
import socket as _socket
import sqlite3
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from capture.config import Config, load_config
from capture.vtrack_watcher import _connect, _ensure_db, insert_shot

log = logging.getLogger(__name__)

_MPH_TO_MPS = 0.44704
_YARDS_TO_M = 0.9144

# Single-line ACK returned when GSPro is unreachable. Matches the shape GSPro
# itself would send (Code/Message at minimum) so ToolKit's response parser
# doesn't choke.
_OFFLINE_ACK = json.dumps(
    {"Code": 200, "Message": "SwingSage-captured, GSPro offline"}
).encode("utf-8") + b"\n"


# ─── JSON → shots-row mapping ─────────────────────────────────────────
def _mph_to_mps(v: Any) -> float | None:
    return float(v) * _MPH_TO_MPS if v is not None else None


def _yards_to_m(v: Any) -> float | None:
    return float(v) * _YARDS_TO_M if v is not None else None


def _as_float(v: Any) -> float | None:
    return float(v) if v is not None else None


def _parse_openconnect_shot(raw: dict, json_text: str) -> dict | None:
    """Flatten an OpenConnect v1 message into a `shots` table row.

    Returns None for heartbeats or any message without a BallData block —
    those are control frames, not shots.

    VTrackToolKit sets IsHeartBeat=true even on real shot frames (non-standard
    vs the GSPro spec), so we treat presence of BallData as the authoritative
    "this is a shot" signal. The heartbeat flag only kicks us out when there's
    no ball data to capture.
    """
    ball = raw.get("BallData")
    if not ball:
        return None

    club = raw.get("ClubData") or {}

    ball_speed_mps = _mph_to_mps(ball.get("Speed"))
    club_speed_mps = _mph_to_mps(club.get("Speed"))

    # Smash factor = ball speed / club speed. OpenConnect doesn't send it
    # directly, but both components are usually present for real shots.
    smash_factor: float | None = None
    if ball_speed_mps is not None and club_speed_mps and club_speed_mps > 0:
        smash_factor = ball_speed_mps / club_speed_mps

    device_id = raw.get("DeviceID") or "openconnect"
    shot_number = raw.get("ShotNumber")
    # ShotNumber resets across ToolKit sessions so we can't use it alone as
    # a dedupe key. uuid suffix guarantees uniqueness; raw_json keeps the
    # original identifiers for forensic lookup.
    shot_id = f"oc-{device_id}-{shot_number}-{uuid.uuid4().hex[:8]}"

    return {
        "vtrack_shot_id": shot_id,
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "club": None,
        "ball_speed_mps": ball_speed_mps,
        "launch_angle_deg": _as_float(ball.get("VLA")),
        "launch_direction_deg": _as_float(ball.get("HLA")),
        "back_spin_rpm": _as_float(ball.get("BackSpin")),
        "side_spin_rpm": _as_float(ball.get("SideSpin")),
        "spin_axis_deg": _as_float(ball.get("SpinAxis")),
        "carry_distance_m": _yards_to_m(ball.get("CarryDistance")),
        "total_distance_m": None,
        "peak_height_m": None,
        "descent_angle_deg": None,
        "club_speed_mps": club_speed_mps,
        "smash_factor": smash_factor,
        "club_path_deg": _as_float(club.get("Path")),
        "face_angle_deg": _as_float(club.get("FaceToTarget")),
        "attack_angle_deg": _as_float(club.get("AngleOfAttack")),
        "impact_location_x_mm": _as_float(club.get("HorizontalFaceImpact")),
        "impact_location_y_mm": _as_float(club.get("VerticalFaceImpact")),
        "raw_json": json_text,
    }


# ─── Latest-shot lookup (used by analysis pipeline) ───────────────────
_BALL_FLIGHT_COLUMNS = (
    "captured_at",
    "ball_speed_mps",
    "carry_distance_m",
    "total_distance_m",
    "launch_angle_deg",
    "launch_direction_deg",
    "back_spin_rpm",
    "side_spin_rpm",
    "spin_axis_deg",
    "club_speed_mps",
    "club_path_deg",
    "face_angle_deg",
    "attack_angle_deg",
    "smash_factor",
)


def latest_ball_flight(db_path: Path, max_age_sec: int) -> dict | None:
    """Return the most recent shot row (subset of columns) within the age
    window, or None if nothing fresh is available. Safe to call from any
    thread — opens its own short-lived connection."""
    if not db_path.exists():
        return None
    cols = ",".join(_BALL_FLIGHT_COLUMNS)
    sql = (
        f"SELECT {cols} FROM shots "
        "WHERE ingested_at >= datetime('now', ?) "
        "ORDER BY id DESC LIMIT 1"
    )
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(sql, (f"-{int(max_age_sec)} seconds",)).fetchone()
    return dict(row) if row else None


# ─── Bridge (asyncio server) ──────────────────────────────────────────
async def _read_line(reader: asyncio.StreamReader) -> bytes:
    """Read one newline-terminated frame. Returns b'' on clean EOF."""
    return await reader.readline()


def _persist_shot(db_path: Path, json_text: str) -> str | None:
    """Parse + insert one frame. Returns the inserted shot_id or None if the
    frame was a heartbeat / options-only message."""
    try:
        raw = json.loads(json_text)
    except json.JSONDecodeError as exc:
        log.warning("Skipping malformed OpenConnect frame: %s", exc)
        return None

    try:
        row = _parse_openconnect_shot(raw, json_text)
    except Exception as exc:
        log.exception("Parser crashed on frame: %s", exc)
        return None
    if row is None:
        return None

    try:
        _ensure_db(db_path)
        with _connect(db_path) as conn:
            insert_shot(conn, row)
    except Exception as exc:
        log.exception("DB insert failed: %s", exc)
        return None

    log.info(
        "Captured shot %s (ball_speed_mps=%s, carry_m=%s)",
        row["vtrack_shot_id"],
        row["ball_speed_mps"],
        row["carry_distance_m"],
    )
    return row["vtrack_shot_id"]


async def _pump_toolkit_to_gspro(
    toolkit_reader: asyncio.StreamReader,
    gspro_writer: asyncio.StreamWriter | None,
    toolkit_writer: asyncio.StreamWriter,
    db_path: Path,
) -> None:
    while True:
        frame = await _read_line(toolkit_reader)
        if not frame:
            return  # ToolKit closed

        json_text = frame.decode("utf-8", errors="replace").strip()
        if json_text:
            # Raw-frame log so we can see heartbeats, welcomes, and anything
            # we don't recognize. Big cap so shot frames aren't truncated.
            preview = json_text if len(json_text) <= 2000 else json_text[:2000] + "...<truncated>"
            log.info("ToolKit frame: %s", preview)
            _persist_shot(db_path, json_text)

        if gspro_writer is not None:
            gspro_writer.write(frame)
            try:
                await gspro_writer.drain()
            except ConnectionError as exc:
                log.warning("GSPro write failed (%s) — falling back to offline ack", exc)
                gspro_writer = None

        if gspro_writer is None:
            # Ack every frame in offline mode — heartbeats too — so ToolKit
            # keeps the connection open and keeps streaming.
            try:
                toolkit_writer.write(_OFFLINE_ACK)
                await toolkit_writer.drain()
            except Exception as exc:
                log.warning("Offline ack write failed: %s", exc)
                return


async def _pump_gspro_to_toolkit(
    gspro_reader: asyncio.StreamReader,
    toolkit_writer: asyncio.StreamWriter,
) -> None:
    while True:
        frame = await _read_line(gspro_reader)
        if not frame:
            return
        toolkit_writer.write(frame)
        try:
            await toolkit_writer.drain()
        except ConnectionError:
            return


def _disable_nagle(writer: asyncio.StreamWriter) -> None:
    """Disable Nagle on the accepted socket. Shot payloads are small (<1 KB)
    and interactive, so piggyback-batching only adds latency."""
    sock = writer.get_extra_info("socket")
    if sock is None:
        return
    try:
        sock.setsockopt(_socket.IPPROTO_TCP, _socket.TCP_NODELAY, 1)
    except OSError:
        pass


async def _handle_toolkit_connection(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    cfg: Config,
) -> None:
    peer = writer.get_extra_info("peername")
    log.info("ToolKit connected from %s", peer)
    _disable_nagle(writer)

    gspro_reader: asyncio.StreamReader | None = None
    gspro_writer: asyncio.StreamWriter | None = None
    try:
        gspro_reader, gspro_writer = await asyncio.open_connection(
            cfg.openconnect_gspro_host, cfg.openconnect_gspro_port
        )
        _disable_nagle(gspro_writer)
        log.info(
            "Relaying to GSPro at %s:%d",
            cfg.openconnect_gspro_host,
            cfg.openconnect_gspro_port,
        )
    except (ConnectionError, OSError) as exc:
        log.warning(
            "GSPro unreachable at %s:%d (%s) — capturing only, synthesizing acks",
            cfg.openconnect_gspro_host,
            cfg.openconnect_gspro_port,
            exc,
        )

    tasks: list[asyncio.Task] = [
        asyncio.create_task(
            _pump_toolkit_to_gspro(reader, gspro_writer, writer, cfg.db_path)
        )
    ]
    if gspro_reader is not None:
        tasks.append(asyncio.create_task(_pump_gspro_to_toolkit(gspro_reader, writer)))

    try:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        for task in pending:
            try:
                await task
            except asyncio.CancelledError:
                pass
    finally:
        for w in (writer, gspro_writer):
            if w is None:
                continue
            try:
                w.close()
            except Exception:
                pass
            # wait_closed can stall on Windows when the peer already dropped the
            # socket; cap the wait so a single weird disconnect can't wedge the
            # whole bridge.
            try:
                await asyncio.wait_for(w.wait_closed(), timeout=1.0)
            except (asyncio.TimeoutError, Exception):
                pass
        log.info("ToolKit disconnected from %s", peer)


async def make_server(cfg: Config) -> asyncio.Server:
    """Start the OpenConnect listener and return the `asyncio.Server` instance
    so the caller can manage its lifecycle (close / wait_closed).

    Useful for tests and for the FastAPI startup hook that wants to track the
    server directly rather than a long-running task.
    """
    _ensure_db(cfg.db_path)

    async def _on_conn(r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
        await _handle_toolkit_connection(r, w, cfg)

    server = await asyncio.start_server(_on_conn, cfg.openconnect_host, cfg.openconnect_port)
    sockname = ", ".join(str(s.getsockname()) for s in server.sockets or [])
    log.info("OpenConnect bridge listening on %s", sockname)
    return server


async def serve(cfg: Config) -> None:
    """Long-running asyncio server. Returns only on cancellation."""
    server = await make_server(cfg)
    try:
        async with server:
            await server.serve_forever()
    except asyncio.CancelledError:
        log.info("OpenConnect bridge shutting down")
        raise


# ─── CLI entry (smoke test / standalone run) ──────────────────────────
def main() -> None:
    cfg = load_config()
    logging.basicConfig(
        level=cfg.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    if not cfg.openconnect_enabled:
        log.warning(
            "SWINGSAGE_OPENCONNECT_ENABLED is false on %s — enable it explicitly to run the bridge.",
            sys.platform,
        )
    try:
        asyncio.run(serve(cfg))
    except KeyboardInterrupt:
        log.info("Bridge stopped by user")


if __name__ == "__main__":
    main()
