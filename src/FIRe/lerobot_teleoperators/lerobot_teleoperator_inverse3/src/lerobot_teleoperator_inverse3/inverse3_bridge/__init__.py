from __future__ import annotations

import subprocess
from pathlib import Path
from typing import NamedTuple

import numpy as np

_SERVER = Path(__file__).parent / "inverse3_server"


def _require_server() -> None:
    if not _SERVER.exists():
        raise OSError(
            f"[inverse3] Server executable not found: {_SERVER}\n"
            f"Run `make` inside {_SERVER.parent} to build it first."
        )


class DeviceState(NamedTuple):
    position: np.ndarray    # float32 (3,) metres
    velocity: np.ndarray    # float32 (3,) m/s
    quaternion: np.ndarray  # float32 (4,) WXYZ
    buttons: int            # bitmask
    battery: float          # 0-100


class Inverse3Server:
    """Subprocess wrapper around inv3_server — manages both Inverse3 and VerseGrip."""

    def __init__(self, inv3_port: str, versegrip_port: str) -> None:
        _require_server()
        self._proc = subprocess.Popen(
            [str(_SERVER)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._send(f"OPEN {inv3_port} {versegrip_port}")
        resp = self._recv()
        if not resp.startswith("OK"):
            self._proc.terminate()
            raise RuntimeError(f"[Inv3Server] OPEN failed: {resp}")

    def _send(self, msg: str) -> None:
        self._proc.stdin.write(msg + "\n")
        self._proc.stdin.flush()

    def _recv(self) -> str:
        line = self._proc.stdout.readline()
        return line.rstrip("\n")

    def _parse_state(self, line: str) -> DeviceState:
        parts = line.split()
        if parts[0] != "STATE" or len(parts) != 13:
            raise RuntimeError(f"[Inv3Server] Unexpected response: {line!r}")
        v = [float(x) for x in parts[1:]]
        return DeviceState(
            position=np.array(v[0:3], dtype=np.float32),
            velocity=np.array(v[3:6], dtype=np.float32),
            quaternion=np.array(v[6:10], dtype=np.float32),
            buttons=int(v[10]),
            battery=v[11],
        )

    def get_state(self) -> DeviceState:
        self._send("GET_STATE")
        return self._parse_state(self._recv())

    def send_force(self, force: np.ndarray) -> DeviceState:
        f = force.astype(np.float32)
        self._send(f"SEND_FORCE {f[0]} {f[1]} {f[2]}")
        return self._parse_state(self._recv())

    def close(self) -> None:
        if self._proc.poll() is None:
            try:
                self._send("CLOSE")
                self._proc.wait(timeout=2.0)
            except Exception:
                self._proc.kill()

    def __del__(self) -> None:
        self.close()


__all__ = ["Inverse3Server", "DeviceState"]
