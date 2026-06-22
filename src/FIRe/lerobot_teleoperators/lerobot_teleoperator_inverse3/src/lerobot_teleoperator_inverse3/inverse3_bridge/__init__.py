from __future__ import annotations

import select
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

    def __init__(
        self,
        inv3_port: str,
        versegrip_port: str,
        *,
        open_timeout_s: float = 8.0,
        response_timeout_s: float = 2.0,
    ) -> None:
        _require_server()
        self._response_timeout_s = response_timeout_s
        self._proc: subprocess.Popen[str] | None = None

        missing_ports = [
            port for port in (inv3_port, versegrip_port)
            if not Path(port).exists()
        ]
        if missing_ports:
            raise FileNotFoundError(
                "[Inv3Server] Device port not found: "
                + ", ".join(missing_ports)
                + ". Check USB connection and udev symlinks, or pass "
                "--inv3_port/--versegrip_port with the current /dev/ttyACM* paths."
            )

        self._proc = subprocess.Popen(
            [str(_SERVER)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        try:
            self._send(f"OPEN {inv3_port} {versegrip_port} 0 0.0")
            resp = self._recv(timeout_s=open_timeout_s)
            if not resp.startswith("OK"):
                raise RuntimeError(f"[Inv3Server] OPEN failed: {resp}")
        except Exception:
            self.close()
            raise

    def _send(self, msg: str) -> None:
        if self._proc is None or self._proc.stdin is None or self._proc.poll() is not None:
            raise RuntimeError("[Inv3Server] server process is not running")
        self._proc.stdin.write(msg + "\n")
        self._proc.stdin.flush()

    def _recv(self, timeout_s: float | None = None) -> str:
        if self._proc is None or self._proc.stdout is None:
            raise RuntimeError("[Inv3Server] server process is not running")
        timeout = self._response_timeout_s if timeout_s is None else timeout_s
        ready, _, _ = select.select([self._proc.stdout], [], [], timeout)
        if not ready:
            raise TimeoutError(
                f"[Inv3Server] No response from inverse3_server within {timeout:.1f}s"
            )
        line = self._proc.stdout.readline()
        if line == "":
            raise RuntimeError("[Inv3Server] server process exited without a response")
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
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._send("CLOSE")
                self._proc.wait(timeout=2.0)
            except Exception:
                self._proc.kill()

    def __del__(self) -> None:
        self.close()


__all__ = ["Inverse3Server", "DeviceState"]
