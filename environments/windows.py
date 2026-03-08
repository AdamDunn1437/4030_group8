"""Compatibility wrapper around legacy_gdb environment with Windows-safe defaults.

On Windows this wrapper can use dolphin-memory-engine (DME) instead of Dolphin's
GDB TCP stub. Set `DOLPHIN_MEMORY_BACKEND` to `dme` or `gdb`.
"""

import inspect
import os
import platform
import struct
from pathlib import Path

from environments import legacy_gdb as _legacy


class _NoOpController:
    """Controller stub for platforms where Dolphin pipe I/O is unavailable."""

    def __init__(self, pipe_path=None):
        self.pipe_path = pipe_path

    def send_action(self, action_idx: int):
        return

    def release_all(self):
        return

    def close(self):
        return


class _DMEMemoryInterface:
    """Memory adapter backed by `dolphin-memory-engine`."""

    def __init__(self, *args, **kwargs):
        self._dme = None
        self._connected = False

    def _resolve_fn(self, names):
        for name in names:
            fn = getattr(self._dme, name, None)
            if callable(fn):
                return fn
        return None

    def _call(self, names, *args):
        fn = self._resolve_fn(names)
        if fn is None:
            return None
        try:
            return fn(*args)
        except TypeError:
            sig = inspect.signature(fn)
            positional = [
                p
                for p in sig.parameters.values()
                if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            ]
            return fn(*args[: len(positional)])

    def connect(self):
        try:
            import dolphin_memory_engine as dme
        except ImportError as exc:
            raise RuntimeError(
                "dolphin-memory-engine is not installed. Run `pip install dolphin-memory-engine`."
            ) from exc

        self._dme = dme

        # Dolphin can take time to initialize process memory; keep retrying.
        for _ in range(120):
            self._call(("hook", "attach", "connect"))
            hooked = self._call(("is_hooked", "is_attached", "connected"))
            self._connected = bool(hooked) if hooked is not None else True
            if self._connected:
                break
            _legacy.time.sleep(0.5)

        if not self._connected:
            raise RuntimeError(
                "dolphin-memory-engine did not report a connected hook. "
                "Run Dolphin and this Python shell with the same privilege level (both normal or both admin)."
            )

    def disconnect(self):
        if self._dme is None:
            return
        self._call(("un_hook", "unhook", "detach", "disconnect"))
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def read_bytes(self, address: int, length: int) -> bytes:
        data = self._call(("read_bytes", "read_memory"), address, length)
        if data is None:
            read_u8 = self._resolve_fn(("read_u8", "read_byte"))
            if read_u8 is None:
                raise RuntimeError("dolphin-memory-engine read function not found.")
            data = bytes(int(read_u8(address + i)) & 0xFF for i in range(length))

        if isinstance(data, bytes):
            return data
        if isinstance(data, bytearray):
            return bytes(data)
        if isinstance(data, list):
            return bytes(data)
        raise RuntimeError(f"Unexpected DME read result type: {type(data)!r}")

    def read_float(self, address: int) -> float:
        value = self._call(("read_f32", "read_float"), address)
        if value is not None:
            return float(value)
        raw = self.read_bytes(address, 4)
        return struct.unpack(">f", raw)[0]

    def read_uint8(self, address: int) -> int:
        value = self._call(("read_u8", "read_byte"), address)
        if value is not None:
            return int(value)
        return struct.unpack(">B", self.read_bytes(address, 1))[0]

    def read_uint16(self, address: int) -> int:
        value = self._call(("read_u16", "read_short"), address)
        if value is not None:
            return int(value)
        return struct.unpack(">H", self.read_bytes(address, 2))[0]

    def read_uint32(self, address: int) -> int:
        value = self._call(("read_u32", "read_word"), address)
        if value is not None:
            return int(value)
        return struct.unpack(">I", self.read_bytes(address, 4))[0]

    def read_pointer(self, address: int) -> int:
        return self.read_uint32(address)

    def write_bytes(self, address: int, data: bytes):
        result = self._call(("write_bytes", "write_memory"), address, data)
        if result is not None:
            return
        write_u8 = self._resolve_fn(("write_u8", "write_byte"))
        if write_u8 is None:
            raise RuntimeError("dolphin-memory-engine write function not found.")
        for i, b in enumerate(data):
            write_u8(address + i, int(b))


def _default_windows_dolphin_exe() -> str | None:
    candidates = [
        Path("C:/Program Files/Dolphin-x64/Dolphin.exe"),
        Path("C:/Program Files/Dolphin/Dolphin.exe"),
        Path("C:/Program Files (x86)/Dolphin-x64/Dolphin.exe"),
        Path("C:/Program Files (x86)/Dolphin/Dolphin.exe"),
    ]
    for path in candidates:
        if path.is_file():
            return str(path)
    return None


if platform.system() == "Windows":
    backend = os.environ.get("DOLPHIN_MEMORY_BACKEND", "dme").strip().lower()

    def _noop_ensure_pipe(self):
        return

    def _noop_wait_for_gdb(self):
        return

    def _windows_wait_for_gdb(self):
        start = _legacy.time.time()
        while _legacy.time.time() - start < self.hook_timeout:
            if not self.is_running:
                rc = self._process.returncode
                stderr = self._process.stderr.read().decode(errors="replace")
                raise RuntimeError(f"Dolphin exited unexpectedly (code {rc}). stderr:\n{stderr}")
            try:
                sock = _legacy.socket.socket(_legacy.socket.AF_INET, _legacy.socket.SOCK_STREAM)
                sock.settimeout(1.0)
                sock.connect(("127.0.0.1", self.gdb_port))
                sock.close()
                return
            except (ConnectionRefusedError, _legacy.socket.timeout, OSError):
                pass
            _legacy.time.sleep(1.0)
        raise TimeoutError(f"Dolphin GDB stub not reachable on port {self.gdb_port} within {self.hook_timeout}s.")

    def _windows_configure_gdb_stub(self):
        user_dir = self.user_dir or str(Path.home() / "Documents" / "Dolphin Emulator")
        config_dir = Path(user_dir) / "Config"
        config_dir.mkdir(parents=True, exist_ok=True)
        debugger_ini = config_dir / "Debugger.ini"
        debugger_ini.write_text(
            "[General]\n"
            f"GDBPort = {self.gdb_port}\n"
            "AutomaticStart = True\n"
            "Enabled = True\n"
            "\n"
            "[GDB]\n"
            f"Port = {self.gdb_port}\n"
            "AutoStart = True\n"
            "Enabled = True\n",
            encoding="utf-8",
        )

    _orig_init = _legacy.DolphinLauncher.__init__

    def _windows_launcher_init(self, *args, **kwargs):
        kwargs.setdefault("batch_mode", False)
        return _orig_init(self, *args, **kwargs)

    _legacy.DolphinLauncher.__init__ = _windows_launcher_init
    _legacy.DolphinLauncher._ensure_pipe = _noop_ensure_pipe
    _legacy.ControllerInterface = _NoOpController

    if backend == "dme":
        _legacy.DolphinLauncher._wait_for_gdb = _noop_wait_for_gdb
        _legacy.GDBMemoryInterface = _DMEMemoryInterface
    else:
        _legacy.DolphinLauncher._wait_for_gdb = _windows_wait_for_gdb
        _legacy.DolphinLauncher._configure_gdb_stub = _windows_configure_gdb_stub


class MarioKartEnv(_legacy.MarioKartEnv):
    """Drop-in env with Windows-friendly defaults for verification."""

    def __init__(self, *args, **kwargs):
        if platform.system() == "Windows":
            if "pipe_path" not in kwargs:
                kwargs["pipe_path"] = os.environ.get("DOLPHIN_PIPE_PATH", "NUL")
            if "dolphin_exe_path" not in kwargs:
                exe_from_env = os.environ.get("DOLPHIN_EXE_PATH", "").strip()
                kwargs["dolphin_exe_path"] = exe_from_env or _default_windows_dolphin_exe() or "Dolphin.exe"
            if "dolphin_user_dir" not in kwargs:
                kwargs["dolphin_user_dir"] = os.environ.get(
                    "DOLPHIN_USER_DIR", str(Path.home() / "Documents" / "Dolphin Emulator")
                )
        super().__init__(*args, **kwargs)


ACTION_MAP = _legacy.ACTION_MAP
NUM_ACTIONS = _legacy.NUM_ACTIONS
MKWiiAddresses = _legacy.MKWiiAddresses
