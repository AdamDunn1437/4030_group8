"""
environment.py — Gymnasium-style Environment Wrapper for Mario Kart Wii (Dolphin Emulator)
This module provides a custom Gymnasium environment that interfaces with the Dolphin
emulator running Mario Kart Wii. It reads game state via Dolphin's built-in GDB stub
(TCP socket), sends controller inputs via named pipe, and computes a dense reward
signal for RL training.
Memory Access Approach:
    Instead of dolphin-memory-engine (which requires macOS SIP/debug entitlements),
    we use Dolphin's built-in GDB remote debugging stub. This works by:
      1. Enabling the GDB stub in Dolphin's config (Debugger.ini)
      2. Dolphin listens on a TCP port (default 2345)
      3. We connect and read memory using the GDB remote protocol
      4. No special OS permissions needed — Dolphin serves the data itself
API Equivalences (for non-standard Gymnasium environments):
    ┌─────────────────────────┬──────────────────────────────────────────────────┐
    │ Gymnasium Method         │ Dolphin/MKWii Equivalent                        │
    ├─────────────────────────┼──────────────────────────────────────────────────┤
    │ env.reset()             │ Load savestate → read initial observation        │
    │ env.step(action)        │ Write controller input → advance N frames →     │
    │                         │   read new state → compute reward → check done  │
    │ env.observation_space   │ Feature vector from emulator memory addresses   │
    │ env.action_space        │ Discrete set of controller button combinations  │
    │ env.close()             │ Terminate Dolphin process / release pipe         │
    └─────────────────────────┴──────────────────────────────────────────────────┘
Dependencies:
    - gymnasium
    - numpy
    - torch
Usage:
    from environment import MarioKartEnv
    env = MarioKartEnv(iso_path="/path/to/Mario Kart Wii.rvz")
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import struct
import socket
import time
import subprocess
import os
import sys
import platform
import logging

# Auto-load .env file if python-dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ===========================================================================
# Memory Address Table — Mario Kart Wii (NTSC-U / RMCE01)
# ===========================================================================
# These addresses point into the game's RAM via Dolphin. They may vary by
# game region (PAL, NTSC-J). Verify with Dolphin's memory viewer or Cheat
# Engine before training.  Addresses are given as offsets from MEM1 base.
# ---------------------------------------------------------------------------
class MKWiiAddresses:
    """
    Known memory addresses for Mario Kart Wii (NTSC-U).
    These were gathered from the MKWii modding community and Dolphin RAM
    watches. You MUST verify these on your specific game ISO + Dolphin
    version, as they can shift between builds.
    All addresses are 32-bit offsets into MEM1 (0x80000000 base).
    """
    PLAYER_BASE_PTR     = 0x809C18F8
    SPEED_OFFSET        = 0x20
    POS_X_OFFSET        = 0x68
    POS_Y_OFFSET        = 0x6C
    POS_Z_OFFSET        = 0x70
    FACING_DIR_OFFSET   = 0xF0

    LAP_COMPLETION      = 0x809BD730
    LAP_COUNT           = 0x809BD734
    CHECKPOINT_IDX      = 0x809BD738
    MAX_CHECKPOINTS     = 0x809BD73A

    OFF_TRACK_FLAG      = 0x809BD740
    COLLISION_FLAG      = 0x809BD744
    ITEM_ID             = 0x809BD750
    RACE_TIMER_MS       = 0x809BD760


# ===========================================================================
# Action Map
# ===========================================================================
ACTION_MAP = {
    0: {"name": "ACCELERATE",   "A": 1, "B": 0, "MAIN": (0.5, 0.5)},
    1: {"name": "BRAKE",        "A": 0, "B": 1, "MAIN": (0.5, 0.5)},
    2: {"name": "STEER_LEFT",   "A": 1, "B": 0, "MAIN": (0.0, 0.5)},
    3: {"name": "STEER_RIGHT",  "A": 1, "B": 0, "MAIN": (1.0, 0.5)},
    4: {"name": "DRIFT",        "A": 1, "B": 0, "R": 1, "MAIN": (0.5, 0.5)},
    5: {"name": "USE_ITEM",     "A": 1, "B": 0, "L": 1, "MAIN": (0.5, 0.5)},
}
NUM_ACTIONS = len(ACTION_MAP)


# ===========================================================================
# Reward Shaping Coefficients (from Phase 1 proposal)
# ===========================================================================
REWARD_PROGRESS_COEFF   =  1.0
REWARD_SPEED_COEFF      =  0.05
REWARD_COLLISION_PENALTY = -5.0
REWARD_OFFTRACK_PENALTY  = -10.0
REWARD_LAP_BONUS         = 100.0


# ===========================================================================
# Configuration
# ===========================================================================
FRAMES_PER_STEP     = 4
MAX_STEPS_PER_EP    = 5000
DOLPHIN_PIPE_PATH   = "/tmp/dolphin_pipe"
GDB_PORT            = 2345      # Dolphin GDB stub port

PLATFORM = platform.system()

def _default_dolphin_exe() -> str:
    env = os.environ.get("DOLPHIN_EXE_PATH", "")
    if env:
        return env
    if PLATFORM == "Darwin":
        candidates = [
            "/Applications/Dolphin.app/Contents/MacOS/Dolphin",
            os.path.expanduser("~/Applications/Dolphin.app/Contents/MacOS/Dolphin"),
            "/opt/homebrew/bin/dolphin-emu",
            "/usr/local/bin/dolphin-emu",
        ]
        for path in candidates:
            if os.path.isfile(path):
                return path
        return "/Applications/Dolphin.app/Contents/MacOS/Dolphin"
    else:
        return "dolphin-emu"

def _default_dolphin_user_dir() -> str:
    env = os.environ.get("DOLPHIN_USER_DIR", "")
    if env:
        return env
    if PLATFORM == "Darwin":
        return os.path.expanduser("~/Library/Application Support/Dolphin")
    else:
        return os.path.expanduser("~/.dolphin-emu")

def _default_gfx_backend() -> str:
    env = os.environ.get("DOLPHIN_GFX_BACKEND", "")
    if env:
        return env
    return "Null"

DOLPHIN_EXE_PATH    = _default_dolphin_exe()
DOLPHIN_ISO_PATH    = os.environ.get("DOLPHIN_ISO_PATH", "")
DOLPHIN_USER_DIR    = _default_dolphin_user_dir()
DOLPHIN_SAVESTATE   = os.environ.get("DOLPHIN_SAVESTATE", "")
DOLPHIN_GFX_BACKEND = _default_gfx_backend()
DOLPHIN_BATCH_MODE  = True
DOLPHIN_HOOK_TIMEOUT = 90  # Longer timeout — game needs to fully boot


# ===========================================================================
# GDB Remote Protocol — Memory Interface
# ===========================================================================
class GDBMemoryInterface:
    """
    Reads Dolphin's emulated memory via the GDB remote debugging stub.
    Dolphin's GDB stub listens on a TCP port and speaks the GDB remote
    serial protocol. We use a tiny subset:
      - 'c'              → continue execution (game was paused at boot)
      - 'm addr,length'  → read `length` bytes from `addr`, returns hex
    This approach requires NO special OS permissions because Dolphin
    itself serves the memory data over a normal TCP socket.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = GDB_PORT, timeout: float = 5.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._sock = None
        self._connected = False

    def connect(self):
        """Connect to Dolphin's GDB stub."""
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(self.timeout)
        self._sock.connect((self.host, self.port))
        self._connected = True
        logging.info(f"GDBMemoryInterface: connected to Dolphin GDB stub at {self.host}:{self.port}")

        # Drain any initial data Dolphin sends on connect
        try:
            self._sock.recv(4096)
        except socket.timeout:
            pass

        # Resume the game — GDB stub pauses emulation on connect
        self._send_command("c")
        # Give the game a moment to resume
        time.sleep(0.5)

    def disconnect(self):
        """Close the socket."""
        if self._sock:
            try:
                # Send detach command so Dolphin continues running
                self._send_packet("D")
            except Exception:
                pass
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
            self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def _checksum(self, data: str) -> str:
        """Compute GDB protocol checksum (sum of bytes mod 256, as 2-char hex)."""
        return f"{sum(ord(c) for c in data) % 256:02x}"

    def _send_packet(self, data: str) -> str:
        """Send a GDB packet and return the response data."""
        packet = f"${data}#{self._checksum(data)}"
        self._sock.sendall(packet.encode())

        # Read response: expect '+' ack, then '$data#xx'
        response = b""
        try:
            while True:
                chunk = self._sock.recv(4096)
                if not chunk:
                    break
                response += chunk
                # Check if we got a complete packet
                if b"#" in response and len(response) >= response.index(b"#") + 3:
                    break
        except socket.timeout:
            pass

        response_str = response.decode(errors="replace")

        # Extract data between $ and #
        if "$" in response_str and "#" in response_str:
            start = response_str.index("$") + 1
            end = response_str.index("#", start)
            return response_str[start:end]
        return response_str

    def _send_command(self, cmd: str):
        """Send a command without waiting for a data response."""
        packet = f"${cmd}#{self._checksum(cmd)}"
        self._sock.sendall(packet.encode())
        # Brief drain
        try:
            self._sock.settimeout(0.3)
            self._sock.recv(4096)
        except socket.timeout:
            pass
        finally:
            self._sock.settimeout(self.timeout)

    def _interrupt(self):
        """Send a break/interrupt to pause execution so we can read memory."""
        # GDB interrupt is a single 0x03 byte (Ctrl-C)
        self._sock.sendall(b"\x03")
        # Wait for the stop reply
        try:
            self._sock.settimeout(2.0)
            self._sock.recv(4096)
        except socket.timeout:
            pass
        finally:
            self._sock.settimeout(self.timeout)

    def read_bytes(self, address: int, length: int) -> bytes:
        """
        Read `length` bytes from emulated memory at `address`.
        We briefly pause emulation, read, then resume. This adds a tiny
        amount of latency but ensures consistent reads.
        """
        # Interrupt to pause
        self._interrupt()

        # Send memory read: m addr,length
        cmd = f"m{address:x},{length:x}"
        hex_data = self._send_packet(cmd)

        # Resume execution
        self._send_command("c")

        # Parse hex response into bytes
        if hex_data.startswith("E") or not hex_data:
            # Error response or empty — return zeros
            logging.warning(f"GDB memory read error at 0x{address:08X}: {hex_data}")
            return b"\x00" * length

        try:
            return bytes.fromhex(hex_data)
        except ValueError:
            logging.warning(f"GDB invalid hex response at 0x{address:08X}: {hex_data[:40]}")
            return b"\x00" * length

    # --- Convenience read helpers (big-endian, matching Wii byte order) ---
    def read_float(self, address: int) -> float:
        raw = self.read_bytes(address, 4)
        return struct.unpack(">f", raw)[0]

    def read_uint8(self, address: int) -> int:
        raw = self.read_bytes(address, 1)
        return struct.unpack(">B", raw)[0]

    def read_uint16(self, address: int) -> int:
        raw = self.read_bytes(address, 2)
        return struct.unpack(">H", raw)[0]

    def read_uint32(self, address: int) -> int:
        raw = self.read_bytes(address, 4)
        return struct.unpack(">I", raw)[0]

    def read_pointer(self, address: int) -> int:
        return self.read_uint32(address)

    def write_bytes(self, address: int, data: bytes):
        """Write bytes to emulated memory."""
        self._interrupt()
        hex_data = data.hex()
        cmd = f"M{address:x},{len(data):x}:{hex_data}"
        self._send_packet(cmd)
        self._send_command("c")


# ===========================================================================
# Dolphin Launcher
# ===========================================================================
class DolphinLauncher:
    """
    Manages the lifecycle of a Dolphin emulator process with GDB stub enabled.
    On start():
      1. Writes Dolphin config to enable the GDB stub
      2. Creates the named pipe for controller input
      3. Launches Dolphin with the game ISO
      4. Waits for the GDB stub to accept TCP connections
      5. Sends A presses via pipe to navigate past startup screens
    """

    def __init__(
        self,
        exe_path: str = DOLPHIN_EXE_PATH,
        iso_path: str = DOLPHIN_ISO_PATH,
        user_dir: str = DOLPHIN_USER_DIR,
        gfx_backend: str = DOLPHIN_GFX_BACKEND,
        pipe_path: str = DOLPHIN_PIPE_PATH,
        batch_mode: bool = DOLPHIN_BATCH_MODE,
        hook_timeout: float = DOLPHIN_HOOK_TIMEOUT,
        gdb_port: int = GDB_PORT,
    ):
        self.exe_path = exe_path
        self.iso_path = iso_path
        self.user_dir = user_dir
        self.gfx_backend = gfx_backend
        self.pipe_path = pipe_path
        self.batch_mode = batch_mode
        self.hook_timeout = hook_timeout
        self.gdb_port = gdb_port
        self._process = None

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self):
        """Launch Dolphin with GDB stub enabled and wait until connectable."""
        if not self.iso_path:
            raise FileNotFoundError(
                "No ISO path provided. Set DOLPHIN_ISO_PATH env var or pass iso_path=."
            )
        if not os.path.isfile(self.iso_path):
            raise FileNotFoundError(f"ISO not found at: {self.iso_path}")

        # Enable GDB stub in Dolphin's config
        self._configure_gdb_stub()

        # Create named pipe for controller input
        self._ensure_pipe()

        # Build command
        cmd = [self.exe_path]
        if self.batch_mode:
            cmd.append("--batch")
        cmd.extend(["--exec", self.iso_path])
        if self.gfx_backend:
            cmd.extend(["--video_backend", self.gfx_backend])
        if self.user_dir:
            cmd.extend(["--user", self.user_dir])

        logging.info(f"DolphinLauncher: starting with command:\n  {' '.join(cmd)}")

        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError:
            if PLATFORM == "Darwin":
                hint = (
                    f"Dolphin not found at '{self.exe_path}'. "
                    f"Install from https://dolphin-emu.org/download/ and put in /Applications."
                )
            else:
                hint = f"Dolphin not found at '{self.exe_path}'. Install or set DOLPHIN_EXE_PATH."
            raise FileNotFoundError(hint)

        logging.info(f"DolphinLauncher: Dolphin started (PID {self._process.pid}).")

        # Wait for GDB stub to accept connections
        self._wait_for_gdb()

    def _configure_gdb_stub(self):
        """
        Write/update Dolphin's Debugger.ini to enable the GDB stub on our port.
        This is read by Dolphin on startup.
        """
        user_dir = self.user_dir or _default_dolphin_user_dir()
        config_dir = os.path.join(user_dir, "Config")
        os.makedirs(config_dir, exist_ok=True)

        debugger_ini = os.path.join(config_dir, "Debugger.ini")

        # Write a minimal Debugger.ini enabling the GDB stub
        config_content = (
            "[General]\n"
            f"GDBPort = {self.gdb_port}\n"
            "AutomaticStart = True\n"
        )

        with open(debugger_ini, "w") as f:
            f.write(config_content)

        logging.info(f"DolphinLauncher: GDB stub configured on port {self.gdb_port} in {debugger_ini}")

    def _ensure_pipe(self):
        """Create the named pipe (FIFO) for controller input if needed."""
        if os.path.exists(self.pipe_path):
            import stat
            mode = os.stat(self.pipe_path).st_mode
            if not stat.S_ISFIFO(mode):
                os.remove(self.pipe_path)
                os.mkfifo(self.pipe_path)
        else:
            os.mkfifo(self.pipe_path)
            logging.info(f"DolphinLauncher: created controller pipe at {self.pipe_path}")

    def _wait_for_gdb(self):
        """
        Block until Dolphin's GDB stub is accepting TCP connections.
        Also sends A presses through the pipe to get past startup screens.
        """
        logging.info(
            f"DolphinLauncher: waiting up to {self.hook_timeout}s for GDB stub on port {self.gdb_port}..."
        )
        start = time.time()
        attempt = 0
        pipe_fd = None

        while time.time() - start < self.hook_timeout:
            if not self.is_running:
                rc = self._process.returncode
                stderr = self._process.stderr.read().decode(errors="replace")
                raise RuntimeError(
                    f"Dolphin exited unexpectedly (code {rc}). stderr:\n{stderr}"
                )

            # Try opening the pipe to send A presses past startup screens
            if pipe_fd is None:
                try:
                    pipe_fd = os.open(self.pipe_path, os.O_WRONLY | os.O_NONBLOCK)
                except OSError:
                    pass

            if pipe_fd is not None:
                try:
                    os.write(pipe_fd, b"SET A 1\n")
                    time.sleep(0.1)
                    os.write(pipe_fd, b"SET A 0\n")
                except OSError:
                    try:
                        os.close(pipe_fd)
                    except OSError:
                        pass
                    pipe_fd = None

            # Try connecting to GDB stub
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1.0)
                sock.connect(("127.0.0.1", self.gdb_port))
                sock.close()
                elapsed = time.time() - start
                logging.info(f"DolphinLauncher: GDB stub is ready after {elapsed:.1f}s.")
                if pipe_fd is not None:
                    try:
                        os.close(pipe_fd)
                    except OSError:
                        pass
                return
            except (ConnectionRefusedError, socket.timeout, OSError):
                pass

            attempt += 1
            if attempt % 10 == 0:
                elapsed = time.time() - start
                logging.info(
                    f"DolphinLauncher: still waiting for GDB... ({elapsed:.0f}s elapsed)"
                )

            time.sleep(1.0)

        if pipe_fd is not None:
            try:
                os.close(pipe_fd)
            except OSError:
                pass

        raise TimeoutError(
            f"Dolphin GDB stub not reachable on port {self.gdb_port} within {self.hook_timeout}s.\n"
            f"  Check that Dolphin started and the game loaded.\n"
            f"  Config written to: {os.path.join(self.user_dir or _default_dolphin_user_dir(), 'Config', 'Debugger.ini')}"
        )

    def stop(self):
        """Gracefully terminate Dolphin (SIGTERM, then SIGKILL after 5s)."""
        if self._process is None:
            return
        if self.is_running:
            logging.info(f"DolphinLauncher: terminating Dolphin (PID {self._process.pid})...")
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=3)
        self._process = None

        if os.path.exists(self.pipe_path):
            try:
                os.remove(self.pipe_path)
            except OSError:
                pass

    def restart(self):
        """Stop and re-launch Dolphin."""
        self.stop()
        time.sleep(1)
        self.start()


# ===========================================================================
# Controller Interface
# ===========================================================================
class ControllerInterface:
    """
    Sends discrete actions to Dolphin via named pipe.
    Dolphin must be configured: Controller Settings → Device → Pipe/0/dolphin_pipe.
    """

    def __init__(self, pipe_path: str = DOLPHIN_PIPE_PATH):
        self.pipe_path = pipe_path
        self._pipe_fd = None
        self._open_pipe()

    def _open_pipe(self):
        if not os.path.exists(self.pipe_path):
            os.mkfifo(self.pipe_path)
        try:
            self._pipe_fd = os.open(self.pipe_path, os.O_WRONLY | os.O_NONBLOCK)
            logging.info(f"ControllerInterface: opened pipe {self.pipe_path}")
        except OSError as e:
            raise RuntimeError(
                f"Could not open controller pipe at {self.pipe_path}. "
                f"Is Dolphin running and configured to read from this pipe? Error: {e}"
            )

    def send_action(self, action_idx: int):
        if action_idx not in ACTION_MAP:
            raise ValueError(f"Invalid action: {action_idx}. Must be 0–{NUM_ACTIONS-1}.")
        action = ACTION_MAP[action_idx]
        commands = []
        for btn in ["A", "B", "R", "L"]:
            commands.append(f"SET {btn} {action.get(btn, 0)}")
        if "MAIN" in action:
            x, y = action["MAIN"]
            commands.append(f"SET MAIN {x:.4f} {y:.4f}")
        msg = "\n".join(commands) + "\n"
        try:
            os.write(self._pipe_fd, msg.encode())
        except OSError as e:
            logging.error(f"Pipe write failed: {e}")

    def release_all(self):
        try:
            os.write(self._pipe_fd, b"SET A 0\nSET B 0\nSET R 0\nSET L 0\nSET MAIN 0.5 0.5\n")
        except OSError:
            pass

    def close(self):
        if self._pipe_fd is not None:
            try:
                os.close(self._pipe_fd)
            except OSError:
                pass


# ===========================================================================
# Main Environment
# ===========================================================================
class MarioKartEnv(gym.Env):
    """
    Gymnasium-compatible environment for Mario Kart Wii via Dolphin emulator.
    On construction:
      1. Launches Dolphin with GDB stub enabled
      2. Waits for the GDB stub to accept connections
      3. Connects the GDB memory reader and controller pipe
    On close():
      1. Disconnects from GDB
      2. Terminates Dolphin
    Parameters
    ----------
    iso_path : str
        Path to the Mario Kart Wii ISO/RVZ/WBFS file.
    frames_per_step : int
        Emulator frames to advance per RL step.
    max_steps : int
        Max steps before episode truncation.
    gfx_backend : str
        "Null" = headless, "Metal" = visible (macOS), "Vulkan"/"OGL" (Linux).
    gdb_port : int
        TCP port for Dolphin's GDB stub (default 2345).
    """

    metadata = {"render_modes": ["human", None], "render_fps": 15}

    def __init__(
        self,
        iso_path: str = DOLPHIN_ISO_PATH,
        frames_per_step: int = FRAMES_PER_STEP,
        max_steps: int = MAX_STEPS_PER_EP,
        render_mode: str = None,
        dolphin_exe_path: str = DOLPHIN_EXE_PATH,
        dolphin_user_dir: str = DOLPHIN_USER_DIR,
        gfx_backend: str = DOLPHIN_GFX_BACKEND,
        pipe_path: str = DOLPHIN_PIPE_PATH,
        gdb_port: int = GDB_PORT,
    ):
        super().__init__()

        self.frames_per_step = frames_per_step
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.pipe_path = pipe_path
        self.gdb_port = gdb_port

        # Observation: [speed, d_center, heading_error, d_wall, progress, item_state]
        self.observation_space = spaces.Box(
            low=np.array([0.0, -1.0, -np.pi, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0,  1.0,  np.pi, 1.0, 1.0, 1.0], dtype=np.float32),
            shape=(6,),
            dtype=np.float32,
        )

        # Actions: ACCELERATE, BRAKE, STEER_LEFT, STEER_RIGHT, DRIFT, USE_ITEM
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        # Launch Dolphin with GDB stub
        self.dolphin = DolphinLauncher(
            exe_path=dolphin_exe_path,
            iso_path=iso_path,
            user_dir=dolphin_user_dir,
            gfx_backend=gfx_backend,
            pipe_path=self.pipe_path,
            gdb_port=self.gdb_port,
        )
        self.dolphin.start()

        # Connect to GDB stub for memory reads
        self.memory = GDBMemoryInterface(port=self.gdb_port)
        self.memory.connect()

        # Connect controller pipe
        self.controller = ControllerInterface(pipe_path=self.pipe_path)

        # Episode state
        self._step_count = 0
        self._prev_progress = 0.0
        self._prev_speed = 0.0
        self._total_collisions = 0
        self._total_offtrack_steps = 0
        self._lap_count = 0
        self._episode_start_time = None

        # Normalization constants (tune per observed ranges)
        self.MAX_SPEED = 120.0
        self.MAX_CENTER_DIST = 500.0

        logging.info("MarioKartEnv initialized.")
        logging.info(f"  Observation space: {self.observation_space}")
        logging.info(f"  Action space:      {self.action_space}")

    # -------------------------------------------------------------------
    # Gymnasium API
    # -------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._step_count = 0
        self._prev_progress = 0.0
        self._prev_speed = 0.0
        self._total_collisions = 0
        self._total_offtrack_steps = 0
        self._lap_count = 0
        self._episode_start_time = time.time()

        # Crash recovery
        if not self.dolphin.is_running:
            logging.warning("Dolphin died. Restarting...")
            self.dolphin.restart()
            self.memory = GDBMemoryInterface(port=self.gdb_port)
            self.memory.connect()
            self.controller = ControllerInterface(pipe_path=self.pipe_path)

        self._load_savestate()
        time.sleep(0.1)
        self.controller.release_all()

        obs = self._read_observation()
        return obs, self._get_info()

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        self.controller.send_action(action)
        self._advance_frames(self.frames_per_step)

        obs = self._read_observation()
        reward, reward_info = self._compute_reward(obs)

        self._step_count += 1
        terminated = self._check_terminated()
        truncated = self._step_count >= self.max_steps

        info = self._get_info()
        info.update(reward_info)
        return obs, reward, terminated, truncated, info

    def close(self):
        self.controller.release_all()
        self.controller.close()
        self.memory.disconnect()
        self.dolphin.stop()
        logging.info("MarioKartEnv closed.")

    def render(self):
        pass  # Dolphin renders its own window

    # -------------------------------------------------------------------
    # State Reading
    # -------------------------------------------------------------------

    def _read_observation(self) -> np.ndarray:
        mem = self.memory
        addr = MKWiiAddresses

        raw_speed    = mem.read_float(addr.PLAYER_BASE_PTR + addr.SPEED_OFFSET)
        raw_pos_x    = mem.read_float(addr.PLAYER_BASE_PTR + addr.POS_X_OFFSET)
        raw_facing   = mem.read_float(addr.PLAYER_BASE_PTR + addr.FACING_DIR_OFFSET)
        raw_progress = mem.read_float(addr.LAP_COMPLETION)
        raw_item     = mem.read_uint8(addr.ITEM_ID)
        off_track    = mem.read_uint8(addr.OFF_TRACK_FLAG)
        collision    = mem.read_uint8(addr.COLLISION_FLAG)

        d_center = raw_pos_x / self.MAX_CENTER_DIST
        heading_error = raw_facing
        d_wall = 1.0 - off_track

        speed = np.clip(raw_speed / self.MAX_SPEED, 0.0, 1.0)
        d_center = np.clip(d_center, -1.0, 1.0)
        heading_error = np.clip(heading_error, -np.pi, np.pi)
        d_wall = np.clip(d_wall, 0.0, 1.0)
        progress = np.clip(raw_progress, 0.0, 1.0)
        item_state = np.clip(raw_item / 15.0, 0.0, 1.0)

        if collision:
            self._total_collisions += 1
        if off_track:
            self._total_offtrack_steps += 1

        return np.array(
            [speed, d_center, heading_error, d_wall, progress, item_state],
            dtype=np.float32,
        )

    # -------------------------------------------------------------------
    # Reward
    # -------------------------------------------------------------------

    def _compute_reward(self, obs: np.ndarray):
        speed    = obs[0]
        progress = obs[4]

        delta_progress = progress - self._prev_progress
        if delta_progress < -0.5:
            delta_progress += 1.0
        self._prev_progress = progress

        collision = 1 if self.memory.read_uint8(MKWiiAddresses.COLLISION_FLAG) else 0
        off_track = 1 if self.memory.read_uint8(MKWiiAddresses.OFF_TRACK_FLAG) else 0

        current_lap = self.memory.read_uint8(MKWiiAddresses.LAP_COUNT)
        lap_completed = current_lap > self._lap_count
        if lap_completed:
            self._lap_count = current_lap

        r_progress  = REWARD_PROGRESS_COEFF * delta_progress * 100
        r_speed     = REWARD_SPEED_COEFF * speed
        r_collision = REWARD_COLLISION_PENALTY * collision
        r_offtrack  = REWARD_OFFTRACK_PENALTY * off_track
        r_lap       = REWARD_LAP_BONUS if lap_completed else 0.0

        reward = r_progress + r_speed + r_collision + r_offtrack + r_lap

        info = {
            "reward_progress": r_progress,
            "reward_speed": r_speed,
            "reward_collision": r_collision,
            "reward_offtrack": r_offtrack,
            "reward_lap_bonus": r_lap,
            "collision": collision,
            "off_track": off_track,
            "lap_completed": lap_completed,
        }
        self._prev_speed = speed
        return reward, info

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    def _check_terminated(self) -> bool:
        TARGET_LAPS = 1
        return self._lap_count >= TARGET_LAPS

    def _advance_frames(self, n: int):
        frame_time = n / 60.0
        time.sleep(frame_time)

    def _load_savestate(self):
        """Load savestate via platform-specific hotkey."""
        if PLATFORM == "Darwin":
            try:
                applescript = '''
                    tell application "Dolphin"
                        activate
                    end tell
                    tell application "System Events"
                        key code 122
                    end tell
                '''
                subprocess.run(["osascript", "-e", applescript], capture_output=True, timeout=3)
                logging.debug("Savestate loaded via osascript F1.")
                return
            except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                logging.debug(f"osascript failed: {e}")
        else:
            try:
                result = subprocess.run(
                    ["xdotool", "search", "--name", "Dolphin"],
                    capture_output=True, text=True, timeout=2,
                )
                wids = result.stdout.strip().split("\n")
                if wids and wids[0]:
                    subprocess.run(["xdotool", "key", "--window", wids[0], "F1"], timeout=2)
                    return
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass

        logging.warning("Could not load savestate automatically.")

    def _get_info(self) -> dict:
        elapsed = time.time() - self._episode_start_time if self._episode_start_time else 0
        return {
            "step_count": self._step_count,
            "lap_count": self._lap_count,
            "total_collisions": self._total_collisions,
            "total_offtrack_steps": self._total_offtrack_steps,
            "elapsed_time_s": elapsed,
        }

    def get_env_summary(self) -> dict:
        return {
            "observation_space": {
                "shape": self.observation_space.shape,
                "dtype": str(self.observation_space.dtype),
                "low": self.observation_space.low.tolist(),
                "high": self.observation_space.high.tolist(),
                "features": [
                    "speed (normalized, 0–1)",
                    "d_center (lateral offset, -1–1)",
                    "heading_error (radians, -π–π)",
                    "d_wall (wall distance, 0–1)",
                    "progress (lap fraction, 0–1)",
                    "item_state (encoded item, 0–1)",
                ],
            },
            "action_space": {
                "type": "Discrete",
                "n": self.action_space.n,
                "actions": {i: ACTION_MAP[i]["name"] for i in range(NUM_ACTIONS)},
            },
            "reward_structure": "Dense (progress + speed − collision − offtrack + lap bonus)",
            "frames_per_step": self.frames_per_step,
            "max_steps_per_episode": self.max_steps,
            "dolphin": {
                "running": self.dolphin.is_running,
                "exe_path": self.dolphin.exe_path,
                "iso_path": self.dolphin.iso_path,
                "gfx_backend": self.dolphin.gfx_backend,
                "gdb_port": self.gdb_port,
                "pid": self.dolphin._process.pid if self.dolphin._process else None,
            },
        }


# ===========================================================================
# Self-test: python environment.py
# ===========================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    iso = os.environ.get("DOLPHIN_ISO_PATH", "")
    if not iso:
        print("ERROR: DOLPHIN_ISO_PATH is not set.")
        print('  export DOLPHIN_ISO_PATH="/path/to/Mario Kart Wii.rvz"')
        print("  Or add it to your .env file.")
        sys.exit(1)

    print("=" * 60)
    print("MarioKartEnv — Live Self-Test (GDB Memory Interface)")
    print("=" * 60)
    print(f"  ISO:      {iso}")
    print(f"  Platform: {PLATFORM}")
    print(f"  Dolphin:  {_default_dolphin_exe()}")
    print(f"  GDB port: {GDB_PORT}")
    print()

    print("Launching Dolphin...")
    env = MarioKartEnv(
        iso_path=iso,
        gfx_backend=os.environ.get("DOLPHIN_GFX_BACKEND", "Metal"),
    )

    summary = env.get_env_summary()
    print(f"\nObservation Space: {env.observation_space}")
    print(f"  Shape: {summary['observation_space']['shape']}")
    print(f"\nAction Space: Discrete({env.action_space.n})")
    for idx, name in summary["action_space"]["actions"].items():
        print(f"  {idx}: {name}")

    print(f"\nDolphin PID: {summary['dolphin']['pid']}")
    print(f"GDB connected: {env.memory.is_connected()}")

    print("\nResetting environment...")
    obs, info = env.reset()
    print(f"  Observation: {obs}")

    print("\nRunning 5 test steps (ACCELERATE)...")
    for i in range(5):
        obs, reward, terminated, truncated, info = env.step(0)
        print(f"  Step {i+1}: reward={reward:+.4f}, speed={obs[0]:.3f}, progress={obs[4]:.4f}")
        if terminated or truncated:
            break

    print("\nClosing...")
    env.close()
    print("Done.")