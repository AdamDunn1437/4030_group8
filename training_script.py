import argparse
import json
import os
import socket
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    from dotenv import load_dotenv

    load_dotenv(override=True)
except Exception:
    pass

from environment import MarioKartEnv


OBS_FEATURES = ["speed", "d_center", "heading_error", "d_wall", "progress", "item_state"]


def _to_builtin(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    return value


def _port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect((host, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def _parse_ini(path: Path) -> dict:
    data = {}
    if not path.is_file():
        return data
    section = None
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith(";"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
            data.setdefault(section, {})
            continue
        if "=" in line and section:
            k, v = line.split("=", 1)
            data[section][k.strip()] = v.strip()
    return data


def _compute_diagnostics(reset_obs, step_results):
    if not step_results:
        return {
            "steps_run": 0,
            "total_reward": 0.0,
            "avg_reward": 0.0,
            "terminated": False,
            "truncated": False,
            "observation_changed": False,
            "feature_change_counts": {name: 0 for name in OBS_FEATURES},
        }

    obs_matrix = np.array([s["observation"] for s in step_results], dtype=np.float32)
    rewards = np.array([s["reward"] for s in step_results], dtype=np.float32)
    reset_vec = np.array(reset_obs, dtype=np.float32)

    feature_change_counts = {}
    for i, name in enumerate(OBS_FEATURES):
        feature_change_counts[name] = int(np.sum(np.abs(obs_matrix[:, i] - reset_vec[i]) > 1e-6))

    observation_changed = bool(np.any(np.abs(obs_matrix - reset_vec) > 1e-6))

    return {
        "steps_run": int(len(step_results)),
        "total_reward": float(np.sum(rewards)),
        "avg_reward": float(np.mean(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "terminated": bool(step_results[-1]["terminated"]),
        "truncated": bool(step_results[-1]["truncated"]),
        "observation_changed": observation_changed,
        "feature_change_counts": feature_change_counts,
        "final_observation": step_results[-1]["observation"],
    }


def _results_dir_stats(results_dir: Path):
    if not results_dir.exists():
        return {"exists": False, "json_files": 0}

    json_files = sorted(results_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    latest = json_files[0] if json_files else None
    return {
        "exists": True,
        "json_files": len(json_files),
        "latest_json": str(latest) if latest else None,
    }


def _write_summary(records: dict, output_file: Path, summary_file: Path):
    diagnostics = records.get("diagnostics", {})
    preflight = records.get("preflight", {})

    lines = []
    lines.append("MarioKart Environment Verification Summary")
    lines.append("=" * 50)
    lines.append(f"Timestamp: {records.get('timestamp')}")
    lines.append(f"Status: {str(records.get('status', '')).upper()}")
    lines.append(f"ISO: {records.get('iso_path')}")
    lines.append(f"Backend: {records.get('gfx_backend')}")
    lines.append(f"Memory backend: {preflight.get('memory_backend')}")
    lines.append(f"Duration (s): {records.get('duration_seconds')}")
    lines.append(f"JSON report: {output_file}")
    lines.append("")

    if records.get("status") == "success":
        lines.append("Run Diagnostics")
        lines.append("-" * 50)
        lines.append(f"Steps run: {diagnostics.get('steps_run')}")
        lines.append(f"Total reward: {diagnostics.get('total_reward'):.6f}")
        lines.append(f"Average reward: {diagnostics.get('avg_reward'):.6f}")
        lines.append(f"Reward range: [{diagnostics.get('min_reward'):.6f}, {diagnostics.get('max_reward'):.6f}]")
        lines.append(f"Observation changed from reset: {diagnostics.get('observation_changed')}")
        lines.append(f"Terminated: {diagnostics.get('terminated')}  Truncated: {diagnostics.get('truncated')}")
        lines.append("Feature change counts from reset:")
        for k, v in diagnostics.get("feature_change_counts", {}).items():
            lines.append(f"  - {k}: {v}")
    else:
        lines.append("Failure")
        lines.append("-" * 50)
        lines.append(f"Error: {records.get('error', 'unknown error')}")

    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_preflight(iso_path: str, gfx_backend: str, gdb_port: int = 2345):
    exe_path = os.environ.get("DOLPHIN_EXE_PATH", "")
    user_dir = os.environ.get("DOLPHIN_USER_DIR", "")
    memory_backend = os.environ.get("DOLPHIN_MEMORY_BACKEND", "dme").strip().lower()
    config_dir = Path(user_dir) / "Config" if user_dir else None
    debugger_ini = config_dir / "Debugger.ini" if config_dir else None
    dolphin_ini = config_dir / "Dolphin.ini" if config_dir else None

    dolphin_parsed = _parse_ini(dolphin_ini) if dolphin_ini else {}
    ra_section = dolphin_parsed.get("RetroAchievements", {})
    ra_enabled = ra_section.get("Enabled", "False")
    ra_hardcore = ra_section.get("HardcoreMode", "False")

    report = {
        "iso_path": iso_path,
        "iso_exists": Path(iso_path).is_file() if iso_path else False,
        "dolphin_exe_path": exe_path,
        "dolphin_exe_exists": Path(exe_path).is_file() if exe_path else False,
        "dolphin_user_dir": user_dir,
        "dolphin_user_dir_exists": Path(user_dir).is_dir() if user_dir else False,
        "gfx_backend": gfx_backend,
        "memory_backend": memory_backend,
        "gdb_port": gdb_port,
        "gdb_port_open_now": _port_open("127.0.0.1", gdb_port),
        "debugger_ini_path": str(debugger_ini) if debugger_ini else "",
        "debugger_ini_exists": debugger_ini.is_file() if debugger_ini else False,
        "dolphin_ini_path": str(dolphin_ini) if dolphin_ini else "",
        "dolphin_ini_exists": dolphin_ini.is_file() if dolphin_ini else False,
        "ra_enabled": ra_enabled,
        "ra_hardcore_mode": ra_hardcore,
        "likely_blockers": [],
    }

    if debugger_ini and debugger_ini.is_file():
        report["debugger_ini_content"] = debugger_ini.read_text(encoding="utf-8", errors="replace")

    if dolphin_ini and dolphin_ini.is_file():
        report["dolphin_ini_excerpt"] = {
            "Core": dolphin_parsed.get("Core", {}),
            "Interface": dolphin_parsed.get("Interface", {}),
            "RetroAchievements": ra_section,
        }

    if not report["iso_exists"]:
        report["likely_blockers"].append("ISO path does not exist")
    if not report["dolphin_exe_exists"]:
        report["likely_blockers"].append("Dolphin executable path does not exist")
    if not report["dolphin_user_dir_exists"]:
        report["likely_blockers"].append("Dolphin user dir path does not exist")
    if memory_backend == "gdb" and not report["debugger_ini_exists"]:
        report["likely_blockers"].append("Debugger.ini not found in configured user dir")
    if str(ra_enabled).lower() in {"true", "1", "yes"} and str(ra_hardcore).lower() in {"true", "1", "yes"}:
        report["likely_blockers"].append("RetroAchievements Hardcore mode enabled (can disable GDB stub)")
    if memory_backend == "gdb" and not report["gdb_port_open_now"]:
        report["likely_blockers"].append("GDB backend selected and port 2345 is closed")

    print("-" * 68)
    print("Preflight")
    print("-" * 68)
    print(f"ISO exists:        {report['iso_exists']} -> {iso_path}")
    print(f"EXE exists:        {report['dolphin_exe_exists']} -> {exe_path}")
    print(f"User dir exists:   {report['dolphin_user_dir_exists']} -> {user_dir}")
    print(f"Memory backend:    {memory_backend}")
    print(f"Debugger.ini:      {report['debugger_ini_exists']} -> {report['debugger_ini_path']}")
    print(f"Dolphin.ini:       {report['dolphin_ini_exists']} -> {report['dolphin_ini_path']}")
    print(f"RA enabled:        {ra_enabled}")
    print(f"RA hardcore:       {ra_hardcore}")
    print(f"Port {gdb_port} open now: {report['gdb_port_open_now']}")
    if report["likely_blockers"]:
        print("Likely blockers:")
        for item in report["likely_blockers"]:
            print(f"  - {item}")

    return report


def run_verification(iso_path: str, steps: int, gfx_backend: str, output_file: Path, summary_file: Path) -> int:
    records = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "iso_path": iso_path,
        "steps_requested": steps,
        "gfx_backend": gfx_backend,
        "status": "started",
        "preflight": run_preflight(iso_path, gfx_backend),
    }

    env = None
    start = time.time()

    try:
        env = MarioKartEnv(iso_path=iso_path, gfx_backend=gfx_backend)
        summary = env.get_env_summary()
        records["environment_summary"] = _to_builtin(summary)

        obs, info = env.reset()
        reset_obs = _to_builtin(obs)
        records["reset"] = {
            "observation": reset_obs,
            "info": _to_builtin(info),
        }

        step_results = []
        for i in range(steps):
            obs, reward, terminated, truncated, info = env.step(0)
            step_results.append(
                {
                    "step": i + 1,
                    "action": 0,
                    "observation": _to_builtin(obs),
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "info": _to_builtin(info),
                }
            )
            if terminated or truncated:
                break

        records["steps"] = step_results
        records["diagnostics"] = _compute_diagnostics(reset_obs, step_results)
        records["results_dir"] = _results_dir_stats(output_file.parent)
        records["status"] = "success"
        records["duration_seconds"] = round(time.time() - start, 3)
        return_code = 0

    except Exception as exc:
        records["status"] = "failed"
        records["duration_seconds"] = round(time.time() - start, 3)
        records["error"] = str(exc)
        records["traceback"] = traceback.format_exc()
        records["results_dir"] = _results_dir_stats(output_file.parent)
        return_code = 1

    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass

        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(records, indent=2), encoding="utf-8")
        _write_summary(records, output_file, summary_file)

        print("=" * 68)
        print("MarioKart Environment Verification")
        print("=" * 68)
        print(f"ISO:         {iso_path}")
        print(f"Backend:     {gfx_backend}")
        print(f"Steps asked: {steps}")
        print(f"Result:      {records['status'].upper()}")
        print(f"Report:      {output_file}")
        print(f"Summary:     {summary_file}")
        if records["status"] == "success":
            d = records.get("diagnostics", {})
            print("Reset:       OK")
            print(f"Steps run:   {d.get('steps_run', 0)}")
            print(f"Obs changed: {d.get('observation_changed', False)}")
            print(f"Total reward:{d.get('total_reward', 0.0): .6f}")
        else:
            print(f"Error:       {records.get('error', 'unknown error')}")

    return return_code


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify MarioKartEnv startup, reset(), and step() and save output."
    )
    parser.add_argument(
        "--iso",
        default=os.environ.get("DOLPHIN_ISO_PATH", ""),
        help="Path to Mario Kart ISO/WBFS/RVZ. Defaults to DOLPHIN_ISO_PATH.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=5,
        help="How many step(action=0) calls to run after reset.",
    )
    parser.add_argument(
        "--gfx",
        default=os.environ.get("DOLPHIN_GFX_BACKEND", "Null"),
        help="Dolphin video backend (e.g., Null, Vulkan, D3D12).",
    )
    parser.add_argument(
        "--output",
        default="dqn_results/environment_verification.json",
        help="Path to JSON report output.",
    )
    parser.add_argument(
        "--summary-output",
        default="dqn_results/environment_verification_summary.txt",
        help="Path to human-readable summary output.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Run config/path/port checks without launching environment.",
    )

    args = parser.parse_args()

    if not args.iso:
        print("ERROR: No ISO path provided.")
        print("Set DOLPHIN_ISO_PATH in .env or pass --iso \"C:/path/to/Mario Kart.wbfs\"")
        return 2

    if args.preflight_only:
        run_preflight(args.iso, args.gfx)
        return 0

    output_path = Path(args.output)
    summary_path = Path(args.summary_output)
    return run_verification(args.iso, max(1, args.steps), args.gfx, output_path, summary_path)


if __name__ == "__main__":
    sys.exit(main())
