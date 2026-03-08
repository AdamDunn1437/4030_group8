"""Project environment entrypoint (auto-select by OS)."""

from environments import ACTION_MAP, MKWiiAddresses, MarioKartEnv, NUM_ACTIONS

__all__ = ["MarioKartEnv", "ACTION_MAP", "NUM_ACTIONS", "MKWiiAddresses"]
