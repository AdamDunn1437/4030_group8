"""OS-specific environment router package."""

import platform

if platform.system() == "Windows":
    from environments.windows import ACTION_MAP, MKWiiAddresses, MarioKartEnv, NUM_ACTIONS
else:
    from environments.mac import ACTION_MAP, MKWiiAddresses, MarioKartEnv, NUM_ACTIONS

__all__ = ["MarioKartEnv", "ACTION_MAP", "NUM_ACTIONS", "MKWiiAddresses"]
