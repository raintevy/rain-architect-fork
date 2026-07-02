"""Per-robot configuration classes.

Usage:
    from architect.robots import get_robot_config
    config = get_robot_config("franka")
"""

from architect.robots.franka import FrankaConfig

_REGISTRY = {
    "franka": FrankaConfig,
}


def get_robot_config(name: str):
    """Return an instantiated RobotConfig for the given robot name."""
    cls = _REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown robot '{name}'. Available: {list(_REGISTRY)}")
    return cls()
