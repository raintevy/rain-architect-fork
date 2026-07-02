"""Abstract base class for per-robot configuration."""

from __future__ import annotations

from pathlib import Path


class RobotConfig:
    """Base class for robot-specific configuration.

    Subclasses provide the robot's API spec, skill files directory, and
    methods to build the execution namespace (dry-run or live).
    """

    #: Short identifier used for the --robot CLI flag.
    name: str = ""

    #: Human-readable label injected into LLM system prompts so the model
    #: knows which robot it's writing programs for. Distinct from ``name``
    #: (which is a CLI key); e.g. ``"Franka Panda + Robotiq 2F-85"``.
    display_name: str = ""

    @property
    def api_spec(self) -> str:
        """Full API spec string injected into the LLM system prompt."""
        raise NotImplementedError

    @property
    def skills_dir(self) -> Path:
        """Directory containing robot-specific skill markdown files."""
        raise NotImplementedError

    def make_dry_run_namespace(self, console) -> dict:
        """Return a namespace of stub callables for --dry-run mode."""
        raise NotImplementedError

    def make_live_namespace(self) -> tuple[dict, object]:
        """Initialise the robot client and return (namespace dict, client).

        The caller is responsible for calling shutdown(client) on exit.
        """
        raise NotImplementedError

    def shutdown(self, client) -> None:
        """Tear down the robot client and its ROS runtime."""
        raise NotImplementedError
