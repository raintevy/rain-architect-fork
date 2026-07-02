"""Live ROS1 service client for the Franka Panda + Robotiq 2F-85 robot API.

Each public method corresponds to one entry in FrankaConfig.api_spec and is
backed by a ROS1 service call using the custom srv types from
``robot_api_interfaces``:

    RobotQuery   — no request fields; response: {result_code, data: JSON string}
    RobotCommand — request: {req: JSON string}; response: {result_code, data}

Service name pattern: /robot/{category}/{function_name}

ROS environment: ROS1 Noetic (rospy). The ARCHITECT process must source the catkin
workspace before importing this module, e.g.:
    source <catkin_ws>/devel/setup.bash
"""

from __future__ import annotations

import json

try:
    import rospy
    from robot_api_interfaces.srv import (  # type: ignore[import]
        RobotCommand,
        RobotCommandRequest,
        RobotQuery,
        RobotQueryRequest,
    )
    _ROS_AVAILABLE = True
except ImportError:
    _ROS_AVAILABLE = False


class FrankaRobotClient:
    """ROS1 client that exposes the Franka robot API as Python method calls.

    Service proxies are created lazily and cached so we don't reconnect on
    every call. All methods are synchronous (rospy service calls block until
    the service responds).
    """

    def __init__(self) -> None:
        if not _ROS_AVAILABLE:
            raise RuntimeError(
                "rospy / robot_api_interfaces not available. "
                "Source the catkin workspace before running in live mode."
            )
        rospy.init_node("architect_franka_client", anonymous=True, disable_signals=True)
        self._query_proxies: dict[str, rospy.ServiceProxy] = {}
        self._command_proxies: dict[str, rospy.ServiceProxy] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _query(self, service_name: str) -> dict:
        """Call a RobotQuery service (no input) and return parsed JSON data."""
        if service_name not in self._query_proxies:
            rospy.wait_for_service(service_name)
            self._query_proxies[service_name] = rospy.ServiceProxy(
                service_name, RobotQuery
            )
        resp = self._query_proxies[service_name](RobotQueryRequest())
        if resp.result_code.result_code != 0:
            raise RuntimeError(resp.result_code.message)
        return json.loads(resp.data)

    def _command(self, service_name: str, payload: dict) -> dict:
        """Call a RobotCommand service with a JSON payload and return parsed data."""
        if service_name not in self._command_proxies:
            rospy.wait_for_service(service_name)
            self._command_proxies[service_name] = rospy.ServiceProxy(
                service_name, RobotCommand
            )
        req = RobotCommandRequest(req=json.dumps(payload))
        resp = self._command_proxies[service_name](req)
        if resp.result_code.result_code != 0:
            raise RuntimeError(resp.result_code.message)
        return json.loads(resp.data)

    def destroy_node(self) -> None:
        """Shut down the ROS1 node."""
        rospy.signal_shutdown("ARCHITECT session ended")

    # ------------------------------------------------------------------
    # Perception
    # ------------------------------------------------------------------

    def detect_markers(self) -> dict:
        return self._query("/robot/perception/detect_markers")

    def detect_objects(
        self,
        text_prompt: str,
        x_offset: float = 0.0,
        y_offset: float = 0.0,
        z_offset: float = 0.0,
        camera: str = "scene",
    ) -> dict:
        """Detect an object via Grounded-SAM2 + AnyGrasp, optionally offset.

        ``x_offset`` / ``y_offset`` / ``z_offset`` shift the returned grasp pose
        in end-effector local frame meters. Useful when the detected grasp lands a few
        centimeters off from where the actual pickup needs to happen — pass
        an offset instead of post-processing the dict in the program.
        """
        return self._command(
            "/robot/perception/detect_objects",
            {
                "text_prompt": text_prompt,
                "x_offset": x_offset,
                "y_offset": y_offset,
                "z_offset": z_offset,
                "camera": camera,
            },
        )

    def get_vqa_response(self, prompt: str, camera: str) -> dict:
        return self._command(
            "/robot/perception/get_vqa_response", {"prompt": prompt, "camera": camera}
        )
    
    def verify_grasp(self, object_name: str) -> dict:
        """Verify if the current grasp is stable for the named object (e.g. "red apple", "baseball", etc.)."""
        return self._command(
            "/robot/perception/verify_grasp", {"object_name": object_name}
        )

    def get_placement_pose(self, base_text_prompt: str, target_text_prompt: str, x_offset: float = 0.0, y_offset: float = 0.0, z_offset: float = 0.0) -> dict:
        """Return a recommended placement pose for ``target_text_prompt`` on top
        of / next to ``base_text_prompt``. Combines scene segmentation with
        spatial reasoning to pick a free surface or stacking position."""
        return self._command(
            "/robot/perception/get_placement_pose",
            {
                "base_text_prompt": base_text_prompt,
                "target_text_prompt": target_text_prompt,
                "x_offset": x_offset,
                "y_offset": y_offset,
                "z_offset": z_offset,
            },
        )

    def get_keypoints(self, text_prompt: str, task: str) -> dict:
        """Return task-conditioned keypoints on ``text_prompt``-named object.

        Backed by ReKep-style perception: the keypoints are semantic anchors
        the manipulation task should care about (e.g. a mug's handle endpoints,
        a drawer's pull-edge, a cup's rim). ``task`` is a natural-language
        description so the same object can yield different keypoints for
        different tasks (grasp-the-handle vs pour-from-spout)."""
        return self._command(
            "/robot/perception/get_keypoints",
            {"text_prompt": text_prompt, "task": task},
        )

    def get_keypoints_trajectory(self, text_prompt: str, task: str) -> dict:
        """Return a task-conditioned waypoint trajectory through keypoints.

        Same backend as :meth:`get_keypoints` but the response is an ordered
        sequence of end-effector waypoints — useful for articulated motions
        (open a drawer, pour from a mug) where the path through space matters,
        not just the endpoint."""
        return self._command(
            "/robot/perception/get_keypoints_trajectory",
            {"text_prompt": text_prompt, "task": task},
        )

    # ------------------------------------------------------------------
    # Proprioception
    # ------------------------------------------------------------------

    def get_current_joints(self) -> dict:
        return self._query("/robot/proprioception/get_current_joints")["joints"]

    def get_current_gripper_width(self) -> dict:
        return self._query("/robot/proprioception/get_gripper_width")

    def get_current_ee_pose(self) -> dict:
        return self._query("/robot/proprioception/get_current_ee_pose")["ee_pose"]

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def set_gripper_width(self, width: float) -> bool:
        return self._command(
            "/robot/control/set_gripper_width", {"width": width}
        )["success"]

    def move_ee_to_pose(self, target_pose: dict) -> bool:
        return self._command(
            "/robot/control/move_ee_to_pose", {"target_pose": target_pose}
        )["success"]

    def move_ee_to_rel_pose(self, delta_position: dict) -> bool:
        return self._command(
            "/robot/control/move_ee_to_rel_pose", {"delta_position": delta_position}
        )["success"]

    def move_ee_guarded(self, axis: str, distance: float, force_threshold: float) -> dict:
        """Move the EE along ``axis`` for up to ``distance`` meters, stopping
        early if the measured contact force exceeds ``force_threshold`` (N).

        Returns a dict with the actual distance traveled and whether contact
        was detected — useful for "press until contact" patterns where you
        don't know the exact contact depth in advance."""
        result = self._command(
            "/robot/control/move_ee_guarded",
            {"axis": axis, "distance": distance, "force_threshold": force_threshold},
        )
        return result.get("data", {})

    def rotate_wrist(self, angle_degrees: float, direction: str) -> bool:
        """Rotate the wrist by ``angle_degrees`` in the named ``direction``
        (``"cw"`` / ``"ccw"``). Used for screw / unscrew motions and other
        in-place reorientations that move_ee_to_pose can't express cleanly."""
        return self._command(
            "/robot/control/rotate_wrist",
            {"angle_degrees": angle_degrees, "direction": direction},
        )["success"]

    def execute_waypoint_trajectory(self, waypoints: list[dict]) -> dict:
        """Execute a smooth trajectory through the given ``waypoints``. Used
        with the trajectory output of :meth:`get_keypoints_trajectory` to
        execute articulated motions (open a door, pour, etc.). Each waypoint
        is a full EE pose dict in base-link frame."""
        result = self._command(
            "/robot/control/execute_waypoint_trajectory",
            {"waypoints": waypoints},
        )
        return result.get("data", {})

    def reset_robot(self) -> bool:
        return self._query("/robot/control/reset_robot")["success"]
