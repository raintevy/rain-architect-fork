"""Franka Panda + Robotiq 2F-85 robot configuration (ROS1 Noetic)."""

from __future__ import annotations

from pathlib import Path

from architect.robots.franka_api import API_SPEC
from architect.robots.base import RobotConfig

_SKILLS_DIR = Path(__file__).parent.parent.parent / "skills" / "franka"


class FrankaConfig(RobotConfig):
    name = "franka"
    display_name = "Franka Panda + Robotiq 2F-85 (ROS1 Noetic)"

    @property
    def api_spec(self) -> str:
        return API_SPEC

    @property
    def skills_dir(self) -> Path:
        return _SKILLS_DIR

    def make_dry_run_namespace(self, console) -> dict:
        def _stub(name, *args, **kwargs):
            arg_strs = [repr(a) for a in args] + [f"{k}={v!r}" for k, v in kwargs.items()]
            console.print(f"  [dim]\\[dry-run][/dim] [cyan]{name}[/cyan]({', '.join(arg_strs)})")

        return {
            "detect_markers": lambda: (
                _stub("detect_markers") or {"markers": []}
            ),
            "detect_objects": lambda text_prompt, x_offset=0.0, y_offset=0.0, z_offset=0.0, camera="scene": (
                _stub("detect_objects", text_prompt,
                      x_offset=x_offset, y_offset=y_offset, z_offset=z_offset, camera=camera) or {
                    "best_grasp": {
                        "translation_wrt_base": [
                            0.45 + x_offset, 0.0 + y_offset, 0.3 + z_offset,
                        ],
                        "quaternion_wrt_base": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                        "score": 0.9,
                        "width": 0.05,
                    }
                }
            ),
            "get_vqa_response": lambda prompt, camera: (
                _stub("get_vqa_response", prompt, camera) or {"data": {"answer": "yes"}}
            ),
            "verify_grasp": lambda object_name: (
                _stub("verify_grasp", object_name) or {"data": {"grasped": True, "reason": "stable grasp"}}
            ),
            "get_placement_pose": lambda base_text_prompt, target_text_prompt, x_offset=0.0, y_offset=0.0, z_offset=0.0: (
                _stub("get_placement_pose", base_text_prompt, target_text_prompt, x_offset, y_offset, z_offset) or {
                    "placement_pose_base": {
                        "position": {"x": 0.45 + x_offset, "y": 0.0 + y_offset, "z": 0.345 + z_offset},
                        "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                    }
                }
            ),
            "get_keypoints": lambda text_prompt, task: (
                _stub("get_keypoints", text_prompt, task) or {
                    "keypoints": [
                        {"position": {"x": 0.45, "y": 0.0, "z": 0.30}, "label": "center"},
                    ]
                }
            ),
            "get_keypoints_trajectory": lambda text_prompt, task: (
                _stub("get_keypoints_trajectory", text_prompt, task) or {
                    "waypoints": [
                        {
                            "position": {"x": 0.45, "y": 0.0, "z": 0.30},
                            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                        }
                    ]
                }
            ),
            "get_current_joints": lambda: (
                _stub("get_current_joints") or {
                    f"panda_joint{i}": {"position": 0.0, "velocity": 0.0, "effort": 0.0}
                    for i in range(1, 8)
                }
            ),
            "get_current_gripper_width": lambda: (
                _stub("get_current_gripper_width") or {"gripper_width": 0.085}
            ),
            "get_current_ee_pose": lambda: (
                _stub("get_current_ee_pose") or {
                    "position": {"x": 0.4, "y": 0.0, "z": 0.4},
                    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                }
            ),
            "set_gripper_width": lambda width: (
                _stub("set_gripper_width", width) or True
            ),
            "move_ee_to_pose": lambda target_pose: (
                _stub("move_ee_to_pose", target_pose) or True
            ),
            "move_ee_to_rel_pose": lambda delta_position: (
                _stub("move_ee_to_rel_pose", delta_position) or True
            ),
            "move_ee_guarded": lambda axis, distance, force_threshold: (
                _stub("move_ee_guarded", axis, distance, force_threshold) or {
                    "actual_distance": distance,
                    "contact_detected": False,
                }
            ),
            "rotate_wrist": lambda angle_degrees, direction: (
                _stub("rotate_wrist", angle_degrees, direction) or True
            ),
            "execute_waypoint_trajectory": lambda waypoints: (
                _stub("execute_waypoint_trajectory", waypoints) or {
                    "completed": True,
                    "n_waypoints_reached": len(waypoints),
                }
            ),
            "reset_robot": lambda: (
                _stub("reset_robot") or True
            ),
            "insert": lambda obj, socket: (
                _stub("insert", obj, socket) or {
                    "success": True,
                    "reason": "success",
                    "depth": 0.025,
                    "steps": 87,
                }
            ),
        }

    def make_live_namespace(self) -> tuple[dict, object]:
        from architect.robots.franka_client import FrankaRobotClient

        client = FrankaRobotClient()
        namespace = {
            "detect_markers": client.detect_markers,
            "detect_objects": client.detect_objects,
            "get_vqa_response": client.get_vqa_response,
            "verify_grasp": client.verify_grasp,
            "get_placement_pose": client.get_placement_pose,
            "get_keypoints": client.get_keypoints,
            "get_keypoints_trajectory": client.get_keypoints_trajectory,
            "get_current_joints": client.get_current_joints,
            "get_current_gripper_width": client.get_current_gripper_width,
            "get_current_ee_pose": client.get_current_ee_pose,
            "set_gripper_width": client.set_gripper_width,
            "move_ee_to_pose": client.move_ee_to_pose,
            "move_ee_to_rel_pose": client.move_ee_to_rel_pose,
            "move_ee_guarded": client.move_ee_guarded,
            "rotate_wrist": client.rotate_wrist,
            "execute_waypoint_trajectory": client.execute_waypoint_trajectory,
            "reset_robot": client.reset_robot,
        }
        return namespace, client

    def shutdown(self, client) -> None:
        client.destroy_node()
