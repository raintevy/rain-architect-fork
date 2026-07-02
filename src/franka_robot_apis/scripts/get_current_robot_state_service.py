#!/usr/bin/env python3
"""Expose the Franka robot's current end-effector pose and joint states as ROS services.

Subscribes to /franka_state_controller/franka_states and serves the latest
end-effector pose and joint states as JSON-encoded service responses.

Services:
    /robot/proprioception/get_current_ee_pose (robot_api_interfaces/RobotQuery)
    /robot/proprioception/get_current_joints (robot_api_interfaces/RobotQuery)

Request (JSON in `req`):
    empty

Response (JSON in `data`):
    get_current_ee_pose:
    {
        "ee_pose": {
            "position": {"x": 0.0, "y": 0.0, "z": 0.0},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
        }
    }

    get_current_joints:
    {
        "joints": {
            "panda_joint1": {"position": 0.0, "velocity": 0.0, "effort": 0.0},
            ...
        }
    }
"""

import json
import math

import rospy
from franka_msgs.msg import FrankaState

from robot_api_interfaces.srv import RobotQuery, RobotQueryResponse
from robot_api_interfaces.msg import ResultCode


class GetCurrentRobotStateNode:
    """ROS1 service node providing the current end-effector pose and joint states.

    Subscribes to /franka_state_controller/franka_states and extracts the
    O_T_EE (4x4 homogeneous transform) to compute position and orientation.
    """

    def __init__(self):
        """Initialize the node, parameters, subscriber, and services."""
        rospy.init_node("get_current_robot_state_service")

        self.tf_timeout = rospy.get_param("~tf_timeout", 1.0)
        self.state_timeout = rospy.get_param("~state_timeout", 1.0)

        # Store latest EE transform (16-element array: 4x4 homogeneous matrix)
        # Format: [r11, r12, r13, tx, r21, r22, r23, ty, r31, r32, r33, tz, 0, 0, 0, 1]
        self.latest_o_tee = None
        self.latest_q = None
        self.latest_dq = None
        self.latest_tau_J = None
        self.latest_stamp = None
        self.has_received_data = False

        self.sub = rospy.Subscriber(
            "/franka_state_controller/franka_states",
            FrankaState,
            self._franka_state_callback,
            queue_size=1,
        )

        # Both services are served from the same FrankaState topic.
        self.ee_pose_service = rospy.Service(
            "/robot/proprioception/get_current_ee_pose",
            RobotQuery,
            self._handle_get_current_ee_pose,
        )

        self.joints_service = rospy.Service(
            "/robot/proprioception/get_current_joints",
            RobotQuery,
            self._handle_get_current_joints,
        )

        rospy.loginfo(
            "Services initialized. "
            "Subscribing to: /franka_state_controller/franka_states"
        )

    def _franka_state_callback(self, msg):
        """Cache the latest FrankaState transform and joint data.

        Args:
            msg: FrankaState message containing O_T_EE transform and joint data.
        """
        self.latest_o_tee = msg.O_T_EE
        self.latest_q = msg.q
        self.latest_dq = msg.dq
        self.latest_tau_J = msg.tau_J
        self.latest_stamp = msg.header.stamp
        self.has_received_data = True
        rospy.logdebug_throttle(1.0, "Received FrankaState update")

    def _rotation_matrix_to_quaternion(self, r):
        """Convert a 3x3 rotation matrix to a normalized quaternion.

        Uses the largest-diagonal branch for numerical stability.

        Args:
            r: 9-element rotation matrix [r11, r12, r13, r21, r22, r23, r31, r32, r33].

        Returns:
            tuple: (x, y, z, w) quaternion components.
        """
        r11, r12, r13 = r[0], r[1], r[2]
        r21, r22, r23 = r[3], r[4], r[5]
        r31, r32, r33 = r[6], r[7], r[8]

        trace = r11 + r22 + r33

        if trace > 0:
            s = math.sqrt(trace + 1.0) * 2.0
            w = 0.25 * s
            x = (r32 - r23) / s
            y = (r13 - r31) / s
            z = (r21 - r12) / s

        elif (r11 > r22) and (r11 > r33):
            s = math.sqrt(1.0 + r11 - r22 - r33) * 2.0
            w = (r32 - r23) / s
            x = 0.25 * s
            y = (r12 + r21) / s
            z = (r13 + r31) / s

        elif r22 > r33:
            s = math.sqrt(1.0 + r22 - r11 - r33) * 2.0
            w = (r13 - r31) / s
            x = (r12 + r21) / s
            y = 0.25 * s
            z = (r23 + r32) / s

        else:
            s = math.sqrt(1.0 + r33 - r11 - r22) * 2.0
            w = (r21 - r12) / s
            x = (r13 + r31) / s
            y = (r23 + r32) / s
            z = 0.25 * s

        norm = math.sqrt(x*x + y*y + z*z + w*w)
        if norm > 1e-10:
            x, y, z, w = x/norm, y/norm, z/norm, w/norm

        return x, y, z, w

    def _extract_pose_from_o_tee(self, o_tee):
        """Extract position and orientation from an O_T_EE homogeneous transform.

        Args:
            o_tee: 16-element homogeneous transform array (column-major 4x4).

        Returns:
            dict: { "position": {x,y,z}, "orientation": {x,y,z,w} }.

        Raises:
            ValueError: If o_tee is None or does not have exactly 16 elements.
        """
        if o_tee is None or len(o_tee) != 16:
            raise ValueError("Invalid O_T_EE data")

        # Translation lives in elements 12, 13, 14.
        tx, ty, tz = o_tee[12], o_tee[13], o_tee[14]

        # Rotation matrix rows extracted from the 4x4 transform.
        rotation_matrix = [
            o_tee[0], o_tee[1], o_tee[2],
            o_tee[4], o_tee[5], o_tee[6],
            o_tee[8], o_tee[9], o_tee[10],
        ]

        qx, qy, qz, qw = self._rotation_matrix_to_quaternion(rotation_matrix)

        return {
            "position": {
                "x": float(tx),
                "y": float(ty),
                "z": float(tz),
            },
            "orientation": {
                "x": float(qx),
                "y": float(qy),
                "z": float(qz),
                "w": float(qw),
            },
        }

    def _handle_get_current_ee_pose(self, req):
        """Serve the current end-effector pose as a JSON response.

        Args:
            req: Empty RobotQuery request.

        Returns:
            RobotQueryResponse with pose data or error details.
        """
        response = RobotQueryResponse()

        try:
            if not self.has_received_data:
                error_msg = "No FrankaState data received yet. Robot may be disconnected."
                rospy.logwarn(error_msg)

                response.result_code.result_code = ResultCode.SERVICE_NOT_RUNNING
                response.result_code.message = error_msg
                response.data = json.dumps({"error": error_msg})
                return response

            if self.latest_stamp is not None:
                time_since_update = (rospy.Time.now() - self.latest_stamp).to_sec()
                if time_since_update > self.state_timeout:
                    error_msg = f"Stale data: last update {time_since_update:.2f}s ago (timeout: {self.state_timeout}s)"
                    rospy.logwarn(error_msg)

                    response.result_code.result_code = ResultCode.TIMEOUT
                    response.result_code.message = error_msg
                    response.data = json.dumps({"error": error_msg})
                    return response

            pose = self._extract_pose_from_o_tee(self.latest_o_tee)

            payload = {
                "ee_pose": pose,
            }

            response.result_code.result_code = ResultCode.SUCCESS
            response.result_code.message = "Successfully retrieved end-effector pose"
            response.data = json.dumps(payload)

            rospy.loginfo("Successfully retrieved end-effector pose")

        except ValueError as e:
            error_msg = f"Invalid O_T_EE data: {str(e)}"
            rospy.logwarn(error_msg)

            response.result_code.result_code = ResultCode.FAILURE
            response.result_code.message = error_msg
            response.data = json.dumps({"error": error_msg})

        except Exception as e:
            error_msg = f"Unexpected error while retrieving end-effector pose: {str(e)}"
            rospy.logerr(error_msg)

            response.result_code.result_code = ResultCode.FAILURE
            response.result_code.message = error_msg
            response.data = json.dumps({"error": error_msg})

        return response

    def _handle_get_current_joints(self, req):
        """Serve the current joint positions, velocities, and efforts as a JSON response.

        Args:
            req: Empty RobotQuery request.

        Returns:
            RobotQueryResponse with joint states or error details.
        """
        response = RobotQueryResponse()

        try:
            if not self.has_received_data:
                error_msg = "No FrankaState data received yet. Robot may be disconnected."
                rospy.logwarn(error_msg)

                response.result_code.result_code = ResultCode.SERVICE_NOT_RUNNING
                response.result_code.message = error_msg
                response.data = json.dumps({"error": error_msg})
                return response

            if self.latest_stamp is not None:
                time_since_update = (rospy.Time.now() - self.latest_stamp).to_sec()
                if time_since_update > self.state_timeout:
                    error_msg = f"Stale data: last update {time_since_update:.2f}s ago (timeout: {self.state_timeout}s)"
                    rospy.logwarn(error_msg)

                    response.result_code.result_code = ResultCode.TIMEOUT
                    response.result_code.message = error_msg
                    response.data = json.dumps({"error": error_msg})
                    return response

            # Franka Panda has 7 joints; q/dq/tau_J map to position/velocity/effort.
            joint_names = ["panda_joint1", "panda_joint2", "panda_joint3",
                           "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7"]

            joints_data = {}
            error_messages = []

            for i, name in enumerate(joint_names):
                try:
                    position = None
                    velocity = None
                    effort = None

                    if self.latest_q is not None and i < len(self.latest_q):
                        position = float(self.latest_q[i])
                    if self.latest_dq is not None and i < len(self.latest_dq):
                        velocity = float(self.latest_dq[i])
                    if self.latest_tau_J is not None and i < len(self.latest_tau_J):
                        effort = float(self.latest_tau_J[i])

                    joints_data[name] = {
                        "position": position,
                        "velocity": velocity,
                        "effort": effort,
                    }

                except Exception as e:
                    error_msg = f"Error processing joint '{name}': {str(e)}"
                    rospy.logwarn(error_msg)
                    error_messages.append(error_msg)

            payload = {
                "joints": joints_data,
            }

            response.result_code.result_code = ResultCode.SUCCESS
            if error_messages:
                response.result_code.message = (
                    f"Retrieved {len(joints_data)} joint state(s). "
                    f"Errors: {'; '.join(error_messages[:2])}"
                )
            else:
                response.result_code.message = (
                    f"Successfully retrieved {len(joints_data)} joint state(s)"
                )
            response.data = json.dumps(payload)

            rospy.loginfo(f"Successfully retrieved {len(joints_data)} joint states")

        except Exception as e:
            error_msg = f"Unexpected error while retrieving joint states: {str(e)}"
            rospy.logerr(error_msg)

            response.result_code.result_code = ResultCode.FAILURE
            response.result_code.message = error_msg
            response.data = json.dumps({"error": error_msg})

        return response


def main():
    """Initialize and run the node."""
    try:
        node = GetCurrentRobotStateNode()
        rospy.loginfo("Starting get current ee pose and joints service...")
        rospy.spin()

    except rospy.ROSInterruptException:
        rospy.loginfo("Shutting down get current ee pose node...")
    except Exception as e:
        rospy.logerr(f"Failed to start get current ee pose node: {str(e)}")
        raise


if __name__ == "__main__":
    main()