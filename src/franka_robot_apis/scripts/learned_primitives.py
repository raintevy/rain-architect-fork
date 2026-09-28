#!/usr/bin/env python3
"""ROS1 node exposing trained policies

Calls trained policies as functions

Services:
    /robot/control/insert       (robot_api_interfaces/RobotCommand)

Request (JSON in `req`):
    insert:
        {"obj":.., "socket": {"position": {"x":.., "y":.., "z":..},
                            "orientation": {"x":.., "y":.., "z":.., "w":..}},

Response (JSON in `data`):
    {"success": bool, "reason": str, "depth": float, "steps": int}

Example service requests:
    rosservice call /robot/control/insert \
      "req: '{\"obj\": \"peg\", {\"socket\": {\"x\": 0.0, \"y\": 0.1, \"z\": 0.3}}}'"
"""

import json
import math
import torch

import numpy as np
from scipy.spatial.transform import Rotation as R

import rospy
from robot_api_interfaces.srv import RobotCommand, RobotCommandResponse
from robot_api_interfaces.msg import ResultCode
from franka_msgs.msg import FrankaState

class InsertNode:

    """ROS node serving the Insert policy

        Loads the policy, builds the observation
    """

    def __init__(self):

        """Initialize params, load policy, evaluate"""

        rospy.init_node("insert_node")

        policy_path = rospy.get_param(
            "~insert_policy_path",
            "/home/hcrlab/isaac/UWLab/logs/rsl_rl/factory/2026-09-01_peginsert/exported/policy.pt",
        )
        self.device = rospy.get_param("~device", "cuda:0")

        self.insert_policy = torch.jit.load(policy_path, map_location=self.device)
        self.insert_policy.eval()

        self.obs_dim = 28
        self.action_dim = 8

        # Initialize action configuration parameters
        self.num_arm_joints = 7
        self.action_scale = 0.02

        self.FLANGE_TO_FINGERTIP_Z = 0.18   # TODO: measure on hardware

        self.precond_xy    = 0.02
        self.precond_z     = (0.047, 0.057)
        self.precond_yaw   = 0.785
        self.precond_roll  = np.pi
        self.precond_roll_tol = 0.1   # training pinned roll at pi exactly

        self.postcond_force_threshold = rospy.get_param("~postcond_force_threshold", 15.0)
        # TODO: calibrate from a slow scripted approach to contact

        self.max_steps = rospy.get_param("~max_steps", 150)   # 10s episode at 15Hz
        self.latest_franka_state = None
        rospy.Subscriber("/franka_state_controller/franka_states",
                        FrankaState, self._state_cb, queue_size=1)

        # ===== ROS Services =====
        rospy.Service("/robot/control/insert", RobotCommand, self._handle_insert)
        rospy.loginfo("InsertNode ready: /robot/control/insert")

    def _state_cb(self, msg):
        """Subscriber callback to store the incoming robot telemetry state."""
        self.latest_franka_state = msg

    def _send_joint_targets(self, target_q):
        """Sends the joint positions to the downstream controller.
        
        Args:
            target_q: List or numpy array of 7 target joint angles.
        """
        # TODO: Implement connection to a joint position controller (e.g., publishing to a 
        # ROS topic or contacting a continuous joint streaming service).
        rospy.logdebug(f"Sending joint targets: {np.array2string(target_q, precision=4, separator=', ')}")

    def insert(self, obj, socket_pos, socket_quat_wxyz):
        """Run the insert policy execution loop under active pre/post-condition checks.
        
        Returns:
            dict: Standardized outcome summary mapping run metrics or failure diagnostics.
        """
        # 1. Corrected Wait Loop: Attribute exists but value begins as None.
        start_wait_time = rospy.Time.now()
        while self.latest_franka_state is None and not rospy.is_shutdown():
            if (rospy.Time.now() - start_wait_time).to_sec() > 5.0:
                return {"success": False, "reason": "no_franka_state_received", "steps": 0}
            rospy.sleep(0.01)

        # Cache a local snapshot for the precondition check
        initial_state = self.latest_franka_state

        # Precondition check — bail before moving if outside the trained region
        ok, failures = self._check_precondition(
            o_tee=initial_state.O_T_EE, 
            socket_pos=socket_pos, 
            socket_quat_wxyz=socket_quat_wxyz
        )
        if not ok:
            return {"success": False, "reason": "precondition_violated", "failures": failures, "steps": 0}

        # 2. Per-call tracking states
        prev_action = np.zeros(self.action_dim, dtype=np.float32)
        prev_pos = None
        prev_quat = None
        prev_t = None
        post = {}

        # 3. Execution Loop
        rate = rospy.Rate(15)
        for step in range(self.max_steps):
            if rospy.is_shutdown():
                return {"success": False, "reason": "ros_shutdown", "steps": step, **post}

            # Capture snapshot locally at loop boundary to protect against callback mutations mid-step
            state = self.latest_franka_state  
            t_now = state.header.stamp.to_sec()

            # 3. Staleness Check: Catch if the subscriber freezes or goes out-of-sync
            if prev_t is not None and math.isclose(t_now, prev_t):
                # We log a warning but proceed with a defensive catch rather than throwing to avoid unsafe pauses
                rospy.logwarn_throttle(1.0, "Stale Franka state detected inside insertion loop!")

            # Build the policy observation tensor using the tracked kinematic window
            obs = self.build_obs(
                o_tee=state.O_T_EE,
                socket_pos=socket_pos,
                socket_quat_wxyz=socket_quat_wxyz,
                t_now=t_now,
                prev_pos=prev_pos,
                prev_quat_wxyz=prev_quat,
                prev_t=prev_t,
                prev_action=prev_action
            )

            # Evaluate policy
            with torch.no_grad():
                action = self.insert_policy(obs)

            # 2. Defensive check: Slice state.q to guarantee 7 elements (ignoring gripper variations)
            target_q = self._decode_action(action, state.q[:7])
            self._send_joint_targets(target_q)

            # Evaluate postcondition metrics
            post = self._check_postcondition(
                o_tee=state.O_T_EE,
                socket_pos=socket_pos,
                socket_quat_wxyz=socket_quat_wxyz,
                wrench=state.O_F_ext_hat_K
            )

            if post["seated_geom"] or post["seated_force"]:
                return {"success": True, "reason": "seated", "steps": step + 1, **post}

            # Cache values for the finite-difference history step (recomputes pose efficiently)
            ft_pos_tensor, ft_quat_tensor = self._fingertip_pose(state.O_T_EE)
            
            prev_action = action
            prev_pos = ft_pos_tensor.numpy()
            prev_quat = ft_quat_tensor.numpy()
            prev_t = t_now

            rate.sleep()

        # 4. Timeout fallback
        return {"success": False, "reason": "timeout", "steps": self.max_steps, **post}


    # =========================================================================
    # Service Handlers
    # =========================================================================

    def _handle_insert(self, req):
        """/robot/control/insert — run the insert policy.
        
        JSON request fields:
        obj (str, required)   : name of the held object
        socket (dict, required) : {"position": {x,y,z}, "orientation": {x,y,z,w}} # ROS xyzw
        
        Response data:
        {"success": bool, "message": str, "data": {"reason": str, "steps": int, ...}}
        """
        response = RobotCommandResponse()
        
        # 1. Parse and validate
        try:
            data = json.loads(req.req)
        except (json.JSONDecodeError, AttributeError) as e:
            response.result_code.result_code = ResultCode.INVALID_INPUT
            response.result_code.message = f"Failed to parse JSON string: {str(e)}"
            return response

        if "obj" not in data or "socket" not in data:
            response.result_code.result_code = ResultCode.INVALID_INPUT
            response.result_code.message = "Missing required top-level key 'obj' or 'socket'."
            return response
        
        socket = data["socket"]
        if not isinstance(socket, dict) or "position" not in socket or "orientation" not in socket:
            response.result_code.result_code = ResultCode.INVALID_INPUT
            response.result_code.message = "'socket' must be an object containing 'position' and 'orientation'."
            return response
            
        pos = socket["position"]
        quat = socket["orientation"]
        
        # Validate nested keys
        required_pos = ['x', 'y', 'z']
        required_quat = ['x', 'y', 'z', 'w']
        
        if not (isinstance(pos, dict) and all(k in pos for k in required_pos)):
            response.result_code.result_code = ResultCode.INVALID_INPUT
            response.result_code.message = f"Position structure must contain keys: {required_pos}"
            return response
            
        if not (isinstance(quat, dict) and all(k in quat for k in required_quat)):
            response.result_code.result_code = ResultCode.INVALID_INPUT
            response.result_code.message = f"Orientation quaternion must contain keys: {required_quat}"
            return response

        # 2. Pull out the values
        obj = data["obj"]
        socket_pos = [pos['x'], pos['y'], pos['z']]
        
        # Convert orientation from ROS xyzw -> wxyz
        socket_quat_wxyz = [quat['w'], quat['x'], quat['y'], quat['z']]
        
        # 3 & 4. Call the policy with safety wrapper and pack response
        try:
            result = self.insert(obj, socket_pos, socket_quat_wxyz)
            
            response.result_code.result_code = ResultCode.SUCCESS
            response.data = json.dumps({
                "success": result.get("success", False),
                "message": result.get("reason", ""),
                "data": result,
            })
        except Exception as e:
            response.result_code.result_code = ResultCode.FAILURE
            response.result_code.message = f"Policy execution failed: {str(e)}"
            response.data = json.dumps({
                "success": False, 
                "error": str(e)
            })
            
        return response

    
    # =========================================================================
    # Shared helpers
    # =========================================================================

    def _fingertip_pose(self, o_tee):
        """Fingertip pose in robot base frame.

        Args:
            o_tee: 16 floats from FrankaState.O_T_EE, column-major 4x4.

        Returns:
            (pos, quat) — pos is torch.Tensor [x,y,z], quat is torch.Tensor
            [w,x,y,z].
        """
        # 1. Reshape the flat 16-element array into a 4x4 matrix (column-major order)
        mat = np.array(o_tee).reshape((4, 4), order="F")

        # 2. Extract base flange position and rotation matrix
        flange_pos_np = mat[0:3, 3]
        r_mat = mat[0:3, 0:3]

        # 3. Instantiate the SciPy Rotation object
        rotation = R.from_matrix(r_mat)

        # 4. Calculate fingertip offset in the robot base frame
        # Local +z always points straight out from the palm/flange face.
        local_offset = np.array([0.0, 0.0, self.FLANGE_TO_FINGERTIP_Z])
        base_offset = rotation.apply(local_offset)

        # 5. Combine flange position and offset, then convert to PyTorch
        fingertip_pos = torch.tensor(
            flange_pos_np + base_offset, dtype=torch.float32
        )

        # 6. Extract quaternion and reorder from [x, y, z, w] to [w, x, y, z]
        scipy_quat = rotation.as_quat()
        quat = torch.tensor(
            [scipy_quat[-1], scipy_quat[0], scipy_quat[1], scipy_quat[2]],
            dtype=torch.float32,
        )

        return fingertip_pos, quat

    @staticmethod
    def hole_in_fingertip_frame(hole_pos, hole_quat_wxyz, ft_pos, ft_quat_wxyz):
        """
        Computes the position and orientation of the hole relative to the fingertip frame.
        
        Args:
            hole_pos: Array-like [x, y, z] position of the hole/socket.
            hole_quat_wxyz: Array-like [w, x, y, z] quaternion of the hole/socket.
            ft_pos: Array-like [x, y, z] position of the fingertip.
            ft_quat_wxyz: Array-like [w, x, y, z] quaternion of the fingertip.
            
        Returns:
            tuple: (rel_pos, rel_quat_wxyz) as 1D numpy arrays (float32).
        """
        h_pos = np.array(hole_pos, dtype=np.float32)
        h_q = np.array(hole_quat_wxyz, dtype=np.float32)
        f_pos = np.array(ft_pos, dtype=np.float32)
        f_q = np.array(ft_quat_wxyz, dtype=np.float32)

        # Convert wxyz -> xyzw for scipy spatial transforms
        ft_quat_xyzw = np.array([f_q[1], f_q[2], f_q[3], f_q[0]])
        hole_quat_xyzw = np.array([h_q[1], h_q[2], h_q[3], h_q[0]])
        
        # Instantiate rotation objects
        r_ft = R.from_quat(ft_quat_xyzw)
        r_hole = R.from_quat(hole_quat_xyzw)
        r_ft_inv = r_ft.inv()
        
        # 1. Relative position: (hole_pos - ft_pos) rotated into the fingertip frame
        pos_diff = h_pos - f_pos
        rel_pos = r_ft_inv.apply(pos_diff).astype(np.float32)
        
        # 2. Relative rotation: compose inverse fingertip rotation with hole rotation
        r_rel = r_ft_inv * r_hole
        rel_quat_xyzw = r_rel.as_quat()
        
        # Convert xyzw back to wxyz sequence
        rel_quat_wxyz = np.array([
            rel_quat_xyzw[3], # w
            rel_quat_xyzw[0], # x
            rel_quat_xyzw[1], # y
            rel_quat_xyzw[2]  # z
        ], dtype=np.float32)
        
        return rel_pos, rel_quat_wxyz

    @staticmethod
    def fingertip_velocity(ft_pos, ft_quat_wxyz, t_now, prev_pos, prev_quat_wxyz, prev_t):
        """Finite-difference fingertip velocity in the base frame.

        Returns:
            (lin_vel[3], ang_vel[3]) as float32 numpy arrays.
            Zeros when there is no previous sample or dt is degenerate.
        """
        if prev_pos is None or prev_t is None:
            return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

        dt = t_now - prev_t
        if dt <= 1e-6:
            return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

        # 1. Linear velocity
        lin_vel = ((np.asarray(ft_pos, dtype=np.float32) - np.asarray(prev_pos, dtype=np.float32)) / dt).astype(np.float32)

        # Helper to convert [w, x, y, z] to SciPy's required [x, y, z, w]
        def _to_xyzw(q):
            q = np.asarray(q, dtype=np.float32)
            return np.array([q[1], q[2], q[3], q[0]])

        # 2. Angular velocity
        r_prev = R.from_quat(_to_xyzw(prev_quat_wxyz))
        r_now = R.from_quat(_to_xyzw(ft_quat_wxyz))

        # Compute delta rotation in the base frame: R_now = R_delta * R_prev -> R_delta = R_now * R_prev^-1
        r_delta = r_now * r_prev.inv()
        
        # Extract the rotation vector (axis scaled by angle) and divide by time step
        ang_vel = (r_delta.as_rotvec() / dt).astype(np.float32)

        return lin_vel, ang_vel

    def build_obs(self, o_tee, socket_pos, socket_quat_wxyz, t_now, prev_pos, prev_quat_wxyz, prev_t, prev_action):
        """Builds the observation vector for the policy.
        
        Args:
            o_tee: 16 floats from FrankaState.O_T_EE (column-major 4x4 matrix).
            socket_pos: Array-like [x, y, z] position of the socket.
            socket_quat_wxyz: Array-like [w, x, y, z] orientation of the socket.
            t_now: float, timestamp from the FrankaState message header.
            prev_pos: Array-like [x, y, z] or None, previous fingertip position.
            prev_quat_wxyz: Array-like [w, x, y, z] or None, previous fingertip orientation.
            prev_t: float or None, previous timestamp.
            prev_action: Array-like or torch.Tensor, previous action taken.
            
        Returns:
            torch.Tensor: A (1, self.obs_dim) tensor on the policy's target device.
        """
        # 1. Compute current fingertip pose in the robot base frame
        ft_pos, ft_quat = self._fingertip_pose(o_tee)
        
        # 2. Convert socket orientation from wxyz -> xyzw for SciPy
        socket_quat_xyzw = np.array([
            socket_quat_wxyz[1], socket_quat_wxyz[2], socket_quat_wxyz[3], socket_quat_wxyz[0]
        ], dtype=np.float32)
        r_socket = R.from_quat(socket_quat_xyzw)
        
        # 3. Rotate the target asset offset [0, 0, 0.025] from socket frame to robot base frame
        local_offset = np.array([0.0, 0.0, 0.025], dtype=np.float32)
        base_offset = r_socket.apply(local_offset)
        hole_pos = np.array(socket_pos, dtype=np.float32) + base_offset
        
        # 4. Compute the hole pose relative to the current fingertip frame (Slots 0-6)
        rel_pos, rel_quat = self.hole_in_fingertip_frame(hole_pos, socket_quat_wxyz, ft_pos.numpy(), ft_quat.numpy())
        
        # 5. Compute finite-difference fingertip velocities (Slots 7-12)
        lin_vel, ang_vel = self.fingertip_velocity(ft_pos.numpy(), ft_quat.numpy(), t_now, prev_pos, prev_quat_wxyz, prev_t)
        
        # 6. Validate and flatten prev_action (Slots 20-27)
        if isinstance(prev_action, torch.Tensor):
            prev_action_np = prev_action.detach().cpu().numpy().reshape(-1)
        else:
            prev_action_np = np.asarray(prev_action, dtype=np.float32).reshape(-1)
            
        if prev_action_np.shape[0] != self.action_dim:
            raise ValueError(f"Expected prev_action to flatten to length {self.action_dim}, got {prev_action_np.shape[0]}")
        
        # 7. Concatenate all segments into a flat observation array
        slots_0_6 = np.concatenate([rel_pos, rel_quat])
        slots_7_12 = np.concatenate([lin_vel, ang_vel])
        slots_13_19 = np.concatenate([ft_pos.numpy(), ft_quat.numpy()])
        
        flat_obs = np.concatenate([slots_0_6, slots_7_12, slots_13_19, prev_action_np])
        
        # 8. Assert length against the expected observation dimension to catch anomalies before reshaping
        if flat_obs.shape[0] != self.obs_dim:
            raise ValueError(f"Built observation shape {flat_obs.shape[0]} does not match self.obs_dim ({self.obs_dim})")
        
        # 9. Convert to PyTorch tensor, map to device, and shape to (1, self.obs_dim)
        obs_tensor = torch.tensor(flat_obs, dtype=torch.float32, device=self.device).reshape(1, self.obs_dim)
        
        return obs_tensor

    def _decode_action(self, action, current_q):
        """Turn the policy's raw output into target joint angles.
        
        Args:
            action: the policy's 8 numbers, shape (1,8) or (8,)
            current_q: current joint angles, 7 floats from FrankaState.q
            
        Returns:
            7 target joint angles.
        """
        # 0. Validate current_q length to catch mismatch versions (e.g., including fingers)
        if len(current_q) != self.num_arm_joints:
            raise ValueError(
                f"Expected current_q to have {self.num_arm_joints} elements, "
                f"but got {len(current_q)}."
            )

        # 1. Flatten and convert to a flat numpy array of 8
        if hasattr(action, 'detach'):
            action = action.detach().cpu().numpy()
        action = np.array(action).flatten()
        
        # 2. Clip to [-1, 1]
        action = np.clip(action, -1.0, 1.0)
        
        # 3. Take the arm joints, scale by configured scale factor
        scaled_action = action[:self.num_arm_joints] * self.action_scale
        
        # 4. Add to current_q element-wise
        target_q = np.array(current_q) + scaled_action
        
        return target_q

    def _check_precondition(self, o_tee, socket_pos, socket_quat_wxyz):
        """Is the fingertip inside the region the policy trained from?
        Ranges come from reset_end_effector_around_fixed_asset in env.yaml.

        Returns:
            (ok, failures) — ok is bool, failures is a list of strings naming which checks failed.
        """
        failures = []

        # 1. fingertip pose in base frame
        ft_pos, ft_quat = self._fingertip_pose(o_tee)
        ft_pos_np = ft_pos.numpy()
        ft_quat_np = ft_quat.numpy()

        # 2. offset hole pose in base frame (same +0.025 as the observation)
        socket_quat_xyzw = np.array([
            socket_quat_wxyz[1],
            socket_quat_wxyz[2],
            socket_quat_wxyz[3],
            socket_quat_wxyz[0]
        ], dtype=np.float32)
        r_socket = R.from_quat(socket_quat_xyzw)

        local_offset = np.array([0.0, 0.0, 0.025], dtype=np.float32)
        base_offset = r_socket.apply(local_offset)
        hole_pos = np.array(socket_pos, dtype=np.float32) + base_offset

        # 3. fingertip expressed in the hole frame
        ft_quat_xyzw = np.array([
            ft_quat_np[1],
            ft_quat_np[2],
            ft_quat_np[3],
            ft_quat_np[0]
        ], dtype=np.float32)
        r_ft = R.from_quat(ft_quat_xyzw)

        r_socket_inv = r_socket.inv()
        pos_diff = ft_pos_np - hole_pos
        rel_pos = r_socket_inv.apply(pos_diff)

        r_rel_rot = r_socket_inv * r_ft
        rel_euler = r_rel_rot.as_euler('xyz', degrees=False)  # [roll, pitch, yaw]
        roll, pitch, yaw = rel_euler[0], rel_euler[1], rel_euler[2]

        # 4. compare each component against its range, collecting failures
        # Translation Checks
        if not (-self.precond_xy <= rel_pos[0] <= self.precond_xy):
            failures.append(f"pos_x out of bounds: {rel_pos[0]:.4f} (expected [{-self.precond_xy}, {self.precond_xy}])")

        if not (-self.precond_xy <= rel_pos[1] <= self.precond_xy):
            failures.append(f"pos_y out of bounds: {rel_pos[1]:.4f} (expected [{-self.precond_xy}, {self.precond_xy}])")

        if not (self.precond_z[0] <= rel_pos[2] <= self.precond_z[1]):
            failures.append(f"pos_z out of bounds: {rel_pos[2]:.4f} (expected {self.precond_z})")

        # Rotation Checks (Yaw and Wrapped Roll)
        if not (-self.precond_yaw <= yaw <= self.precond_yaw):
            failures.append(f"rot_yaw out of bounds: {yaw:.4f} (expected [{-self.precond_yaw}, {self.precond_yaw}])")

        if abs(abs(roll) - self.precond_roll) > self.precond_roll_tol:
            failures.append(f"rot_roll out of bounds: {roll:.4f} (expected near ±{self.precond_roll:.4f})")

        # Pitch is not randomized/specified in env.yaml (implied 0); checking against small tolerance
        pitch_tol = 0.05
        if not (-pitch_tol <= pitch <= pitch_tol):
            failures.append(f"rot_pitch out of bounds: {pitch:.4f} (expected near 0.0)")

        # 5. return (len(failures) == 0, failures)
        return len(failures) == 0, failures

    def _check_postcondition(self, o_tee, socket_pos, socket_quat_wxyz, wrench):
        """Evaluate if the peg is successfully seated inside the socket hole.
        
        Computes both a geometric depth clearance and an axial force metric 
        to handle real-world hardware variation.
        
        Args:
            o_tee: 16 floats from FrankaState.O_T_EE (column-major 4x4 matrix).
            socket_pos: Array-like [x, y, z] position of the socket.
            socket_quat_wxyz: Array-like [w, x, y, z] orientation of the socket.
            wrench: geometry_msgs/WrenchStamped or raw force vector [fx, fy, fz, tx, ty, tz].
            
        Returns:
            dict: Diagnostic data containing 'seated_geom', 'seated_force', 
                  estimated 'fingertip_z_clearance', and 'axial_force'.
        """
        # 1. Compute current fingertip position in base frame
        ft_pos, _ = self._fingertip_pose(o_tee)
        ft_pos_np = ft_pos.numpy()

        # 2. Extract socket frame transformation
        socket_quat_xyzw = np.array([
            socket_quat_wxyz[1], socket_quat_wxyz[2], socket_quat_wxyz[3], socket_quat_wxyz[0]
        ], dtype=np.float32)
        r_socket = R.from_quat(socket_quat_xyzw)
        r_socket_inv = r_socket.inv()

        # 3. Calculate target hole position (+0.025 along socket +z)
        local_hole_offset = np.array([0.0, 0.0, 0.025], dtype=np.float32)
        hole_pos = np.array(socket_pos, dtype=np.float32) + r_socket.apply(local_hole_offset)

        # 4. Compute fingertip's z clearance relative to the hole frame
        # TODO: A raw subtraction of grasp_offset_z along socket z assumes perfect alignment. 
        # Instead, we report raw fingertip z clearance. Verify the expected full-seating clearance 
        # via hardware measurement or check what the sim's success reward compares.
        pos_diff = ft_pos_np - hole_pos
        rel_pos_socket = r_socket_inv.apply(pos_diff)
        fingertip_z_clearance = rel_pos_socket[2]

        # 5. Extract force vector acting along the insertion path (socket axis z)
        if hasattr(wrench, 'wrench'):
            f_base = np.array([wrench.wrench.force.x, wrench.wrench.force.y, wrench.wrench.force.z])
        elif hasattr(wrench, 'force'):
            f_base = np.array([wrench.force.x, wrench.force.y, wrench.force.z])
        else:
            f_base = np.array(wrench[:3])

        # Transform raw base forces into the socket frame to get clean axial thrust
        f_socket = r_socket_inv.apply(f_base)
        axial_force = f_socket[2]

        # 6. Evaluate independent geometric vs mechanical criteria
        # Sim ground truth success limit: 1mm (0.001 m)
        # TODO: Replace 0.001 with measured target value offset once calibrated on hardware
        seated_geom = bool(abs(fingertip_z_clearance) <= 0.001)
        
        # Check against the absolute force value to handle unverified reaction coordinate signs safely
        # TODO: Verify the explicit sign of O_F_ext_hat_K vs socket frame orientation from real logs
        seated_force = bool(abs(axial_force) >= self.postcond_force_threshold)

        return {
            "seated_geom": seated_geom,
            "seated_force": seated_force,
            "fingertip_z_clearance": float(fingertip_z_clearance),
            "axial_force": float(axial_force)
        }

def main():
    try:
        node = InsertNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Shutting down...")
    except Exception as e:
        rospy.logerr(f"Failed: {e}")
        raise


if __name__ == "__main__":
    main()