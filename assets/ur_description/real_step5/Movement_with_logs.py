import os, time, math
import numpy as np
import pybullet as p
import pybullet_data
import cv2
from filterpy.kalman import KalmanFilter
import pandas as pd  # for saving tracking_log.csv

# =================================================================
# 1. CONFIGURATION & PARAMETERS
# =================================================================
# FILE PATHS:
UR3_ROOT = "E:/Projects/Dual-Arm_Robotic_System_Synchronisation/ur_description" #Define your path
FRANKA_ROOT = "E:/Projects/Dual-Arm_Robotic_System_Synchronisation/franka_panda/franka_h2" #Define your path

# VISUAL PARAMETERS:
WIDTH, HEIGHT = 320, 240
FOV, NEAR, FAR = 75, 0.01, 3.0
AREA_TRACK = 120
AREA_LOST  = 50

# PHYSICS:
SIM_DT = 1.0 / 240.0

# --- Franka Camera Mounting Settings ---
FRANKA_TOOL_LINK = 7     # end-effector link index for camera
CAM_LENS_OFFSET = 0.1    # 10 cm in front of wrist
CAM_LOOK_DIST = 0.5      # look-ahead distance

# --- Visual Servo Gains & Limits ---
KPIX = 0.01              # pixel error -> camera velocity gain
VEL_FILTER_ALPHA = 0.2   # smoothing on camera velocity
MAX_JOINT_VEL = 4.0      # safety limit on joint speeds
DAMPING_LAMBDA = 0.05    # DLS damping

# --- Color Detection Thresholds (HSV) ---
LOWER_RED1 = np.array([0, 50, 50])
UPPER_RED1 = np.array([10, 255, 255])
LOWER_RED2 = np.array([170, 50, 50])
UPPER_RED2 = np.array([180, 255, 255])

# --- TRAJECTORY PARAMETERS ("Sleeping Eight" / Infinity) ---
CENTER_X = 0.20
CENTER_Y = 0.0
CENTER_Z = 0.70

# Slower, slightly smaller infinity for better tracking
AMP_Y = 0.18   # ~36 cm span in Y
AMP_Z = 0.18   # ~36 cm span in Z
SPEED = 0.45 #0.7    # slower figure-8

# --- Robustness Parameters ---
# Tracking state machine
TRACK_STATE = "SEARCH"      # ["SEARCH", "TRACKING", "TEMP_LOST", "LOST"]
last_seen_time = 0.0
last_cx, last_cy = WIDTH // 2, HEIGHT // 2
AREA_MIN = 80               # min blob area for high-confidence detection
TEMP_LOST_TIMEOUT = 0.5     # seconds before we consider it truly LOST

# Robust handover from Franka -> UR3 (stable 3D pick target)
pick_candidates = []        # buffered Y,Z world coords from camera
MAX_PICK_SPREAD = 0.03      # meters: max cluster radius to accept target

# UR3 goal smoothing
ur3_goal_pos_smoothed = None
UR3_GOAL_ALPHA = 0.7        # 0..1, smaller = heavier smoothing (we use 0.7 + state logic)
ur3_pos_prev = None

# Franka camera offset relative to UR3 EE (in world frame)
FRANKA_VIEW_OFFSET = np.array([0.30, 0.0, 0.25])
FRANKA_FOLLOW_GAIN = 6.0   # trajectory-following gain
FRANKA_TRACK_GAIN  = 1.5   # vision correction gain

# Visual servo pixel → velocity
KPIX = 0.015                # was 0.01


# --- Kalman initialization block ---
kf = KalmanFilter(dim_x=4, dim_z=2)
dt = SIM_DT

kf.F = np.array([
    [1,0,dt,0],
    [0,1,0,dt],
    [0,0,1,0 ],
    [0,0,0,1 ]
], dtype=float)

kf.H = np.array([
    [1,0,0,0],
    [0,1,0,0]
], dtype=float)

kf.P *= 0.05
kf.R *= 0.02
kf.Q *= 0.001
kf.x = np.zeros(4)



# --- Logging for analysis / plots ---
log = []  # list of dicts, filled each timestep

# =================================================================
# 2. ENVIRONMENT SETUP
# =================================================================
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetSimulation()
p.setGravity(0, 0, -9.81)
p.loadURDF("plane.urdf")

# --- Load UR3 ---
ur3_urdf = os.path.join(UR3_ROOT, "urdf", "ur3.urdf")
robotUR3 = p.loadURDF(
    ur3_urdf,
    [0, 0, 0.5],
    p.getQuaternionFromEuler([0, 0, 0]),
    useFixedBase=True
)

num_joints_ur3 = p.getNumJoints(robotUR3)
ur3_joints = []
ur3_ee_link = 9  # fallback
for i in range(num_joints_ur3):
    info = p.getJointInfo(robotUR3, i)
    if info[2] == p.JOINT_REVOLUTE:
        ur3_joints.append(i)
    name = info[1].decode("utf-8")
    if "tool0" in name or "ee_link" in name:
        ur3_ee_link = i

# --- Load Franka ---
franka_urdf = os.path.join(FRANKA_ROOT, "urdf", "panda_robot.urdf")
robotFranka = p.loadURDF(
    franka_urdf,
    [1.35, 0, 0.0],
    p.getQuaternionFromEuler([0, 0, np.pi]),
    useFixedBase=True
)
franka_joints = [
    i for i in range(p.getNumJoints(robotFranka))
    if p.getJointInfo(robotFranka, i)[2] == p.JOINT_REVOLUTE
]

# Calibrated initial pose for Franka (so camera looks at UR3 workspace)
calibrated_pose = [0.0, -1.2, 0.0, -1.8, 0.0, 1.80, 0.8]
for idx, q in zip(franka_joints, calibrated_pose):
    p.resetJointState(robotFranka, idx, q)

# --- Load Target Cube ---
cube_start = [0.35, 0.0, 0.6]
cube_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.035] * 3)
cube_vis = p.createVisualShape(
    p.GEOM_BOX,
    halfExtents=[0.035] * 3,
    rgbaColor=[1, 0, 0, 1]
)
cube_id = p.createMultiBody(
    baseMass=0.05,
    baseCollisionShapeIndex=cube_col,
    baseVisualShapeIndex=cube_vis,
    basePosition=cube_start
)
# make static initially
p.changeDynamics(cube_id, -1, mass=0)

# =================================================================
# 3. HELPER FUNCTIONS
# =================================================================

def get_camera_data(robot_id, link_index):
    """
    Get RGB image + camera pose from a robot link.
    Returns: rgb, R_cam (3x3), cam_pos, cam_forward (3,)
    """
    ls = p.getLinkState(robot_id, link_index)
    link_pos = np.array(ls[0])
    link_rot = np.array(p.getMatrixFromQuaternion(ls[1])).reshape(3, 3)

    # In this model: Z ~ forward, Y ~ up
    cam_forward = link_rot[:,2]   # Z-axis is camera forward
    cam_up      = link_rot[:,1]      # Panda tool Y is up
    
    cam_eye_pos = link_pos + cam_forward * CAM_LENS_OFFSET
    cam_target  = cam_eye_pos + cam_forward * CAM_LOOK_DIST

    view_mat = p.computeViewMatrix(cam_eye_pos, cam_target, cam_up)
    proj_mat = p.computeProjectionMatrixFOV(FOV, WIDTH / HEIGHT, NEAR, FAR)
    img = p.getCameraImage(WIDTH, HEIGHT, view_mat, proj_mat)
    rgb = np.ascontiguousarray(
        np.array(img[2], dtype=np.uint8).reshape((HEIGHT, WIDTH, 4))[:, :, :3]
    )
    return rgb, link_rot, cam_eye_pos, cam_forward


def get_dls_joint_vel(robot, joint_indices, tool_link, v_lin, v_ang, lam=DAMPING_LAMBDA):
    states = p.getJointStates(robot, joint_indices)
    q = [s[0] for s in states]
    dq = [0.0] * len(q)

    jac_t, jac_r = p.calculateJacobian(robot, tool_link, [0, 0, 0], q, dq, dq)

    J = np.vstack([jac_t, jac_r])   # 6 x n
    v = np.hstack([v_lin, v_ang])   # 6

    JJt = J.dot(J.T)
    inv = np.linalg.inv(JJt + lam * np.eye(6))
    return J.T.dot(inv.dot(v))


def detect_blob(rgb):
    """
    Detect red blob and return:
    (found, cx, cy, area, mask)
    area = blob area (for confidence/occlusion handling).
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = (
        cv2.inRange(hsv, LOWER_RED1, UPPER_RED1)
        | cv2.inRange(hsv, LOWER_RED2, UPPER_RED2)
    )
    M = cv2.moments(mask)

    if M["m00"] > 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        area = cv2.countNonZero(mask)
        return True, cx, cy, area, mask

    return False, 0, 0, 0, mask


def calculate_cube_location(cam_pos, cam_dir):
    """
    Project the camera ray onto a horizontal plane (approx table).
    Used for handover of cube (Y,Z) location to UR3.
    """
    TARGET_Z = 0.635  # ~ table height
    if abs(cam_dir[2]) < 0.001:
        return None  # parallel to plane

    t = (TARGET_Z - cam_pos[2]) / cam_dir[2]
    return cam_pos + t * cam_dir

def rotmat_to_quat(R):
    """3x3 rotation matrix to quaternion"""
    qw = np.sqrt(1.0 + R[0,0] + R[1,1] + R[2,2]) / 2.0
    qx = (R[2,1] - R[1,2]) / (4*qw)
    qy = (R[0,2] - R[2,0]) / (4*qw)
    qz = (R[1,0] - R[0,1]) / (4*qw)
    return [qx, qy, qz, qw]


# =================================================================
# 4. MAIN SIMULATION SETUP
# =================================================================
print(">>> STARTING TEST 1: BIG RETRACTED INFINITY (ROBUST) <<<")

# --- UR3 STATE MACHINE ---
UR3_STATE = "WAIT_FOR_CAM"  # WAIT_FOR_CAM -> APPROACH -> LIFT -> MOVE_8
target_acquired = False

start_time = time.time()
oscillation_start_time = 0.0
step_count = 0

# Hard-coded X for UR3 (safety margin)
HARDCODED_X = cube_start[0]
pick_target = np.array([HARDCODED_X, 0.0, 0.0])  # will be overwritten once target is stable

# UR3 orientation: gripper pointing strictly down
ur3_orn = p.getQuaternionFromEuler([np.pi, 0, 0])

""" # Low-pass filter state for camera velocity
vel_cam_filtered = np.zeros(3) """

# =================================================================
# 5. MAIN LOOP (wrapped in try/finally so logs always save)
# =================================================================
try:
    while True:
        p.stepSimulation()
        time.sleep(SIM_DT)
        step_count += 1
        t_now = time.time() - start_time

        # ========================================================
        # A. UR3 LOGIC (Worker)
        # ========================================================
        ur3_ee_state = p.getLinkState(robotUR3, ur3_ee_link)
        ur3_pos = np.array(ur3_ee_state[0])
        if ur3_pos_prev is None:
            ur3_vel = np.zeros(3)
        else:
            ur3_vel = (ur3_pos - ur3_pos_prev) / SIM_DT
                
        ur3_goal_pos = ur3_pos.copy()  # will be overwritten per state

        if UR3_STATE == "WAIT_FOR_CAM":
            ur3_goal_pos = ur3_pos
            if step_count % 50 == 0:
                print("UR3: Waiting for Franka to find cube...")

        elif UR3_STATE == "APPROACH":
            ur3_goal_pos = pick_target
            dist = np.linalg.norm(ur3_pos - pick_target)
            if dist < 0.05:
                print(f"[{t_now:.2f}] UR3: Attaching Cube!")
                UR3_STATE = "LIFT"
                # "Grasp": give cube mass and fix it to EE
                p.changeDynamics(cube_id, -1, mass=0.1)
                p.createConstraint(
                    robotUR3, ur3_ee_link,
                    cube_id, -1,
                    p.JOINT_FIXED,
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]
                )

        elif UR3_STATE == "LIFT":
            ur3_goal_pos = [CENTER_X, CENTER_Y, CENTER_Z]
            dist = np.linalg.norm(ur3_pos - ur3_goal_pos)
            if dist < 0.05:
                print(f"[{t_now:.2f}] UR3: Retracted & Centered. Starting Big Infinity!")
                UR3_STATE = "MOVE_8"
                oscillation_start_time = t_now

        elif UR3_STATE == "MOVE_8":
            t_osc = t_now - oscillation_start_time
            y_new = CENTER_Y + AMP_Y * math.sin(SPEED * t_osc)
            z_new = CENTER_Z + AMP_Z * math.sin(2 * SPEED * t_osc)
            ur3_goal_pos = [CENTER_X, y_new, z_new]

            if step_count % 50 == 0:
                err = np.linalg.norm(ur3_pos - ur3_goal_pos)
                if err > 0.1:
                    print(f"Lagging! EE error: {err:.3f} m")

        # --- UR3 GOAL SMOOTHING (state-dependent to avoid discontinuities) ---
        if ur3_goal_pos_smoothed is None:
            ur3_goal_pos_smoothed = np.array(ur3_goal_pos, dtype=float)
        else:
            # Heavier smoothing in non-periodic phases, no smoothing in continuous infinity
            alpha = UR3_GOAL_ALPHA if UR3_STATE != "MOVE_8" else 1.0
            ur3_goal_pos_smoothed = (
                alpha * np.array(ur3_goal_pos)
                + (1.0 - alpha) * ur3_goal_pos_smoothed
            )

        # Solve IK and send position commands
        joint_poses = p.calculateInverseKinematics(
            robotUR3,
            ur3_ee_link,
            ur3_goal_pos_smoothed.tolist(),
            ur3_orn
        )
        for i, idx in enumerate(ur3_joints):
            p.setJointMotorControl2(
                robotUR3,
                idx,
                p.POSITION_CONTROL,
                targetPosition=joint_poses[i],
                force=800,
                positionGain=0.35,
                velocityGain=0.8
            )

        # ========================================================
        # B. FRANKA LOGIC (Observer / Tracker)
        # ========================================================
        rgb, R_cam, eye_pos, cam_forward = get_camera_data(robotFranka, FRANKA_TOOL_LINK)
        found, cx, cy, area, mask = detect_blob(rgb)

        # default pixel error for logging in case LOST
        dx, dy = 0.0, 0.0

        # --- Tracking State Machine (robust to occlusions) ---
        if found and area > AREA_TRACK:
            TRACK_STATE = "TRACKING"
            last_seen_time = t_now
            last_cx, last_cy = cx, cy
        
        elif found and area > AREA_LOST:
            TRACK_STATE = "TEMP_LOST"
        
        else:
            if t_now - last_seen_time < TEMP_LOST_TIMEOUT:
                TRACK_STATE = "TEMP_LOST"
            else:
                TRACK_STATE = "LOST"


        # --- Visual Servo based on TRACK_STATE ---
        if TRACK_STATE in ["TRACKING", "TEMP_LOST"]:
            if TRACK_STATE == "TRACKING":
                dx = float(cx - WIDTH // 2)
                dy = float(cy - HEIGHT // 2)
            else:
                # TEMP_LOST: reuse last good centroid
                dx = float(last_cx - WIDTH // 2)
                dy = float(last_cy - HEIGHT // 2)

           # =========================
            # Franka phase-locked camera boom
            # =========================
            
            # Lock Franka directly to UR3's smoothed trajectory
            ur3_ref = ur3_goal_pos_smoothed.copy()
            
            franka_target_pos = ur3_ref + FRANKA_VIEW_OFFSET

            # Look at the real cube
            cube_pos, _ = p.getBasePositionAndOrientation(cube_id)
            look_vec = np.array(cube_pos) - franka_target_pos
            look_vec /= np.linalg.norm(look_vec) + 1e-6
            
            # Camera forward
            cam_forward = look_vec
            
            # True lens position
            cam_pos = franka_target_pos + cam_forward * CAM_LENS_OFFSET
            
            # Recompute exact optical axis
            look_vec = np.array(cube_pos) - cam_pos
            look_vec /= np.linalg.norm(look_vec) + 1e-6
            cam_forward = look_vec
            
            # -------------------------------
            # Visual centering correction
            # -------------------------------
            err_x = (cx - WIDTH//2) / WIDTH
            err_y = (cy - HEIGHT//2) / HEIGHT
            
            cam_right = np.cross(cam_forward, np.array([0,0,1.0]))
            cam_right /= np.linalg.norm(cam_right) + 1e-6
            cam_up = np.cross(cam_right, cam_forward)
            
            # Only small corrections now
            VIS_GAIN = 0.04
            franka_target_pos += VIS_GAIN * err_x * cam_right
            franka_target_pos += VIS_GAIN * err_y * cam_up
            
            # -------------------------------
            # Build camera orientation
            # -------------------------------
            tool_z = cam_forward                      # Panda looks along +Z
            up = np.array([0,0,1.0])
            
            # avoid singularity
            if abs(np.dot(tool_z, up)) > 0.95:
                up = np.array([1.0,0,0])
            
            tool_x = np.cross(up, tool_z)
            tool_x /= np.linalg.norm(tool_x) + 1e-6
            tool_y = np.cross(tool_z, tool_x)

            R = np.column_stack((tool_x, tool_y, tool_z))
            franka_target_orn = rotmat_to_quat(R)
            
            # Send to IK
            joint_targets = p.calculateInverseKinematics(
                robotFranka,
                FRANKA_TOOL_LINK,
                franka_target_pos.tolist(),
                franka_target_orn
            )
            
            for i, idx in enumerate(franka_joints):
                p.setJointMotorControl2(
                    robotFranka,
                    idx,
                    p.POSITION_CONTROL,
                    targetPosition=joint_targets[i],
                    force=800,
                    positionGain=0.5,
                    velocityGain=1.2
                )

            # =================================================================
            # C. Robust Coordinate Handover (Franka -> UR3)
            # =================================================================
            # Only once, and only from good tracking.
            if not target_acquired and step_count > 10 and TRACK_STATE == "TRACKING":
                _, _, eye_pos2, cam_forward2 = get_camera_data(robotFranka, FRANKA_TOOL_LINK)
                world_loc = calculate_cube_location(eye_pos2, cam_forward2)
                if world_loc is not None:
                    meas = np.array([world_loc[1], world_loc[2]])
                    kf.predict()
                    kf.update(meas)
                    pick_candidates.append([kf.x[0], kf.x[1]])

                    # Require a small cluster of stable measurements
                    if len(pick_candidates) >= 5:
                        arr = np.array(pick_candidates)
                        mean = arr.mean(axis=0)
                        spread = np.max(np.linalg.norm(arr - mean, axis=1))
                        if spread < MAX_PICK_SPREAD:
                            print(
                                f">>> TARGET ACQUIRED (stable): Y={mean[0]:.3f}, Z={mean[1]:.3f} "
                                f"(spread={spread:.3f} m) <<<"
                            )
                            pick_target = np.array([HARDCODED_X, mean[0], mean[1] + 0.04])
                            target_acquired = True
                            UR3_STATE = "APPROACH"
                        else:
                            # too noisy → reset buffer
                            pick_candidates = []

        # ========================================================
        # C. LOGGING FOR PERFORMANCE ANALYSIS
        # ========================================================
        if ur3_goal_pos_smoothed is not None:
            log.append({
                "t": float(t_now),
                "track_state": TRACK_STATE,
                "px_err_x": float(dx),
                "px_err_y": float(dy),
                "ee_x": float(ur3_pos[0]),
                "ee_y": float(ur3_pos[1]),
                "ee_z": float(ur3_pos[2]),
                "ref_x": float(ur3_goal_pos_smoothed[0]),
                "ref_y": float(ur3_goal_pos_smoothed[1]),
                "ref_z": float(ur3_goal_pos_smoothed[2]),
            })

         
        ur3_pos_prev = ur3_pos.copy()
   

        # ========================================================
        # D. Visualization
        # ========================================================
        if found:
            cv2.circle(rgb, (cx, cy), 5, (0, 255, 0), -1)
        status_text = f"{TRACK_STATE}"
        cv2.putText(rgb, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0) if TRACK_STATE != "LOST" else (0, 0, 255),
                    2)

        cv2.imshow("Test 1: Infinity (Robust)", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        # IMPORTANT: to exit cleanly and save log, press 'q' while this window is focused
        if cv2.waitKey(1) == ord('q'):
            print("Received 'q' – exiting main loop.")
            break

finally:
    # =================================================================
    # 6. SAVE LOG & CLEANUP (always executed)
    # =================================================================
    try:
        if len(log) > 0:
            df = pd.DataFrame(log)
            out_path = os.path.abspath("tracking_log.csv")
            df.to_csv(out_path, index=False)
            print(f"Saved log to {out_path} (rows: {len(log)})")
        else:
            print("Log was empty, nothing saved.")
    except Exception as e:
        print("Error while saving log:", e)

    p.disconnect()
    cv2.destroyAllWindows()
