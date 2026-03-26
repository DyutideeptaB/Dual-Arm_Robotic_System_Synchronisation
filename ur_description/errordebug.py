import pybullet as p
import pybullet_data
import os

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetSimulation()

# --- Load your Franka robot only ---
panda_root = "C:/Users/lenv/Desktop/robotics/franka_panda/franka_h2"
panda_urdf = os.path.join(panda_root, "urdf", "panda_robot.urdf")
p.setAdditionalSearchPath(panda_root)
robotB_id = p.loadURDF(panda_urdf, [0,0,0], p.getQuaternionFromEuler([0,0,0]), useFixedBase=True)

print("\nJoint info for Franka Panda:")
for i in range(p.getNumJoints(robotB_id)):
    info = p.getJointInfo(robotB_id, i)
    print(f"{i:2d} | name={info[1].decode('utf-8')} | parent={info[16]}")

p.disconnect()
