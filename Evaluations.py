# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("Result_visualisations/tracking_log.csv")
t = df["t"].values

# Pixel error
ex = df["px_err_x"].values
ey = df["px_err_y"].values
enorm = np.sqrt(ex**2 + ey**2)

plt.figure()
plt.plot(t, ex, label="e_x (pixels)")
plt.plot(t, ey, label="e_y (pixels)")
plt.plot(t, enorm, label="||e||", linestyle="--")
plt.xlabel("Time [s]")
plt.ylabel("Pixel Error")
plt.title("Visual Tracking Error")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Result_visualisations/fig_pixel_error.png", dpi=200)

# EE tracking error
p_ee = df[["ee_x", "ee_y", "ee_z"]].values
p_ref = df[["ref_x", "ref_y", "ref_z"]].values
ee_err = np.linalg.norm(p_ee - p_ref, axis=1)

plt.figure()
plt.plot(t, ee_err)
plt.xlabel("Time [s]")
plt.ylabel("EE Tracking Error [m]")
plt.title("UR3 End-Effector Tracking Error vs Reference")
plt.grid(True)
plt.tight_layout()
plt.savefig("Result_visualisations/fig_ee_error.png", dpi=200)

# YZ trajectory
plt.figure()
plt.plot(df["ref_y"], df["ref_z"], label="Reference Trajectory")
plt.plot(df["ee_y"], df["ee_z"], label="Actual EE Trajectory", alpha=0.7)
plt.xlabel("Y [m]")
plt.ylabel("Z [m]")
plt.axis("equal")
plt.title("Infinity Trajectory in Y–Z Plane")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Result_visualisations/fig_yz_trajectory.png", dpi=200)

print("Saved plots: fig_pixel_error.png, fig_ee_error.png, fig_yz_trajectory.png")
plt.show()
