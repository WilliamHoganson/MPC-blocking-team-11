# import matplotlib.pyplot as plt
import numpy as np
#from scipy import interpolate
import bezier

# corners are a set of 4 points, corner point, entry, control, exit

control_corner1 = np.array([9.7, 0.])
control_corner2 = np.array([9.7, 8.7])
control_corner3 = np.array([-13.6, 8.7])
control_corner4 = np.array([-13.6, 0.])

wall_corner1 = np.array([8.93, 0.69])
wall_corner2 = np.array([8.89, 7.78])
wall_corner3 = np.array([-12.73, 7.73])
wall_corner4 = np.array([-12.89, 0.72])

corner_radius = 0.7

entry_corner1 = np.array([control_corner1[0]-corner_radius, control_corner1[1]])
exit_corner1 = np.array([control_corner1[0], control_corner1[1]+corner_radius])

entry_corner2 = np.array([control_corner2[0], control_corner2[1]-corner_radius])
exit_corner2 = np.array([control_corner2[0]-corner_radius, control_corner2[1]])

entry_corner3 = np.array([control_corner3[0]+corner_radius, control_corner3[1]])
exit_corner3 = np.array([control_corner3[0], control_corner3[1]-corner_radius])

entry_corner4 = np.array([control_corner4[0], control_corner4[1]+corner_radius])
exit_corner4 = np.array([control_corner4[0]+corner_radius, control_corner4[1]])

corner1 = np.array([wall_corner1, entry_corner1, control_corner1, exit_corner1])
corner2 = np.array([wall_corner2, entry_corner2, control_corner2, exit_corner2])
corner3 = np.array([wall_corner3, entry_corner3, control_corner3, exit_corner3])
corner4 = np.array([wall_corner4, entry_corner4, control_corner4, exit_corner4])
corners = np.stack([corner1, corner2, corner3, corner4])

straight1 = np.linspace(corner1[-1], corner2[1], int(np.linalg.norm(corner1[-1] - corner2[1]) * 10), endpoint=False) #corner1 to corner2
straight2 = np.linspace(corner2[-1], corner3[1], int(np.linalg.norm(corner2[-1] - corner3[1]) * 10), endpoint=False) #corner2 to corner3
straight3 = np.linspace(corner3[-1], corner4[1], int(np.linalg.norm(corner3[-1] - corner4[1]) * 10), endpoint=False) #corner3 to corner4
straight4 = np.linspace(corner4[-1], corner1[1], int(np.linalg.norm(corner4[-1] - corner1[1]) * 10), endpoint=True) #corner4 to corner1

interp_corners = []
for corner in corners:

    curve = bezier.Curve(corner[1:].T, degree=2)
    spline_x = np.linspace(0, 1, 10)
    res = curve.evaluate_multi(spline_x)
    interp_corners.append(res.T)

    min_dist = np.linalg.norm(res.T - corner[0], axis=1)
    # print(f"Min dist from corner: {min_dist.min()}")

full_path = np.concatenate([interp_corners[0], straight1, interp_corners[1], straight2, interp_corners[2], straight3, interp_corners[3], straight4])
slowdown_entry = int(1.2 * 10.) # how far before entering the corner to start slowing down
speedup_exit = int(0 * 10.) # how far before exiting the corner to start speeding up
straight_speed = 6.
slowdown_speed = 2.
speed_lookup = np.concatenate([np.ones(len(interp_corners[0])) * slowdown_speed, np.linspace(slowdown_speed, straight_speed, speedup_exit), 
    np.ones(len(straight1[speedup_exit:len(straight1)-slowdown_entry])) * straight_speed, np.ones(len(straight1[len(straight1)-slowdown_entry:])) * slowdown_speed,
    np.ones(len(interp_corners[1])) * slowdown_speed, np.linspace(slowdown_speed, straight_speed, speedup_exit),
    np.ones(len(straight2[speedup_exit:len(straight2)-slowdown_entry])) * straight_speed, np.ones(len(straight2[len(straight2)-slowdown_entry:])) * slowdown_speed,
    np.ones(len(interp_corners[2])) * slowdown_speed, np.linspace(slowdown_speed, straight_speed, speedup_exit),
    np.ones(len(straight3[speedup_exit:len(straight3)-slowdown_entry])) * straight_speed, np.ones(len(straight3[len(straight3)-slowdown_entry:])) * slowdown_speed,
    np.ones(len(interp_corners[3])) * slowdown_speed, np.linspace(slowdown_speed, straight_speed, speedup_exit),
    np.ones(len(straight4[speedup_exit:len(straight4)-slowdown_entry])) * straight_speed, np.ones(len(straight4[len(straight4)-slowdown_entry:])) * slowdown_speed])
assert len(full_path) == len(speed_lookup)

def get_waypoints():
    return full_path

def get_speed_lookup():
    return speed_lookup