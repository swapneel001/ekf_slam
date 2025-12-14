# EKF SLAM for Landmark-Based Navigation

An implementation of the Extended Kalman Filter (EKF) SLAM algorithm based on Thrun's "Probabilistic Robotics" for online simultaneous localization and mapping with known data association.

## Overview

This ROS 2 node implements the EKF SLAM algorithm for a TurtleBot3 robot navigating in a simulated environment with 5 colored cylindrical landmarks. The robot uses range-bearing measurements derived from corner detection to simultaneously build a map of landmark positions while localizing itself.

![EKF SLAM Demo](Assets/EKFSLAM1.gif)

## Algorithm

The implementation follows the EKF SLAM algorithm from Thrun, Burgard and Fox's "Probabilistic Robotics":

**State Vector:**
```
μ = [x, y, θ, m1_x, m1_y, m2_x, m2_y, ..., mn_x, mn_y]ᵀ
```

**Covariance Matrix:**
```
Σ = [ Σ_rr   Σ_rm ]
    [ Σ_mr   Σ_mm ]
```

Where:
- `Σ_rr`: Robot pose covariance (3×3)
- `Σ_rm`, `Σ_mr`: Robot-landmark cross-covariance
- `Σ_mm`: Landmark-landmark covariance

**Key Steps:**
1. **Prediction**: Update robot pose and covariance using odometry motion model
2. **Measurement Update**: Correct state estimate using range-bearing observations
3. **Landmark Initialization**: Add new landmarks to state vector on first observation

## Demo Videos 

### EKF Slam in World Configuration 1 
[<img src="https://img.youtube.com/vi/k7Gz0vrzSmE/hqdefault.jpg" width="600" height="300"
/>](https://www.youtube.com/embed/k7Gz0vrzSmE)

### EKF Slam in World Configuration 2
[<img src="https://img.youtube.com/vi/SzE4AFgWFig/hqdefault.jpg" width="600" height="300"
/>](https://www.youtube.com/embed/SzE4AFgWFig)

### EKF Slam in World Configuration 2 with Robot Error Plot
[<img src="https://img.youtube.com/vi/bVRGi56nFlA/hqdefault.jpg" width="600" height="300"
/>](https://www.youtube.com/embed/bVRGi56nFlA)

## Installation

This package is designed to be used as a submodule in the [prob_rob_labs_ros_2](https://github.com/swapneel001/prob_rob_labs_ros_2) repository, which contains coursework for Professor Ilija Hadzic's Probabilistic Robotics course at Columbia University (Fall 2025).
```bash
# Clone the parent repository
git clone --recursive https://github.com/swapneel001/prob_rob_labs_ros_2.git

# Or add as submodule to existing prob_rob_labs_ros_2 repo, refer to parent repository for structure information
git submodule add https://github.com/swapneel001/ekf_slam.git src/ekf_slam
```

## Usage

### Option 1: Launch the SLAM System Manually
```bash
# Terminal 1: Launch Gazebo world with landmarks
ros2 launch prob_rob_labs turtlebot3_among_landmarks_launch.py

# Terminal 2: Launch EKF SLAM nodes
ros2 launch prob_rob_labs ekf_slam.launch.py

# Terminal 3: Control the robot
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```
### Option 2 (Recommended) : 
Make use of the `run_ekf_slam.sh` script in the parent repository. 

### Published Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/ekf_slam_pose` | `nav_msgs/Odometry` | Estimated robot pose with covariance |
| `/ekf_slam_landmark_covariances` | `visualization_msgs/MarkerArray` | Landmark uncertainty ellipses |
| `/tf` | `tf2_msgs/TFMessage` | Landmark transforms in map frame |

### Subscribed Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/odom` | `nav_msgs/Odometry` | Robot odometry for prediction step |
| `/vision_<color>/corners` | `prob_rob_msgs/Point2DArrayStamped` | Detected corners for each landmark |
| `/camera/camera_info` | `sensor_msgs/CameraInfo` | Camera intrinsics |

## World Configuration

The simulation world contains 5 colored cylindrical landmarks, used in different positions. 
The ground truth for these landmarks is obtained by subscribing to the `/gazebo/link_states` topic. 

## Known Limitations

### Reference Frame Offset

The SLAM algorithm initializes the robot at the origin `(0, 0, 0)`, while in the Gazebo simulation the robot actually starts at `(-1.5, 0, 0)`. This results in a consistent bias in the estimated landmark positions relative to Gazebo ground truth.
This is intentional — in real-world SLAM, the robot does not know its initial global position. The map is built relative to the robot's starting pose, which becomes the origin of the map frame.


## Tuning Parameters

Key parameters in `ekf_slam.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `M[0,0]` | 1.0e-05 | Linear velocity variance |
| `M[1,1]` | 0.001 | Angular velocity variance |
| `landmark_height` | 0.5 | Landmark height for range calculation |
| `init_inflation` | 10.0 | Inflation factor for initial landmark covariance |
| `var_d_base` | 0.1467 | Base range measurement variance - inflate by 3 outside stable range|
| `var_theta_base` | 0.000546 | Base bearing measurement variance - inflate by 3 outside stable range|

## References

- Thrun, S., Burgard, W., & Fox, D. (2005). *Probabilistic Robotics*. MIT Press.
- Course: EEME6911 Probabilistic Robotics, Columbia University, Fall 2025


## License
This project is part of academic coursework. Please refer to the parent repository for license information.
