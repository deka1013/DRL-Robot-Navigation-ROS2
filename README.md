# DRL-Robot-Navigation-ROS2


Deep Reinforcement Learning for mobile robot navigation in ROS2 Gazebo simulator. Using DRL neural network (TD3, SAC), a robot learns to navigate to a random goal point in a simulated environment while avoiding obstacles. Obstacles are detected by laser readings and a goal is given to the robot in polar coordinates. Trained in ROS Gazebo simulator with PyTorch.  Originally tested with ROS 2 Foxy. The local configuration now targets Ubuntu 22.04, ROS 2 Humble, Python 3.10 and Gazebo Classic 11. CPU inference and a training update have been checked for SAC and TD3; full simulation training still requires runtime validation.

Training example:
<p align="center">
    <img width=100% src="https://github.com/reiniscimurs/DRL-Robot-Navigation-ROS2/blob/main/gif.gif">
</p>


ROS2 adapted from: https://github.com/tomasvr/turtlebot3_drlnav \
TD3 adapted from: https://github.com/reiniscimurs/DRL-robot-navigation \
SAC adapted from: https://github.com/denisyarats/pytorch_sac 


## Installation (Ubuntu 22.04 / ROS 2 Humble)

Use Bash and the Linux Python 3.10 interpreter. ROS 2 Humble must already be installed in `/opt/ros/humble`.

```bash
cd ~/DRL-Robot-Navigation-ROS2
source /opt/ros/humble/setup.bash
sudo apt update
sudo apt install python3-rosdep2 python3-colcon-common-extensions python3-pip python3-venv ros-humble-gazebo-ros-pkgs ros-humble-robot-state-publisher
# On a fresh installation only: sudo rosdep init
rosdep update
rosdep install --from-paths src --ignore-src --rosdistro humble -y
colcon build
```

The current machine already has the required training imports. Check before installing:

```bash
python3 -c "import torch, numpy, yaml, squaternion, tqdm; from torch.utils.tensorboard import SummaryWriter"
```

For an isolated Python environment that can also import system ROS packages:

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`requirements.txt` and `pyproject.toml` declare the core training dependencies. ROS packages are installed through apt/rosdep. The old Poetry lock file described unrelated dependencies and was removed; Poetry users must run `poetry lock` to generate a new lock before installing. These version ranges are not an exact environment snapshot.

## Run

In **each Bash terminal**, load the workspace (and activate `.venv` if you created it):

```bash
cd ~/DRL-Robot-Navigation-ROS2
source src/test.env
# Optional: source .venv/bin/activate
```

The environment script derives the project path automatically, loads Humble and the workspace, and defaults to `ROS_DOMAIN_ID=1` and `TURTLEBOT3_MODEL=waffle`. Training currently uses the waffle robot name; keep this model for training. All terminals must use the same domain ID.

Start the simulator in the first terminal:

```bash
ros2 launch turtlebot3_gazebo ros2_drl.launch.py
# Without the Gazebo window:
# ros2 launch turtlebot3_gazebo ros2_drl.launch.py gui:=false
```

Start training from the project root in a second terminal:

```bash
python3 src/drl_navigation_ros2/train.py
```

Training defaults to SAC with saved experience pretraining enabled. It selects CUDA when available and otherwise uses CPU. The default pretraining performs 25,000 updates and may take considerable time on CPU. To skip it, set `pretrain = False` in `train.py`.

Optional tools, from another configured terminal:

```bash
rviz2
# Or:
python3 -m tensorboard.main --logdir runs
```

## WSL2

Gazebo's GUI needs working WSLg/OpenGL. `gui:=false` disables its window, but camera sensors may still require rendering support. Check GPU availability inside the same Linux environment:

```bash
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
nvidia-smi
```

If CUDA is unavailable, training uses CPU. This project configuration does not change the Windows GPU driver or WSL GPU access.

## CUDA numerical compatibility

On the local WSL2 / RTX 3050 Laptop GPU / PyTorch 2.14.1+cu130 setup,
training intermittently produced an enormous Critic output despite finite,
small inputs and weights. Re-evaluating the saved batch on CPU and CUDA
returned normal values. `src/test.env` and `train.py` therefore default
`DISABLE_ADDMM_CUDA_LT=1` before CUDA linear operations to avoid the fused
cuBLASLt addmm path. CUDA training remains enabled. An explicit environment
value takes precedence; restart the Python process after changing it.

SAC target-network and actor update frequencies are counted per gradient
update, including across calls to `train()`. Non-finite gradient checks
remain enabled so invalid values cannot silently update model weights.
