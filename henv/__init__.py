import gymnasium
from .customized_humanoid import CustomizedHumanoidEnv
import gym

gymnasium.register(
    id="CustomizedHumanoid-v4",
    entry_point=CustomizedHumanoidEnv,
    max_episode_steps=1000,
)
from humanoid_bench.env import ROBOTS, TASKS
from . import customized_humanoid_bench
for robot in ROBOTS:
    if robot == "g1" or robot == "digit":
        control = "torque"
    else:
        control = "pos"
    for task, task_info in TASKS.items():
        task_info = task_info()
        kwargs = task_info.kwargs.copy()
        kwargs["robot"] = robot
        kwargs["control"] = control
        kwargs["task"] = task
        gymnasium.register(
            id=f"{robot}-{task}-customized-v0",
            entry_point=customized_humanoid_bench.HumanoidEnv,
            max_episode_steps=task_info.max_episode_steps,
            kwargs=kwargs,
        )
for robot in ROBOTS:
    if robot == "g1" or robot == "digit":
        control = "torque"
    else:
        control = "pos"
    for task, task_info in TASKS.items():
        task_info = task_info()
        kwargs = task_info.kwargs.copy()
        kwargs["robot"] = robot
        kwargs["control"] = control
        kwargs["task"] = task
        gym.register(
            id=f"{robot}-{task}-customized-v0",
            entry_point=customized_humanoid_bench.HumanoidEnv,
            max_episode_steps=task_info.max_episode_steps,
            kwargs=kwargs,
        )
