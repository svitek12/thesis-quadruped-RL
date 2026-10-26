"""
Author: Witold Pietron
E-Mail: witek.pietron@gmail.com

Środowisko Gymnasium dla czworonożnego robota kroczącego symulowanego w CoppeliaSim
(połączenie przez ZeroMQ Remote API). Parametry nagrody, limity stawów, curriculum
i randomizacja domeny są wczytywane z pliku YAML (patrz config/env_config.example.yaml).
"""

import os
from collections import deque
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from coppeliasim_zmqremoteapi_client import RemoteAPIClient
#API functions: https://manual.coppeliarobotics.com/en/apiFunctions.htm

from LivePlot import LivePlot

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "env_config.yaml"
TRAINING_LOGS_DIR = REPO_ROOT / "training_logs"
EPOCHS_LOG_PATH = TRAINING_LOGS_DIR / "All_training_epochs_log.txt"


def load_config(config_path=DEFAULT_CONFIG_PATH):
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}. "
            f"Copy config/env_config.example.yaml to config/env_config.yaml and fill in the values."
        )
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    def find_missing(node, prefix=""):
        if isinstance(node, dict):
            return [m for k, v in node.items() for m in find_missing(v, f"{prefix}{k}.")]
        if isinstance(node, list):
            return [m for i, v in enumerate(node) for m in find_missing(v, f"{prefix}{i}.")]
        return [prefix.rstrip(".")] if node is None else []

    missing = find_missing(config)
    if missing:
        raise ValueError(f"Missing values in {config_path}: {', '.join(missing)}")
    return config


class CoppeliaSteppingRobotGymEnv(gym.Env):
    def __init__(self, control_step=10, show_reward_plot=False, observation_seq_len=2, zmq_port=23000,
                 starting_curriculum_weight=0.0, save_stats=False, config_path=DEFAULT_CONFIG_PATH):
        print(f'Control step: {control_step}')
        self.number_of_sim_steps_between_control_steps = control_step
        self.steps_per_second = 20//control_step
        assert control_step % 2 == 0, "Control step must be divisible by 2"
        self.show_reward_plot = show_reward_plot
        self.zmq_port = zmq_port
        self.observation_seq_len = observation_seq_len
        self.num_of_critic_only_observations = 4

        self.connect_to_Coppelia_API()
        self.init_parameters(config_path)

        self.observation_history_metric = deque(maxlen=2)
        self.observation_history_base = deque(maxlen=self.observation_seq_len)
        self.observation_history_critic = deque(maxlen=self.observation_seq_len)
        self.episode_steps_counter = 0
        self.episode_counter = 0
        self.episode_last_x_position = 0.0

        self.save_stats = save_stats
        if self.save_stats:
            self.velocity_list = []
            self.metric_list = []
            os.makedirs(TRAINING_LOGS_DIR / "roll_pitch", exist_ok=True)

        self.curriculum_weight = starting_curriculum_weight
        print(f'Starting curriculum weight: {self.curriculum_weight}')

        self.terminated = False
        self.truncated = False

        observation_dim = 10*2*self.observation_seq_len + self.num_of_critic_only_observations*self.observation_seq_len
        print(f'Observation dimension: {observation_dim}')

        self.observation_space = spaces.Box(
            low=np.array([-1] * observation_dim),
            high=np.array([1] * observation_dim),
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([-1] * 8),
            high=np.array([1] * 8),
            dtype=np.float32
        )

        self.function_tests()
        self.create_reward_plot()

    def init_parameters(self, config_path=DEFAULT_CONFIG_PATH):
        self.config = load_config(config_path)
        episode_cfg = self.config["episode"]
        joints_cfg = self.config["joints"]

        self.max_number_of_steps_per_episode = episode_cfg["max_steps"]
        self.goal_x_position = episode_cfg["goal_x"]

        self.front_hip_min_angle, self.front_hip_max_angle = np.deg2rad(joints_cfg["front_hip_range_deg"])
        self.back_hip_min_angle, self.back_hip_max_angle = np.deg2rad(joints_cfg["back_hip_range_deg"])
        self.knee_min_angle, self.knee_max_angle = np.deg2rad(joints_cfg["knee_range_deg"])

        self.max_hip_velocity = np.deg2rad(joints_cfg["max_hip_velocity_deg"])
        self.max_knee_velocity = np.deg2rad(joints_cfg["max_knee_velocity_deg"])

        self.max_abs_robot_list = np.deg2rad(joints_cfg["max_abs_tilt_deg"])

        self.episode_reward = 0.0
        self.episode_velocity_reward = 0.0
        self.episode_height_penalty = 0.0
        self.episode_rotation_penalty = 0.0
        self.episode_parallel_legs_penalty = 0.0
        self.action_regularisation_penalty = 0.0

        self.previous_action = None

        self.BackRightKneeTorques = []
        self.BackLeftKneeTorques = []

        self.BackRightHipTorques = []
        self.BackLeftHipTorques = []

        self.roll_list = []
        self.pitch_list = []

    def create_reward_plot(self):
        if self.show_reward_plot:
            self.reward_plot = LivePlot()

    def function_tests(self):
        assert abs(self.from_action_to_hip_position_space(-1, back_hip=False) - self.front_hip_min_angle) < 0.0001
        assert abs(self.from_action_to_hip_position_space(-1, back_hip=True) - self.back_hip_min_angle) < 0.0001
        assert abs(self.from_action_to_hip_position_space(0, back_hip=False) - (self.front_hip_min_angle+self.front_hip_max_angle)/2) < 0.0001
        assert abs(self.from_action_to_hip_position_space(0, back_hip=True) - (self.back_hip_min_angle+self.back_hip_max_angle)/2) < 0.0001
        assert abs(self.from_action_to_hip_position_space(1, back_hip=False) - self.front_hip_max_angle) < 0.0001
        assert abs(self.from_action_to_hip_position_space(1, back_hip=True) - self.back_hip_max_angle) < 0.0001

        assert abs(self.from_action_to_knee_position_space(-1) - self.knee_min_angle) < 0.0001
        assert abs(self.from_action_to_knee_position_space(0) - (self.knee_min_angle+self.knee_max_angle)/2) < 0.0001
        assert abs(self.from_action_to_knee_position_space(1) - self.knee_max_angle) < 0.0001

        assert self.from_hip_position_to_observation_space(self.front_hip_min_angle, back_hip=False) == -1
        assert self.from_hip_position_to_observation_space(self.back_hip_min_angle, back_hip=True) == -1
        assert self.from_hip_position_to_observation_space((self.front_hip_min_angle+self.front_hip_max_angle)/2, back_hip=False) == 0.0
        assert self.from_hip_position_to_observation_space((self.back_hip_min_angle+self.back_hip_max_angle)/2, back_hip=True) == 0.0
        assert self.from_hip_position_to_observation_space(self.front_hip_max_angle, back_hip=False) == 1
        assert self.from_hip_position_to_observation_space(self.back_hip_max_angle, back_hip=True) == 1

        assert self.from_knee_position_to_observation_space(self.knee_min_angle) == -1
        assert abs(self.from_knee_position_to_observation_space((self.knee_min_angle+self.knee_max_angle)/2) - 0.0) < 0.0001
        assert self.from_knee_position_to_observation_space(self.knee_max_angle) == 1

        assert self.from_observation_to_gyroscope_space(-1) == -self.max_abs_robot_list
        assert self.from_observation_to_gyroscope_space(0.0) == 0.0
        assert self.from_observation_to_gyroscope_space(1) == self.max_abs_robot_list

        assert self.from_gyroscope_to_observation_space(-self.max_abs_robot_list-1) == -1
        assert self.from_gyroscope_to_observation_space(-self.max_abs_robot_list) == -1
        assert self.from_gyroscope_to_observation_space(0) == 0
        assert self.from_gyroscope_to_observation_space(self.max_abs_robot_list) == 1
        assert self.from_gyroscope_to_observation_space(self.max_abs_robot_list+1) == 1

    def connect_to_Coppelia_API(self):
        # create a client to connect to zmqRemoteApi server:
        # (creation arguments can specify different host/port,
        # defaults are host='localhost', port=23000)
        client = RemoteAPIClient(port=self.zmq_port)

        # get a remote object:
        self.sim = client.require('sim')

        self.sim.setStepping(True)
        self.sim.startSimulation()

        self.FrontRightHipMotorHandle = self.sim.getObject('/Baza/PrzedniPrawyStawBiodrowy')
        self.FrontLeftHipMotorHandle = self.sim.getObject('/Baza/PrzedniLewyStawBiodrowy')
        self.BackRightHipMotorHandle = self.sim.getObject('/Baza/TylnyPrawyStawBiodrowy')
        self.BackLeftHipMotorHandle = self.sim.getObject('/Baza/TylnyLewyStawBiodrowy')

        self.FrontRightKneeMotorHandle = self.sim.getObject('/Baza/PrzedniPrawyStawKolanowy')
        self.FrontLeftKneeMotorHandle = self.sim.getObject('/Baza/PrzedniLewyStawKolanowy')
        self.BackRightKneeMotorHandle = self.sim.getObject('/Baza/TylnyPrawyStawKolanowy')
        self.BackLeftKneeMotorHandle = self.sim.getObject('/Baza/TylnyLewyStawKolanowy')

        self.RobotBaseHandle = self.sim.getObject('/Baza')

    def stop_simulation(self):
        self.sim.stopSimulation()
        while self.sim.getSimulationState()!=self.sim.simulation_stopped:
            pass

    def restart_simulation(self):
        self.stop_simulation()
        self.sim.setStepping(True)
        self.sim.startSimulation()

    def close(self):
        try:
            self.stop_simulation()
        except Exception as e:
            print(f'[{self.zmq_port}] close(): could not stop simulation cleanly: {e}')

    def curriculum_range(self, base_and_gain_deg):
        base, gain = base_and_gain_deg
        return np.deg2rad(base + gain*self.curriculum_weight)

    def init_joint_positions(self):
        init_front_hip_positions = self.from_action_to_hip_position_space(0.0, back_hip=False)
        init_back_hip_positions = self.from_action_to_hip_position_space(0.0, back_hip=True)
        init_knee_states = self.from_action_to_knee_position_space(0.0)
        max_tilt = self.curriculum_range(self.config["randomization"]["joint_noise_deg"])

        self.sim.setJointTargetPosition(self.FrontRightHipMotorHandle, init_front_hip_positions+ np.random.uniform(-max_tilt, max_tilt), [self.max_hip_velocity, None, None])
        self.sim.setJointTargetPosition(self.FrontLeftHipMotorHandle, init_front_hip_positions+ np.random.uniform(-max_tilt, max_tilt), [self.max_hip_velocity, None, None])
        self.sim.setJointTargetPosition(self.BackRightHipMotorHandle, init_back_hip_positions+ np.random.uniform(-max_tilt, max_tilt), [self.max_hip_velocity, None, None])
        self.sim.setJointTargetPosition(self.BackLeftHipMotorHandle, init_back_hip_positions+ np.random.uniform(-max_tilt, max_tilt), [self.max_hip_velocity, None, None])

        self.sim.setJointTargetPosition(self.FrontRightKneeMotorHandle, init_knee_states+ np.random.uniform(-max_tilt, max_tilt), [self.max_knee_velocity, None, None])
        self.sim.setJointTargetPosition(self.FrontLeftKneeMotorHandle, init_knee_states+ np.random.uniform(-max_tilt, max_tilt), [self.max_knee_velocity, None, None])
        self.sim.setJointTargetPosition(self.BackRightKneeMotorHandle, init_knee_states+ np.random.uniform(-max_tilt, max_tilt), [self.max_knee_velocity, None, None])
        self.sim.setJointTargetPosition(self.BackLeftKneeMotorHandle, init_knee_states+ np.random.uniform(-max_tilt, max_tilt), [self.max_knee_velocity, None, None])

        self.make_simulation_steps(5)

    def get_current_joint_positions(self):

        FrontRightHipPosition = self.from_hip_position_to_observation_space(self.sim.getJointPosition(self.FrontRightHipMotorHandle))
        FrontLeftHipPosition = self.from_hip_position_to_observation_space(self.sim.getJointPosition(self.FrontLeftHipMotorHandle))
        BackRightHipPosition = self.from_hip_position_to_observation_space(self.sim.getJointPosition(self.BackRightHipMotorHandle))
        BackLeftHipPosition = self.from_hip_position_to_observation_space(self.sim.getJointPosition(self.BackLeftHipMotorHandle))

        RobotRoll, RobotPitch, _ = self.sim.getObjectOrientation(self.RobotBaseHandle,-1)
        RobotRoll, RobotPitch = self.from_gyroscope_to_observation_space(RobotRoll), self.from_gyroscope_to_observation_space(RobotPitch)

        FrontRightKneePosition = self.from_knee_position_to_observation_space(self.sim.getJointPosition(self.FrontRightKneeMotorHandle))
        FrontLeftKneePosition = self.from_knee_position_to_observation_space(self.sim.getJointPosition(self.FrontLeftKneeMotorHandle))
        BackRightKneePosition = self.from_knee_position_to_observation_space(self.sim.getJointPosition(self.BackRightKneeMotorHandle))
        BackLeftKneePosition = self.from_knee_position_to_observation_space(self.sim.getJointPosition(self.BackLeftKneeMotorHandle))

        observation_dict = {
            "FrontRightHipPosition": FrontRightHipPosition,
            "FrontLeftHipPosition": FrontLeftHipPosition,
            "BackRightHipPosition": BackRightHipPosition,
            "BackLeftHipPosition": BackLeftHipPosition,
            "FrontRightKneePosition": FrontRightKneePosition,
            "FrontLeftKneePosition": FrontLeftKneePosition,
            "BackRightKneePosition": BackRightKneePosition,
            "BackLeftKneePosition": BackLeftKneePosition,
            "RobotRoll": RobotRoll,
            "RobotPitch": RobotPitch
        }

        return observation_dict

    def prepare_observation(self, observation_dict):
        observation = [
            observation_dict["FrontRightHipPosition"], observation_dict["FrontLeftHipPosition"],
            observation_dict["BackRightHipPosition"], observation_dict["BackLeftHipPosition"],
            observation_dict["FrontRightKneePosition"], observation_dict["FrontLeftKneePosition"],
            observation_dict["BackRightKneePosition"], observation_dict["BackLeftKneePosition"],
            observation_dict["RobotRoll"], observation_dict["RobotPitch"]
        ]
        return np.array(observation, dtype=np.float32)

    def concatenate_observations(self):
        observation_agent = np.array(self.observation_history_base, dtype=np.float32).reshape(-1)
        observation_critic = np.array(self.observation_history_critic, dtype=np.float32).reshape(-1)
        whole_observation = np.concatenate((observation_agent, observation_agent, observation_critic), axis=0)
        return np.array(whole_observation, dtype=np.float32)

    def make_simulation_steps(self, number_of_steps=1):
        for _ in range(number_of_steps):
            self.sim.step()

    def from_action_to_hip_position_space(self, act, back_hip=False):
        if back_hip:
            return float((act+1)*(self.back_hip_max_angle-self.back_hip_min_angle)/2+self.back_hip_min_angle)
        else:
            return float((act+1)*(self.front_hip_max_angle-self.front_hip_min_angle)/2+self.front_hip_min_angle)

    def from_action_to_knee_position_space(self, act):
        return float((act+1)*(self.knee_max_angle-self.knee_min_angle)/2+self.knee_min_angle)

    def from_hip_position_to_observation_space(self, hip_position, back_hip=False):
        if back_hip:
            return float((hip_position-self.back_hip_min_angle)/(self.back_hip_max_angle-self.back_hip_min_angle)*2-1)
        else:
            return float((hip_position-self.front_hip_min_angle)/(self.front_hip_max_angle-self.front_hip_min_angle)*2-1)

    def from_knee_position_to_observation_space(self, knee_position):
        return float((knee_position-self.knee_min_angle)/(self.knee_max_angle-self.knee_min_angle)*2-1)

    def from_gyroscope_to_observation_space(self, gyro_position):
        return np.clip(float((gyro_position+self.max_abs_robot_list)/(2*self.max_abs_robot_list)*2-1), -1, 1)

    def from_observation_to_gyroscope_space(self, act):
        return float((act+1)*(self.max_abs_robot_list)-self.max_abs_robot_list)

    def print_joint_velocities(self):
        knee_velocities = [
            np.rad2deg(self.sim.getJointVelocity(self.FrontRightKneeMotorHandle)),
            np.rad2deg(self.sim.getJointVelocity(self.FrontLeftKneeMotorHandle)),
            np.rad2deg(self.sim.getJointVelocity(self.BackRightKneeMotorHandle)),
            np.rad2deg(self.sim.getJointVelocity(self.BackLeftKneeMotorHandle))
        ]
        hip_velocities = [
            np.rad2deg(self.sim.getJointVelocity(self.FrontRightHipMotorHandle)),
            np.rad2deg(self.sim.getJointVelocity(self.FrontLeftHipMotorHandle)),
            np.rad2deg(self.sim.getJointVelocity(self.BackRightHipMotorHandle)),
            np.rad2deg(self.sim.getJointVelocity(self.BackLeftHipMotorHandle))
        ]
        print(f'Knee velocities: {knee_velocities}')
        print(f'Hip velocities: {hip_velocities}')

    def get_joint_torques(self):
        knee_torques = [
            self.sim.getJointForce(self.BackRightKneeMotorHandle),
            self.sim.getJointForce(self.BackLeftKneeMotorHandle)
        ]
        hip_torques = [
            self.sim.getJointForce(self.BackRightHipMotorHandle),
            self.sim.getJointForce(self.BackLeftHipMotorHandle)
        ]
        self.BackRightKneeTorques.append(knee_torques[0])
        self.BackLeftKneeTorques.append(knee_torques[1])
        self.BackRightHipTorques.append(hip_torques[0])
        self.BackLeftHipTorques.append(hip_torques[1])

    def backward_diff_3point(self, x_t, x_t_h, x_t_2h, h):
        return (3*x_t-4*x_t_h+x_t_2h)/(2*h)

    def update_curriculum(self):
        curriculum_cfg = self.config["curriculum"]
        #episode_counter liczy się dla każdej instancji środowiska osobno
        if self.episode_counter % curriculum_cfg["episodes_per_increase"] == 0 and self.curriculum_weight < curriculum_cfg["max_weight"]:
            previous_weight = self.curriculum_weight
            self.curriculum_weight = min(curriculum_cfg["max_weight"], self.curriculum_weight + curriculum_cfg["increment"])
            os.makedirs(TRAINING_LOGS_DIR, exist_ok=True)
            with open(EPOCHS_LOG_PATH, "a") as f:
                f.write(f"Curriculum weight: {previous_weight} -> {self.curriculum_weight}\n")

    def reset(self, seed=None, options=None):

        print(f'EPISODE LEN {self.episode_steps_counter} REW: {self.episode_reward:.4f} VEL REW: {self.episode_velocity_reward:.4f} HEIGHT PEN: {self.episode_height_penalty:.4f} ROTATION PEN: {self.episode_rotation_penalty:.4f} PARALLEL PEN: {self.episode_parallel_legs_penalty:.4f} ACTION REG PEN: {self.action_regularisation_penalty:.4f}')
        if self.episode_steps_counter > 0:
            print(f'EPISODE MEAN VELOCITY {self.episode_last_x_position/self.episode_steps_counter*self.steps_per_second:.4f}')
            if self.save_stats:
                self.velocity_list.append(self.episode_last_x_position/self.episode_steps_counter*self.steps_per_second)
                self.metric_list.append(self.episode_reward)

        self.episode_steps_counter = 0
        self.episode_counter += 1

        self.episode_reward = 0.0
        self.episode_velocity_reward = 0.0
        self.episode_height_penalty = 0.0
        self.episode_rotation_penalty = 0.0
        self.episode_parallel_legs_penalty = 0.0
        self.action_regularisation_penalty = 0.0

        if len(self.roll_list) > 0 and len(self.pitch_list) > 0:
            mean_roll = np.mean(self.roll_list)
            mean_pitch = np.mean(self.pitch_list)
            df = pd.DataFrame({'Roll': self.roll_list, 'Pitch': self.pitch_list})
            df.to_csv(TRAINING_LOGS_DIR / "roll_pitch" / f"Roll_Pitch_{self.episode_counter}.csv", index=False)
            print(f'Mean roll: {mean_roll:.4f} deg, Mean pitch: {mean_pitch:.4f} deg')
        self.roll_list = []
        self.pitch_list = []

        self.episode_last_x_position = 0.0

        self.update_curriculum()

        super().reset(seed=seed)
        self.restart_simulation()

        self.init_joint_positions()

        max_tilt = self.curriculum_range(self.config["randomization"]["base_tilt_deg"])
        roll  = np.random.uniform(-max_tilt, max_tilt)
        pitch = np.random.uniform(-max_tilt, max_tilt)
        self.sim.setObjectOrientation(self.RobotBaseHandle, -1, [roll, pitch, 0])
        self.make_simulation_steps(2)

        current_observation_dict = self.get_current_joint_positions()
        current_observation = self.prepare_observation(current_observation_dict)

        for _ in range(2):
            self.observation_history_metric.append(current_observation)

        for _ in range(self.observation_seq_len):
            self.observation_history_base.append(current_observation)
            self.observation_history_critic.append(np.zeros(self.num_of_critic_only_observations))

        whole_observation = self.concatenate_observations()

        self.previous_action = current_observation[:8]
        imu_bias_std = np.deg2rad(self.config["randomization"]["imu_bias_std_deg"])
        self.imu_bias_roll = np.random.normal(0, imu_bias_std)
        self.imu_bias_pitch = np.random.normal(0, imu_bias_std)

        info = {}

        self.terminated = False
        self.truncated = False
        if self.show_reward_plot:
            self.reward_plot.clear()

        return whole_observation, info

    def get_real_joint_positions(self):
        FrontRightHipPosition = self.sim.getJointPosition(self.FrontRightHipMotorHandle)
        FrontLeftHipPosition = self.sim.getJointPosition(self.FrontLeftHipMotorHandle)
        BackRightHipPosition = self.sim.getJointPosition(self.BackRightHipMotorHandle)
        BackLeftHipPosition = self.sim.getJointPosition(self.BackLeftHipMotorHandle)

        FrontRightKneePosition = self.sim.getJointPosition(self.FrontRightKneeMotorHandle)
        FrontLeftKneePosition = self.sim.getJointPosition(self.FrontLeftKneeMotorHandle)
        BackRightKneePosition = self.sim.getJointPosition(self.BackRightKneeMotorHandle)
        BackLeftKneePosition = self.sim.getJointPosition(self.BackLeftKneeMotorHandle)

        return np.array([FrontRightHipPosition, FrontLeftHipPosition, BackRightHipPosition, BackLeftHipPosition,
                FrontRightKneePosition, FrontLeftKneePosition, BackRightKneePosition, BackLeftKneePosition])

    def action_to_hip(self, action):
        FrontRightHipPosition = float(self.from_action_to_hip_position_space(action[0], back_hip=False))
        FrontLeftHipPosition = float(self.from_action_to_hip_position_space(action[1], back_hip=False))
        BackRightHipPosition = float(self.from_action_to_hip_position_space(action[2], back_hip=True))
        BackLeftHipPosition = float(self.from_action_to_hip_position_space(action[3], back_hip=True))

        FrontRightKneePosition = float(self.from_action_to_knee_position_space(action[4]))
        FrontLeftKneePosition = float(self.from_action_to_knee_position_space(action[5]))
        BackRightKneePosition = float(self.from_action_to_knee_position_space(action[6]))
        BackLeftKneePosition = float(self.from_action_to_knee_position_space(action[7]))

        return np.array([FrontRightHipPosition, FrontLeftHipPosition, BackRightHipPosition, BackLeftHipPosition,
                FrontRightKneePosition, FrontLeftKneePosition, BackRightKneePosition, BackLeftKneePosition])

    def calculate_reward(self, RobotBasePosition, RobotBaseOrientation, velocity, action):
        reward_cfg = self.config["reward"]
        termination_cfg = self.config["termination"]

        velocity_reward = velocity*reward_cfg["velocity_weight"]

        if velocity_reward < 0:
            velocity_reward = velocity_reward*reward_cfg["backward_velocity_multiplier"]

        height_penalty = - max(0, reward_cfg["min_height"]-RobotBasePosition[2])*reward_cfg["height_weight"]

        # kara tylko za przechył przekraczający tolerancję
        roll, pitch = abs(RobotBaseOrientation[0]), abs(RobotBaseOrientation[1])
        excess = reward_cfg["pitch_excess_scale"] * max(0.0, pitch - np.deg2rad(reward_cfg["pitch_tolerance_deg"])) \
                 + max(0.0, roll - np.deg2rad(reward_cfg["roll_tolerance_deg"]))
        rotation_penalty = -excess * reward_cfg["rotation_weight"]

        # kara za równoległy ruch bioder pary nóg (promuje chód naprzemienny)
        curr_obs = self.observation_history_metric[-1]
        prev_obs = self.observation_history_metric[-2]
        delta_front_right_hip = curr_obs[0] - prev_obs[0]
        delta_front_left_hip  = curr_obs[1] - prev_obs[1]
        delta_back_right_hip = curr_obs[2] - prev_obs[2]
        delta_back_left_hip  = curr_obs[3] - prev_obs[3]
        parallel_legs_penalty = -abs(delta_back_right_hip + delta_back_left_hip)*reward_cfg["parallel_legs_weight"]*self.curriculum_weight
        parallel_legs_penalty += -abs(delta_front_right_hip + delta_front_left_hip)*reward_cfg["parallel_legs_weight"]*self.curriculum_weight*reward_cfg["front_legs_parallel_scale"]

        alive_reward = reward_cfg["alive_reward"]

        # kara za gwałtowne zmiany akcji między krokami sterowania
        if action is not None:
            action_regularisation = -np.mean(np.abs(action - self.previous_action))*reward_cfg["action_regularisation_weight"]*self.curriculum_weight
            self.previous_action = action
        else:
            action_regularisation = 0.0

        reward = velocity_reward+height_penalty+rotation_penalty+parallel_legs_penalty+alive_reward + action_regularisation

        if abs(RobotBaseOrientation[0]) > np.deg2rad(termination_cfg["max_roll_deg"]):
            self.terminated = True
            reward -= termination_cfg["roll_penalty"]
        elif abs(RobotBaseOrientation[2]) > np.deg2rad(termination_cfg["max_yaw_deg"]):
            self.terminated = True
            reward -= termination_cfg["yaw_penalty"]
        elif RobotBasePosition[0] >= self.goal_x_position:
            self.terminated = True
            reward += termination_cfg["goal_bonus"]

        self.episode_reward += reward
        self.episode_velocity_reward += velocity_reward
        self.episode_height_penalty += height_penalty
        self.episode_rotation_penalty += rotation_penalty
        self.episode_parallel_legs_penalty += parallel_legs_penalty
        self.action_regularisation_penalty += action_regularisation

        return reward

    def step(self, action):
        # biodra i kolana ustawiane z przesunięciem o pół kroku sterowania
        self.sim.setJointTargetPosition(self.FrontRightHipMotorHandle, float(self.from_action_to_hip_position_space(action[0], back_hip=False)), [self.max_hip_velocity, None, None])
        self.sim.setJointTargetPosition(self.FrontLeftHipMotorHandle, float(self.from_action_to_hip_position_space(action[1], back_hip=False)), [self.max_hip_velocity, None, None])
        self.sim.setJointTargetPosition(self.BackRightHipMotorHandle, float(self.from_action_to_hip_position_space(action[2], back_hip=True)), [self.max_hip_velocity, None, None])
        self.sim.setJointTargetPosition(self.BackLeftHipMotorHandle, float(self.from_action_to_hip_position_space(action[3], back_hip=True)), [self.max_hip_velocity, None, None])

        self.make_simulation_steps(self.number_of_sim_steps_between_control_steps//2)

        self.sim.setJointTargetPosition(self.FrontRightKneeMotorHandle, float(self.from_action_to_knee_position_space(action[4])), [self.max_knee_velocity, None, None])
        self.sim.setJointTargetPosition(self.FrontLeftKneeMotorHandle, float(self.from_action_to_knee_position_space(action[5])), [self.max_knee_velocity, None, None])
        self.sim.setJointTargetPosition(self.BackRightKneeMotorHandle, float(self.from_action_to_knee_position_space(action[6])), [self.max_knee_velocity, None, None])
        self.sim.setJointTargetPosition(self.BackLeftKneeMotorHandle, float(self.from_action_to_knee_position_space(action[7])), [self.max_knee_velocity, None, None])

        self.make_simulation_steps(self.number_of_sim_steps_between_control_steps//2)

        RobotBasePosition = self.sim.getObjectPosition(self.RobotBaseHandle,-1)
        RobotBaseOrientation = self.sim.getObjectOrientation(self.RobotBaseHandle,-1)

        velocity = self.sim.getObjectVelocity(self.RobotBaseHandle)[0][0]

        orientation_noise_std = np.deg2rad(self.config["randomization"]["imu_noise_std_deg"])
        noisy_roll  = RobotBaseOrientation[0] + self.imu_bias_roll + np.random.normal(0.0, orientation_noise_std)
        noisy_pitch = RobotBaseOrientation[1] + self.imu_bias_pitch + np.random.normal(0.0, orientation_noise_std)

        if self.save_stats:
            self.get_joint_torques()
            self.roll_list.append(np.rad2deg(RobotBaseOrientation[0]))
            self.pitch_list.append(np.rad2deg(RobotBaseOrientation[1]))

        obs_actor = [
            action[0], action[1], action[2], action[3],
            action[4], action[5], action[6], action[7],
            self.from_gyroscope_to_observation_space(noisy_roll), self.from_gyroscope_to_observation_space(noisy_pitch)
            ]

        obs_critic_only = [
            np.clip(float((RobotBasePosition[0]+1)/5), -1, 1), np.clip(float((RobotBasePosition[1])), -1, 1),
            np.clip(velocity, -1, 1),self.from_gyroscope_to_observation_space(RobotBaseOrientation[2])
        ]

        self.observation_history_metric.append(obs_actor)
        self.observation_history_base.append(obs_actor)
        self.observation_history_critic.append(obs_critic_only)

        observation = self.concatenate_observations()

        reward = self.calculate_reward(RobotBasePosition, RobotBaseOrientation, velocity, action)

        if self.show_reward_plot:
            self.reward_plot.update(self.episode_steps_counter, self.episode_reward, RobotBasePosition[0])

        self.episode_steps_counter += 1
        self.episode_last_x_position = RobotBasePosition[0]

        if self.episode_steps_counter >= self.max_number_of_steps_per_episode:
            self.truncated = True

        #observation, reward, terminated, truncated , info
        return observation, reward, self.terminated, self.truncated, {}

    def calculate_avg_stats(self):
        print(self.velocity_list)
        print(self.metric_list)
        print(f'Average velocity: {np.mean(self.velocity_list):.4f}')
        print(f'Average metric: {np.mean(self.metric_list):.4f}')
        print(f'Metric std.: {np.std(self.metric_list):.4f}')

    def plot_torques(self):
        hip_theoretic_max_abs_torque = 0.402*1.8
        knee_theoretic_max_abs_torque = 0.707*1.185
        plt.figure(figsize=(14, 10))

        plt.subplot(2, 2, 1)
        plt.hist(self.BackRightKneeTorques, bins=20, color='green', alpha=0.7)
        plt.title(f'Back Right Knee Torque Mean Abs: {np.mean(np.abs(self.BackRightKneeTorques)):.4f} N*m')
        plt.xlabel('Torque')
        plt.grid()
        plt.axvline(x=-knee_theoretic_max_abs_torque, color='r', linestyle='--')
        plt.axvline(x=knee_theoretic_max_abs_torque, color='r', linestyle='--')
        plt.ylabel('Frequency')

        plt.subplot(2, 2, 2)
        plt.hist(self.BackLeftKneeTorques, bins=20, color='green', alpha=0.7)
        plt.title(f'Back Left Knee Torque Mean Abs: {np.mean(np.abs(self.BackLeftKneeTorques)):.4f} N*m')
        plt.xlabel('Torque')
        plt.grid()
        plt.axvline(x=-knee_theoretic_max_abs_torque, color='r', linestyle='--')
        plt.axvline(x=knee_theoretic_max_abs_torque, color='r', linestyle='--')
        plt.ylabel('Frequency')

        plt.subplot(2, 2, 3)
        plt.hist(self.BackRightHipTorques, bins=20, color='blue', alpha=0.7)
        plt.title(f'Back Right Hip Torque Mean Abs: {np.mean(np.abs(self.BackRightHipTorques)):.4f} N*m')
        plt.xlabel('Torque')
        plt.grid()
        plt.axvline(x=-hip_theoretic_max_abs_torque, color='r', linestyle='--')
        plt.axvline(x=hip_theoretic_max_abs_torque, color='r', linestyle='--')
        plt.ylabel('Frequency')

        plt.subplot(2, 2, 4)
        plt.hist(self.BackLeftHipTorques, bins=20, color='blue', alpha=0.7)
        plt.title(f'Back Left Hip Torque Mean Abs: {np.mean(np.abs(self.BackLeftHipTorques)):.4f} N*m')
        plt.xlabel('Torque')
        plt.grid()
        plt.axvline(x=-hip_theoretic_max_abs_torque, color='r', linestyle='--')
        plt.axvline(x=hip_theoretic_max_abs_torque, color='r', linestyle='--')
        plt.ylabel('Frequency')

        plt.tight_layout()
        plt.show()
