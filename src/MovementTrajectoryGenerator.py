"""
Author: Witold Pietron
E-Mail: witek.pietron@gmail.com

Generator trajektorii chodu (kinematyka nogi + zadawanie pozycji stawów).
Ręcznie zaprojektowany chód w otwartej pętli - punkt odniesienia dla polityk RL
(patrz notebooks/MovementTrajectoryGeneration.ipynb).
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import deque

from CoppeliaSteppingRobotGymEnv import CoppeliaSteppingRobotGymEnv


class TrajectoryGenerator(CoppeliaSteppingRobotGymEnv):
    def __init__(self, step_frequency, step_length=0.08, step_height=0.04, step_offset=0.0):
        self.step_frequency = step_frequency
        print(f'Step freq: {step_frequency}')
        self.step_length = step_length
        self.step_height = step_height
        self.step_offset = step_offset
        self.initial_y = -0.10
        self.L1 = 0.065  # długość uda
        self.L2 = 0.055  # długość podudzia
        self.min_hip_action = 10000
        self.max_hip_action = -10000
        self.min_knee_action = 10000
        self.max_knee_action = -10000
        self.knee_action_list = []
        self.hip_action_list = []
        self.zmq_port = 23000
        self.number_of_sim_steps_between_control_steps = 1
        self.observation_seq_len = 2
        self.max_number_of_steps_per_episode = 150
        self.observation_history_metric = deque(maxlen=self.observation_seq_len)
        self.observation_history_critic = deque(maxlen=self.observation_seq_len)
        self.num_of_critic_only_observations = 4
        self.curriculum_weight = 1.0

    def reset(self):

        self.episode_reward = 0.0
        self.episode_steps_counter = 0
        self.cumulated_episode_reward = np.zeros(self.max_number_of_steps_per_episode)
        self.x_position = -1.0

        self.connect_to_Coppelia_API()
        self.init_parameters()

        current_observation_dict = self.get_current_joint_positions()
        current_observation = self.prepare_observation(current_observation_dict)

        for _ in range(self.observation_seq_len):
            self.observation_history_metric.append(current_observation)
            self.observation_history_critic.append(np.zeros(self.num_of_critic_only_observations))

    def foot_trajectory(self, t, phase_offset=0.0, stance_fraction=0.5):
        """
        stance_fraction ∈ (0, 1)
        - stance: stopa na ziemi, ruch do tyłu
        - swing: stopa w powietrzu, ruch do przodu
        """
        assert 0.0 < stance_fraction < 1.0, "stance_fraction must be in (0, 1)"

        tau = (t * self.step_frequency + phase_offset) % 1.0  # faza kroku

        if tau < stance_fraction:  # STANCE
            alpha = tau / stance_fraction
            x = (
                self.step_offset
                + self.step_length / 2
                - self.step_length * alpha
            )
            y = self.initial_y
        else:  # SWING
            alpha = (tau - stance_fraction) / (1.0 - stance_fraction)
            x = (
                self.step_offset
                - self.step_length / 2
                + self.step_length * alpha
            )
            y = self.initial_y + self.step_height * np.sin(np.pi * alpha)

        return x, y


    def leg_ik(self, x, y):
        c2 = x**2 + y**2

        cos_beta = (c2 - self.L1**2 - self.L2**2) / (2.0 * self.L1 * self.L2)

        cos_beta = np.clip(cos_beta, -1.0, 1.0)

        sin_beta = np.sqrt(1.0 - cos_beta**2)
        knee = np.arctan2(sin_beta, cos_beta)

        base_angle = np.arctan2(-x, -y)

        num = self.L2 * np.sin(knee)
        den = self.L1 + self.L2 * np.cos(knee)
        triangle_angle = np.arctan2(num, den)

        hip = base_angle - triangle_angle

        return hip, knee

    def forward_kinematics(self, hip, knee):
        """
        Proste zadanie kinematyki - pozycja końcówki stopy względem stawu biodrowego.
        Odwrotność leg_ik: forward_kinematics(*leg_ik(x, y)) == (x, y).

        hip  - kąt w stawie biodrowym [rad] (alpha)
        knee - kąt w stawie kolanowym [rad] (beta)
        """
        x = -self.L1 * np.sin(hip) - self.L2 * np.sin(hip + knee)
        y = -self.L1 * np.cos(hip) - self.L2 * np.cos(hip + knee)

        return x, y


    def generate_action(self, t):
        phases = {
            "LF": 0.0,
            "RF": 0.5,
            "LB": 0.5,
            "RB": 0.0,
        }

        action = []
        for leg in ["RF", "LF", "RB", "LB"]:
            x, y = self.foot_trajectory(t, phases[leg], stance_fraction=0.5)

            hip, knee = self.leg_ik(-x, y) # -x bo przy wyznaczniu trajektorii stopy zakładamy że x rośnie w stronę w którą idzie robot
            action.extend([hip, knee])

        return np.array(action, dtype=np.float32)

    def execute_action(self, action, movement_type="both"):
        assert movement_type in ["both", "hips", "knees", "None"]
        if self.min_hip_action > np.min(action[::2]):
            self.min_hip_action = np.min(action[::2])
        if self.max_hip_action < np.max(action[::2]):
            self.max_hip_action = np.max(action[::2])
        if self.min_knee_action > np.min(action[1::2]):
            self.min_knee_action = np.min(action[1::2])
        if self.max_knee_action < np.max(action[1::2]):
            self.max_knee_action = np.max(action[1::2])
        self.hip_action_list.extend(action[::2])
        self.knee_action_list.extend(action[1::2])

        if movement_type == "hips" or movement_type == "both":
            self.sim.setJointTargetPosition(self.FrontRightHipMotorHandle, action[0], [self.max_hip_velocity, None, None])
            self.sim.setJointTargetPosition(self.FrontLeftHipMotorHandle, action[2], [self.max_hip_velocity, None, None])
            self.sim.setJointTargetPosition(self.BackRightHipMotorHandle, action[4], [self.max_hip_velocity, None, None])
            self.sim.setJointTargetPosition(self.BackLeftHipMotorHandle, action[6], [self.max_hip_velocity, None, None])

        if movement_type == "knees" or movement_type == "both":
            self.sim.setJointTargetPosition(self.FrontRightKneeMotorHandle, action[1], [self.max_knee_velocity, None, None])
            self.sim.setJointTargetPosition(self.FrontLeftKneeMotorHandle, action[3], [self.max_knee_velocity, None, None])
            self.sim.setJointTargetPosition(self.BackRightKneeMotorHandle, action[5], [self.max_knee_velocity, None, None])
            self.sim.setJointTargetPosition(self.BackLeftKneeMotorHandle, action[7], [self.max_knee_velocity, None, None])

    def print_min_max_actions(self):
        print("min_hip_action:", np.rad2deg(self.min_hip_action))
        print("max_hip_action:", np.rad2deg(self.max_hip_action))
        print("min_knee_action:", np.rad2deg(self.min_knee_action))
        print("max_knee_action:", np.rad2deg(self.max_knee_action))

    def plot_trajectory(self, sample_period_ms=100):
            period = 1.0 / self.step_frequency
            dt = sample_period_ms / 1000.0

            time_steps = np.arange(0, period, dt)
            if time_steps[-1] < period - 1e-5:
                time_steps = np.append(time_steps, period)

            phase_lb = 0.5

            dense_t = np.linspace(0, period, 200)
            lb_dense_x, lb_dense_y = [], []
            for t in dense_t:
                x_l, y_l = self.foot_trajectory(t, phase_lb, stance_fraction=0.5)
                lb_dense_x.append(x_l)
                lb_dense_y.append(y_l)

            plt.figure(figsize=(14, 9))
            plt.plot(lb_dense_x, lb_dense_y, color='#1f77b4', linestyle='-', linewidth=2, label='Trajektoria stopy')
            plt.axhline(self.initial_y, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Poziom podłoża')

            stance_counter = 0

            for t in time_steps:
                x_lb, y_lb = self.foot_trajectory(t, phase_lb, stance_fraction=0.5)

                plt.scatter(x_lb, y_lb, color='#1f77b4', edgecolors='black', s=45, zorder=5)
                t_ms = int(round(t * 1000))

                is_on_ground = y_lb <= self.initial_y + 0.001
                y_offset = 15 if not is_on_ground else 15
                x_offset = 15 if t_ms < 350 else (-15 if t_ms < 700 else 0)



                if not is_on_ground or (stance_counter % 2 == 1):
                    plt.annotate(
                        f't = {t_ms} ms',
                        (x_lb, y_lb),
                        textcoords="offset points",
                        xytext=(x_offset, y_offset),
                        ha='center',
                        fontsize=12,
                        color='#114b73',
                        fontweight='bold'
                    )

                if is_on_ground:
                    stance_counter += 1

            plt.gca().invert_xaxis()

            plt.xlabel("Pozycja pozioma stopy $X$ [m]", fontsize=16)
            plt.ylabel("Pozycja pionowa stopy $Y$ [m]", fontsize=16)

            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)

            plt.grid(True, linestyle=':', alpha=0.6)
            plt.ylim(-0.105, -0.045)

            plt.legend(loc='upper right', fontsize=14, frameon=True, facecolor='white', edgecolor='lightgray')
            plt.tight_layout()

            plt.show()
