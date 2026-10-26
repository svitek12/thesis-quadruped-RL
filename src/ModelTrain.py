"""
Author: Witold Pietron
E-Mail: witek.pietron@gmail.com
"""

import argparse
import json
import os
from collections import deque
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO, A2C, SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from CoppeliaSteppingRobotGymEnv import CoppeliaSteppingRobotGymEnv, REPO_ROOT, TRAINING_LOGS_DIR, EPOCHS_LOG_PATH
from CustomSeparatePolicy import CustomSeparatePolicy, ExportableActor

TRAINED_MODELS_DIR = REPO_ROOT / "trained_models"
PORTS = [23000, 23001, 23002, 23003]


def export_model_weights(model, path=TRAINED_MODELS_DIR / "WEIGHTS" / "ulab_weights_MLP.json"):
    # Export for micropython with ulab
    state = model.state_dict()  # OrderedDict - zachowuje kolejność zapisu modelu
    layers = {}
    idx = 0

    for key in state.keys():
        if key.endswith(".weight"):
            tensor = state[key]
            if tensor.dim() != 2:
                continue

            W = tensor.detach().cpu().numpy()
            base = key.rsplit(".", 1)[0]
            bias_key = base + ".bias"

            if bias_key in state:
                b = state[bias_key].detach().cpu().numpy()
            else:
                b = np.zeros(W.shape[0], dtype=W.dtype)

            layers[str(idx)] = {
                "name": base,
                "in": int(W.shape[1]),
                "out": int(W.shape[0]),
                "W": W.tolist(),
                "b": b.tolist()
            }
            idx += 1

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(layers, f, indent=2)


def export_actor_onnx(model, path=TRAINED_MODELS_DIR / "ONNX" / "Actor.onnx"):
    actor_only_net = ExportableActor(model.policy.cpu()).eval()
    obs_dim = model.observation_space.shape[0]
    dummy = torch.randn(1, obs_dim)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.onnx.export(actor_only_net, dummy, str(path),
        input_names=["obs"], output_names=["action"], opset_version=11)


def make_env(control_step, port, obs_seq_len, start_curriculum_weight=0.1):
    def _init():
        return CoppeliaSteppingRobotGymEnv(control_step=control_step, zmq_port=port, observation_seq_len=obs_seq_len, starting_curriculum_weight=start_curriculum_weight)
    return _init


def make_vec_env(training_nickname, control_step, obs_seq_len, start_curriculum_weight=0.1):
    os.makedirs(TRAINING_LOGS_DIR, exist_ok=True)
    vec_env = SubprocVecEnv([make_env(control_step, port, obs_seq_len, start_curriculum_weight) for port in PORTS])
    return VecMonitor(vec_env, filename=str(TRAINING_LOGS_DIR / f"training_{training_nickname}_cs{control_step}_{str(date.today())}.csv"))


def separate_policy_kwargs(obs_seq_len):
    return dict(activation_fn=nn.ReLU,
                net_arch_actor=[80, 40], net_arch_critic=[80, 40],
                actor_dim=10*obs_seq_len, critic_dim=14*obs_seq_len)


def train_PPO():
    control_step = 2
    obs_seq_len = 2
    training_nickname = "PPO_v63"
    model_name = 'PPO'

    SteppingRobotEnv = make_vec_env(training_nickname, control_step, obs_seq_len, 0.1)

    model = PPO(
    CustomSeparatePolicy,
    SteppingRobotEnv,
    policy_kwargs=separate_policy_kwargs(obs_seq_len),
    verbose=1,
    n_steps=3000, # 3000 dla control_step=2, 2000 dla control_step=4
    batch_size=500, # 500 dla control_step=2, 300 dla control_step=4
    learning_rate=0.00025
    )
    print(model.policy)

    return training_loop(model, SteppingRobotEnv, training_nickname, control_step, model_name)


def train_A2C():
    control_step = 2
    obs_seq_len = 2
    training_nickname = "A2C_v8"
    model_name = 'A2C'

    SteppingRobotEnv = make_vec_env(training_nickname, control_step, obs_seq_len)

    model = A2C(
        CustomSeparatePolicy,
        SteppingRobotEnv,
        policy_kwargs=separate_policy_kwargs(obs_seq_len),
        verbose=1,
        n_steps=250,
        learning_rate=1e-4,
        gamma=0.99,
        gae_lambda = 0.95,
        ent_coef = 0.015,
        vf_coef=0.7,
        max_grad_norm=0.3
    )
    print(model.policy)

    return training_loop(model, SteppingRobotEnv, training_nickname, control_step, model_name)


def train_SAC():
    control_step = 2
    obs_seq_len = 2
    training_nickname = "SAC_v6"
    model_name = 'SAC'

    SteppingRobotEnv = make_vec_env(training_nickname, control_step, obs_seq_len, 0.5)

    policy_kwargs = dict(
        activation_fn=nn.ReLU,
        net_arch=dict( pi=[128, 64], qf=[128, 64] )
    )

    model = SAC(
        "MlpPolicy",
        SteppingRobotEnv,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        buffer_size=1000000,
        learning_starts=10000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=2,
        ent_coef="auto",
        verbose=0
    )
    print(model.policy)

    return training_loop_SAC(model, SteppingRobotEnv, training_nickname, control_step, model_name)


def log_training_start(mean_reward, std_reward):
    with open(EPOCHS_LOG_PATH, "a") as f:
        f.write("="*50 + "\n")
        f.write(f"Nauka {str(date.today())}\n")
        f.write(f"Start: {mean_reward:.2f} +/- {std_reward:.2f}\n")


def training_loop(model, SteppingRobotEnv, training_nickname, control_step, model_name):
    os.makedirs(TRAINED_MODELS_DIR, exist_ok=True)
    mean_reward_table = []
    std_reward_table = []

    mean_reward, std_reward = evaluate_policy(model, SteppingRobotEnv, n_eval_episodes=10, deterministic=True)

    print(f'Starting mean reward: {np.floor(mean_reward)} +/- {std_reward}')
    mean_reward_table.append(mean_reward)
    std_reward_table.append(std_reward)

    max_reward = np.floor(mean_reward)
    log_training_start(mean_reward, std_reward)

    for i in range(150):
        model.learn(total_timesteps=30000, progress_bar=True) # 30000 dla control_step=2, 20000 dla control_step=4
        mean_reward, std_reward = evaluate_policy(model, SteppingRobotEnv, n_eval_episodes=10, deterministic=True)

        mean_reward_table.append(mean_reward)
        std_reward_table.append(std_reward)
        with open(EPOCHS_LOG_PATH, "a") as f:
            f.write(f"Ep{i+1}: {mean_reward:.2f} +/- {std_reward:.2f}\n")
        print(f'Epoch {i+1} - Mean reward: {mean_reward} +/- {std_reward}')

        is_new_max = mean_reward > max_reward
        if is_new_max:
            max_reward = mean_reward
            print(f"New max reward: {max_reward}")

        # zapis co 10 epok oraz przy każdym nowym maksimum
        if (i+1) % 10 == 0 or is_new_max:
            model.save(TRAINED_MODELS_DIR / f"{model_name}_SteppingRobotEnv_{training_nickname}_continous_step{control_step}_separate_obs_ep{i+1}_mean_{int(mean_reward)}_std_{int(std_reward)}.zip")

    SteppingRobotEnv.close()

    return mean_reward_table, std_reward_table


def training_loop_SAC(model, SteppingRobotEnv, training_nickname, control_step, model_name):
    os.makedirs(TRAINED_MODELS_DIR, exist_ok=True)
    mean_reward_table = []
    std_reward_table = []

    mean_reward, std_reward = evaluate_policy(model, SteppingRobotEnv, n_eval_episodes=10, deterministic=True)

    print(f'Starting mean reward: {np.floor(mean_reward)} +/- {std_reward}')
    mean_reward_table.append(mean_reward)
    std_reward_table.append(std_reward)

    max_reward = np.floor(mean_reward)
    log_training_start(mean_reward, std_reward)

    total_timesteps = 3_000_000
    eval_freq = 100_000

    for step in range(0, total_timesteps, eval_freq):
        model.learn(
            total_timesteps=eval_freq,
            reset_num_timesteps=False,
            progress_bar=True
        )

        mean_reward, std_reward = evaluate_policy(
            model,
            SteppingRobotEnv,
            n_eval_episodes=10,
            deterministic=True
        )

        mean_reward_table.append(mean_reward)
        std_reward_table.append(std_reward)

        print(f"Step {step + eval_freq} - {mean_reward:.2f} +/- {std_reward:.2f}")

        with open(EPOCHS_LOG_PATH, "a") as f:
            f.write(f"Step {step + eval_freq}: {mean_reward:.2f} +/- {std_reward:.2f}\n")

        if mean_reward > max_reward:
            max_reward = mean_reward

            model.save(TRAINED_MODELS_DIR / f"{training_nickname}_step{step + eval_freq}_mean_{int(mean_reward)}.zip")

    SteppingRobotEnv.close()

    return mean_reward_table, std_reward_table


def visualize_policy_network(model, output_path="actor_network.png", max_neurons_per_layer=20):

    actor = ExportableActor(model.policy.cpu()).eval()

    weights = [p.detach().cpu().numpy() for n, p in actor.named_parameters() if n.endswith(".weight")]

    layer_sizes = [weights[0].shape[1]] + [W.shape[0] for W in weights]
    display_sizes = [min(s, max_neurons_per_layer) for s in layer_sizes]

    _, ax = plt.subplots(figsize=(2 + 2.2 * len(layer_sizes), 7))
    x_positions = np.linspace(0, 1, len(layer_sizes))

    hidden_half = 0.45
    edge_half = 0.32

    node_coords = []
    for li, (n_real, n_show) in enumerate(zip(layer_sizes, display_sizes)):
        x = x_positions[li]
        is_edge = (li == 0 or li == len(layer_sizes) - 1)
        half_height = edge_half if is_edge else hidden_half
        y_top, y_bot = 0.5 + half_height, 0.5 - half_height

        if n_real > n_show:
            n_top = n_show // 2
            gap = 0.05 * (half_height / hidden_half) + 0.05
            top_ys = np.linspace(y_top, 0.5 + gap, n_top)
            bot_ys = np.linspace(0.5 - gap, y_bot, n_show - n_top)
            ys = np.concatenate([top_ys, bot_ys])
            for dy in (-0.02, 0.0, 0.02):
                ax.scatter(x, 0.5 + dy, s=25, color="#555555", zorder=4)
        else:
            ys = np.linspace(y_top, y_bot, n_show) if n_show > 1 else np.array([0.5])

        coords = [(x, y) for y in ys]
        node_coords.append(coords)
        ax.scatter([x] * n_show, ys, s=120, color="#1f77b4", zorder=3)

    for li, W in enumerate(weights):
        src = node_coords[li]
        dst = node_coords[li + 1]
        vmax = np.max(np.abs(W)) + 1e-9
        for i, (x1, y1) in enumerate(src):
            for j, (x2, y2) in enumerate(dst):
                w = W[j % W.shape[0], i % W.shape[1]]
                color = "tab:red" if w < 0 else "tab:green"
                ax.plot([x1, x2], [y1, y2], color=color,
                        alpha=min(0.9, abs(w) / vmax),
                        linewidth=0.3 + 1.6 * abs(w) / vmax, zorder=1)

    for li in range(1, len(layer_sizes) - 1):
        x_mid = (x_positions[li] + x_positions[li + 1]) / 2
        ax.text(x_mid, 1.06, "ReLU", ha="center", fontsize=9, color="darkorange",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#fff3e0",
                          edgecolor="darkorange", linewidth=1))

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.08, 1.13)
    ax.axis("off")
    plt.savefig(output_path, dpi=150)
    plt.tight_layout()
    plt.show()

    model.policy.to(model.device)


def test_NN_functioning(model_path):
    obs_seq_len = 2
    model = PPO.load(model_path)

    actor_only_net = ExportableActor(model.policy.cpu()).eval()

    observation_deque = deque(maxlen=obs_seq_len)
    for _ in range(obs_seq_len):
        observation_deque.append([0.29]*4+[0.33]*4)

    for i in range(20):
        nn_input = []
        for j in range(obs_seq_len):
            nn_input += observation_deque[j]
            nn_input += [0.0, 0.0]

        obs = torch.Tensor([nn_input])
        out = actor_only_net(obs)

        action = out.detach().cpu().numpy()[0]
        action = np.clip(action, -1.0, 1.0)
        print("==========================")
        print(f"Manual step {i}: Action: \n{np.array2string(action, precision=2, suppress_small=True)}")
        observation_deque.append(action.tolist())

        obs_model = np.array(nn_input*2 + [0]*8)
        action, _ = model.predict(obs_model, deterministic=True)
        print(f"Automatic step {i}: Action: \n{np.array2string(action, precision=2, suppress_small=True)}")


ALGORITHMS = {"ppo": PPO, "a2c": A2C, "sac": SAC}


def evaluate_model(algo, model_path, n_eval_episodes=10):
    control_step = 2
    obs_seq_len = 2
    SteppingRobotEnv = CoppeliaSteppingRobotGymEnv(control_step=control_step, zmq_port=PORTS[0], observation_seq_len=obs_seq_len, starting_curriculum_weight=0.8, save_stats=True)

    model = ALGORITHMS[algo].load(model_path)
    print(f"Loaded model from: {model_path}")

    # PPO i A2C oceniane z polityką stochastyczną, SAC - deterministyczną
    mean_reward, std_reward = evaluate_policy(model, SteppingRobotEnv, n_eval_episodes=n_eval_episodes, deterministic=(algo == "sac"))
    print(f'Test mean reward: {mean_reward} +/- {std_reward}')

    SteppingRobotEnv.calculate_avg_stats()
    SteppingRobotEnv.plot_torques()

    return mean_reward, std_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or evaluate the stepping robot policy.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--algo", choices=ALGORITHMS.keys(), default="ppo")

    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("--algo", choices=ALGORITHMS.keys(), default="ppo")
    eval_parser.add_argument("--model", required=True, help="path to a saved .zip model")
    eval_parser.add_argument("--episodes", type=int, default=10)

    args = parser.parse_args()

    if args.command == "train":
        {"ppo": train_PPO, "a2c": train_A2C, "sac": train_SAC}[args.algo]()
    else:
        evaluate_model(args.algo, args.model, args.episodes)
