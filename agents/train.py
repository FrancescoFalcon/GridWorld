"""Training pipeline for the GridWorld project."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gridworld import utils
from gridworld.env import GridWorldConfig, GridWorldEnv
from gridworld.level_generator import (
    DEFAULT_LEVEL_METADATA,
    build_default_level_pack,
    generate_level,
)

OUTPUT_DIR = Path("output")
MODELS_DIR = OUTPUT_DIR / "models"
LOGS_DIR = OUTPUT_DIR / "logs"
GIF_DIR = OUTPUT_DIR / "gifs"
PLOTS_DIR = OUTPUT_DIR / "plots"
TRAJ_DIR = OUTPUT_DIR / "trajectories"
LEVEL_DIR = Path("levels")
MAX_OBS_GRID = max(meta.grid_size for meta in DEFAULT_LEVEL_METADATA)


class GridWorldCNN(BaseFeaturesExtractor):
    """
    Custom CNN for GridWorld.
    Input shape: (6, H, W)
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


class RewardLoggerCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.record: List[Dict[str, float]] = []
        self.current_reward = 0.0

    def _on_step(self) -> bool:
        reward = float(self.locals["rewards"][0])
        self.current_reward += reward
        if self.locals["dones"][0]:
            self.record.append({"timesteps": self.num_timesteps, "reward": self.current_reward})
            self.current_reward = 0.0
        return True


class ProceduralWrapper(gym.Wrapper):
    def __init__(self, env: GridWorldEnv, min_diff: int, max_diff: int, seed: Optional[int] = None):
        super().__init__(env)
        self.min_diff = min_diff
        self.max_diff = max_diff
        self.rng = np.random.default_rng(seed)

    def reset(self, **kwargs):
        difficulty = int(self.rng.integers(self.min_diff, self.max_diff + 1))
        level_cfg = generate_level(difficulty, seed=int(self.rng.integers(0, 1_000_000)))
        return self.env.reset(options={"config": level_cfg}, **kwargs)


class RandomFixedLevelWrapper(gym.Wrapper):
    def __init__(self, env: GridWorldEnv, level_dir: Path, levels: List[int], seed: Optional[int] = None):
        super().__init__(env)
        self.level_dir = level_dir
        self.levels = levels
        self.rng = np.random.default_rng(seed)

    def reset(self, **kwargs):
        level_idx = self.rng.choice(self.levels)
        level_path = self.level_dir / f"level_{level_idx}.json"
        level_cfg = utils.load_level_from_json(str(level_path))
        return self.env.reset(options={"config": level_cfg}, **kwargs)


class MixedWrapper(gym.Wrapper):
    def __init__(self, env: GridWorldEnv, level_dir: Path, min_diff: int, max_diff: int, seed: Optional[int] = None, procedural_prob: float = 0.8):
        super().__init__(env)
        self.level_dir = level_dir
        self.min_diff = min_diff
        self.max_diff = max_diff
        self.rng = np.random.default_rng(seed)
        self.fixed_levels = [1, 2, 3, 4, 5]
        self.procedural_prob = procedural_prob

    def reset(self, **kwargs):
        # Use procedural_prob to decide between procedural and fixed
        if self.rng.random() < self.procedural_prob:
            # Procedural
            difficulty = int(self.rng.integers(self.min_diff, self.max_diff + 1))
            level_cfg = generate_level(difficulty, seed=int(self.rng.integers(0, 1_000_000)))
        else:
            # Fixed Suite
            level_idx = self.rng.choice(self.fixed_levels)
            level_path = self.level_dir / f"level_{level_idx}.json"
            level_cfg = utils.load_level_from_json(str(level_path))
        
        return self.env.reset(options={"config": level_cfg}, **kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training GridWorld")
    parser.add_argument("--algo", choices=["dqn", "ppo"], default="dqn")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid-size", type=int, default=None)
    parser.add_argument("--difficulty", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--train-on-procedural", action="store_true")
    parser.add_argument("--train-on-suite", action="store_true", help="Train on the fixed suite of levels 1-5")
    parser.add_argument("--train-mixed", action="store_true", help="Train on both procedural and fixed suite (50/50)")
    parser.add_argument("--config", type=str, help="Percorso a livello custom")
    parser.add_argument("--load-model", type=str, help="Percorso del modello da cui riprendere il training")
    parser.add_argument("--tensorboard", action="store_true")
    return parser.parse_args()


def ensure_assets() -> None:
    for path in [MODELS_DIR, LOGS_DIR, GIF_DIR, PLOTS_DIR, TRAJ_DIR]:
        path.mkdir(parents=True, exist_ok=True)
    build_default_level_pack(str(LEVEL_DIR))


def load_level_config(args: argparse.Namespace) -> Dict:
    if args.config:
        return utils.load_level_from_json(args.config)
    if args.train_on_procedural:
        return generate_level(args.difficulty, seed=args.seed)
    # Default fallback
    level_path = LEVEL_DIR / f"level_{args.difficulty}.json"
    if not level_path.exists():
        # If level doesn't exist yet, generate a dummy one just for init
        return generate_level(1, seed=42)
    return utils.load_level_from_json(str(level_path))


def make_env(args: argparse.Namespace, base_config: Dict) -> DummyVecEnv:
    obs_grid = max(base_config.get("grid_size", MAX_OBS_GRID), MAX_OBS_GRID)

    def _init():
        env = GridWorldEnv(GridWorldConfig.from_dict(base_config), obs_grid_size=obs_grid)
        if args.train_mixed:
            env = MixedWrapper(env, LEVEL_DIR, 1, 5, seed=args.seed, procedural_prob=0.8)
        elif args.train_on_procedural:
            # Genera livelli casuali da difficoltà 1 a 5 per massima generalizzazione
            env = ProceduralWrapper(env, 1, 5, seed=args.seed)
        elif args.train_on_suite:
            env = RandomFixedLevelWrapper(env, LEVEL_DIR, [1, 2, 3, 4, 5], seed=args.seed)
        return env

    return DummyVecEnv([_init])


def create_model(algo: str, env: DummyVecEnv, tensorboard: bool, seed: int, load_path: Optional[str] = None, is_procedural: bool = False, is_mixed: bool = False) -> BaseAlgorithm:
    device = "cuda" if th.cuda.is_available() else "cpu"
    print(f"[Train] Using {device.upper()} for policy updates")
    
    if load_path:
        print(f"[Train] Loading model from {load_path}")
        if algo == "dqn":
            model = DQN.load(load_path, env=env, device=device)
        else:
            model = PPO.load(load_path, env=env, device=device)
        
        if algo == "dqn":
            if is_procedural or is_mixed:
                # CONSOLIDATION PHASE: Moderate exploration to regain generalization
                model.exploration_initial_eps = 0.15
                model.exploration_final_eps = 0.02
                model.exploration_fraction = 0.2
                print("[Train] Adjusted exploration MODERATE for generalization recovery")
            else:
                # Lower exploration for fine-tuning on fixed suite
                model.exploration_initial_eps = 0.2
                model.exploration_final_eps = 0.02
                model.exploration_fraction = 0.2
                print("[Train] Adjusted exploration LOW for fine-tuning")
            
            # Force update of exploration schedule
            model.exploration_schedule = get_linear_fn(
                model.exploration_initial_eps,
                model.exploration_final_eps,
                model.exploration_fraction,
            )
        return model

    common_kwargs = {"seed": seed, "verbose": 1, "device": device}
    tb_log = str(LOGS_DIR / "tensorboard") if tensorboard else None
    
    policy_kwargs = dict(
        features_extractor_class=GridWorldCNN,
        features_extractor_kwargs=dict(features_dim=512),
    )

    if algo == "dqn":
        model = DQN(
            "CnnPolicy",
            env,
            learning_rate=1e-4,  # Increased learning rate
            buffer_size=500_000,
            exploration_fraction=0.9, # Explore for 90% of training
            exploration_final_eps=0.05, # Slightly higher final epsilon
            batch_size=128, # Increased batch size
            tau=1.0,
            target_update_interval=1000, # More frequent updates
            train_freq=4,
            gradient_steps=1,
            tensorboard_log=tb_log,
            policy_kwargs=policy_kwargs,
            **common_kwargs,
        )
    else:
        model = PPO(
            "CnnPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gae_lambda=0.95,
            clip_range=0.2,
            tensorboard_log=tb_log,
            policy_kwargs=policy_kwargs,
            **common_kwargs,
        )
    return model


def save_metadata(model_path: Path, algo: str) -> None:
    meta_path = Path(f"{model_path}.meta.json")
    with open(meta_path, "w", encoding="utf-8") as fp:
        json.dump({"algo": algo}, fp, indent=2)


def export_training_logs(records: List[Dict[str, float]]) -> Path:
    df = pd.DataFrame(records)
    csv_path = LOGS_DIR / "training_logs.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def plot_reward(records: List[Dict[str, float]]) -> Path:
    if not records:
        raise ValueError("Nessun record di reward")
    timesteps = [row["timesteps"] for row in records]
    rewards = [row["reward"] for row in records]
    output = PLOTS_DIR / "reward_curve.png"
    utils.plot_reward_curve(timesteps, rewards, str(output))
    return output


def capture_heatmap(model: BaseAlgorithm) -> np.ndarray:
    level3 = LEVEL_DIR / "level_3.json"
    env = GridWorldEnv(
        GridWorldConfig.from_dict(utils.load_level_from_json(str(level3))),
        obs_grid_size=MAX_OBS_GRID,
    )
    obs, _ = env.reset()
    done = False
    truncated = False
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, truncated, _ = env.step(int(action))
    visitation = env.visitation.astype(float)
    if visitation.max() > 0:
        visitation /= visitation.max()
    return visitation


def build_effectiveness_plot(
    reward_history: List[Dict[str, float]],
    suite_success: Dict[int, float],
    heatmap: np.ndarray,
) -> Path:
    output = PLOTS_DIR / "model_effectiveness.png"
    timesteps = [row["timesteps"] for row in reward_history]
    rewards = [row["reward"] for row in reward_history]
    levels = sorted(suite_success.keys())
    success_values = [suite_success.get(lvl, 0.0) for lvl in levels]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].plot(timesteps, rewards, color="tab:blue")
    axes[0, 0].set_title("Reward per episodio")
    axes[0, 0].set_xlabel("Timesteps")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(levels, success_values, marker="o", color="tab:green")
    axes[0, 1].set_title("Success rate suite")
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_xlabel("Livello")
    axes[0, 1].set_ylabel("Success rate")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].bar(levels, success_values, color="tab:orange")
    axes[1, 0].set_title("Confronto livello")
    axes[1, 0].set_xlabel("Livello")
    axes[1, 0].set_ylabel("Success rate")
    axes[1, 0].set_ylim(0, 1)

    im = axes[1, 1].imshow(heatmap, cmap="magma")
    axes[1, 1].set_title("Heatmap visite livello 3")
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)
    for ax in axes.flat:
        ax.label_outer()
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return output


def auto_generate_gif(model: BaseAlgorithm, algo: str) -> Optional[str]:
    from agents import evaluate as eval_mod

    hardest_level = LEVEL_DIR / "level_5.json"
    env = eval_mod.make_env(str(hardest_level))
    gif_path = GIF_DIR / "best_episode.gif"
    try:
        eval_mod.save_replay_gif(model, env, str(gif_path))
        trajectory = eval_mod.play_episode(model, env, deterministic=True, capture_frames=False)[
            "trajectory"
        ]
        env.save_trajectory_png(trajectory, str(TRAJ_DIR / "best_episode.png"))
        return str(gif_path)
    except Exception:
        return None


def main() -> None:
    args = parse_args()
    ensure_assets()
    base_config = load_level_config(args)
    env = make_env(args, base_config)
    callback = RewardLoggerCallback()
    model = create_model(args.algo, env, args.tensorboard, args.seed, args.load_model, is_procedural=args.train_on_procedural, is_mixed=args.train_mixed)
    
    # If fine-tuning, maybe we want to reset the exploration schedule?
    # For DQN, exploration_schedule is internal. 
    # If we load a fully trained model, exploration might be low.
    # Let's ensure we have some exploration for fine-tuning if it's DQN.
    if args.load_model and args.algo == "dqn" and not args.train_on_procedural and not args.train_mixed:
        # This block is now handled inside create_model, but keeping a safety check or logging is fine.
        pass

    model.learn(total_timesteps=args.timesteps, callback=callback)

    model_path = MODELS_DIR / f"{args.algo}_final.zip"
    model.save(str(model_path))
    save_metadata(model_path, args.algo)

    export_training_logs(callback.record)
    plot_reward(callback.record)

    from agents import evaluate as eval_mod

    suite = eval_mod.run_test_suite(str(model_path), algo=args.algo)
    heatmap = capture_heatmap(model)
    build_effectiveness_plot(callback.record, {int(k): v for k, v in suite["success_rates"].items()}, heatmap)
    auto_generate_gif(model, args.algo)

    print("Training completato. Modello salvato in", model_path)


if __name__ == "__main__":
    main()
