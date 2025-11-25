"""Evaluation utilities for the GridWorld RL project."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.base_class import BaseAlgorithm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gridworld.env import GridWorldConfig, GridWorldEnv
from gridworld.level_generator import DEFAULT_LEVEL_METADATA, build_default_level_pack
from gridworld import utils

LEVEL_DIR = Path("levels")
OUTPUT_DIR = Path("output")
REPORT_TXT = OUTPUT_DIR / "effectiveness_report.txt"
REPORT_CSV = OUTPUT_DIR / "test_suite_report.csv"
COMPARE_CSV = OUTPUT_DIR / "model_comparison.csv"
COMPARE_PNG = OUTPUT_DIR / "model_comparison.png"
GIF_DIR = OUTPUT_DIR / "gifs"
PLOTS_DIR = OUTPUT_DIR / "plots"
MAX_OBS_GRID = max(meta.grid_size for meta in DEFAULT_LEVEL_METADATA)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def ensure_levels() -> None:
    build_default_level_pack(str(LEVEL_DIR))


def detect_algo(model_path: str, requested: Optional[str]) -> str:
    if requested:
        return requested.lower()
    meta_path = Path(f"{model_path}.meta.json")
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as fp:
            meta = json.load(fp)
        return meta.get("algo", "dqn")
    name = Path(model_path).name.lower()
    if "ppo" in name:
        return "ppo"
    return "dqn"


def load_model(model_path: str, algo: Optional[str] = None) -> Tuple[BaseAlgorithm, str]:
    algorithm = detect_algo(model_path, algo)
    if algorithm == "ppo":
        model = PPO.load(model_path)
    elif algorithm == "dqn":
        model = DQN.load(model_path)
    else:
        raise ValueError(f"Algoritmo non supportato: {algorithm}")
    return model, algorithm


def make_env(level_path: str) -> GridWorldEnv:
    config = utils.load_level_from_json(level_path)
    grid_config = GridWorldConfig.from_dict(config)
    obs_grid = max(grid_config.grid_size, MAX_OBS_GRID)
    return GridWorldEnv(grid_config, obs_grid_size=obs_grid)


# ---------------------------------------------------------------------------
# Evaluation routines
# ---------------------------------------------------------------------------


def play_episode(
    model: BaseAlgorithm,
    env: GridWorldEnv,
    deterministic: bool = True,
    capture_frames: bool = False,
) -> Dict:
    obs, _ = env.reset()
    done = False
    truncated = False
    reward_sum = 0.0
    steps = 0
    steps_to_key = None
    steps_to_goal = None
    door_open_step = None
    failure_reason = "max_steps"
    frames: List[np.ndarray] = []
    trajectory: List[Tuple[int, int]] = [env.agent_pos]
    action_counts = {}

    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=deterministic)
        act_int = int(action)
        action_counts[act_int] = action_counts.get(act_int, 0) + 1
        
        obs, reward, done, truncated, info = env.step(act_int)
        reward_sum += reward
        steps += 1
        if info.get("has_key") and steps_to_key is None:
            steps_to_key = steps
        if info.get("door_open") and door_open_step is None:
            door_open_step = steps
        if done:
            steps_to_goal = steps
            failure_reason = "success"
        elif truncated:
            failure_reason = "timeout"
        if capture_frames:
            frames.append(_grid_to_image(env._symbolic_grid(), step=steps))
        trajectory.append(env.agent_pos)
    
    # Optional: print action distribution for debugging
    # print(f"Actions: {action_counts}")
    
    return {
        "success": done,
        "reward": reward_sum,
        "steps": steps,
        "steps_to_key": steps_to_key,
        "steps_key_to_goal": steps_to_goal - steps_to_key if (steps_to_goal and steps_to_key) else None,
        "door_open_step": door_open_step,
        "failure_reason": failure_reason,
        "trajectory": trajectory,
        "frames": frames,
    }


def evaluate_model(
    model: BaseAlgorithm,
    env: GridWorldEnv,
    episodes: int = 5,
    deterministic: bool = True,
) -> Dict:
    episode_stats = []
    for _ in range(episodes):
        result = play_episode(model, env, deterministic=deterministic, capture_frames=False)
        episode_stats.append(result)
    success_rate = np.mean([1 if r["success"] else 0 for r in episode_stats])
    mean_reward = np.mean([r["reward"] for r in episode_stats])
    std_reward = np.std([r["reward"] for r in episode_stats])
    return {
        "success_rate": float(success_rate),
        "mean_reward": float(mean_reward),
        "std_reward": float(std_reward),
        "episodes": episode_stats,
    }


def evaluate_and_print_trajectory(
    model: BaseAlgorithm,
    env: GridWorldEnv,
    episodes: int = 5,
    deterministic: bool = True,
) -> None:
    for ep in range(episodes):
        print(f"\nEpisodio {ep + 1}")
        obs, _ = env.reset()
        done = False
        truncated = False
        steps = 0
        while not (done or truncated):
            env.render()
            action, _states = model.predict(obs, deterministic=deterministic)
            obs, reward, done, truncated, info = env.step(int(action))
            steps += 1
        env.render()
        print(f"Terminato in {steps} step | Successo: {done}")


def save_replay_gif(
    model: BaseAlgorithm,
    env: GridWorldEnv,
    output_path: str,
    deterministic: bool = True,
) -> str:
    result = play_episode(model, env, deterministic=deterministic, capture_frames=True)
    frames = result["frames"]
    if not frames:
        raise RuntimeError("Nessun frame catturato per la GIF")
    utils.ensure_dir(Path(output_path).parent)
    imageio.mimsave(output_path, frames, duration=0.4)
    return output_path


def run_custom_suite(
    model_path: str,
    level_files: List[Path],
    algo: Optional[str] = None,
    num_episodes: int = 10,
    deterministic: bool = True,
) -> Dict:
    model, algorithm = load_model(model_path, algo)
    all_records: List[Dict] = []
    
    print(f"Valutazione su {len(level_files)} livelli custom...")
    for lvl_path in level_files:
        try:
            env = make_env(str(lvl_path))
            # Extract difficulty/name from filename or json if possible, else use filename
            lvl_name = lvl_path.stem
            
            for episode in range(num_episodes):
                result = play_episode(model, env, deterministic=deterministic, capture_frames=False)
                all_records.append(
                    {
                        "level": lvl_name,
                        "episode": episode,
                        "success": result["success"],
                        "reward": result["reward"],
                        "steps": result["steps"],
                        "steps_to_key": result["steps_to_key"],
                        "steps_key_to_goal": result["steps_key_to_goal"],
                        "failure_reason": result["failure_reason"],
                    }
                )
        except Exception as e:
            print(f"Errore nel valutare {lvl_path}: {e}")

    df = pd.DataFrame(all_records)
    utils.ensure_dir(OUTPUT_DIR)
    report_csv = OUTPUT_DIR / "custom_suite_report.csv"
    df.to_csv(report_csv, index=False)
    
    # Simple summary
    success_rate = df["success"].mean()
    print(f"Successo complessivo suite custom: {success_rate:.2f}")
    
    return {
        "records": df,
        "success_rate": success_rate,
        "report_csv": str(report_csv),
    }


def run_test_suite(
    model_path: str,
    algo: Optional[str] = None,
    num_episodes: int = 20,
    deterministic: bool = True,
) -> Dict:
    ensure_levels()
    model, algorithm = load_model(model_path, algo)
    all_records: List[Dict] = []
    for meta in DEFAULT_LEVEL_METADATA:
        level_path = LEVEL_DIR / f"level_{meta.difficulty}.json"
        env = make_env(str(level_path))
        for episode in range(num_episodes):
            result = play_episode(model, env, deterministic=deterministic, capture_frames=False)
            all_records.append(
                {
                    "level": meta.difficulty,
                    "episode": episode,
                    "success": result["success"],
                    "reward": result["reward"],
                    "steps": result["steps"],
                    "steps_to_key": result["steps_to_key"],
                    "steps_key_to_goal": result["steps_key_to_goal"],
                    "failure_reason": result["failure_reason"],
                }
            )
    df = pd.DataFrame(all_records)
    utils.ensure_dir(OUTPUT_DIR)
    df.to_csv(REPORT_CSV, index=False)
    summary_lines = _summarize_suite(df)
    utils.write_report(REPORT_TXT, summary_lines)
    success_rates = df.groupby("level")["success"].mean().to_dict()
    return {
        "records": df,
        "success_rates": success_rates,
        "report_csv": str(REPORT_CSV),
        "report_txt": str(REPORT_TXT),
        "algo": algorithm,
    }


def compare_models(model_paths: Sequence[str], algo: Optional[str] = None) -> str:
    results = []
    for path in model_paths:
        suite = run_test_suite(path, algo=algo)
        rates = suite["success_rates"]
        results.append({"model": Path(path).name, **{f"level_{k}": v for k, v in rates.items()}})
    df = pd.DataFrame(results)
    df.to_csv(COMPARE_CSV, index=False)
    utils.ensure_dir(PLOTS_DIR)
    plt.figure(figsize=(8, 5))
    for _, row in df.iterrows():
        levels = sorted(int(col.split("_")[1]) for col in row.index if col.startswith("level_"))
        values = [row[f"level_{lvl}"] for lvl in levels]
        plt.plot(levels, values, marker="o", label=row["model"])
    plt.xticks(levels)
    plt.ylim(0, 1)
    plt.xlabel("Livello")
    plt.ylabel("Success rate")
    plt.title("Comparazione modelli")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(COMPARE_PNG)
    plt.close()
    return str(COMPARE_CSV)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _grid_to_image(grid: List[List[str]], step: int) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(3, 3))
    table = ax.table(cellText=grid, loc="center", cellLoc="center")
    table.scale(1, 2)
    ax.axis("off")
    ax.set_title(f"Step {step}")
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    buffer = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    image = buffer.reshape((height, width, 4))[:, :, :3]
    plt.close(fig)
    return image


def _summarize_suite(df: pd.DataFrame) -> List[str]:
    lines = ["=== Test Suite GridWorld ==="]
    failure_counts = df["failure_reason"].value_counts().to_dict()
    for level, group in df.groupby("level"):
        success = group["success"].mean()
        mean_steps = group["steps"].mean()
        mean_reward = group["reward"].mean()
        key_time = group["steps_to_key"].dropna().mean()
        bridge_time = group["steps_key_to_goal"].dropna().mean()
        lines.append(
            f"Livello {level}: success={success:.2f}, steps={mean_steps:.1f}, reward={mean_reward:.2f}, key={key_time:.1f}, key->goal={bridge_time:.1f}"
        )
    overall = df["success"].mean()
    lines.append(f"Successo complessivo: {overall:.2f}")
    lines.append(
        f"Reward medio globale: {df['reward'].mean():.2f} +/- {df['reward'].std():.2f}"
    )
    lines.append(
        f"Passi medi per la chiave: {df['steps_to_key'].dropna().mean():.1f}"
    )
    lines.append(
        f"Passi medi chiave->porta->goal: {df['steps_key_to_goal'].dropna().mean():.1f}"
    )
    lines.append("Note fallimenti: " + ", ".join(f"{k}={v}" for k, v in failure_counts.items()))
    return lines


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Valutazione GridWorld")
    parser.add_argument("--model_path", type=str, help="Percorso del modello SB3", nargs="?")
    parser.add_argument("--algo", type=str, choices=["dqn", "ppo"], help="Algoritmo del modello", nargs="?")
    parser.add_argument("--level", type=int, choices=range(1, 6), help="Livello singolo da valutare")
    parser.add_argument("--level_file", type=str, help="Percorso file JSON livello specifico")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--run_suite", action="store_true")
    parser.add_argument("--test_folder", type=str, help="Cartella con livelli JSON per test custom")
    parser.add_argument("--compare_models", nargs="*", help="Confronta modelli molteplici")
    parser.add_argument("--save_gif", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_levels()

    if args.compare_models:
        compare_models(args.compare_models, algo=args.algo)
        return

    if not args.model_path:
        raise SystemExit("Serve --model_path o --compare_models")

    model, algo = load_model(args.model_path, args.algo)

    if args.run_suite:
        run_test_suite(args.model_path, algo=algo)

    if args.test_folder:
        folder = Path(args.test_folder)
        if folder.exists() and folder.is_dir():
            levels = sorted(list(folder.glob("*.json")))
            run_custom_suite(args.model_path, levels, algo=algo, num_episodes=args.episodes, deterministic=args.deterministic)
        else:
            print(f"Cartella non trovata: {args.test_folder}")
        return

    if args.level_file:
        level_path = Path(args.level_file)
    elif args.level:
        level_path = LEVEL_DIR / f"level_{args.level}.json"
    else:
        level_path = LEVEL_DIR / "level_1.json"

    env = make_env(str(level_path))
    stats = evaluate_model(model, env, episodes=args.episodes, deterministic=args.deterministic)
    print(json.dumps(stats, indent=2))

    if args.save_gif:
        gif_path = GIF_DIR / "best_episode.gif"
        save_replay_gif(model, env, str(gif_path))


if __name__ == "__main__":
    main()
