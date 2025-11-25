"""Custom GridWorld environment with key-door-goal mechanics."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from . import utils

Cell = Tuple[int, int]


@dataclass
class GridWorldConfig:
    grid_size: int = 7
    start: Cell = (0, 0)
    key: Cell = (0, 0)
    door: Cell = (0, 0)
    goal: Cell = (0, 0)
    obstacles: List[Cell] = field(default_factory=list)
    risk_zones: List[Cell] = field(default_factory=list)
    max_steps: Optional[int] = None
    door_penalty: float = -1.0
    risk_penalty: float = -3.0
    step_penalty: float = -0.1
    wall_penalty: float = -0.5
    repetition_penalty: float = -0.2
    exploration_bonus: float = 0.1
    key_reward: float = 3.0
    goal_reward: float = 20.0

    @classmethod
    def from_dict(cls, data: Dict) -> "GridWorldConfig":
        return cls(
            grid_size=data["grid_size"],
            start=tuple(data["start"]),
            key=tuple(data["key"]),
            door=tuple(data["door"]),
            goal=tuple(data["goal"]),
            obstacles=[tuple(cell) for cell in data.get("obstacles", [])],
            risk_zones=[tuple(cell) for cell in data.get("risk_zones", [])],
            max_steps=data.get("max_steps"),
            door_penalty=data.get("door_penalty", -1.0),
            risk_penalty=data.get("risk_penalty", -3.0),
            step_penalty=data.get("step_penalty", -0.15),
            wall_penalty=data.get("wall_penalty", -0.3),
            repetition_penalty=data.get("repetition_penalty", -0.2),
            exploration_bonus=data.get("exploration_bonus", 0.05),
            key_reward=data.get("key_reward", 3.0),
            goal_reward=data.get("goal_reward", 20.0),
        )

    def to_dict(self) -> Dict:
        return {
            "grid_size": self.grid_size,
            "start": list(self.start),
            "key": list(self.key),
            "door": list(self.door),
            "goal": list(self.goal),
            "obstacles": [list(cell) for cell in self.obstacles],
            "risk_zones": [list(cell) for cell in self.risk_zones],
            "max_steps": self.max_steps,
            "door_penalty": self.door_penalty,
            "risk_penalty": self.risk_penalty,
            "step_penalty": self.step_penalty,
            "wall_penalty": self.wall_penalty,
            "repetition_penalty": self.repetition_penalty,
            "exploration_bonus": self.exploration_bonus,
            "key_reward": self.key_reward,
            "goal_reward": self.goal_reward,
        }


class GridWorldEnv(gym.Env):
    """GridWorld with a locked door that requires a key."""

    metadata = {"render.modes": ["human", "ansi"], "render_fps": 10}

    def __init__(
        self,
        config: Optional[GridWorldConfig] = None,
        level_path: Optional[str] = None,
        seed: Optional[int] = None,
        obs_grid_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        if config is None:
            if level_path is None:
                raise ValueError("Provide either config or level_path")
            config = GridWorldConfig.from_dict(utils.load_level_from_json(level_path))
        self.config = config
        self.grid_size = config.grid_size
        self.obs_grid_size = obs_grid_size or config.grid_size
        if self.obs_grid_size < self.grid_size:
            raise ValueError("obs_grid_size must be >= config.grid_size")
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(6, self.obs_grid_size, self.obs_grid_size),
            dtype=np.float32,
        )
        self.rng = np.random.default_rng(seed)
        self.agent_pos: Cell = config.start
        self.has_key = False
        self.door_open = False
        self.steps = 0
        self.max_steps = config.max_steps or (self.grid_size * self.grid_size * 2)
        self.visitation = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.recent_positions = []  # Sliding window for loop detection
        self.visited_cells = set()  # Track first-time visits for exploration bonus

    # ---------------------- Gym API ----------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if options and "config" in options:
            self.config = GridWorldConfig.from_dict(options["config"])
            self.grid_size = self.config.grid_size
            if self.grid_size > self.obs_grid_size:
                raise ValueError("Config grid_size exceeds observation grid size")
            self.max_steps = self.config.max_steps or (self.grid_size * self.grid_size * 2)
        self.agent_pos = self.config.start
        self.has_key = False
        self.door_open = False
        self.steps = 0
        self.visitation = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.recent_positions = [self.agent_pos]
        self.visited_cells = {self.agent_pos}
        obs = self._get_observation()
        info = self._extra_info()
        return obs, info

    def step(self, action: int):
        assert self.action_space.contains(action)
        reward = self.config.step_penalty
        terminated = False
        truncated = False

        target_pos = self._next_position(action)
        moved = False

        if not self._in_bounds(target_pos):
            reward += self.config.wall_penalty
        elif target_pos in self.config.obstacles:
            reward += self.config.wall_penalty
        elif target_pos == self.config.door:
            if self.has_key:
                self.door_open = True
                self.agent_pos = target_pos
                moved = True
            else:
                reward += self.config.door_penalty
        else:
            self.agent_pos = target_pos
            moved = True

        if moved:
            # Track recent positions to penalize loops (3+ visits in window of 10)
            self.recent_positions.append(self.agent_pos)
            if len(self.recent_positions) > 10:
                self.recent_positions.pop(0)

            if self.recent_positions.count(self.agent_pos) >= 3:
                reward += self.config.repetition_penalty

            if self.agent_pos not in self.visited_cells:
                reward += self.config.exploration_bonus
                self.visited_cells.add(self.agent_pos)

            self.visitation[self.agent_pos] += 1
            if self.agent_pos == self.config.key and not self.has_key:
                self.has_key = True
                reward += self.config.key_reward
            if self.agent_pos in self.config.risk_zones:
                reward += self.config.risk_penalty
            if self.agent_pos == self.config.goal and self.door_open:
                reward += self.config.goal_reward
                terminated = True

        self.steps += 1
        if self.steps >= self.max_steps and not terminated:
            truncated = True

        obs = self._get_observation()
        info = self._extra_info()
        return obs, reward, terminated, truncated, info

    # ---------------------- Helpers ----------------------
    def _next_position(self, action: int) -> Cell:
        row, col = self.agent_pos
        if action == 0:  # up
            row -= 1
        elif action == 1:  # down
            row += 1
        elif action == 2:  # left
            col -= 1
        elif action == 3:  # right
            col += 1
        return row, col

    def _in_bounds(self, pos: Cell) -> bool:
        r, c = pos
        return 0 <= r < self.grid_size and 0 <= c < self.grid_size

    def _get_observation(self) -> np.ndarray:
        obs = np.zeros((6, self.obs_grid_size, self.obs_grid_size), dtype=np.float32)
        obs[0][self.agent_pos] = 1.0
        for cell in self.config.obstacles:
            obs[1][cell] = 1.0
        if not self.has_key:
            obs[2][self.config.key] = 1.0
        obs[3][self.config.door] = 1.0 if not self.door_open else 0.5
        obs[4][self.config.goal] = 1.0
        for cell in self.config.risk_zones:
            obs[5][cell] = 1.0
        return obs

    def _extra_info(self) -> Dict:
        return {
            "agent_pos": self.agent_pos,
            "has_key": self.has_key,
            "door_open": self.door_open,
            "steps": self.steps,
            "max_steps": self.max_steps,
        }

    # ---------------------- Rendering ----------------------
    def render(self):
        grid = self._symbolic_grid()
        rendered = "\n".join(" ".join(row) for row in grid)
        print(rendered)
        return rendered

    def _symbolic_grid(self) -> List[List[str]]:
        grid = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        for r, c in self.config.obstacles:
            grid[r][c] = "#"
        for r, c in self.config.risk_zones:
            grid[r][c] = "!"
        if not self.has_key:
            kr, kc = self.config.key
            grid[kr][kc] = "K"
        dr, dc = self.config.door
        grid[dr][dc] = "D" if not self.door_open else "d"
        gr, gc = self.config.goal
        grid[gr][gc] = "G"
        ar, ac = self.agent_pos
        grid[ar][ac] = "A"
        return grid

    def save_trajectory_png(self, trajectory: Sequence[Cell], output_path: str) -> None:
        grid = np.zeros((self.grid_size, self.grid_size))
        for r, c in self.config.obstacles:
            grid[r, c] = -1
        for r, c in self.config.risk_zones:
            grid[r, c] = -0.5
        for idx, (r, c) in enumerate(trajectory):
            grid[r, c] = idx + 1
        utils.render_grid_heatmap(
            grid,
            title="Traiettoria agente",
            output_path=output_path,
            cmap="viridis",
            show_colorbar=True,
        )

    # ---------------------- Serialization ----------------------
    @classmethod
    def from_json(cls, path: str) -> "GridWorldEnv":
        data = utils.load_level_from_json(path)
        return cls(GridWorldConfig.from_dict(data))

    def to_json(self, path: str) -> None:
        utils.save_level_to_json(self.config.to_dict(), path)


# ---------------------- Registration ----------------------


def register_env(env_id: str = "GridWorld-Themed-v0") -> None:
    if env_id in gym.registry:
        return
    gym.register(
        id=env_id,
        entry_point="gridworld.env:GridWorldEnv",
    )
