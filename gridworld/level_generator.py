"""Procedural level generation for the GridWorld environment."""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import utils

Cell = Tuple[int, int]


@dataclass
class LevelMeta:
    difficulty: int
    grid_size: int
    base_obstacles: int
    base_risks: int
    description: str


DEFAULT_LEVEL_METADATA: Sequence[LevelMeta] = [
    LevelMeta(1, 7, 3, 1, "Introduzione"),
    LevelMeta(2, 8, 4, 2, "Corridoi stretti"),
    LevelMeta(3, 9, 5, 3, "Labirinto moderato"),
    LevelMeta(4, 10, 6, 4, "Labirinto avanzato"),
    LevelMeta(5, 11, 7, 5, "Sfida finale"),
]


def build_default_level_pack(output_dir: str) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for meta in DEFAULT_LEVEL_METADATA:
        level = generate_level(meta.difficulty, seed=meta.difficulty)
        path = output / f"level_{meta.difficulty}.json"
        save_level_to_json(level, str(path))


def save_level_to_json(level: Dict, path: str) -> None:
    utils.save_level_to_json(level, path)


def load_level_from_json(path: str) -> Dict:
    return utils.load_level_from_json(path)


def generate_level(difficulty: int, seed: Optional[int] = None) -> Dict:
    meta = DEFAULT_LEVEL_METADATA[difficulty - 1]
    rng = np.random.default_rng(seed)
    grid_size = meta.grid_size
    start = (0, 0)
    door_row = grid_size // 2
    door_col = rng.integers(grid_size // 3, grid_size - grid_size // 3)
    door = (door_row, int(door_col))
    goal = (grid_size - 2, grid_size - 2)
    key_row = rng.integers(1, door_row - 1)
    key_col = rng.integers(1, grid_size - 2)
    key = (int(key_row), int(key_col))

    obstacles = _generate_obstacles(meta, rng, grid_size, door, start, goal, key)
    risk_zones = _generate_risks(meta, rng, grid_size, {door, start, goal, key}, obstacles)

    max_steps = grid_size * grid_size * 2

    level = {
        "name": f"Procedural_Level_{difficulty}",
        "difficulty": difficulty,
        "grid_size": grid_size,
        "start": list(start),
        "key": list(key),
        "door": list(door),
        "goal": list(goal),
        "obstacles": [list(cell) for cell in obstacles],
        "risk_zones": [list(cell) for cell in risk_zones],
        "max_steps": max_steps,
    }
    if not _validate_paths(level):
        return generate_level(difficulty, seed=(seed or 0) + 1337)
    return level


def _generate_obstacles(
    meta: LevelMeta,
    rng: np.random.Generator,
    grid_size: int,
    door: Cell,
    start: Cell,
    goal: Cell,
    key: Cell,
) -> List[Cell]:
    num_obstacles = meta.base_obstacles * meta.difficulty
    cells = [(r, c) for r in range(grid_size) for c in range(grid_size)]
    forbidden = {start, goal, door, key}
    door_row = door[0]

    # Build a solid barrier across the door row so the door is the only passage.
    barrier = {
        (door_row, c)
        for c in range(grid_size)
        if (door_row, c) not in forbidden and c != door[1]
    }

    obstacles = set(barrier)
    target_total = len(barrier) + num_obstacles
    while len(obstacles) < target_total:
        cell = tuple(rng.choice(cells))
        if cell in forbidden or cell == door:
            continue
        if cell in obstacles:
            continue
        if cell[0] == door_row:
            continue
        obstacles.add(cell)
    return sorted(obstacles)


def _generate_risks(
    meta: LevelMeta,
    rng: np.random.Generator,
    grid_size: int,
    critical: set,
    obstacles: List[Cell],
) -> List[Cell]:
    num_risk = meta.base_risks * meta.difficulty
    risks = set()
    occupied = set(obstacles) | critical
    while len(risks) < num_risk:
        cell = (rng.integers(0, grid_size), rng.integers(0, grid_size))
        if cell in occupied:
            continue
        risks.add(cell)
    return sorted(risks)


def _validate_paths(level: Dict) -> bool:
    grid_size = level["grid_size"]
    start = tuple(level["start"])
    key = tuple(level["key"])
    door = tuple(level["door"])
    goal = tuple(level["goal"])
    obstacles = {tuple(cell) for cell in level["obstacles"]}

    def bfs(source: Cell, target: Cell, door_closed: bool) -> bool:
        queue = [source]
        visited = {source}
        while queue:
            r, c = queue.pop(0)
            if (r, c) == target:
                return True
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < grid_size and 0 <= nc < grid_size):
                    continue
                cell = (nr, nc)
                if cell in visited or cell in obstacles:
                    continue
                if door_closed and cell == door:
                    continue
                visited.add(cell)
                queue.append(cell)
        return False

    return bfs(start, key, True) and bfs(key, goal, False)
