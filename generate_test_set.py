
import os
from pathlib import Path
from gridworld.level_generator import generate_level, save_level_to_json

TEST_LEVEL_DIR = Path("levels/test_set")
TEST_LEVEL_DIR.mkdir(parents=True, exist_ok=True)

def generate_test_set():
    print(f"Generating 10 test levels in {TEST_LEVEL_DIR}...")
    
    # Generate 2 levels for each difficulty (1-5)
    level_idx = 1
    for difficulty in range(1, 6):
        for _ in range(2):
            # Use a seed that is likely different from training seeds (which usually start low)
            # We use a large offset + index
            seed = 10000 + level_idx 
            level = generate_level(difficulty, seed=seed)
            
            # Rename to be clear it's a test level
            level["name"] = f"Test_Level_{level_idx}_Diff_{difficulty}"
            
            filename = f"test_level_{level_idx}.json"
            save_path = TEST_LEVEL_DIR / filename
            save_level_to_json(level, str(save_path))
            print(f"Generated {filename} (Difficulty: {difficulty})")
            level_idx += 1

if __name__ == "__main__":
    generate_test_set()
