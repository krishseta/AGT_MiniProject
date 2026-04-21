import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.engine import GameEngine
from training.rule_bot import RuleBot

def generate_replay():
    print("Generating a test replay with RuleBots...")
    engine = GameEngine(seed=42, h=24, w=24, num_players=2)
    engine.reset()

    bots = [RuleBot(i, engine) for i in range(2)]

    for t in range(50):
        actions = {}
        for p in range(2):
            if engine.player_alive[p]:
                actions[p] = bots[p].get_action()
            else:
                actions[p] = None
                
        _, _, terminal, _ = engine.step(actions)
        if terminal:
            break

    os.makedirs("scratch", exist_ok=True)
    out_path = os.path.join("scratch", "game.json")
    engine.save_log(out_path)
    print(f"Replay successfully generated and saved to: {out_path}")

if __name__ == "__main__":
    generate_replay()
