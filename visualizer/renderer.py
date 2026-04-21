import pygame
import numpy as np
import json
import sys
import os
import time
import threading
import queue

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from core.engine import GameEngine
from core.config import (
    TerrainType, UnitType, ActionType,
    NUM_ACTION_TYPES, DIRECTION_DELTAS
)
from training.rule_bot import RuleBot

TILE_SIZE = 40
SIDEBAR_WIDTH = 280
FPS = 10

TERRAIN_COLORS = {
    TerrainType.PLAINS: (180, 220, 150),
    TerrainType.FOREST: (34, 100, 34),
    TerrainType.MOUNTAIN: (120, 110, 110),
    TerrainType.WATER: (60, 170, 250),
    TerrainType.CITY: (230, 200, 150),
}

PLAYER_COLORS = [
    (255, 50, 50),     # Red
    (0, 255, 255),     # Cyan
    (255, 220, 0),     # Yellow
    (200, 50, 255),    # Purple
]

UNIT_SYMBOLS = {
    UnitType.INFANTRY: "I",
    UnitType.CAVALRY: "C",
    UnitType.ARTILLERY: "A",
    UnitType.KNIGHT: "K",
    UnitType.HEAVY: "H",
}


class Visualizer:

    def __init__(self, engine, grid_h=24, grid_w=24):
        pygame.init()
        self.engine = engine
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.screen_w = grid_w * TILE_SIZE + SIDEBAR_WIDTH
        self.screen_h = max(grid_h * TILE_SIZE, 600)
        self.screen = pygame.display.set_mode((self.screen_w, self.screen_h))
        pygame.display.set_caption("Micro-4X MARL Telemetry")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 14)
        self.font_large = pygame.font.SysFont("consolas", 20, bold=True)
        self.font_unit = pygame.font.SysFont("consolas", 18, bold=True)
        self.running = True
        self.paused = False
        self.selected_tile = None
        self.events_log = []
        self.turn_history = []

    def draw_terrain(self):
        for y in range(self.grid_h):
            for x in range(self.grid_w):
                terrain = int(self.engine.terrain[y, x])
                color = TERRAIN_COLORS.get(terrain, (100, 100, 100))
                rect = pygame.Rect(
                    x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE
                )
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (60, 60, 60), rect, 1)

    def draw_territory(self):
        from core.game_theory import compute_ownership_grid
        ownership = compute_ownership_grid(
            self.engine.economy, self.grid_h, self.grid_w
        )
        overlay = pygame.Surface(
            (self.grid_w * TILE_SIZE, self.grid_h * TILE_SIZE), pygame.SRCALPHA
        )
        for y in range(self.grid_h):
            for x in range(self.grid_w):
                owner = int(ownership[y, x])
                if owner >= 0:
                    color = PLAYER_COLORS[owner % len(PLAYER_COLORS)]
                    rect = pygame.Rect(
                        x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE
                    )
                    pygame.draw.rect(
                        overlay, (*color, 40), rect
                    )
        self.screen.blit(overlay, (0, 0))

    def draw_cities(self):
        city_yx = np.argwhere(self.engine.terrain == TerrainType.CITY)
        for i in range(len(city_yx)):
            y, x = int(city_yx[i, 0]), int(city_yx[i, 1])
            owner = int(self.engine.economy.city_owner[y, x])
            infra = int(self.engine.economy.infrastructure[y, x])
            cx = x * TILE_SIZE + TILE_SIZE // 2
            cy = y * TILE_SIZE + TILE_SIZE // 2

            if owner >= 0:
                color = PLAYER_COLORS[owner % len(PLAYER_COLORS)]
                pygame.draw.rect(
                    self.screen, color,
                    (x * TILE_SIZE + 2, y * TILE_SIZE + 2,
                     TILE_SIZE - 4, TILE_SIZE - 4),
                    3,
                )

            label = self.font.render(str(infra), True, (255, 255, 255))
            lr = label.get_rect(center=(cx, cy))
            self.screen.blit(label, lr)

    def draw_units(self):
        alive_ids = np.where(self.engine.units.alive)[0]
        for uid in alive_ids:
            y = int(self.engine.units.positions[uid, 0])
            x = int(self.engine.units.positions[uid, 1])
            owner = int(self.engine.units.owner[uid])
            utype = int(self.engine.units.unit_type[uid])
            hp = float(self.engine.units.hp[uid])
            max_hp = float(self.engine.units.max_hp[uid])

            color = PLAYER_COLORS[owner % len(PLAYER_COLORS)]
            cx = x * TILE_SIZE + TILE_SIZE // 2
            cy = y * TILE_SIZE + TILE_SIZE // 2

            pygame.draw.circle(self.screen, color, (cx, cy), TILE_SIZE // 3)
            pygame.draw.circle(self.screen, (0, 0, 0), (cx, cy), TILE_SIZE // 3, 1)

            sym = UNIT_SYMBOLS.get(utype, "?")
            label = self.font_unit.render(sym, True, (255, 255, 255))
            lr = label.get_rect(center=(cx, cy))
            self.screen.blit(label, lr)

            hp_ratio = hp / max_hp if max_hp > 0 else 0
            bar_w = TILE_SIZE - 8
            bar_h = 4
            bar_x = x * TILE_SIZE + 4
            bar_y = y * TILE_SIZE + TILE_SIZE - 7
            pygame.draw.rect(
                self.screen, (80, 0, 0),
                (bar_x, bar_y, bar_w, bar_h),
            )
            pygame.draw.rect(
                self.screen, (0, 200, 0),
                (bar_x, bar_y, int(bar_w * hp_ratio), bar_h),
            )

    def draw_resource_caches(self):
        for ci in range(len(self.engine.resource_caches)):
            cy, cx = int(self.engine.resource_caches[ci, 0]), int(self.engine.resource_caches[ci, 1])
            if cy < 0:
                continue
            px = cx * TILE_SIZE + TILE_SIZE // 2
            py = cy * TILE_SIZE + TILE_SIZE // 2
            pygame.draw.polygon(
                self.screen, (255, 215, 0),
                [(px, py - 8), (px + 7, py + 5), (px - 7, py + 5)],
            )

    def draw_sidebar(self):
        sidebar_x = self.grid_w * TILE_SIZE
        pygame.draw.rect(
            self.screen, (30, 30, 40),
            (sidebar_x, 0, SIDEBAR_WIDTH, self.screen_h),
        )

        y_offset = 10
        title = self.font_large.render("MICRO-4X TELEMETRY", True, (220, 220, 240))
        self.screen.blit(title, (sidebar_x + 10, y_offset))
        y_offset += 35

        turn_text = self.font.render(
            f"Turn: {self.engine.turn}", True, (180, 180, 200)
        )
        self.screen.blit(turn_text, (sidebar_x + 10, y_offset))
        y_offset += 25

        status = "PAUSED" if self.paused else "RUNNING"
        status_color = (255, 180, 50) if self.paused else (100, 255, 100)
        status_text = self.font.render(f"Status: {status}", True, status_color)
        self.screen.blit(status_text, (sidebar_x + 10, y_offset))
        y_offset += 35

        pygame.draw.line(
            self.screen, (80, 80, 100),
            (sidebar_x + 10, y_offset), (sidebar_x + SIDEBAR_WIDTH - 10, y_offset),
        )
        y_offset += 15

        for p in range(self.engine.num_players):
            color = PLAYER_COLORS[p % len(PLAYER_COLORS)]
            alive = bool(self.engine.player_alive[p])

            pygame.draw.rect(
                self.screen, color, (sidebar_x + 10, y_offset, 12, 12)
            )
            pname = f"Player {p}" if alive else f"Player {p} [DEAD]"
            text_color = color if alive else (120, 120, 120)
            label = self.font.render(pname, True, text_color)
            self.screen.blit(label, (sidebar_x + 28, y_offset - 1))
            y_offset += 20

            if alive:
                currency = float(self.engine.economy.currency[p])
                cities = self.engine.economy.player_city_count(p)
                units = self.engine.units.count_player_alive(p)
                pop = int(self.engine.economy.population[p])

                stats = [
                    f"  ${currency:.0f}  Cities:{cities}  Units:{units}",
                    f"  Pop:{pop}",
                ]
                for s in stats:
                    st = self.font.render(s, True, (160, 160, 180))
                    self.screen.blit(st, (sidebar_x + 10, y_offset))
                    y_offset += 18

            y_offset += 10

        pygame.draw.line(
            self.screen, (80, 80, 100),
            (sidebar_x + 10, y_offset), (sidebar_x + SIDEBAR_WIDTH - 10, y_offset),
        )
        y_offset += 15

        events_title = self.font.render("Recent Events:", True, (200, 200, 220))
        self.screen.blit(events_title, (sidebar_x + 10, y_offset))
        y_offset += 22

        display_events = self.events_log[-12:]
        for evt_str in display_events:
            evt_text = self.font.render(evt_str[:35], True, (140, 140, 160))
            self.screen.blit(evt_text, (sidebar_x + 10, y_offset))
            y_offset += 16

        y_offset = self.screen_h - 200
        pygame.draw.line(
            self.screen, (80, 80, 100),
            (sidebar_x + 10, y_offset), (sidebar_x + SIDEBAR_WIDTH - 10, y_offset),
        )
        y_offset += 10

        legend_title = self.font.render("LEGEND:", True, (200, 200, 220))
        self.screen.blit(legend_title, (sidebar_x + 10, y_offset))
        y_offset += 20
        
        symbols_str = " ".join([f"{k.name[:1]}={v}" for k, v in UNIT_SYMBOLS.items()])
        legend_symbols = self.font.render(symbols_str, True, (180, 180, 200))
        self.screen.blit(legend_symbols, (sidebar_x + 10, y_offset))
        y_offset += 20
        
        terrain_str = "Plains Forest Mntn Water City"
        legend_terrain = self.font.render(terrain_str, True, (180, 180, 200))
        self.screen.blit(legend_terrain, (sidebar_x + 10, y_offset))
        y_offset += 10

        from core.config import TerrainType
        tx = sidebar_x + 10
        for t_type in [TerrainType.PLAINS, TerrainType.FOREST, TerrainType.MOUNTAIN, TerrainType.WATER, TerrainType.CITY]:
            pygame.draw.rect(self.screen, TERRAIN_COLORS[t_type], (tx, y_offset, 25, 15))
            tx += 53
            
        y_offset += 30

        pygame.draw.line(
            self.screen, (80, 80, 100),
            (sidebar_x + 10, y_offset), (sidebar_x + SIDEBAR_WIDTH - 10, y_offset),
        )
        y_offset += 10

        controls = [
            "[SPACE] Pause/Resume",
            "[S] Step (when paused)",
            "[R] Reset",
            "[Q/ESC] Quit",
        ]
        for c in controls:
            ct = self.font.render(c, True, (120, 120, 140))
            self.screen.blit(ct, (sidebar_x + 10, y_offset))
            y_offset += 16

    def draw_selection(self):
        if self.selected_tile is None:
            return
        y, x = self.selected_tile
        rect = pygame.Rect(
            x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE
        )
        pygame.draw.rect(self.screen, (255, 255, 0), rect, 2)

    def format_events(self, events):
        for evt in events:
            etype = evt.get("type", "?")
            if etype == "spawn":
                self.events_log.append(
                    f"P{evt['player']} spawns unit@{evt['pos']}"
                )
            elif etype == "move":
                self.events_log.append(
                    f"P{evt['player']} move {evt['from']}->{evt['to']}"
                )
            elif etype == "attack":
                kill_str = " KILL" if evt.get("killed") else ""
                self.events_log.append(
                    f"P{evt['attacker_player']} atk dmg={evt['damage']:.0f}{kill_str}"
                )
            elif etype == "capture_city":
                self.events_log.append(
                    f"P{evt['new_owner']} captures city@{evt['pos']}"
                )
            elif etype == "invest":
                self.events_log.append(
                    f"P{evt['player']} invests @{evt['pos']} lv{evt['new_level']}"
                )
            elif etype == "research":
                self.events_log.append(
                    f"P{evt['player']} researches {evt['tech_name']}"
                )
            elif etype == "ipd":
                self.events_log.append(
                    f"IPD P{evt['players']}: {evt['choices']}"
                )
            elif etype == "stag_hunt":
                self.events_log.append(
                    f"Stag P{evt['players']}: {evt['choices']}"
                )

    def run_live(self):
        bots = [RuleBot(i, self.engine) for i in range(self.engine.num_players)]
        self.engine.reset()
        step_requested = False

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        self.running = False
                    elif event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_s and self.paused:
                        step_requested = True
                    elif event.key == pygame.K_r:
                        self.engine.reset()
                        self.events_log = []
                        bots = [RuleBot(i, self.engine) for i in range(self.engine.num_players)]
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = event.pos
                    tx = mx // TILE_SIZE
                    ty = my // TILE_SIZE
                    if 0 <= tx < self.grid_w and 0 <= ty < self.grid_h:
                        self.selected_tile = (ty, tx)
                    else:
                        self.selected_tile = None

            if (not self.paused or step_requested) and np.any(self.engine.player_alive):
                step_requested = False
                actions = {}
                for p in range(self.engine.num_players):
                    if self.engine.player_alive[p]:
                        actions[p] = bots[p].get_action()
                    else:
                        actions[p] = np.zeros(
                            (self.grid_h, self.grid_w), dtype=np.int32
                        )

                _, _, terminal, infos = self.engine.step(actions)

                if len(self.engine.turn_log) > 0:
                    last_state = self.engine.turn_log[-1]
                    self.format_events(last_state.get("events", []))

                if terminal:
                    self.paused = True
                    alive_players = np.where(self.engine.player_alive)[0]
                    if len(alive_players) == 1:
                        self.events_log.append(
                            f"=== PLAYER {alive_players[0]} WINS ==="
                        )
                    else:
                        self.events_log.append("=== GAME OVER (draw) ===")

            self.screen.fill((20, 20, 30))
            self.draw_terrain()
            self.draw_territory()
            self.draw_cities()
            self.draw_resource_caches()
            self.draw_units()
            self.draw_selection()
            self.draw_sidebar()

            pygame.display.flip()
            self.clock.tick(FPS)

        pygame.quit()

    def run_replay(self, log_path):
        with open(log_path, "r") as f:
            log_data = json.load(f)

        frame_idx = 0
        total_frames = len(log_data)

        while self.running and frame_idx < total_frames:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        self.running = False
                    elif event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_s and self.paused:
                        frame_idx = min(frame_idx + 1, total_frames - 1)

            if not self.paused:
                frame_idx += 1
                if frame_idx >= total_frames:
                    self.paused = True
                    frame_idx = total_frames - 1

            self.screen.fill((20, 20, 30))

            frame = log_data[min(frame_idx, total_frames - 1)]
            self._draw_replay_frame(frame)

            self.draw_sidebar()
            pygame.display.flip()
            self.clock.tick(FPS)

        pygame.quit()

    def _draw_replay_frame(self, frame):
        terrain = np.array(frame["terrain"], dtype=np.int32)
        for y in range(self.grid_h):
            for x in range(self.grid_w):
                t = int(terrain[y, x])
                color = TERRAIN_COLORS.get(t, (100, 100, 100))
                rect = pygame.Rect(
                    x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE
                )
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (60, 60, 60), rect, 1)

        for unit in frame.get("units", []):
            y = unit["y"]
            x = unit["x"]
            owner = unit["owner"]
            utype = unit["type"]
            hp = unit["hp"]
            max_hp = unit["max_hp"]

            color = PLAYER_COLORS[owner % len(PLAYER_COLORS)]
            cx = x * TILE_SIZE + TILE_SIZE // 2
            cy = y * TILE_SIZE + TILE_SIZE // 2

            pygame.draw.circle(self.screen, color, (cx, cy), TILE_SIZE // 3)

            sym = UNIT_SYMBOLS.get(utype, "?")
            label = self.font_unit.render(sym, True, (255, 255, 255))
            lr = label.get_rect(center=(cx, cy))
            self.screen.blit(label, lr)

            if max_hp > 0:
                hp_ratio = hp / max_hp
                bar_w = TILE_SIZE - 8
                bar_h = 4
                bar_x = x * TILE_SIZE + 4
                bar_y = y * TILE_SIZE + TILE_SIZE - 7
                pygame.draw.rect(
                    self.screen, (80, 0, 0),
                    (bar_x, bar_y, bar_w, bar_h),
                )
                pygame.draw.rect(
                    self.screen, (0, 200, 0),
                    (bar_x, bar_y, int(bar_w * hp_ratio), bar_h),
                )


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", type=str, default=None)
    parser.add_argument("--grid-h", type=int, default=24)
    parser.add_argument("--grid-w", type=int, default=24)
    parser.add_argument("--players", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.replay:
        engine = GameEngine(
            seed=args.seed, h=args.grid_h, w=args.grid_w,
            num_players=args.players
        )
        viz = Visualizer(engine, args.grid_h, args.grid_w)
        viz.run_replay(args.replay)
    else:
        engine = GameEngine(
            seed=args.seed, h=args.grid_h, w=args.grid_w,
            num_players=args.players
        )
        viz = Visualizer(engine, args.grid_h, args.grid_w)
        viz.run_live()


if __name__ == "__main__":
    main()
