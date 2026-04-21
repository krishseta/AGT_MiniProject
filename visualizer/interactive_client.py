import pygame
import numpy as np
import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from env.micro4x_env import Micro4XEnv
from core.config import NUM_ACTION_TYPES, ActionType, UnitType
from visualizer.renderer import Visualizer, TILE_SIZE, SIDEBAR_WIDTH, PLAYER_COLORS, UNIT_SYMBOLS
from env.action_mask_model import GridWiseActionMaskModel
from training.rule_bot import RuleBot


# ── UI Constants ──────────────────────────────────────────────────────
SIDEBAR_BG = (30, 30, 40)
HEADER_COLOR = (220, 220, 240)
SUBHEADER_COLOR = (180, 180, 200)
DIM_TEXT = (120, 120, 140)
GOLD = (255, 215, 0)
BTN_VALID = (60, 160, 80)
BTN_INVALID = (60, 60, 70)
BTN_QUEUED = (200, 170, 40)
BTN_HOVER = (80, 200, 100)
BTN_TEXT_VALID = (240, 240, 240)
BTN_TEXT_INVALID = (90, 90, 100)

ACTION_BTN_H = 26
ACTION_BTN_GAP = 3
ACTION_BTN_W = SIDEBAR_WIDTH - 40
SCROLL_BAR_W = 10

HEADER_ZONE_H = 140          # reserved for title + turn + tile info
FOOTER_ZONE_H = 60           # reserved for END TURN button
ACTION_ZONE_TOP = HEADER_ZONE_H
ACTION_ZONE_BOTTOM_PAD = FOOTER_ZONE_H


class InteractiveClient:
    def __init__(self, grid_h=24, grid_w=24, checkpoint_path=None, use_rule_bot=True):
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.turn_number = 0
        self.game_over = False
        self.game_over_msg = ""

        self.env = Micro4XEnv(grid_h=grid_h, grid_w=grid_w)
        self._mask_size = grid_h * grid_w * NUM_ACTION_TYPES
        self.obs, _ = self.env.reset()

        self.selected_pos = None
        self.human_action_grid = np.full(grid_h * grid_w, ActionType.NO_OP, dtype=np.int32)

        self.visual_positions = {}
        self.renderer = Visualizer(self.env.engine, grid_h=grid_h, grid_w=grid_w)

        # ── CPU opponent ──
        self.use_rule_bot = use_rule_bot
        if use_rule_bot:
            self.cpu_bot = RuleBot(1, self.env.engine)
            self.cpu_model = None
            print("CPU opponent: RuleBot (heuristic)")
        else:
            self.cpu_bot = None
            self.cpu_model = self._load_cpu_model(checkpoint_path)

        # ── Scroll state ──
        self.scroll_offset = 0
        self.max_scroll = 0

    # ── CPU model loading ─────────────────────────────────────────────
    def _load_cpu_model(self, checkpoint_path=None):
        model = GridWiseActionMaskModel(
            obs_space=None,
            action_space=None,
            num_outputs=self.grid_h * self.grid_w * NUM_ACTION_TYPES,
            model_config={"custom_model_config": {"grid_h": self.grid_h, "grid_w": self.grid_w}},
            name="local_cpu"
        )

        if checkpoint_path and os.path.isfile(checkpoint_path):
            print(f"Loading CPU model weights from: {checkpoint_path}")
            state = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(state)
            print("  Checkpoint loaded successfully.")
        elif checkpoint_path:
            print(f"WARNING: Checkpoint path not found: {checkpoint_path}")
            print("  Falling back to random weights.")
        else:
            print("No checkpoint specified — CPU plays with random weights.")

        model.eval()
        return model

    # ── Action helpers ────────────────────────────────────────────────
    def _get_action_entries(self):
        """Build the list of (action_id, name, is_valid, is_queued) for the selected tile."""
        if not self.selected_pos:
            return []
        x, y = self.selected_pos
        base_idx = (y * self.grid_w + x) * NUM_ACTION_TYPES
        # Flat obs: first _mask_size floats = action mask, rest = observation
        full_mask = self.obs["player_0"][:self._mask_size]
        mask = full_mask[base_idx: base_idx + NUM_ACTION_TYPES]
        flat_coord = y * self.grid_w + x
        queued_act = int(self.human_action_grid[flat_coord])

        entries = []
        for act in range(1, NUM_ACTION_TYPES):  # skip NO_OP
            valid = mask[act] > 0.5
            queued = (queued_act == act)
            entries.append((act, ActionType(act).name, valid, queued))
        return entries

    def _content_height(self, entries):
        return len(entries) * (ACTION_BTN_H + ACTION_BTN_GAP)

    # ── Click handling ────────────────────────────────────────────────
    def handle_click(self, pos):
        sidebar_x = self.grid_w * TILE_SIZE
        gx = pos[0] // TILE_SIZE
        gy = pos[1] // TILE_SIZE

        # Click on grid
        if pos[0] < sidebar_x and 0 <= gx < self.grid_w and 0 <= gy < self.grid_h:
            self.selected_pos = (gx, gy)
            self.scroll_offset = 0
            return

        # Click on sidebar action buttons
        if pos[0] >= sidebar_x:
            self._handle_sidebar_click(pos)

    def _handle_sidebar_click(self, pos):
        sidebar_x = self.grid_w * TILE_SIZE

        # Check END TURN button
        btn_rect = pygame.Rect(
            sidebar_x + 20,
            self.renderer.screen_h - FOOTER_ZONE_H + 10,
            ACTION_BTN_W, 40
        )
        if btn_rect.collidepoint(pos):
            self.step_turn()
            return

        if not self.selected_pos:
            return

        entries = self._get_action_entries()
        action_zone_h = self.renderer.screen_h - ACTION_ZONE_TOP - ACTION_ZONE_BOTTOM_PAD
        btn_x = sidebar_x + 20

        for idx, (act, name, valid, queued) in enumerate(entries):
            btn_y = ACTION_ZONE_TOP + idx * (ACTION_BTN_H + ACTION_BTN_GAP) - self.scroll_offset
            if btn_y + ACTION_BTN_H < ACTION_ZONE_TOP or btn_y > ACTION_ZONE_TOP + action_zone_h:
                continue  # clipped
            rect = pygame.Rect(btn_x, btn_y, ACTION_BTN_W, ACTION_BTN_H)
            if rect.collidepoint(pos) and valid:
                x, y = self.selected_pos
                flat_coord = y * self.grid_w + x
                current = int(self.human_action_grid[flat_coord])
                if current == act:
                    # deselect (toggle off)
                    self.human_action_grid[flat_coord] = ActionType.NO_OP
                    print(f"Action deselected: {name} at ({x}, {y})")
                else:
                    self.human_action_grid[flat_coord] = act
                    print(f"Action queued: {name} at ({x}, {y})")
                return

    def handle_scroll(self, dy):
        """dy is positive when scrolling up (away), negative when scrolling down."""
        scroll_speed = (ACTION_BTN_H + ACTION_BTN_GAP) * 2
        self.scroll_offset -= dy * scroll_speed
        self.scroll_offset = max(0, min(self.scroll_offset, self.max_scroll))

    # ── Drawing ───────────────────────────────────────────────────────
    def draw_sidebar_ui(self):
        sidebar_x = self.grid_w * TILE_SIZE
        screen = self.renderer.screen
        font = self.renderer.font
        font_lg = self.renderer.font_large

        # Background
        pygame.draw.rect(screen, SIDEBAR_BG, (sidebar_x, 0, SIDEBAR_WIDTH, self.renderer.screen_h))

        y_off = 10

        # ── Title ──
        title = font_lg.render("COMMAND PANEL", True, HEADER_COLOR)
        screen.blit(title, (sidebar_x + 12, y_off))
        y_off += 28

        # ── Turn counter ──
        turn_text = font.render(f"Turn: {self.turn_number}  /  {self.env.max_turns}", True, GOLD)
        screen.blit(turn_text, (sidebar_x + 12, y_off))
        y_off += 22

        # ── Player stats ──
        for p in range(self.env.num_players):
            alive = bool(self.env.engine.player_alive[p])
            color = PLAYER_COLORS[p % len(PLAYER_COLORS)] if alive else (80, 80, 80)
            tag = "YOU" if p == 0 else "CPU"
            pygame.draw.rect(screen, color, (sidebar_x + 12, y_off, 10, 10))
            lbl = font.render(f" P{p} ({tag}): ${self.env.engine.economy.currency[p]:.0f}  "
                              f"U:{self.env.engine.units.count_player_alive(p)}  "
                              f"C:{self.env.engine.economy.player_city_count(p)}",
                              True, color)
            screen.blit(lbl, (sidebar_x + 26, y_off - 2))
            y_off += 18

        y_off += 4

        # ── Selected tile info ──
        if self.selected_pos:
            x, y = self.selected_pos
            sel_text = font.render(f"Tile: ({x}, {y})", True, SUBHEADER_COLOR)
            screen.blit(sel_text, (sidebar_x + 12, y_off))
            y_off += 18

            # Show unit on tile if any
            unit_grid = self.env.engine.units.build_unit_grid(self.grid_h, self.grid_w)
            uid = unit_grid[y, x]
            if uid >= 0:
                u = self.env.engine.units
                utype = int(u.unit_type[uid])
                owner = int(u.owner[uid])
                hp = float(u.hp[uid])
                max_hp = float(u.max_hp[uid])
                sym = UNIT_SYMBOLS.get(utype, "?")
                tag = "Your" if owner == 0 else "Enemy"
                unit_info = font.render(f"{tag} {UnitType(utype).name} [{sym}] HP:{hp:.0f}/{max_hp:.0f}", True, SUBHEADER_COLOR)
                screen.blit(unit_info, (sidebar_x + 12, y_off))
            y_off += 18
        else:
            hint = font.render("Click a tile to act", True, DIM_TEXT)
            screen.blit(hint, (sidebar_x + 12, y_off))
            y_off += 18

        # ── Separator ──
        pygame.draw.line(screen, (60, 60, 80),
                         (sidebar_x + 10, HEADER_ZONE_H - 4),
                         (sidebar_x + SIDEBAR_WIDTH - 10, HEADER_ZONE_H - 4))

        # ── Scrollable action buttons ──
        entries = self._get_action_entries()
        action_zone_h = self.renderer.screen_h - ACTION_ZONE_TOP - ACTION_ZONE_BOTTOM_PAD
        content_h = self._content_height(entries)
        self.max_scroll = max(0, content_h - action_zone_h)

        # Clip region
        clip_rect = pygame.Rect(sidebar_x, ACTION_ZONE_TOP, SIDEBAR_WIDTH, action_zone_h)
        screen.set_clip(clip_rect)

        mouse_pos = pygame.mouse.get_pos()
        btn_x = sidebar_x + 20

        for idx, (act, name, valid, queued) in enumerate(entries):
            btn_y = ACTION_ZONE_TOP + idx * (ACTION_BTN_H + ACTION_BTN_GAP) - self.scroll_offset
            rect = pygame.Rect(btn_x, btn_y, ACTION_BTN_W, ACTION_BTN_H)

            # Skip if completely out of view
            if btn_y + ACTION_BTN_H < ACTION_ZONE_TOP or btn_y > ACTION_ZONE_TOP + action_zone_h:
                continue

            # Color logic
            if queued:
                bg = BTN_QUEUED
                fg = (0, 0, 0)
            elif valid and rect.collidepoint(mouse_pos):
                bg = BTN_HOVER
                fg = (0, 0, 0)
            elif valid:
                bg = BTN_VALID
                fg = BTN_TEXT_VALID
            else:
                bg = BTN_INVALID
                fg = BTN_TEXT_INVALID

            pygame.draw.rect(screen, bg, rect, border_radius=4)
            label = font.render(name, True, fg)
            screen.blit(label, (btn_x + 8, btn_y + 5))

            if queued:
                check = font.render("✓", True, (0, 0, 0))
                screen.blit(check, (btn_x + ACTION_BTN_W - 22, btn_y + 4))

        screen.set_clip(None)

        # ── Scroll bar ──
        if self.max_scroll > 0:
            bar_track_y = ACTION_ZONE_TOP
            bar_track_h = action_zone_h
            bar_h = max(20, int(bar_track_h * (action_zone_h / content_h)))
            bar_y = bar_track_y + int((self.scroll_offset / self.max_scroll) * (bar_track_h - bar_h))
            bar_x = sidebar_x + SIDEBAR_WIDTH - SCROLL_BAR_W - 4
            pygame.draw.rect(screen, (50, 50, 60), (bar_x, bar_track_y, SCROLL_BAR_W, bar_track_h), border_radius=3)
            pygame.draw.rect(screen, (120, 120, 140), (bar_x, bar_y, SCROLL_BAR_W, bar_h), border_radius=3)

        # ── Footer separator ──
        footer_y = self.renderer.screen_h - FOOTER_ZONE_H
        pygame.draw.line(screen, (60, 60, 80),
                         (sidebar_x + 10, footer_y),
                         (sidebar_x + SIDEBAR_WIDTH - 10, footer_y))

        # ── END TURN button ──
        btn_rect = pygame.Rect(sidebar_x + 20, footer_y + 10, ACTION_BTN_W, 40)
        btn_color = (180, 50, 50)
        if btn_rect.collidepoint(mouse_pos):
            btn_color = (220, 70, 70)
        pygame.draw.rect(screen, btn_color, btn_rect, border_radius=8)
        btn_label = font_lg.render("END TURN  [SPACE]", True, (255, 255, 255))
        lbl_rect = btn_label.get_rect(center=btn_rect.center)
        screen.blit(btn_label, lbl_rect)

        # ── Game over overlay ──
        if self.game_over:
            overlay = pygame.Surface((SIDEBAR_WIDTH, 40), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            screen.blit(overlay, (sidebar_x, footer_y - 44))
            go_text = font_lg.render(self.game_over_msg, True, GOLD)
            screen.blit(go_text, (sidebar_x + 12, footer_y - 38))

    # ── Main loop ─────────────────────────────────────────────────────
    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE and not self.game_over:
                        self.step_turn()
                    elif event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1 and not self.game_over:
                        self.handle_click(event.pos)
                    elif event.button == 4:   # scroll up
                        self.handle_scroll(1)
                    elif event.button == 5:   # scroll down
                        self.handle_scroll(-1)
                elif event.type == pygame.MOUSEWHEEL:
                    self.handle_scroll(event.y)

            self.renderer.screen.fill((20, 20, 30))
            self.renderer.draw_terrain()
            self.renderer.draw_territory()
            self.renderer.draw_cities()

            # Highlight selected tile
            if self.selected_pos:
                gx, gy = self.selected_pos
                sel_rect = pygame.Rect(gx * TILE_SIZE, gy * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                pygame.draw.rect(self.renderer.screen, (255, 255, 255), sel_rect, 3)

                # Also highlight tiles with queued actions
                for tile_idx in range(self.grid_h * self.grid_w):
                    if self.human_action_grid[tile_idx] != ActionType.NO_OP:
                        ty = tile_idx // self.grid_w
                        tx = tile_idx % self.grid_w
                        q_rect = pygame.Rect(tx * TILE_SIZE + 2, ty * TILE_SIZE + 2, TILE_SIZE - 4, TILE_SIZE - 4)
                        pygame.draw.rect(self.renderer.screen, GOLD, q_rect, 2)

            self.update_unit_lerps()
            self.draw_sidebar_ui()

            pygame.display.flip()
            self.renderer.clock.tick(60)

        pygame.quit()

    def update_unit_lerps(self):
        units = self.env.engine.units
        for u_id in range(units.max_units):
            if units.alive[u_id]:
                target_y, target_x = units.positions[u_id]
                target_px = float(target_x * TILE_SIZE)
                target_py = float(target_y * TILE_SIZE)

                if u_id not in self.visual_positions:
                    self.visual_positions[u_id] = [target_px, target_py]
                else:
                    vp = self.visual_positions[u_id]
                    vp[0] += (target_px - vp[0]) * 0.2
                    vp[1] += (target_py - vp[1]) * 0.2

                vx, vy = self.visual_positions[u_id]
                cx = int(vx) + TILE_SIZE // 2
                cy = int(vy) + TILE_SIZE // 2

                owner = int(units.owner[u_id])
                utype = int(units.unit_type[u_id])
                color = PLAYER_COLORS[owner % len(PLAYER_COLORS)]

                pygame.draw.circle(self.renderer.screen, color, (cx, cy), TILE_SIZE // 3)
                pygame.draw.circle(self.renderer.screen, (0, 0, 0), (cx, cy), TILE_SIZE // 3, 1)

                sym = UNIT_SYMBOLS.get(utype, "?")
                label = self.renderer.font_unit.render(sym, True, (255, 255, 255))
                self.renderer.screen.blit(label, label.get_rect(center=(cx, cy)))

                # HP bar
                hp = float(units.hp[u_id])
                max_hp = float(units.max_hp[u_id])
                if max_hp > 0:
                    hp_ratio = hp / max_hp
                    bar_w = TILE_SIZE - 8
                    bar_x = int(vx) + 4
                    bar_y = int(vy) + TILE_SIZE - 7
                    pygame.draw.rect(self.renderer.screen, (80, 0, 0), (bar_x, bar_y, bar_w, 4))
                    pygame.draw.rect(self.renderer.screen, (0, 200, 0), (bar_x, bar_y, int(bar_w * hp_ratio), 4))
            else:
                # Remove dead units from visual cache
                self.visual_positions.pop(u_id, None)

    # ── Turn step ─────────────────────────────────────────────────────
    def step_turn(self):
        if self.game_over:
            return

        # ── CPU action ──
        if self.use_rule_bot:
            cpu_action_grid = self.cpu_bot.get_action()
            cpu_actions_flat = cpu_action_grid.flatten()
        else:
            cpu_actions_flat = self._get_model_actions()

        actions = {
            "player_0": self.human_action_grid.copy(),
            "player_1": cpu_actions_flat,
        }

        self.human_action_grid.fill(ActionType.NO_OP)
        self.selected_pos = None
        self.scroll_offset = 0

        self.obs, rewards, terms, truncs, infos = self.env.step(actions)
        self.turn_number = self.env.engine.turn

        # Check game over
        p0_done = terms.get("player_0", False) or truncs.get("player_0", False)
        p1_done = terms.get("player_1", False) or truncs.get("player_1", False)

        if p0_done or p1_done:
            self.game_over = True
            r0 = rewards.get("player_0", 0)
            r1 = rewards.get("player_1", 0)
            if r0 > r1:
                self.game_over_msg = "YOU WIN!"
            elif r1 > r0:
                self.game_over_msg = "CPU WINS!"
            else:
                self.game_over_msg = "DRAW"
            print(f"Game Over — Turn {self.turn_number}: {self.game_over_msg} (P0={r0:.1f}, P1={r1:.1f})")

    def _get_model_actions(self):
        """Get CPU actions from the neural net model."""
        flat = self.obs["player_1"]
        cpu_mask = flat[:self._mask_size]
        cpu_obs_flat = flat[self._mask_size:]

        # Model now expects flat input: [mask | obs]
        flat_t = torch.tensor(flat).unsqueeze(0)

        with torch.no_grad():
            input_dict = {"obs": flat_t}
            logits, _ = self.cpu_model(input_dict, None, None)

            grid_logits = logits.squeeze(0).view(self.grid_h * self.grid_w, NUM_ACTION_TYPES)
            cpu_actions = torch.argmax(grid_logits, dim=-1).numpy()

        # Enforce action mask safety
        for i in range(len(cpu_actions)):
            act = cpu_actions[i]
            if cpu_mask[i * NUM_ACTION_TYPES + act] < 0.5:
                cpu_actions[i] = ActionType.NO_OP

        return cpu_actions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Micro-4X Human vs CPU Interactive Client")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to a PyTorch checkpoint (.pt) file for the CPU model. "
             "If not provided, the CPU is a RuleBot."
    )
    parser.add_argument("--grid-h", type=int, default=24)
    parser.add_argument("--grid-w", type=int, default=24)
    parser.add_argument(
        "--nn", action="store_true",
        help="Use neural network model instead of RuleBot for the CPU."
    )
    args = parser.parse_args()

    use_rulebot = not args.nn
    client = InteractiveClient(
        grid_h=args.grid_h,
        grid_w=args.grid_w,
        checkpoint_path=args.checkpoint,
        use_rule_bot=use_rulebot,
    )
    client.run()
