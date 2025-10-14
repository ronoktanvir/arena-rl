"""Advanced two-lane Clash Royale style environment (CRLiteEnv)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pygame
from gymnasium import Env, spaces

from scripted.baseline_opponents import get_opponent


@dataclass(frozen=True)
class Card:
    """Static definition of a card that can be drawn into a player's hand."""

    name: str
    cost: int
    hp: float
    dps: float
    speed: float
    attack_range: float
    target: str
    color: Tuple[int, int, int]


@dataclass
class Unit:
    """Runtime state for a spawned unit on a specific lane."""

    card: Card
    side: int  # 0 = player, 1 = opponent
    lane: int  # 0 = left, 1 = right
    position: float
    hp: float
    attack_cooldown: float = 0.0

    def advance(self, dt: float, lane_length: float) -> None:
        direction = 1.0 if self.side == 0 else -1.0
        self.position = float(np.clip(self.position + direction * self.card.speed * dt, 0.0, lane_length))

    @property
    def alive(self) -> bool:
        return self.hp > 0.0


@dataclass
class Tower:
    """Tower representation with combat statistics."""

    name: str
    side: int
    lane: Optional[int]
    position: float
    hp: float
    max_hp: float
    dps: float
    attack_range: float

    @property
    def alive(self) -> bool:
        return self.hp > 0.0


class CRLiteEnv(Env[np.ndarray, int]):
    """A simplified two-lane Clash Royale inspired environment."""

    metadata = {"render_fps": 10}

    _frame_size: Tuple[int, int] = (96, 96)
    _lane_length: float = 100.0
    _dt: float = 0.1
    _max_steps: int = 600
    _elixir_regen: float = 0.05
    _tower_max_hp: float = 1000.0

    def __init__(
        self,
        opponent_policy: str = "greedy",
        *,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        if render_mode not in (None, "rgb_array"):
            raise ValueError("render_mode must be None or 'rgb_array'.")
        self.render_mode = render_mode

        height, width = self._frame_size[1], self._frame_size[0]
        self.observation_space = spaces.Box(low=0, high=255, shape=(height, width, 3), dtype=np.uint8)
        self.action_space = spaces.Discrete(9)

        self._seed_value = seed
        self._rng = np.random.default_rng(seed)

        self._surface: Optional[pygame.Surface] = None
        self._font: Optional[pygame.font.Font] = None

        self._opponent_name = opponent_policy
        self._opponent_fn = get_opponent(opponent_policy)

        self._player_deck: List[Card] = []
        self._opponent_deck: List[Card] = []
        self._player_hand: List[Card] = []
        self._opponent_hand: List[Card] = []
        self._player_draw_index = 0
        self._opponent_draw_index = 0

        self.units: List[Unit] = []
        self.player_towers: List[Tower] = []
        self.opponent_towers: List[Tower] = []

        self.player_elixir: float = 5.0
        self.opponent_elixir: float = 5.0
        self.elapsed_steps: int = 0
        self._last_player_lane: Optional[int] = None

    # ------------------------------------------------------------------
    # Gym API
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._seed_value = seed
        if self._seed_value is not None:
            self._rng = np.random.default_rng(self._seed_value)
        else:
            self._rng = np.random.default_rng()

        self.elapsed_steps = 0
        self.player_elixir = 5.0
        self.opponent_elixir = 5.0
        self._last_player_lane = None

        self._player_deck = self._build_deck()
        self._opponent_deck = self._build_deck()
        self._rng.shuffle(self._player_deck)
        self._rng.shuffle(self._opponent_deck)

        self._player_hand = list(self._player_deck[:4])
        self._opponent_hand = list(self._opponent_deck[:4])
        self._player_draw_index = 4
        self._opponent_draw_index = 4

        self.units = []
        self.player_towers = self._build_towers(side=0)
        self.opponent_towers = self._build_towers(side=1)

        observation = self._render()
        info = self._build_info(result=None, win=False)
        return observation, info

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError("Action out of bounds for Discrete(9) space.")

        reward = -0.01  # baseline step penalty
        terminated = False
        truncated = False
        result: Optional[str] = None

        action_mask = self._compute_action_mask()
        if not action_mask[action]:
            reward -= 0.01  # invalid action penalty
            action = 0

        if action != 0:
            card_index = (action - 1) // 2
            lane = (action - 1) % 2
            self._play_card(side=0, card_index=card_index, lane=lane)
            self._last_player_lane = lane

        opponent_action = self._opponent_turn()
        if opponent_action is not None:
            opp_card_index, opp_lane = opponent_action
            self._play_card(side=1, card_index=opp_card_index, lane=opp_lane)

        prev_enemy_hp = self._total_tower_hp(self.opponent_towers)
        prev_player_hp = self._total_tower_hp(self.player_towers)

        self._advance_units()
        self._resolve_combat()

        self.player_elixir = float(np.clip(self.player_elixir + self._elixir_regen, 0.0, 10.0))
        self.opponent_elixir = float(np.clip(self.opponent_elixir + self._elixir_regen, 0.0, 10.0))

        enemy_hp = self._total_tower_hp(self.opponent_towers)
        player_hp = self._total_tower_hp(self.player_towers)
        reward += (prev_enemy_hp - enemy_hp) / 100.0
        reward -= (prev_player_hp - player_hp) / 100.0

        enemy_king_alive = any(t.name.endswith("king") and t.side == 1 and t.alive for t in self.opponent_towers)
        player_king_alive = any(t.name.endswith("king") and t.side == 0 and t.alive for t in self.player_towers)

        if not enemy_king_alive:
            reward += 5.0
            terminated = True
            result = "win"
        elif not player_king_alive:
            reward -= 5.0
            terminated = True
            result = "loss"

        self.elapsed_steps += 1
        if self.elapsed_steps >= self._max_steps and not terminated:
            truncated = True

        observation = self._render()
        info = self._build_info(result=result, win=(result == "win"))
        info["action_mask"] = self._compute_action_mask()

        return observation, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Decks, hands, and card play
    def _build_deck(self) -> List[Card]:
        palette = [
            (231, 76, 60),
            (46, 204, 113),
            (52, 152, 219),
            (155, 89, 182),
            (241, 196, 15),
            (230, 126, 34),
            (26, 188, 156),
            (149, 165, 166),
        ]
        costs = [2, 3, 3, 4, 4, 5, 2, 5]
        hps = [200, 240, 300, 450, 500, 600, 180, 650]
        dps_values = [55, 60, 70, 80, 90, 110, 50, 120]
        speeds = [12, 10, 11, 9, 8, 7, 13, 6]
        ranges = [1.0, 1.5, 3.0, 3.0, 1.0, 1.5, 3.0, 2.0]
        targets = ["ground", "ground", "any", "any", "ground", "ground", "any", "ground"]
        deck: List[Card] = []
        for idx in range(8):
            deck.append(
                Card(
                    name=f"card_{idx}",
                    cost=costs[idx],
                    hp=float(hps[idx]),
                    dps=float(dps_values[idx]),
                    speed=float(speeds[idx]),
                    attack_range=float(ranges[idx]),
                    target=targets[idx],
                    color=palette[idx],
                )
            )
        return deck

    def _build_towers(self, side: int) -> List[Tower]:
        towers: List[Tower] = []
        front = 20.0 if side == 0 else 80.0
        king_pos = 10.0 if side == 0 else 90.0
        towers.append(
            Tower(
                name=f"side{side}_left_tower",
                side=side,
                lane=0,
                position=front,
                hp=self._tower_max_hp,
                max_hp=self._tower_max_hp,
                dps=70.0,
                attack_range=6.0,
            )
        )
        towers.append(
            Tower(
                name=f"side{side}_right_tower",
                side=side,
                lane=1,
                position=front,
                hp=self._tower_max_hp,
                max_hp=self._tower_max_hp,
                dps=70.0,
                attack_range=6.0,
            )
        )
        towers.append(
            Tower(
                name=f"side{side}_king",
                side=side,
                lane=None,
                position=king_pos,
                hp=self._tower_max_hp,
                max_hp=self._tower_max_hp,
                dps=90.0,
                attack_range=8.0,
            )
        )
        return towers

    def _compute_action_mask(self) -> np.ndarray:
        mask = np.zeros(9, dtype=bool)
        mask[0] = True
        for idx, card in enumerate(self._player_hand):
            affordable = self.player_elixir >= card.cost
            mask[1 + idx * 2] = affordable
            mask[1 + idx * 2 + 1] = affordable
        return mask

    def _draw_next_card(self, side: int) -> Card:
        if side == 0:
            deck = self._player_deck
            index = self._player_draw_index
            card = deck[index % len(deck)]
            self._player_draw_index = (index + 1) % len(deck)
            return card
        deck = self._opponent_deck
        index = self._opponent_draw_index
        card = deck[index % len(deck)]
        self._opponent_draw_index = (index + 1) % len(deck)
        return card

    def _play_card(self, side: int, card_index: int, lane: int) -> None:
        hand = self._player_hand if side == 0 else self._opponent_hand
        elixir_attr = "player_elixir" if side == 0 else "opponent_elixir"
        if card_index < 0 or card_index >= len(hand):
            return
        card = hand[card_index]
        if getattr(self, elixir_attr) < card.cost:
            return

        setattr(self, elixir_attr, getattr(self, elixir_attr) - card.cost)

        start_pos = 0.0 if side == 0 else self._lane_length
        self.units.append(Unit(card=card, side=side, lane=lane, position=start_pos, hp=card.hp))

        next_card = self._draw_next_card(side)
        hand[card_index] = next_card

    def _opponent_turn(self) -> Optional[Tuple[int, int]]:
        affordable = [idx for idx, c in enumerate(self._opponent_hand) if self.opponent_elixir >= c.cost]
        lane_counts = {0: 0, 1: 0}
        for unit in self.units:
            if unit.side == 0:
                lane_counts[unit.lane] += 1
        state = {
            "step": self.elapsed_steps,
            "enemy_elixir": self.opponent_elixir,
            "affordable_idxs": affordable,
            "lane_load": {"L": lane_counts[0], "R": lane_counts[1]},
            "last_player_lane": None if self._last_player_lane is None else ("L" if self._last_player_lane == 0 else "R"),
            "hand_costs": [card.cost for card in self._opponent_hand],
            "rng": self._rng,
        }
        action = self._opponent_fn(state)
        if action is None:
            return None
        hand_idx, lane = action
        if hand_idx not in affordable:
            return None
        if isinstance(lane, str):
            lane = 0 if lane.upper().startswith("L") else 1
        lane = int(np.clip(int(lane), 0, 1))
        return hand_idx, lane

    # ------------------------------------------------------------------
    # Simulation updates
    def _advance_units(self) -> None:
        for unit in self.units:
            unit.advance(self._dt, self._lane_length)
        self.units = [unit for unit in self.units if unit.alive]

    def _resolve_combat(self) -> None:
        lane_units = {0: [], 1: []}
        for unit in self.units:
            lane_units[unit.lane].append(unit)

        for lane_list in lane_units.values():
            lane_list.sort(key=lambda u: u.position)

        # Units attack
        for lane, units in lane_units.items():
            player_units = [u for u in units if u.side == 0]
            opponent_units = [u for u in units if u.side == 1]
            for unit in player_units:
                target = self._select_unit_target(unit, opponent_units)
                if target is not None:
                    target.hp -= unit.card.dps * self._dt
                else:
                    self._attack_tower(unit, self.opponent_towers, lane)
            for unit in opponent_units:
                target = self._select_unit_target(unit, player_units)
                if target is not None:
                    target.hp -= unit.card.dps * self._dt
                else:
                    self._attack_tower(unit, self.player_towers, lane)

        # Towers attack
        for tower in self.player_towers:
            if not tower.alive:
                continue
            target = self._tower_target(tower, [u for u in self.units if u.side == 1])
            if target is not None:
                target.hp -= tower.dps * self._dt
        for tower in self.opponent_towers:
            if not tower.alive:
                continue
            target = self._tower_target(tower, [u for u in self.units if u.side == 0])
            if target is not None:
                target.hp -= tower.dps * self._dt

        self.units = [unit for unit in self.units if unit.alive]

    def _select_unit_target(self, unit: Unit, enemies: List[Unit]) -> Optional[Unit]:
        best_target: Optional[Unit] = None
        best_distance = float("inf")
        for enemy in enemies:
            distance = abs(enemy.position - unit.position)
            if distance <= unit.card.attack_range and distance < best_distance:
                best_distance = distance
                best_target = enemy
        return best_target

    def _attack_tower(self, unit: Unit, towers: List[Tower], lane: int) -> None:
        lane_tower = next((t for t in towers if t.lane == lane and t.alive), None)
        target = lane_tower
        if target is None:
            target = next((t for t in towers if t.lane is None and t.alive), None)
        if target is None:
            return
        distance = abs(target.position - unit.position)
        if distance <= unit.card.attack_range + 1.0:  # allow slight overlap fudge
            target.hp = max(0.0, target.hp - unit.card.dps * self._dt)

    def _tower_target(self, tower: Tower, enemies: List[Unit]) -> Optional[Unit]:
        best_target: Optional[Unit] = None
        best_distance = float("inf")
        for enemy in enemies:
            if tower.lane is not None and enemy.lane != tower.lane:
                continue
            distance = abs(enemy.position - tower.position)
            if distance <= tower.attack_range and distance < best_distance:
                best_distance = distance
                best_target = enemy
        return best_target

    def _total_tower_hp(self, towers: List[Tower]) -> float:
        return sum(tower.hp for tower in towers)

    # ------------------------------------------------------------------
    # Rendering and info helpers
    def _render(self) -> np.ndarray:
        width, height = self._frame_size
        if self._surface is None:
            pygame.init()
            self._surface = pygame.Surface((width, height))
            self._font = pygame.font.SysFont("Arial", 8)
        surface = self._surface
        surface.fill((30, 30, 40))

        mid_x = width // 2
        pygame.draw.line(surface, (80, 80, 90), (mid_x, 0), (mid_x, height), 2)

        # Draw lanes
        pygame.draw.rect(surface, (40, 60, 90), (0, height // 2 - 12, width // 2, 24))
        pygame.draw.rect(surface, (40, 90, 60), (mid_x, height // 2 - 12, width // 2, 24))

        # Draw towers
        def draw_tower(tower: Tower) -> None:
            x = int((tower.position / self._lane_length) * (width - 1))
            if tower.side == 1:
                x = width - 1 - x
            y = height // 2 - 20 if tower.lane == 0 else height // 2 + 4
            if tower.lane is None:
                y = height - 18 if tower.side == 0 else 6
            size = 6 if tower.lane is not None else 8
            hp_ratio = tower.hp / tower.max_hp
            color = (int(200 * hp_ratio), int(80 + 120 * hp_ratio), int(60)) if tower.side == 0 else (
                int(180 * hp_ratio), int(60), int(180 * hp_ratio)
            )
            pygame.draw.rect(surface, color, (x - size, y - size, size * 2, size * 2))

        for tower in self.player_towers + self.opponent_towers:
            draw_tower(tower)

        # Draw units
        for unit in self.units:
            x = int((unit.position / self._lane_length) * (width - 1))
            if unit.side == 1:
                x = width - 1 - x
            y = height // 2 - 10 if unit.lane == 0 else height // 2 + 10
            pygame.draw.circle(surface, unit.card.color, (x, y), 3)

        # Draw elixir bars
        def draw_elixir(value: float, y: int, color: Tuple[int, int, int]) -> None:
            bar_width = int((value / 10.0) * (width - 20))
            pygame.draw.rect(surface, color, (10, y, bar_width, 6))
            pygame.draw.rect(surface, (200, 200, 200), (10, y, width - 20, 6), 1)

        draw_elixir(self.player_elixir, height - 12, (120, 200, 250))
        draw_elixir(self.opponent_elixir, 6, (250, 140, 180))

        # Draw hand HUD
        hud_y = height - 20
        for idx, card in enumerate(self._player_hand):
            x = 6 + idx * 20
            pygame.draw.rect(surface, card.color, (x, hud_y, 16, 12))
            if self._font is not None:
                text = self._font.render(str(card.cost), True, (0, 0, 0))
                surface.blit(text, (x + 4, hud_y + 2))

        frame = pygame.surfarray.array3d(surface)
        frame = np.transpose(frame, (1, 0, 2))
        return frame.astype(np.uint8)

    def _build_info(self, *, result: Optional[str], win: bool) -> Dict[str, object]:
        tower_hp = {
            "player": {
                "left": next((t.hp for t in self.player_towers if t.lane == 0), 0.0),
                "right": next((t.hp for t in self.player_towers if t.lane == 1), 0.0),
                "king": next((t.hp for t in self.player_towers if t.lane is None), 0.0),
            },
            "opponent": {
                "left": next((t.hp for t in self.opponent_towers if t.lane == 0), 0.0),
                "right": next((t.hp for t in self.opponent_towers if t.lane == 1), 0.0),
                "king": next((t.hp for t in self.opponent_towers if t.lane is None), 0.0),
            },
        }
        return {
            "result": result,
            "win": win,
            "tower_hp": tower_hp,
            "action_mask": self._compute_action_mask(),
        }

    # ------------------------------------------------------------------
    def close(self) -> None:  # pragma: no cover - standard Env cleanup
        if self._surface is not None:
            pygame.quit()
            self._surface = None
            self._font = None


__all__ = ["CRLiteEnv", "Card", "Unit", "Tower"]
