"""CR-Lite: a compact Clash Royale inspired two-lane Gymnasium environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pygame
from gymnasium import Env, spaces

from scripted.baseline_opponents import OpponentAction, get_opponent


@dataclass(frozen=True)
class Card:
    """Immutable template describing a unit card."""

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
    """A spawned unit travelling down a lane."""

    card: Card
    side: int  # 0 = player, 1 = opponent
    lane: int  # 0 = left, 1 = right
    position: float
    hp: float

    def advance(self, *, dt: float, lane_length: float) -> None:
        if self.side == 0:
            self.position = min(lane_length, self.position + self.card.speed * dt)
        else:
            self.position = max(0.0, self.position - self.card.speed * dt)

    @property
    def alive(self) -> bool:
        return self.hp > 0


@dataclass
class Tower:
    """Tower with combat attributes."""

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
        return self.hp > 0


class CRLiteEnv(Env[np.ndarray, int]):
    """Two-lane tower defense arena with card-based deployment and elixir economy."""

    metadata = {"render_fps": 10}

    _frame_size: Tuple[int, int] = (96, 96)
    _lane_length: float = 100.0
    _dt: float = 0.1
    _max_steps: int = 600
    _elixir_regen: float = 0.05
    _tower_max_hp: float = 1000.0
    _lanes: Tuple[int, int] = (0, 1)

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
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(height, width, 3),
            dtype=np.uint8,
        )
        self.action_space = spaces.Discrete(9)

        self._rng = np.random.default_rng(seed)
        self._opponent_name = opponent_policy
        self._opponent_fn = get_opponent(opponent_policy)

        self._surface: Optional[pygame.Surface] = None
        self._font: Optional[pygame.font.Font] = None
        self._ensure_pygame()

        self.player_elixir: float = 5.0
        self.opponent_elixir: float = 5.0
        self.elapsed_steps: int = 0

        self._player_deck: List[Card] = []
        self._opponent_deck: List[Card] = []
        self._player_draw_index: int = 0
        self._opponent_draw_index: int = 0
        self.player_hand: List[Card] = []
        self.opponent_hand: List[Card] = []

        self.player_towers: Dict[str, Tower] = {}
        self.opponent_towers: Dict[str, Tower] = {}
        self.units: List[Unit] = []

        self._last_player_lane: Optional[int] = None

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        _ = options
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        else:
            self._rng = self.np_random

        self.player_elixir = 5.0
        self.opponent_elixir = 5.0
        self.elapsed_steps = 0
        self._last_player_lane = None

        self.player_towers = self._create_towers(side=0)
        self.opponent_towers = self._create_towers(side=1)
        self.units = []

        self._player_deck = self._initial_deck()
        self._opponent_deck = self._initial_deck()
        self._rng.shuffle(self._player_deck)
        self._rng.shuffle(self._opponent_deck)

        self.player_hand = self._player_deck[:4]
        self.opponent_hand = self._opponent_deck[:4]
        self._player_draw_index = 4 % len(self._player_deck)
        self._opponent_draw_index = 4 % len(self._opponent_deck)

        observation = self._render_frame()
        info = {
            "action_mask": self._action_mask(self.player_hand, self.player_elixir),
            "win": False,
            "result": None,
            "tower_hp": self._tower_hp_snapshot(),
        }
        return observation, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        if not self.action_space.contains(action):
            raise ValueError("Action outside of action space range.")

        previous_step = self.elapsed_steps
        self.elapsed_steps += 1
        reward = -0.01  # baseline step penalty
        terminated = False
        truncated = False
        result: Optional[str] = None

        mask = self._action_mask(self.player_hand, self.player_elixir)
        played_lane: Optional[int] = None

        if action != 0:
            hand_idx = (action - 1) // 2
            lane = (action - 1) % 2
            if (
                0 <= hand_idx < len(self.player_hand)
                and mask[action]
            ):
                self._play_card(
                    hand_idx=hand_idx,
                    lane=lane,
                    hand=self.player_hand,
                    deck=self._player_deck,
                    draw_index_attr="_player_draw_index",
                    side=0,
                )
                played_lane = lane
            else:
                reward -= 0.01
        self._last_player_lane = played_lane if played_lane is not None else self._last_player_lane

        opponent_action = self._opponent_step(previous_step)
        if opponent_action is not None:
            opp_idx, opp_lane = opponent_action
            if (
                0 <= opp_idx < len(self.opponent_hand)
                and opp_lane in (0, 1)
                and self.opponent_hand[opp_idx].cost <= self.opponent_elixir
            ):
                self._play_card(
                    hand_idx=opp_idx,
                    lane=opp_lane,
                    hand=self.opponent_hand,
                    deck=self._opponent_deck,
                    draw_index_attr="_opponent_draw_index",
                    side=1,
                )

        self.player_elixir = min(10.0, self.player_elixir + self._elixir_regen)
        self.opponent_elixir = min(10.0, self.opponent_elixir + self._elixir_regen)

        player_tower_hp_before = {k: t.hp for k, t in self.player_towers.items()}
        opponent_tower_hp_before = {k: t.hp for k, t in self.opponent_towers.items()}

        self._simulate_combat()

        for name, hp_before in opponent_tower_hp_before.items():
            hp_after = self.opponent_towers[name].hp
            dealt = max(0.0, hp_before - hp_after)
            if dealt > 0:
                reward += dealt / 100.0
        for name, hp_before in player_tower_hp_before.items():
            hp_after = self.player_towers[name].hp
            taken = max(0.0, hp_before - hp_after)
            if taken > 0:
                reward -= taken / 100.0

        if not self.opponent_towers["king"].alive:
            reward += 5.0
            terminated = True
            result = "win"
        elif not self.player_towers["king"].alive:
            reward -= 5.0
            terminated = True
            result = "loss"

        if not terminated and self.elapsed_steps >= self._max_steps:
            truncated = True

        observation = self._render_frame()
        info = {
            "action_mask": self._action_mask(self.player_hand, self.player_elixir),
            "win": result == "win",
            "result": result,
            "tower_hp": self._tower_hp_snapshot(),
        }
        return observation, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Core simulation helpers
    # ------------------------------------------------------------------
    def _initial_deck(self) -> List[Card]:
        colors = [
            (235, 99, 99),
            (99, 176, 235),
            (137, 235, 99),
            (235, 210, 99),
            (235, 140, 196),
            (167, 99, 235),
            (99, 235, 214),
            (235, 168, 99),
        ]
        templates = [
            ("Spearman", 2, 150.0, 35.0, 18.0, 3.0, "any"),
            ("Knight", 3, 300.0, 55.0, 14.0, 1.0, "any"),
            ("Archer", 3, 160.0, 45.0, 16.0, 3.0, "any"),
            ("Giant", 5, 600.0, 60.0, 10.0, 1.5, "tower"),
            ("Musketeer", 4, 220.0, 65.0, 15.0, 3.0, "any"),
            ("MiniPekka", 4, 350.0, 90.0, 16.0, 1.0, "any"),
            ("Bomber", 3, 180.0, 75.0, 15.0, 2.5, "any"),
            ("BabyDragon", 5, 420.0, 70.0, 14.0, 3.0, "any"),
        ]
        return [
            Card(
                name=name,
                cost=cost,
                hp=hp,
                dps=dps,
                speed=speed,
                attack_range=rng,
                target=target,
                color=colors[i % len(colors)],
            )
            for i, (name, cost, hp, dps, speed, rng, target) in enumerate(templates)
        ]

    def _create_towers(self, side: int) -> Dict[str, Tower]:
        lane_front = 24.0
        if side == 1:
            lane_front = self._lane_length - lane_front
        king_position = 0.0 if side == 0 else self._lane_length
        lane_positions = {
            "left": lane_front,
            "right": lane_front,
        }
        towers = {
            "left": Tower(
                name="left",
                side=side,
                lane=0,
                position=lane_positions["left"],
                hp=self._tower_max_hp,
                max_hp=self._tower_max_hp,
                dps=65.0,
                attack_range=14.0,
            ),
            "right": Tower(
                name="right",
                side=side,
                lane=1,
                position=lane_positions["right"],
                hp=self._tower_max_hp,
                max_hp=self._tower_max_hp,
                dps=65.0,
                attack_range=14.0,
            ),
            "king": Tower(
                name="king",
                side=side,
                lane=None,
                position=king_position,
                hp=self._tower_max_hp,
                max_hp=self._tower_max_hp,
                dps=85.0,
                attack_range=18.0,
            ),
        }
        return towers

    def _play_card(
        self,
        *,
        hand_idx: int,
        lane: int,
        hand: List[Card],
        deck: List[Card],
        draw_index_attr: str,
        side: int,
    ) -> None:
        card = hand[hand_idx]
        if side == 0:
            self.player_elixir -= card.cost
        else:
            self.opponent_elixir -= card.cost

        position = 0.0 if side == 0 else self._lane_length
        self.units.append(
            Unit(card=card, side=side, lane=lane, position=position, hp=card.hp)
        )

        draw_index = getattr(self, draw_index_attr)
        next_card = deck[draw_index]
        setattr(self, draw_index_attr, (draw_index + 1) % len(deck))
        hand[hand_idx] = next_card

    def _opponent_step(self, step: int) -> OpponentAction:
        affordable = [i for i, card in enumerate(self.opponent_hand) if card.cost <= self.opponent_elixir]
        state = {
            "step": step,
            "elixir": self.opponent_elixir,
            "affordable_indices": tuple(affordable),
            "hand_costs": tuple(card.cost for card in self.opponent_hand),
            "lane_load": tuple(self._lane_unit_count(side=1, lane=lane) for lane in self._lanes),
            "last_player_lane": self._last_player_lane,
            "rng": self._rng,
        }
        return self._opponent_fn(state)

    def _lane_unit_count(self, *, side: int, lane: int) -> int:
        return sum(1 for unit in self.units if unit.side == side and unit.lane == lane and unit.alive)

    def _simulate_combat(self) -> None:
        for unit in self.units:
            if unit.alive:
                unit.advance(dt=self._dt, lane_length=self._lane_length)

        for lane in self._lanes:
            player_units = [u for u in self.units if u.side == 0 and u.lane == lane and u.alive]
            opponent_units = [u for u in self.units if u.side == 1 and u.lane == lane and u.alive]

            for unit in player_units:
                self._unit_attack(
                    unit,
                    enemy_units=opponent_units,
                    enemy_towers=self._towers_for_lane(side=1, lane=lane),
                )

            for unit in opponent_units:
                self._unit_attack(
                    unit,
                    enemy_units=player_units,
                    enemy_towers=self._towers_for_lane(side=0, lane=lane),
                )

        self.units = [unit for unit in self.units if unit.alive]
        self._towers_attack()
        self.units = [unit for unit in self.units if unit.alive]

    def _unit_attack(
        self,
        unit: Unit,
        *,
        enemy_units: Sequence[Unit],
        enemy_towers: Sequence[Tower],
    ) -> None:
        damage = unit.card.dps * self._dt
        if unit.card.target != "tower":
            target_unit = self._closest_unit(unit.position, enemy_units)
            if target_unit is not None:
                distance = abs(target_unit.position - unit.position)
                if distance <= unit.card.attack_range:
                    target_unit.hp = max(0.0, target_unit.hp - damage)
                    return
        tower = self._closest_tower(unit.position, enemy_towers)
        if tower is not None:
            distance = abs(tower.position - unit.position)
            if distance <= unit.card.attack_range and tower.alive:
                tower.hp = max(0.0, tower.hp - damage)
        return

    def _closest_unit(self, position: float, units: Sequence[Unit]) -> Optional[Unit]:
        closest: Optional[Unit] = None
        min_distance = float("inf")
        for enemy in units:
            if not enemy.alive:
                continue
            distance = abs(enemy.position - position)
            if distance < min_distance:
                min_distance = distance
                closest = enemy
        return closest

    def _closest_tower(self, position: float, towers: Sequence[Tower]) -> Optional[Tower]:
        closest: Optional[Tower] = None
        min_distance = float("inf")
        for tower in towers:
            if not tower.alive:
                continue
            distance = abs(tower.position - position)
            if distance < min_distance:
                min_distance = distance
                closest = tower
        return closest

    def _towers_for_lane(self, *, side: int, lane: int) -> List[Tower]:
        towers = self.player_towers if side == 0 else self.opponent_towers
        lane_key = "left" if lane == 0 else "right"
        candidates = [towers[lane_key], towers["king"]]
        return [tower for tower in candidates if tower.alive]

    def _towers_attack(self) -> None:
        damage = lambda tower: tower.dps * self._dt
        for towers, enemy_side in ((self.player_towers, 1), (self.opponent_towers, 0)):
            for tower in towers.values():
                if not tower.alive:
                    continue
                enemy_units = [
                    unit
                    for unit in self.units
                    if unit.side == enemy_side
                    and unit.alive
                    and (tower.lane is None or unit.lane == tower.lane)
                ]
                target = self._closest_unit(tower.position, enemy_units)
                if target is None:
                    continue
                distance = abs(target.position - tower.position)
                if distance <= tower.attack_range:
                    target.hp = max(0.0, target.hp - damage(tower))

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------
    def _render_frame(self) -> np.ndarray:
        self._ensure_pygame()
        assert self._surface is not None

        surface = self._surface
        surface.fill((25, 32, 38))

        width, height = surface.get_size()
        lane_height = height // 2
        for lane in self._lanes:
            lane_rect = pygame.Rect(0, lane * lane_height, width, lane_height)
            pygame.draw.rect(surface, (40, 70, 80), lane_rect)
            pygame.draw.rect(surface, (30, 50, 60), lane_rect, 1)

        self._draw_towers(surface)
        self._draw_units(surface)
        self._draw_elixir(surface)
        self._draw_hand(surface)

        frame = pygame.surfarray.array3d(surface)
        frame = np.transpose(frame, (1, 0, 2))
        return np.ascontiguousarray(frame, dtype=np.uint8)

    def _draw_towers(self, surface: pygame.Surface) -> None:
        width, height = surface.get_size()
        lane_height = height // 2
        tower_size = (10, 14)

        def draw_tower(tower: Tower) -> None:
            if not tower.alive:
                color = (70, 30, 30) if tower.side == 0 else (30, 30, 70)
            else:
                color = (200, 80, 80) if tower.side == 1 else (80, 200, 120)
            if tower.lane is None:
                y = height // 2 - tower_size[1] // 2
            else:
                lane_top = tower.lane * lane_height
                y = lane_top + lane_height - tower_size[1] - 4 if tower.side == 0 else lane_top + 4
            x = int((tower.position / self._lane_length) * (width - tower_size[0]))
            rect = pygame.Rect(x, y, *tower_size)
            pygame.draw.rect(surface, color, rect)
            health_ratio = tower.hp / tower.max_hp
            bar_width = max(1, int(tower_size[0] * health_ratio))
            bar_color = (0, 200, 0) if tower.side == 0 else (200, 0, 0)
            bar_rect = pygame.Rect(x, y - 2 if tower.side == 0 else y + tower_size[1] + 1, bar_width, 2)
            pygame.draw.rect(surface, bar_color, bar_rect)

        for tower in self.player_towers.values():
            draw_tower(tower)
        for tower in self.opponent_towers.values():
            draw_tower(tower)

    def _draw_units(self, surface: pygame.Surface) -> None:
        width, height = surface.get_size()
        lane_height = height // 2
        for unit in self.units:
            if not unit.alive:
                continue
            x = int((unit.position / self._lane_length) * width)
            lane_top = unit.lane * lane_height
            y = lane_top + lane_height - 8 if unit.side == 0 else lane_top + 8
            radius = 3
            color = unit.card.color if unit.side == 0 else tuple(min(255, c + 40) for c in unit.card.color)
            pygame.draw.circle(surface, color, (x, y), radius)
            hp_ratio = max(0.0, unit.hp / unit.card.hp)
            hp_width = max(1, int(6 * hp_ratio))
            hp_color = (0, 255, 0) if unit.side == 0 else (255, 0, 0)
            bar_rect = pygame.Rect(x - 3, y - radius - 3, hp_width, 2)
            pygame.draw.rect(surface, hp_color, bar_rect)

    def _draw_elixir(self, surface: pygame.Surface) -> None:
        assert self._font is not None
        width, _ = surface.get_size()
        bar_width = width // 3
        def draw_bar(value: float, y: int, color: Tuple[int, int, int]) -> None:
            background = pygame.Rect(4, y, bar_width, 8)
            pygame.draw.rect(surface, (25, 25, 25), background)
            fill_width = int(bar_width * (value / 10.0))
            fill_rect = pygame.Rect(4, y, fill_width, 8)
            pygame.draw.rect(surface, color, fill_rect)
            text = self._font.render(f"{value:.1f}", True, (220, 220, 220))
            surface.blit(text, (4 + bar_width + 4, y - 1))

        draw_bar(self.player_elixir, surface.get_height() - 16, (100, 60, 220))
        draw_bar(self.opponent_elixir, 6, (220, 80, 80))

    def _draw_hand(self, surface: pygame.Surface) -> None:
        width, height = surface.get_size()
        card_width = width // 6
        spacing = 4
        y = height - 32
        for idx, card in enumerate(self.player_hand):
            x = spacing + idx * (card_width + spacing)
            rect = pygame.Rect(x, y, card_width, 24)
            pygame.draw.rect(surface, card.color, rect)
            pygame.draw.rect(surface, (20, 20, 20), rect, 1)
            cost_text = self._font.render(str(card.cost), True, (10, 10, 10))
            surface.blit(cost_text, (x + 4, y + 4))

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _action_mask(self, hand: Sequence[Card], elixir: float) -> np.ndarray:
        mask = np.zeros(9, dtype=bool)
        mask[0] = True
        for idx, card in enumerate(hand[:4]):
            allowed = card.cost <= elixir
            base = 1 + idx * 2
            mask[base] = allowed
            mask[base + 1] = allowed
        return mask

    def _tower_hp_snapshot(self) -> Dict[str, Dict[str, float]]:
        return {
            "player": {name: tower.hp for name, tower in self.player_towers.items()},
            "opponent": {name: tower.hp for name, tower in self.opponent_towers.items()},
        }

    def _ensure_pygame(self) -> None:
        if not pygame.get_init():
            pygame.init()
            pygame.font.init()
        if self._surface is None:
            self._surface = pygame.Surface(self._frame_size)
        if self._font is None:
            self._font = pygame.font.SysFont("arial", 10)


__all__ = ["CRLiteEnv"]
