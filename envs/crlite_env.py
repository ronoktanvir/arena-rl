"""Simplified Clash-Royale style environment for Arena RL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pygame
from gymnasium import Env, spaces

from scripted.baseline_opponents import OpponentAction, get_opponent


@dataclass(eq=False)
class Card:
    """Immutable unit template used in decks and hands."""

    name: str
    cost: int
    hp: float
    dps: float
    speed: float
    attack_range: float
    target: str
    color: Tuple[int, int, int]


@dataclass(eq=False)
class Unit:
    """Live unit marching down a lane."""

    card: Card
    side: int
    lane: int
    position: float
    hp: float

    def advance(self, *, step: float, lane_length: float) -> None:
        if self.side == 0:
            self.position = min(lane_length, self.position + self.card.speed * step)
        else:
            self.position = max(0.0, self.position - self.card.speed * step)


@dataclass(eq=False)
class Tower:
    """Tower with health and simple attack stats."""

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
    """Two-lane tower defense environment inspired by Clash Royale."""

    metadata = {"render_fps": 10}

    _frame_size: Tuple[int, int] = (96, 96)
    _lane_length: float = 100.0
    _spawn_offset: float = 12.0
    _dt: float = 0.1
    _elixir_regen: float = 0.05
    _max_steps: int = 600

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
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self._frame_size[1], self._frame_size[0], 3),
            dtype=np.uint8,
        )
        self.action_space = spaces.Discrete(9)

        self._rng = np.random.default_rng(seed)
        self._opponent_policy = opponent_policy
        self._opponent_fn = get_opponent(opponent_policy)

        self.player_elixir: float = 5.0
        self.opponent_elixir: float = 5.0
        self.elapsed_steps: int = 0

        self._player_deck: List[Card] = []
        self._opponent_deck: List[Card] = []
        self._player_draw_index: int = 0
        self._opponent_draw_index: int = 0
        self.player_hand: List[Card] = []
        self.opponent_hand: List[Card] = []
        self._last_player_play: Optional[Tuple[str, int]] = None

        self.player_towers: Dict[str, Tower] = {}
        self.opponent_towers: Dict[str, Tower] = {}
        self.units: List[Unit] = []

        self._surface: Optional[pygame.Surface] = None
        self._font: Optional[pygame.font.Font] = None
        self._ensure_pygame()

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
        _ = options  # Unused but part of Gymnasium signature.
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        else:
            self._rng = self.np_random

        self.player_elixir = 5.0
        self.opponent_elixir = 5.0
        self.elapsed_steps = 0
        self._last_player_play = None

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
        info = {"action_mask": self._action_mask(self.player_hand, self.player_elixir)}
        return observation, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        if not self.action_space.contains(action):
            raise ValueError("Action out of bounds.")

        reward = -0.01  # Step penalty
        terminated = False
        truncated = False

        player_mask = self._action_mask(self.player_hand, self.player_elixir)
        valid = player_mask[action]

        if valid and action != 0:
            card_index = (action - 1) // 2
            lane = (action - 1) % 2
            if card_index < len(self.player_hand):
                card = self.player_hand[card_index]
                if card.cost <= self.player_elixir:
                    self._play_player_card(card_index, lane)
                else:
                    valid = False
            else:
                valid = False

        if not valid:
            reward -= 0.01  # Invalid action penalty
            self._last_player_play = None

        opponent_decision = self._opponent_act()
        if opponent_decision is not None:
            opp_index, opp_lane = opponent_decision
            if opp_index < len(self.opponent_hand):
                self._play_opponent_card(opp_index, opp_lane)

        reward += self._simulate_combat()

        self.player_elixir = min(10.0, self.player_elixir + self._elixir_regen)
        self.opponent_elixir = min(10.0, self.opponent_elixir + self._elixir_regen)
        self.elapsed_steps += 1

        if not self.opponent_towers["king"].alive:
            reward += 5.0
            terminated = True
        if not self.player_towers["king"].alive:
            reward -= 5.0
            terminated = True
        if self.elapsed_steps >= self._max_steps:
            truncated = True

        observation = self._render_frame()
        info = {"action_mask": self._action_mask(self.player_hand, self.player_elixir)}
        return observation, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Card play helpers
    # ------------------------------------------------------------------
    def _play_player_card(self, card_index: int, lane: int) -> None:
        card = self.player_hand[card_index]
        self._spawn_unit(card, lane, side=0)
        self.player_elixir -= card.cost
        self._cycle_hand(self.player_hand, self._player_deck, card_index, is_player=True)
        self._last_player_play = (card.name, lane)

    def _play_opponent_card(self, card_index: int, lane: int) -> None:
        card = self.opponent_hand[card_index]
        self._spawn_unit(card, lane, side=1)
        self.opponent_elixir -= card.cost
        self._cycle_hand(self.opponent_hand, self._opponent_deck, card_index, is_player=False)

    def _cycle_hand(
        self,
        hand: List[Card],
        deck: Sequence[Card],
        card_index: int,
        *,
        is_player: bool,
    ) -> None:
        hand.pop(card_index)
        next_card = self._draw_next_card(deck, is_player=is_player)
        hand.append(next_card)

    def _draw_next_card(self, deck: Sequence[Card], *, is_player: bool) -> Card:
        if not deck:
            raise RuntimeError("Deck is empty.")
        if is_player:
            idx = self._player_draw_index
            self._player_draw_index = (self._player_draw_index + 1) % len(deck)
        else:
            idx = self._opponent_draw_index
            self._opponent_draw_index = (self._opponent_draw_index + 1) % len(deck)
        return deck[idx]

    def _spawn_unit(self, card: Card, lane: int, *, side: int) -> None:
        spawn = self._spawn_offset if side == 0 else self._lane_length - self._spawn_offset
        lane_clamped = lane if lane in (0, 1) else 0
        unit = Unit(card=card, side=side, lane=lane_clamped, position=spawn, hp=card.hp)
        self.units.append(unit)

    # ------------------------------------------------------------------
    # Combat simulation
    # ------------------------------------------------------------------
    def _simulate_combat(self) -> float:
        """Advance the simulation and compute reward from tower damage."""
        reward_delta = 0.0
        pending_damage: Dict[Unit, float] = {}

        for unit in self.units:
            unit.advance(step=self._dt, lane_length=self._lane_length)

        lane_player_units: Dict[int, List[Unit]] = {}
        lane_opponent_units: Dict[int, List[Unit]] = {}

        for lane in (0, 1):
            player_units = [u for u in self.units if u.side == 0 and u.lane == lane]
            opponent_units = [u for u in self.units if u.side == 1 and u.lane == lane]
            player_units.sort(key=lambda u: u.position)
            opponent_units.sort(key=lambda u: u.position)

            lane_player_units[lane] = player_units
            lane_opponent_units[lane] = opponent_units

            reward_delta += self._resolve_attacks(
                attackers=player_units,
                defenders=opponent_units,
                enemy_towers=self.opponent_towers,
                pending_damage=pending_damage,
                lane=lane,
                side=0,
            )
            reward_delta += self._resolve_attacks(
                attackers=opponent_units,
                defenders=player_units,
                enemy_towers=self.player_towers,
                pending_damage=pending_damage,
                lane=lane,
                side=1,
            )

        for lane in (0, 1):
            self._tower_attack(
                self.player_towers.get(f"lane_{lane}"),
                lane_opponent_units.get(lane, []),
                pending_damage,
            )
            self._tower_attack(
                self.opponent_towers.get(f"lane_{lane}"),
                lane_player_units.get(lane, []),
                pending_damage,
            )

        all_opponents = lane_opponent_units.get(0, []) + lane_opponent_units.get(1, [])
        all_players = lane_player_units.get(0, []) + lane_player_units.get(1, [])
        self._tower_attack(self.player_towers.get("king"), all_opponents, pending_damage)
        self._tower_attack(self.opponent_towers.get("king"), all_players, pending_damage)

        for unit, damage in pending_damage.items():
            unit.hp -= damage

        self.units = [u for u in self.units if u.hp > 0]
        return reward_delta

    def _resolve_attacks(
        self,
        *,
        attackers: Sequence[Unit],
        defenders: Sequence[Unit],
        enemy_towers: Dict[str, Tower],
        pending_damage: Dict[Unit, float],
        lane: int,
        side: int,
    ) -> float:
        """Resolve attacks for a group of units against enemy units and towers."""
        reward_delta = 0.0

        for unit in attackers:
            if unit.hp <= 0:
                continue
            damage = unit.card.dps * self._dt
            target = self._select_unit_target(unit, defenders)
            if target is not None:
                pending_damage[target] = pending_damage.get(target, 0.0) + damage
                continue

            target_tower = self._target_tower(enemy_towers, lane)
            if target_tower is None or not target_tower.alive:
                continue

            distance = (
                target_tower.position - unit.position
                if side == 0
                else unit.position - target_tower.position
            )
            if distance < 0 or distance > unit.card.attack_range:
                continue

            actual_damage = min(damage, target_tower.hp)
            target_tower.hp -= actual_damage
            if side == 0:
                reward_delta += actual_damage / 100.0
            else:
                reward_delta -= actual_damage / 100.0

        return reward_delta

    def _tower_attack(
        self,
        tower: Optional[Tower],
        enemies: Sequence[Unit],
        pending_damage: Dict[Unit, float],
    ) -> None:
        """Apply a single tower attack against the closest enemy in range."""
        if tower is None or not tower.alive or not enemies:
            return
        target = self._select_tower_target(tower, enemies)
        if target is None:
            return
        damage = tower.dps * self._dt
        pending_damage[target] = pending_damage.get(target, 0.0) + damage

    def _select_unit_target(
        self, attacker: Unit, enemies: Sequence[Unit]
    ) -> Optional[Unit]:
        """Return the nearest enemy in range ahead of the attacker, if any."""
        best: Optional[Unit] = None
        best_distance = float("inf")
        for enemy in enemies:
            if attacker.side == 0 and enemy.position < attacker.position:
                continue
            if attacker.side == 1 and enemy.position > attacker.position:
                continue
            distance = abs(enemy.position - attacker.position)
            if distance <= attacker.card.attack_range and distance < best_distance:
                best_distance = distance
                best = enemy
        return best

    def _select_tower_target(
        self, tower: Tower, enemies: Sequence[Unit]
    ) -> Optional[Unit]:
        """Return the closest enemy unit that the tower can attack."""
        best: Optional[Unit] = None
        best_distance = float("inf")
        for unit in enemies:
            if tower.side == 0 and unit.position < tower.position:
                continue
            if tower.side == 1 and unit.position > tower.position:
                continue
            if tower.lane is not None and unit.lane != tower.lane:
                continue
            distance = abs(unit.position - tower.position)
            if distance <= tower.attack_range and distance < best_distance:
                best_distance = distance
                best = unit
        return best

    def _target_tower(self, towers: Dict[str, Tower], lane: int) -> Optional[Tower]:
        """Select the appropriate enemy tower target for the given lane."""
        lane_key = f"lane_{lane}"
        lane_tower = towers.get(lane_key)
        if lane_tower is not None and lane_tower.alive:
            return lane_tower
        return towers.get("king")

    # ------------------------------------------------------------------
    # Opponent behaviour
    # ------------------------------------------------------------------
    def _opponent_act(self) -> OpponentAction:
        """Return the opponent's chosen (card_index, lane) action if any."""

        affordable_indices = [
            idx
            for idx, card in enumerate(self.opponent_hand[:4])
            if card.cost <= self.opponent_elixir
        ]
        state = {
            "step": self.elapsed_steps,
            "elixir": self.opponent_elixir,
            "affordable_indices": affordable_indices,
            "hand_costs": [card.cost for card in self.opponent_hand[:4]],
            "lane_load": [
                sum(1 for u in self.units if u.side == 0 and u.lane == lane)
                for lane in (0, 1)
            ],
            "last_player_lane": None if self._last_player_play is None else self._last_player_play[1],
            "rng": self._rng,
        }

        decision = self._opponent_fn(state)
        if decision is None:
            return None

        index, lane = decision
        if index not in affordable_indices:
            return None
        if lane not in (0, 1):
            return None
        return index, lane

    # ------------------------------------------------------------------
    # Environment construction helpers
    # ------------------------------------------------------------------
    def _initial_deck(self) -> List[Card]:
        """Create the ordered list of cards used to construct decks."""
        palette = [
            (220, 80, 80),
            (80, 140, 220),
            (120, 200, 120),
            (200, 160, 60),
            (160, 120, 200),
            (200, 200, 200),
            (240, 120, 180),
            (120, 220, 220),
        ]
        cards = [
            Card("Knight", 3, 320.0, 60.0, 16.0, 8.0, "any", palette[0]),
            Card("Archers", 3, 180.0, 50.0, 18.0, 28.0, "any", palette[1]),
            Card("Giant", 5, 900.0, 55.0, 12.0, 10.0, "tower", palette[2]),
            Card("Rogue", 2, 200.0, 70.0, 26.0, 8.0, "any", palette[3]),
            Card("Musketeer", 4, 240.0, 80.0, 18.0, 32.0, "any", palette[4]),
            Card("Goblins", 2, 140.0, 45.0, 28.0, 6.0, "any", palette[5]),
            Card("Bomber", 3, 200.0, 85.0, 18.0, 12.0, "tower", palette[6]),
            Card("Sentinel", 4, 420.0, 65.0, 14.0, 10.0, "any", palette[7]),
        ]
        return cards

    def _create_towers(self, side: int) -> Dict[str, Tower]:
        """Instantiate tower objects for the requested side."""
        if side == 0:
            king_pos = 10.0
            lane_pos = 30.0
        else:
            king_pos = self._lane_length - 10.0
            lane_pos = self._lane_length - 30.0

        towers = {
            "king": Tower("king", side, None, king_pos, 1000.0, 1000.0, 70.0, 32.0),
            "lane_0": Tower("lane", side, 0, lane_pos, 1000.0, 1000.0, 60.0, 26.0),
            "lane_1": Tower("lane", side, 1, lane_pos, 1000.0, 1000.0, 60.0, 26.0),
        }
        return towers

    def _action_mask(self, hand: Sequence[Card], elixir: float) -> np.ndarray:
        """Build an action mask encoding which card plays are currently legal."""
        mask = np.zeros(9, dtype=bool)
        mask[0] = True
        for idx in range(4):
            if idx >= len(hand):
                continue
            affordable = hand[idx].cost <= elixir
            mask[idx * 2 + 1] = affordable
            mask[idx * 2 + 2] = affordable
        return mask

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------
    def _ensure_pygame(self) -> None:
        if not pygame.get_init():
            pygame.init()
        if self._surface is None:
            self._surface = pygame.Surface(self._frame_size)
        if self._font is None:
            self._font = pygame.font.Font(None, 12)

    def _render_frame(self) -> np.ndarray:
        """Render the current state to a uint8 RGB frame."""
        self._ensure_pygame()
        assert self._surface is not None
        surface = self._surface
        surface.fill((28, 40, 60))

        width, height = self._frame_size
        lane_height = 40
        lane_spacing = 8
        lane_top = 8
        lane_width = width // 2 - 14

        for lane in (0, 1):
            rect = pygame.Rect(
                8 + lane * (lane_width + lane_spacing),
                lane_top,
                lane_width,
                lane_height,
            )
            pygame.draw.rect(surface, (52, 90, 110), rect, border_radius=6)

        self._draw_towers(surface, lane_top, lane_height, lane_width, lane_spacing)
        self._draw_units(surface, lane_top, lane_height, lane_width, lane_spacing)
        self._draw_elixir(surface, (10, height - 10), self.player_elixir, label="P")
        self._draw_elixir(surface, (width - 24, height - 10), self.opponent_elixir, label="O")
        self._draw_hand(surface, (8, height - 20))

        frame = pygame.surfarray.array3d(surface)
        frame = np.transpose(frame, (1, 0, 2))
        return frame.astype(np.uint8)

    def _draw_towers(
        self,
        surface: pygame.Surface,
        lane_top: int,
        lane_height: int,
        lane_width: int,
        lane_spacing: int,
    ) -> None:
        for side, towers in ((0, self.player_towers), (1, self.opponent_towers)):
            for key, tower in towers.items():
                if not tower.alive:
                    continue
                lane_idx = 0 if key == "king" else tower.lane or 0
                base_x = 8 + lane_idx * (lane_width + lane_spacing) + lane_width // 2
                y = lane_top + lane_height + 6 if side == 0 else lane_top - 6
                size = 6 if key == "king" else 4
                color = (210, 210, 100) if side == 0 else (220, 130, 130)
                pygame.draw.rect(
                    surface,
                    color,
                    pygame.Rect(base_x - size, y - size, size * 2, size * 2),
                    border_radius=2,
                )

    def _draw_units(
        self,
        surface: pygame.Surface,
        lane_top: int,
        lane_height: int,
        lane_width: int,
        lane_spacing: int,
    ) -> None:
        for unit in self.units:
            lane_x = 8 + unit.lane * (lane_width + lane_spacing) + lane_width // 2
            progress = unit.position / self._lane_length
            y = lane_top + lane_height - int(progress * lane_height)
            y = max(lane_top + 4, min(lane_top + lane_height - 4, y))
            color = unit.card.color if unit.side == 0 else tuple(int(c * 0.7) for c in unit.card.color)
            pygame.draw.circle(surface, color, (lane_x, y), 3)

    def _draw_elixir(
        self,
        surface: pygame.Surface,
        origin: Tuple[int, int],
        value: float,
        *,
        label: str,
    ) -> None:
        assert self._font is not None
        total_height = 18
        bar_rect = pygame.Rect(origin[0], origin[1] - total_height, 12, total_height)
        pygame.draw.rect(surface, (18, 18, 24), bar_rect, border_radius=3)
        filled = int((value / 10.0) * total_height)
        if filled > 0:
            fill_rect = pygame.Rect(origin[0], origin[1] - filled, 12, filled)
            pygame.draw.rect(surface, (130, 90, 230), fill_rect, border_radius=3)
        text = self._font.render(label, True, (240, 240, 240))
        surface.blit(text, (origin[0] - 2, origin[1] - total_height - 10))

    def _draw_hand(self, surface: pygame.Surface, origin: Tuple[int, int]) -> None:
        x, y = origin
        for idx, card in enumerate(self.player_hand[:4]):
            rect = pygame.Rect(x + idx * 20, y, 18, 10)
            pygame.draw.rect(surface, card.color, rect, border_radius=2)

    # ------------------------------------------------------------------
    # Optional API helpers
    # ------------------------------------------------------------------
    def render(self) -> np.ndarray:
        return self._render_frame()


__all__ = ["CRLiteEnv"]
