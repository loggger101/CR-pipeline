"""Rendering utilities for the simulation engine.

Provides:
- Lightweight frame rendering for visualization
- Arena state to image conversion
- Unit/tower visualization
- Action preview rendering
"""

from __future__ import annotations

import numpy as np
from typing import Optional, List, Dict, Tuple

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


class SimulationRenderer:
    """Renders simulation state to images for visualization.

    Creates lightweight frames showing:
    - Arena layout with towers and units
    - Unit positions, health, and types
    - Spell effects (poison, stun zones)
    - Action previews
    """

    # Color palette
    COLORS = {
        "player_tower": (0, 100, 255),
        "opponent_tower": (255, 50, 50),
        "player_unit": (0, 150, 255),
        "opponent_unit": (255, 80, 80),
        "bridge": (150, 150, 150),
        "background": (30, 30, 30),
        "grid_line": (60, 60, 60),
        "poison_zone": (100, 200, 50),
        "stun_indicator": (255, 255, 0),
        "spell_effect": (255, 200, 50),
        "text": (255, 255, 255),
        "text_dark": (0, 0, 0),
    }

    def __init__(
        self,
        width: int = 512,
        height: int = 512,
        grid_cols: int = 8,
        grid_rows: int = 6,
    ):
        self.width = width
        self.height = height
        self.grid_cols = grid_cols
        self.grid_rows = grid_rows
        self.cell_width = width / grid_cols
        self.cell_height = height / grid_rows

    def render_arena(
        self,
        player_towers: List,
        opponent_towers: List,
        player_units: List,
        opponent_units: List,
        spell_effects: Optional[List[Dict]] = None,
        action_preview: Optional[Tuple[float, float, str]] = None,
    ) -> np.ndarray:
        """Render the arena state to an RGB image.

        Args:
            player_towers: List of player tower UnitState objects.
            opponent_towers: List of opponent tower UnitState objects.
            player_units: List of player unit UnitState objects.
            opponent_units: List of opponent unit UnitState objects.
            spell_effects: List of spell effect dicts {col, row, radius, type}.
            action_preview: (col, row, type) for preview rendering.

        Returns:
            RGB numpy array of shape (height, width, 3).
        """
        if not HAS_PIL:
            # Fallback: return a solid color array
            return np.full((self.height, self.width, 3), 30, dtype=np.uint8)

        img = Image.new("RGB", (self.width, self.height), self.COLORS["background"])
        draw = ImageDraw.Draw(img)

        # Draw grid
        self._draw_grid(draw)

        # Draw bridges
        self._draw_bridges(draw)

        # Draw spell effects
        if spell_effects:
            for effect in spell_effects:
                self._draw_spell_effect(draw, effect)

        # Draw action preview
        if action_preview:
            col, row, ptype = action_preview
            self._draw_action_preview(draw, col, row, ptype)

        # Draw towers
        for tower in player_towers:
            if tower.is_alive:
                self._draw_tower(draw, tower, self.COLORS["player_tower"])
        for tower in opponent_towers:
            if tower.is_alive:
                self._draw_tower(draw, tower, self.COLORS["opponent_tower"])

        # Draw units
        for unit in player_units:
            if unit.is_alive:
                self._draw_unit(draw, unit, self.COLORS["player_unit"])
        for unit in opponent_units:
            if unit.is_alive:
                self._draw_unit(draw, unit, self.COLORS["opponent_unit"])

        # Draw status indicators
        for unit in player_units + opponent_units:
            if unit.is_alive and hasattr(unit, 'status') and unit.status > 0:
                self._draw_status_indicator(draw, unit)

        return np.array(img)

    def _draw_grid(self, draw: ImageDraw.ImageDraw) -> None:
        """Draw the arena grid."""
        for i in range(self.grid_cols + 1):
            x = int(i * self.cell_width)
            draw.line([(x, 0), (x, self.height)], fill=self.COLORS["grid_line"], width=1)
        for j in range(self.grid_rows + 1):
            y = int(j * self.cell_height)
            draw.line([(0, y), (self.width, y)], fill=self.COLORS["grid_line"], width=1)

    def _draw_bridges(self, draw: ImageDraw.ImageDraw) -> None:
        """Draw the bridge areas."""
        bridge_rows = [2, 3]  # Bridge rows
        bridge_cols = [3, 4]   # Bridge columns
        
        for br in bridge_rows:
            for bc in bridge_cols:
                x1 = int(bc * self.cell_width)
                y1 = int(br * self.cell_height)
                x2 = int((bc + 1) * self.cell_width)
                y2 = int((br + 1) * self.cell_height)
                draw.rectangle([x1, y1, x2, y2], fill=self.COLORS["bridge"])

    def _draw_tower(self, draw: ImageDraw.ImageDraw, tower, color: Tuple) -> None:
        """Draw a tower on the arena."""
        cx = int((tower.col + 0.5) * self.cell_width)
        cy = int((tower.row + 0.5) * self.cell_height)
        radius = min(self.cell_width, self.cell_height) * 0.4

        # Tower base
        draw.ellipse(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            fill=color,
            outline=(255, 255, 255),
            width=2,
        )

        # HP bar
        hp_ratio = tower.hp / tower.max_hp if hasattr(tower, 'max_hp') and tower.max_hp > 0 else 0
        bar_width = int(radius * 2 * hp_ratio)
        bar_height = 4
        bar_x = cx - radius
        bar_y = cy + radius + 2
        draw.rectangle([bar_x, bar_y, bar_x + bar_width, bar_y + bar_height],
                       fill=(255, 0, 0) if hp_ratio < 0.3 else (0, 255, 0))

        # King tower indicator
        if hasattr(tower, 'is_building') and not tower.is_building:
            # Draw a crown symbol for king tower
            draw.ellipse(
                [cx - radius * 0.5, cy - radius * 0.5,
                 cx + radius * 0.5, cy + radius * 0.5],
                fill=(255, 215, 0),
            )

    def _draw_unit(self, draw: ImageDraw.ImageDraw, unit, color: Tuple) -> None:
        """Draw a unit on the arena."""
        cx = int((unit.col + 0.5) * self.cell_width)
        cy = int((unit.row + 0.5) * self.cell_height)
        radius = min(self.cell_width, self.cell_height) * 0.25

        # Unit circle
        draw.ellipse(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            fill=color,
            outline=(255, 255, 255),
            width=1,
        )

        # HP bar
        hp_ratio = unit.hp / unit.max_hp if hasattr(unit, 'max_hp') and unit.max_hp > 0 else 0
        bar_width = int(radius * 2 * hp_ratio)
        bar_height = 3
        bar_x = cx - radius
        bar_y = cy + radius + 1
        draw.rectangle([bar_x, bar_y, bar_x + bar_width, bar_y + bar_height],
                       fill=(255, 0, 0) if hp_ratio < 0.3 else (0, 255, 0))

    def _draw_spell_effect(self, draw: ImageDraw.ImageDraw, effect: Dict) -> None:
        """Draw a spell effect zone."""
        cx = int((effect.get("col", 0) + 0.5) * self.cell_width)
        cy = int((effect.get("row", 0) + 0.5) * self.cell_height)
        radius = int(effect.get("radius", 1.0) * self.cell_width)

        color = self.COLORS.get(effect.get("type", "damage"), self.COLORS["spell_effect"])
        alpha_color = tuple(int(c * 0.3) for c in color) + (128,)

        draw.ellipse(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            fill=color,
            outline=color,
            width=2,
        )

    def _draw_action_preview(self, draw: ImageDraw.ImageDraw, col: float,
                             row: float, ptype: str) -> None:
        """Draw an action preview marker."""
        cx = int((col + 0.5) * self.cell_width)
        cy = int((row + 0.5) * self.cell_height)
        radius = int(self.cell_width * 0.3)

        color = (0, 255, 0) if ptype == "unit" else (255, 255, 0)
        draw.ellipse(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            outline=color,
            width=2,
        )

    def _draw_status_indicator(self, draw: ImageDraw.ImageDraw, unit) -> None:
        """Draw a status effect indicator on a unit."""
        cx = int((unit.col + 0.5) * self.cell_width)
        cy = int((unit.row + 0.5) * self.cell_height)
        radius = int(min(self.cell_width, self.cell_height) * 0.35)

        status_colors = {
            1: self.COLORS["stun_indicator"],  # Stunned
            2: (100, 100, 255),  # Slowed
            3: self.COLORS["poison_zone"],  # Poisoned
        }

        status = getattr(unit, 'status', 0)
        if status in status_colors:
            color = status_colors[status]
            draw.ellipse(
                [cx - radius, cy - radius, cx + radius, cy + radius],
                outline=color,
                width=2,
            )

    def render_minimap(
        self,
        state: np.ndarray,
        resolution: int = 64,
    ) -> np.ndarray:
        """Render a minimap from the state tensor.

        Args:
            state: State tensor of shape (channels, H, W).
            resolution: Output resolution.

        Returns:
            RGB numpy array.
        """
        if not HAS_PIL:
            return np.full((resolution, resolution, 3), 30, dtype=np.uint8)

        # Extract key channels
        unit_heat = state[6] if state.shape[0] > 6 else np.zeros((6, 8))
        tower_health = state[7] if state.shape[0] > 7 else np.zeros((6, 8))

        # Create image
        img = Image.new("RGB", (resolution, resolution), (20, 20, 20))
        draw = ImageDraw.Draw(img)

        # Draw unit density
        h, w = unit_heat.shape
        for r in range(h):
            for c in range(w):
                val = unit_heat[r, c]
                x1 = int(c * resolution / w)
                y1 = int(r * resolution / h)
                x2 = int((c + 1) * resolution / w)
                y2 = int((r + 1) * resolution / h)
                intensity = int(val * 255)
                draw.rectangle([x1, y1, x2, y2], fill=(0, intensity, 0))

        # Draw tower health
        for r in range(h):
            for c in range(w):
                val = tower_health[r, c]
                x1 = int(c * resolution / w)
                y1 = int(r * resolution / h)
                x2 = int((c + 1) * resolution / w)
                y2 = int((r + 1) * resolution / h)
                intensity = int(val * 255)
                draw.rectangle([x1, y1, x2, y2], fill=(intensity, 0, 0))

        return np.array(img)
