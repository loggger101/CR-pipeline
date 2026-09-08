"""Evolvable policy: the mapping from game state to action that GA operates on.

This module exists because the evolutionary loop needs a genome that actually
*determines behaviour*. The parallel runner previously called an ``_infer_action``
stub that seeded an RNG from the tick counter and ignored the weight vector
entirely, so every agent in the population played identically and fitness was
pure noise -- evolution had nothing to select on.

Design constraints that shape what is here:

* **Pure NumPy.** Genomes are shipped to worker processes as plain arrays and
  evaluated thousands of times per match. Rebuilding a Torch module per worker
  (or per tick) would dominate runtime, and Torch modules pickle poorly.
* **Small enough for evolution, deep enough to play.** The genome is a flat
  vector over an N-hidden-layer tanh MLP: the default shape is 66 -> 96 -> 72
  -> 56 -> 40 -> 7 (20,071 parameters). That is ~2.2x more capacity than the
  previous three-layer net (9,207 params) and far below the 9.28M-parameter
  Torch genomes the old code path evolved -- none of which influenced play. The
  shape funnels wide-to-narrow: a broad first layer captures interactions among
  the 66 raw features, then narrows as it abstracts toward the action head.
* **Side-symmetric.** Features are encoded from the acting player's point of
  view (the arena is mirrored for the opponent), so one genome can play either
  side. Self-play depends on this.

The deep Torch architectures in ``architecture.py`` remain available for
architecture search, export, and ensembling; this is the representation the
genetic algorithm optimises.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

# Feature layout ─ keep in sync with `encode_features`.
NUM_GLOBAL_FEATURES = 8
NUM_HAND_SLOTS = 4
NUM_FEATURES_PER_SLOT = 5
NUM_TOWER_FEATURES = 6   # own L/R/king + enemy L/R/king HP ratios, explicit order
POOL_ROWS = 3
POOL_COLS = 4
NUM_ARENA_FEATURES = 2 * POOL_ROWS * POOL_COLS  # own + enemy troop density
NUM_LANE_FEATURES = 8

FEATURE_DIM = (
    NUM_GLOBAL_FEATURES
    + NUM_HAND_SLOTS * NUM_FEATURES_PER_SLOT
    + NUM_TOWER_FEATURES
    + NUM_ARENA_FEATURES
    + NUM_LANE_FEATURES
)

# 4 card logits + 1 "pass" logit + 2 placement coordinates.
NUM_OUTPUTS = 7


@dataclass(frozen=True)
class PolicySpec:
    """Shape of the evolvable policy network.

    A stack of ``hidden_dims`` tanh hidden layers between the feature vector
    and the output head (4 card logits + pass logit + 2 placement values).
    ``num_params`` is the exact length of the flat genome the GA evolves:

        sum over each layer L_k with fan-in in_k, width w_k:  in_k * w_k + w_k

    The default shape (66 -> 96 -> 72 -> 56 -> 40 -> 7) gives 20,071 parameters.
    """
    feature_dim: int = FEATURE_DIM
    hidden_dims: Tuple[int, ...] = (96, 72, 56, 40)
    num_outputs: int = NUM_OUTPUTS

    def __post_init__(self):
        if not self.hidden_dims or any(h <= 0 for h in self.hidden_dims):
            raise ValueError(
                f"hidden_dims must be a non-empty sequence of positive ints, "
                f"got {self.hidden_dims!r}")

    @property
    def num_params(self) -> int:
        total = 0
        fan_in = self.feature_dim
        for width in self.hidden_dims:
            total += fan_in * width + width   # W_k (fan_in x width) + b_k
            fan_in = width
        total += fan_in * self.num_outputs + self.num_outputs  # head layer
        return total

    def unpack(self, genome: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Split a flat genome into ``(W1, b1, W2, b2, ..., Wo, bo)`` views.

        Layer k has shape ``in_k x w_k`` where ``in_0 = feature_dim`` and
        ``in_k = hidden_dims[k-1]``; the final layer maps to ``num_outputs``.

        Raises:
            ValueError: if ``genome`` does not match ``num_params``.
        """
        genome = np.asarray(genome, dtype=np.float64).ravel()
        if genome.size != self.num_params:
            raise ValueError(
                f"genome has {genome.size} parameters, expected {self.num_params} "
                f"for {self}. Use PolicySpec.random_genome() to create one."
            )
        sizes = list(self.hidden_dims) + [self.num_outputs]
        parts: List[np.ndarray] = []
        fan_in = self.feature_dim
        i = 0
        for width in sizes:
            w = genome[i:i + fan_in * width].reshape(fan_in, width); i += fan_in * width
            b = genome[i:i + width];                   i += width
            parts.append(w)
            parts.append(b)
            fan_in = width
        return tuple(parts)

    def random_genome(self, rng: np.random.RandomState) -> np.ndarray:
        """Create a genome with per-layer Xavier-style initialisation.

        Scaling each layer by its own fan-in keeps activations in the useful
        range of ``tanh``; a flat scale over all parameters saturates or dies
        as depth and feature dimension grow. Biases start at zero so early
        layers are roughly symmetric (no dead units).
        """
        chunks: List[np.ndarray] = []
        fan_in = self.feature_dim
        for width in list(self.hidden_dims) + [self.num_outputs]:
            chunks.append(rng.randn(fan_in * width) * np.sqrt(1.0 / fan_in))
            chunks.append(np.zeros(width))
            fan_in = width
        return np.concatenate(chunks).astype(np.float64)


DEFAULT_POLICY_SPEC = PolicySpec()


def make_spec(hidden_dims: Optional[Sequence[int]] = None) -> PolicySpec:
    """Build a spec from an explicit hidden-layer list.

    ``None`` returns the default (96, 72, 56, 40); pass e.g. ``(128,)`` for a
    wide single layer or ``(128, 96, 72)`` to go wider in experiments. The list
    is read left-to-right: the first width sits next to the feature vector and
    the last feeds the action head.
    """
    if hidden_dims is None:
        return DEFAULT_POLICY_SPEC
    dims = tuple(int(h) for h in hidden_dims)
    return PolicySpec(feature_dim=FEATURE_DIM, hidden_dims=dims)


# Card registry, resolved on first use. It cannot be imported at module scope:
# ``src.env.sim`` imports the parallel runner, which imports this module, so a
# top-level import here would be a cycle. Caching it matters because the
# lookup sits on the per-tick path -- the import machinery alone was visible
# in match profiles.
_CARD_DEFS = None


def _card_defs():
    """The card registry, imported once and reused."""
    global _CARD_DEFS
    if _CARD_DEFS is None:
        from ..env.sim.entities import CARD_DEFS
        _CARD_DEFS = CARD_DEFS
    return _CARD_DEFS


def compile_genome(genome, spec: PolicySpec = DEFAULT_POLICY_SPEC):
    """Unpack a genome into layer arrays once, for reuse across ticks.

    A match runs the policy on every tick for both sides -- thousands of calls
    -- and re-slicing the flat genome each time is pure overhead, since the
    genome is fixed for the whole match. Callers that run a match should
    compile once and pass the result to :func:`policy_forward`.

    Args:
        genome: Flat parameter vector, ``None``, or an already-compiled tuple.
        spec: Network shape (only used when ``genome`` is a flat vector).

    Returns:
        ``(W1, b1, ..., Wo, bo)``, or ``None`` if ``genome`` was ``None``.
    """
    if genome is None:
        return None
    if isinstance(genome, tuple):
        return genome            # already compiled
    return spec.unpack(np.asarray(genome).ravel())


def _is_compiled(obj) -> bool:
    """A compiled policy is a non-empty tuple whose first entry is 2-D."""
    return (isinstance(obj, tuple) and len(obj) >= 4
            and isinstance(obj[0], np.ndarray) and obj[0].ndim == 2)


def _forward_layers(layers: Tuple[np.ndarray, ...], features: np.ndarray):
    """Run a compiled layer stack; returns the raw output vector."""
    x = features
    for k in range(0, len(layers), 2):
        w, b = layers[k], layers[k + 1]
        if k == len(layers) - 2:          # head is linear (logits)
            return x @ w + b
        x = np.tanh(x @ w + b)


def policy_forward(genome, features: np.ndarray,
                   spec: PolicySpec = DEFAULT_POLICY_SPEC
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Run the policy.

    Args:
        genome: Flat parameter vector of length ``spec.num_params``, or a
            compiled ``(W1, b1, ..., Wo, bo)`` tuple from :func:`compile_genome`
            (or any layer stack whose first matrix has shape matching
            ``features.size``).
        features: Feature vector of length ``spec.feature_dim``.
        spec: Network shape (only used for flat genomes).

    Returns:
        ``(card_logits, placement)`` where ``card_logits`` has length 5
        (4 hand slots + pass) and ``placement`` is 2 values in [-1, 1].
    """
    if _is_compiled(genome):
        out = _forward_layers(genome, features)
    else:
        layers = spec.unpack(np.asarray(genome).ravel())
        out = _forward_layers(layers, features)
    return out[:5], np.tanh(out[5:7])


def _card_kind(card_def) -> Tuple[float, float, float]:
    """One-hot over (unit, spell, building) for a card definition."""
    if card_def is None:
        return 0.0, 0.0, 0.0
    kind = getattr(card_def, "card_type", "unit")
    return (
        1.0 if kind == "unit" else 0.0,
        1.0 if kind == "spell" else 0.0,
        1.0 if kind == "building" else 0.0,
    )


def _tower_hp_ratio(t) -> float:
    """HP ratio of a tower (dead towers read as 0)."""
    return (t.hp / t.max_hp) if (t.is_alive and t.max_hp) else 0.0


def encode_features(engine, side: str = "player") -> np.ndarray:
    """Encode the engine state into the policy's input vector.

    Features are expressed from ``side``'s point of view: the arena is mirrored
    for the opponent so that "forward" (towards the enemy king) is always
    decreasing row. This lets a single genome play either side, which self-play
    and tournament evaluation both rely on.

    Layout (length ``FEATURE_DIM``):
      * 8 globals: elixir x3, time remaining, overtime flag, crown lead,
        own-king active, double-elixir-overtime-active;
      * 4 hand slots x5: ready, cost/10, unit/spell/building one-hot;
      * 6 tower HP ratios in explicit order (own L/R/king, enemy L/R/king);
      * 24 arena density (3x4 pools, own then enemy, squashed counts);
      * 8 lane summaries (L/R count + HP per side).

    Args:
        engine: A ``SimulationEngine``.
        side: ``"player"`` or ``"opponent"``.

    Returns:
        Float vector of length ``FEATURE_DIM``.
    """
    card_defs = _card_defs()

    is_player = side == "player"
    rows = engine.GRID_ROWS
    cols = engine.GRID_COLS

    if is_player:
        own_units, foe_units = engine.player_units, engine.opponent_units
        own_towers, foe_towers = engine.player_towers, engine.opponent_towers
        own_elixir, foe_elixir = engine.player_elixir, engine.opponent_elixir
        own_crowns, foe_crowns = engine.player_trophies, engine.opponent_trophies
        hand = engine.player_hand
        cooldowns = engine.player_cooldowns
    else:
        own_units, foe_units = engine.opponent_units, engine.player_units
        own_towers, foe_towers = engine.opponent_towers, engine.player_towers
        own_elixir, foe_elixir = engine.opponent_elixir, engine.player_elixir
        own_crowns, foe_crowns = engine.opponent_trophies, engine.player_trophies
        hand = engine.opponent_hand
        cooldowns = engine.opponent_cooldowns

    # Features are accumulated into a Python list and converted once. Writing
    # values one at a time into a NumPy array costs a setitem apiece, and this
    # runs on every tick of every match for both sides; batches keep it fast.
    elixir_max = engine.elixir_max
    total_ticks = engine.match_duration_ticks + (
        engine.overtime_ticks if engine.is_overtime else 0)
    crown_lead = (own_crowns - foe_crowns) / 3.0

    # ── Globals ─────────────────────────────────────────────────────────────
    feats: List[float] = [
        own_elixir / elixir_max,
        foe_elixir / elixir_max,
        (own_elixir - foe_elixir) / elixir_max,
        max(0.0, 1.0 - engine.tick / max(total_ticks, 1)),
        1.0 if engine.is_overtime else 0.0,
        # min/max rather than np.clip: on Python scalars np.clip goes through
        # the full array machinery and was ~10% of a match's runtime alone.
        -1.0 if crown_lead < -1.0 else (1.0 if crown_lead > 1.0 else crown_lead),
    ]

    # King activation state: whether own king is fighting matters for when to
    # commit troops; it flips exactly once per match, so a single bit suffices.
    own_king = next((t for t in own_towers if t.is_king), None)
    feats.append(1.0 if (own_king is not None and own_king.is_active) else 0.0)

    # Double-elixir-overtime-active: distinct from the plain overtime flag --
    # with double_elixir_overtime disabled, OT runs at normal rate and this is
    # 0 while overtime is 1. A full bar is worth more in real OT, so the net
    # elixir position combines with these two bits to price that difference.
    feats.append(1.0 if (engine.is_overtime and engine.double_elixir_overtime) else 0.0)

    # ── Hand: affordability, cost, and card kind per slot ───────────────────
    num_hand = len(hand)
    num_cooldowns = len(cooldowns)
    for slot in range(NUM_HAND_SLOTS):
        card_def = card_defs.get(hand[slot]) if slot < num_hand else None
        cost = getattr(card_def, "cost", 0.0) if card_def else 0.0
        ready = (slot < num_cooldowns and cooldowns[slot] <= 0
                 and card_def is not None and own_elixir >= cost)
        unit_f, spell_f, building_f = _card_kind(card_def)
        feats += (1.0 if ready else 0.0, cost / 10.0,
                  unit_f, spell_f, building_f)

    # ── Tower health in explicit order: own L/R/king then enemy L/R/king ────
    def _ordered(towers):
        """(left princess, right princess, king); missing entries read as 0."""
        left = next((t for t in towers if not t.is_king and t.col < engine.KING_COL), None)
        right = next((t for t in towers if not t.is_king and t.col >= engine.KING_COL), None)
        king = next((t for t in towers if t.is_king), None)
        return (left, right, king)

    own_l, own_r, own_k = _ordered(own_towers)
    foe_l, foe_r, foe_k = _ordered(foe_towers)
    feats += [_tower_hp_ratio(t) for t in (own_l, own_r, own_k)]
    feats += [_tower_hp_ratio(t) for t in (foe_l, foe_r, foe_k)]

    # ── Troop density and lane summaries, one pass per side ─────────────────
    # Pooling and the lane summaries both walk the same unit lists, so they
    # are gathered together and written out in feature order afterwards.
    mid = cols / 2.0
    pools: List[List[float]] = []
    lanes: List[Tuple[int, float, int, float]] = []
    for units in (own_units, foe_units):
        pool = [0.0] * (POOL_ROWS * POOL_COLS)
        left_count = right_count = 0
        left_hp = right_hp = 0.0
        for u in units:
            if not u.is_alive or u.is_building:
                continue
            col = u.col
            # Row in the acting side's frame (own base high, enemy base low).
            r = u.row if is_player else (rows - 1 - u.row)
            rb = min(POOL_ROWS - 1, max(0, int(r / rows * POOL_ROWS)))
            cb = min(POOL_COLS - 1, max(0, int(col / cols * POOL_COLS)))
            max_hp = u.max_hp
            pool[rb * POOL_COLS + cb] += (u.hp / max_hp if max_hp else 0.0)

            if 0.0 <= col < mid:
                left_count += 1
                left_hp += u.hp
            elif mid <= col < cols:
                right_count += 1
                right_hp += u.hp

        # Squash counts into [0, 1) so a big push cannot dominate the input.
        pools.append([v / (v + 1.0) for v in pool])
        lanes.append((left_count, left_hp, right_count, right_hp))

    feats += pools[0]
    feats += pools[1]
    for left_count, left_hp, right_count, right_hp in lanes:
        feats += (left_count / (left_count + 1.0),
                  left_hp / (left_hp + 2000.0),
                  right_count / (right_count + 1.0),
                  right_hp / (right_hp + 2000.0))

    return np.array(feats, dtype=np.float64)
