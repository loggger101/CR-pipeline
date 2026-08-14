"""Tests for the evolvable policy (src/models/policy.py).

The policy is the representation the genetic algorithm optimises. Its central
contract -- that the genome determines behaviour -- is what these tests pin
down, because the code path this replaced ignored the genome entirely.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.env.sim.engine import SimulationEngine
from src.models.policy import (
    DEFAULT_POLICY_SPEC, FEATURE_DIM, NUM_OUTPUTS, PolicySpec,
    encode_features, policy_forward,
)


def _genome(seed: int, spec: PolicySpec = DEFAULT_POLICY_SPEC) -> np.ndarray:
    return spec.random_genome(np.random.RandomState(seed))


class TestPolicySpec:
    """Shape and packing contract."""

    def test_num_params_matches_layer_sizes(self):
        spec = PolicySpec(feature_dim=8, hidden_dim=4, num_outputs=3)
        assert spec.num_params == 8 * 4 + 4 + 4 * 3 + 3

    def test_random_genome_has_exact_length(self):
        spec = DEFAULT_POLICY_SPEC
        assert _genome(0, spec).shape == (spec.num_params,)

    def test_unpack_returns_correctly_shaped_matrices(self):
        spec = PolicySpec(feature_dim=8, hidden_dim=4, num_outputs=3)
        w1, b1, w2, b2 = spec.unpack(np.arange(spec.num_params, dtype=float))
        assert w1.shape == (8, 4)
        assert b1.shape == (4,)
        assert w2.shape == (4, 3)
        assert b2.shape == (3,)

    def test_unpack_rejects_wrong_sized_genome(self):
        """A size mismatch must fail loudly, not silently misinterpret memory."""
        spec = DEFAULT_POLICY_SPEC
        with pytest.raises(ValueError, match="expected"):
            spec.unpack(np.zeros(spec.num_params + 1))

    def test_random_genome_is_seed_reproducible(self):
        assert np.array_equal(_genome(3), _genome(3))
        assert not np.array_equal(_genome(3), _genome(4))


class TestPolicyForward:
    """Forward-pass contract."""

    def test_output_shapes(self):
        eng = SimulationEngine(seed=1, record_replay=False)
        eng.reset()
        logits, placement = policy_forward(_genome(0), encode_features(eng))
        assert logits.shape == (5,)      # 4 hand slots + pass
        assert placement.shape == (2,)
        assert NUM_OUTPUTS == 7

    def test_placement_is_bounded(self):
        """Placement is squashed, so a large genome cannot escape the arena."""
        eng = SimulationEngine(seed=1, record_replay=False)
        eng.reset()
        huge = np.full(DEFAULT_POLICY_SPEC.num_params, 50.0)
        _, placement = policy_forward(huge, encode_features(eng))
        assert np.all(placement >= -1.0) and np.all(placement <= 1.0)

    def test_outputs_are_finite(self):
        eng = SimulationEngine(seed=1, record_replay=False)
        eng.reset()
        logits, placement = policy_forward(_genome(2), encode_features(eng))
        assert np.all(np.isfinite(logits))
        assert np.all(np.isfinite(placement))

    def test_different_genomes_give_different_outputs(self):
        """The regression that matters: behaviour must depend on the genome.

        The replaced implementation seeded an RNG from the tick counter and
        never read the weights, so every agent produced identical output and
        fitness could not distinguish them.
        """
        eng = SimulationEngine(seed=1, record_replay=False)
        eng.reset()
        features = encode_features(eng)
        outputs = [policy_forward(_genome(i), features)[0] for i in range(5)]
        distinct = {tuple(np.round(o, 6)) for o in outputs}
        assert len(distinct) == 5

    def test_same_genome_and_state_is_deterministic(self):
        eng = SimulationEngine(seed=1, record_replay=False)
        eng.reset()
        features = encode_features(eng)
        g = _genome(9)
        a_logits, a_place = policy_forward(g, features)
        b_logits, b_place = policy_forward(g, features)
        assert np.array_equal(a_logits, b_logits)
        assert np.array_equal(a_place, b_place)


class TestFeatureEncoding:
    """The observation the policy sees."""

    def test_shape_and_bounds(self):
        eng = SimulationEngine(seed=4, record_replay=False)
        eng.reset()
        feats = encode_features(eng, "player")
        assert feats.shape == (FEATURE_DIM,)
        assert np.all(np.isfinite(feats))
        assert np.all(np.abs(feats) <= 1.0 + 1e-9)

    def test_encodes_both_sides(self):
        eng = SimulationEngine(seed=4, record_replay=False)
        eng.reset()
        assert encode_features(eng, "opponent").shape == (FEATURE_DIM,)

    def test_features_track_elixir(self):
        eng = SimulationEngine(seed=4, record_replay=False)
        eng.reset()
        before = encode_features(eng, "player")[0]
        eng.player_elixir = 10.0
        after = encode_features(eng, "player")[0]
        assert after > before
        assert after == pytest.approx(1.0)

    def test_features_track_tower_damage(self):
        eng = SimulationEngine(seed=4, record_replay=False)
        eng.reset()
        before = encode_features(eng, "player")
        tower = next(t for t in eng.player_towers if not t.is_king)
        tower.take_damage(tower.max_hp * 0.5)
        after = encode_features(eng, "player")
        assert not np.array_equal(before, after)

    def test_sides_see_mirrored_boards(self):
        """One genome must be able to play either side, so 'forward' flips."""
        eng = SimulationEngine(seed=4, record_replay=False)
        eng.reset()
        # A lone player troop deep in its own half.
        eng._spawn_unit("knight", 3.0, 5.0, "player")

        own = encode_features(eng, "player")
        foe = encode_features(eng, "opponent")
        # The player sees it as its own troop; the opponent sees an enemy.
        assert not np.array_equal(own, foe)

    def test_troop_presence_changes_features(self):
        eng = SimulationEngine(seed=4, record_replay=False)
        eng.reset()
        empty = encode_features(eng, "player")
        eng._spawn_unit("knight", 3.0, 4.0, "player")
        occupied = encode_features(eng, "player")
        assert not np.array_equal(empty, occupied)
