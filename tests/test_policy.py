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
    compile_genome, encode_features, policy_forward,
)


def _genome(seed: int, spec: PolicySpec = DEFAULT_POLICY_SPEC) -> np.ndarray:
    return spec.random_genome(np.random.RandomState(seed))


class TestPolicySpec:
    """Shape and packing contract."""

    def test_num_params_matches_layer_sizes(self):
        spec = PolicySpec(feature_dim=8, hidden_dims=(4,), num_outputs=3)
        assert spec.num_params == 8 * 4 + 4 + 4 * 3 + 3

    def test_num_params_sums_all_layers_for_deep_net(self):
        """Each layer contributes fan_in*width + width; the head counts too."""
        spec = PolicySpec(feature_dim=10, hidden_dims=(7, 5), num_outputs=2)
        assert spec.num_params == (10 * 7 + 7) + (7 * 5 + 5) + (5 * 2 + 2)

    def test_default_spec_is_four_layer_funnel(self):
        """The evolved policy must be deeper than the old single layer.

        The current default funnels wide-to-narrow (96 -> 72 -> 56 -> 40) so a
        broad first layer captures interactions among the raw features before
        abstracting toward the action head."""
        assert DEFAULT_POLICY_SPEC.hidden_dims == (96, 72, 56, 40)
        # 66 -> 96 -> 72 -> 56 -> 40 -> 7.
        expected = ((66 * 96 + 96) + (96 * 72 + 72) + (72 * 56 + 56)
                    + (56 * 40 + 40) + (40 * 7 + 7))
        assert DEFAULT_POLICY_SPEC.num_params == expected

    def test_random_genome_has_exact_length(self):
        spec = DEFAULT_POLICY_SPEC
        assert _genome(0, spec).shape == (spec.num_params,)

    def test_unpack_returns_correctly_shaped_matrices(self):
        """A deep spec unpacks into one (W_k, b_k) pair per layer."""
        spec = PolicySpec(feature_dim=8, hidden_dims=(4, 3), num_outputs=2)
        layers = spec.unpack(np.arange(spec.num_params, dtype=float))
        assert len(layers) == 6          # W1,b1,W2,b2,W3,b3
        w1, b1, w2, b2, wo, bo = layers
        assert w1.shape == (8, 4) and b1.shape == (4,)
        assert w2.shape == (4, 3) and b2.shape == (3,)
        assert wo.shape == (3, 2) and bo.shape == (2,)

    def test_unpack_rejects_wrong_sized_genome(self):
        """A size mismatch must fail loudly, not silently misinterpret memory."""
        spec = DEFAULT_POLICY_SPEC
        with pytest.raises(ValueError, match="expected"):
            spec.unpack(np.zeros(spec.num_params + 1))

    def test_random_genome_is_seed_reproducible(self):
        assert np.array_equal(_genome(3), _genome(3))
        assert not np.array_equal(_genome(3), _genome(4))

    def test_empty_hidden_dims_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            PolicySpec(feature_dim=8, hidden_dims=(), num_outputs=2)

    def test_make_spec_builds_custom_depth_and_width(self):
        from src.models.policy import make_spec
        wide = make_spec((128,))
        assert wide.hidden_dims == (128,)
        deep = make_spec((96, 72, 56, 40, 32))   # one layer deeper than default
        assert deep.num_params > DEFAULT_POLICY_SPEC.num_params


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

    def test_king_active_flag_tracks_activation(self):
        """The king-active global flips when the king wakes."""
        eng = SimulationEngine(seed=4, record_replay=False)
        eng.reset()
        feats_off = encode_features(eng, "player")
        assert feats_off[6] == 0.0          # king dormant at match start

        king = next(t for t in eng.player_towers if t.is_king)
        king.take_damage(king.max_hp * 0.5)  # past the 25% wake threshold
        eng._activate_king("player")
        feats_on = encode_features(eng, "player")
        assert feats_on[6] == 1.0

    def test_double_elixir_ot_flag_tracks_overtime(self):
        """The double-elixir OT global is off in regulation and on in real OT."""
        eng = SimulationEngine(seed=4, record_replay=False)
        eng.reset()
        assert encode_features(eng, "player")[7] == 0.0

        eng.is_overtime = True
        feats_ot = encode_features(eng, "player")
        assert feats_ot[4] == 1.0           # plain overtime flag (index 4)
        assert feats_ot[7] == 1.0           # double-elixir OT (enabled by default)

        eng.double_elixir_overtime = False
        assert encode_features(eng, "player")[7] == 0.0


class TestFeatureEncodingDimension:
    """The feature vector must match what the policy consumes."""

    def test_feature_dim_matches_spec(self):
        from src.models.policy import DEFAULT_POLICY_SPEC
        eng = SimulationEngine(seed=4, record_replay=False)
        eng.reset()
        assert encode_features(eng).shape == (DEFAULT_POLICY_SPEC.feature_dim,)


class TestCompiledGenome:
    """A genome unpacked once must behave exactly like the flat vector.

    Match loops compile the genome before the first tick rather than
    re-slicing it on each of the several thousand that follow, so the two
    paths have to agree exactly -- not approximately, since a different action
    choice would change the match and therefore the fitness.
    """

    def test_compiled_forward_matches_flat_forward(self):
        eng = SimulationEngine(seed=4, record_replay=False)
        eng.reset()
        genome = _genome(11)
        features = encode_features(eng, "player")

        flat_logits, flat_placement = policy_forward(genome, features)
        compiled = compile_genome(genome)
        comp_logits, comp_placement = policy_forward(compiled, features)

        assert np.array_equal(flat_logits, comp_logits)
        assert np.array_equal(flat_placement, comp_placement)

    def test_compiling_twice_is_a_no_op(self):
        compiled = compile_genome(_genome(3))
        assert compile_genome(compiled) is compiled

    def test_none_compiles_to_none(self):
        assert compile_genome(None) is None

    def test_compiled_policy_plays_the_same_match(self):
        """The whole match, tick for tick, not just one forward pass."""
        from src.env.sim.parallel_runner import _policy_action

        genome = _genome(21)
        compiled = compile_genome(genome)

        flat_engine = SimulationEngine(seed=9, record_replay=False)
        flat_engine.reset()
        compiled_engine = SimulationEngine(seed=9, record_replay=False)
        compiled_engine.reset()

        for _ in range(400):
            if flat_engine.terminated or compiled_engine.terminated:
                break
            flat_action = _policy_action(genome, flat_engine, "player")
            compiled_action = _policy_action(compiled, compiled_engine, "player")
            assert flat_action.action_type == compiled_action.action_type
            assert flat_action.card_index == compiled_action.card_index
            flat_engine.step(flat_action, None)
            compiled_engine.step(compiled_action, None)

        assert flat_engine.tick == compiled_engine.tick
        assert flat_engine.player_trophies == compiled_engine.player_trophies
