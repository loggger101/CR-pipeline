"""Tests for continuing training from earlier work.

Two ways to carry on:

* **Resume a run** -- pick up a previous run exactly where it stopped, keeping
  the population, generation counter, hall of fame and ELO ratings.
* **Seed from agents** -- start a new run from specific agents you liked,
  wherever they came from.

Resume was previously broken end to end: the checkpoint could not even be read
on current PyTorch, and what little did load discarded the generation counter,
hall of fame and ratings while overwriting the caller's config.
"""

import json
import shutil
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import torch

from src.models.policy import DEFAULT_POLICY_SPEC
from src.models.population import Population
from src.serialization import load_agent_genome, load_checkpoint
from src.train.trainer import EvolutionTrainer, TrainingConfig


def _config(run_dir, **overrides):
    params = dict(
        population_size=8, elite_count=2, max_generations=4, num_workers=2,
        match_duration="short", runs_dir=run_dir, seed=3,
        checkpoint_interval=2, curriculum_learning=False,
        diversity_preservation=False, tournament_matches=2,
        hall_of_fame_size=3,
    )
    params.update(overrides)
    return TrainingConfig(**params)


@pytest.fixture(scope="module")
def trained_run():
    """A finished 4-generation run, shared read-only by the tests below.

    Training it once keeps the suite quick. Any test that trains *into* a run
    must use ``resumable_run`` instead, which hands out a private copy --
    resuming writes new checkpoints, and sharing one directory would let each
    test see the previous test's extra generations.
    """
    tmp = tempfile.mkdtemp()
    run = os.path.join(tmp, "run1")
    os.makedirs(run, exist_ok=True)
    with EvolutionTrainer(_config(run)) as trainer:
        trainer.train()
        summary = {
            "run": run,
            "root": tmp,
            "generations": len(trainer.population.fitness_history["best"]),
            "best_fitness": trainer.best_fitness,
            "hall_of_fame": len(trainer.hall_of_fame),
            "elo_entries": len(trainer.elo_ratings),
            "agent": os.path.join(run, "best", "best_agent.pt"),
        }
    return summary


@pytest.fixture
def resumable_run(trained_run):
    """A private copy of the trained run, safe to train into."""
    with tempfile.TemporaryDirectory() as tmp:
        target = os.path.join(tmp, "run1")
        shutil.copytree(trained_run["run"], target)
        yield dict(trained_run, run=target,
                   agent=os.path.join(target, "best", "best_agent.pt"))


class TestCheckpointLoading:
    """The checkpoint files must actually be readable."""

    def test_population_checkpoint_round_trips(self):
        """torch.load defaults to weights_only=True since 2.6, which rejects
        the NumPy arrays in these files -- resume raised UnpicklingError."""
        population = Population(population_size=4, elite_count=1)
        population.initialize(seed=1)
        original = population.get_population_weights()

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "population.pt")
            population.save_checkpoint(path)

            restored = Population(population_size=4, elite_count=1)
            restored.load_checkpoint(path)

        for before, after in zip(original, restored.get_population_weights()):
            assert np.array_equal(before, after)

    def test_load_checkpoint_helper_reads_numpy_payloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "x.pt")
            torch.save({"array": np.arange(5), "note": "hello"}, path)
            payload = load_checkpoint(path)
        assert np.array_equal(payload["array"], np.arange(5))
        assert payload["note"] == "hello"


class TestCheckpointDiscovery:

    def test_finds_latest_checkpoint_in_a_run_directory(self, trained_run):
        found = EvolutionTrainer.find_population_checkpoint(trained_run["run"])
        assert found is not None
        assert found.name == "population.pt"
        # checkpoint_interval=2 over 4 generations -> gen_0002 and gen_0004
        assert found.parent.name == "gen_0004"

    def test_accepts_a_generation_folder(self, trained_run):
        gen_dir = os.path.join(trained_run["run"], "gen_0002")
        found = EvolutionTrainer.find_population_checkpoint(gen_dir)
        assert found is not None and found.parent.name == "gen_0002"

    def test_accepts_the_file_itself(self, trained_run):
        direct = os.path.join(trained_run["run"], "gen_0002", "population.pt")
        found = EvolutionTrainer.find_population_checkpoint(direct)
        assert str(found) == direct

    def test_returns_none_when_there_is_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            assert EvolutionTrainer.find_population_checkpoint(tmp) is None


class TestResume:

    def test_continues_from_where_it_stopped(self, resumable_run):
        with EvolutionTrainer(_config(resumable_run["run"],
                                      resume_from=resumable_run["run"],
                                      additional_generations=2)) as trainer:
            trainer.train()
            history = trainer.population.fitness_history["best"]

        # 4 generations before, 2 more now.
        assert len(history) == resumable_run["generations"] + 2

    def test_additional_generations_extends_the_target(self, trained_run):
        with EvolutionTrainer(_config(trained_run["run"],
                                      resume_from=trained_run["run"],
                                      additional_generations=3)) as trainer:
            trainer._resume_from_checkpoint()
            assert trainer.config.max_generations == trainer.generation + 3

    def test_carries_the_hall_of_fame_and_ratings(self, trained_run):
        """Losing these would reset the benchmark that anchors ELO across
        generations, making the resumed run's ratings incomparable."""
        with EvolutionTrainer(_config(trained_run["run"],
                                      resume_from=trained_run["run"],
                                      additional_generations=1)) as trainer:
            trainer._resume_from_checkpoint()
            assert len(trainer.hall_of_fame) == trained_run["hall_of_fame"]
            assert len(trainer.elo_ratings) == trained_run["elo_entries"]
            assert trainer.best_fitness == pytest.approx(
                trained_run["best_fitness"])
            assert trainer.best_genome is not None

    def test_keeps_the_population_rather_than_starting_over(self, trained_run):
        checkpoint = EvolutionTrainer.find_population_checkpoint(
            trained_run["run"])
        saved = load_checkpoint(str(checkpoint))
        saved_first = np.asarray(saved["agents"][0]["weights"])

        with EvolutionTrainer(_config(trained_run["run"],
                                      resume_from=trained_run["run"],
                                      additional_generations=1)) as trainer:
            trainer._resume_from_checkpoint()
            loaded = trainer.population.get_population_weights()

        assert np.array_equal(loaded[0], saved_first)

    def test_does_not_discard_the_new_config(self, trained_run):
        """Resume used to overwrite the caller's config with the saved one, so
        asking for different settings had no effect."""
        with EvolutionTrainer(_config(trained_run["run"],
                                      resume_from=trained_run["run"],
                                      additional_generations=1,
                                      tournament_matches=7)) as trainer:
            trainer._resume_from_checkpoint()
            assert trainer.config.tournament_matches == 7

    def test_missing_checkpoint_is_reported_clearly(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = os.path.join(tmp, "empty")
            os.makedirs(run)
            with EvolutionTrainer(_config(run, resume_from=run)) as trainer:
                with pytest.raises(FileNotFoundError, match="No population"):
                    trainer.train()

    def test_nothing_left_to_do_is_reported_clearly(self, resumable_run):
        with EvolutionTrainer(_config(resumable_run["run"],
                                      resume_from=resumable_run["run"],
                                      max_generations=1)) as trainer:
            with pytest.raises(ValueError, match="already reached generation"):
                trainer.train()


class TestSeedFromAgents:

    def test_keeps_the_chosen_agents_intact(self, trained_run):
        """The point is to carry on from what was learned, so the seeds
        themselves must survive into the population unchanged."""
        seed = load_agent_genome(trained_run["agent"])
        with tempfile.TemporaryDirectory() as tmp:
            with EvolutionTrainer(_config(tmp, max_generations=1)) as trainer:
                trainer._seed_population_from_agents([trained_run["agent"]])
                genomes = trainer.population.get_population_weights()

        assert sum(1 for g in genomes if np.array_equal(g, seed)) == 1

    def test_fills_the_rest_with_variants(self, trained_run):
        """Copying the seed into every slot would leave selection nothing to
        choose between."""
        seed = load_agent_genome(trained_run["agent"])
        with tempfile.TemporaryDirectory() as tmp:
            with EvolutionTrainer(_config(tmp, max_generations=1)) as trainer:
                trainer._seed_population_from_agents([trained_run["agent"]])
                genomes = trainer.population.get_population_weights()

        assert len(genomes) == 8
        assert len({g.tobytes() for g in genomes}) == len(genomes)
        for genome in genomes:
            assert genome.shape == seed.shape
            # Variants stay in the seed's neighbourhood.
            assert np.linalg.norm(genome - seed) < 20

    def test_multiple_seeds_are_all_preserved(self, trained_run):
        second = os.path.join(trained_run["root"], "second.pt")
        other = DEFAULT_POLICY_SPEC.random_genome(np.random.RandomState(9))
        torch.save({"genome": other, "param_kind": "genome"}, second)

        with tempfile.TemporaryDirectory() as tmp:
            with EvolutionTrainer(_config(tmp, max_generations=1)) as trainer:
                trainer._seed_population_from_agents(
                    [trained_run["agent"], second])
                genomes = trainer.population.get_population_weights()

        first = load_agent_genome(trained_run["agent"])
        assert any(np.array_equal(g, first) for g in genomes)
        assert any(np.array_equal(g, other) for g in genomes)

    def test_training_runs_from_a_seeded_population(self, trained_run):
        with tempfile.TemporaryDirectory() as tmp:
            run = os.path.join(tmp, "seeded")
            os.makedirs(run)
            config = _config(run, max_generations=2,
                             seed_agents=[trained_run["agent"]])
            with EvolutionTrainer(config) as trainer:
                trainer.train()
                assert len(trainer.population.fitness_history["best"]) == 2
                assert trainer.best_genome is not None

    def test_rejects_a_torch_network_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "net.pt")
            torch.save({"weights": np.zeros(9_277_223, dtype=np.float32)}, path)
            with EvolutionTrainer(_config(tmp, max_generations=1)) as trainer:
                with pytest.raises(ValueError, match="Torch network"):
                    trainer._seed_population_from_agents([path])


class TestUiWiring:
    """The Train tab builds configs through build_training_config."""

    def _values(self, **overrides):
        from src.ui.operations import START_FRESH

        values = dict(
            population_size=8, max_generations=5, elite_count=2, num_workers=2,
            seed=1, tournament_format="swiss", tournament_matches=2,
            hall_of_fame_size=2, match_duration="short",
            opponent_type="balanced", matches_per_agent=2,
            tournament_mode=True, start_mode=START_FRESH,
            resume_from="", seed_agents=[],
        )
        values.update(overrides)
        return values

    def test_continue_mode_sets_additional_generations(self, trained_run):
        from src.ui.operations import START_CONTINUE, build_training_config

        config = build_training_config(
            self._values(start_mode=START_CONTINUE,
                         resume_from=trained_run["run"], max_generations=6),
            trained_run["run"])
        assert config.resume_from == trained_run["run"]
        assert config.additional_generations == 6

    def test_continue_requires_a_run_with_checkpoints(self):
        from src.ui.operations import START_CONTINUE, build_training_config

        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(ValueError, match="no saved population"):
                build_training_config(
                    self._values(start_mode=START_CONTINUE, resume_from=tmp),
                    tmp)

    def test_continue_requires_a_choice(self):
        from src.ui.operations import START_CONTINUE, build_training_config

        with pytest.raises(ValueError, match="choose a run"):
            build_training_config(
                self._values(start_mode=START_CONTINUE, resume_from=""), "x")

    def test_seed_mode_validates_the_agents(self, trained_run):
        from src.ui.operations import START_SEED, build_training_config

        config = build_training_config(
            self._values(start_mode=START_SEED,
                         seed_agents=[trained_run["agent"]]), "x")
        assert config.seed_agents == [trained_run["agent"]]
        assert config.resume_from is None

    def test_seed_mode_requires_an_agent(self):
        from src.ui.operations import START_SEED, build_training_config

        with pytest.raises(ValueError, match="at least one agent"):
            build_training_config(
                self._values(start_mode=START_SEED, seed_agents=[]), "x")

    def test_continuing_writes_back_to_the_same_run(self, trained_run):
        """Otherwise a run's history ends up split across directories."""
        from src.ui.operations import run_dir_for_resume

        assert run_dir_for_resume(trained_run["run"]) == trained_run["run"]
        gen_dir = os.path.join(trained_run["run"], "gen_0002")
        assert run_dir_for_resume(gen_dir) == trained_run["run"]
        pop_file = os.path.join(gen_dir, "population.pt")
        assert run_dir_for_resume(pop_file) == trained_run["run"]
