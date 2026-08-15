# CR-Pipeline — Session Handoff

## Headline

Training did not work. The suite was green at 211 tests, but the evolutionary
loop could not learn, for three independent reasons that all had to be fixed
before any of them mattered.

The loop now learns: mean population fitness rises measurably over generations,
and `tests/test_training_signal.py` asserts that property so it cannot regress
silently. Suite is at **279 passing**.

---

## The three blocking defects

### 1. `ParallelRunner.evaluate_population` crashed on every call

```python
self.pool.starmap(_run_matches,
                  [(t[1], t[2], t[3], t[4], t[5]) for t in tasks], ...)
```

`tasks` held 5-tuples, so `t[5]` raised `IndexError` — and the arguments were
misaligned anyway (`config` would have landed in `worker_id`). A bare
`except Exception` caught it and returned `fitness=0.0` for the entire
population. Every generation, every agent, scored exactly zero. Training ran to
completion and reported progress while selecting on a constant.

**Fixed:** pass `tasks` straight through (the tuple already matches
`_run_matches`' signature) and let failures propagate rather than flattening
them into zeros.

### 2. The genome never reached the game

`_infer_action(weights, state)` ignored `weights` entirely:

```python
rng = np.random.RandomState(hash(state.tick) % 2**31)
card_logits = rng.randn(5) * 0.1
```

Output depended only on the tick counter, so every agent in the population
played identically. Fitness carried no information about the genome.

**Fixed:** added `src/models/policy.py` — a 2,311-parameter NumPy policy
(64 features → 32 tanh hidden → 5 card logits + 2 placement) that the runner
evaluates directly. Features are encoded from the acting side's point of view,
mirroring the arena for the opponent, so one genome can play either side.

### 3. Evolution optimised parameters nothing read

`Population.initialize` created a 1000-element vector, stored it unused on the
agent, and `get_population_weights()` returned the agent's **9.28M-parameter
Torch network** instead. Selection, crossover, and mutation ran on ~7GB of
float traffic per generation at population 200 — none of which any match
consulted.

**Fixed:** `AgentRecord` now carries the policy `genome`, and all population
weight accessors use it. `EvolutionaryAgent` builds its Torch network lazily,
so population init went from ~0.87s per 20 agents to ~0.02s. The Torch
architectures remain for architecture search, export, and ensembling.

---

## Simulation correctness

The sim was also producing a degenerate game. Notable fixes:

| Defect | Effect |
|---|---|
| King tower not flagged as a building | The king **walked across the arena**; it also attacked twice per tick (once as a "unit", once as a tower) |
| King placed level with the princesses at col 3.5 | It sat between the bridges, so every match was a king rush ending in ~7% of regulation |
| King activation logic inverted | The king was always live; princess towers were the gated ones |
| Crowns only awarded for a king kill | Any match reaching time compared 0 to 0 — a guaranteed draw |
| `_cycle_hand` replaced all 4 slots every 3rd tick | A hand index had no stable meaning; unlearnable |
| Spells restricted to the caster's own half | Damage spells could never reach an enemy tower or push |
| `apply_status` compared `UnitStatus` members to bare ints | Every stun/slow speed modifier silently no-opped |
| Dead units never removed | 12 of 16 list entries were corpses after 126 ticks; scanned every tick |
| `60 * attack_speed` treated as a 60Hz clock | ~72 ticks (7s) between attacks |
| Movement stopped at 0.1 tiles | Ranged troops walked into melee before firing |
| Spell damage bypassed death handling | Tower kills by spell awarded nothing; damage booked as `max_hp` |
| `_process_death_effects` nested loop | Ran per side over the combined list; keyed on `elixir_golem`/`mini_golem`, neither in the registry |
| Spell air/ground filter read the *victim's* `can_target_*` | Coverage was effectively arbitrary |
| Lane pressure added each unit to both lanes | Recomputed the unit count; carried no positional information |
| Head-to-head computed both agents as the player side, then recorded the swapped block as if sides had changed | Tournament/ELO win attribution inverted in half the matches |
| Opponent AIs reseeded from entropy every tick | Fitness irreproducible |
| Every match in a worker reset to the same seed | N matches per agent measured as much as one |

### Card data

- Princess towers were 1400 HP / 70 dmg / 1.2s — far below card power, so one
  mid-cost troop could solo a tower. Now Level 11: 2534 HP / 109 dmg / 0.8s
  (king 4008 / 109 / 1.0). Match outcomes went from 100% king rush at ~13s to a
  realistic mix (≈⅓ reaching time, mean length ~150s).
- `spawn_count`/`spawned_unit` conflated "deploys as N copies" (Minions) with
  "splits into N on death" (Golem). Golem deployed as two Golem Minis and Lava
  Hound as a single Lava Pup, discarding the tank. Added explicit
  `death_spawn_count`/`death_spawned_unit`.
- Minions, Minion, Minion Horde, and Royal Ghost were flagged
  `target_ground=False` — none of them can hit ground in that state. Royal
  Ghost was also typed `AIR`, making it immune to every ground-only attacker.

---

---

## Round two: search quality, persistence, and baselines

With the loop learning, the next round addressed *how well* it learns and
whether the result can be kept. Suite is at **341 passing**.

### The trained agent was never saved

`EvolutionaryAgent.save_checkpoint` persisted `get_weights()` — the **Torch
network**, which the evolutionary path never touches and which is freshly
initialised. Training would finish, write `best_agent.pt`, and that file held a
random 9.28M-parameter network. Reloading gave a random agent, silently.

Checkpoints now carry the genome under `genome`, with `param_kind` recording
which parameter set is authoritative, and `network_weights` written only if a
network was ever built. `scripts/evaluate.py` rejects network-only checkpoints
explicitly rather than returning an unusable vector.

Related: `trainer.best_agent` held a **live reference** into the population.
`set_population_weights` rewrites `AgentRecord`s in place, so the "best agent"
silently became whichever genome later occupied that slot. The trainer now
snapshots `best_genome` on improvement.

### Rank selection was pointing backwards

```python
ranked_indices = np.argsort(fitnesses)[::-1]   # computed...
parent1 = r.choice(n, p=probs)                 # ...and never used
```

`probs` is indexed by *rank*, but the draw was returned as a *population
index*. Measured over 4000 draws on a worst-first population, the least fit
individual was selected 1079 times and the fittest 37 — selection pressure ran
in reverse. Fixed by mapping the draw back through `ranked_indices`.

Also in the GA:

- **`EvolutionStrategy` was entropy-seeded**, so `TrainingConfig.seed` had no
  effect on evolution whatsoever. Added `EvolutionConfig.seed`, plumbed the
  trainer's seed into it, and passed `rng` into mutation (which had been
  drawing from the operator's own unseeded stream).
- **Parent re-draw loops were unbounded.** With
  `tournament_size >= population_size` every tournament sees the whole
  population and returns the same winner deterministically, so
  `while second == first` could never terminate — a hang for small
  populations. All four selectors now use a bounded helper.
- **`AdaptiveMutation.mutate` omitted `rng`**, so a positional call bound the
  `RandomState` to `current_fitness` and died on a float comparison.

### Baselines were walkovers, and evaluation was unpaired

Untrained genomes beat every scripted opponent 75–100% of the time. The
opponents picked a card by cost and dropped it at a uniformly random column and
row — they never defended a lane or responded to a push.

Replaced with one heuristic core (`_heuristic_opponent_action`) plus per
personality `OpponentProfile`s. They answer threats that cross the river with a
counter chosen for staying power and damage per elixir (and required to be able
to hit air or ground as needed), place it between the threat and their towers,
bank elixir when unthreatened, push at the bridge into the weakest lane, and
answer clumps with a damage spell. Untrained win rates are now 25–43%.

Separately, **each agent was evaluated on its own match seeds**
(`seed + i * 1000`), so fitness differences were substantially draw luck. All
agents in a generation now share seeds and the same sequence of opponent decks
— common random numbers, which makes fitness comparison a paired test. The
shared seed advances per generation so the population is not graded repeatedly
on one fixed set of games.

Measured effect on mean-fitness improvement over 12 generations against the
now-competitive `balanced` opponent: **+0.32 → +1.32** after switching to
per-match deck variation under common random numbers.

---

---

## Round three: tournament matchmaking as the main training loop

Agents are now trained by playing each other. Fitness is a competitor's
standing in a Swiss tournament against the population plus the hall of fame.
Suite is at **377 passing**.

### The tournament system had never run

Every format ended with:

```python
stats.win_rate = stats.wins / max(1, stats.matches)
```

`win_rate` is a computed property with no setter, so this raised
`AttributeError` in the tail of `run_round_robin`, `run_single_elimination`,
`run_double_elimination` and `run_league` — before any ranking was returned.
Tournament mode could never have worked.

Alongside it, the scoring was wrong in ways that would have made rankings
meaningless even if it had run:

- `AgentTournamentStats.update_elo` derived "actual" from the ratings
  (`actual = 1 - expected`) and **ignored the result argument entirely**, so
  the update reduced to `k*(1 - 2*expected)`: ratings moved without ever
  consulting who won.
- `_update_elo_pair` divided by `wins + draws + 1`, which neither counts losses
  nor sums to 1 across the pair — a clean 2-0 sweep scored 0.67.
- `compute_composite_score` added `0.01 * avg_duration`. With durations in the
  hundreds of ticks that term outweighed every win, so ranking largely
  reflected which agent happened to play longer games.
- `TournamentEvolutionStrategy` selected parents only from **non-elite**
  indices, so top performers were copied forward but barred from passing on
  any genes.

### Swiss pairing makes it affordable

Round-robin is O(N²) and ran one matchup at a time. Added `run_swiss`:
`⌈log₂ N⌉` rounds, every entrant paired against someone on a similar score,
rematches avoided, byes scored as half a point, and each round dispatched
across the worker pool via the new `ParallelRunner.run_pairings`.

At 200 agents that is 800 matchups instead of 19,900. At 16 agents Swiss
reproduces the full round-robin ranking with rank correlation **+0.83** at ~5×
less compute.

### Relative fitness needed an absolute anchor

The subtle failure here: tournament fitness is a standing *within the current
field*. It can rise while nothing actually improves — the population drifts, or
cycles against its own meta, and every agent still looks better than its peers.

The first version of this hit exactly that. `test_trained_champion_beats_its_own_ancestor`
failed at 7W/12L: mean fitness climbed every generation while the champion was
no stronger than its own ancestor.

Two changes fixed it:

- **Hall of fame.** Past champions enter every tournament as non-reproducing
  benchmark opponents, giving successive generations a fixed reference. They
  keep the id they were admitted under, so a carried-over rating follows the
  genome rather than the slot.
- **Track the run's best agent by ELO, not by fitness.** Fitness from
  generation 2 and generation 9 are standings in different fields and are not
  comparable; ELO is, because the hall of fame is common to both.

Result over 10 generations x 16 agents (~100s):

| Check | Generation-0 champion | Trained champion |
|---|---|---|
| Head-to-head vs its own ancestor | — | **12W / 2D / 6L (60%)** |
| vs `balanced` (never seen in training) | 1W / 5L | **3W / 3L** |
| vs `defensive` | 1W / 5L | **3W / 3L** |
| vs `aggressive` | 2W / 4L | **4W / 2L** |

The transfer to scripted opponents matters: those were never part of the
training signal, so improvement against them is not the metric being gamed.

---

---

## Round four: a desktop app to drive it

`python scripts/crp_gui.py` (or `CR-Pipeline.bat`) opens a native Tk window
with four tabs: **Train** (configure, start/stop, live fitness/ELO chart),
**Watch** (an agent playing a match on the arena, with scrubbing and speed),
**Runs** (browse and compare past runs), **Agents** (play a saved agent against
the baselines or head-to-head against another). `packaging/build_exe.py`
freezes it into `dist/CR-Pipeline/CR-Pipeline.exe`.

Design notes worth keeping:

- Work runs on a thread and reports through a queue the UI drains on a timer,
  so widget access stays on the main thread and the window never blocks.
  Stopping is cooperative — the trainer checks between generations.
- `src/ui/operations.py` holds the pipeline actions with no Tk import, so the
  whole thing is testable headlessly; only the widget tests need a display.
- The arena is drawn as canvas items rather than going through
  `src/viz/rendering.py`, which returns numpy arrays and would have pulled in
  Pillow.
- The frozen entry point calls `multiprocessing.freeze_support()` first. Miss
  that and every pool worker re-executes the `.exe` and opens another window.

### Three more pipeline bugs the UI surfaced

Building a UI that actually exercises the pipeline turned up defects the test
suite had been stepping around:

- **Checkpoint saving crashed every run.** `TrainingConfig` passes strategy
  names as strings (`"tournament"`, `"blend"`, `"gaussian"`) but
  `EvolutionConfig` is typed for enums, so `get_config_summary()` raised
  `AttributeError` on `.name`. Default `checkpoint_interval` is 10, so any run
  reaching generation 10 died — every earlier test had used
  `checkpoint_interval=100` and never hit it.
- **Selection strategy was silently ignored.** Same root cause: the string
  never equalled any enum member, so `_init_selection` fell through to its
  `else` branch. Every trainer-driven run used *roulette* selection no matter
  what was configured. Fixed by coercing names to enums in
  `EvolutionConfig.__post_init__`, which now also rejects typos loudly.
- **Runs left no top-level record.** `fitness_history.json` and `metrics.json`
  were written only inside `gen_XXXX/` checkpoint folders, so a run directory
  had nothing to read at its root — the layout the README documents did not
  exist. The trainer now writes a run-level summary each generation.

---

---

## Round five: continuing training, and a readiness pass

### Continue from earlier work

Two ways, both in the Train tab's **Start from** control and in
`TrainingConfig`:

- **Continue a previous run** (`resume_from` + `additional_generations`) —
  picks up the population, generation counter, hall of fame and ELO ratings,
  and writes back into the same run folder.
- **Start from chosen agents** (`seed_agents`) — the picked genomes go into the
  population unchanged, remaining slots are filled with mutated copies
  (`seed_mutation_std`). Copying the seed into every slot would leave selection
  nothing to choose between.

`resume_from` accepts a run directory, a `gen_XXXX` folder, or a
`population.pt` file.

### Resume had never worked

- `Population.load_checkpoint` called `torch.load(path)`. PyTorch 2.6 flipped
  `weights_only` to True, and these checkpoints hold NumPy arrays, so loading
  raised `UnpicklingError` outright. **Ten call sites** across the trainer, CLI
  and scripts had the same defect; they now share
  `src/serialization.load_checkpoint`, which documents why full unpickling is
  correct for our own files.
- What little did load threw away the generation counter, hall of fame and ELO
  ratings, and replaced the caller's config with the saved one — so asking to
  continue for more generations was silently discarded. Trainer state is now
  written beside each population checkpoint and restored on resume.

### Readiness pass findings

Running the documented commands for real turned up three more:

- **`crp export` crashed** with `NameError: name 'np' is not defined` — numpy
  was never imported in `scripts/crp.py`. The documented export command had
  never worked.
- **`evaluate.py --opponent` was fiction**: its choices were
  `random, greedy, elite` ("elite" is not an opponent, and balanced/aggressive/
  defensive were missing), and the flag was ignored anyway because the code
  hardcoded its own list. It also called `evaluate_single`, which does not
  exist on `FitnessEvaluator`.
- **`EvolutionTrainer.__del__` raised** `AttributeError` whenever construction
  failed before `self.runner` was set, printing a traceback on every rejected
  config.

Everything else checked out: 35 entry points, all CLI subcommands, config
parsing, the 140-card registry, model export, and a full
train → save → continue → seed round trip.

---

---

## Round six: what two real runs revealed

Inspecting actual runs on disk (240 agents / Swiss, and 24 agents / round-robin
continued from an earlier run) surfaced four defects the whole test suite had
been stepping around, because they are about *what a run leaves behind* rather
than whether the code raises.

| Symptom on disk | Cause |
|---|---|
| `best_agent.pt` was **108 MB** for an 18 KB genome | `update_best()` called `get_weights()`, which builds the lazily-created 9.28M-parameter Torch network just to copy it; `save_checkpoint` then wrote both `network_weights` and `best_weights` at 37 MB each |
| An 11-generation run had **no checkpoint at all** | `checkpoint_interval` was 50 (the UI used `generations // 4`), so 19 minutes of training could not be continued |
| `training.log` was **zero bytes** in both runs | The module logger's level was never set, so it inherited the root's WARNING and dropped every INFO record |
| `metrics.json` said `population_size: 240` for a run training **24** agents | Resuming kept the requested size instead of the one the checkpoint actually holds |

Fixes: the genome is snapshotted directly and Torch parameters are only written
for agents that have no genome (108 MB → 58 KB); a checkpoint is always written
when training ends or is stopped, and the UI checkpoints every 10 generations;
the file handler sets its own level; and resume reconciles `population_size`
with the checkpoint, warning when they differ.

### Fitness cannot show progress in tournament mode

Both runs looked flat: mean fitness moved from 0.664 to 0.700 across 61
generations. That is not a training failure — it is arithmetic. Tournament
fitness is points per match inside a closed field, so its mean is pinned near
0.5 however strong the population becomes.

The real signal was already in the data: hall-of-fame champions' ratings
*decline* relative to the field as the population outgrows them (`hof_gen0`
1483 → 1441, `hof_gen1` 1464 → 1422). Progress snapshots now carry
`population_elo` and `hall_of_fame_elo`, and the Train tab charts ratings
rather than the flat fitness curve.

### A correction

My first reading of these runs was that Swiss (103 s/generation) was slower
than round-robin (28 s/generation) and therefore broken. It is not: the
round-robin run had 24 agents and the Swiss run had 240. The run directory's
`metrics.json` disagreed with its own `config.json` — which is what led to the
misreported-population fix above.

---

## Where to pick up

1. **The frozen build is heavy.** PyTorch dominates the bundle. If size
   matters, a CPU-only torch wheel would cut it substantially, and the app
   itself only needs torch for checkpoint I/O.
2. **Watch tab replays one match at a time.** Watching two saved agents play
   each other would reuse `play_match` with `opponent_genome`, which is already
   wired but not exposed in the UI.
3. **Fitness transfer is real but modest.** 10 generations x 16 agents is a
   small run; the numbers above are a smoke test, not a trained agent. Longer
   runs at larger population are the obvious next step now that the loop scales.
2. **Hall of fame keeps only the most recent champions.** A diverse archive
   (sampling across the whole run rather than a sliding window) resists cycling
   better; worth trying if agents start beating recent champions while losing
   to older ones.
3. **`configs/sim_game.yaml` is documentation, not configuration.** The engine
   hardcodes its layout and rules; the file matches the engine but nothing
   reads the tower section. Either wire it up or drop it.
4. **Status effects are single-slot** — applying a stun replaces an active
   poison. Real stacking would need a list of active effects.
5. **`OpponentProfile.min_play_gap` is nearly inert.** At the real elixir rate,
   affording a card takes ~84 ticks, so a 3–6 tick gap rarely binds. It only
   matters when the opponent has a large bank and cheap cards.
