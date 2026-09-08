# CR-Pipeline — Project State & History

## Headline

Version **0.3.0**. The pipeline trains Clash Royale AI agents via evolutionary
strategies with tournament matchmaking. The core loop is stable, the sim is
internally consistent, and the desktop app drives everything without a terminal.

Suite: **513 tests passing** across simulation engine, evolution strategies,
tournament system, checkpoint/resume, run artifacts, desktop UI, visualization,
monitoring, and integration.

---

## What this project does

| Layer | Module(s) | Purpose |
|-------|-----------|---------|
| **Simulation** | `src/env/sim/engine.py` | Tick-based (10 Hz) Clash Royale sim: 8×6 arena, king activation, crown scoring, overtime, card cycling, status effects, death spawns. |
| **Parallel runner** | `src/env/sim/parallel_runner.py` | Multiprocessing worker pool (`os.cpu_count() - 1`). Each worker runs matches with either scripted opponents or the evolved policy (pure NumPy). Common random numbers across agents per generation. |
| **Evolvable policy** | `src/models/policy.py` | 2,311-parameter NumPy MLP: 64 features → tanh(32) → 5 card logits + 2 placement coordinates. Side-symmetric (arena mirrored for opponent). Genome is the flat parameter vector; `compile_genome()` caches unpacked layers per match. |
| **Agent** | `src/models/agent.py` | Wraps a genome or a Torch network, handles action selection with exploration strategies (ε-greedy, Boltzmann, entropy-regularized), saves/loads checkpoints (`genome` + `param_kind`). Network is built lazily — never constructed for population init. |
| **Population** | `src/models/population.py` | Manages agent records, fitness ranking, diversity tracking, speciation. Weight accessors all route through the policy genome, not the Torch network. |
| **GA operators** | `src/models/evolution.py` | Selection (tournament, rank, roulette, tournament-elite), crossover (blend, single-point, uniform, arithmetic), mutation (Gaussian, uniform, adaptive). All selectors use bounded parent re-draw (`_MAX_DISTINCT_ATTEMPTS = 32`) to prevent hangs on small populations. |
| **Tournament** | `src/train/evaluator.py` | Swiss pairing (default), round-robin, single/double elimination, league. ELO tracking with K=32. Hall of fame carries past champions across generations as non-reproducing benchmarks. |
| **Trainer** | `src/train/trainer.py` | Orchestrates the evolution loop: population init → evaluate → evolve → checkpoint → repeat. Handles resume from run dir / gen folder / population file. Seeds chosen agents intact, fills remaining slots with mutated copies. Writes run-level metrics every generation (not just inside checkpoints). |
| **Architecture search** | `src/models/architecture_search.py` | Evolves network topologies across layer types, filter sizes, attention heads, etc. Keeps the evolved policy as the primary representation; architectures are for NAS/export/ensembling. |
| **Ensemble** | `src/models/ensemble.py` | Weight averaging (performance-weighted), geometric mean, stacking with meta-learner. Optimizes combination weights from tournament fitness. |
| **Desktop UI** | `src/ui/app.py` + `_tabs*` | Tkinter window with four tabs: Train (configure/start/stop training, live chart), Watch (replay on arena canvas with scrubbing/speed), Runs (browse past runs, compare fitness curves), Agents (play saved agent vs baselines or head-to-head). Work runs on a thread; UI drains events via `after(poll)`. |
| **CLI** | `scripts/crp.py` | 11 subcommands: `train`, `tournament`, `hpo`, `export`, `report`, `dashboard`, `compare`, `experiments`, `pipelines`, `search`, `benchmark`. |
| **Visualization** | `src/viz/dashboard.py` | Streamlit dashboard (8 tabs): Fitness, Statistics, Tournament, Comparison, Runs, Config, Monitoring, Card Meta. Plotly charts with smoothing and statistical significance testing. |
| **Monitoring** | `src/train/monitoring/` | Metrics collection, GPU/CPU resource monitoring, bottleneck detection. Optional during training. |
| **Alerting** | `src/alerting/__init__.py` | Convergence, bottleneck, fitness milestone, early-stop, GPU error alerts with template formatting and multiple channels (console/file). |
| **Registry** | `src/registry.py` | Model versioning registry: saves model metadata alongside checkpoints for later retrieval. |
| **Data pipeline** | `src/train/data_pipeline/` | Match collector + dataset builder for capturing training episodes for offline analysis. |
| **Config system** | `src/config/validation.py` + `generator/` | Schema-based validation, templating, environment variable substitution (`${VAR:-default}`), config inheritance via `_base`. |
| **Data augmentation** | `src/data/augmentation.py` | Deck composition variation, opponent strategy variation, game condition variation (elixir rate, timing). |
| **Deploy/Export** | `src/deploy/export.py` | Export evolved policy genomes to ONNX, TorchScript, NumPy (.npy), JSON, pickle. Includes benchmarking and compression helpers. |
| **Serialization** | `src/serialization.py` | Centralised `torch.load(..., weights_only=False)` for this project's own checkpoints (NumPy arrays + metadata). Also `load_agent_genome()` with length-based validation to reject Torch-network-only files. |
| **Card data** | `assets/card_data.json` | 140+ card definitions at Level 11 stats: HP, damage, speed, cost, targeting mode, death spawns. Tower defs included separately. |

---

## Evolution of the project (chronological fix history)

### Round 1 — The loop was blind (3 blocking defects)

| Defect | Effect | Fix |
|--------|--------|-----|
| `_run_matches` unpacked 5-tuples as `t[0]…t[4]` but the caller passed a 6th element (`config`) → `IndexError`. Bare `except Exception` swallowed it, returning `fitness=0.0` for every agent every generation. | Training ran to completion reporting progress while selecting on constant zeros. | Pass tasks straight through; let failures propagate instead of flattening them into zeros. |
| `_infer_action(weights, state)` ignored `weights`: output was `rng.randn(5) * 0.1`, seeded from the tick counter only. Every agent played identically. | Fitness carried no information about the genome. No learning possible. | Created `src/models/policy.py` — a compact NumPy policy that reads `genome` on every tick. Features are side-symmetric (arena mirrored for opponent) so one genome plays either side. |
| `Population.initialize()` created an arbitrary 1000-element vector and stored it unused; `get_population_weights()` returned the agent's **9.28M-parameter Torch network** instead. | Selection/crossover/mutation operated on ~7 GB of float traffic per generation (pop 200) — none of which any match consulted. | `AgentRecord` now carries the policy `genome`; all weight accessors use it. Torch network is built lazily (population init went from ~0.87 s / 20 agents to ~0.02 s). Architectures remain for NAS/export/ensembling. |

### Round 2 — Simulation correctness & baseline strength

| Defect | Effect | Fix |
|--------|--------|-----|
| King tower not flagged as `building` → king walked across the arena and attacked twice per tick (once as unit, once as tower). | Match chaos; guaranteed king rush in many cases. | Set `is_building=True`; king only fires when activated. |
| King placed at row 0 level with princesses instead of behind them. | Attackers met the king before any princess — every match ended ~7% in regulation by instant king kill. | Moved king to its own back row; princess towers sit closer to the river. |
| King activation logic inverted (king always live, princess gated). | Towers fired at wrong times; crown awards nonsensical. | Fixed: king activates on damage or losing a princess tower. |
| Crowns only awarded for a king kill. Matches reaching time = 0-0 draw. | No meaningful match outcomes except instant wins. | Crown-based scoring: 1 per princess, 3 for king; overtime if tied at regulation. |
| `_cycle_hand` replaced all four slots every third tick regardless of play. | Hand indices had no stable meaning — unlearnable. | Cycle only the played slot (push to back, draw front). |
| Spells restricted to caster's half. Damage spells could never reach enemy towers or pushes. | Spell cards useless; deck composition broken. | Spells target entire arena. |
| `apply_status` compared `UnitStatus` members to bare ints → every stun/slow silently no-opped. | Status effects did nothing. | Compare status flags correctly (`& UnitStatus.STUNTED`). |
| Dead units never removed from unit lists. 12 of 16 entries corpses after 126 ticks; scanned every tick. | Performance degraded over match length; stale data polluted logic. | Reclaim dead units each tick via `_reclaim_dead`. |
| `60 * attack_speed` treated as a 60 Hz clock → ~72 ticks (7 s) between attacks. | Units attacked far too slowly. | Use engine's `TICKS_PER_SECOND = 10`; interval = `attack_speed * TICKS_PER_SECOND`. |
| Movement stopped at 0.1 tiles threshold for ranged troops walking into melee before firing. | Ranged units closed distance then attacked point-blank instead of from range. | Fix stopping logic: stop at attack range, not a fixed epsilon. |
| Spell damage bypassed death handling → tower kills by spell awarded no crowns; damage booked as `max_hp`. | Spells could destroy towers without scoring. | Route spell damage through `_process_damage` which handles crown awards on death. |
| `_process_death_effects` nested loop keyed on `elixir_golem`/`mini_golem`, neither in registry. | Death spawns never triggered. | Key on actual card names from the registry; check both `death_spawn_count` and `death_spawned_unit`. |
| Spell air/ground filter read victim's `can_target_*` instead of caster's. | Coverage effectively arbitrary — spells hit wrong targets. | Read targeting mode from the spell card definition, not the target unit. |
| Lane pressure added each unit to both lanes regardless of position. | Recomputed count was meaningless; carried no positional info. | Track per-lane counts with actual column positions. |
| Head-to-head computed both agents as player side then recorded swapped block as if sides changed → ELO win attribution inverted half the time. | Rankings were noisy and sometimes backwards. | Record winner/loser explicitly from match result, not inferred from side swaps. |
| Opponent AIs reseeded from entropy every tick → fitness irreproducible. | Same agent scored differently across runs; no convergence. | Seed once at engine init; do not reseed per tick. |
| Every match in a worker reset to the same seed (no per-match variation). | N matches measured as much as one. | Advance seed per match: `seed + i * 1000`. |

**Card data fixes:**
- Princess towers were 1400 HP / 70 dmg — far below card power, so one mid-cost troop could solo a tower. Updated to Level 11 stats (2534 HP / 109 dmg). King: 4008 HP. Match outcomes shifted from 100% king rush at ~13 s to a realistic mix (~⅓ time-out, mean length ~150 s).
- `spawn_count`/`spawned_unit` conflated deploy-as-N (Minions) with death-split N (Golem). Added explicit `death_spawn_count` / `death_spawned_unit`. Golem deploys as two Golem Minis; Lava Hound deploys as one Lava Pup — the tank is preserved.
- Minions, Minion Horde, Royal Ghost flagged `target_ground=False`; Royal Ghost typed `AIR` making it immune to every ground attacker. Fixed targeting flags.

### Round 3 — Baselines & evaluation fairness

| Defect | Effect | Fix |
|--------|--------|-----|
| Scripted opponents picked a card by cost, dropped at random column/row — never defended or responded to pushes. Untrained genomes beat them 75–100% of the time. | Baselines were walkovers; training had nothing meaningful to learn against. | Replaced with one heuristic core (`_heuristic_opponent_action`) plus per-personality `OpponentProfile`s. They counter-threats that cross the river, bank elixir when unthreatened, push at weakest lane, answer clumps with spells. Untrained win rates now 25–43%. |
| Each agent evaluated on its own match seeds (`seed + i * 1000`). Fitness differences were substantially draw luck. | Selection sorted noise rather than signal. | All agents in a generation share the same seeds and opponent deck sequence (common random numbers). Seed advances per generation so population is never graded repeatedly on one fixed set of games. |

**Measured effect:** mean-fitness improvement over 12 generations vs `balanced` opponent: **+0.32 → +1.32**.

### Round 4 — Tournament matchmaking as the main loop

| Defect | Effect | Fix |
|--------|--------|-----|
| Every tournament format raised `AttributeError` on `stats.win_rate = …` (computed property with no setter) before returning any ranking. | Tournament mode never worked at all. | Fixed setter; return rankings in every format. |
| `update_elo` derived "actual" from ratings (`1 - expected`) and ignored the result argument → rating moved without consulting who won. | ELO was a function of pre-match ratings only, not outcomes. | Pass actual score (win/loss/draw) into update; compute expected correctly. |
| `_update_elo_pair` divided by `wins + draws + 1` — neither counted losses nor summed to 1 across the pair. A clean 2-0 sweep scored 0.67. | Rankings reflected arithmetic artifact more than play. | Use wins + draws + losses as denominator; normalise correctly. |
| `compute_composite_score` added `0.01 * avg_duration`. With durations in hundreds of ticks, that term outweighed every win. Ranking largely reflected which agent played longer games. | Duration dominated ranking instead of performance. | Remove duration term from composite score. Points per match + tower differential only. |
| `TournamentEvolutionStrategy` selected parents only from **non-elite** indices → top performers copied forward but barred from reproducing. | Worst agents bred the next generation; selection pressure inverted for elites. | Include all agents in parent pool; elitism preserves top N unchanged. |
| No hall of fame → each generation measured against its own current field only. Population could drift/cycle without anything actually improving. | Fitness rose while champion couldn't beat its own ancestor (`test_trained_champion_beats_its_own_ancestor` failed 7W/12L). | Past champions enter every tournament as non-reproducing benchmarks. Track best agent by ELO (comparable across generations) not fitness (not comparable between fields). |
| Round-robin O(N²), one matchup at a time → unaffordable for real population sizes. | Could only evaluate tiny populations. | Added `run_swiss`: ⌈log₂ N⌉ rounds, paired against similar-score opponents, byes scored as half points, dispatched across worker pool via new `ParallelRunner.run_pairings`. At 200 agents: 800 matchups vs 19,900 for round-robin. |

**Results over 10 generations × 16 agents (~100 s):**

| Check | Generation-0 champion | Trained champion |
|-------|----------------------|------------------|
| Head-to-head vs ancestor | — | **12W / 2D / 6L (60%)** |
| vs `balanced` (never seen in training) | 1W / 5L | **3W / 3L** |
| vs `defensive` | 1W / 5L | **3W / 3L** |
| vs `aggressive` | 2W / 4L | **4W / 2L** |

### Round 5 — Desktop app & persistence bugs it surfaced

- **Checkpoint saving crashed every run.** `TrainingConfig` passes strategy names as strings (`"tournament"`, `"blend"`) but `EvolutionConfig.__post_init__` expected enums → `AttributeError: 'str' has no attribute '.name'`. Default `checkpoint_interval` is 10, so any run reaching gen 10 died. Fixed by coercing string names to enum members in `__post_init__`, which now also rejects typos loudly with the field name and valid options listed.
- **Selection strategy silently ignored.** Same root cause: string never matched enum member → fell through to default (roulette). Every trainer-driven run used roulette regardless of config. Fixed by the same coercion path.
- **Runs left no top-level record.** `fitness_history.json` and `metrics.json` written only inside `gen_XXXX/` checkpoint folders, so a run directory had nothing at its root — the layout README documented did not exist. Trainer now writes a run-level summary each generation.

### Round 6 — Continuing training & readiness pass

**Resume never worked:**
- `Population.load_checkpoint()` called bare `torch.load(path)`. PyTorch 2.6 flipped `weights_only` to True; these checkpoints hold NumPy arrays → `UnpicklingError`. Ten call sites had the same defect; all now share `src/serialization.load_checkpoint` with documented rationale.
- What little did load threw away generation counter, hall of fame and ELO ratings, replacing caller's config with saved one → asking to continue for more generations silently discarded. Trainer state written beside each population checkpoint and restored on resume.

**Readiness pass:**
- **`crp export` crashed** — `NameError: name 'np' is not defined`. numpy never imported in `scripts/crp.py`. Documented command had never worked. Fixed import.
- **`evaluate.py --opponent` was fiction**: choices were `random, greedy, elite` ("elite" isn't an opponent; balanced/aggressive/defensive missing); flag ignored because code hardcoded its own list. Fixed to accept all five profiles dynamically.
- **`EvolutionTrainer.__del__` raised** `AttributeError` whenever construction failed before `self.runner` was set — traceback printed on every rejected config. Guarded with `hasattr`.

### Round 7 — Two real runs on disk revealed

| Symptom | Cause | Fix |
|---------|-------|-----|
| `best_agent.pt` **108 MB** for an 18 KB genome | `update_best()` called `get_weights()`, building the lazily-created 9.28M-parameter Torch network just to copy it; checkpoint then wrote both `network_weights` and `best_weights`. | Genome snapshotted directly; Torch parameters only written for agents that have no genome. 108 MB → 58 KB. |
| An 11-generation run had **no checkpoint at all** | `checkpoint_interval` was 50 (UI used `generations // 4`) — a 19-minute training could not be continued. | Checkpoint always written when training ends or is stopped; UI checkpoints every 10 generations. |
| `training.log` was **zero bytes** in both runs | Module logger's level never set → inherited root WARNING, dropped every INFO record. | File handler sets its own level explicitly. |
| `metrics.json` said `population_size: 240` for a run training **24** agents | Resume kept the requested size instead of the one the checkpoint actually holds. | Resume reconciles `population_size` with checkpoint, warns when they differ. |

### Round 8 — Sim fidelity to real Clash Royale + watchable training

| Change | Before | After |
|--------|--------|-------|
| Per-slot deploy lockout (`DEPLOY_COOLDOWN`) | Cards locked for ticks after playing | Removed: real CR has no per-card cooldown; elixir is the only gate. Mechanism kept for feature encoding. |
| King activation | Woke on **any** damage | Wakes at ≤75% max HP (25%-lost threshold, `KING_ACTIVATION_HP_FRACTION`). ⚠️ Threshold chosen from game knowledge — user has not confirmed the exact real-game rule; one-line constant to verify. |
| Overtime | 120 ticks, first crown wins | Sudden death up to 60 s (`overtime_ticks` default 600); both kings activate when OT starts (real CR behaviour). |
| Generator template elixir rate | `elixir_regen_rate: 0.3` (= 8× the real rate) in both sim templates | Fixed to 0.0357 (~1 per 2.8 s, matching engine default). |
| `configs/sim_game.yaml` | Dead keys (`overtime_duration_ticks`, `deployment_cooldown`) | Renamed/removed; `from_file` still accepts the legacy key. |

**Watchable training.** Training previously showed only charts and a log while workers ran. Now, after each completed generation, the Train tab plays one recorded match between champions — reigning champion vs its newest distinct hall-of-fame predecessor (scripted opponent when they are the same genome) — with play/pause and speed controls, auto-advancing to the next generation's match. The match is simulated in the training worker thread (~1–2 s per generation, sequential with the pool rather than competing for CPU), captured via the existing `play_match`/snapshot path and posted as a new `spectator` job event.

| Defect found by smoke testing | Effect | Fix |
|--------|--------|-----|
| Worker thread called Tkinter's `BooleanVar.get()` to read the live spectate toggle | `RuntimeError: main thread is not in main loop`, swallowed by the trainer's `on_generation` guard — spectating silently disabled with no error anywhere visible | Toggle copied into a plain Python flag on the main thread; workers never touch Tcl. A raising toggle now degrades to a `spectate:` line in the UI log pane (regression-tested). |

### Fitness cannot show progress in tournament mode

Both real runs looked flat: mean fitness moved from 0.664 to 0.700 across 61 generations. That is **arithmetic**, not a training failure — tournament fitness is points per match inside a closed field, so its mean is pinned near 0.5 however strong the population becomes.

The real signal was already in the data: hall-of-fame champions' ratings decline relative to the field as the population outgrows them (`hof_gen0` 1483 → 1441, `hof_gen1` 1464 → 1422). Progress snapshots now carry `population_elo` and `hall_of_fame_elo`; Train tab charts ratings rather than the flat fitness curve.

---

## Known limitations & future work

| Area | Status | Notes |
|------|--------|-------|
| Frozen build size | ⚠️ Heavy (~4 GB with CUDA) | CPU-only torch wheel cuts it dramatically; app only needs torch for checkpoint I/O (genome is NumPy). `CRP_CONSOLE=1` for debug builds. OneDrive/Dropbox sync folders cause `PermissionError` during build — use `--output PATH`. |
| Watch tab | Partial | Replays one match at a time. Watching two saved agents play each other would reuse `play_match` with `opponent_genome` (already wired, not exposed in UI). |
| Fitness transfer | Real but modest | 10 gens × 16 agents = smoke test. Longer runs at larger population are the next step now that the loop scales. |
| Hall of fame diversity | Could improve | Keeps only most recent champions. A diverse archive (sampling across whole run rather than sliding window) resists cycling better — worth trying if agents beat recent champions while losing to older ones. |
| `configs/sim_game.yaml` | Documentation only | Engine hardcodes its layout and rules; nothing reads the tower section. Either wire it up or drop it. |
| Status effects | Single-slot | Applying a stun replaces an active poison. Real stacking would need a list of active effects per unit. |
| `OpponentProfile.min_play_gap` | Nearly inert | At real elixir rate, affording a card takes ~84 ticks; a 3–6 tick gap rarely binds. Only matters when opponent has large bank + cheap cards. |
| All 20+ phases in roadmap | ✅ Complete through Phase 18 | Phases 19-24 (live-game prototype, full sim, live fine-tuning, distributed Ray training, API docs, Jupyter tutorials) remain open. |

---

## Where to pick up

1. **Longer tournament runs at larger populations** — the loop is stable; scaling is the next experiment.
2. **Hall of fame diversity** — sampling across the whole run rather than a sliding window may improve resistance to cycling.
3. **Watch tab head-to-head** — expose `play_match(opponent_genome)` in the UI so two saved agents can play each other.
4. **Live-game interaction** (Phase 19) — `src/env/live/` has screen capture, game state extraction, and action mapper stubs but is not wired into training yet.
5. **Full card registry** (Phase 20) — currently ~140 cards; real Clash Royale has 110+ unique cards at various levels. The registry structure supports expansion.
