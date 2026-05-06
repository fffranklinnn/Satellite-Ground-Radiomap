# Satellite-Ground-Radiomap Scene-Adaptive Layer Policy Plan

## Goal Description

Refine the Satellite-Ground-Radiomap codebase so the propagation pipeline is no longer treated as a fixed three-layer stack that must always run L1 macro, L2 terrain, and L3 urban refinement together. The new goal is to make the runtime scene-adaptive while preserving the canonical semantic contracts already established by the semantic-unification refactor. The core observation is that different physical scenes are governed by different dominant factors: dense urban plains may require strong L3 building refinement but only weak L2 terrain correction; mountainous rural regions may require strong L2 terrain computation but no L3 at all; sparse flat regions may only justify L1. The implementation therefore introduces explicit scene profiles, a shared layer-policy decision model, and manifest/runtime contracts that distinguish "layer intentionally disabled by scene policy" from "layer expected but unavailable because of missing data." This keeps the code scientifically interpretable, avoids forcing unnecessary computation, and makes benchmark behavior match the actual physics and data availability of each scene.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.

- AC-1: Scene profile drives layer enablement explicitly
  - Positive Tests (expected to PASS):
    - A config with `scene.profile = "urban_flat"` resolves to a canonical runtime contract where `l1_macro` is required, `l3_urban` is enabled, and `l2_topo` is optional or disabled by policy
    - A config with `scene.profile = "mountain_rural"` resolves to a canonical runtime contract where `l1_macro` and `l2_topo` are enabled and `l3_urban` is disabled by policy
    - A config with `scene.profile = "plain_sparse"` resolves to a contract where only `l1_macro` is required and higher-resolution layers may be disabled
    - Layer enablement results are serializable into a stable runtime policy record
  - Negative Tests (expected to FAIL):
    - An unknown `scene.profile` raises a validation error
    - A runtime path that enables or disables layers ad hoc without going through the shared policy resolver raises or logs a contract violation in strict mode

- AC-2: Disabled layers distinguish scene-policy absence from missing-input failure
  - Positive Tests (expected to PASS):
    - When `l3_urban` is disabled for `mountain_rural`, the manifest records `reason_type = "scene_policy"` rather than `missing_input`
    - When `l3_urban` is expected under `urban_flat` but tile data are missing, the manifest/runtime records `reason_type = "missing_input"`
    - In normal mode, optional layers disabled by policy do not emit missing-data warnings
    - In normal mode, configured-but-missing required layer inputs emit warnings with the correct layer attribution
  - Negative Tests (expected to FAIL):
    - Benchmark mode treats a required-by-scene layer with missing configured input as a hard failure
    - Benchmark mode does not allow a missing-data case to be silently rewritten as a scene-policy disable

- AC-3: Canonical runtime supports partial-layer composition without semantic drift
  - Positive Tests (expected to PASS):
    - `L1 + L3` composition for `urban_flat` scenes produces a valid product on `product_grid` without requiring `L2`
    - `L1 + L2` composition for `mountain_rural` scenes produces a valid product on `product_grid` without requiring `L3`
    - `L1`-only composition for `plain_sparse` scenes produces a valid product and a manifest that explicitly records absent higher-resolution layers
    - Multi-scale composition still requires consistent projected inputs; skipping a layer does not relax projection/grid contracts
  - Negative Tests (expected to FAIL):
    - A runtime path that substitutes zero arrays for a disabled layer without recording the disable reason fails contract checks in strict mode
    - A runtime path that treats "layer absent" and "layer present but numerically zero" as equivalent in manifest/provenance output fails validation

- AC-4: Scene policy is benchmark-visible and reproducible
  - Positive Tests (expected to PASS):
    - The manifest includes `scene_profile`, `enabled_layers`, `disabled_layers`, and per-layer disable reasons as stable provenance fields
    - Same config + same scene profile + same data snapshot + same frame list produces identical layer-policy manifest output
    - Benchmark baselines can be declared per scene family (`urban_flat`, `mountain_rural`, `suburban_mixed`, `plain_sparse`)
    - Benchmark artifacts clearly state whether they cover `L1`, `L1+L2`, `L1+L3`, or `L1+L2+L3`
  - Negative Tests (expected to FAIL):
    - A benchmark artifact that omits the scene profile or layer-policy contract is considered incomplete
    - Two runs with different enabled-layer sets but identical product filenames/IDs fail manifest equivalence checks

- AC-5: Policy resolution is centralized rather than duplicated in scripts
  - Positive Tests (expected to PASS):
    - `main.py`, `BenchmarkRunner`, and multisat scripts all obtain layer enablement decisions from a shared resolver
    - Layer-specific runtime behavior is driven by one shared policy object rather than scattered booleans
    - Legacy scripts can adapt through compatibility wrappers while still producing explicit policy records
  - Negative Tests (expected to FAIL):
    - A new script entry point that bypasses the shared policy resolver fails a policy-contract regression test
    - Divergent layer-selection logic between `main.py` and `BenchmarkRunner` fails deterministic parity tests

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.

### Upper Bound (Maximum Acceptable Scope)
The implementation introduces a fully shared scene-policy subsystem with formal profile definitions, a central `LayerPolicyResolver`, manifest-level layer decision provenance, benchmark grouping by scene family, runtime parity tests across `main.py` and `BenchmarkRunner`, and compatibility wrappers for legacy entry points. Scene profile resolution can use both explicit config and optional derived heuristics, but canonical behavior remains config-driven and deterministic. The system supports `L1`, `L1+L2`, `L1+L3`, and `L1+L2+L3` as first-class benchmark/runtime contracts.

### Lower Bound (Minimum Acceptable Scope)
The implementation introduces explicit `scene.profile` config, a shared resolver that determines which layers are enabled, and manifest fields that distinguish `scene_policy` from `missing_input`. `main.py` and `BenchmarkRunner` both adopt the resolver. Partial-layer products are supported on the canonical path without breaking grid/projection semantics. Benchmark strict mode correctly fails when a layer required by scene policy is missing its configured inputs.

### Allowed Choices
- Can use: explicit scene profiles such as `urban_flat`, `mountain_rural`, `suburban_mixed`, `plain_sparse`
- Can use: a frozen dataclass or small immutable object to represent resolved layer policy
- Can use: manifest fields such as `scene_profile`, `enabled_layers`, `disabled_layers`, `reason_type`, `reason`
- Can use: compatibility wrappers for legacy scripts as long as canonical policy still appears in outputs
- Cannot use: implicit "all layers always on" assumptions in new canonical runtime code
- Cannot use: treating missing data and scene-policy disablement as the same provenance category
- Cannot use: zero-filled placeholder layers as silent substitutes for intentionally absent layers on the canonical path

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

The canonical scene-adaptive runtime after refactoring:

```python
policy = resolve_layer_policy(
    scene_profile=config["scene"]["profile"],
    config=config,
    data_validation=data_validation_report,
    strict=strict,
)

frame = frame_builder.build(timestamp=ts, sat_info=sat, coverage=coverage, strict=True)

entry = l1.propagate_entry(frame) if policy.enable_l1 else None
terrain = l2.propagate_terrain(frame, entry=entry) if policy.enable_l2 else None
urban = l3.refine_urban(frame, entry=entry) if policy.enable_l3 else None

product = compose_scene_adaptive(
    frame=frame,
    entry=entry,
    terrain=terrain,
    urban=urban,
    policy=policy,
)

manifest = ProductManifest.build(
    ...,
    scene_profile=policy.scene_profile,
    enabled_layers=policy.enabled_layers,
    disabled_layers=policy.disabled_layers,
)
```

### Recommended Scene Profiles

| Profile | Dominant Physics | Typical Use | Expected Layer Set |
|--------|-------------------|-------------|--------------------|
| `urban_flat` | Building obstruction/refinement dominates; terrain weak | Xi'an main urban area | `L1 + L3`, `L2` optional |
| `mountain_rural` | Terrain obstruction dominates; urban fabric absent | Qinling, Huashan, mountainous countryside | `L1 + L2`, `L3` disabled |
| `suburban_mixed` | Both terrain and buildings matter | Mountain-front towns, suburban edges | `L1 + L2 + L3` |
| `plain_sparse` | Macro link dominates; both terrain and buildings weak | Rural flat plains, open fields | `L1` primarily |

### Policy Contract

- `scene_profile` is the explicit high-level declaration of what physical scene is being modeled
- `enabled_layers` is the resolved runtime result
- `disabled_layers[layer].reason_type` must be one of:
  - `scene_policy`
  - `missing_input`
  - `manual_override`
- `missing_input` means the layer was expected under the scene profile but could not run
- `scene_policy` means the layer was intentionally excluded because the scene model says it is not relevant
- Benchmark mode should fail on `missing_input` for required layers, but not on `scene_policy`

### Manifest / Provenance Implications

Suggested additions to the runtime contract:

- `scene_profile`
- `enabled_layers`
- `disabled_layers`
- `layer_policy_version`
- `policy_decision_source` (`config_only`, future `config_plus_heuristic`, etc.)

These fields complement, not replace, the existing provenance fields such as `coverage_signature`, `frame_contract_hash`, and `input_snapshot_hash`.

### Benchmark Implications

Benchmark suites should be grouped by scene family rather than assuming that every benchmark must exercise all three refinement layers:

- `benchmark_urban_flat`
- `benchmark_mountain_rural`
- `benchmark_suburban_mixed`
- `benchmark_plain_sparse`

Each benchmark artifact should state the exact active layer contract (`L1`, `L1+L2`, `L1+L3`, `L1+L2+L3`) so comparisons remain physically meaningful.

### Relevant References

- `docs/plan_20260423.md` — Existing semantic-unification refactor plan
- `src/context/coverage_spec.py` — Multi-grid coverage contract already introduced
- `src/context/frame_context.py` — Canonical frame-level contract
- `src/context/multiscale_map.py` — Composition semantics and projected combination
- `src/pipeline/benchmark_runner.py` — Benchmark runtime entry point
- `src/products/manifest.py` — Provenance and input-contract recording
- `main.py` — Main canonical runtime path
- `scripts/generate_multisat_timeseries_radiomap.py` — Multisat runtime path that should eventually share layer policy

## Dependencies and Sequence

### Milestones

1. Milestone 1 — Scene Profile & Policy Contract
   - Define supported scene profiles
   - Add config schema for `scene.profile`
   - Design resolved layer-policy object and disable-reason schema

2. Milestone 2 — Shared Policy Resolution
   - Implement centralized layer-policy resolver
   - Thread resolver into `main.py` and `BenchmarkRunner`
   - Ensure canonical runtime can skip non-required layers cleanly

3. Milestone 3 — Manifest & Provenance Extension
   - Add `scene_profile`, `enabled_layers`, `disabled_layers`, `reason_type`
   - Distinguish `scene_policy` from `missing_input`
   - Add strict benchmark failures for required-but-missing layers

4. Milestone 4 — Partial-Layer Benchmarking
   - Re-scope benchmark artifacts by scene family
   - Add canonical tests for `L1`, `L1+L2`, `L1+L3`, and full-stack runs
   - Verify runtime parity across `main.py` and `BenchmarkRunner`

Milestone 1 must complete before Milestone 2. Milestone 2 must complete before Milestone 3. Milestone 3 must complete before partial-layer benchmark baselines are considered authoritative.

## Task Breakdown

Each task must include exactly one routing tag:
- `coding`: implemented directly in the repo
- `analyze`: exploration / impact assessment

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Analyze existing runtime assumptions that implicitly require all three layers together (`main.py`, `BenchmarkRunner`, multisat script, composition helpers) | AC-1, AC-5 | analyze | - |
| task2 | Define supported `scene.profile` values and config contract; add validation tests for unknown profiles | AC-1 | coding | task1 |
| task3 | Design and implement a shared `LayerPolicyResolver` / policy object that maps scene profile + config + strict mode to enabled/disabled layers | AC-1, AC-5 | coding | task2 |
| task4 | Update `main.py` to resolve layer policy once and execute only the required layers while preserving projection/composition semantics | AC-3, AC-5 | coding | task3 |
| task5 | Update `BenchmarkRunner` to use the same policy object and fail correctly when required-by-scene layers are missing configured inputs | AC-2, AC-4, AC-5 | coding | task3 |
| task6 | Extend `ProductManifest` / runtime provenance schema with `scene_profile`, `enabled_layers`, `disabled_layers`, and disable reasons | AC-2, AC-4 | coding | task4, task5 |
| task7 | Add manifest tests distinguishing `scene_policy` from `missing_input` under both normal mode and benchmark strict mode | AC-2, AC-4 | coding | task6 |
| task8 | Add canonical runtime tests for `urban_flat` (`L1+L3`), `mountain_rural` (`L1+L2`), and `plain_sparse` (`L1`) execution paths | AC-3, AC-5 | coding | task5 |
| task9 | Re-scope benchmark artifact conventions so benchmark reports explicitly declare scene family and active layer set | AC-4 | coding | task7, task8 |
| task10 | Evaluate whether legacy scripts can adopt the policy resolver directly or require compatibility adapters | AC-5 | analyze | task5 |

## Claude-Codex Deliberation

### Agreements
- The existing semantic-unification work remains valid; scene adaptation builds on it rather than replacing it
- L1 remains the universal macro layer, but L2 and L3 should be treated as scene-dependent refinements
- It is scientifically incorrect to conflate "layer absent by design" with "layer missing because data were unavailable"
- Benchmark logic should describe the active layer set explicitly rather than assuming full three-layer execution
- The right abstraction target is a shared layer-policy resolver, not more script-local booleans

### Resolved Disagreements
- **Three-layer requirement**: A naive interpretation of the current pipeline suggests that every scene must run L1/L2/L3. Resolution: the canonical model should support partial-layer execution as a first-class contract, provided the active layer set is explicit and reproducible.
- **Benchmark strictness**: One possible simplification is to let benchmark mode fail whenever any higher-resolution layer is unavailable. Resolution: benchmark mode should fail only when a layer is required by the active scene policy, not when a layer is intentionally excluded by that policy.
- **Scene inference**: Automatic scene classification from data was considered. Resolution: the near-term canonical path should remain config-driven (`scene.profile` explicit) because heuristics would add ambiguity before policy contracts are stable.

### Convergence Status
- Final Status: `converged`
- Rationale: The scene-adaptive policy is compatible with the semantic-unification direction and gives a clearer physical meaning to partial-layer runs. It also reduces unnecessary computation and avoids forcing unavailable data products where they are not scientifically relevant.

## Pending User Decisions

- DEC-1: Near-term policy source
  - Option A: explicit `scene.profile` only
  - Option B: explicit profile with optional future heuristics
  - Tradeoff Summary: Option A is simpler and deterministic; Option B may improve ergonomics later but adds ambiguity now.

- DEC-2: Benchmark baseline strategy
  - Option A: maintain separate baselines per scene family and active layer set
  - Option B: keep one benchmark path but annotate active layers inside it
  - Tradeoff Summary: Option A is cleaner and more scientifically interpretable; Option B is lighter-weight but easier to misuse.
