# Satellite-Ground-Radiomap Scene-Adaptive Layer Policy Plan

## Goal Description

Add a scene-adaptive layer policy to the existing canonical runtime so the project no longer assumes every run must execute `L1 + L2 + L3`. The target outcome is a single project-level contract that maps an explicit `scene.profile` to a deterministic layer-execution decision, records why layers were enabled or disabled, and keeps `main.py`, `BenchmarkRunner`, and the multisatellite timeseries script aligned with the same runtime behavior. The work must build on the current `FrameContext` / `CoverageSpec` / projection / composition / manifest architecture rather than redesigning the propagation stack.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.

- AC-1: Explicit scene profile contract exists at runtime
  - Positive Tests (expected to PASS):
    - Config parsing accepts a declared `scene.profile` and preserves it through the canonical runtime
    - Supported profiles map deterministically to documented default layer sets such as `urban_flat -> L1+L3`, `mountain_rural -> L1+L2`, `suburban_mixed -> L1+L2+L3`, `plain_sparse -> L1`
    - The resolved scene profile is available to downstream runtime and manifest code without each entry point redefining profile semantics
  - Negative Tests (expected to FAIL):
    - An unknown `scene.profile` raises a validation error on canonical paths
    - A run with no resolvable scene profile on a strict path does not silently fall back to ad hoc layer selection

- AC-2: A unified `LayerPolicyResolver` decides enabled and disabled layers
  - Positive Tests (expected to PASS):
    - A single resolver produces `enabled_layers` and `disabled_layers` for all canonical entry points
    - Resolver output includes structured disabled-layer reasons, not just booleans
    - The resolver can factor scene profile, user overrides, strict or benchmark mode, and input-availability checks into one policy object
  - Negative Tests (expected to FAIL):
    - `main.py` and `BenchmarkRunner` producing different layer decisions for the same config and scene is rejected by tests
    - Runtime code bypassing the resolver and directly toggling L2 or L3 on canonical paths is treated as a contract violation
  - AC-2.1: Disabled-layer reason taxonomy is explicit
    - Positive: Disabled reasons distinguish at least `scene_policy`, `missing_input`, and `user_override`
    - Negative: A disabled layer recorded without a reason type or with an ambiguous free-form-only status is rejected by contract tests

- AC-3: Canonical entry points execute only the layers allowed by policy
  - Positive Tests (expected to PASS):
    - `main.py` uses policy resolution before running propagation layers
    - `src/pipeline/benchmark_runner.py` uses the same policy resolution path as `main.py`
    - `scripts/generate_multisat_timeseries_radiomap.py` adopts the same policy contract after the main and benchmark paths
    - Partial-layer combinations `L1`, `L1+L2`, `L1+L3`, and `L1+L2+L3` still produce valid projected composition on `product_grid`
  - Negative Tests (expected to FAIL):
    - A profile that disables L3 does not instantiate or execute urban refinement on the canonical path
    - A profile that disables L2 does not export outputs implying terrain computation occurred
    - A canonical entry point running all three layers unconditionally despite policy resolution fails regression tests

- AC-4: Manifest and provenance explain actual layer-policy decisions
  - Positive Tests (expected to PASS):
    - `ProductManifest` records `scene_profile`, `enabled_layers`, disabled-layer metadata, and policy-version information
    - Manifest output can distinguish “layer not needed for this scene” from “layer required but missing input”
    - Repeated runs with the same config, data snapshot, frame list, and scene profile produce the same policy record
  - Negative Tests (expected to FAIL):
    - A disabled layer recorded only in logs and absent from manifest output fails manifest contract tests
    - A manifest that cannot explain whether a missing layer came from scene policy or missing data is rejected
  - AC-4.1: Missing-input behavior respects mode semantics
    - Positive: In strict or benchmark mode, a layer required by policy but blocked by missing configured input raises a failure
    - Negative: A scene-policy-disabled layer is not treated as an error merely because its backing dataset is absent

- AC-5: Benchmark behavior supports scene-specific valid layer combinations
  - Positive Tests (expected to PASS):
    - Benchmark runs can validate `urban_flat` with `L1+L3`, `mountain_rural` with `L1+L2`, `suburban_mixed` with `L1+L2+L3`, and `plain_sparse` with `L1`
    - Benchmark artifacts record the actual resolved layer set for each scene
    - Strict benchmark checks still enforce required inputs only for layers that policy marks as required
  - Negative Tests (expected to FAIL):
    - Benchmark logic assuming “all scenes must run full three-layer propagation” fails scene-policy regression tests
    - A benchmark artifact omitting the resolved layer combination fails validation

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.

### Upper Bound (Maximum Acceptable Scope)
The implementation introduces a formal `scene.profile` contract, a reusable `LayerPolicyResolver`, shared disabled-reason taxonomy, manifest-level layer-policy recording, strict-mode handling for required inputs, and benchmark coverage for all intended scene combinations. `main.py`, `BenchmarkRunner`, and `generate_multisat_timeseries_radiomap.py` all converge on the same policy path. Existing projection and composition contracts are reused so partial-layer products remain semantically correct.

### Lower Bound (Minimum Acceptable Scope)
The implementation adds a stable `scene.profile` field, a resolver shared by `main.py` and `BenchmarkRunner`, manifest recording for enabled and disabled layers with reason type, and tests covering at least `urban_flat`, `mountain_rural`, and `plain_sparse`. The multisatellite script may remain a second-step migration if the main and benchmark paths are already unified and covered by regression tests.

### Allowed Choices
- Can use: `src/planning/` as the home for `LayerPolicyResolver` and related policy types
- Can use: frozen dataclasses, enums, or validated string constants for policy objects and reason types
- Can use: existing `ProductManifest.metadata` patterns or explicit new manifest fields, provided the result is stable and testable
- Can use: current `FrameContext`, `CoverageSpec`, `project_to_product_grid(...)`, and `MultiScaleMap.compose(...)` contracts without redesigning them
- Cannot use: automatic scene classification as part of this change
- Cannot use: rewriting L1, L2, or L3 physics formulas as a substitute for policy control
- Cannot use: separate ad hoc layer-toggle logic in each entry point
- Cannot use: plan terminology such as `AC-` or `Milestone` in implementation code

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

One workable runtime shape is:

```python
scene_profile = config["scene"]["profile"]
policy = LayerPolicyResolver.from_config(config).resolve(
    scene_profile=scene_profile,
    strict=strict_mode,
    benchmark=benchmark_mode,
    input_availability=input_check,
)

frame = frame_builder.build(timestamp=ts, sat_info=sat_info)
entry = l1.propagate_entry(frame) if policy.is_enabled("l1_macro") else None
terrain = l2.propagate_terrain(frame, entry=entry) if policy.is_enabled("l2_topo") else None
urban = l3.refine_urban(frame, entry=entry) if policy.is_enabled("l3_urban") else None

manifest = ProductManifest.build(
    ...,
    metadata={
        "scene_profile": policy.scene_profile,
        "enabled_layers": list(policy.enabled_layers),
        "disabled_layers": policy.disabled_layers_dict(),
        "layer_policy_version": policy.version,
    },
)
```

The main design constraint is that scene-adaptive behavior should be introduced before layer execution, not buried inside individual layer implementations. The existing composition path is already capable of combining partial-layer projected views, so the new control point should primarily be execution policy and output explainability.

### Relevant References

- [src/planning/satellite_selector.py](/home/Users_Work_Space/zsfang/Satellite-Ground-Radiomap/src/planning/satellite_selector.py) - Existing runtime planning module; natural home for layer-policy resolution
- [main.py](/home/Users_Work_Space/zsfang/Satellite-Ground-Radiomap/main.py) - Canonical single-run entry point that already builds frames and projects to `product_grid`
- [src/pipeline/benchmark_runner.py](/home/Users_Work_Space/zsfang/Satellite-Ground-Radiomap/src/pipeline/benchmark_runner.py) - Deterministic benchmark path that should share policy resolution with `main.py`
- [scripts/generate_multisat_timeseries_radiomap.py](/home/Users_Work_Space/zsfang/Satellite-Ground-Radiomap/scripts/generate_multisat_timeseries_radiomap.py) - High-value script that should adopt the same scene-policy contract after the primary paths
- [src/products/manifest.py](/home/Users_Work_Space/zsfang/Satellite-Ground-Radiomap/src/products/manifest.py) - Manifest and input-collection contract that should record policy decisions
- [src/context/multiscale_map.py](/home/Users_Work_Space/zsfang/Satellite-Ground-Radiomap/src/context/multiscale_map.py) - Composition target that should continue to accept projected partial-layer inputs
- [src/compose/__init__.py](/home/Users_Work_Space/zsfang/Satellite-Ground-Radiomap/src/compose/__init__.py) - Projection entry point that partial-layer execution should continue to reuse

## Dependencies and Sequence

### Milestones
1. Milestone 1: Scene profile contract and policy model
   - Phase A: Define supported scene profiles and default layer sets
   - Phase B: Introduce policy data structures and disabled-reason taxonomy
   - Phase C: Add config validation and resolver tests

2. Milestone 2: Canonical runtime integration
   - Step 1: Integrate policy resolution into `main.py`
   - Step 2: Integrate the same policy resolution into `BenchmarkRunner`
   - Step 3: Ensure partial-layer projected composition still yields valid products

3. Milestone 3: Explainability and strictness
   - Step 1: Extend manifest output with scene and layer-policy records
   - Step 2: Distinguish `scene_policy`, `missing_input`, and `user_override`
   - Step 3: Align strict and benchmark failure behavior with required-by-policy inputs

4. Milestone 4: Secondary entry points and benchmark semantics
   - Step 1: Migrate `generate_multisat_timeseries_radiomap.py` to the shared resolver
   - Step 2: Add scene-specific benchmark coverage and artifact validation
   - Step 3: Lock regression coverage around valid layer combinations per scene

Milestone 1 must complete before runtime integration. Milestone 2 should complete before manifest semantics are finalized, because the policy object has to exist in the runtime first. Milestone 3 must complete before benchmark semantics are considered stable. Milestone 4 can begin after the main and benchmark paths are already converged.

## Task Breakdown

Each task must include exactly one routing tag:
- `coding`: implemented by Claude
- `analyze`: executed via Codex (`/humanize:ask-codex`)

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Analyze current layer enablement, config parsing, and per-entry-point runtime differences across `main.py`, `BenchmarkRunner`, and the multisat script | AC-1, AC-2, AC-3 | analyze | - |
| task2 | Define supported `scene.profile` values and their default layer combinations in a single policy module | AC-1 | coding | task1 |
| task3 | Implement `LayerPolicyResolver` and typed disabled-reason structures in `src/planning/` | AC-2, AC-2.1 | coding | task2 |
| task4 | Add resolver unit tests covering known profiles, unknown profiles, and deterministic output | AC-1, AC-2 | coding | task3 |
| task5 | Integrate policy resolution into `main.py` before layer execution | AC-3 | coding | task3 |
| task6 | Integrate the same policy resolution path into `src/pipeline/benchmark_runner.py` | AC-2, AC-3, AC-5 | coding | task5 |
| task7 | Add integration tests proving canonical entry points no longer run all layers unconditionally | AC-3 | coding | task5, task6 |
| task8 | Analyze current `ProductManifest` schema and decide the narrowest stable location for layer-policy recording | AC-4 | analyze | task6 |
| task9 | Extend manifest output to record `scene_profile`, `enabled_layers`, `disabled_layers`, and `layer_policy_version` | AC-4 | coding | task8 |
| task10 | Implement required-input checks that distinguish `scene_policy`, `missing_input`, and `user_override` outcomes under strict or benchmark mode | AC-4.1, AC-5 | coding | task9 |
| task11 | Add manifest and strict-mode regression tests for disabled-layer explanations and required-input failures | AC-4, AC-4.1 | coding | task10 |
| task12 | Migrate `scripts/generate_multisat_timeseries_radiomap.py` to the shared policy resolver | AC-3 | coding | task6 |
| task13 | Add benchmark-path tests covering `L1`, `L1+L2`, `L1+L3`, and `L1+L2+L3` scene combinations | AC-5 | coding | task11, task12 |

## Claude-Codex Deliberation

### Agreements
- The repository already has a sufficiently strong frame, grid, projection, and manifest backbone to support scene-adaptive execution without redesigning the core runtime
- The next gap is policy control, not additional complexity inside L1, L2, or L3 physics
- A single project-level resolver is preferable to per-script layer-toggle logic
- Manifest explainability is part of the functional requirement, not an optional logging detail
- Benchmark semantics must evolve away from the assumption that all scenes require all three propagation layers

### Resolved Disagreements
- Scene-policy scope: An early interpretation could have expanded into automatic scene detection. Resolution: this plan is limited to explicit `scene.profile` input and deterministic runtime policy. Rationale: it solves the contract problem without introducing unreliable heuristics.
- Integration order: One option was to migrate every script at once. Resolution: main path and benchmark path first, multisat script second. Rationale: this preserves a clear canonical path and lowers rollout risk.
- Manifest placement: One option was to redesign the entire manifest model. Resolution: extend the existing manifest contract narrowly with layer-policy semantics. Rationale: the repo already contains provenance infrastructure worth reusing.

### Convergence Status
- Final Status: `converged`

## Pending User Decisions

- DEC-1: Scene profile vocabulary
  - Claude Position: Keep the initial supported set narrow and explicit: `urban_flat`, `mountain_rural`, `suburban_mixed`, `plain_sparse`
  - Codex Position: Same narrow set for phase one, with room to add aliases later only if tests require them
  - Tradeoff Summary: A small controlled vocabulary reduces ambiguity and keeps policy tests deterministic
  - Decision Status: `PENDING`

- DEC-2: Manifest field placement
  - Claude Position: Prefer explicit first-class manifest fields for scene and layer-policy semantics
  - Codex Position: Using stable metadata fields is acceptable if schema churn must stay low, provided tests lock the contract
  - Tradeoff Summary: First-class fields improve discoverability; metadata reuse minimizes schema disruption
  - Decision Status: `PENDING`

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as `AC-`, `Milestone`, `Step`, `Phase`, or similar workflow markers
- Policy code should use domain names such as `scene_profile`, `enabled_layers`, `disabled_reason`, and `required_inputs`
- Logging is not a substitute for contract output; tests should assert structured runtime and manifest state

