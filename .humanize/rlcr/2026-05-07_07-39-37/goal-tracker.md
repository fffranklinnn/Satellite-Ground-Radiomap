# Goal Tracker

<!--
This file tracks the ultimate goal, acceptance criteria, and plan evolution.
It prevents goal drift by maintaining a persistent anchor across all rounds.

RULES:
- IMMUTABLE SECTION: Do not modify after initialization
- MUTABLE SECTION: Update each round, but document all changes
- Every task must be in one of: Active, Completed, or Deferred
- Deferred items require explicit justification
-->

## IMMUTABLE SECTION
<!-- Do not modify after initialization -->

### Ultimate Goal

Add a scene-adaptive layer policy to the existing canonical runtime so the project no longer assumes every run must execute `L1 + L2 + L3`. The target outcome is a single project-level contract that maps an explicit `scene.profile` to a deterministic layer-execution decision, records why layers were enabled or disabled, and keeps `main.py`, `BenchmarkRunner`, and the multisatellite timeseries script aligned with the same runtime behavior. The work must build on the current `FrameContext` / `CoverageSpec` / projection / composition / manifest architecture rather than redesigning the propagation stack.

## Acceptance Criteria

### Acceptance Criteria
<!-- Each criterion must be independently verifiable -->
<!-- Claude must extract or define these in Round 0 -->


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

---

## MUTABLE SECTION
<!-- Update each round with justification for changes -->

### Plan Version: 1 (Updated: Round 0)

#### Plan Evolution Log
<!-- Document any changes to the plan with justification -->
| Round | Change | Reason | Impact on AC |
|-------|--------|--------|--------------|
| 0 | Initial plan | - | - |
| 0 | Round 0 narrowed to canonical-path alignment for the multisat script plus regression coverage | `main.py`, `BenchmarkRunner`, and the shared policy module are already implemented; the remaining contract gap is the multisat entry point and scene-combination regression coverage | Advances AC-3 and AC-5 directly while preserving AC-1 and AC-2 behavior |

#### Active Tasks
<!-- Mainline tasks only: each task must directly advance the current round objective and carry routing metadata -->
| Task | Target AC | Status | Tag | Owner | Notes |
|------|-----------|--------|-----|-------|-------|
| task12 | AC-3 | completed | coding | claude | Multisat strict-mode checks now normalize project-root-relative input paths before policy resolution and manifest input collection |
| task13 | AC-5 | completed | coding | claude | BenchmarkRunner coverage now exercises the full `plain_sparse`, `mountain_rural`, `urban_flat`, and `suburban_mixed` scene matrix with manifest assertions |

### Blocking Side Issues
<!-- Only issues that directly block current mainline progress belong here -->
| Issue | Discovered Round | Blocking AC | Resolution Path |
|-------|-----------------|-------------|-----------------|
| Multisat strict policy resolution checks relative input paths against the process cwd instead of the script `project_root` | 0 | AC-3 | Normalize configured layer input paths before resolving policy and collecting manifest inputs, or pass project-root-aware `input_availability` into every multisat policy resolution call |
| Benchmark scene-combination matrix coverage is incomplete | 0 | AC-5 | Add a parameterized `BenchmarkRunner.run_frame` regression covering `plain_sparse`, `mountain_rural`, `urban_flat`, and `suburban_mixed`, asserting manifest metadata and layer method call counts for each |

### Queued Side Issues
<!-- Non-blocking issues stay queued and must NOT replace the round objective -->
| Issue | Discovered Round | Why Not Blocking | Revisit Trigger |
|-------|-----------------|------------------|-----------------|
| `bitlesson-selector` hangs instead of returning lesson IDs | 0 | Existing BitLesson KB already identifies the relevant runtime-wiring lesson; the hang does not block code changes | Revisit if future rounds require selecting among multiple candidate lessons |

### Completed and Verified
<!-- Only move tasks here after Codex verification -->
| AC | Task | Completed Round | Verified Round | Evidence |
|----|------|-----------------|----------------|----------|
| AC-3 | task12 | 1 | 1 | `pytest tests/test_runtime_layer_policy.py` passed with a project-root-relative strict-path regression |
| AC-5 | task13 | 1 | 1 | `pytest tests/test_pipeline/test_benchmark_runner.py` passed with parameterized scene matrix coverage |

### Explicitly Deferred
<!-- Items here require strong justification -->
| Task | Original AC | Deferred Since | Justification | When to Reconsider |
|------|-------------|----------------|---------------|-------------------|
