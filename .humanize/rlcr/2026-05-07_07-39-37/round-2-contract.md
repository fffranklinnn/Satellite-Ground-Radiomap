## Round 2 Contract

### Mainline Objective
Make the canonical runtime policy-aware before layer construction and close the multisat strict-path gap for nested `layers.l1_macro.tle.file`.

### Target ACs
- AC-3
- AC-5

### Blocking Issues
- `main.py` still instantiates layers from raw config before policy resolution, which allows policy-disabled layers to be constructed.
- `scripts/generate_multisat_timeseries_radiomap.py` still treats nested `layers.l1_macro.tle.file` as missing in strict required-data validation.

### Queued Issues
- Goal tracker immutable acceptance criteria remain truncated relative to `docs/plan.md`.
- No manifest-schema restructuring in this round.

### Success Criteria
- `run_simulation(...)` resolves policy before layer construction and does not instantiate disabled layers.
- `check_required_data(...)` accepts nested `layers.l1_macro.tle.file` under strict project-root-relative configs.
- Regression tests prove policy-disabled layer construction is skipped and nested TLE strict validation succeeds from a cwd outside `project_root`.
