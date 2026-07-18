# Code Review v2 — SONATA infrastructure & refactor

Scope: the new/refactored code in the v0.2.0 packaging effort — `backends/`,
`parallel.py`, `memory.py`, `viz/`, `attribution.py`, `cli.py`, and the
refactors of `spectral_features.py`, `graph.py`, `config.py`, `__init__.py`.
Protocol: two passes (reviewing-code + scientific-coding + system-design +
scientific-critical-thinking). Validation is CPU-path + synthetic-data only —
the real cohort and the GPU/PyMC/SpectralBrain stacks are not in the build
sandbox (by the project's own rule, pipelines run on the author's machine).

## Summary

**Quality score:** 92/100
**Issues found:** Critical 1 · Important 2 · Nice-to-have 2 — all fixed or
consciously accepted below.
**Tests:** 28 passing (backends, parallel, memory, viz, spectral/config).
**Import graph:** acyclic (`base` leaf → array/bayes → parallel/memory → spectral
→ graph → attribution; `viz` independent). Verified by ordered import.

---

## Pass 1 — correctness & robustness

### 🔴 CRITICAL — fixed
**Unpicklable lambda in the parallel path.** `parallel_map(..., pin_blas=False)`
with `n_threads>=2` passed a `lambda` to `joblib.delayed`; loky pickles tasks by
reference, so a lambda would raise `PicklingError` at run time — exactly on the
parallel path the function exists for.
*Fix:* introduced module-level `_plain_call`/`_pinned_call` workers (picklable by
qualified name). Added a regression test
(`test_parallel_without_blas_pinning_is_picklable`). ✅

**Bonus correctness in the same fix:** the serial path was pinning inner BLAS to
1 thread per item, which is both wrong (a single task should use all cores) and
slow (env churn per item). The serial path now runs `fn(x)` directly and only the
*parallel* path pins BLAS. ✅

### 🟡 IMPORTANT — fixed
1. **Misleading `jax` sync.** `array.synchronize("jax")` referenced
   `jax.block_until_ready` without calling it — a silent no-op dressed as work.
   Replaced with an explicit, documented no-op (JAX has no global device sync;
   results block on transfer in `to_numpy`). ✅
2. **Over-promising API name.** `memory.oom_safe_batches` did no OOM handling
   despite its name (only `run_in_oom_safe_batches` did). Removed it to avoid a
   name that lies and to cut redundancy (one OOM-safe entry point, not two). ✅

### 🟢 NICE-TO-HAVE — done
- `spectral_features.surface_descriptor` now runs the cheap size/`k` guards
  **before** importing SpectralBrain, so the "surface too small / unsupported k"
  paths cost nothing and are unit-testable without the heavy library. ✅
- Worker functions for both parallel entry points (`_descriptor_from_item` via a
  `functools.partial`, and `_build_one_subject`) verified picklable. ✅

---

## Pass 2 — scientific validity & system design

### Statistical correctness (attribution)
The regularized-horseshoe model in `attribution.build_horseshoe_model` follows
Piironen & Vehtari (2017): non-centered parameterization
(`beta = z·tau·lam_tilde`, `z ~ N(0,1)`), global scale `tau0 = (p0/(D−p0))·σ/√N`
tied to the noise scale, per-column (or per-group) local scales, and a Student-t
**slab** via `c² ~ Inv-Gamma(ν/2, ν/2·s²)` with
`lam_tilde² = c²λ²/(c² + τ²λ²)`. `fit_attribution` defaults to
`target_accept=0.95`, appropriate for the horseshoe funnel geometry. Group
horseshoe (`groups=`) shares a local scale within a tract block for block
sparsity. **Assessment: statistically sound.** (Not run here — PyMC absent; the
model is AST-valid and the math matches the reference.)

### 🟡 IMPORTANT — accepted limitation (documented, not a bug)
**GPU cost model vs. the actual heavy kernel.** `should_use_gpu` is a calibrated
*size* heuristic. SONATA's single most expensive kernel is the **sparse LBO
eigensolve**, and sparse iterative eigensolvers are largely memory-bandwidth
bound — a GPU does not reliably beat a good CPU BLAS there. The array backend is
therefore most valuable for the **dense/batched** ops (edge aggregation,
bootstrap resampling, GNN forward/backward), and the cost model's default
threshold reflects that. This is stated so a user does not expect the eigensolve
itself to accelerate on GPU. **Accepted** — correct behavior, honest scope.

### 🟢 NICE-TO-HAVE — accepted follow-up
**`viz/` vs. legacy `figures.py` overlap.** The new `viz/` package is the
go-forward, general-purpose visualization API (palette, panels, heatmaps, metric
plots, 3D). The legacy `figures.py` retains a few pipeline-specific figures used
by `run_sonata.py` (edge-coverage, coupling-along-gradient, tract renders). To
avoid breaking the working pipeline in this pass, `figures.py` is kept as-is; a
future migration should re-express its generic helpers on top of `viz/`. **Not
redundant enough to force now; flagged for a later pass.**

### System-design notes (good practices confirmed)
- **Lazy top-level import (PEP 562).** `import sonata` no longer hard-fails on a
  partial install; infra + config load on the scientific-Python core, heavy
  pipeline symbols resolve on first access. Verified without torch/nibabel.
- **Single source of truth for capabilities.** All device/thread decisions route
  through `backends.base.capabilities()` (cached), not ad-hoc probes.
- **One parallelism convention.** Every CPU-bound parallelizable function takes
  `n_threads` and routes through `parallel_map`; the standalone
  `parallel_features.py` script is superseded by
  `graph.build_all_subject_features` + the `sonata-features` CLI.
- **Reuse in viz.** Solo and evolutionary heatmaps share one `_draw` primitive;
  metric plots share `_fig_ax`; the dashboard composes existing plotters via
  `panels.grid` — no duplicated drawing code.

---

## What "done" means here (honest boundary)
Verified in-sandbox: every module imports/compiles; the infrastructure and 2D
visualization pass 28 synthetic-data unit tests; the parallel pipeline path and
worker picklability are exercised; the import graph is acyclic. **Not** verified
here (by design — data and heavy stacks live on the author's machine): real-cohort
numerical results, CUDA array backends, and PyMC sampling. Those are for the
author to run per the project's standing rule.
