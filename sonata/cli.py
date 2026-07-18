# sonata/cli.py
"""Command-line entry points for SONATA.

Two subcommands wrap the library so common runs need no bespoke script:

* ``sonata-features`` — build (and cache) per-subject features in parallel,
  superseding the old standalone ``parallel_features.py``.
* ``sonata-run`` — build features if needed, then run leakage-safe CV + baselines.

Both read paths and hyperparameters from :class:`sonata.config.SonataConfig`, so
the only thing to edit before a run is that one config (or pass ``--manifest``).
"""

from __future__ import annotations

import argparse
import sys


def features_main(argv: list[str] | None = None) -> int:
    """Entry point for ``sonata-features``: parallel per-subject feature build."""
    ap = argparse.ArgumentParser(prog="sonata-features",
                                 description="Build per-subject SONATA features (parallel).")
    ap.add_argument("--n-threads", type=int, default=-1,
                    help="1 serial, >=2 workers, -1 all usable cores (capped 22).")
    ap.add_argument("--force", action="store_true", help="recompute cached subjects.")
    ap.add_argument("--manifest", default=None, help="override manifest CSV path.")
    args = ap.parse_args(argv)

    from .config import SonataConfig
    from .graph import build_all_subject_features
    from .utils import get_logger, load_manifest

    log = get_logger("sonata.cli")
    cfg = SonataConfig()
    cfg.ensure()
    manifest_path = args.manifest or cfg.paths.manifest_csv
    manifest = load_manifest(manifest_path)
    log.info("building features for %d subjects (n_threads=%s)", len(manifest), args.n_threads)

    results = build_all_subject_features(manifest, cfg, n_threads=args.n_threads,
                                         force=args.force)
    ok = sum(r["status"] in ("ok", "cached") for r in results)
    failed = [r for r in results if r["status"] == "FAILED"]
    log.info("done: %d/%d ok, %d failed", ok, len(results), len(failed))
    for r in failed:
        log.error("  %s: %s", r["sid"], r.get("err", ""))
    return 1 if failed else 0


def run_main(argv: list[str] | None = None) -> int:
    """Entry point for ``sonata-run``: features (if needed) then CV + baselines."""
    ap = argparse.ArgumentParser(prog="sonata-run",
                                 description="Run the SONATA pipeline (CV + baselines).")
    ap.add_argument("--n-threads", type=int, default=-1)
    ap.add_argument("--tag", default=None, help="output tag (defaults to cfg.cohort).")
    ap.add_argument("--skip-features", action="store_true",
                    help="assume the feature cache is already populated.")
    args = ap.parse_args(argv)

    from .config import SonataConfig
    from .graph import build_all_subject_features
    from .graph import load_all_cached_features
    from .train import run_cv
    from .utils import get_logger, load_manifest

    log = get_logger("sonata.cli")
    cfg = SonataConfig()
    cfg.ensure()
    manifest = load_manifest(cfg.paths.manifest_csv)

    if not args.skip_features:
        build_all_subject_features(manifest, cfg, n_threads=args.n_threads)

    feats = load_all_cached_features(cfg)
    tag = args.tag or cfg.cohort
    log.info("running CV on %d cached subjects (tag=%s)", len(feats), tag)
    run_cv(feats, cfg, tag=tag)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(features_main())
