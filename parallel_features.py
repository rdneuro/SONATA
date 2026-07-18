#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DEPRECATED — superseded by the integrated parallel builder.

The subject-level parallel feature build now lives inside the library and honours
the unified ``n_threads`` convention. Use either:

    from sonata.graph import build_all_subject_features   # in Python / Spyder
    # or, from the shell:
    sonata-features --n-threads -1

This shim forwards to the new CLI so existing muscle memory keeps working.
"""
from __future__ import annotations

import sys

if __name__ == "__main__":
    from sonata.cli import features_main

    print("[deprecated] parallel_features.py -> sonata-features "
          "(sonata.graph.build_all_subject_features)")
    sys.exit(features_main(sys.argv[1:]))
