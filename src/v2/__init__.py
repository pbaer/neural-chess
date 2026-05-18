# -*- coding: utf-8 -*-
"""v2 chess architecture: 21-plane history-aware input, 8x8x73 move output,
policy + value heads. T0a baseline is a plain ResNet; later tiers add a
differentiable lookahead block.

See data/v2/README.md for the corpus this trains on.
See memory/project-principles.md for the experimental constraints
(only games are training signal — no computed chess features anywhere).
"""
