# -*- coding: utf-8 -*-
"""Architecture-invariant code for neural-chess.

Modules in this package work with any model version through the PolicyEngine
abstraction; version-specific code (model definition, featurization, dataset
format, inference details) lives under src/v1, src/v2, etc.
"""
