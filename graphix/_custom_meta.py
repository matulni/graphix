"""Custom metaclasses."""

# TODO: Maybe merge with repr_mixings.py in a single enum_utils.py module
from __future__ import annotations

from abc import ABCMeta
from enum import EnumMeta


class ABCEnumMeta(ABCMeta, EnumMeta):
    """Custom metaclass to allow multiple inheritance from ``Enum`` and ``ABC``."""
