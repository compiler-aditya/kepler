"""Trusted code that fits theories and judges them. The agent can't change it."""

from .fit import ClipData, FitResult, fit_all, fit_theory, verdict

__all__ = ["ClipData", "FitResult", "fit_all", "fit_theory", "verdict"]
