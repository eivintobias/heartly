#!/usr/bin/env python3
"""
boundary_head.py — Optional quality gate for Heartly model output.

The boundary head is a trained logistic regression classifier that reads
the model's internal hidden state at the <verify> token position and
predicts whether the model actually knows the answer.

This is an OPTIONAL speed optimization:
- If the boundary head says "unknown" with high confidence, skip the
  expensive Docker evaluation and mark the program as low fitness.
- If it says "known", proceed normally.

The boundary head is trained by heartly-qwen-code/train_probe_code.py
and saved as probe_head.pkl.

NOTE: The boundary head is blind to confident confabulations (cases where
the model thinks it knows but the code is wrong). The EvaluatorAgent's
test suite is the real quality check.
"""
import logging
import pickle
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class BoundaryHead:
    """Logistic regression probe on hidden states at <verify> position."""

    def __init__(self, probe_path: str):
        """Load a trained probe from disk.

        Args:
            probe_path: Path to the probe_head.pkl file.
        """
        probe_path = Path(probe_path)
        if not probe_path.exists():
            raise FileNotFoundError(f"Boundary head not found at {probe_path}")

        with open(probe_path, "rb") as f:
            self.probe = pickle.load(f)
        logger.info(f"Boundary head loaded from {probe_path}")

    def predict(self, hidden_state: np.ndarray) -> int:
        """Predict known (1) vs unknown (0).

        Args:
            hidden_state: Hidden state vector from the model's last layer
                          at the <verify> token position.

        Returns:
            1 if known, 0 if unknown.
        """
        return int(self.probe.predict(hidden_state.reshape(1, -1))[0])

    def predict_proba(self, hidden_state: np.ndarray) -> float:
        """Get confidence score for 'known' class.

        Args:
            hidden_state: Hidden state vector.

        Returns:
            Probability of 'known' class (0.0 to 1.0).
        """
        return float(self.probe.predict_proba(hidden_state.reshape(1, -1))[0, 1])

    def is_known(self, hidden_state: np.ndarray, threshold: float = 0.5) -> bool:
        """Check if the model likely knows the answer.

        Args:
            hidden_state: Hidden state vector.
            threshold: Confidence threshold (default 0.5).

        Returns:
            True if known confidence >= threshold.
        """
        return self.predict_proba(hidden_state) >= threshold
