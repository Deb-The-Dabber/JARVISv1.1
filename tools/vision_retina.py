import cv2
import numpy as np
import time

# ============================================================
# JARVIS ARTIFICIAL VISION v0.3
# Enhanced 320 × 180 SPIKING RETINA WITH MOTION & NOISE HANDLING
# ============================================================

# ============================================================
# CONFIGURATION
# ============================================================
RETINA_WIDTH = 320
RETINA_HEIGHT = 180

THRESHOLD = 0.15          # Spike initiation threshold
LEAK = 0.90               # Membrane leak factor
REFRACTORY_PERIOD = 2     # Refractory timesteps
NOISE_LEVEL = 0.02        # Fraction of max pixel noise to inject
MOTION_SENSITIVITY = 0.05 # Minimum change magnitude to consider motion

# ============================================================
# ARTIFICIAL RETINA
# ============================================================
class ArtificialRetina:
    """Converts camera frames into temporal change maps."""
    def __init__(self, width=RETINA_WIDTH, height=RETINA_HEIGHT):
        self.width = width
        self.height = height
        self.previous = None

    def process(self, frame):
        """Return absolute brightness change (temporal edge)"""
        # Grayscale conversion
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Resize to retinal resolution
        gray = cv2.resize(
            gray