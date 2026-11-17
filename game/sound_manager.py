"""
Procedural sound generator for game audio effects without requiring external asset files.
Generates sound effects in-memory using NumPy and Pygame mixer.
"""

import numpy as np
import pygame


class SoundManager:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.enabled = True
        self.sounds = {}

        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=self.sample_rate, size=-16, channels=2, buffer=512)
            self._generate_procedural_sounds()
        except Exception as e:
            print(f"[SoundManager] Audio initialization failed: {e}. Running in silent mode.")
            self.enabled = False

    def _make_stereo_sound(self, mono_samples):
        """Converts a 1D float array [-1.0, 1.0] to a 2D int16 stereo Pygame Sound."""
        mono_int16 = (np.clip(mono_samples, -1.0, 1.0) * 32767).astype(np.int16)
        stereo = np.column_stack([mono_int16, mono_int16])
        return pygame.sndarray.make_sound(stereo)

    def _generate_procedural_sounds(self):
        """Synthesizes sound effects procedurally."""
        sr = self.sample_rate

        # 1. Morph Sound (Sci-Fi Laser / Frequency Sweep)
        duration_morph = 0.18
        t = np.linspace(0, duration_morph, int(sr * duration_morph), endpoint=False)
        freq = np.linspace(280, 880, len(t))
        wave_morph = np.sin(2 * np.pi * freq * t) * np.linspace(0.4, 0.0, len(t))
        self.sounds["morph"] = self._make_stereo_sound(wave_morph)

        # 2. Score / Obstacle Clear Chime (Dual Sine Chord)
        duration_clear = 0.22
        t = np.linspace(0, duration_clear, int(sr * duration_clear), endpoint=False)
        wave_clear = (0.5 * np.sin(2 * np.pi * 587.33 * t) +   # D5
                      0.5 * np.sin(2 * np.pi * 880.00 * t))    # A5
        envelope = np.exp(-12 * t)
        self.sounds["clear"] = self._make_stereo_sound(wave_clear * envelope * 0.45)

        # 3. Crash / Explosion Sound (White noise + decaying low rumble)
        duration_crash = 0.35
        t = np.linspace(0, duration_crash, int(sr * duration_crash), endpoint=False)
        noise = np.random.uniform(-1.0, 1.0, len(t))
        rumble = np.sin(2 * np.pi * 75 * t)
        env = np.exp(-7.5 * t)
        wave_crash = (noise * 0.65 + rumble * 0.35) * env * 0.65
        self.sounds["crash"] = self._make_stereo_sound(wave_crash)

        # 4. Calibration Done Chime (Ascending triad)
        duration_cal = 0.30
        t = np.linspace(0, duration_cal, int(sr * duration_cal), endpoint=False)
        f_cal = np.where(t < 0.1, 440, np.where(t < 0.2, 554, 659))
        wave_cal = np.sin(2 * np.pi * f_cal * t) * np.linspace(0.4, 0.05, len(t))
        self.sounds["calibrated"] = self._make_stereo_sound(wave_cal)

    def play(self, sound_name):
        """Plays a registered procedural sound effect."""
        if not self.enabled:
            return
        snd = self.sounds.get(sound_name)
        if snd:
            snd.play()
