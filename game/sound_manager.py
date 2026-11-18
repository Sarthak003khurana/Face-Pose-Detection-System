"""
Procedural sound & synthwave background music synthesizer.
Generates all SFX and a seamless looping 120 BPM retro synthwave soundtrack
in-memory using NumPy and Pygame mixer (zero external audio files required).
"""

import numpy as np
import pygame


class SoundManager:
    def __init__(self, sample_rate=44100, music_enabled=True):
        self.sample_rate = sample_rate
        self.enabled = True
        self.music_enabled = bool(music_enabled)
        self.sounds = {}
        self.music_sound = None
        self.music_channel = None

        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=self.sample_rate, size=-16, channels=2, buffer=512)
            pygame.mixer.set_num_channels(16)
            self.music_channel = pygame.mixer.Channel(0)
            self._generate_procedural_sounds()
            self._generate_synthwave_loop()
            if self.music_enabled:
                self.start_music()
        except Exception as e:
            print(f"[SoundManager] Audio initialization failed: {e}. Running in silent mode.")
            self.enabled = False

    def _make_stereo_sound(self, mono_samples, volume=1.0):
        """Converts a 1D float array [-1.0, 1.0] to a 2D int16 stereo Pygame Sound."""
        scaled = np.clip(mono_samples * volume, -1.0, 1.0)
        mono_int16 = (scaled * 32767).astype(np.int16)
        stereo = np.column_stack([mono_int16, mono_int16])
        return pygame.sndarray.make_sound(stereo)

    def _generate_procedural_sounds(self):
        """Synthesizes arcade SFX procedurally."""
        sr = self.sample_rate

        # 1. Morph Sound (Sci-Fi Frequency Sweep)
        dur = 0.16
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        freq = np.linspace(260, 840, len(t))
        wave_morph = np.sin(2 * np.pi * freq * t) * np.linspace(0.35, 0.0, len(t))
        self.sounds["morph"] = self._make_stereo_sound(wave_morph)

        # 2. Obstacle Clear Chime (Dual Sine Chord)
        dur = 0.20
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        wave_clear = (0.5 * np.sin(2 * np.pi * 587.33 * t) +
                      0.5 * np.sin(2 * np.pi * 880.00 * t))
        self.sounds["clear"] = self._make_stereo_sound(wave_clear * np.exp(-12 * t) * 0.40)

        # 3. Coin Pickup (Classic B5 -> E6 Bright Ping)
        dur = 0.18
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        f_coin = np.where(t < 0.06, 987.77, 1318.51)
        wave_coin = (np.sin(2 * np.pi * f_coin * t) + 0.3 * np.sin(4 * np.pi * f_coin * t))
        self.sounds["coin"] = self._make_stereo_sound(wave_coin * np.exp(-11 * t) * 0.35)

        # 4. Streak Multiplier Up (Ascending 4-Note Synth Arpeggio)
        dur = 0.34
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        notes = [523.25, 659.25, 783.99, 1046.50]  # C5 E5 G5 C6
        idx = np.clip((t / dur * len(notes)).astype(int), 0, len(notes) - 1)
        f_arp = np.array(notes)[idx]
        local_t = (t % (dur / len(notes)))
        env = np.exp(-16 * local_t)
        wave_streak = np.sin(2 * np.pi * f_arp * t) * env * 0.42
        self.sounds["streak"] = self._make_stereo_sound(wave_streak)

        # 5. Near-Miss Whoosh Ping
        dur = 0.20
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        f_nm = np.linspace(620, 1180, len(t))
        wave_nm = np.sin(2 * np.pi * f_nm * t) * np.exp(-10 * t) * 0.32
        self.sounds["near_miss"] = self._make_stereo_sound(wave_nm)

        # 6. Biome Transition Shimmer
        dur = 0.45
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        wave_bio = (
            0.4 * np.sin(2 * np.pi * 440.0 * t) +
            0.35 * np.sin(2 * np.pi * 659.25 * t) +
            0.25 * np.sin(2 * np.pi * 880.0 * t)
        ) * np.sin(np.pi * t / dur) * 0.38
        self.sounds["biome"] = self._make_stereo_sound(wave_bio)

        # 7. Crash / Explosion Sound (Noise + Low Rumble)
        dur = 0.35
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        noise = np.random.uniform(-1.0, 1.0, len(t))
        rumble = np.sin(2 * np.pi * 72 * t)
        wave_crash = (noise * 0.65 + rumble * 0.35) * np.exp(-7.5 * t) * 0.60
        self.sounds["crash"] = self._make_stereo_sound(wave_crash)

        # 8. Calibration Done Chime
        dur = 0.30
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        f_cal = np.where(t < 0.1, 440, np.where(t < 0.2, 554.37, 659.25))
        wave_cal = np.sin(2 * np.pi * f_cal * t) * np.linspace(0.38, 0.05, len(t))
        self.sounds["calibrated"] = self._make_stereo_sound(wave_cal)

        # 9. Jump Spring Whoosh (Eyebrow Raise)
        dur = 0.22
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        f_jmp = 220.0 + 520.0 * (t / dur) ** 0.7
        wave_jmp = np.sin(2 * np.pi * np.cumsum(f_jmp) / sr) * np.exp(-7.0 * t) * 0.42
        self.sounds["jump"] = self._make_stereo_sound(wave_jmp)

        # 10. Energy Shield + Nitro Power-Up Chord (Wink)
        dur = 0.38
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        f_sweep = np.linspace(330.0, 990.0, len(t))
        wave_sh = (
            0.45 * np.sin(2 * np.pi * np.cumsum(f_sweep) / sr) +
            0.35 * np.sin(2 * np.pi * 659.25 * t) +
            0.20 * np.sin(2 * np.pi * 987.77 * t)
        ) * np.sin(np.pi * t / dur) * 0.45
        self.sounds["shield"] = self._make_stereo_sound(wave_sh)

        # 11. Shield Smash Impact (Bonus shatter)
        dur = 0.24
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        noise = np.random.uniform(-1.0, 1.0, len(t))
        ring = np.sin(2 * np.pi * 1120.0 * t)
        wave_smash = (0.45 * noise + 0.55 * ring) * np.exp(-11.0 * t) * 0.48
        self.sounds["smash"] = self._make_stereo_sound(wave_smash)

    def _generate_synthwave_loop(self):
        """
        Synthesizes an 8-second (4 bars @ 120 BPM) seamless looping retro synthwave track:
        Driving 8th-note Am -> F -> C -> G saw/triangle bassline + shimmering arpeggio + kick.
        """
        sr = self.sample_rate
        bpm = 120.0
        beat_sec = 60.0 / bpm            # 0.5 s per beat
        total_beats = 16                 # 4 bars (8.0 seconds)
        total_sec = beat_sec * total_beats
        n_samples = int(sr * total_sec)
        t = np.linspace(0, total_sec, n_samples, endpoint=False)

        # Chord progression: 4 beats each of Am, F, C, G
        bass_freqs = [110.0, 87.31, 130.81, 98.0]  # A2, F2, C3, G2
        arp_chords = [
            [220.0, 261.63, 329.63, 440.0],        # Am
            [174.61, 220.0, 261.63, 349.23],       # F
            [261.63, 329.63, 392.0, 523.25],       # C
            [196.0, 246.94, 293.66, 392.0],        # G
        ]

        bar_idx = np.clip((t / (beat_sec * 4)).astype(int), 0, 3)

        # 1. Driving 8th-note Synth Bass (saw-ish harmonics with plucky envelope)
        eighth_dur = beat_sec / 2.0
        t_eighth = t % eighth_dur
        f_bass = np.array(bass_freqs)[bar_idx]
        phase_bass = 2 * np.pi * np.cumsum(f_bass) / sr
        bass_raw = (
            0.65 * np.sin(phase_bass) +
            0.25 * np.sin(2 * phase_bass) +
            0.12 * np.sin(3 * phase_bass)
        )
        bass_env = np.exp(-9.5 * t_eighth) * np.minimum(1.0, t_eighth / 0.006)
        bass = bass_raw * bass_env * 0.42

        # 2. Sparkling 16th-note Retro Arpeggio
        sixteenth_dur = beat_sec / 4.0
        step_idx = (t / sixteenth_dur).astype(int) % 4
        t_sixteenth = t % sixteenth_dur
        chord_table = np.array(arp_chords)
        f_arp = chord_table[bar_idx, step_idx]
        phase_arp = 2 * np.pi * np.cumsum(f_arp) / sr
        arp_raw = 0.7 * np.sin(phase_arp) + 0.25 * np.sin(2 * phase_arp)
        arp_env = np.exp(-16.0 * t_sixteenth) * np.minimum(1.0, t_sixteenth / 0.004)
        arp = arp_raw * arp_env * 0.22

        # 3. Four-on-the-Floor Punchy Kick Drum (every beat)
        t_beat = t % beat_sec
        kick_freq = 48.0 + 95.0 * np.exp(-32.0 * t_beat)
        kick_phase = 2 * np.pi * np.cumsum(kick_freq) / sr
        kick = np.sin(kick_phase) * np.exp(-14.0 * t_beat) * 0.34

        # Mix & soft fade at exact loop boundaries to avoid click
        mix = bass + arp + kick
        fade = int(sr * 0.01)
        mix[:fade] *= np.linspace(0.0, 1.0, fade)
        mix[-fade:] *= np.linspace(1.0, 0.0, fade)

        self.music_sound = self._make_stereo_sound(mix, volume=0.28)

    def start_music(self):
        """Starts or resumes the procedural synthwave background loop."""
        if not self.enabled or not self.music_sound or not self.music_channel:
            return
        self.music_enabled = True
        if not self.music_channel.get_busy():
            self.music_channel.play(self.music_sound, loops=-1)

    def stop_music(self):
        """Stops the background synthwave music."""
        self.music_enabled = False
        if self.music_channel:
            self.music_channel.stop()

    def toggle_music(self):
        """Toggles background synthwave music ON/OFF and returns new state."""
        if self.music_enabled:
            self.stop_music()
        else:
            self.start_music()
        return self.music_enabled

    def play(self, sound_name):
        """Plays a registered procedural sound effect on any free SFX channel."""
        if not self.enabled:
            return
        snd = self.sounds.get(sound_name)
        if snd:
            ch = pygame.mixer.find_channel()
            if ch and ch != self.music_channel:
                ch.play(snd)
            else:
                snd.play()
