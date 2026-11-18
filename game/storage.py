"""
Persistent local JSON storage for Top-5 High Scores and User Sensitivity/Audio Settings.
"""

import json
import os
from datetime import datetime

from game.config import (
    HIGHSCORE_FILE, SETTINGS_FILE, MAX_HIGHSCORES, DEFAULT_SETTINGS
)


def load_highscores(filepath=HIGHSCORE_FILE):
    """Loads sorted list of top high score dicts from disk."""
    if not os.path.exists(filepath):
        return []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return sorted(data, key=lambda x: x.get("score", 0), reverse=True)[:MAX_HIGHSCORES]
    except Exception:
        pass
    return []


def save_highscore(score, distance, max_streak, coins=0, filepath=HIGHSCORE_FILE):
    """
    Records a completed run if score > 0, keeps top MAX_HIGHSCORES,
    and returns (updated_list, is_new_record).
    """
    scores = load_highscores(filepath)
    prev_best = scores[0]["score"] if scores else 0
    is_new_record = score > prev_best and score > 0

    if score > 0:
        entry = {
            "score": int(score),
            "distance": int(distance),
            "max_streak": int(max_streak),
            "coins": int(coins),
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
        scores.append(entry)
        scores = sorted(scores, key=lambda x: x.get("score", 0), reverse=True)[:MAX_HIGHSCORES]
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(scores, f, indent=2)
        except Exception as e:
            print(f"[Storage] Could not save highscores: {e}")

    return scores, is_new_record


def load_settings(filepath=SETTINGS_FILE):
    """Loads saved user sensitivity and audio settings with safe defaults."""
    settings = dict(DEFAULT_SETTINGS)
    if not os.path.exists(filepath):
        return settings
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            for k in DEFAULT_SETTINGS:
                if k in loaded:
                    settings[k] = loaded[k]
    except Exception:
        pass
    return settings


def save_settings(settings, filepath=SETTINGS_FILE):
    """Persists user sensitivity and audio settings to disk."""
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)
    except Exception as e:
        print(f"[Storage] Could not save settings: {e}")
