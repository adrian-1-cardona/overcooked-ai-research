"""Visual styling, modern dark color themes, fonts, and constants for Overcooked GUI."""

from __future__ import annotations

import pygame
from typing import Tuple

Color = Tuple[int, int, int]
ColorA = Tuple[int, int, int, int]

# Window and layout sizing
WINDOW_DEFAULT_WIDTH = 1120
WINDOW_DEFAULT_HEIGHT = 760
SIDEBAR_WIDTH = 340
HEADER_HEIGHT = 50

# Modern Dark Theme Palette
BG_WINDOW: Color = (22, 25, 34)            # Deep dark charcoal
BG_GAME_FRAME: Color = (15, 17, 23)        # Slightly darker canvas backing
BG_SIDEBAR: Color = (28, 32, 45)           # Elegant sidebar panel
BG_CARD: Color = (36, 41, 56)              # Card background
BG_CARD_HOVER: Color = (46, 52, 70)        # Card hover state
BG_INPUT: Color = (30, 34, 48)             # Dropdown / input background
BG_INPUT_HOVER: Color = (42, 48, 66)       # Dropdown hover
BG_INPUT_ACTIVE: Color = (52, 60, 82)      # Active / pressed input

# Borders and Dividers
BORDER_DEFAULT: Color = (50, 58, 80)
BORDER_FOCUS: Color = (99, 102, 241)       # Indigo highlight
BORDER_SUBTLE: Color = (38, 44, 62)
DIVIDER_COLOR: Color = (45, 52, 72)

# Typography Colors
TEXT_PRIMARY: Color = (243, 244, 246)      # Pure crisp off-white
TEXT_SECONDARY: Color = (156, 163, 175)    # Muted silver
TEXT_MUTED: Color = (107, 114, 128)        # Darker gray
TEXT_DARK: Color = (17, 24, 39)            # Dark for light badges

# Accents
COLOR_AGENT_RED: Color = (239, 68, 68)     # Agent 1 Red Hat
COLOR_AGENT_RED_BG: Color = (80, 20, 25)
COLOR_AGENT_BLUE: Color = (59, 130, 246)   # Agent 2 Blue Hat
COLOR_AGENT_BLUE_BG: Color = (20, 45, 85)

COLOR_RUN: Color = (16, 185, 129)          # Emerald green
COLOR_RUN_HOVER: Color = (5, 150, 105)
COLOR_RUN_TEXT: Color = (255, 255, 255)

COLOR_PAUSE: Color = (245, 158, 11)        # Warm amber
COLOR_PAUSE_HOVER: Color = (217, 119, 6)
COLOR_PAUSE_TEXT: Color = (255, 255, 255)

COLOR_RESET: Color = (75, 85, 99)          # Slate gray
COLOR_RESET_HOVER: Color = (107, 114, 128)
COLOR_RESET_TEXT: Color = (255, 255, 255)

COLOR_GRAPH: Color = (139, 92, 246)        # Purple / Violet (Streamlit vibe)
COLOR_GRAPH_HOVER: Color = (124, 58, 237)
COLOR_GRAPH_TEXT: Color = (255, 255, 255)

COLOR_DONE_BADGE: Color = (16, 185, 129)   # Celebratory green
COLOR_DONE_BADGE_BG: Color = (20, 60, 40)


from pathlib import Path


class FontManager:
    """Manages system fonts across different sizes with crisp TrueType rendering."""

    _fonts: dict[tuple[str, int, bool], pygame.font.Font] = {}

    @classmethod
    def get_font(cls, size: int = 14, bold: bool = False) -> pygame.font.Font:
        pygame.font.init()
        key = ("default", size, bold)
        if key not in cls._fonts:
            font: pygame.font.Font | None = None
            # Match genuine TrueType font files with native bold weights to avoid blurry fake bolding
            font_candidates = ["arial", "trebuchetms", "verdana", "dejavusans"]
            for name in font_candidates:
                path = pygame.font.match_font(name, bold=bold)
                if path and Path(path).is_file():
                    try:
                        font = pygame.font.Font(path, size)
                        break
                    except Exception:
                        continue
            if font is None:
                try:
                    font = pygame.font.SysFont("arial", size, bold=bold)
                except Exception:
                    font = pygame.font.Font(None, size)
            cls._fonts[key] = font
        return cls._fonts[key]
