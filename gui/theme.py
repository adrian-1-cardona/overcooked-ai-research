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

# RetroUI Gray Palette (Inspired by retroui.io/components with neutral gray & neo-brutalist styling)
BG_WINDOW: Color = (232, 227, 231)            # RetroUI canvas stone gray (#e8e3e7)
BG_GAME_FRAME: Color = (24, 25, 28)           # Inset dark canvas backing for kitchen
BG_SIDEBAR: Color = (220, 218, 222)           # Retro sidebar panel gray
BG_CARD: Color = (255, 255, 255)              # Crisp white retro card
BG_CARD_ALT: Color = (245, 245, 248)          # Light pewter secondary card
BG_CARD_HOVER: Color = (240, 242, 246)        # Card hover state
BG_INPUT: Color = (255, 255, 255)             # Retro input surface
BG_INPUT_HOVER: Color = (236, 238, 242)       # Input hover
BG_INPUT_ACTIVE: Color = (210, 214, 220)      # Active / pressed input

# Retro Solid Borders and Hard Shadows
BORDER_BLACK: Color = (0, 0, 0)
SHADOW_BLACK: Color = (0, 0, 0)
BORDER_DEFAULT: Color = (0, 0, 0)
BORDER_FOCUS: Color = (0, 0, 0)
BORDER_SUBTLE: Color = (0, 0, 0)
DIVIDER_COLOR: Color = (0, 0, 0)

# Typography Colors (Maximum contrast to prevent fuzziness on Retina screens)
TEXT_PRIMARY: Color = (0, 0, 0)               # Pure crisp solid black
TEXT_SECONDARY: Color = (65, 70, 80)          # Slate gray
TEXT_MUTED: Color = (110, 115, 125)           # Muted gray
TEXT_DARK: Color = (0, 0, 0)

# Accents (Muted retro tones with solid black borders)
COLOR_AGENT_RED: Color = (215, 55, 55)        # Agent 1 Red Hat
COLOR_AGENT_RED_BG: Color = (255, 232, 232)
COLOR_AGENT_BLUE: Color = (45, 100, 205)      # Agent 2 Blue Hat
COLOR_AGENT_BLUE_BG: Color = (230, 240, 255)

COLOR_RUN: Color = (210, 214, 220)            # Retro gray button
COLOR_RUN_HOVER: Color = (195, 200, 208)
COLOR_RUN_TEXT: Color = (0, 0, 0)

COLOR_PAUSE: Color = (225, 185, 110)          # Retro warm amber
COLOR_PAUSE_HOVER: Color = (210, 170, 95)
COLOR_PAUSE_TEXT: Color = (0, 0, 0)

COLOR_RESET: Color = (210, 214, 220)          # Retro gray
COLOR_RESET_HOVER: Color = (195, 200, 208)
COLOR_RESET_TEXT: Color = (0, 0, 0)

COLOR_GRAPH: Color = (195, 200, 210)          # Retro slate gray
COLOR_GRAPH_HOVER: Color = (180, 185, 195)
COLOR_GRAPH_TEXT: Color = (0, 0, 0)

COLOR_DONE_BADGE: Color = (45, 135, 80)       # Retro muted green
COLOR_DONE_BADGE_BG: Color = (230, 245, 235)


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
            # Match genuine TrueType font files: Geneva provides classic retro Mac OS crispness
            font_candidates = ["geneva", "arial", "helvetica", "dejavusans"]
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
                    font = pygame.font.SysFont("geneva", size, bold=bold)
                except Exception:
                    font = pygame.font.Font(None, size)
            cls._fonts[key] = font
        return cls._fonts[key]

    @classmethod
    def get_mono_font(cls, size: int = 13, bold: bool = False) -> pygame.font.Font:
        """Returns a crisp monospace font for numbers, telemetry, and metrics."""
        pygame.font.init()
        key = ("mono", size, bold)
        if key not in cls._fonts:
            font: pygame.font.Font | None = None
            mono_candidates = ["monaco", "menlo", "couriernew", "dejavusansmono"]
            for name in mono_candidates:
                path = pygame.font.match_font(name, bold=bold)
                if path and Path(path).is_file():
                    try:
                        font = pygame.font.Font(path, size)
                        break
                    except Exception:
                        continue
            if font is None:
                try:
                    font = pygame.font.SysFont("monaco", size, bold=bold)
                except Exception:
                    font = cls.get_font(size, bold=bold)
            cls._fonts[key] = font
        return cls._fonts[key]
