"""Visual styling, retro arcade color themes, fonts, and constants for Overcooked GUI."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple
import pygame

Color = Tuple[int, int, int]
ColorA = Tuple[int, int, int, int]

REPO_ROOT = Path(__file__).resolve().parents[1]
MINECRAFT_FONT = (
    REPO_ROOT
    / "overcooked-agent-eval"
    / ".venv"
    / "lib"
    / "python3.10"
    / "site-packages"
    / "gym"
    / "envs"
    / "toy_text"
    / "font"
    / "Minecraft.ttf"
)

# Window and layout sizing
WINDOW_DEFAULT_WIDTH = 1120
WINDOW_DEFAULT_HEIGHT = 760
SIDEBAR_WIDTH = 340
HEADER_HEIGHT = 50

# Retro Arcade Dark Theme Palette
BG_WINDOW: Color = (16, 18, 28)            # Deep arcade midnight
BG_GAME_FRAME: Color = (10, 12, 18)        # Crisp screen backing
BG_SIDEBAR: Color = (24, 28, 42)           # Arcade cabinet sidebar
BG_CARD: Color = (32, 38, 56)              # Beveled card module
BG_CARD_HOVER: Color = (42, 50, 74)        # Card hover state
BG_INPUT: Color = (20, 24, 36)             # Recessed dark slot
BG_INPUT_HOVER: Color = (34, 40, 60)       # Dropdown hover
BG_INPUT_ACTIVE: Color = (46, 54, 82)      # Active / pressed input

# Retro 3D Bevel Borders
BORDER_DEFAULT: Color = (55, 65, 95)
BORDER_FOCUS: Color = (129, 140, 248)       # Neon highlight
BORDER_SUBTLE: Color = (38, 45, 68)
BORDER_RETRO_LIGHT: Color = (85, 100, 145) # Bevel top-left light
BORDER_RETRO_DARK: Color = (12, 14, 22)    # Bevel bottom-right shadow
DIVIDER_COLOR: Color = (45, 54, 80)

# Typography Colors
TEXT_PRIMARY: Color = (245, 245, 252)      # Crisp off-white pixel text
TEXT_SECONDARY: Color = (180, 190, 215)    # Arcade silver
TEXT_MUTED: Color = (120, 130, 155)        # Muted gray
TEXT_GOLD: Color = (255, 215, 64)          # Arcade score gold
TEXT_DARK: Color = (17, 24, 39)

# Retro Arcade Accents
COLOR_AGENT_RED: Color = (235, 55, 70)     # 1P Arcade Red
COLOR_AGENT_BLUE: Color = (45, 125, 245)   # 2P Arcade Blue

COLOR_RUN: Color = (0, 205, 115)           # Arcade Start Green
COLOR_RUN_HOVER: Color = (0, 175, 95)
COLOR_RUN_TEXT: Color = (255, 255, 255)

COLOR_PAUSE: Color = (250, 160, 25)        # Arcade Pause Amber
COLOR_PAUSE_HOVER: Color = (220, 140, 15)
COLOR_PAUSE_TEXT: Color = (255, 255, 255)

COLOR_RESET: Color = (235, 65, 80)         # Arcade Reset Ruby/Crimson
COLOR_RESET_HOVER: Color = (205, 45, 60)
COLOR_RESET_TEXT: Color = (255, 255, 255)

COLOR_GRAPH: Color = (155, 90, 245)        # Retro Neon Violet
COLOR_GRAPH_HOVER: Color = (135, 70, 225)
COLOR_GRAPH_TEXT: Color = (255, 255, 255)

COLOR_DONE_BADGE: Color = (0, 220, 130)    # Victory Green
COLOR_DONE_BADGE_BG: Color = (15, 55, 35)


class FontManager:
    """Manages retro arcade pixel fonts with fallback to system monospace."""

    _fonts: dict[tuple[str, int, bool], pygame.font.Font] = {}

    @classmethod
    def get_font(cls, size: int = 14, bold: bool = False) -> pygame.font.Font:
        pygame.font.init()
        key = ("retro", size, bold)
        if key not in cls._fonts:
            font = None
            if MINECRAFT_FONT.is_file():
                try:
                    font = pygame.font.Font(str(MINECRAFT_FONT), size)
                except Exception:
                    font = None

            if font is None:
                try:
                    # Fallback to system monospace fonts
                    font_names = ["Courier New", "Menlo", "Courier", "Andale Mono"]
                    font = pygame.font.SysFont(font_names, size, bold=bold)
                except Exception:
                    font = pygame.font.Font(None, size)

            cls._fonts[key] = font
        return cls._fonts[key]
