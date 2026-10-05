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

# RetroUI "Muddy" Brown Theme Palette (from retroui.io/components)
BG_WINDOW: Color = (191, 184, 172)          # RetroUI pageBg (#bfb8ac)
BG_GAME_FRAME: Color = (28, 22, 16)         # Deep contrast backing for game canvas
BG_SIDEBAR: Color = (191, 184, 172)         # RetroUI muddy sidebar (#bfb8ac)
BG_GRID: Color = (178, 171, 158)            # Subtle 24px pixel grid line
BG_CARD: Color = (255, 255, 255)            # RetroUI cardBg (#ffffff)
BG_CARD_HOVER: Color = (250, 248, 242)      # Soft card hover
BG_INPUT: Color = (221, 206, 180)           # RetroUI muddy bg tan (#ddceb4)
BG_INPUT_HOVER: Color = (233, 221, 201)     # Warm input hover
BG_INPUT_ACTIVE: Color = (206, 188, 158)    # Active / pressed input

# RetroUI Chunky Pixel Borders & Hard Drop-Shadows
BORDER_DEFAULT: Color = (48, 33, 11)        # RetroUI muddy border & shadow (#30210b)
BORDER_FOCUS: Color = (98, 71, 31)          # RetroUI cocoa accent (#62471f)
BORDER_SUBTLE: Color = (150, 135, 115)      # Warm subtle divider/border
BORDER_RETRO_LIGHT: Color = (245, 238, 225) # Soft highlight
BORDER_RETRO_DARK: Color = (48, 33, 11)     # Hard pixel shadow (#30210b)
DIVIDER_COLOR: Color = (48, 33, 11)         # Solid 3px divider (#30210b)

# Typography Colors
TEXT_PRIMARY: Color = (48, 33, 11)          # RetroUI deep espresso text (#30210b)
TEXT_SECONDARY: Color = (104, 82, 60)       # Warm medium brown (#68523c)
TEXT_MUTED: Color = (140, 122, 102)         # Muted brown
TEXT_CREAM: Color = (254, 252, 208)         # RetroUI cream text (#fefcd0)
TEXT_GOLD: Color = (196, 126, 26)           # Warm retro score gold (#c47e1a)
TEXT_DARK: Color = (48, 33, 11)

# RetroUI Component Accents
COLOR_ACCENT_BROWN: Color = (98, 71, 31)    # RetroUI signature cocoa brown (#62471f)
COLOR_AGENT_RED: Color = (201, 64, 56)      # Retro 1P Warm Red (#c94038)
COLOR_AGENT_BLUE: Color = (59, 119, 168)    # Retro 2P Slate Blue (#3b77a8)

# Buttons (RetroUI Hard Pixel Drop-Shadow Style)
COLOR_RUN: Color = (98, 71, 31)             # RetroUI brown primary accent (#62471f)
COLOR_RUN_HOVER: Color = (120, 88, 40)
COLOR_RUN_TEXT: Color = (254, 252, 208)     # Cream text on dark cocoa

COLOR_PAUSE: Color = (207, 134, 42)         # Warm honey amber (#cf862a)
COLOR_PAUSE_HOVER: Color = (185, 118, 32)
COLOR_PAUSE_TEXT: Color = (255, 255, 255)

COLOR_RESET: Color = (196, 67, 58)          # Warm terracotta red (#c4433a)
COLOR_RESET_HOVER: Color = (175, 52, 44)
COLOR_RESET_TEXT: Color = (255, 255, 255)

COLOR_GRAPH: Color = (98, 71, 31)           # RetroUI cocoa brown (#62471f)
COLOR_GRAPH_HOVER: Color = (120, 88, 40)
COLOR_GRAPH_TEXT: Color = (254, 252, 208)

COLOR_DONE_BADGE: Color = (68, 128, 74)     # Retro sage/olive green (#44804a)
COLOR_DONE_BADGE_BG: Color = (235, 245, 235)


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
