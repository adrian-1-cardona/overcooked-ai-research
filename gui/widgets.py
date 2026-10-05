"""Reusable retro-styled UI widgets for the Overcooked interactive application.

Includes retro arcade buttons, dropdown select menus with scrolling, banners, and cards.
"""

from __future__ import annotations

import pygame
from typing import Any, Callable, Sequence, Tuple
from gui.theme import (
    FontManager,
    BG_CARD,
    BG_INPUT,
    BG_INPUT_HOVER,
    BG_INPUT_ACTIVE,
    BORDER_DEFAULT,
    BORDER_FOCUS,
    BORDER_SUBTLE,
    BORDER_RETRO_LIGHT,
    BORDER_RETRO_DARK,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TEXT_MUTED,
)


def _adjust_color(color: tuple[int, int, int], amount: int) -> tuple[int, int, int]:
    return (
        max(0, min(255, color[0] + amount)),
        max(0, min(255, color[1] + amount)),
        max(0, min(255, color[2] + amount)),
    )


class UIButton:
    """A RetroUI pixel-perfect button with chunky border and hard drop-shadow."""

    def __init__(
        self,
        rect: pygame.Rect | tuple[int, int, int, int],
        text: str,
        on_click: Callable[[], None] | None = None,
        bg_color: tuple[int, int, int] = BG_INPUT,
        hover_color: tuple[int, int, int] = BG_INPUT_HOVER,
        text_color: tuple[int, int, int] = TEXT_PRIMARY,
        font_size: int = 14,
        bold: bool = True,
        border_radius: int = 0,
        border_color: tuple[int, int, int] | None = BORDER_DEFAULT,
    ) -> None:
        self.rect = pygame.Rect(rect)
        self.text = text
        self.on_click = on_click
        self.bg_color = bg_color
        self.hover_color = hover_color
        self.text_color = text_color
        self.font_size = font_size
        self.bold = bold
        self.border_radius = border_radius
        self.border_color = border_color
        self.is_hovered = False
        self.is_enabled = True
        self.is_visible = True

    def handle_event(self, event: pygame.event.Event) -> bool:
        if not self.is_visible or not self.is_enabled:
            self.is_hovered = False
            return False

        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
            return False

        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                if self.on_click:
                    self.on_click()
                return True

        return False

    def draw(self, surface: pygame.Surface) -> None:
        if not self.is_visible:
            return

        fill_color = self.hover_color if (self.is_hovered and self.is_enabled) else self.bg_color
        if not self.is_enabled:
            fill_color = (205, 195, 180)

        # RetroUI Hard Pixel Drop Shadow (3px offset)
        if self.is_enabled:
            shadow_rect = self.rect.move(3, 3)
            pygame.draw.rect(surface, BORDER_DEFAULT, shadow_rect)

        # Button Body
        pygame.draw.rect(surface, fill_color, self.rect)
        # Chunky 3px Border in deep espresso #30210b
        border_col = self.border_color or BORDER_DEFAULT
        if not self.is_enabled:
            border_col = BORDER_SUBTLE
        pygame.draw.rect(surface, border_col, self.rect, width=3)

        font = FontManager.get_font(self.font_size, bold=self.bold)
        txt_col = self.text_color if self.is_enabled else TEXT_MUTED
        txt_surf = font.render(self.text, True, txt_col)
        txt_rect = txt_surf.get_rect(center=self.rect.center)
        surface.blit(txt_surf, txt_rect)


class UIDropdown:
    """A retro arcade dropdown select menu with support for many items and scrolling."""

    def __init__(
        self,
        rect: pygame.Rect | tuple[int, int, int, int],
        options: Sequence[tuple[str, Any] | str],
        selected_value: Any = None,
        label: str = "",
        on_change: Callable[[Any], None] | None = None,
        max_visible_items: int = 7,
        font_size: int = 13,
    ) -> None:
        self.rect = pygame.Rect(rect)
        self.label = label
        self.on_change = on_change
        self.max_visible_items = max_visible_items
        self.font_size = font_size
        self.is_open = False
        self.scroll_offset = 0
        self.hovered_index = -1
        self.is_hovered = False
        self.is_enabled = True

        self.options: list[tuple[str, Any]] = []
        for opt in options:
            if isinstance(opt, tuple):
                self.options.append(opt)
            else:
                self.options.append((str(opt), opt))

        if selected_value is not None:
            self.selected_value = selected_value
        elif self.options:
            self.selected_value = self.options[0][1]
        else:
            self.selected_value = None

    def set_options(self, options: Sequence[tuple[str, Any] | str], preserve_selection: bool = True) -> None:
        current_val = self.selected_value
        self.options = [opt if isinstance(opt, tuple) else (str(opt), opt) for opt in options]
        if preserve_selection and any(v == current_val for _, v in self.options):
            self.selected_value = current_val
        elif self.options:
            self.selected_value = self.options[0][1]
        else:
            self.selected_value = None
        self.scroll_offset = 0

    @property
    def selected_label(self) -> str:
        for lbl, val in self.options:
            if val == self.selected_value:
                return lbl
        return str(self.selected_value) if self.selected_value is not None else "Select..."

    def get_popup_rect(self) -> pygame.Rect:
        num_visible = min(len(self.options), self.max_visible_items)
        item_height = self.rect.height
        return pygame.Rect(
            self.rect.x,
            self.rect.bottom + 2,
            self.rect.width,
            num_visible * item_height + 4,
        )

    def handle_event(self, event: pygame.event.Event) -> bool:
        if not self.is_enabled:
            self.is_open = False
            self.is_hovered = False
            return False

        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
            if self.is_open:
                popup_rect = self.get_popup_rect()
                if popup_rect.collidepoint(event.pos):
                    rel_y = event.pos[1] - (popup_rect.y + 2)
                    item_idx = self.scroll_offset + int(rel_y // self.rect.height)
                    if 0 <= item_idx < len(self.options):
                        self.hovered_index = item_idx
                    else:
                        self.hovered_index = -1
                    return True
                else:
                    self.hovered_index = -1

        elif event.type == pygame.MOUSEWHEEL and self.is_open:
            max_scroll = max(0, len(self.options) - self.max_visible_items)
            self.scroll_offset = max(0, min(max_scroll, self.scroll_offset - event.y))
            return True

        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.is_open = not self.is_open
                if self.is_open:
                    for i, (_, v) in enumerate(self.options):
                        if v == self.selected_value:
                            max_scroll = max(0, len(self.options) - self.max_visible_items)
                            self.scroll_offset = max(0, min(max_scroll, i - 2))
                            break
                return True

            if self.is_open:
                popup_rect = self.get_popup_rect()
                if popup_rect.collidepoint(event.pos):
                    rel_y = event.pos[1] - (popup_rect.y + 2)
                    idx = self.scroll_offset + int(rel_y // self.rect.height)
                    if 0 <= idx < len(self.options):
                        new_val = self.options[idx][1]
                        changed = (new_val != self.selected_value)
                        self.selected_value = new_val
                        self.is_open = False
                        if changed and self.on_change:
                            self.on_change(new_val)
                    return True
                else:
                    self.is_open = False
                    return False

        elif event.type == pygame.KEYDOWN and self.is_open:
            if event.key == pygame.K_ESCAPE:
                self.is_open = False
                return True

        return False

    def draw(self, surface: pygame.Surface) -> None:
        """Draw the closed base dropdown slot with RetroUI chunky border and hard drop-shadow."""
        if not self.is_enabled:
            bg_col = (205, 195, 180)
            border_col = BORDER_SUBTLE
            txt_col = TEXT_MUTED
            arrow_char = "—"
            arrow_col = TEXT_MUTED
        else:
            bg_col = BG_INPUT_HOVER if (self.is_hovered or self.is_open) else BG_INPUT
            border_col = BORDER_DEFAULT
            txt_col = TEXT_PRIMARY
            arrow_char = "▲" if self.is_open else "▼"
            arrow_col = TEXT_PRIMARY

            # RetroUI hard pixel drop shadow (3px)
            shadow_rect = self.rect.move(3, 3)
            pygame.draw.rect(surface, BORDER_DEFAULT, shadow_rect)

        # Base slot with chunky 3px border
        pygame.draw.rect(surface, bg_col, self.rect)
        pygame.draw.rect(surface, border_col, self.rect, width=3)

        # Selected text
        font = FontManager.get_font(self.font_size, bold=True)
        txt = self.selected_label
        avail_w = self.rect.width - 32
        txt_surf = font.render(txt, True, txt_col)
        if txt_surf.get_width() > avail_w:
            while len(txt) > 3 and font.size(txt + "...")[0] > avail_w:
                txt = txt[:-1]
            txt_surf = font.render(txt + "...", True, txt_col)

        surface.blit(txt_surf, (self.rect.x + 8, self.rect.y + (self.rect.height - txt_surf.get_height()) // 2))

        # Arrow indicator
        arrow_font = FontManager.get_font(10, bold=True)
        arrow_surf = arrow_font.render(arrow_char, True, arrow_col)
        surface.blit(arrow_surf, (self.rect.right - 18, self.rect.y + (self.rect.height - arrow_surf.get_height()) // 2))

    def draw_overlay(self, surface: pygame.Surface) -> None:
        """Draw floating popup options menu with RetroUI pixel styling."""
        if not self.is_open:
            return

        popup_rect = self.get_popup_rect()
        # Hard pixel drop shadow (4px)
        shadow_rect = popup_rect.move(4, 4)
        pygame.draw.rect(surface, BORDER_DEFAULT, shadow_rect)
        # Popup white body with chunky 3px border
        pygame.draw.rect(surface, BG_CARD, popup_rect)
        pygame.draw.rect(surface, BORDER_DEFAULT, popup_rect, width=3)

        item_height = self.rect.height
        visible_count = min(len(self.options), self.max_visible_items)
        font = FontManager.get_font(self.font_size, bold=False)

        for i in range(visible_count):
            opt_idx = self.scroll_offset + i
            if opt_idx >= len(self.options):
                break

            opt_lbl, opt_val = self.options[opt_idx]
            item_rect = pygame.Rect(
                popup_rect.x + 3,
                popup_rect.y + 3 + i * item_height,
                popup_rect.width - 6,
                item_height,
            )

            is_selected = (opt_val == self.selected_value)
            is_item_hovered = (opt_idx == self.hovered_index)

            if is_selected:
                # Active selection: RetroUI cocoa brown #62471f
                pygame.draw.rect(surface, BORDER_FOCUS, item_rect)
            elif is_item_hovered:
                # Hover: RetroUI warm tan #ddceb4
                pygame.draw.rect(surface, BG_INPUT, item_rect)

            txt_color = (254, 252, 208) if is_selected else TEXT_PRIMARY
            txt_s = opt_lbl
            txt_rend = font.render(txt_s, True, txt_color)
            if txt_rend.get_width() > item_rect.width - 16:
                while len(txt_s) > 3 and font.size(txt_s + "...")[0] > item_rect.width - 16:
                    txt_s = txt_s[:-1]
                txt_rend = font.render(txt_s + "...", True, txt_color)

            surface.blit(txt_rend, (item_rect.x + 8, item_rect.y + (item_rect.height - txt_rend.get_height()) // 2))

        # RetroUI Scrollbar
        if len(self.options) > self.max_visible_items:
            bar_track_h = popup_rect.height - 8
            thumb_h = max(16, int(bar_track_h * (self.max_visible_items / len(self.options))))
            max_scroll = len(self.options) - self.max_visible_items
            scroll_pct = self.scroll_offset / max_scroll if max_scroll > 0 else 0
            thumb_y = popup_rect.y + 4 + int(scroll_pct * (bar_track_h - thumb_h))
            thumb_rect = pygame.Rect(popup_rect.right - 8, thumb_y, 5, thumb_h)
            pygame.draw.rect(surface, BORDER_DEFAULT, thumb_rect)


class UICard:
    """A RetroUI pixel-perfect card container with chunky 3px border and 3px hard drop-shadow."""

    @staticmethod
    def draw(
        surface: pygame.Surface,
        rect: pygame.Rect | tuple[int, int, int, int],
        bg_color: tuple[int, int, int] = BG_CARD,
        border_color: tuple[int, int, int] = BORDER_DEFAULT,
        border_radius: int = 0,
    ) -> None:
        r = pygame.Rect(rect)
        # RetroUI Hard Pixel Drop Shadow (3px offset)
        shadow_rect = r.move(3, 3)
        pygame.draw.rect(surface, border_color, shadow_rect)
        # Card Body
        pygame.draw.rect(surface, bg_color, r)
        # Chunky 3px Border in deep espresso #30210b
        pygame.draw.rect(surface, border_color, r, width=3)


class UIBanner:
    """A RetroUI pixel tag/badge with solid color, 2px border, and hard pixel drop-shadow."""

    @staticmethod
    def draw(
        surface: pygame.Surface,
        rect: pygame.Rect | tuple[int, int, int, int],
        text: str,
        bg_color: tuple[int, int, int],
        text_color: tuple[int, int, int] = (255, 255, 255),
        font_size: int = 11,
        bold: bool = True,
        border_radius: int = 0,
    ) -> pygame.Rect:
        r = pygame.Rect(rect)
        # Hard pixel drop shadow (2px offset)
        shadow_rect = r.move(2, 2)
        pygame.draw.rect(surface, BORDER_DEFAULT, shadow_rect)
        # Tag Body
        pygame.draw.rect(surface, bg_color, r)
        # 2px border in #30210b
        pygame.draw.rect(surface, BORDER_DEFAULT, r, width=2)

        font = FontManager.get_font(font_size, bold=bold)
        txt_surf = font.render(text, True, text_color)
        txt_rect = txt_surf.get_rect(center=r.center)
        surface.blit(txt_surf, txt_rect)
        return r
