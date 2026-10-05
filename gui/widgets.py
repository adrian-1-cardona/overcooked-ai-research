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
    """A retro arcade button with 3D chiseled bevel borders and hover states."""

    def __init__(
        self,
        rect: pygame.Rect | tuple[int, int, int, int],
        text: str,
        on_click: Callable[[], None] | None = None,
        bg_color: tuple[int, int, int] = BG_INPUT,
        hover_color: tuple[int, int, int] = BG_INPUT_HOVER,
        text_color: tuple[int, int, int] = TEXT_PRIMARY,
        font_size: int = 15,
        bold: bool = True,
        border_radius: int = 4,
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
            fill_color = tuple(max(20, c // 2) for c in self.bg_color)

        pygame.draw.rect(surface, fill_color, self.rect, border_radius=self.border_radius)

        # 3D Arcade Bevel Effect
        if self.is_enabled:
            light = _adjust_color(fill_color, 45)
            dark = _adjust_color(fill_color, -50)
            # Top edge and Left edge (Highlight)
            pygame.draw.line(surface, light, (self.rect.left, self.rect.top), (self.rect.right - 1, self.rect.top), 2)
            pygame.draw.line(surface, light, (self.rect.left, self.rect.top), (self.rect.left, self.rect.bottom - 1), 2)
            # Bottom edge and Right edge (Shadow)
            pygame.draw.line(surface, dark, (self.rect.left, self.rect.bottom - 1), (self.rect.right - 1, self.rect.bottom - 1), 2)
            pygame.draw.line(surface, dark, (self.rect.right - 1, self.rect.top), (self.rect.right - 1, self.rect.bottom - 1), 2)

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
        """Draw the closed base dropdown slot with retro bevels."""
        if not self.is_enabled:
            bg_col = (20, 24, 34)
            border_col = BORDER_SUBTLE
            txt_col = TEXT_MUTED
            arrow_char = "—"
            arrow_col = (65, 72, 92)
        else:
            bg_col = BG_INPUT_HOVER if (self.is_hovered or self.is_open) else BG_INPUT
            border_col = BORDER_FOCUS if (self.is_hovered or self.is_open) else BORDER_DEFAULT
            txt_col = TEXT_PRIMARY
            arrow_char = "▲" if self.is_open else "▼"
            arrow_col = TEXT_MUTED if not self.is_open else TEXT_PRIMARY

        pygame.draw.rect(surface, bg_col, self.rect, border_radius=4)
        pygame.draw.rect(surface, border_col, self.rect, width=1, border_radius=4)

        # Selected text
        font = FontManager.get_font(self.font_size, bold=False)
        txt = self.selected_label
        avail_w = self.rect.width - 32
        txt_surf = font.render(txt, True, txt_col)
        if txt_surf.get_width() > avail_w:
            while len(txt) > 3 and font.size(txt + "...")[0] > avail_w:
                txt = txt[:-1]
            txt_surf = font.render(txt + "...", True, txt_col)

        surface.blit(txt_surf, (self.rect.x + 8, self.rect.y + (self.rect.height - txt_surf.get_height()) // 2))

        # Arrow indicator
        arrow_font = FontManager.get_font(10, bold=False)
        arrow_surf = arrow_font.render(arrow_char, True, arrow_col)
        surface.blit(arrow_surf, (self.rect.right - 18, self.rect.y + (self.rect.height - arrow_surf.get_height()) // 2))

    def draw_overlay(self, surface: pygame.Surface) -> None:
        """Draw floating popup options menu with retro arcade styling."""
        if not self.is_open:
            return

        popup_rect = self.get_popup_rect()
        shadow_rect = popup_rect.move(3, 3)
        pygame.draw.rect(surface, (8, 10, 15), shadow_rect, border_radius=4)
        pygame.draw.rect(surface, BG_CARD, popup_rect, border_radius=4)
        pygame.draw.rect(surface, BORDER_FOCUS, popup_rect, width=2, border_radius=4)

        item_height = self.rect.height
        visible_count = min(len(self.options), self.max_visible_items)
        font = FontManager.get_font(self.font_size, bold=False)

        for i in range(visible_count):
            opt_idx = self.scroll_offset + i
            if opt_idx >= len(self.options):
                break

            opt_lbl, opt_val = self.options[opt_idx]
            item_rect = pygame.Rect(
                popup_rect.x + 2,
                popup_rect.y + 2 + i * item_height,
                popup_rect.width - 4,
                item_height,
            )

            is_selected = (opt_val == self.selected_value)
            is_item_hovered = (opt_idx == self.hovered_index)

            if is_selected:
                pygame.draw.rect(surface, BG_INPUT_ACTIVE, item_rect, border_radius=2)
            elif is_item_hovered:
                pygame.draw.rect(surface, BG_INPUT_HOVER, item_rect, border_radius=2)

            txt_color = TEXT_PRIMARY if (is_selected or is_item_hovered) else TEXT_SECONDARY
            txt_s = opt_lbl
            txt_rend = font.render(txt_s, True, txt_color)
            if txt_rend.get_width() > item_rect.width - 16:
                while len(txt_s) > 3 and font.size(txt_s + "...")[0] > item_rect.width - 16:
                    txt_s = txt_s[:-1]
                txt_rend = font.render(txt_s + "...", True, txt_color)

            surface.blit(txt_rend, (item_rect.x + 8, item_rect.y + (item_rect.height - txt_rend.get_height()) // 2))

        # Retro Scrollbar
        if len(self.options) > self.max_visible_items:
            bar_track_h = popup_rect.height - 8
            thumb_h = max(16, int(bar_track_h * (self.max_visible_items / len(self.options))))
            max_scroll = len(self.options) - self.max_visible_items
            scroll_pct = self.scroll_offset / max_scroll if max_scroll > 0 else 0
            thumb_y = popup_rect.y + 4 + int(scroll_pct * (bar_track_h - thumb_h))
            thumb_rect = pygame.Rect(popup_rect.right - 6, thumb_y, 4, thumb_h)
            pygame.draw.rect(surface, (120, 135, 175), thumb_rect, border_radius=2)


class UICard:
    """A retro arcade card container with 3D chiseled bevel borders."""

    @staticmethod
    def draw(
        surface: pygame.Surface,
        rect: pygame.Rect | tuple[int, int, int, int],
        bg_color: tuple[int, int, int] = BG_CARD,
        border_color: tuple[int, int, int] = BORDER_DEFAULT,
        border_radius: int = 4,
    ) -> None:
        r = pygame.Rect(rect)
        pygame.draw.rect(surface, bg_color, r, border_radius=border_radius)
        # 3D Retro bevel edges
        pygame.draw.line(surface, BORDER_RETRO_LIGHT, (r.left, r.top), (r.right - 1, r.top), 1)
        pygame.draw.line(surface, BORDER_RETRO_LIGHT, (r.left, r.top), (r.left, r.bottom - 1), 1)
        pygame.draw.line(surface, BORDER_RETRO_DARK, (r.left, r.bottom - 1), (r.right - 1, r.bottom - 1), 1)
        pygame.draw.line(surface, BORDER_RETRO_DARK, (r.right - 1, r.top), (r.right - 1, r.bottom - 1), 1)


class UIBanner:
    """A clean rectangular retro banner/tag styled with solid color and 3D bevels."""

    @staticmethod
    def draw(
        surface: pygame.Surface,
        rect: pygame.Rect | tuple[int, int, int, int],
        text: str,
        bg_color: tuple[int, int, int],
        text_color: tuple[int, int, int] = (255, 255, 255),
        font_size: int = 11,
        bold: bool = True,
        border_radius: int = 4,
    ) -> pygame.Rect:
        r = pygame.Rect(rect)
        pygame.draw.rect(surface, bg_color, r, border_radius=border_radius)

        # Subtle bevel
        light = _adjust_color(bg_color, 45)
        dark = _adjust_color(bg_color, -45)
        pygame.draw.line(surface, light, (r.left, r.top), (r.right - 1, r.top), 1)
        pygame.draw.line(surface, light, (r.left, r.top), (r.left, r.bottom - 1), 1)
        pygame.draw.line(surface, dark, (r.left, r.bottom - 1), (r.right - 1, r.bottom - 1), 1)
        pygame.draw.line(surface, dark, (r.right - 1, r.top), (r.right - 1, r.bottom - 1), 1)

        font = FontManager.get_font(font_size, bold=bold)
        txt_surf = font.render(text, True, text_color)
        txt_rect = txt_surf.get_rect(center=r.center)
        surface.blit(txt_surf, txt_rect)
        return r
