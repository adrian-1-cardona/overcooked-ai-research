"""Reusable UI widgets for the Overcooked interactive application.

Includes buttons, dropdown select menus with scrolling, banners, badges, and cards,
implemented with clean RetroUI styling (2px black borders, hard offset drop-shadows,
retro gray surfaces, and crisp typography).
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
    BORDER_BLACK,
    SHADOW_BLACK,
    BORDER_DEFAULT,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TEXT_MUTED,
)


class UIButton:
    """A RetroUI styled push button with hard offset shadows and tactile button-press."""

    def __init__(
        self,
        rect: pygame.Rect | tuple[int, int, int, int],
        text: str,
        on_click: Callable[[], None] | None = None,
        bg_color: tuple[int, int, int] = (210, 214, 220),
        hover_color: tuple[int, int, int] = (195, 200, 208),
        text_color: tuple[int, int, int] = TEXT_PRIMARY,
        font_size: int = 14,
        bold: bool = True,
        border_radius: int = 0,
        border_color: tuple[int, int, int] | None = BORDER_BLACK,
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
        self.is_pressed = False
        self.is_enabled = True
        self.is_visible = True

    def handle_event(self, event: pygame.event.Event) -> bool:
        if not self.is_visible or not self.is_enabled:
            self.is_hovered = False
            self.is_pressed = False
            return False

        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
            return False

        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.is_pressed = True
                if self.on_click:
                    self.on_click()
                return True

        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.is_pressed = False
            return False

        return False

    def draw(self, surface: pygame.Surface) -> None:
        if not self.is_visible:
            return

        if not self.is_enabled:
            fill_color = (220, 222, 226)
            txt_col = TEXT_MUTED
            pygame.draw.rect(surface, fill_color, self.rect, border_radius=self.border_radius)
            pygame.draw.rect(surface, BORDER_BLACK, self.rect, width=2, border_radius=self.border_radius)
            font = FontManager.get_font(self.font_size, bold=self.bold)
            txt_surf = font.render(self.text, True, txt_col)
            surface.blit(txt_surf, txt_surf.get_rect(center=self.rect.center))
            return

        # Hard drop shadow & tactile press
        if self.is_pressed:
            draw_rect = self.rect.move(2, 2)
            shadow_rect = self.rect.move(1, 1)
        elif self.is_hovered:
            draw_rect = self.rect.move(1, 1)
            shadow_rect = self.rect.move(3, 3)
        else:
            draw_rect = self.rect
            shadow_rect = self.rect.move(3, 3)

        pygame.draw.rect(surface, SHADOW_BLACK, shadow_rect, border_radius=self.border_radius)
        fill_color = self.hover_color if self.is_hovered else self.bg_color
        pygame.draw.rect(surface, fill_color, draw_rect, border_radius=self.border_radius)
        pygame.draw.rect(surface, BORDER_BLACK, draw_rect, width=2, border_radius=self.border_radius)

        font = FontManager.get_font(self.font_size, bold=self.bold)
        txt_surf = font.render(self.text, True, self.text_color)
        txt_rect = txt_surf.get_rect(center=draw_rect.center)
        surface.blit(txt_surf, txt_rect)


class UIDropdown:
    """A RetroUI styled dropdown select menu with crisp borders and scrolling popup."""

    def __init__(
        self,
        rect: pygame.Rect | tuple[int, int, int, int],
        options: Sequence[tuple[str, Any]],
        selected_value: Any = None,
        on_change: Callable[[Any], None] | None = None,
        font_size: int = 13,
        max_visible_items: int = 6,
    ) -> None:
        self.rect = pygame.Rect(rect)
        self.options = list(options)
        self.selected_value = selected_value
        self.on_change = on_change
        self.font_size = font_size
        self.max_visible_items = max_visible_items

        self.is_open = False
        self.is_hovered = False
        self.hovered_index = -1
        self.scroll_offset = 0
        self.is_enabled = True

        if self.selected_value is None and self.options:
            self.selected_value = self.options[0][1]

    @property
    def selected_label(self) -> str:
        for lbl, val in self.options:
            if val == self.selected_value:
                return lbl
        return str(self.selected_value)

    def set_options(self, options: Sequence[tuple[str, Any]], new_selected: Any = None) -> None:
        self.options = list(options)
        if new_selected is not None:
            self.selected_value = new_selected
        elif self.options:
            match = next((v for l, v in self.options if v == self.selected_value), None)
            if match is None:
                self.selected_value = self.options[0][1]
        self.scroll_offset = 0

    def get_popup_rect(self) -> pygame.Rect:
        item_count = min(len(self.options), self.max_visible_items)
        total_h = item_count * self.rect.height
        return pygame.Rect(self.rect.x, self.rect.bottom + 2, self.rect.width, total_h)

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
                    self.hovered_index = self.scroll_offset + int(rel_y // self.rect.height)
                else:
                    self.hovered_index = -1
            return False

        elif event.type == pygame.MOUSEWHEEL and self.is_open:
            max_scroll = max(0, len(self.options) - self.max_visible_items)
            self.scroll_offset = max(0, min(max_scroll, self.scroll_offset - event.y))
            return True

        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.is_open = not self.is_open
                if self.is_open:
                    for i, (_, val) in enumerate(self.options):
                        if val == self.selected_value:
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
                    return True

        elif event.type == pygame.KEYDOWN and self.is_open:
            if event.key == pygame.K_ESCAPE:
                self.is_open = False
                return True

        return False

    def draw(self, surface: pygame.Surface) -> None:
        """Draw the closed/base dropdown control in RetroUI style."""
        if not self.is_enabled:
            bg_col = (226, 228, 232)
            txt_col = TEXT_MUTED
            arrow_char = "-"
            pygame.draw.rect(surface, bg_col, self.rect)
            pygame.draw.rect(surface, BORDER_BLACK, self.rect, width=2)
        else:
            bg_col = BG_INPUT_HOVER if (self.is_hovered or self.is_open) else BG_INPUT
            txt_col = TEXT_PRIMARY
            arrow_char = "^" if self.is_open else "v"
            # Hard offset shadow
            pygame.draw.rect(surface, SHADOW_BLACK, self.rect.move(2, 2))
            pygame.draw.rect(surface, bg_col, self.rect)
            pygame.draw.rect(surface, BORDER_BLACK, self.rect, width=2)

        font = FontManager.get_font(self.font_size, bold=False)
        txt = self.selected_label
        avail_w = self.rect.width - 32
        txt_surf = font.render(txt, True, txt_col)
        if txt_surf.get_width() > avail_w:
            while len(txt) > 3 and font.size(txt + "...")[0] > avail_w:
                txt = txt[:-1]
            txt_surf = font.render(txt + "...", True, txt_col)

        surface.blit(txt_surf, (self.rect.x + 10, self.rect.y + (self.rect.height - txt_surf.get_height()) // 2))

        arrow_font = FontManager.get_font(11, bold=True)
        arrow_surf = arrow_font.render(arrow_char, True, txt_col)
        surface.blit(arrow_surf, (self.rect.right - 20, self.rect.y + (self.rect.height - arrow_surf.get_height()) // 2))

    def draw_overlay(self, surface: pygame.Surface) -> None:
        """Draw the floating popup menu with RetroUI hard shadow and borders."""
        if not self.is_open:
            return

        popup_rect = self.get_popup_rect()
        # Retro hard drop shadow
        pygame.draw.rect(surface, SHADOW_BLACK, popup_rect.move(3, 3))

        # Popup background & 2px border
        pygame.draw.rect(surface, (255, 255, 255), popup_rect)
        pygame.draw.rect(surface, BORDER_BLACK, popup_rect, width=2)

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
                pygame.draw.rect(surface, (210, 214, 222), item_rect)
                pygame.draw.rect(surface, BORDER_BLACK, item_rect, width=1)
            elif is_item_hovered:
                pygame.draw.rect(surface, (235, 238, 244), item_rect)

            txt_color = TEXT_PRIMARY
            txt_s = opt_lbl
            txt_rend = font.render(txt_s, True, txt_color)
            if txt_rend.get_width() > item_rect.width - 16:
                while len(txt_s) > 3 and font.size(txt_s + "...")[0] > item_rect.width - 16:
                    txt_s = txt_s[:-1]
                txt_rend = font.render(txt_s + "...", True, txt_color)

            surface.blit(txt_rend, (item_rect.x + 8, item_rect.y + (item_rect.height - txt_rend.get_height()) // 2))

        # Scrollbar if more items exist
        if len(self.options) > self.max_visible_items:
            bar_track_h = popup_rect.height - 8
            thumb_h = max(16, int(bar_track_h * (self.max_visible_items / len(self.options))))
            max_scroll = len(self.options) - self.max_visible_items
            scroll_pct = self.scroll_offset / max_scroll if max_scroll > 0 else 0
            thumb_y = popup_rect.y + 4 + int(scroll_pct * (bar_track_h - thumb_h))
            thumb_rect = pygame.Rect(popup_rect.right - 8, thumb_y, 6, thumb_h)
            pygame.draw.rect(surface, (120, 125, 135), thumb_rect)
            pygame.draw.rect(surface, BORDER_BLACK, thumb_rect, width=1)


class UICard:
    """A RetroUI styled card container with hard drop shadows and 2px borders."""

    @staticmethod
    def draw(
        surface: pygame.Surface,
        rect: pygame.Rect | tuple[int, int, int, int],
        bg_color: tuple[int, int, int] = BG_CARD,
        border_color: tuple[int, int, int] = BORDER_BLACK,
        border_radius: int = 0,
        shadow_offset: int = 3,
    ) -> None:
        r = pygame.Rect(rect)
        if shadow_offset > 0:
            pygame.draw.rect(surface, SHADOW_BLACK, r.move(shadow_offset, shadow_offset), border_radius=border_radius)
        pygame.draw.rect(surface, bg_color, r, border_radius=border_radius)
        pygame.draw.rect(surface, border_color, r, width=2, border_radius=border_radius)


class UIBadge:
    """A small colorful status badge/pill with retro border."""

    @staticmethod
    def draw(
        surface: pygame.Surface,
        center_x: int,
        center_y: int,
        text: str,
        bg_color: tuple[int, int, int],
        text_color: tuple[int, int, int] = TEXT_PRIMARY,
        font_size: int = 12,
        bold: bool = True,
        padding_x: int = 10,
        padding_y: int = 4,
    ) -> pygame.Rect:
        font = FontManager.get_font(font_size, bold=bold)
        txt_surf = font.render(text, True, text_color)
        w = txt_surf.get_width() + padding_x * 2
        h = txt_surf.get_height() + padding_y * 2
        rect = pygame.Rect(0, 0, w, h)
        rect.center = (center_x, center_y)
        pygame.draw.rect(surface, SHADOW_BLACK, rect.move(1, 1))
        pygame.draw.rect(surface, bg_color, rect)
        pygame.draw.rect(surface, BORDER_BLACK, rect, width=2)
        surface.blit(txt_surf, txt_surf.get_rect(center=rect.center))
        return rect


class UIBanner:
    """A clean rectangular banner/tag styled with retro borders matching RetroUI."""

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
        pygame.draw.rect(surface, SHADOW_BLACK, r.move(1, 1), border_radius=border_radius)
        pygame.draw.rect(surface, bg_color, r, border_radius=border_radius)
        pygame.draw.rect(surface, BORDER_BLACK, r, width=2, border_radius=border_radius)
        font = FontManager.get_font(font_size, bold=bold)
        txt_surf = font.render(text, True, text_color)
        txt_rect = txt_surf.get_rect(center=r.center)
        surface.blit(txt_surf, txt_rect)
        return r


class UISwitchbar:
    """A segmented tab switchbar in RetroUI style ([ Live Match | Matrix Test ])."""

    def __init__(
        self,
        rect: pygame.Rect | tuple[int, int, int, int],
        options: Sequence[tuple[str, str]],
        selected_value: str,
        on_change: Callable[[str], None] | None = None,
        bg_color: tuple[int, int, int] = (245, 245, 247),
        active_color: tuple[int, int, int] = (190, 195, 204),
        border_color: tuple[int, int, int] = BORDER_BLACK,
    ) -> None:
        self.rect = pygame.Rect(rect)
        self.options = list(options)
        self.selected_value = selected_value
        self.on_change = on_change
        self.bg_color = bg_color
        self.active_color = active_color
        self.border_color = border_color

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                num_opts = max(1, len(self.options))
                seg_w = self.rect.width / num_opts
                rel_x = event.pos[0] - self.rect.x
                idx = min(num_opts - 1, int(rel_x // seg_w))
                new_val = self.options[idx][1]
                if new_val != self.selected_value:
                    self.selected_value = new_val
                    if self.on_change:
                        self.on_change(new_val)
                    return True
        return False

    def draw(self, surface: pygame.Surface) -> None:
        # Hard drop shadow
        pygame.draw.rect(surface, SHADOW_BLACK, self.rect.move(2, 2))
        pygame.draw.rect(surface, self.bg_color, self.rect)
        pygame.draw.rect(surface, self.border_color, self.rect, width=2)

        num_opts = max(1, len(self.options))
        seg_w = self.rect.width / num_opts

        for idx, (label, val) in enumerate(self.options):
            seg_rect = pygame.Rect(
                int(self.rect.x + idx * seg_w),
                self.rect.y,
                int(seg_w),
                self.rect.height,
            )
            is_active = (val == self.selected_value)
            if is_active:
                pygame.draw.rect(surface, self.active_color, seg_rect)
                pygame.draw.rect(surface, BORDER_BLACK, seg_rect, width=1)

            font = FontManager.get_font(12, bold=is_active)
            txt_color = TEXT_PRIMARY if is_active else TEXT_MUTED
            txt_surf = font.render(label, True, txt_color)
            txt_rect = txt_surf.get_rect(center=seg_rect.center)
            surface.blit(txt_surf, txt_rect)


class UICheckbox:
    """A RetroUI styled toggle checkbox with hard shadow and pixel box."""

    def __init__(
        self,
        rect: pygame.Rect | tuple[int, int, int, int],
        text: str,
        checked: bool = True,
        on_change: Callable[[bool], None] | None = None,
    ) -> None:
        self.rect = pygame.Rect(rect)
        self.text = text
        self.checked = checked
        self.on_change = on_change
        self.is_hovered = False

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.checked = not self.checked
                if self.on_change:
                    self.on_change(self.checked)
                return True
        return False

    def draw(self, surface: pygame.Surface) -> None:
        box_size = 18
        box_rect = pygame.Rect(self.rect.x, self.rect.y + (self.rect.height - box_size) // 2, box_size, box_size)

        # Retro hard drop shadow
        pygame.draw.rect(surface, SHADOW_BLACK, box_rect.move(2, 2))
        pygame.draw.rect(surface, (255, 255, 255), box_rect)
        pygame.draw.rect(surface, BORDER_BLACK, box_rect, width=2)

        if self.checked:
            # Retro solid black square pip
            pip_rect = pygame.Rect(box_rect.x + 4, box_rect.y + 4, 10, 10)
            pygame.draw.rect(surface, (0, 0, 0), pip_rect)

        font = FontManager.get_font(12, bold=self.checked)
        txt_color = TEXT_PRIMARY if self.checked else TEXT_SECONDARY
        txt_surf = font.render(self.text, True, txt_color)
        surface.blit(txt_surf, (box_rect.right + 10, self.rect.y + (self.rect.height - txt_surf.get_height()) // 2))


