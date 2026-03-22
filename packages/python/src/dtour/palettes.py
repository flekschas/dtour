"""Color palettes matching the scatter engine's built-in categorical palettes.

Use :func:`build_color_map` to construct a ``color_map`` dict suitable for
:class:`~dtour.Widget`.
"""

from __future__ import annotations

OKABE_ITO: list[tuple[int, int, int]] = [
    (0, 114, 178),
    (230, 159, 0),
    (0, 158, 115),
    (204, 121, 167),
    (86, 180, 233),
    (213, 94, 0),
    (240, 228, 66),
]

GLASBEY_DARK: list[tuple[int, int, int]] = [
    (215, 0, 0),
    (0, 151, 0),
    (0, 0, 255),
    (255, 186, 0),
    (0, 208, 255),
    (255, 0, 186),
    (143, 90, 0),
    (0, 255, 151),
    (169, 0, 255),
    (129, 129, 129),
    (255, 129, 97),
    (0, 107, 79),
    (109, 109, 255),
    (255, 0, 57),
    (137, 171, 0),
    (176, 106, 255),
    (0, 168, 168),
    (255, 131, 222),
    (104, 70, 43),
    (175, 185, 143),
    (163, 0, 96),
    (93, 136, 200),
    (230, 215, 127),
    (0, 64, 36),
    (62, 53, 91),
    (199, 186, 255),
    (127, 0, 18),
    (85, 233, 188),
    (70, 80, 0),
    (255, 195, 182),
    (106, 50, 89),
    (0, 107, 140),
    (238, 114, 0),
    (168, 161, 195),
    (148, 85, 115),
    (0, 172, 95),
    (199, 0, 171),
    (135, 141, 113),
    (27, 36, 65),
    (185, 111, 84),
    (113, 109, 60),
    (0, 232, 87),
    (162, 55, 0),
    (0, 157, 218),
    (231, 164, 255),
    (96, 131, 120),
    (96, 76, 255),
    (192, 155, 103),
    (207, 0, 116),
    (36, 65, 103),
    (157, 205, 255),
    (182, 97, 176),
    (89, 186, 138),
    (255, 78, 147),
    (81, 113, 0),
    (255, 168, 105),
    (75, 0, 48),
    (117, 217, 0),
    (0, 90, 179),
    (214, 194, 0),
    (0, 74, 56),
    (237, 237, 220),
    (90, 12, 0),
    (147, 161, 236),
]

GLASBEY_LIGHT: list[tuple[int, int, int]] = [
    (214, 0, 0),
    (1, 135, 0),
    (181, 0, 255),
    (5, 172, 198),
    (151, 255, 0),
    (255, 165, 47),
    (255, 142, 200),
    (121, 82, 94),
    (0, 253, 207),
    (175, 165, 255),
    (147, 172, 131),
    (154, 105, 0),
    (54, 105, 98),
    (211, 0, 140),
    (253, 244, 144),
    (200, 110, 102),
    (158, 226, 255),
    (0, 200, 70),
    (168, 119, 172),
    (184, 186, 1),
    (244, 191, 177),
    (255, 40, 253),
    (242, 205, 255),
    (0, 158, 124),
    (255, 98, 0),
    (86, 100, 42),
    (149, 63, 31),
    (144, 49, 142),
    (255, 52, 100),
    (160, 228, 145),
    (140, 154, 177),
    (130, 144, 38),
    (174, 8, 63),
    (119, 198, 186),
    (188, 145, 87),
    (228, 142, 255),
    (114, 184, 255),
    (198, 165, 193),
    (255, 144, 112),
    (211, 195, 124),
    (188, 237, 219),
    (107, 133, 103),
    (145, 110, 86),
    (249, 255, 0),
    (186, 193, 223),
    (172, 86, 124),
    (255, 205, 3),
    (255, 73, 177),
    (193, 86, 3),
    (93, 140, 144),
    (193, 68, 188),
    (0, 117, 63),
    (186, 110, 253),
    (0, 212, 147),
    (0, 255, 117),
    (73, 161, 80),
    (204, 151, 144),
    (0, 235, 237),
    (219, 126, 1),
    (247, 117, 137),
    (184, 149, 0),
    (200, 66, 72),
    (0, 207, 249),
    (117, 87, 38),
]


def _rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{r:02x}{g:02x}{b:02x}"


def _build_palette(n: int, theme: str) -> list[tuple[int, int, int]]:
    """Return a palette of *n* colors matching the scatter engine logic."""
    if n <= len(OKABE_ITO):
        return OKABE_ITO[:n]
    glasbey = GLASBEY_LIGHT if theme == "light" else GLASBEY_DARK
    extra_needed = n - len(OKABE_ITO)
    reps = -(-extra_needed // len(glasbey))  # ceil div
    return list(OKABE_ITO) + (glasbey * reps)[:extra_needed]


def build_color_map(
    labels: list[str],
    *,
    theme: str | None = None,
    overrides: dict[str, str | dict[str, str]] | None = None,
) -> dict[str, str | dict[str, str]]:
    """Build a ``color_map`` for :class:`~dtour.Widget`.

    Assigns colours from the built-in OKABE_ITO + GLASBEY palettes (matching
    the scatter engine's auto-assignment) and applies *overrides* on top.

    Parameters
    ----------
    labels:
        Sorted unique category labels (same order the engine sees).
    theme:
        ``"light"`` or ``"dark"`` to produce a single-theme map (all values
        are plain hex strings).  When *None* (default), returns theme-aware
        ``{"light": "#...", "dark": "#..."}`` dicts where the two palettes
        differ.
    overrides:
        Per-label colour overrides.  Values may be a hex string (same for both
        themes) or ``{"light": "#...", "dark": "#..."}``.
    """
    if theme is not None:
        pal = _build_palette(len(labels), theme)
        cmap: dict[str, str | dict[str, str]] = {
            label: _rgb_to_hex(*pal[i]) for i, label in enumerate(labels)
        }
    else:
        dark_pal = _build_palette(len(labels), "dark")
        light_pal = _build_palette(len(labels), "light")
        cmap = {}
        for i, label in enumerate(labels):
            dark_hex = _rgb_to_hex(*dark_pal[i])
            light_hex = _rgb_to_hex(*light_pal[i])
            if dark_hex == light_hex:
                cmap[label] = dark_hex
            else:
                cmap[label] = {"light": light_hex, "dark": dark_hex}

    if overrides:
        cmap.update(overrides)

    return cmap
