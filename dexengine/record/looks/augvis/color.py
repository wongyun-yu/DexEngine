"""Colour math of the visual augment (standard library only): sRGB <-> linear, CIELAB, ΔE76, chroma, luminance and the
tint multiply colour. The catalog's scene rules (augvis.catalog) and the scene writer (augvis.scene) use it."""

from __future__ import annotations

import math


def srgb_to_lab(rgb) -> tuple[float, float, float]:
    """sRGB (0-1, gamma) -> CIELAB (D65)."""

    def lin(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (lin(float(c)) for c in rgb)
    X = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047
    Y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    Z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883

    def f(t):
        return t ** (1 / 3) if t > 216 / 24389 else (24389 / 27 * t + 16) / 116

    fx, fy, fz = f(X), f(Y), f(Z)
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)


def delta_e(a, b) -> float:
    la, lb = srgb_to_lab(a), srgb_to_lab(b)
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(la, lb, strict=True)))


def lin(c: float) -> float:
    c = float(c)
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def srgb(c: float) -> float:
    c = max(0.0, float(c))
    return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055


def tint_rgb(tint) -> tuple[float, float, float]:
    """(hue °, saturation) -> linear multiply colour (HSV, value 1: the brightest channel 1; only darkens)."""
    h, sat = float(tint[0]) % 360.0, float(tint[1])
    c = sat
    x = c * (1 - abs((h / 60.0) % 2 - 1))
    r, g, b = [(c, x, 0), (x, c, 0), (0, c, x), (0, x, c), (x, 0, c), (c, 0, x)][int(h // 60) % 6]
    m = 1.0 - c
    return (round(r + m, 4), round(g + m, 4), round(b + m, 4))


def chroma(rgb) -> float:
    """sRGB -> CIELAB chroma C*."""
    _L, a, b = srgb_to_lab(rgb)
    return math.hypot(a, b)


def luma(rgb) -> float:
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
