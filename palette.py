from typing import List, Tuple, Optional


def color_distance(c1: Tuple[int, int, int], c2: Tuple[int, int, int]) -> int:
    dr = c1[0] - c2[0]
    dg = c1[1] - c2[1]
    db = c1[2] - c2[2]
    return dr * dr + dg * dg + db * db


def find_nearest_palette(color: Tuple[int, int, int], palette: List[Tuple[int, int, int]]) -> int:
    best_idx = 0
    best_dist = None
    for i, pal_color in enumerate(palette):
        dist = color_distance(color, pal_color)
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_idx = i
    return best_idx


def median_cut_quantize(colors: List[Tuple[int, int, int]], target_colors: int) -> List[Tuple[int, int, int]]:
    if not colors:
        return [(0, 0, 0)]

    unique_colors = list(set(colors))
    if len(unique_colors) <= target_colors:
        return unique_colors[:target_colors]

    def average_color(bucket):
        n = len(bucket)
        if n == 0:
            return (0, 0, 0)
        r = sum(c[0] for c in bucket) // n
        g = sum(c[1] for c in bucket) // n
        b = sum(c[2] for c in bucket) // n
        return (r, g, b)

    def split_bucket(bucket):
        if len(bucket) <= 1:
            return [bucket], []

        rs = [c[0] for c in bucket]
        gs = [c[1] for c in bucket]
        bs = [c[2] for c in bucket]

        r_range = max(rs) - min(rs)
        g_range = max(gs) - min(gs)
        b_range = max(bs) - min(bs)

        if r_range >= g_range and r_range >= b_range:
            sorted_bucket = sorted(bucket, key=lambda c: c[0])
        elif g_range >= b_range:
            sorted_bucket = sorted(bucket, key=lambda c: c[1])
        else:
            sorted_bucket = sorted(bucket, key=lambda c: c[2])

        mid = len(sorted_bucket) // 2
        return sorted_bucket[:mid], sorted_bucket[mid:]

    buckets = [unique_colors]
    while len(buckets) < target_colors:
        max_size = -1
        max_idx = -1
        for i, b in enumerate(buckets):
            if len(b) > max_size and len(b) > 1:
                max_size = len(b)
                max_idx = i

        if max_idx == -1:
            break

        left, right = split_bucket(buckets[max_idx])
        if not right:
            break

        buckets.pop(max_idx)
        buckets.append(left)
        buckets.append(right)

    palette = [average_color(b) for b in buckets]

    while len(palette) < target_colors:
        palette.append(palette[-1] if palette else (0, 0, 0))

    return palette[:target_colors]


class QuantizationResult:
    def __init__(self):
        self.palette = []
        self.indices = []
        self.is_lossy = False
        self.exact_match_count = 0
        self.total_pixels = 0
        self.max_error = 0
        self.original_colors = 0
        self.palette_colors = 0


class PaletteError(Exception):
    def __init__(self, message, color_count=None, max_colors=None):
        super().__init__(message)
        self.color_count = color_count
        self.max_colors = max_colors


def build_palette_and_quantize(
    pixels,
    bits_per_pixel: int,
    strategy: str = "auto",
) -> QuantizationResult:
    """
    Build palette from pixel data and quantize.

    bits_per_pixel: index bit depth (1, 2, 4, 8). Palette capacity = 1 << bits_per_pixel.
    strategy:
      - "error": raise PaletteError if colors exceed palette capacity
      - "exact": only accept if all colors fit, otherwise error
      - "quantize": use median-cut quantization to reduce colors
      - "auto": exact if fits, quantize otherwise
    """
    result = QuantizationResult()

    if bits_per_pixel <= 0:
        raise ValueError(f"bits_per_pixel must be positive, got {bits_per_pixel}")
    palette_capacity = 1 << bits_per_pixel if bits_per_pixel <= 8 else 256

    flat_colors = []
    width = len(pixels[0]) if pixels else 0
    height = len(pixels)
    result.total_pixels = width * height

    for row in pixels:
        for p in row:
            if isinstance(p, tuple):
                r, g, b = p[0], p[1], p[2]
            else:
                r = g = b = p
            flat_colors.append((r, g, b))

    unique_colors = list(set(flat_colors))
    result.original_colors = len(unique_colors)

    if strategy in ("error", "exact") and len(unique_colors) > palette_capacity:
        raise PaletteError(
            f"Image has {len(unique_colors)} unique colors, but {bits_per_pixel}-bit index mode "
            f"can only hold {palette_capacity} colors. "
            f"Use strategy='quantize' to auto-reduce, or switch to true-color mode.",
            color_count=len(unique_colors),
            max_colors=palette_capacity,
        )

    if len(unique_colors) <= palette_capacity:
        result.palette = unique_colors[:]
        result.is_lossy = False
        result.exact_match_count = result.total_pixels
        result.palette_colors = len(result.palette)
    else:
        result.palette = median_cut_quantize(unique_colors, palette_capacity)
        result.is_lossy = True
        result.palette_colors = len(result.palette)

    while len(result.palette) < palette_capacity:
        result.palette.append(result.palette[-1] if result.palette else (0, 0, 0))
    result.palette = result.palette[:palette_capacity]

    color_to_idx = {}
    for i, c in enumerate(result.palette):
        color_to_idx[c] = i

    result.indices = []
    for row in pixels:
        idx_row = []
        for p in row:
            if isinstance(p, tuple):
                color = (p[0], p[1], p[2])
            else:
                color = (p, p, p)

            if color in color_to_idx:
                idx = color_to_idx[color]
                result.exact_match_count += 1
            else:
                idx = find_nearest_palette(color, result.palette)
                matched = result.palette[idx]
                err = color_distance(color, matched)
                if err > result.max_error:
                    result.max_error = err
            idx_row.append(idx)
        result.indices.append(idx_row)

    return result
