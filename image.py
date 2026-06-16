from typing import List, Tuple, Optional
from bmp_codec import BMPImage, read_bmp, write_bmp, BMPError, bmp_bytes_per_row
from png_codec import PNGImage, read_png, write_png, PNGError, COLOR_TYPE_NAMES
from palette import build_palette_and_quantize, PaletteError, QuantizationResult


class Image:
    def __init__(self, width=0, height=0):
        self.width = width
        self.height = height
        self.pixels: List[List[Tuple]] = []
        self.palette: Optional[List[Tuple]] = None
        self._last_quant_result: Optional[QuantizationResult] = None
        self._meta = {}
        if width > 0 and height > 0:
            self.pixels = [[(0, 0, 0, 255) for _ in range(width)] for _ in range(height)]

    @property
    def last_quantization_info(self):
        return self._last_quant_result

    @property
    def unique_colors(self):
        colors = set()
        for row in self.pixels:
            for p in row:
                if isinstance(p, tuple):
                    colors.add(p[:3])
                else:
                    colors.add((p, p, p))
        return len(colors)

    def set_meta(self, key, value):
        self._meta[key] = value

    def get_meta(self, key, default=None):
        return self._meta.get(key, default)

    @staticmethod
    def from_bmp(data: bytes) -> 'Image':
        bmp = read_bmp(data)
        img = Image(bmp.width, bmp.height)
        img.palette = None
        img.set_meta("source_format", "bmp")
        img.set_meta("source_bpp", bmp.bits_per_pixel)
        img.set_meta("source_compression", bmp.compression)

        if bmp.palette is not None:
            img.pixels = [[bmp.get_pixel(x, y) for x in range(bmp.width)] for y in range(bmp.height)]
        else:
            img.pixels = []
            for y in range(bmp.height):
                row = []
                for x in range(bmp.width):
                    p = bmp.pixels[y][x]
                    if len(p) == 3:
                        row.append((p[0], p[1], p[2], 255))
                    else:
                        row.append(p)
                img.pixels.append(row)

        return img

    @staticmethod
    def from_png(data: bytes, strict_crc=True) -> 'Image':
        png = read_png(data, strict_crc=strict_crc)
        img = Image(png.width, png.height)
        img.palette = None
        img.set_meta("source_format", "png")
        img.set_meta("source_color_type", png.color_type)
        img.set_meta("source_bit_depth", png.bit_depth)
        img.set_meta("idat_chunks", png.idat_chunk_count)
        img.set_meta("compressed_size", png.compressed_idat_size)
        img.set_meta("raw_size", png.raw_idat_size)

        if png.color_type == 3 and png.palette is not None:
            img.pixels = [[png.palette[png.pixels[y][x]] for x in range(png.width)] for y in range(png.height)]
            for row in img.pixels:
                for i in range(len(row)):
                    p = row[i]
                    row[i] = (p[0], p[1], p[2], 255)
        elif png.color_type == 0:
            img.pixels = []
            for y in range(png.height):
                row = []
                for x in range(png.width):
                    v = png.pixels[y][x]
                    row.append((v, v, v, 255))
                img.pixels.append(row)
        else:
            img.pixels = []
            for y in range(png.height):
                row = []
                for x in range(png.width):
                    p = png.pixels[y][x]
                    if len(p) == 3:
                        row.append((p[0], p[1], p[2], 255))
                    else:
                        row.append(p)
                img.pixels.append(row)

        return img

    @staticmethod
    def from_file(data: bytes, fmt: str = None, **kwargs) -> 'Image':
        if fmt is None:
            if data[:2] == b'BM':
                fmt = 'bmp'
            elif data[:8] == b'\x89PNG\r\n\x1a\n':
                fmt = 'png'
            else:
                raise ValueError("Cannot determine image format from signature")

        fmt = fmt.lower()
        if fmt == 'bmp':
            return Image.from_bmp(data)
        elif fmt == 'png':
            return Image.from_png(data, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {fmt}")

    def to_bmp(self, bits_per_pixel=24, palette_strategy="quantize") -> bytes:
        bmp = BMPImage()
        bmp.width = self.width
        bmp.height = self.height
        bmp.bits_per_pixel = bits_per_pixel

        if bits_per_pixel <= 8:
            stripped = []
            for row in self.pixels:
                new_row = []
                for p in row:
                    if isinstance(p, tuple):
                        new_row.append((p[0], p[1], p[2]))
                    else:
                        new_row.append((p, p, p))
                stripped.append(new_row)

            quant = build_palette_and_quantize(stripped, bits_per_pixel, strategy=palette_strategy)
            self._last_quant_result = quant
            bmp.palette = quant.palette
            bmp.pixels = quant.indices
        else:
            bmp.pixels = []
            for row in self.pixels:
                new_row = []
                for p in row:
                    if bits_per_pixel == 32:
                        if len(p) == 4:
                            new_row.append(p)
                        else:
                            new_row.append((p[0], p[1], p[2], 255))
                    else:
                        new_row.append(p[:3])
                bmp.pixels.append(new_row)

        return write_bmp(bmp, palette_strategy=palette_strategy)

    def to_png(self, color_type=6, bit_depth=8, filter_type=4,
               palette_strategy="quantize", idat_split=None) -> bytes:
        png = PNGImage()
        png.width = self.width
        png.height = self.height
        png.bit_depth = bit_depth
        png.color_type = color_type

        if color_type == 3:
            stripped = []
            for row in self.pixels:
                new_row = []
                for p in row:
                    if isinstance(p, tuple):
                        new_row.append((p[0], p[1], p[2]))
                    else:
                        new_row.append((p, p, p))
                stripped.append(new_row)
            quant = build_palette_and_quantize(stripped, bit_depth, strategy=palette_strategy)
            self._last_quant_result = quant
            png.palette = quant.palette
            png.pixels = quant.indices
        elif color_type == 2:
            png.pixels = [[(p[0], p[1], p[2]) if isinstance(p, tuple) else (p, p, p) for p in row] for row in self.pixels]
        elif color_type == 6:
            png.pixels = []
            for row in self.pixels:
                new_row = []
                for p in row:
                    if isinstance(p, tuple):
                        if len(p) == 4:
                            new_row.append(p)
                        else:
                            new_row.append((p[0], p[1], p[2], 255))
                    else:
                        new_row.append((p, p, p, 255))
                png.pixels.append(new_row)
        elif color_type == 0:
            def _to_gray(p):
                if isinstance(p, tuple):
                    return int(0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2])
                return p
            png.pixels = [[_to_gray(p) for p in row] for row in self.pixels]

        return write_png(png, filter_type=filter_type,
                         palette_strategy=palette_strategy,
                         idat_split=idat_split)

    def to_file(self, fmt: str, **kwargs) -> bytes:
        fmt = fmt.lower()
        if fmt == 'bmp':
            return self.to_bmp(**kwargs)
        elif fmt == 'png':
            return self.to_png(**kwargs)
        else:
            raise ValueError(f"Unsupported format: {fmt}")

    def describe(self):
        lines = []
        lines.append(f"  Dimensions : {self.width} x {self.height}")
        uc = self.unique_colors
        lines.append(f"  Unique RGB : {uc}")
        src_fmt = self.get_meta("source_format", "memory")
        lines.append(f"  Source     : {src_fmt.upper()}")
        if src_fmt == "png":
            ct = self.get_meta("source_color_type", "?")
            lines.append(f"  Orig color : {ct} ({COLOR_TYPE_NAMES.get(ct, 'Unknown')})")
            lines.append(f"  Orig depth : {self.get_meta('source_bit_depth', '?')}")
            idat_c = self.get_meta("idat_chunks", 0)
            if idat_c:
                comp = self.get_meta("compressed_size", 0)
                raw = self.get_meta("raw_size", 0)
                lines.append(f"  IDAT chunks: {idat_c}  ({comp:,} -> {raw:,} bytes)")
        elif src_fmt == "bmp":
            lines.append(f"  Orig bpp   : {self.get_meta('source_bpp', '?')}")
        return "\n".join(lines)

    def get_pixel(self, x: int, y: int) -> Tuple:
        return self.pixels[y][x]

    def set_pixel(self, x: int, y: int, color: Tuple):
        self.pixels[y][x] = color

    def pixels_equal(self, other: 'Image', tolerance=0, ignore_alpha=False) -> bool:
        if self.width != other.width or self.height != other.height:
            return False
        for y in range(self.height):
            for x in range(self.width):
                a = self.pixels[y][x]
                b = other.pixels[y][x]
                if ignore_alpha:
                    a_cmp = a[:3]
                    b_cmp = b[:3]
                else:
                    if len(a) != len(b):
                        if a[:3] != b[:3]:
                            if tolerance == 0:
                                return False
                        continue
                    a_cmp = a
                    b_cmp = b
                if tolerance == 0:
                    if a_cmp != b_cmp:
                        return False
                else:
                    for ca, cb in zip(a_cmp, b_cmp):
                        if abs(ca - cb) > tolerance:
                            return False
        return True

    def count_differences(self, other: 'Image', ignore_alpha=False):
        if self.width != other.width or self.height != other.height:
            return None, float('inf')
        count = 0
        max_diff = 0
        for y in range(self.height):
            for x in range(self.width):
                a = self.pixels[y][x]
                b = other.pixels[y][x]
                n = 3 if ignore_alpha else min(len(a), len(b))
                for i in range(n):
                    d = abs(a[i] - b[i])
                    if d > 0:
                        count += 1
                        if d > max_diff:
                            max_diff = d
        total = self.width * self.height * (4 if not ignore_alpha else 3)
        return count, max_diff
