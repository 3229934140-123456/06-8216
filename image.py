from typing import List, Tuple, Optional
from bmp_codec import BMPImage, read_bmp, write_bmp
from png_codec import PNGImage, read_png, write_png


class Image:
    def __init__(self, width=0, height=0):
        self.width = width
        self.height = height
        self.pixels: List[List[Tuple]] = []
        self.palette: Optional[List[Tuple]] = None
        if width > 0 and height > 0:
            self.pixels = [[(0, 0, 0) for _ in range(width)] for _ in range(height)]

    @staticmethod
    def from_bmp(data: bytes) -> 'Image':
        bmp = read_bmp(data)
        img = Image(bmp.width, bmp.height)
        img.palette = bmp.palette
        if bmp.palette is not None:
            img.pixels = [[bmp.get_pixel(x, y) for x in range(bmp.width)] for y in range(bmp.height)]
        else:
            img.pixels = bmp.pixels
        return img

    @staticmethod
    def from_png(data: bytes) -> 'Image':
        png = read_png(data)
        img = Image(png.width, png.height)
        img.palette = png.palette
        if png.color_type == 3 and png.palette is not None:
            img.pixels = [[png.palette[png.pixels[y][x]] for x in range(png.width)] for y in range(png.height)]
        elif png.color_type == 0:
            img.pixels = [[(v, v, v, 255) if isinstance(v, int) else v for v in row] for row in png.pixels]
        else:
            img.pixels = png.pixels
        return img

    @staticmethod
    def from_file(data: bytes, fmt: str = None) -> 'Image':
        if fmt is None:
            if data[:2] == b'BM':
                fmt = 'bmp'
            elif data[:8] == b'\x89PNG\r\n\x1a\n':
                fmt = 'png'
            else:
                raise ValueError("Cannot determine image format")

        fmt = fmt.lower()
        if fmt == 'bmp':
            return Image.from_bmp(data)
        elif fmt == 'png':
            return Image.from_png(data)
        else:
            raise ValueError(f"Unsupported format: {fmt}")

    def to_bmp(self, bits_per_pixel=24) -> bytes:
        bmp = BMPImage()
        bmp.width = self.width
        bmp.height = self.height
        bmp.bits_per_pixel = bits_per_pixel

        if bits_per_pixel <= 8:
            if self.palette:
                bmp.palette = self.palette
                palette_map = {color: i for i, color in enumerate(self.palette)}
                bmp.pixels = [[palette_map.get(p, 0) for p in row] for row in self.pixels]
            else:
                palette = []
                seen = {}
                idx = 0
                indexed_pixels = []
                for row in self.pixels:
                    idx_row = []
                    for p in row:
                        color = p[:3] if isinstance(p, tuple) else (p, p, p)
                        if color not in seen:
                            if idx >= (1 << bits_per_pixel):
                                color = (idx % 256, idx % 256, idx % 256)
                            seen[color] = idx
                            palette.append(color)
                            idx += 1
                        idx_row.append(seen[color])
                    indexed_pixels.append(idx_row)
                bmp.palette = palette
                bmp.pixels = indexed_pixels
        else:
            bmp.pixels = self.pixels

        return write_bmp(bmp)

    def to_png(self, color_type=6, bit_depth=8, filter_type=4) -> bytes:
        png = PNGImage()
        png.width = self.width
        png.height = self.height
        png.bit_depth = bit_depth
        png.color_type = color_type

        if color_type == 3:
            if self.palette:
                png.palette = self.palette
                palette_map = {color: i for i, color in enumerate(self.palette)}
                png.pixels = [[palette_map.get(p[:3] if isinstance(p, tuple) else (p, p, p), 0) for p in row] for row in self.pixels]
            else:
                palette = []
                seen = {}
                idx = 0
                indexed_pixels = []
                for row in self.pixels:
                    idx_row = []
                    for p in row:
                        color = p[:3] if isinstance(p, tuple) else (p, p, p)
                        if color not in seen:
                            if idx >= 256:
                                color = (idx % 256, idx % 256, idx % 256)
                            seen[color] = idx
                            palette.append(color)
                            idx += 1
                        idx_row.append(seen[color])
                    indexed_pixels.append(idx_row)
                png.palette = palette
                png.pixels = indexed_pixels
        elif color_type == 2:
            png.pixels = [[(p[0], p[1], p[2]) if isinstance(p, tuple) else (p, p, p) for p in row] for row in self.pixels]
        elif color_type == 6:
            png.pixels = [[p if len(p) == 4 else (p[0], p[1], p[2], 255) if isinstance(p, tuple) else (p, p, p, 255) for p in row] for row in self.pixels]
        elif color_type == 0:
            def _to_gray(p):
                if isinstance(p, tuple):
                    return int(0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2])
                return p
            png.pixels = [[_to_gray(p) for p in row] for row in self.pixels]

        return write_png(png, filter_type=filter_type)

    def to_file(self, fmt: str, **kwargs) -> bytes:
        fmt = fmt.lower()
        if fmt == 'bmp':
            return self.to_bmp(**kwargs)
        elif fmt == 'png':
            return self.to_png(**kwargs)
        else:
            raise ValueError(f"Unsupported format: {fmt}")

    def get_pixel(self, x: int, y: int) -> Tuple:
        return self.pixels[y][x]

    def set_pixel(self, x: int, y: int, color: Tuple):
        self.pixels[y][x] = color

    def pixels_equal(self, other: 'Image') -> bool:
        if self.width != other.width or self.height != other.height:
            return False
        for y in range(self.height):
            for x in range(self.width):
                a = self.pixels[y][x]
                b = other.pixels[y][x]
                if len(a) != len(b):
                    if a[:3] != b[:3]:
                        return False
                elif a != b:
                    return False
        return True
