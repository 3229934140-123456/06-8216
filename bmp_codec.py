import struct
from typing import List, Tuple, Optional
from palette import build_palette_and_quantize, PaletteError


def bmp_bytes_per_row(width, bits_per_pixel):
    row_bytes = (width * bits_per_pixel + 7) // 8
    padded_row_bytes = (row_bytes + 3) & ~3
    return padded_row_bytes


def bmp_row_padding(width, bits_per_pixel):
    row_bytes = (width * bits_per_pixel + 7) // 8
    padded_row_bytes = (row_bytes + 3) & ~3
    return padded_row_bytes - row_bytes


class BMPError(Exception):
    def __init__(self, code, message, details=None):
        super().__init__(message)
        self.code = code
        self.details = details

    def user_message(self):
        lines = [f"[BMP Error] {self.code}: {self}"]
        if self.details:
            lines.append(f"  Details: {self.details}")
        lines.append("")
        lines.append("  Troubleshooting:")
        if self.code == "SIGNATURE":
            lines.append("  - File is not a BMP, or has corrupt header")
            lines.append("  - Expected first 2 bytes to be 'BM'")
        elif self.code == "TRUNCATED":
            lines.append("  - File data was truncated")
            lines.append("  - Check if file is complete and uncorrupted")
        elif self.code == "HEADER":
            lines.append("  - Invalid BMP or DIB header fields")
        elif self.code == "FEATURE":
            lines.append("  - Uses features not supported by this decoder")
            lines.append("  - Only uncompressed BMP (BI_RGB / compression=0) is supported")
        elif self.code == "PALETTE":
            lines.append("  - Problem with indexed-color palette")
            lines.append("  - Verify palette size matches bit depth")
        return "\n".join(lines)


class BMPImage:
    def __init__(self):
        self.width = 0
        self.height = 0
        self.bits_per_pixel = 24
        self.pixels = []
        self.palette = None
        self.compression = 0
        self.file_size = 0
        self.pixel_offset = 0
        self.top_down = False

    def get_pixel(self, x, y):
        if self.palette is not None:
            idx = self.pixels[y][x]
            if idx < len(self.palette):
                return self.palette[idx]
            return (0, 0, 0)
        return self.pixels[y][x]

    def set_pixel(self, x, y, color):
        self.pixels[y][x] = color

    def describe(self):
        lines = []
        lines.append(f"  Dimensions : {self.width} x {self.height}")
        lines.append(f"  Bit depth  : {self.bits_per_pixel}")
        lines.append(f"  Row order  : {'Top-down' if self.top_down else 'Bottom-up (standard)'}")
        if self.palette:
            lines.append(f"  Palette    : {len(self.palette)} entries")
        row_bytes = bmp_bytes_per_row(self.width, self.bits_per_pixel)
        pad = bmp_row_padding(self.width, self.bits_per_pixel)
        lines.append(f"  Row bytes  : {row_bytes} (including {pad} padding bytes)")
        lines.append(f"  Compression: {self.compression} (BI_RGB=uncompressed)")
        if self.file_size:
            lines.append(f"  File size  : {self.file_size:,} bytes")
        return "\n".join(lines)


def _validate_bmp(data):
    if len(data) < 2:
        raise BMPError("TRUNCATED", f"File too small for BMP signature ({len(data)} bytes < 2)")
    if data[0:2] != b'BM':
        got = data[0:2].hex()
        raise BMPError("SIGNATURE", f"Not a BMP file: first 2 bytes are 0x{got} instead of 'BM' (0x424D)")
    if len(data) < 54:
        raise BMPError("TRUNCATED",
                       f"File too small for minimum BMP header: {len(data)} bytes < 54",
                       details={"need_at_least": 54})


def read_bmp(data: bytes) -> BMPImage:
    _validate_bmp(data)

    file_size = struct.unpack_from('<I', data, 2)[0]
    pixel_offset = struct.unpack_from('<I', data, 10)[0]
    dib_size = struct.unpack_from('<I', data, 14)[0]

    if dib_size != 40:
        raise BMPError("HEADER",
                       f"Unsupported DIB header size: {dib_size}. Only BITMAPINFOHEADER (size=40) is supported.",
                       details={"dib_size": dib_size, "supported_sizes": [40]})

    if pixel_offset < 14 + dib_size:
        raise BMPError("HEADER",
                       f"Pixel data offset {pixel_offset} is inside header area",
                       details={"pixel_offset": pixel_offset, "header_end": 14 + dib_size})

    width = struct.unpack_from('<i', data, 18)[0]
    height = struct.unpack_from('<i', data, 22)[0]
    planes = struct.unpack_from('<H', data, 26)[0]
    bits_per_pixel = struct.unpack_from('<H', data, 28)[0]
    compression = struct.unpack_from('<I', data, 30)[0]

    if planes != 1:
        raise BMPError("HEADER",
                       f"Invalid planes field: {planes}. Must be 1 for standard BMP.",
                       details={"planes": planes})

    if compression != 0:
        raise BMPError("FEATURE",
                       f"Compression method {compression} is not supported. Only uncompressed BMP (BI_RGB=0) works.",
                       details={"compression": compression, "supported": [0]})

    if bits_per_pixel not in (1, 2, 4, 8, 16, 24, 32):
        raise BMPError("HEADER",
                       f"Unsupported bit depth: {bits_per_pixel}.",
                       details={"bits_per_pixel": bits_per_pixel,
                                "supported": [1, 2, 4, 8, 16, 24, 32]})

    colors_used = struct.unpack_from('<I', data, 46)[0]

    top_down = False
    if height < 0:
        top_down = True
        height = -height

    if width <= 0 or height <= 0:
        raise BMPError("HEADER",
                       f"Invalid dimensions: {width}x{height}",
                       details={"width": width, "height": height})

    img = BMPImage()
    img.width = width
    img.height = height
    img.bits_per_pixel = bits_per_pixel
    img.compression = compression
    img.file_size = file_size
    img.pixel_offset = pixel_offset
    img.top_down = top_down

    palette = None
    if bits_per_pixel <= 8:
        max_colors = 1 << bits_per_pixel
        palette_size = colors_used if colors_used > 0 else max_colors
        palette_size = min(palette_size, max_colors)

        palette_start = 14 + dib_size
        needed_end = palette_start + palette_size * 4
        if len(data) < needed_end:
            raise BMPError("TRUNCATED",
                           f"Palette data truncated: need {needed_end} bytes, have {len(data)}",
                           details={"palette_start": palette_start, "palette_size": palette_size})

        palette = []
        for i in range(palette_size):
            offset = palette_start + i * 4
            b = data[offset]
            g = data[offset + 1]
            r = data[offset + 2]
            palette.append((r, g, b))
        img.palette = palette

    row_bytes = bmp_bytes_per_row(width, bits_per_pixel)
    padding = bmp_row_padding(width, bits_per_pixel)

    total_pixel_bytes = row_bytes * height
    if pixel_offset + total_pixel_bytes > len(data):
        raise BMPError("TRUNCATED",
                       f"Pixel data truncated: need {pixel_offset + total_pixel_bytes} bytes, have {len(data)}",
                       details={"expected_end": pixel_offset + total_pixel_bytes,
                                "file_size": len(data),
                                "width": width, "height": height,
                                "bytes_per_row": row_bytes})

    pixels = []
    for row in range(height):
        if top_down:
            file_row = row
        else:
            file_row = height - 1 - row

        row_offset = pixel_offset + file_row * row_bytes
        row_end = row_offset + row_bytes
        if row_end > len(data):
            raise BMPError("TRUNCATED",
                           f"Row {row} (file row {file_row}) extends past end of file",
                           details={"row": row, "file_row": file_row,
                                    "row_start": row_offset, "row_end": row_end, "file_size": len(data)})
        row_data = data[row_offset:row_end]

        pixel_row = _decode_row(row_data, width, bits_per_pixel, palette)

        if bits_per_pixel <= 8 and palette:
            for idx_val in pixel_row:
                if idx_val >= len(palette):
                    raise BMPError("PALETTE",
                                   f"Row {row} contains palette index {idx_val} but palette only has {len(palette)} entries",
                                   details={"row": row, "index": idx_val, "palette_size": len(palette)})

        pixels.append(pixel_row)

    img.pixels = pixels
    return img


def _decode_row(row_data: bytes, width: int, bits_per_pixel: int, palette) -> List:
    if bits_per_pixel == 24:
        pixels = []
        for x in range(width):
            offset = x * 3
            b = row_data[offset]
            g = row_data[offset + 1]
            r = row_data[offset + 2]
            pixels.append((r, g, b))
        return pixels
    elif bits_per_pixel == 32:
        pixels = []
        for x in range(width):
            offset = x * 4
            b = row_data[offset]
            g = row_data[offset + 1]
            r = row_data[offset + 2]
            a = row_data[offset + 3]
            pixels.append((r, g, b, a))
        return pixels
    elif bits_per_pixel == 16:
        pixels = []
        for x in range(width):
            offset = x * 2
            val = struct.unpack_from('<H', row_data, offset)[0]
            r = ((val >> 10) & 0x1F) << 3
            g = ((val >> 5) & 0x1F) << 3
            b = (val & 0x1F) << 3
            pixels.append((r, g, b))
        return pixels
    elif bits_per_pixel == 8:
        return list(row_data[:width])
    elif bits_per_pixel == 4:
        pixels = []
        for x in range(width):
            byte_idx = x // 2
            if x % 2 == 0:
                val = (row_data[byte_idx] >> 4) & 0x0F
            else:
                val = row_data[byte_idx] & 0x0F
            pixels.append(val)
        return pixels
    elif bits_per_pixel == 2:
        pixels = []
        for x in range(width):
            byte_idx = x // 4
            shift = 6 - 2 * (x % 4)
            val = (row_data[byte_idx] >> shift) & 0x03
            pixels.append(val)
        return pixels
    elif bits_per_pixel == 1:
        pixels = []
        for x in range(width):
            byte_idx = x // 8
            bit_idx = 7 - (x % 8)
            val = (row_data[byte_idx] >> bit_idx) & 1
            pixels.append(val)
        return pixels
    else:
        raise BMPError("HEADER", f"Cannot decode row with {bits_per_pixel} bpp")


def _encode_row(pixels, width, bits_per_pixel, palette) -> bytes:
    if bits_per_pixel == 24:
        row = bytearray()
        for x in range(width):
            r, g, b = pixels[x][:3]
            row.append(b)
            row.append(g)
            row.append(r)
        return bytes(row)
    elif bits_per_pixel == 32:
        row = bytearray()
        for x in range(width):
            pixel = pixels[x]
            if len(pixel) == 4:
                r, g, b, a = pixel
            else:
                r, g, b = pixel
                a = 255
            row.append(b)
            row.append(g)
            row.append(r)
            row.append(a)
        return bytes(row)
    elif bits_per_pixel == 16:
        row = bytearray()
        for x in range(width):
            r, g, b = pixels[x][:3]
            r5 = (r >> 3) & 0x1F
            g5 = (g >> 3) & 0x1F
            b5 = (b >> 3) & 0x1F
            val = (r5 << 10) | (g5 << 5) | b5
            row.extend(struct.pack('<H', val))
        return bytes(row)
    elif bits_per_pixel == 8:
        max_idx = 255
        for x in range(width):
            if pixels[x] > max_idx:
                raise PaletteError(f"Index {pixels[x]} exceeds 8-bit max 255 at col {x}")
        return bytes(pixels[:width])
    elif bits_per_pixel == 4:
        row = bytearray()
        max_idx = 15
        for x in range(0, width, 2):
            if pixels[x] > max_idx:
                raise PaletteError(f"Index {pixels[x]} exceeds 4-bit max 15 at col {x}")
            high = (pixels[x] & 0x0F) << 4
            if x + 1 < width:
                if pixels[x + 1] > max_idx:
                    raise PaletteError(f"Index {pixels[x + 1]} exceeds 4-bit max 15 at col {x + 1}")
                low = pixels[x + 1] & 0x0F
            else:
                low = 0
            row.append(high | low)
        return bytes(row)
    elif bits_per_pixel == 2:
        row = bytearray()
        max_idx = 3
        for x in range(0, width, 4):
            byte_val = 0
            for s in range(4):
                col = x + s
                if col < width:
                    if pixels[col] > max_idx:
                        raise PaletteError(f"Index {pixels[col]} exceeds 2-bit max 3 at col {col}")
                    v = pixels[col] & 0x03
                else:
                    v = 0
                byte_val |= v << (6 - 2 * s)
            row.append(byte_val)
        return bytes(row)
    elif bits_per_pixel == 1:
        row = bytearray()
        byte_val = 0
        bit_pos = 7
        for x in range(width):
            v = pixels[x] & 1
            byte_val |= v << bit_pos
            bit_pos -= 1
            if bit_pos < 0:
                row.append(byte_val)
                byte_val = 0
                bit_pos = 7
        if bit_pos != 7:
            row.append(byte_val)
        return bytes(row)
    else:
        raise BMPError("HEADER", f"Cannot encode row with {bits_per_pixel} bpp")


def write_bmp(img: BMPImage, palette_strategy="quantize") -> bytes:
    width = img.width
    height = img.height
    bits_per_pixel = img.bits_per_pixel

    dib_size = 40
    file_header_size = 14

    palette_data = b''
    if bits_per_pixel <= 8:
        if not img.palette:
            quant = build_palette_and_quantize(img.pixels, bits_per_pixel, strategy=palette_strategy)
            img.palette = quant.palette
            img.pixels = quant.indices

        palette_size = len(img.palette)
        max_palette = 1 << bits_per_pixel
        if palette_size > max_palette:
            raise PaletteError(
                f"Palette has {palette_size} colors, but {bits_per_pixel}-bit BMP can only hold {max_palette}. "
                f"Reduce colors, or use strategy='quantize' to auto-reduce.",
                color_count=palette_size, max_colors=max_palette
            )

        palette_data = bytearray()
        for i in range(palette_size):
            r, g, b = img.palette[i][:3]
            palette_data.append(b)
            palette_data.append(g)
            palette_data.append(r)
            palette_data.append(0)
        palette_data = bytes(palette_data)
    else:
        img.palette = None

    row_bytes = bmp_bytes_per_row(width, bits_per_pixel)
    padding = bmp_row_padding(width, bits_per_pixel)
    pixel_data_size = row_bytes * height

    pixel_offset = file_header_size + dib_size + len(palette_data)
    file_size = pixel_offset + pixel_data_size

    result = bytearray()

    result.extend(b'BM')
    result.extend(struct.pack('<I', file_size))
    result.extend(struct.pack('<HH', 0, 0))
    result.extend(struct.pack('<I', pixel_offset))

    result.extend(struct.pack('<I', dib_size))
    result.extend(struct.pack('<i', width))
    result.extend(struct.pack('<i', height))
    result.extend(struct.pack('<H', 1))
    result.extend(struct.pack('<H', bits_per_pixel))
    result.extend(struct.pack('<I', 0))
    result.extend(struct.pack('<I', pixel_data_size))
    result.extend(struct.pack('<i', 2835))
    result.extend(struct.pack('<i', 2835))
    result.extend(struct.pack('<I', len(img.palette) if img.palette else 0))
    result.extend(struct.pack('<I', 0))

    result.extend(palette_data)

    for row in range(height):
        source_row = height - 1 - row
        row_data = _encode_row(img.pixels[source_row], width, bits_per_pixel, img.palette)
        result.extend(row_data)
        if padding > 0:
            result.extend(b'\x00' * padding)

    return bytes(result)
