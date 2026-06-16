import struct
from typing import List, Tuple, Optional


def bmp_bytes_per_row(width, bits_per_pixel):
    row_bytes = (width * bits_per_pixel + 7) // 8
    padded_row_bytes = (row_bytes + 3) & ~3
    return padded_row_bytes


def bmp_row_padding(width, bits_per_pixel):
    row_bytes = (width * bits_per_pixel + 7) // 8
    padded_row_bytes = (row_bytes + 3) & ~3
    return padded_row_bytes - row_bytes


class BMPImage:
    def __init__(self):
        self.width = 0
        self.height = 0
        self.bits_per_pixel = 24
        self.pixels = []
        self.palette = None
        self.compression = 0

    def get_pixel(self, x, y):
        if self.palette is not None:
            idx = self.pixels[y][x]
            return self.palette[idx]
        return self.pixels[y][x]

    def set_pixel(self, x, y, color):
        self.pixels[y][x] = color


def read_bmp(data: bytes) -> BMPImage:
    if len(data) < 54:
        raise ValueError("BMP data too short")

    if data[0:2] != b'BM':
        raise ValueError("Not a BMP file")

    file_size = struct.unpack_from('<I', data, 2)[0]
    pixel_offset = struct.unpack_from('<I', data, 10)[0]
    dib_size = struct.unpack_from('<I', data, 14)[0]

    if dib_size != 40:
        raise ValueError(f"Unsupported DIB header size: {dib_size}")

    width = struct.unpack_from('<i', data, 18)[0]
    height = struct.unpack_from('<i', data, 22)[0]
    planes = struct.unpack_from('<H', data, 26)[0]
    bits_per_pixel = struct.unpack_from('<H', data, 28)[0]
    compression = struct.unpack_from('<I', data, 30)[0]

    if compression != 0:
        raise ValueError(f"Compressed BMP not supported")

    colors_used = struct.unpack_from('<I', data, 46)[0]

    top_down = False
    if height < 0:
        top_down = True
        height = -height

    img = BMPImage()
    img.width = width
    img.height = height
    img.bits_per_pixel = bits_per_pixel
    img.compression = compression

    palette = None
    if bits_per_pixel <= 8:
        palette_size = colors_used if colors_used > 0 else (1 << bits_per_pixel)
        palette = []
        palette_start = 14 + dib_size
        for i in range(palette_size):
            offset = palette_start + i * 4
            b = data[offset]
            g = data[offset + 1]
            r = data[offset + 2]
            palette.append((r, g, b))
        img.palette = palette

    row_bytes = bmp_bytes_per_row(width, bits_per_pixel)
    padding = bmp_row_padding(width, bits_per_pixel)

    pixels = []
    for row in range(height):
        if top_down:
            file_row = row
        else:
            file_row = height - 1 - row

        row_offset = pixel_offset + file_row * row_bytes
        row_data = data[row_offset:row_offset + row_bytes]

        pixel_row = _decode_row(row_data, width, bits_per_pixel, palette)
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
    elif bits_per_pixel == 1:
        pixels = []
        for x in range(width):
            byte_idx = x // 8
            bit_idx = 7 - (x % 8)
            val = (row_data[byte_idx] >> bit_idx) & 1
            pixels.append(val)
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
    else:
        raise ValueError(f"Unsupported bits per pixel: {bits_per_pixel}")


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
    elif bits_per_pixel == 8:
        return bytes(pixels[:width])
    elif bits_per_pixel == 4:
        row = bytearray()
        for x in range(0, width, 2):
            high = (pixels[x] & 0x0F) << 4
            if x + 1 < width:
                low = pixels[x + 1] & 0x0F
            else:
                low = 0
            row.append(high | low)
        return bytes(row)
    elif bits_per_pixel == 1:
        row = bytearray()
        byte_val = 0
        bit_pos = 7
        for x in range(width):
            byte_val |= (pixels[x] & 1) << bit_pos
            bit_pos -= 1
            if bit_pos < 0:
                row.append(byte_val)
                byte_val = 0
                bit_pos = 7
        if bit_pos != 7:
            row.append(byte_val)
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
    else:
        raise ValueError(f"Unsupported bits per pixel: {bits_per_pixel}")


def write_bmp(img: BMPImage) -> bytes:
    width = img.width
    height = img.height
    bits_per_pixel = img.bits_per_pixel

    dib_size = 40
    file_header_size = 14

    palette_data = b''
    if bits_per_pixel <= 8:
        if img.palette:
            palette_size = len(img.palette)
        else:
            palette_size = 1 << bits_per_pixel
            img.palette = [(i, i, i) for i in range(palette_size)]

        palette_data = bytearray()
        for i in range(palette_size):
            r, g, b = img.palette[i][:3]
            palette_data.append(b)
            palette_data.append(g)
            palette_data.append(r)
            palette_data.append(0)
        palette_data = bytes(palette_data)

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
