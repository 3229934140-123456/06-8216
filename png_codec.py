import struct
import zlib
from deflate import zlib_compress, zlib_decompress


PNG_SIGNATURE = b'\x89PNG\r\n\x1a\n'


def crc32(data):
    return zlib.crc32(data) & 0xFFFFFFFF


def paeth_predictor(a, b, c):
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    elif pb <= pc:
        return b
    else:
        return c


class PNGChunk:
    def __init__(self, chunk_type, data):
        self.chunk_type = chunk_type
        self.data = data

    def encode(self):
        length = struct.pack('>I', len(self.data))
        chunk_data = self.chunk_type + self.data
        crc = struct.pack('>I', crc32(chunk_data))
        return length + chunk_data + crc


def parse_chunks(data):
    pos = len(PNG_SIGNATURE)
    chunks = []
    while pos < len(data):
        if pos + 8 > len(data):
            break
        length = struct.unpack_from('>I', data, pos)[0]
        chunk_type = data[pos + 4:pos + 8]
        chunk_data = data[pos + 8:pos + 8 + length]
        stored_crc = struct.unpack_from('>I', data, pos + 8 + length)[0]

        calc_crc = crc32(chunk_type + chunk_data)
        if calc_crc != stored_crc:
            raise ValueError(f"CRC mismatch for chunk {chunk_type}")

        chunks.append(PNGChunk(chunk_type, chunk_data))
        pos += 12 + length

        if chunk_type == b'IEND':
            break
    return chunks


class PNGImage:
    def __init__(self):
        self.width = 0
        self.height = 0
        self.bit_depth = 8
        self.color_type = 6
        self.compression = 0
        self.filter = 0
        self.interlace = 0
        self.palette = None
        self.pixels = []

    @property
    def channels(self):
        if self.color_type == 0:
            return 1
        elif self.color_type == 2:
            return 3
        elif self.color_type == 3:
            return 1
        elif self.color_type == 4:
            return 2
        elif self.color_type == 6:
            return 4
        return 1

    @property
    def bytes_per_pixel(self):
        return max(1, (self.bit_depth * self.channels + 7) // 8)


def filter_none(scanline, prev_scanline, bpp):
    return scanline[:]


def filter_sub(scanline, prev_scanline, bpp):
    result = bytearray()
    for i in range(len(scanline)):
        left = scanline[i - bpp] if i >= bpp else 0
        result.append((scanline[i] - left) & 0xFF)
    return bytes(result)


def filter_up(scanline, prev_scanline, bpp):
    result = bytearray()
    for i in range(len(scanline)):
        up = prev_scanline[i] if prev_scanline else 0
        result.append((scanline[i] - up) & 0xFF)
    return bytes(result)


def filter_average(scanline, prev_scanline, bpp):
    result = bytearray()
    for i in range(len(scanline)):
        left = scanline[i - bpp] if i >= bpp else 0
        up = prev_scanline[i] if prev_scanline else 0
        result.append((scanline[i] - (left + up) // 2) & 0xFF)
    return bytes(result)


def filter_paeth(scanline, prev_scanline, bpp):
    result = bytearray()
    for i in range(len(scanline)):
        left = scanline[i - bpp] if i >= bpp else 0
        up = prev_scanline[i] if prev_scanline else 0
        up_left = prev_scanline[i - bpp] if (prev_scanline and i >= bpp) else 0
        result.append((scanline[i] - paeth_predictor(left, up, up_left)) & 0xFF)
    return bytes(result)


def unfilter_none(filtered, prev_scanline, bpp):
    return filtered[:]


def unfilter_sub(filtered, prev_scanline, bpp):
    result = bytearray()
    for i in range(len(filtered)):
        left = result[i - bpp] if i >= bpp else 0
        result.append((filtered[i] + left) & 0xFF)
    return bytes(result)


def unfilter_up(filtered, prev_scanline, bpp):
    result = bytearray()
    for i in range(len(filtered)):
        up = prev_scanline[i] if prev_scanline else 0
        result.append((filtered[i] + up) & 0xFF)
    return bytes(result)


def unfilter_average(filtered, prev_scanline, bpp):
    result = bytearray()
    for i in range(len(filtered)):
        left = result[i - bpp] if i >= bpp else 0
        up = prev_scanline[i] if prev_scanline else 0
        result.append((filtered[i] + (left + up) // 2) & 0xFF)
    return bytes(result)


def unfilter_paeth(filtered, prev_scanline, bpp):
    result = bytearray()
    for i in range(len(filtered)):
        left = result[i - bpp] if i >= bpp else 0
        up = prev_scanline[i] if prev_scanline else 0
        up_left = prev_scanline[i - bpp] if (prev_scanline and i >= bpp) else 0
        result.append((filtered[i] + paeth_predictor(left, up, up_left)) & 0xFF)
    return bytes(result)


FILTER_FUNCTIONS = [
    filter_none,
    filter_sub,
    filter_up,
    filter_average,
    filter_paeth,
]

UNFILTER_FUNCTIONS = [
    unfilter_none,
    unfilter_sub,
    unfilter_up,
    unfilter_average,
    unfilter_paeth,
]


def _raw_row_bytes(width, bit_depth, channels):
    return (width * bit_depth * channels + 7) // 8


def read_png(data):
    if data[:8] != PNG_SIGNATURE:
        raise ValueError("Not a PNG file")

    chunks = parse_chunks(data)

    img = PNGImage()
    idat_data = bytearray()

    for chunk in chunks:
        if chunk.chunk_type == b'IHDR':
            if len(chunk.data) != 13:
                raise ValueError("Invalid IHDR size")
            img.width = struct.unpack_from('>I', chunk.data, 0)[0]
            img.height = struct.unpack_from('>I', chunk.data, 4)[0]
            img.bit_depth = chunk.data[8]
            img.color_type = chunk.data[9]
            img.compression = chunk.data[10]
            img.filter = chunk.data[11]
            img.interlace = chunk.data[12]

            if img.compression != 0:
                raise ValueError("Unsupported compression method")
            if img.filter != 0:
                raise ValueError("Unsupported filter method")
            if img.interlace != 0:
                raise ValueError("Interlacing not supported")

        elif chunk.chunk_type == b'PLTE':
            palette_size = len(chunk.data) // 3
            img.palette = []
            for i in range(palette_size):
                r = chunk.data[i * 3]
                g = chunk.data[i * 3 + 1]
                b = chunk.data[i * 3 + 2]
                img.palette.append((r, g, b))

        elif chunk.chunk_type == b'IDAT':
            idat_data.extend(chunk.data)

        elif chunk.chunk_type == b'IEND':
            break

    if img.color_type == 3 and not img.palette:
        raise ValueError("Indexed color requires PLTE chunk")

    decompressed = zlib_decompress(bytes(idat_data))

    bpp = img.bytes_per_pixel
    stride = _raw_row_bytes(img.width, img.bit_depth, img.channels)

    raw_rows = []
    prev_row = None
    pos = 0

    for row in range(img.height):
        if pos >= len(decompressed):
            raise ValueError("Truncated image data")

        filter_byte = decompressed[pos]
        pos += 1

        filtered_row = decompressed[pos:pos + stride]
        pos += stride

        if len(filtered_row) < stride:
            filtered_row = filtered_row + b'\x00' * (stride - len(filtered_row))

        unfilter_fn = UNFILTER_FUNCTIONS[filter_byte]
        raw_row = unfilter_fn(filtered_row, prev_row, bpp)
        raw_rows.append(raw_row)
        prev_row = raw_row

    img.pixels = _decode_pixels(raw_rows, img)
    return img


def _decode_pixels(raw_rows, img):
    width = img.width
    height = img.height
    bit_depth = img.bit_depth
    color_type = img.color_type
    channels = img.channels

    pixels = []

    for y in range(height):
        row = raw_rows[y]
        pixel_row = []
        pos = 0

        if color_type == 2 and bit_depth == 8:
            for x in range(width):
                r = row[pos]
                g = row[pos + 1]
                b = row[pos + 2]
                pixel_row.append((r, g, b))
                pos += 3
        elif color_type == 6 and bit_depth == 8:
            for x in range(width):
                r = row[pos]
                g = row[pos + 1]
                b = row[pos + 2]
                a = row[pos + 3]
                pixel_row.append((r, g, b, a))
                pos += 4
        elif color_type == 3 and bit_depth == 8:
            for x in range(width):
                pixel_row.append(row[x])
        elif color_type == 3 and bit_depth == 4:
            for x in range(width):
                byte_idx = x // 2
                if x % 2 == 0:
                    val = (row[byte_idx] >> 4) & 0x0F
                else:
                    val = row[byte_idx] & 0x0F
                pixel_row.append(val)
        elif color_type == 3 and bit_depth == 1:
            for x in range(width):
                byte_idx = x // 8
                bit_idx = 7 - (x % 8)
                val = (row[byte_idx] >> bit_idx) & 1
                pixel_row.append(val)
        elif color_type == 0 and bit_depth == 8:
            for x in range(width):
                gray = row[x]
                pixel_row.append(gray)
        else:
            raise ValueError(f"Unsupported color type {color_type} with bit depth {bit_depth}")

        pixels.append(pixel_row)

    return pixels


def write_png(img, filter_type=0):
    width = img.width
    height = img.height
    bit_depth = img.bit_depth
    color_type = img.color_type
    channels = img.channels

    result = bytearray()
    result.extend(PNG_SIGNATURE)

    ihdr_data = struct.pack('>II', width, height)
    ihdr_data += bytes([bit_depth, color_type, 0, 0, 0])
    result.extend(PNGChunk(b'IHDR', ihdr_data).encode())

    if color_type == 3 and img.palette:
        plte_data = bytearray()
        for color in img.palette:
            plte_data.append(color[0])
            plte_data.append(color[1])
            plte_data.append(color[2])
        result.extend(PNGChunk(b'PLTE', bytes(plte_data)).encode())

    raw_rows = _encode_pixels(img)

    bpp = img.bytes_per_pixel
    stride = _raw_row_bytes(width, bit_depth, channels)

    filtered_data = bytearray()
    prev_row = None

    for y in range(height):
        raw_row = raw_rows[y]
        filter_fn = FILTER_FUNCTIONS[filter_type]
        filtered_row = filter_fn(raw_row, prev_row, bpp)
        filtered_data.append(filter_type)
        filtered_data.extend(filtered_row)
        prev_row = raw_row

    compressed = zlib_compress(bytes(filtered_data))
    result.extend(PNGChunk(b'IDAT', compressed).encode())
    result.extend(PNGChunk(b'IEND', b'').encode())

    return bytes(result)


def _encode_pixels(img):
    width = img.width
    height = img.height
    bit_depth = img.bit_depth
    color_type = img.color_type

    raw_rows = []

    for y in range(height):
        row = img.pixels[y]
        raw_row = bytearray()

        if color_type == 2 and bit_depth == 8:
            for x in range(width):
                r, g, b = row[x][:3]
                raw_row.append(r)
                raw_row.append(g)
                raw_row.append(b)
        elif color_type == 6 and bit_depth == 8:
            for x in range(width):
                pixel = row[x]
                if len(pixel) == 4:
                    r, g, b, a = pixel
                else:
                    r, g, b = pixel
                    a = 255
                raw_row.append(r)
                raw_row.append(g)
                raw_row.append(b)
                raw_row.append(a)
        elif color_type == 3 and bit_depth == 8:
            for x in range(width):
                raw_row.append(row[x] & 0xFF)
        elif color_type == 3 and bit_depth == 4:
            for x in range(0, width, 2):
                high = (row[x] & 0x0F) << 4
                if x + 1 < width:
                    low = row[x + 1] & 0x0F
                else:
                    low = 0
                raw_row.append(high | low)
        elif color_type == 3 and bit_depth == 1:
            byte_val = 0
            bit_pos = 7
            for x in range(width):
                byte_val |= (row[x] & 1) << bit_pos
                bit_pos -= 1
                if bit_pos < 0:
                    raw_row.append(byte_val)
                    byte_val = 0
                    bit_pos = 7
            if bit_pos != 7:
                raw_row.append(byte_val)
        elif color_type == 0 and bit_depth == 8:
            for x in range(width):
                if isinstance(row[x], tuple):
                    gray = row[x][0]
                else:
                    gray = row[x]
                raw_row.append(gray)
        else:
            raise ValueError(f"Unsupported color type {color_type} with bit depth {bit_depth}")

        raw_rows.append(bytes(raw_row))

    return raw_rows
