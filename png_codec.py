import struct
import zlib
from deflate import zlib_compress, zlib_decompress
from palette import build_palette_and_quantize, PaletteError


PNG_SIGNATURE = b'\x89PNG\r\n\x1a\n'

FILTER_NAMES = {0: "None", 1: "Sub", 2: "Up", 3: "Average", 4: "Paeth"}
COLOR_TYPE_NAMES = {0: "Grayscale", 2: "RGB", 3: "Indexed", 4: "Grayscale+Alpha", 6: "RGBA"}


class PNGError(Exception):
    def __init__(self, code, message, details=None):
        super().__init__(message)
        self.code = code
        self.details = details

    def user_message(self):
        lines = [f"[PNG Error] {self.code}: {self}"]
        if self.details:
            lines.append(f"  Details: {self.details}")
        lines.append("")
        lines.append("  Troubleshooting:")
        if self.code == "SIGNATURE":
            lines.append("  - File may be corrupt or not a PNG at all")
            lines.append("  - Check file extension matches actual format")
        elif self.code == "TRUNCATED":
            lines.append("  - File was cut off during download/save")
            lines.append("  - Redownload or regenerate the file")
        elif self.code == "CRC":
            lines.append("  - Chunk data corrupted in transit or storage")
            lines.append("  - Consider re-encoding from source")
        elif self.code == "FILTER":
            lines.append("  - Invalid filter byte indicates corrupt or malformed data")
            lines.append("  - Only filter types 0-4 (None/Sub/Up/Average/Paeth) are valid")
        elif self.code == "MISSING_CHUNK":
            lines.append(f"  - Required chunk {self.details.get('chunk', '')} is missing")
            lines.append("  - IHDR must be first chunk, IEND must be last")
        elif self.code == "IDAT_EMPTY":
            lines.append("  - No IDAT chunks found; image has no pixel data")
        elif self.code == "DECOMPRESS":
            lines.append("  - Compressed IDAT stream is corrupt")
            lines.append("  - This often happens with partial file downloads")
        elif self.code == "COLOR_MODE":
            lines.append("  - Unsupported combination of color type and bit depth")
            lines.append("  - Supported: 8-bit for all modes, plus 1/4-bit for Indexed")
        elif self.code == "INDEXED_NO_PALETTE":
            lines.append("  - Indexed color (type 3) requires a PLTE chunk before IDAT")
        return "\n".join(lines)


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

    @property
    def is_critical(self):
        return (self.chunk_type[0] & 0x20) == 0

    @property
    def is_public(self):
        return (self.chunk_type[1] & 0x20) == 0

    def encode(self):
        length = struct.pack('>I', len(self.data))
        chunk_data = self.chunk_type + self.data
        crc = struct.pack('>I', crc32(chunk_data))
        return length + chunk_data + crc


def parse_chunks(data, strict_crc=True):
    if len(data) < 8:
        raise PNGError("TRUNCATED", "File too small to contain PNG signature",
                       details={"got_bytes": len(data), "required": 8})

    if data[:8] != PNG_SIGNATURE:
        raise PNGError("SIGNATURE", "Invalid PNG signature - file is not a PNG",
                       details={"got": data[:8].hex()})

    pos = 8
    chunks = []
    chunk_order = 0

    while pos < len(data):
        if pos + 12 > len(data):
            raise PNGError("TRUNCATED",
                           f"Unexpected end of file at chunk header, byte {pos}/{len(data)}",
                           details={"position": pos, "remaining": len(data) - pos})

        length = struct.unpack_from('>I', data, pos)[0]

        if pos + 12 + length > len(data):
            raise PNGError("TRUNCATED",
                           f"Chunk at {pos} declares {length} bytes but only {len(data) - pos - 12} remain",
                           details={"chunk_offset": pos, "declared_length": length,
                                    "available": len(data) - pos - 12})

        chunk_type = bytes(data[pos + 4:pos + 8])
        chunk_data = bytes(data[pos + 8:pos + 8 + length])
        stored_crc = struct.unpack_from('>I', data, pos + 8 + length)[0]

        calc_crc = crc32(chunk_type + chunk_data)
        if calc_crc != stored_crc:
            type_str = chunk_type.decode('ascii', errors='replace')
            if strict_crc:
                raise PNGError("CRC",
                               f"CRC mismatch for chunk #{chunk_order} '{type_str}'",
                               details={"chunk_type": type_str, "chunk_index": chunk_order,
                                        "expected": f"0x{calc_crc:08x}",
                                        "stored": f"0x{stored_crc:08x}"})

        is_critical = (chunk_type[0] & 0x20) == 0
        if is_critical and chunk_type not in (b'IHDR', b'PLTE', b'IDAT', b'IEND', b'sRGB', b'gAMA', b'pHYs'):
            pass

        chunks.append(PNGChunk(chunk_type, chunk_data))
        pos += 12 + length
        chunk_order += 1

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
        self.raw_idat_size = 0
        self.compressed_idat_size = 0
        self.idat_chunk_count = 0
        self.chunks_found = []

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

    def describe(self):
        lines = []
        lines.append(f"  Dimensions: {self.width} x {self.height}")
        lines.append(f"  Bit depth : {self.bit_depth}")
        lines.append(f"  Color type: {self.color_type} ({COLOR_TYPE_NAMES.get(self.color_type, 'Unknown')})")
        lines.append(f"  Channels  : {self.channels}")
        if self.palette:
            lines.append(f"  Palette   : {len(self.palette)} entries")
        if self.idat_chunk_count:
            lines.append(f"  IDAT chunks: {self.idat_chunk_count}")
            lines.append(f"  Compressed  : {self.compressed_idat_size:,} bytes")
            lines.append(f"  Decompressed: {self.raw_idat_size:,} bytes")
            if self.compressed_idat_size > 0:
                ratio = self.raw_idat_size / self.compressed_idat_size
                lines.append(f"  Ratio       : {ratio:.2f}x")
        return "\n".join(lines)


def _validate_ihdr(bit_depth, color_type):
    valid = {
        0: [1, 2, 4, 8, 16],
        2: [8, 16],
        3: [1, 2, 4, 8],
        4: [8, 16],
        6: [8, 16],
    }
    if color_type not in valid:
        raise PNGError("COLOR_MODE", f"Invalid color type {color_type}",
                       details={"color_type": color_type})
    if bit_depth not in valid[color_type]:
        raise PNGError("COLOR_MODE",
                       f"Bit depth {bit_depth} not valid for color type {color_type} ({COLOR_TYPE_NAMES.get(color_type, 'Unknown')})",
                       details={"color_type": color_type, "bit_depth": bit_depth,
                                "valid_depths": valid[color_type]})


FILTER_FUNCTIONS = None
UNFILTER_FUNCTIONS = None


def _init_filter_fns():
    global FILTER_FUNCTIONS, UNFILTER_FUNCTIONS
    if FILTER_FUNCTIONS is not None:
        return

    def filter_none(s, p, b): return s[:]
    def filter_sub(s, p, b):
        r = bytearray()
        for i in range(len(s)):
            left = s[i - b] if i >= b else 0
            r.append((s[i] - left) & 0xFF)
        return bytes(r)
    def filter_up(s, p, b):
        r = bytearray()
        for i in range(len(s)):
            up = p[i] if p else 0
            r.append((s[i] - up) & 0xFF)
        return bytes(r)
    def filter_avg(s, p, b):
        r = bytearray()
        for i in range(len(s)):
            left = s[i - b] if i >= b else 0
            up = p[i] if p else 0
            r.append((s[i] - (left + up) // 2) & 0xFF)
        return bytes(r)
    def filter_paeth(s, p, b):
        r = bytearray()
        for i in range(len(s)):
            left = s[i - b] if i >= b else 0
            up = p[i] if p else 0
            ul = p[i - b] if (p and i >= b) else 0
            r.append((s[i] - paeth_predictor(left, up, ul)) & 0xFF)
        return bytes(r)

    def unf_none(f, p, b): return f[:]
    def unf_sub(f, p, b):
        r = bytearray()
        for i in range(len(f)):
            left = r[i - b] if i >= b else 0
            r.append((f[i] + left) & 0xFF)
        return bytes(r)
    def unf_up(f, p, b):
        r = bytearray()
        for i in range(len(f)):
            up = p[i] if p else 0
            r.append((f[i] + up) & 0xFF)
        return bytes(r)
    def unf_avg(f, p, b):
        r = bytearray()
        for i in range(len(f)):
            left = r[i - b] if i >= b else 0
            up = p[i] if p else 0
            r.append((f[i] + (left + up) // 2) & 0xFF)
        return bytes(r)
    def unf_paeth(f, p, b):
        r = bytearray()
        for i in range(len(f)):
            left = r[i - b] if i >= b else 0
            up = p[i] if p else 0
            ul = p[i - b] if (p and i >= b) else 0
            r.append((f[i] + paeth_predictor(left, up, ul)) & 0xFF)
        return bytes(r)

    FILTER_FUNCTIONS = [filter_none, filter_sub, filter_up, filter_avg, filter_paeth]
    UNFILTER_FUNCTIONS = [unf_none, unf_sub, unf_up, unf_avg, unf_paeth]


_init_filter_fns()


def _raw_row_bytes(width, bit_depth, channels):
    return (width * bit_depth * channels + 7) // 8


def read_png(data, strict_crc=True):
    chunks = parse_chunks(data, strict_crc=strict_crc)

    img = PNGImage()
    idat_data = bytearray()
    ihdr_seen = False
    iend_seen = False
    plte_seen = False

    for idx, chunk in enumerate(chunks):
        ct = chunk.chunk_type
        img.chunks_found.append(ct.decode('ascii', errors='replace'))

        if ct == b'IHDR':
            if idx != 0:
                raise PNGError("MISSING_CHUNK", "IHDR must be the first chunk",
                               details={"chunk": "IHDR", "actual_first": chunks[0].chunk_type.decode('ascii', errors='replace')})
            if len(chunk.data) != 13:
                raise PNGError("HEADER", f"IHDR has invalid size {len(chunk.data)} (expected 13)",
                               details={"ihdr_size": len(chunk.data)})
            img.width = struct.unpack_from('>I', chunk.data, 0)[0]
            img.height = struct.unpack_from('>I', chunk.data, 4)[0]
            img.bit_depth = chunk.data[8]
            img.color_type = chunk.data[9]
            img.compression = chunk.data[10]
            img.filter = chunk.data[11]
            img.interlace = chunk.data[12]

            _validate_ihdr(img.bit_depth, img.color_type)

            if img.compression != 0:
                raise PNGError("FEATURE", f"Unsupported compression method {img.compression} (only method 0/deflate supported)")
            if img.filter != 0:
                raise PNGError("FEATURE", f"Unsupported filter method {img.filter} (only method 0/adaptive supported)")
            if img.interlace != 0:
                raise PNGError("FEATURE", "Adam7 interlacing is not implemented in this decoder",
                               details={"interlace_method": img.interlace})
            ihdr_seen = True

        elif ct == b'PLTE':
            if not ihdr_seen:
                raise PNGError("ORDER", "PLTE appeared before IHDR",
                               details={"chunk": "PLTE"})
            if img.idat_chunk_count > 0:
                raise PNGError("ORDER", "PLTE appeared after IDAT (must come before)",
                               details={"chunk": "PLTE"})
            if len(chunk.data) % 3 != 0:
                raise PNGError("HEADER", f"PLTE size {len(chunk.data)} is not a multiple of 3",
                               details={"plte_size": len(chunk.data)})
            palette_size = len(chunk.data) // 3
            img.palette = []
            for i in range(palette_size):
                r = chunk.data[i * 3]
                g = chunk.data[i * 3 + 1]
                b = chunk.data[i * 3 + 2]
                img.palette.append((r, g, b))
            plte_seen = True

        elif ct == b'IDAT':
            if not ihdr_seen:
                raise PNGError("ORDER", "IDAT appeared before IHDR",
                               details={"chunk": "IDAT"})
            idat_data.extend(chunk.data)
            img.idat_chunk_count += 1

        elif ct == b'IEND':
            iend_seen = True
            break

        else:
            is_critical = (chunk.chunk_type[0] & 0x20) == 0
            if is_critical:
                pass

    if not ihdr_seen:
        raise PNGError("MISSING_CHUNK", "No IHDR chunk found - file is corrupt",
                       details={"chunk": "IHDR"})

    if img.idat_chunk_count == 0:
        raise PNGError("IDAT_EMPTY", "No IDAT chunks found - image contains no pixel data",
                       details={"chunks_found": img.chunks_found})

    if img.color_type == 3 and not img.palette:
        raise PNGError("INDEXED_NO_PALETTE",
                       f"Color type 3 (Indexed) requires PLTE but none was found",
                       details={"color_type": 3})

    if not iend_seen:
        raise PNGError("MISSING_CHUNK", "No IEND chunk found - file appears to be truncated",
                       details={"chunk": "IEND", "chunks_found": img.chunks_found})

    img.compressed_idat_size = len(idat_data)

    try:
        decompressed = zlib_decompress(bytes(idat_data))
    except Exception as e:
        raise PNGError("DECOMPRESS",
                       f"Failed to decompress IDAT stream: {e}",
                       details={"compressed_size": len(idat_data), "error": str(e)})

    img.raw_idat_size = len(decompressed)

    bpp = img.bytes_per_pixel
    stride = _raw_row_bytes(img.width, img.bit_depth, img.channels)
    expected_size = img.height * (stride + 1)

    if len(decompressed) < expected_size:
        raise PNGError("TRUNCATED",
                       f"Decompressed IDAT too small: {len(decompressed)} bytes, expected at least {expected_size} "
                       f"({img.height} rows × ({stride}+1 filter byte))",
                       details={"got": len(decompressed), "expected": expected_size,
                                "width": img.width, "height": img.height, "stride": stride})

    raw_rows = []
    prev_row = None
    pos = 0

    for row_idx in range(img.height):
        if pos >= len(decompressed):
            raise PNGError("TRUNCATED",
                           f"Ran out of decompressed data at row {row_idx}/{img.height}",
                           details={"position": pos, "total_decompressed": len(decompressed)})

        filter_byte = decompressed[pos]
        pos += 1

        if filter_byte > 4:
            raise PNGError("FILTER",
                           f"Invalid filter type {filter_byte} at row {row_idx}. "
                           f"Allowed: 0=None, 1=Sub, 2=Up, 3=Average, 4=Paeth",
                           details={"row": row_idx, "filter_byte": filter_byte,
                                    "filter_name": FILTER_NAMES.get(filter_byte, "INVALID")})

        remaining = len(decompressed) - pos
        if remaining < stride:
            raise PNGError("TRUNCATED",
                           f"Row {row_idx} needs {stride} bytes after filter byte, only {remaining} left",
                           details={"row": row_idx, "needed": stride, "available": remaining})

        filtered_row = decompressed[pos:pos + stride]
        pos += stride

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
        elif color_type == 3:
            if bit_depth == 8:
                for x in range(width):
                    idx = row[x]
                    if img.palette and idx >= len(img.palette):
                        raise PNGError("INDEXED_OOB",
                                       f"Row {y} col {x}: palette index {idx} >= palette size {len(img.palette)}",
                                       details={"row": y, "col": x, "index": idx, "palette_size": len(img.palette)})
                    pixel_row.append(idx)
            elif bit_depth == 4:
                for x in range(width):
                    byte_idx = x // 2
                    if x % 2 == 0:
                        idx = (row[byte_idx] >> 4) & 0x0F
                    else:
                        idx = row[byte_idx] & 0x0F
                    if img.palette and idx >= len(img.palette):
                        raise PNGError("INDEXED_OOB",
                                       f"Row {y} col {x}: palette index {idx} >= palette size {len(img.palette)}",
                                       details={"row": y, "col": x, "index": idx, "palette_size": len(img.palette)})
                    pixel_row.append(idx)
            elif bit_depth == 1:
                for x in range(width):
                    byte_idx = x // 8
                    bit_idx = 7 - (x % 8)
                    idx = (row[byte_idx] >> bit_idx) & 1
                    if img.palette and idx >= len(img.palette):
                        raise PNGError("INDEXED_OOB",
                                       f"Row {y} col {x}: palette index {idx} >= palette size {len(img.palette)}",
                                       details={"row": y, "col": x, "index": idx, "palette_size": len(img.palette)})
                    pixel_row.append(idx)
            elif bit_depth == 2:
                for x in range(width):
                    byte_idx = x // 4
                    shift = 6 - 2 * (x % 4)
                    idx = (row[byte_idx] >> shift) & 0x03
                    if img.palette and idx >= len(img.palette):
                        raise PNGError("INDEXED_OOB",
                                       f"Row {y} col {x}: palette index {idx} >= palette size {len(img.palette)}",
                                       details={"row": y, "col": x, "index": idx, "palette_size": len(img.palette)})
                    pixel_row.append(idx)
        elif color_type == 0 and bit_depth == 8:
            for x in range(width):
                gray = row[x]
                pixel_row.append(gray)
        else:
            raise PNGError("COLOR_MODE",
                           f"Unsupported decoder combination: color_type={color_type}, bit_depth={bit_depth}",
                           details={"implemented_modes": ["8-bit RGB(2)", "8-bit RGBA(6)", "1/2/4/8-bit Indexed(3)", "8-bit Grayscale(0)"]})

        pixels.append(pixel_row)

    return pixels


def write_png(img, filter_type=0, palette_strategy="quantize", idat_split=None):
    _init_filter_fns()

    width = img.width
    height = img.height
    bit_depth = img.bit_depth
    color_type = img.color_type
    channels = img.channels

    if color_type == 3 and not img.palette:
        quant = build_palette_and_quantize(img.pixels, bit_depth, strategy=palette_strategy)
        img.palette = quant.palette
        img.pixels = quant.indices

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

    bpp = max(1, (bit_depth * channels + 7) // 8)
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

    if idat_split and idat_split > 0 and len(compressed) > idat_split:
        for i in range(0, len(compressed), idat_split):
            part = compressed[i:i + idat_split]
            result.extend(PNGChunk(b'IDAT', part).encode())
    else:
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
        elif color_type == 3:
            max_idx = (1 << bit_depth) - 1
            if bit_depth == 8:
                for x in range(width):
                    val = row[x] & 0xFF
                    if val > max_idx:
                        raise PaletteError(
                            f"Index {val} exceeds maximum for {bit_depth}-bit (max {max_idx}). "
                            f"Either reduce palette size or increase bit depth.",
                            color_count=val, max_colors=max_idx + 1
                        )
                    raw_row.append(val)
            elif bit_depth == 4:
                for x in range(0, width, 2):
                    val1 = row[x] & 0x0F
                    if val1 > max_idx:
                        raise PaletteError(f"Index {val1} > {max_idx} at col {x}")
                    if x + 1 < width:
                        val2 = row[x + 1] & 0x0F
                        if val2 > max_idx:
                            raise PaletteError(f"Index {val2} > {max_idx} at col {x + 1}")
                    else:
                        val2 = 0
                    raw_row.append((val1 << 4) | val2)
            elif bit_depth == 2:
                for x in range(0, width, 4):
                    byte_val = 0
                    for s in range(4):
                        col = x + s
                        if col < width:
                            v = row[col] & 0x03
                            if v > max_idx:
                                raise PaletteError(f"Index {v} > {max_idx} at col {col}")
                        else:
                            v = 0
                        byte_val |= v << (6 - 2 * s)
                    raw_row.append(byte_val)
            elif bit_depth == 1:
                byte_val = 0
                bit_pos = 7
                for x in range(width):
                    v = row[x] & 1
                    if v > max_idx:
                        raise PaletteError(f"Index {v} > {max_idx} at col {x}")
                    byte_val |= v << bit_pos
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
            raise ValueError(f"Unsupported encoder combination: color_type={color_type}, bit_depth={bit_depth}")

        raw_rows.append(bytes(raw_row))

    return raw_rows
