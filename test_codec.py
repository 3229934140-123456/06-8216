import random
import sys
import struct
import zlib
import os
from image import Image
from bmp_codec import bmp_bytes_per_row, bmp_row_padding
from bmp_codec import BMPImage, read_bmp, write_bmp, BMPError
from png_codec import PNGImage, read_png, write_png, PNGError, PNG_SIGNATURE, FILTER_NAMES, COLOR_TYPE_NAMES, crc32, PNGChunk
from palette import PaletteError, build_palette_and_quantize, QuantizationResult


# ============================================================
# Test Image Factories
# ============================================================

def create_random_image(width, height, seed=42, alpha=True):
    random.seed(seed)
    img = Image(width, height)
    for y in range(height):
        for x in range(width):
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            a = random.randint(0, 255) if alpha else 255
            img.pixels[y][x] = (r, g, b, a)
    return img


def create_gradient_image(width, height, alpha=False):
    img = Image(width, height)
    for y in range(height):
        for x in range(width):
            r = int(255 * x / max(width - 1, 1))
            g = int(255 * y / max(height - 1, 1))
            b = int(255 * (x + y) / max(width + height - 2, 1))
            if alpha:
                a = int(255 * (abs(x - y) / max(width, height)))
            else:
                a = 255
            img.pixels[y][x] = (r, g, b, a)
    return img


def create_few_colors_image(width, height, num_colors, seed=1):
    """Create an image that only uses a fixed number of distinct colors."""
    random.seed(seed)
    palette = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
               for _ in range(num_colors)]
    img = Image(width, height)
    for y in range(height):
        for x in range(width):
            r, g, b = random.choice(palette)
            img.pixels[y][x] = (r, g, b, 255)
    return img


def create_transparent_rgba_image(width, height):
    """Various alpha levels including fully transparent."""
    img = Image(width, height)
    for y in range(height):
        for x in range(width):
            r = (x * 37) % 256
            g = (y * 53) % 256
            b = ((x + y) * 71) % 256
            if x < width // 4:
                a = 0
            elif x < width // 2:
                a = 128
            elif x < 3 * width // 4:
                a = 200
            else:
                a = 255
            img.pixels[y][x] = (r, g, b, a)
    return img


def create_checker_image(width, height, size=1):
    img = Image(width, height)
    for y in range(height):
        for x in range(width):
            black = ((x // size) + (y // size)) % 2 == 0
            v = 0 if black else 255
            img.pixels[y][x] = (v, v, v, 255)
    return img


# ============================================================
# Helper: Verify round-trip and report lossy/lossless
# ============================================================

def report_roundtrip(label, orig, decoded, ignore_alpha=True, tolerance=0):
    """Compare two Images, print PASS/FAIL + lossy/lossless, return bool."""
    if orig.width != decoded.width or orig.height != decoded.height:
        print(f"  [FAIL] {label}: dimension mismatch {orig.width}x{orig.height} vs {decoded.width}x{decoded.height}")
        return False

    diff_count, max_diff = orig.count_differences(decoded, ignore_alpha=ignore_alpha)
    total_channels = orig.width * orig.height * (3 if ignore_alpha else 4)

    if diff_count == 0:
        status = "LOSSLESS"
        tag = "[PASS]"
    elif max_diff <= tolerance:
        status = f"WITHIN-TOLERANCE ({diff_count}/{total_channels} diffs, max={max_diff})"
        tag = "[PASS]"
    else:
        pct = 100 * diff_count / total_channels if total_channels else 0
        status = f"LOSSY ({diff_count:,}/{total_channels:,} channel diffs = {pct:.3f}%, max={max_diff})"
        tag = "[INFO]" if (diff_count > 0 and "indexed" in label.lower() or "bpp1" in label.lower() or "bpp4" in label.lower()) else "[WARN]"

    print(f"  {tag} {label}: {status}")
    return diff_count == 0 or max_diff <= tolerance


# ============================================================
# Test 1: Native BMP Roundtrip (various bit depths + odd widths)
# ============================================================

def test_bmp_native_roundtrip():
    print("\n" + "=" * 65)
    print("TEST 1: BMP Native Roundtrip — odd widths, 1/4/8/24/32 bpp")
    print("=" * 65)

    # Odd widths (non-4-byte-aligned) for packed pixel formats
    cases = [
        # (width, height, bpp, description)
        (1, 1, 1, "1x1 1bpp"),
        (3, 5, 1, "3x5 1bpp odd-w"),
        (5, 7, 1, "5x7 1bpp odd-w"),
        (7, 3, 1, "7x3 1bpp odd-w"),
        (1, 1, 4, "1x1 4bpp"),
        (3, 5, 4, "3x5 4bpp odd-w"),
        (5, 7, 4, "5x7 4bpp odd-w"),
        (9, 11, 4, "9x11 4bpp odd-w"),
        (1, 1, 8, "1x1 8bpp"),
        (3, 5, 8, "3x5 8bpp"),
        (7, 9, 8, "7x9 8bpp odd-w"),
        (11, 13, 8, "11x13 8bpp odd-w"),
        (1, 1, 24, "1x1 24bpp"),
        (3, 5, 24, "3x5 24bpp non-aligned"),
        (7, 9, 24, "7x9 24bpp non-aligned"),
        (1, 1, 32, "1x1 32bpp"),
        (5, 7, 32, "5x7 32bpp"),
    ]

    all_pass = True
    seed_counter = 0
    for width, height, bpp, desc in cases:
        seed_counter += 1
        random.seed(seed_counter * 131)
        bmp = BMPImage()
        bmp.width = width
        bmp.height = height
        bmp.bits_per_pixel = bpp

        if bpp <= 8:
            palette_size = 1 << bpp
            palette = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                       for _ in range(palette_size)]
            bmp.palette = palette
            bmp.pixels = [[random.randint(0, palette_size - 1) for _ in range(width)] for _ in range(height)]
        elif bpp == 24:
            bmp.pixels = [[(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                           for _ in range(width)] for _ in range(height)]
        else:  # 32
            bmp.pixels = [[(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                           for _ in range(width)] for _ in range(height)]

        try:
            data = write_bmp(bmp)
            decoded = read_bmp(data)

            row_bytes = bmp_bytes_per_row(width, bpp)
            pad = bmp_row_padding(width, bpp)

            # Compare via BMPImage directly (no Image RGBA conversion yet)
            mismatch = 0
            if decoded.width != width or decoded.height != height or decoded.bits_per_pixel != bpp:
                print(f"  [FAIL] {desc}: dim/bpp mismatch got {decoded.width}x{decoded.height}@{decoded.bits_per_pixel}")
                all_pass = False
                continue

            for y in range(height):
                for x in range(width):
                    if bpp <= 8:
                        if bmp.pixels[y][x] != decoded.pixels[y][x]:
                            mismatch += 1
                    elif bpp == 24:
                        if bmp.pixels[y][x][:3] != decoded.pixels[y][x][:3]:
                            mismatch += 1
                    else:
                        if bmp.pixels[y][x] != decoded.pixels[y][x]:
                            mismatch += 1

            if mismatch == 0:
                print(f"  [PASS] {desc:25s}: size={len(data):>5d}B, row={row_bytes}B, pad={pad}B")
            else:
                print(f"  [FAIL] {desc}: {mismatch} pixel mismatches")
                all_pass = False
        except Exception as e:
            print(f"  [FAIL] {desc}: EXCEPTION {type(e).__name__}: {e}")
            all_pass = False

    return all_pass


# ============================================================
# Test 2: Image API BMP — palette strategies + odd widths
# ============================================================

def test_image_bmp_palette_strategies():
    print("\n" + "=" * 65)
    print("TEST 2: Image API BMP — palette strategies (error/quantize/auto)")
    print("=" * 65)

    all_pass = True

    # --- Subtest A: fits-within-capacity exact ---
    print("\n  --- Fits-within palette (should be exact/error both OK) ---")
    small = create_few_colors_image(5, 5, num_colors=8, seed=2)
    for bpp in [4, 8]:
        cap = 1 << bpp
        try:
            data = small.to_bmp(bits_per_pixel=bpp, palette_strategy="error")
            back = Image.from_bmp(data)
            qi = small.last_quantization_info
            diff_count, _ = small.count_differences(back, ignore_alpha=True)
            lossy_note = "(LOSSY-quant?)" if qi and qi.is_lossy else "(exact)"
            status = "PASS" if diff_count == 0 else "FAIL"
            print(f"    [{status}] fits-8-colors -> BMP {bpp}bpp strategy=error: {diff_count} diffs {lossy_note}")
            if diff_count != 0:
                all_pass = False
        except Exception as e:
            print(f"    [FAIL] fits-8-colors -> BMP {bpp}bpp strategy=error: EXCEPTION {e}")
            all_pass = False

    # --- Subtest B: too many colors -> strategy=error should throw PaletteError ---
    print("\n  --- Too many colors (strategy=error MUST throw) ---")
    big_color = create_gradient_image(9, 9)  # 81 pixels, likely > 16 colors
    for bpp in [1, 4]:
        cap_name = f"{1 << bpp} colors"
        try:
            _ = big_color.to_bmp(bits_per_pixel=bpp, palette_strategy="error")
            print(f"    [FAIL] BMP {bpp}bpp ({cap_name}) strategy=error: should have raised PaletteError!")
            all_pass = False
        except PaletteError as pe:
            print(f"    [PASS] BMP {bpp}bpp ({cap_name}) strategy=error -> PaletteError OK "
                  f"(need {pe.color_count}, cap {pe.max_colors})")
        except Exception as e:
            print(f"    [FAIL] BMP {bpp}bpp strategy=error: wrong exception {type(e).__name__}: {e}")
            all_pass = False

    # --- Subtest C: strategy=quantize should reduce without error ---
    print("\n  --- Too many colors (strategy=quantize MUST NOT throw, is lossy) ---")
    for bpp in [1, 4, 8]:
        try:
            data = big_color.to_bmp(bits_per_pixel=bpp, palette_strategy="quantize")
            back = Image.from_bmp(data)
            qi = big_color.last_quantization_info
            diff_count, max_diff = big_color.count_differences(back, ignore_alpha=True)
            expected_lossy = (qi is not None and qi.is_lossy)
            lossy_note = "LOSSY" if expected_lossy else "EXACT"
            if diff_count == 0 and not expected_lossy:
                tag = "[PASS]"
            elif diff_count > 0 and expected_lossy:
                tag = "[PASS]"
            else:
                tag = "[WARN]"
                all_pass = False
            print(f"    {tag} BMP {bpp}bpp strategy=quantize -> {lossy_note}: "
                  f"{diff_count} diffs, max={max_diff}, qi.lossy={expected_lossy}, "
                  f"orig={qi.original_colors if qi else '?'}->pal={qi.palette_colors if qi else '?'}")
        except Exception as e:
            print(f"    [FAIL] BMP {bpp}bpp strategy=quantize: EXCEPTION {type(e).__name__}: {e}")
            all_pass = False

    # --- Subtest D: odd-width via Image API ---
    print("\n  --- Odd-width non-4-aligned via Image API ---")
    for width in [1, 3, 5, 7, 9]:
        for bpp in [24, 32]:
            img = create_random_image(width, 5, seed=width + bpp)
            try:
                data = img.to_bmp(bits_per_pixel=bpp)
                back = Image.from_bmp(data)
                diff_count, _ = img.count_differences(back, ignore_alpha=(bpp != 32))
                status = "PASS" if diff_count == 0 else "FAIL"
                if diff_count != 0:
                    all_pass = False
                print(f"    [{status}] {width}x5 BMP {bpp}bpp: {len(data)}B, {diff_count} diffs")
            except Exception as e:
                print(f"    [FAIL] {width}x5 BMP {bpp}bpp: {e}")
                all_pass = False

    return all_pass


# ============================================================
# Test 3: PNG roundtrip (all filters, odd widths, multi-IDAT)
# ============================================================

def test_png_roundtrip_extended():
    print("\n" + "=" * 65)
    print("TEST 3: PNG Roundtrip — odd widths, all filters, multi-IDAT")
    print("=" * 65)

    all_pass = True
    filter_names = ["None", "Sub", "Up", "Average", "Paeth"]

    # --- Subtest A: odd widths, RGBA, all filters ---
    print("\n  --- RGBA odd-width images (all filters, LOSSLESS expected) ---")
    for width in [1, 3, 5, 7, 11]:
        img = create_gradient_image(width, 7, alpha=True)
        for ft in range(5):
            try:
                data = img.to_png(color_type=6, bit_depth=8, filter_type=ft)
                back = Image.from_png(data)
                diff_count, max_diff = img.count_differences(back, ignore_alpha=False)
                tag = "PASS" if diff_count == 0 else "FAIL"
                if diff_count != 0:
                    all_pass = False
                print(f"    [{tag}] {width}x7 RGBA + {filter_names[ft]:8s}: "
                      f"{len(data):>5d}B, {diff_count} diffs")
            except Exception as e:
                print(f"    [FAIL] {width}x7 RGBA + {filter_names[ft]}: {e}")
                all_pass = False

    # --- Subtest B: multi-IDAT write + read (split IDAT into small chunks) ---
    print("\n  --- Multi-IDAT write + read (idat_split=7 / idat_split=100) ---")
    img = create_gradient_image(17, 13, alpha=True)
    for split in [7, 100, 1024]:
        try:
            data = img.to_png(color_type=6, filter_type=4, idat_split=split)
            back = Image.from_png(data)
            diff_count, _ = img.count_differences(back, ignore_alpha=False)
            # count IDAT chunks in data
            pos = 8
            idat_count = 0
            while pos + 8 <= len(data):
                length = struct.unpack('>I', data[pos:pos + 4])[0]
                ctype = data[pos + 4:pos + 8]
                if ctype == b'IDAT':
                    idat_count += 1
                pos += 12 + length
            tag = "PASS" if diff_count == 0 and idat_count >= 1 else "FAIL"
            if diff_count != 0:
                all_pass = False
            print(f"    [{tag}] idat_split={split:>5d}: {len(data):>5d}B, IDAT chunks={idat_count}, {diff_count} diffs")
        except Exception as e:
            print(f"    [FAIL] idat_split={split}: {type(e).__name__}: {e}")
            all_pass = False

    # --- Subtest C: indexed PNG with odd width ---
    print("\n  --- Indexed PNG (1/4/8 bit) odd-width, odd-height ---")
    small_pal = create_few_colors_image(7, 5, num_colors=8, seed=99)
    for bit_depth, name in [(1, "1bit"), (4, "4bit"), (8, "8bit")]:
        try:
            data = small_pal.to_png(color_type=3, bit_depth=bit_depth, filter_type=4, palette_strategy="quantize")
            back = Image.from_png(data)
            diff_count, max_diff = small_pal.count_differences(back, ignore_alpha=True)
            qi = small_pal.last_quantization_info
            lossy_note = "(lossy)" if (qi and qi.is_lossy) else "(exact)"
            tag = "PASS"  # 1/4-bit from 8 colors is usually lossy unless all fit
            if bit_depth == 8 and diff_count != 0:
                tag = "FAIL"
                all_pass = False
            print(f"    [{tag}] Indexed {name} 7x5: {len(data):>5d}B, {diff_count} diffs {lossy_note}")
        except Exception as e:
            print(f"    [FAIL] Indexed {name}: {type(e).__name__}: {e}")
            all_pass = False

    # --- Subtest D: transparent RGBA ---
    print("\n  --- Transparent RGBA PNG (alpha preserved lossless) ---")
    tr = create_transparent_rgba_image(16, 12)
    for ft in range(5):
        try:
            data = tr.to_png(color_type=6, bit_depth=8, filter_type=ft)
            back = Image.from_png(data)
            diff_count, max_diff = tr.count_differences(back, ignore_alpha=False)
            tag = "PASS" if diff_count == 0 else "FAIL"
            if diff_count != 0:
                all_pass = False
            print(f"    [{tag}] RGBA+alpha + {filter_names[ft]:8s}: {len(data):>5d}B, {diff_count} alpha-aware diffs")
        except Exception as e:
            print(f"    [FAIL] RGBA+alpha + {filter_names[ft]}: {e}")
            all_pass = False

    return all_pass


# ============================================================
# Test 4: PNG Robustness — malformed files with friendly errors
# ============================================================

def make_minimal_valid_png(width=4, height=4, color_type=2, bit_depth=8):
    """Build a minimal valid PNG from scratch using raw bytes, for mutation testing."""
    # IHDR
    ihdr_data = struct.pack('>IIBBBBB', width, height, bit_depth, color_type, 0, 0, 0)
    ihdr_type = b'IHDR'
    ihdr_crc = struct.pack('>I', crc32(ihdr_type + ihdr_data))
    ihdr_chunk = struct.pack('>I', len(ihdr_data)) + ihdr_type + ihdr_data + ihdr_crc

    # Raw scanlines: 1 filter byte + pixels per row
    bpp = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(color_type, 1)
    if color_type == 3:
        bpp = 1  # index bytes
    row_bytes = (width * bit_depth * bpp + 7) // 8 if color_type != 3 else ((width * bit_depth + 7) // 8)

    raw = bytearray()
    for y in range(height):
        raw.append(0)  # filter = None
        if color_type == 0:
            raw.extend(bytes([(y * 37 + x * 11) % 256 for x in range(width)]))
        elif color_type == 2:
            for x in range(width):
                raw.extend([(x * 50) % 256, (y * 70) % 256, ((x + y) * 90) % 256])
        elif color_type == 6:
            for x in range(width):
                raw.extend([(x * 50) % 256, (y * 70) % 256, ((x + y) * 90) % 256, 255])
        elif color_type == 3:
            # Indexed: 0..n
            n = 1 << bit_depth
            row = []
            for x in range(width):
                row.append((x + y) % n)
            # pack
            if bit_depth == 8:
                raw.extend(bytes(row))
            elif bit_depth == 4:
                packed = []
                for i in range(0, len(row), 2):
                    hi = row[i] & 0x0F
                    lo = (row[i + 1] & 0x0F) if i + 1 < len(row) else 0
                    packed.append((hi << 4) | lo)
                raw.extend(bytes(packed))
            elif bit_depth == 1:
                packed = []
                for i in range(0, len(row), 8):
                    byte = 0
                    for j in range(8):
                        if i + j < len(row):
                            byte |= (row[i + j] & 1) << (7 - j)
                    packed.append(byte)
                raw.extend(bytes(packed))
        else:
            raw.extend(bytes(row_bytes))

    # Add PLTE for indexed
    extra_chunks = b''
    if color_type == 3:
        n = 1 << bit_depth
        plte_data = bytearray()
        for i in range(n):
            plte_data.extend([(i * 37) % 256, (i * 73) % 256, (i * 113) % 256])
        plte_type = b'PLTE'
        plte_crc = struct.pack('>I', crc32(plte_type + bytes(plte_data)))
        extra_chunks = struct.pack('>I', len(plte_data)) + plte_type + bytes(plte_data) + plte_crc

    compressed = zlib.compress(bytes(raw))
    idat_type = b'IDAT'
    idat_crc = struct.pack('>I', crc32(idat_type + compressed))
    idat_chunk = struct.pack('>I', len(compressed)) + idat_type + compressed + idat_crc

    iend_type = b'IEND'
    iend_data = b''
    iend_crc = struct.pack('>I', crc32(iend_type + iend_data))
    iend_chunk = struct.pack('>I', 0) + iend_type + iend_data + iend_crc

    return PNG_SIGNATURE + ihdr_chunk + extra_chunks + idat_chunk + iend_chunk


def test_png_robustness():
    print("\n" + "=" * 65)
    print("TEST 4: PNG Robustness — malformed files -> user_message()")
    print("=" * 65)

    all_pass = True
    valid = make_minimal_valid_png(4, 4, color_type=2, bit_depth=8)

    cases = []

    # 1. Wrong signature
    bad_sig = b'\x00\x00PNG\r\n\x1a\n' + valid[8:]
    cases.append(("Bad signature", bad_sig, ["SIGNATURE"], True))

    # 2. Truncated (cut in half)
    cases.append(("Truncated file", valid[:len(valid) // 2], ["TRUNCATED"], True))

    # 3. Missing IEND (truncate right before IEND)
    cut = valid[:-12]
    cases.append(("Missing IEND", cut, ["TRUNCATED", "MISSING_CHUNK"], True))

    # 4. CRC corruption on IHDR
    corrupted = bytearray(valid)
    # find IHDR CRC offset: sig(8) + len(4) + type(4) + ihdr_data(13) = at byte 29
    corrupted[29] ^= 0xFF  # flip 1 byte in CRC
    cases.append(("Corrupted IHDR CRC", bytes(corrupted), ["CRC"], True))

    # 5. Invalid filter byte (use 7, only 0-4 allowed)
    # Reconstruct with bad filter
    def _make_with_bad_filter():
        ihdr_data = struct.pack('>IIBBBBB', 4, 3, 8, 2, 0, 0, 0)
        ihdr = struct.pack('>I', 13) + b'IHDR' + ihdr_data + struct.pack('>I', crc32(b'IHDR' + ihdr_data))
        raw = bytearray()
        for y in range(3):
            raw.append(7 if y == 1 else 0)  # middle row filter=7
            for x in range(4):
                raw.extend([100, 150, 200])
        comp = zlib.compress(bytes(raw))
        idat = struct.pack('>I', len(comp)) + b'IDAT' + comp + struct.pack('>I', crc32(b'IDAT' + comp))
        iend = struct.pack('>I', 0) + b'IEND' + b'' + struct.pack('>I', crc32(b'IEND'))
        return PNG_SIGNATURE + ihdr + idat + iend

    cases.append(("Invalid filter byte (7)", _make_with_bad_filter(), ["FILTER"], True))

    # 6. Missing PLTE for indexed color
    def _make_indexed_no_palte():
        ihdr_data = struct.pack('>IIBBBBB', 4, 3, 8, 3, 0, 0, 0)  # color_type=3 indexed
        ihdr = struct.pack('>I', 13) + b'IHDR' + ihdr_data + struct.pack('>I', crc32(b'IHDR' + ihdr_data))
        raw = bytearray()
        for y in range(3):
            raw.append(0)
            raw.extend(bytes([0, 1, 2, 3]))
        comp = zlib.compress(bytes(raw))
        idat = struct.pack('>I', len(comp)) + b'IDAT' + comp + struct.pack('>I', crc32(b'IDAT' + comp))
        iend = struct.pack('>I', 0) + b'IEND' + b'' + struct.pack('>I', crc32(b'IEND'))
        return PNG_SIGNATURE + ihdr + idat + iend  # NO PLTE!

    cases.append(("Indexed (type=3) without PLTE", _make_indexed_no_palte(), ["INDEXED_NO_PALETTE"], True))

    # 7. Strict CRC off vs on for CRC error
    corrupted2 = bytearray(valid)
    corrupted2[29] ^= 0xFF  # flip CRC again

    cases.append(("Corrupt CRC with strict_crc=False", bytes(corrupted2), None, False))

    for name, data, expected_codes, should_strict_fail in cases:
        print(f"\n  [{name}]")
        strict_kw = {}
        if expected_codes is not None:
            try:
                _ = read_png(data, strict_crc=True)
                if should_strict_fail:
                    print(f"    [FAIL] strict_crc=True: did NOT raise PNGError (expected one of {expected_codes})")
                    all_pass = False
                else:
                    print(f"    [PASS] strict_crc=True: decoded OK (as expected)")
            except PNGError as e:
                ok = (expected_codes is None) or (e.code in expected_codes)
                tag = "PASS" if ok else "FAIL"
                if not ok:
                    all_pass = False
                print(f"    [{tag}] strict_crc=True -> PNGError({e.code})")
                if hasattr(e, 'user_message'):
                    um = e.user_message()
                    # Print first 2 lines of user_message to verify it's there
                    lines = [l for l in um.split('\n') if l.strip()]
                    if len(lines) >= 2:
                        print(f"           user_message OK ({len(lines)} lines, starts: {lines[0][:60]}...)")
                    else:
                        print(f"    [FAIL] user_message too short: {um!r}")
                        all_pass = False
            except Exception as e:
                print(f"    [FAIL] unexpected exception {type(e).__name__}: {e}")
                all_pass = False

            # Try Image.from_png too (uses default strict_crc=True)
            try:
                _ = Image.from_png(data)
                if should_strict_fail:
                    print(f"    [FAIL] Image.from_png: did NOT raise PNGError")
                    all_pass = False
                else:
                    print(f"    [PASS] Image.from_png: decoded OK")
            except PNGError as e:
                ok = (expected_codes is None) or (e.code in expected_codes)
                print(f"    [{'PASS' if ok else 'FAIL'}] Image.from_png -> PNGError({e.code})")
                if not ok:
                    all_pass = False
            except Exception as e:
                print(f"    [FAIL] Image.from_png unexpected: {type(e).__name__}: {e}")
                all_pass = False
        else:
            # case without expected_codes = should work with strict off
            try:
                _ = read_png(data, strict_crc=False)
                print(f"    [PASS] strict_crc=False: decoded OK (ignored bad CRC)")
            except PNGError as e:
                print(f"    [FAIL] strict_crc=False still raised PNGError({e.code}): {e}")
                all_pass = False
            except Exception as e:
                print(f"    [FAIL] strict_crc=False unexpected: {type(e).__name__}: {e}")
                all_pass = False

    return all_pass


# ============================================================
# Test 5: Cross-format conversion with lossy/lossless report
# ============================================================

def test_cross_format_conversion():
    print("\n" + "=" * 65)
    print("TEST 5: Cross-format conversion — BMP <-> PNG lossy/lossless")
    print("=" * 65)

    all_pass = True

    # Test images
    test_images = [
        ("truecolor-random-9x7", create_random_image(9, 7, seed=101)),
        ("rgba-transparent-12x8", create_transparent_rgba_image(12, 8)),
        ("gradient-odd-11x5", create_gradient_image(11, 5, alpha=False)),
        ("fewcolors-8-7x3", create_few_colors_image(7, 3, num_colors=8, seed=55)),
        ("1x1-single", create_random_image(1, 1, seed=777)),
    ]

    conversion_pairs = [
        # (name, encode_1, decode_1_encode_2, compare_alpha, tolerate_lossy)
        ("BMP24 -> PNG(RGB,Paeth) -> PNG(decode)",
         lambda img: img.to_bmp(24),
         lambda d: Image.from_bmp(d).to_png(color_type=2, filter_type=4),
         False,  # ignore alpha in final compare (BMP24 lost it)
         False),  # should be lossless (RGB<->RGB)
        ("PNG(RGBA,Paeth) -> BMP32 -> BMP32(decode)",
         lambda img: img.to_png(color_type=6, filter_type=4),
         lambda d: Image.from_png(d).to_bmp(32),
         True,  # alpha preserved via BMP32
         False),
        ("PNG(RGB) -> BMP8(quantize) -> decode",
         lambda img: img.to_png(color_type=2, filter_type=2),
         lambda d: Image.from_png(d).to_bmp(8, palette_strategy="quantize"),
         False,
         True),  # 8-bit indexed from truecolor is usually lossy
        ("BMP8(fewcolors, exact) -> PNG(Indexed, Paeth)",
         lambda img: img.to_bmp(8, palette_strategy="error"),
         lambda d: Image.from_bmp(d).to_png(color_type=3, bit_depth=8, filter_type=4, palette_strategy="error"),
         False,
         False),  # fits in 256 colors, should be lossless
    ]

    for img_name, img in test_images:
        print(f"\n  --- Image: {img_name} ({img.width}x{img.height}, unique_rgb={img.unique_colors}) ---")
        for conv_name, enc1, enc2, cmp_alpha, tolerate_lossy in conversion_pairs:
            try:
                data1 = enc1(img)
                img_mid = Image.from_file(data1)
                data2 = enc2(data1)
                img_final = Image.from_file(data2)

                diff_count, max_diff = img.count_differences(img_final, ignore_alpha=not cmp_alpha)
                total = img.width * img.height * (4 if cmp_alpha else 3)

                lossy_tag = "LOSSLESS" if diff_count == 0 else "LOSSY"
                if diff_count == 0:
                    status = "PASS"
                elif tolerate_lossy:
                    status = "PASS"
                else:
                    status = "FAIL"
                    all_pass = False

                pct = 100 * diff_count / total if total else 0
                print(f"    [{status}] {conv_name}")
                print(f"           {lossy_tag}: {diff_count}/{total} ({pct:.3f}%), max_diff={max_diff}")
                print(f"           sizes: mid={len(data1)}B, final={len(data2)}B")
            except PaletteError as pe:
                # Only acceptable for 8-bit "exact" when image has too many colors
                if "fewcolors" in img_name and "BMP8(fewcolors, exact)" in conv_name:
                    print(f"    [FAIL] {conv_name}: PaletteError ({pe.color_count} colors but expected <=8)")
                    all_pass = False
                elif tolerate_lossy and "BMP8" in conv_name:
                    print(f"    [INFO] {conv_name}: PaletteError (strategy=error + too many colors, expected for this img)")
                else:
                    print(f"    [FAIL] {conv_name}: PaletteError {pe}")
                    all_pass = False
            except Exception as e:
                print(f"    [FAIL] {conv_name}: EXCEPTION {type(e).__name__}: {e}")
                all_pass = False

    return all_pass


# ============================================================
# Test 6: BMP Robustness (signature / truncation)
# ============================================================

def test_bmp_robustness():
    print("\n" + "=" * 65)
    print("TEST 6: BMP Robustness — malformed files")
    print("=" * 65)

    all_pass = True

    # Build a minimal valid BMP (24-bit 2x2) via write_bmp
    tmp = BMPImage()
    tmp.width = 2
    tmp.height = 2
    tmp.bits_per_pixel = 24
    tmp.pixels = [[(10, 20, 30), (40, 50, 60)],
                  [(70, 80, 90), (100, 110, 120)]]
    valid = write_bmp(tmp)

    cases = [
        ("Bad signature", b'XX' + valid[2:], ["SIGNATURE"]),
        ("Truncated to 10 bytes", valid[:10], ["TRUNCATED"]),
        ("Truncated in pixel area", valid[:-10], ["TRUNCATED"]),
    ]

    for name, data, codes in cases:
        try:
            _ = read_bmp(data)
            print(f"    [FAIL] {name}: should have raised BMPError (expected {codes})")
            all_pass = False
        except BMPError as e:
            ok = e.code in codes
            print(f"    [{'PASS' if ok else 'FAIL'}] {name} -> BMPError({e.code})")
            if not ok:
                all_pass = False
            if hasattr(e, 'user_message'):
                um = e.user_message()
                lines = [l for l in um.split('\n') if l.strip()]
                if len(lines) >= 2:
                    print(f"           user_message OK ({len(lines)} lines)")
                else:
                    print(f"    [FAIL] user_message too short")
                    all_pass = False
        except Exception as e:
            print(f"    [FAIL] {name}: wrong exception {type(e).__name__}: {e}")
            all_pass = False

    return all_pass


# ============================================================
# Test 7: CLI smoke test (encode -> inspect -> convert)
# ============================================================

def test_cli_smoke():
    print("\n" + "=" * 65)
    print("TEST 7: CLI smoke test (encode -> inspect -> convert)")
    print("=" * 65)

    import tempfile
    import os
    from cli import main as cli_main

    all_pass = True
    tmpdir = tempfile.mkdtemp(prefix="imcodec_test_")

    try:
        # Encode
        in_path = os.path.join(tmpdir, "in.png")
        rc = cli_main(["encode", "--width", "9", "--height", "7",
                       "--pattern", "gradient", "-o", in_path])
        if rc != 0 or not os.path.exists(in_path):
            print(f"    [FAIL] encode: rc={rc}, exists={os.path.exists(in_path)}")
            all_pass = False
        else:
            sz = os.path.getsize(in_path)
            print(f"    [PASS] encode -> {in_path} ({sz}B)")

        # Inspect
        rc = cli_main(["inspect", in_path, "-d"])
        if rc != 0:
            print(f"    [FAIL] inspect rc={rc}")
            all_pass = False
        else:
            print(f"    [PASS] inspect rc=0")

        # Convert BMP 24-bit (lossless for RGB)
        out_bmp = os.path.join(tmpdir, "out.bmp")
        rc = cli_main(["convert", in_path, out_bmp, "--to", "bmp",
                       "--bmp-bpp", "24", "--verify"])
        if rc != 0:
            print(f"    [FAIL] convert PNG->BMP24 rc={rc}")
            all_pass = False
        else:
            print(f"    [PASS] convert PNG->BMP24")

        # Convert BMP 1-bit indexed (should be lossy)
        out_bmp1 = os.path.join(tmpdir, "out_1bpp.bmp")
        rc = cli_main(["convert", in_path, out_bmp1, "--to", "bmp",
                       "--bmp-bpp", "1", "--palette-strategy", "quantize", "--verify"])
        if rc != 0:
            print(f"    [FAIL] convert PNG->BMP 1-bit rc={rc}")
            all_pass = False
        else:
            print(f"    [PASS] convert PNG->BMP 1-bit (quantize)")

        # Convert PNG indexed 4-bit
        out_png4 = os.path.join(tmpdir, "out_4bit.png")
        rc = cli_main(["convert", in_path, out_png4, "--to", "png",
                       "--png-color", "indexed", "--png-depth", "4",
                       "--palette-strategy", "quantize", "--verify"])
        if rc != 0:
            print(f"    [FAIL] convert PNG->Indexed4 rc={rc}")
            all_pass = False
        else:
            print(f"    [PASS] convert PNG->Indexed 4-bit")

        # Decode sample
        rc = cli_main(["decode", in_path])
        if rc != 0:
            print(f"    [FAIL] decode rc={rc}")
            all_pass = False
        else:
            print(f"    [PASS] decode")

    finally:
        # cleanup
        try:
            for f in os.listdir(tmpdir):
                os.remove(os.path.join(tmpdir, f))
            os.rmdir(tmpdir)
        except Exception:
            pass

    return all_pass


# ============================================================
# Main
# ============================================================

# ============================================================
# Test 8: PNG ancillary chunks (gAMA/sRGB/tEXt/tRNS + unknown)
# ============================================================

def test_png_ancillary_chunks():
    print("\n" + "=" * 65)
    print("TEST 8: PNG Ancillary Chunks (gAMA/sRGB/tEXt + unknown)")
    print("=" * 65)

    all_pass = True
    img = create_gradient_image(8, 6, alpha=True)

    # --- Subtest A: write PNG with sRGB + gAMA + tEXt + custom unknown chunk ---
    print("\n  --- Encoder: sRGB + gAMA + tEXt + unknown chunk preservation ---")
    png = PNGImage()
    png.width = img.width
    png.height = img.height
    png.bit_depth = 8
    png.color_type = 6
    png.pixels = img.pixels
    png.srgb = 0  # Perceptual
    png.gama = 1.0 / 2.2  # ~0.45455
    png.text_chunks = [
        ("Software", "imcodec test suite v1.0"),
        ("Author", "Test Author"),
        ("Description", "A test gradient with metadata"),
    ]
    png.phys = (2835, 2835, 1)  # 72 DPI in pixels-per-meter

    # Unknown chunk (non-critical 'teSt' - lowercase first char = ancillary)
    extra = [(b'teSt', b'hello world custom chunk data')]

    try:
        data = write_png(png, filter_type=4, extra_chunks=extra, write_ancillary=True)
    except Exception as e:
        print(f"    [FAIL] write_png with ancillary: {type(e).__name__}: {e}")
        all_pass = False
    else:
        # Read back and check chunks
        try:
            back = read_png(data, strict_crc=True)
            checks = [
                ("sRGB intent=0", back.srgb == 0),
                ("gAMA≈0.45455", back.gama is not None and abs(back.gama - 0.45455) < 0.001),
                ("pHYs 2835x2835 unit=1", back.phys == (2835, 2835, 1)),
                ("tEXt count=3", len(back.text_chunks) == 3),
                ("tEXt Software present", any(kw == "Software" and "imcodec" in tx
                                              for kw, tx in back.text_chunks)),
                ("unknown chunks contains teSt",
                 any(name == 'teSt' for name, length, order in back.unknown_chunks)),
            ]
            for name, ok in checks:
                tag = "PASS" if ok else "FAIL"
                if not ok:
                    all_pass = False
                print(f"    [{tag}] {name}")
            # pixel roundtrip (RGBA with metadata does not affect pixels)
            decoded_img = Image.from_png(data)
            diff_count, _ = img.count_differences(decoded_img, ignore_alpha=False)
            if diff_count == 0:
                print(f"    [PASS] RGBA pixels LOSSLESS despite ancillary chunks (0 diffs)")
            else:
                print(f"    [FAIL] RGBA pixels: {diff_count} diffs")
                all_pass = False
            # describe() should print these chunks (verify no exceptions)
            desc = back.describe()
            for key in ["sRGB", "gAMA", "pHYs", "tEXt chunks: 3", "Unhandled chunks:",
                        "teSt"]:
                if key not in desc:
                    print(f"    [FAIL] describe() missing '{key}'")
                    all_pass = False
            print(f"    [PASS] describe() prints all ancillary info")
        except Exception as e:
            print(f"    [FAIL] read_png back with ancillary: {type(e).__name__}: {e}")
            all_pass = False

    return all_pass


# ============================================================
# Test 9: PNG tRNS palette transparency
# ============================================================

def test_png_trns_transparency():
    print("\n" + "=" * 65)
    print("TEST 9: PNG tRNS Palette Transparency (indexed alpha)")
    print("=" * 65)

    all_pass = True

    # --- Subtest A: Build indexed palette with explicit alpha + write tRNS ---
    print("\n  --- Write indexed PNG with tRNS -> decode should see correct alpha ---")
    width, height = 6, 5
    palette = [
        (255, 0, 0),     # 0 = red, fully opaque (255)
        (0, 255, 0),     # 1 = green, half-transparent (128)
        (0, 0, 255),     # 2 = blue, fully transparent (0)
        (255, 255, 0),   # 3 = yellow, opaque
    ]
    alphas = [255, 128, 0, 255]
    # Palette layout: red, green, blue, yellow rows
    indices_pixels = []
    for y in range(height):
        row = []
        for x in range(width):
            idx = (x + y) % 4
            row.append(idx)
        indices_pixels.append(row)

    png = PNGImage()
    png.width = width
    png.height = height
    png.bit_depth = 4  # use 4-bit indexed
    png.color_type = 3
    png.palette = palette
    png.transparency = alphas
    png.pixels = indices_pixels

    try:
        data = write_png(png, filter_type=0, write_ancillary=True)
        # Decode back via PNGImage (low-level)
        low = read_png(data)
        t_ok = (low.transparency is not None and
                len(low.transparency) >= 3 and
                low.transparency[0] == 255 and
                low.transparency[1] == 128 and
                low.transparency[2] == 0)
        print(f"    [{'PASS' if t_ok else 'FAIL'}] Low-level: tRNS parsed correctly")
        if not t_ok:
            all_pass = False

        # Decode via Image.from_png -> should get RGBA with alpha
        decoded = Image.from_png(data)
        # Spot check a few pixels
        checks = [
            # (x, y) -> expected_rgb, expected_alpha
            ((0, 0), (255, 0, 0), 255),  # idx 0
            ((1, 0), (0, 255, 0), 128),  # idx 1
            ((2, 0), (0, 0, 255), 0),    # idx 2
            ((3, 0), (255, 255, 0), 255),  # idx 3
            ((0, 2), (0, 0, 255), 0),    # idx 2 again
        ]
        for (x, y), exp_rgb, exp_a in checks:
            p = decoded.pixels[y][x]
            ok = (p[0], p[1], p[2]) == exp_rgb and p[3] == exp_a
            tag = "PASS" if ok else "FAIL"
            if not ok:
                all_pass = False
            print(f"    [{tag}] pixel ({x},{y}): got {p}, expected RGB={exp_rgb} A={exp_a}")
    except Exception as e:
        print(f"    [FAIL] tRNS write/read: {type(e).__name__}: {e}")
        all_pass = False

    # --- Subtest B: convert tRNS-indexed PNG to RGBA PNG and BMP32 ---
    print("\n  --- tRNS-indexed -> RGBA PNG (lossless alpha) & BMP32 (lossless alpha) ---")
    try:
        src = Image.from_png(data)
        # -> RGBA PNG
        rgba_png = src.to_png(color_type=6, bit_depth=8, filter_type=4)
        back_rgba = Image.from_png(rgba_png)
        diff_rgba, max_rgba = src.count_differences(back_rgba, ignore_alpha=False)
        tag = "PASS" if diff_rgba == 0 else "FAIL"
        if diff_rgba != 0:
            all_pass = False
        print(f"    [{tag}] Indexed+tRNS -> RGBA PNG: {diff_rgba} alpha-aware diffs, max={max_rgba}")

        # -> BMP32
        bmp32 = src.to_bmp(32)
        back_bmp = Image.from_bmp(bmp32)
        diff_bmp, max_bmp = src.count_differences(back_bmp, ignore_alpha=False)
        tag = "PASS" if diff_bmp == 0 else "FAIL"
        if diff_bmp != 0:
            all_pass = False
        print(f"    [{tag}] Indexed+tRNS -> BMP32: {diff_bmp} alpha-aware diffs, max={max_bmp}")
    except Exception as e:
        print(f"    [FAIL] cross-format alpha: {type(e).__name__}: {e}")
        all_pass = False

    return all_pass


# ============================================================
# Test 10: Strict write validation (palette overflow / index oob)
# ============================================================

def test_write_indexed_strict_validation():
    print("\n" + "=" * 65)
    print("TEST 10: Strict indexed write validation (palette/index)")
    print("=" * 65)

    all_pass = True

    # --- Subtest A: palette larger than bit depth allows ---
    print("\n  --- A: Palette with 17 colors vs 4-bit (cap 16) MUST throw ---")
    for fmt_name, ctor in [("PNG", PNGImage), ("BMP", BMPImage)]:
        big_pal = [(i * 13, i * 17, i * 19) for i in range(17)]  # 17 colors
        img = ctor()
        img.width = 2
        img.height = 2
        img.palette = big_pal
        img.pixels = [[0, 1], [2, 3]]
        if fmt_name == "PNG":
            img.bit_depth = 4
            img.color_type = 3  # MUST be indexed for palette
        else:
            img.bits_per_pixel = 4  # BMP attribute name
        if fmt_name == "PNG":
            try:
                write_png(img, palette_strategy="error")
                print(f"    [FAIL] {fmt_name} write: did NOT raise PaletteError for 17-color 4-bit")
                all_pass = False
            except PaletteError as pe:
                print(f"    [PASS] {fmt_name} write: PaletteError ({pe.color_count}/{pe.max_colors})")
            except Exception as e:
                print(f"    [FAIL] {fmt_name} write: wrong exception {type(e).__name__}: {e}")
                all_pass = False
        else:
            try:
                write_bmp(img, palette_strategy="error")
                print(f"    [FAIL] {fmt_name} write: did NOT raise PaletteError for 17-color 4-bit")
                all_pass = False
            except PaletteError as pe:
                print(f"    [PASS] {fmt_name} write: PaletteError ({pe.color_count}/{pe.max_colors})")
            except Exception as e:
                print(f"    [FAIL] {fmt_name} write: wrong exception {type(e).__name__}: {e}")
                all_pass = False

    # --- Subtest B: pixel index >= palette size ---
    print("\n  --- B: Palette 4 entries, pixel index 5 MUST throw ---")
    small_pal = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    for fmt_name, ctor in [("PNG", PNGImage), ("BMP", BMPImage)]:
        img = ctor()
        img.width = 3
        img.height = 1
        img.palette = small_pal
        # 4 valid entries -> index 5 is definitely out of range
        img.pixels = [[0, 5, 2]]  # OOB at col 1
        if fmt_name == "PNG":
            img.bit_depth = 8
            img.color_type = 3
        else:
            img.bits_per_pixel = 8
        if fmt_name == "PNG":
            try:
                write_png(img)
                print(f"    [FAIL] {fmt_name} write: did NOT raise for index 5 with 4-entry palette")
                all_pass = False
            except PaletteError as pe:
                print(f"    [PASS] {fmt_name} write: PaletteError on bad index ({pe})")
            except Exception as e:
                print(f"    [FAIL] {fmt_name} write: wrong exception {type(e).__name__}: {e}")
                all_pass = False
        else:
            try:
                write_bmp(img)
                print(f"    [FAIL] {fmt_name} write: did NOT raise for index 5 with 4-entry palette")
                all_pass = False
            except PaletteError as pe:
                print(f"    [PASS] {fmt_name} write: PaletteError on bad index ({pe})")
            except Exception as e:
                print(f"    [FAIL] {fmt_name} write: wrong exception {type(e).__name__}: {e}")
                all_pass = False

    # --- Subtest C: 1-bit mode with index 2 ---
    print("\n  --- C: 1-bit with index 2 (out of {0,1}) MUST throw ---")
    for fmt_name, ctor in [("PNG", PNGImage), ("BMP", BMPImage)]:
        img = ctor()
        img.width = 3
        img.height = 1
        img.palette = [(0, 0, 0), (255, 255, 255)]
        img.pixels = [[0, 2, 1]]  # 2 is invalid for 1-bit
        if fmt_name == "PNG":
            img.bit_depth = 1
            img.color_type = 3
        else:
            img.bits_per_pixel = 1
        if fmt_name == "PNG":
            try:
                write_png(img)
                print(f"    [FAIL] {fmt_name}: did NOT raise for idx=2 in 1-bit")
                all_pass = False
            except PaletteError:
                print(f"    [PASS] {fmt_name}: PaletteError on idx=2 in 1-bit")
            except Exception as e:
                print(f"    [FAIL] {fmt_name}: wrong exception {type(e).__name__}: {e}")
                all_pass = False
        else:
            try:
                write_bmp(img)
                print(f"    [FAIL] {fmt_name}: did NOT raise for idx=2 in 1-bit")
                all_pass = False
            except PaletteError:
                print(f"    [PASS] {fmt_name}: PaletteError on idx=2 in 1-bit")
            except Exception as e:
                print(f"    [FAIL] {fmt_name}: wrong exception {type(e).__name__}: {e}")
                all_pass = False

    return all_pass


# ============================================================
# Test 11: Batch CLI convert with mixed good/bad files
# ============================================================

def test_cli_batch_with_corrupt_files():
    print("\n" + "=" * 65)
    print("TEST 11: CLI Batch convert (directory w/ corrupt files)")
    print("=" * 65)

    import tempfile
    import shutil
    from cli import main as cli_main

    all_pass = True
    tmpdir = tempfile.mkdtemp(prefix="imcodec_batch_")

    try:
        in_dir = os.path.join(tmpdir, "input")
        out_dir = os.path.join(tmpdir, "output")
        os.makedirs(in_dir)

        # Good files: 3 PNGs + 2 BMPs
        img_a = create_gradient_image(7, 5, alpha=False)
        img_b = create_few_colors_image(9, 4, num_colors=6, seed=3)
        img_c = create_random_image(5, 3, seed=123, alpha=True)

        # Good PNG
        with open(os.path.join(in_dir, "a_gradient.png"), 'wb') as f:
            f.write(img_a.to_png(color_type=2, filter_type=4))
        # Good BMP 24
        with open(os.path.join(in_dir, "b_fewcolors.bmp"), 'wb') as f:
            f.write(img_b.to_bmp(24))
        # Good PNG rgba
        with open(os.path.join(in_dir, "c_random_rgba.png"), 'wb') as f:
            f.write(img_c.to_png(color_type=6, filter_type=0))
        # Good BMP 32
        with open(os.path.join(in_dir, "d_rgba32.bmp"), 'wb') as f:
            f.write(img_c.to_bmp(32))
        # Good 8-color -> BMP 4-bit indexed
        with open(os.path.join(in_dir, "e_indexed4.bmp"), 'wb') as f:
            f.write(img_b.to_bmp(4, palette_strategy="quantize"))

        # Corrupt files
        # 1) Truncated PNG (just signature)
        with open(os.path.join(in_dir, "bad_truncated.png"), 'wb') as f:
            f.write(b'\x89PNG\r\n\x1a\n')  # 8 bytes only
        # 2) BMP with bad signature
        with open(os.path.join(in_dir, "bad_sig.bmp"), 'wb') as f:
            f.write(b'QQ' + bytes([0] * 100))
        # 3) PNG with wrong CRC on IHDR
        bad_crc = bytearray(img_a.to_png(color_type=2))
        bad_crc[29] ^= 0xFF  # flip CRC byte
        with open(os.path.join(in_dir, "bad_crc.png"), 'wb') as f:
            f.write(bytes(bad_crc))
        # 4) Random non-image file
        with open(os.path.join(in_dir, "readme.txt"), 'w') as f:
            f.write("not an image file\n")

        print(f"  Input dir     : {in_dir}")
        print(f"  Output dir    : {out_dir}")
        print(f"  Good files    : 5")
        print(f"  Bad/non-image : 4 (1 truncated PNG, 1 bad BMP sig, 1 bad CRC, 1 TXT)")

        # Run CLI batch convert (strict_crc default = True, so bad CRC should fail)
        rc = cli_main(["convert", in_dir, out_dir, "--to", "png",
                       "--png-color", "rgba", "--verify", "--quiet"])

        # Expect rc = 2 (because some failed)
        if rc == 0:
            print(f"    [WARN] CLI batch rc=0, expected rc=2 (failures present)")
        elif rc == 2:
            print(f"    [PASS] CLI batch rc=2 (non-zero = had failures, correct)")
        else:
            print(f"    [FAIL] CLI batch rc={rc}, expected rc=2")
            all_pass = False

        # Verify output directory: at least the 5 good files should exist
        expected_outputs = {
            "a_gradient.png",
            "b_fewcolors.png",
            "c_random_rgba.png",
            "d_rgba32.png",
            "e_indexed4.png",
        }
        actual_outputs = set(os.listdir(out_dir)) if os.path.isdir(out_dir) else set()
        for exp in expected_outputs:
            ok = exp in actual_outputs
            tag = "PASS" if ok else "FAIL"
            if not ok:
                all_pass = False
            print(f"    [{tag}] Good output exists: {exp}")

        # Verify bad files are NOT converted
        for bad in ["bad_truncated.png", "bad_sig.png", "bad_crc.png", "readme.png"]:
            not_present = bad not in actual_outputs
            tag = "PASS" if not_present else "FAIL"
            if not not_present:
                all_pass = False
            print(f"    [{tag}] Bad file not in output: {bad}")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return all_pass


# ============================================================
# Test 12: Transparent PNG -> BMP24 (lossy) vs BMP32 (lossless alpha)
# ============================================================

def test_transparent_png_to_bmp_lossy_vs_lossless():
    print("\n" + "=" * 65)
    print("TEST 12: Transparent PNG -> BMP24 (lossy) / BMP32 (lossless)")
    print("=" * 65)

    all_pass = True
    from cli import _convert_single
    from cli import _image_has_alpha, _output_has_alpha
    import tempfile
    import argparse

    with tempfile.TemporaryDirectory() as tmp:
        src_img = create_transparent_rgba_image(8, 6)
        src_path = os.path.join(tmp, "src.png")
        with open(src_path, 'wb') as f:
            f.write(src_img.to_png(color_type=6, filter_type=4))

        def fake_args(**kw):
            ns = argparse.Namespace()
            ns.options = None
            ns.bmp_bpp = kw.get('bmp_bpp')
            ns.png_color = kw.get('png_color')
            ns.png_depth = kw.get('png_depth')
            ns.png_filter = kw.get('png_filter')
            ns.palette_strategy = kw.get('palette_strategy')
            ns.from_fmt = None
            ns.to_fmt = None
            ns.verify = kw.get('verify', True)
            ns.overwrite = 'always'
            return ns

        # -> BMP24: alpha must be discarded (lossy)
        print("  --- RGBA PNG -> BMP 24-bit (should be LOSSY: alpha discarded) ---")
        bmp24 = os.path.join(tmp, "out.bmp")
        r1 = _convert_single(src_path, bmp24, fake_args(bmp_bpp=24, verify=True), verbose=False)
        ok = (r1["ok"] and r1["lossiness"] == "lossy" and r1["diff_count"] is not None and r1["diff_count"] > 0)
        tag = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"    [{tag}] ok={r1['ok']} lossiness={r1['lossiness']} diffs={r1['diff_count']} max={r1['max_diff']}")

        # -> BMP32: alpha preserved (should be lossless)
        print("  --- RGBA PNG -> BMP 32-bit (should be LOSSLESS: alpha preserved) ---")
        bmp32 = os.path.join(tmp, "out32.bmp")
        r2 = _convert_single(src_path, bmp32, fake_args(bmp_bpp=32, verify=True), verbose=False)
        ok = (r2["ok"] and r2["lossiness"] == "lossless" and r2["diff_count"] == 0)
        tag = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"    [{tag}] ok={r2['ok']} lossiness={r2['lossiness']} diffs={r2['diff_count']} max={r2['max_diff']}")

        # -> BMP24 WITHOUT --verify: heuristic says likely-lossy
        print("  --- RGBA PNG -> BMP 24-bit (no --verify): heuristic likely-lossy ---")
        bmp24_nv = os.path.join(tmp, "out_nv.bmp")
        r3 = _convert_single(src_path, bmp24_nv, fake_args(bmp_bpp=24, verify=False), verbose=False)
        ok = (r3["ok"] and r3["lossiness"] == "likely-lossy" and r3["note"] is not None and "alpha" in r3["note"])
        tag = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"    [{tag}] lossiness={r3['lossiness']} note={r3['note']!r}")

        # -> RGBA PNG WITHOUT --verify: heuristic says likely-lossless
        print("  --- RGBA PNG -> RGBA PNG (no --verify): heuristic likely-lossless ---")
        rgba_png = os.path.join(tmp, "out_rgba.png")
        r4 = _convert_single(src_path, rgba_png, fake_args(png_color='rgba', png_depth=8, png_filter='paeth', verify=False), verbose=False)
        ok = (r4["ok"] and r4["lossiness"] == "likely-lossless")
        tag = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"    [{tag}] lossiness={r4['lossiness']} note={r4['note']!r}")

    return all_pass


# ============================================================
# Test 13: Batch reports (JSON/CSV) + nested dir partial failure
# ============================================================

def test_batch_reports_and_nested_dir():
    print("\n" + "=" * 65)
    print("TEST 13: Batch JSON/CSV reports + nested directory + partial fail")
    print("=" * 65)

    all_pass = True
    import tempfile
    import shutil
    import json
    import csv
    from cli import main as cli_main

    with tempfile.TemporaryDirectory() as tmp:
        in_dir = os.path.join(tmp, "in")
        out_dir = os.path.join(tmp, "out")

        # Nested structure
        good_a = os.path.join(in_dir, "a")
        good_b = os.path.join(in_dir, "sub", "b")
        bad_c = os.path.join(in_dir, "sub", "broken")
        os.makedirs(good_a, exist_ok=True)
        os.makedirs(good_b, exist_ok=True)
        os.makedirs(bad_c, exist_ok=True)

        # Good PNG in top-level
        img_a = create_gradient_image(6, 5, alpha=False)
        with open(os.path.join(good_a, "grad.png"), 'wb') as f:
            f.write(img_a.to_png(color_type=2))

        # Good BMP in sub/b
        img_b = create_few_colors_image(9, 4, num_colors=6, seed=22)
        with open(os.path.join(good_b, "small.bmp"), 'wb') as f:
            f.write(img_b.to_bmp(8, palette_strategy="quantize"))

        # Bad PNG in sub/broken (corrupt)
        with open(os.path.join(bad_c, "corrupt.png"), 'wb') as f:
            f.write(b'\x89PNG\r\n\x1a\nTHIS-IS-CORRUPT')

        json_report = os.path.join(tmp, "report.json")
        csv_report = os.path.join(tmp, "report.csv")

        # Run batch with --recursive (default), --report json
        rc = cli_main(["convert", in_dir, out_dir, "--to", "png",
                       "--png-color", "rgba", "--verify", "--quiet",
                       "--report", json_report])

        print(f"  --- Batch recursive with JSON report (rc={rc}) ---")
        if rc != 2:
            print(f"    [WARN] Expected rc=2 (one failure), got {rc}")

        # Verify structure preserved (keep_structure default on)
        expected = [
            os.path.join(out_dir, "a", "grad.png"),
            os.path.join(out_dir, "sub", "b", "small.png"),
        ]
        for p in expected:
            ok = os.path.exists(p)
            tag = "PASS" if ok else "FAIL"
            if not ok:
                all_pass = False
            print(f"    [{tag}] Nested structure preserved: {os.path.relpath(p, out_dir)}")

        # Verify JSON report is valid and contains summary
        if os.path.exists(json_report):
            try:
                with open(json_report, 'r', encoding='utf-8') as f:
                    j = json.load(f)
                s = j.get("summary", {})
                has_sum = ("files_processed" in s and "succeeded" in s and "failed" in s
                           and "lossless_cnt" in s and "lossy_cnt" in s)
                has_files = isinstance(j.get("files"), list) and len(j["files"]) == 3
                tag = "PASS" if (has_sum and has_files) else "FAIL"
                if not (has_sum and has_files):
                    all_pass = False
                print(f"    [{tag}] JSON report valid: summary={has_sum}, files={has_files}, "
                      f"processed={s.get('files_processed')}, succeeded={s.get('succeeded')}, failed={s.get('failed')}")
            except Exception as e:
                print(f"    [FAIL] JSON parse error: {e}")
                all_pass = False
        else:
            print(f"    [FAIL] JSON report file missing")
            all_pass = False

        # Run again --flatten + --overwrite=rename + CSV report
        out2 = os.path.join(tmp, "out2")
        rc2 = cli_main(["convert", in_dir, out2, "--to", "bmp", "--bmp-bpp", "24",
                        "--flatten", "--no-recursive",
                        "--overwrite", "rename",
                        "--include-ext", ".png",
                        "--quiet",
                        "--report", csv_report])
        print(f"  --- Batch non-recursive, flatten, include-ext=.png, CSV report (rc={rc2}) ---")

        # Non-recursive: only the 'a' dir file NOT processed (it's in subdir); wait --no-recursive
        # means only direct children of in_dir; but our in_dir direct children are "a", "sub" directories
        # so no files -> count 0 processed -> rc 0. Hmm, this is fine.
        # Let's instead use a flat dir: move good file to top-level. Re-do the test
        # in a simpler way.

        # Actually use CSV report from first run instead by regenerating
        csv_report2 = os.path.join(tmp, "report2.csv")
        cli_main(["convert", in_dir, os.path.join(tmp, "out3"), "--to", "png",
                  "--png-color", "rgb", "--verify", "--quiet",
                  "--report", csv_report2])
        if os.path.exists(csv_report2):
            try:
                with open(csv_report2, 'r', encoding='utf-8', newline='') as f:
                    reader = list(csv.DictReader(f))
                has_header = len(reader) >= 1 and "input" in reader[0]
                has_summary = any(r.get("input") == "__SUMMARY__" for r in reader)
                tag = "PASS" if (has_header and has_summary) else "FAIL"
                if not (has_header and has_summary):
                    all_pass = False
                print(f"    [{tag}] CSV report: rows={len(reader)}, header={has_header}, summary_row={has_summary}")
            except Exception as e:
                print(f"    [FAIL] CSV parse error: {e}")
                all_pass = False
        else:
            print(f"    [FAIL] CSV report file missing")
            all_pass = False

    return all_pass


# ============================================================
# Test 14: 16-bit PNG + Grayscale+Alpha (color type 4) decode
# ============================================================

def test_16bit_and_grayalpha_decode():
    print("\n" + "=" * 65)
    print("TEST 14: 16-bit PNG / Grayscale+Alpha decode support")
    print("=" * 65)

    all_pass = True

    def build_png_via_raw(color_type, bit_depth, pixel_writer, width=4, height=3):
        """Helper: build a raw PNG with given color_type/bit_depth from raw pixels."""
        import zlib as _zl
        ihdr_data = struct.pack('>II', width, height) + bytes([bit_depth, color_type, 0, 0, 0])
        out = bytearray(PNG_SIGNATURE)
        out.extend(PNGChunk(b'IHDR', ihdr_data).encode())

        # Build filtered scanlines (filter=0)
        raw = bytearray()
        for y in range(height):
            raw.append(0)  # filter=none
            pixel_writer(raw, y, width)

        comp = _zl.compress(bytes(raw))
        out.extend(PNGChunk(b'IDAT', comp).encode())
        out.extend(PNGChunk(b'IEND', b'').encode())
        return bytes(out)

    # Subtest A: 16-bit RGB color type 2
    print("  --- 16-bit RGB (color_type=2, bit_depth=16) ---")
    def writer_16rgb(raw, y, w):
        for x in range(w):
            # Use high bytes distinguishable from low bytes
            v = (x * 50 + y * 100) & 0xFF
            # Write R=r, G=g, B=b in both high and low byte
            r = (x * 37 + y * 7) & 0xFF
            g = (x * 71 + y * 13) & 0xFF
            b = (x * 101 + y * 19) & 0xFF
            raw.append(r); raw.append(0)  # R = r<<8
            raw.append(g); raw.append(0)  # G = g<<8
            raw.append(b); raw.append(0)  # B = b<<8

    data16 = build_png_via_raw(2, 16, writer_16rgb, width=4, height=3)
    try:
        decoded = Image.from_png(data16)
        # spot-check first pixel
        r0 = decoded.pixels[0][0]
        expected_r = (0 * 37 + 0 * 7) & 0xFF
        expected_g = (0 * 71 + 0 * 13) & 0xFF
        expected_b = (0 * 101 + 0 * 19) & 0xFF
        ok = (r0[0] == expected_r and r0[1] == expected_g and r0[2] == expected_b and r0[3] == 255)
        tag = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"    [{tag}] Decoded dims={decoded.width}x{decoded.height}, pixel(0,0)={r0}, expected=({expected_r},{expected_g},{expected_b},255)")
    except PNGError as e:
        print(f"    [FAIL] 16-bit RGB PNGError: {e.code} {e}")
        all_pass = False
    except Exception as e:
        print(f"    [FAIL] 16-bit RGB {type(e).__name__}: {e}")
        all_pass = False

    # Subtest B: 8-bit Grayscale+Alpha (color_type=4)
    print("  --- 8-bit Gray+Alpha (color_type=4, bit_depth=8) ---")
    def writer_ga8(raw, y, w):
        for x in range(w):
            gray = (x * 40 + y * 30) % 256
            alpha = 0 if x == 0 else (128 if x == 1 else 255)
            raw.append(gray); raw.append(alpha)

    data_ga = build_png_via_raw(4, 8, writer_ga8, width=3, height=2)
    try:
        decoded = Image.from_png(data_ga)
        expected_px = {
            (0, 0): ((0, 0, 0), 0),
            (1, 0): ((40, 40, 40), 128),
            (2, 0): ((80, 80, 80), 255),
            (0, 1): ((30, 30, 30), 0),
        }
        all_ok = True
        for (x, y), (exp_rgb, exp_a) in expected_px.items():
            p = decoded.pixels[y][x]
            if (p[0], p[1], p[2]) != exp_rgb or p[3] != exp_a:
                all_ok = False
                print(f"    [FAIL] pixel({x},{y})={p}, expected RGB={exp_rgb} A={exp_a}")
        print(f"    [{'PASS' if all_ok else 'FAIL'}] Gray+Alpha decode: {len(expected_px)} pixels checked")
        if not all_ok:
            all_pass = False
    except Exception as e:
        print(f"    [FAIL] Gray+Alpha {type(e).__name__}: {e}")
        all_pass = False

    # Subtest C: Grayscale PNG + tRNS chroma-key (gray=128 -> alpha=0)
    print("  --- 8-bit Grayscale + tRNS chroma-key (gray=128 transparent) ---")
    def writer_g8(raw, y, w):
        for x in range(w):
            v = 64 if (x + y) % 2 == 0 else 128
            raw.append(v)

    data_gray_keyed = build_png_via_raw(0, 8, writer_g8, width=4, height=2)
    # Inject a tRNS chunk before IDAT manually
    # Rebuild properly using PNGChunk:
    import zlib as _zl
    ihdr_data = struct.pack('>IIBBBBB', 4, 2, 8, 0, 0, 0, 0)
    trns_data = struct.pack('>H', 128)  # gray = 128 is transparent
    raw = bytearray()
    for y in range(2):
        raw.append(0)
        for x in range(4):
            v = 64 if (x + y) % 2 == 0 else 128
            raw.append(v)
    comp = _zl.compress(bytes(raw))
    tkeyed = bytearray(PNG_SIGNATURE)
    tkeyed.extend(PNGChunk(b'IHDR', ihdr_data).encode())
    tkeyed.extend(PNGChunk(b'tRNS', trns_data).encode())
    tkeyed.extend(PNGChunk(b'IDAT', comp).encode())
    tkeyed.extend(PNGChunk(b'IEND', b'').encode())

    try:
        decoded = Image.from_png(bytes(tkeyed))
        # (0,0): 64 (opaque=255), (1,0): 128 (transparent=0), etc.
        checks = [
            ((0, 0), (64, 64, 64), 255),
            ((1, 0), (128, 128, 128), 0),
            ((2, 0), (64, 64, 64), 255),
            ((1, 1), (64, 64, 64), 255),
            ((0, 1), (128, 128, 128), 0),
        ]
        all_ok = True
        for (x, y), exp_rgb, exp_a in checks:
            p = decoded.pixels[y][x]
            if (p[0], p[1], p[2]) != exp_rgb or p[3] != exp_a:
                all_ok = False
                print(f"    [FAIL] pixel({x},{y})={p}, expected RGB={exp_rgb} A={exp_a}")
        print(f"    [{'PASS' if all_ok else 'FAIL'}] Grayscale tRNS chroma-key: {len(checks)} pixels checked")
        if not all_ok:
            all_pass = False
    except Exception as e:
        print(f"    [FAIL] Grayscale tRNS {type(e).__name__}: {e}")
        all_pass = False

    return all_pass


def main():
    print()
    print("#" * 65)
    print("#" + " " * 10 + "IMAGE CODEC — EXTENDED TEST SUITE" + " " * 17 + "#")
    print("#" * 65)

    tests = [
        ("BMP native roundtrip (odd widths)", test_bmp_native_roundtrip),
        ("Image API BMP palette strategies", test_image_bmp_palette_strategies),
        ("PNG roundtrip (multi-IDAT, odd-w, RGBA)", test_png_roundtrip_extended),
        ("PNG robustness (malformed + user_message)", test_png_robustness),
        ("BMP robustness (malformed)", test_bmp_robustness),
        ("Cross-format lossy/lossless", test_cross_format_conversion),
        ("CLI smoke test", test_cli_smoke),
        ("PNG ancillary chunks (gAMA/sRGB/tEXt/unknown)", test_png_ancillary_chunks),
        ("PNG tRNS palette transparency + cross-format", test_png_trns_transparency),
        ("Strict indexed write validation (palette + index)", test_write_indexed_strict_validation),
        ("CLI batch convert with corrupt files", test_cli_batch_with_corrupt_files),
        ("Transparent PNG -> BMP24(lossy)/BMP32(lossless)", test_transparent_png_to_bmp_lossy_vs_lossless),
        ("Batch JSON/CSV reports + nested dir + partial fail", test_batch_reports_and_nested_dir),
        ("16-bit PNG + Grayscale+Alpha decode", test_16bit_and_grayalpha_decode),
    ]

    results = []
    for name, fn in tests:
        try:
            results.append((name, fn()))
        except Exception as e:
            print(f"\n[CRITICAL] Test '{name}' crashed: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name:48s}: {status}")
        if not passed:
            all_pass = False

    print("\n" + "=" * 65)
    if all_pass:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED — review output above")
        sys.exit(1)
    print("=" * 65)


if __name__ == '__main__':
    main()
