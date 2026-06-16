import os
import sys
import argparse
from image import Image
from bmp_codec import BMPError, BMPImage, read_bmp, write_bmp, bmp_bytes_per_row
from png_codec import PNGError, PNGImage, read_png, write_png, COLOR_TYPE_NAMES, FILTER_NAMES
from palette import PaletteError


def detect_format(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.bmp', '.dib'):
        return 'bmp'
    elif ext in ('.png',):
        return 'png'
    return None


def banner():
    return "Image Codec Tool  (BMP/PNG read-write engine)"


def print_error(e, fmt=None):
    if hasattr(e, 'user_message'):
        print()
        print(e.user_message())
    elif isinstance(e, PaletteError):
        print()
        print("[Palette Error]")
        print(f"  Message: {e}")
        if e.color_count is not None and e.max_colors is not None:
            print(f"  Colors needed: {e.color_count}, capacity: {e.max_colors}")
        print()
        print("  Troubleshooting:")
        print("  - Use a higher bit depth (e.g. 8-bit = 256 colors, 24-bit = true color)")
        print("  - Use strategy='quantize' to auto-reduce colors via median-cut")
        print("  - For PNG: use --color-type 2 (RGB) or 6 (RGBA) instead of 3 (Indexed)")
    else:
        print(f"[Error] {type(e).__name__}: {e}")


def cmd_inspect(args):
    path = args.input
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return 1

    fmt = detect_format(path) or args.format
    if not fmt:
        with open(path, 'rb') as f:
            sig = f.read(16)
        if sig[:2] == b'BM':
            fmt = 'bmp'
        elif sig[:8] == b'\x89PNG\r\n\x1a\n':
            fmt = 'png'
        else:
            print(f"Cannot detect format for: {path}")
            return 1

    file_size = os.path.getsize(path)
    with open(path, 'rb') as f:
        data = f.read()

    print(f"\n=== Inspection: {path} ===")
    print(f"  File size : {file_size:,} bytes")
    print(f"  Format    : {fmt.upper()}")

    try:
        if fmt == 'bmp':
            bmp = read_bmp(data)
            print()
            print("--- BMP Details ---")
            print(bmp.describe())
        elif fmt == 'png':
            png = read_png(data, strict_crc=not args.ignore_crc)
            print()
            print("--- PNG Details ---")
            print(png.describe())

        if args.decode_and_check:
            img = Image.from_file(data, fmt=fmt)
            print()
            print("--- Unified Decode ---")
            print(img.describe())
    except (BMPError, PNGError, PaletteError) as e:
        print_error(e, fmt)
        return 1
    except Exception as e:
        print_error(e, fmt)
        return 1

    return 0


def _parse_extra_options(extra_str):
    opts = {}
    if not extra_str:
        return opts
    for part in extra_str.split(','):
        if '=' in part:
            k, v = part.split('=', 1)
            k = k.strip()
            v = v.strip()
            if v.lower() in ('true', 'yes', 'on'):
                opts[k] = True
            elif v.lower() in ('false', 'no', 'off'):
                opts[k] = False
            else:
                try:
                    opts[k] = int(v)
                except ValueError:
                    opts[k] = v
        else:
            opts[part.strip()] = True
    return opts


def cmd_convert(args):
    in_path = args.input
    out_path = args.output

    if not os.path.exists(in_path):
        print(f"Input file not found: {in_path}")
        return 1

    in_fmt = detect_format(in_path) or args.from_fmt
    out_fmt = detect_format(out_path) or args.to_fmt

    if not in_fmt:
        with open(in_path, 'rb') as f:
            sig = f.read(16)
        if sig[:2] == b'BM':
            in_fmt = 'bmp'
        elif sig[:8] == b'\x89PNG\r\n\x1a\n':
            in_fmt = 'png'
    if not out_fmt:
        print(f"Cannot detect output format for {out_path}, use --to")
        return 1

    in_size = os.path.getsize(in_path)
    with open(in_path, 'rb') as f:
        in_data = f.read()

    print(f"\n--- Converting: {in_path} ({in_fmt.upper()}) -> {out_path} ({out_fmt.upper()}) ---")
    print(f"  Input size : {in_size:,} bytes")

    try:
        img = Image.from_file(in_data, fmt=in_fmt)
        print()
        print("--- Input decoded ---")
        print(img.describe())

        extra = _parse_extra_options(args.options)

        if out_fmt == 'bmp':
            bpp = args.bmp_bpp or extra.get('bpp', 24)
            strategy = args.palette_strategy or extra.get('strategy', 'quantize')
            out_data = img.to_bmp(bits_per_pixel=bpp, palette_strategy=strategy)
            if bpp <= 8:
                qi = img.last_quantization_info
                if qi and qi.is_lossy:
                    print()
                    print("--- Palette quantization (lossy) ---")
                    print(f"  Original unique colors : {qi.original_colors}")
                    print(f"  Palette size           : {qi.palette_colors} ({bpp}-bit = {1 << bpp} slots)")
                    exact_pct = 100 * qi.exact_match_count / qi.total_pixels
                    print(f"  Exact matches          : {qi.exact_match_count:,}/{qi.total_pixels:,} ({exact_pct:.1f}%)")
                    print(f"  Max distance           : {qi.max_error}")
        else:
            ct_map = {'rgb': 2, 'rgba': 6, 'indexed': 3, 'palette': 3, 'gray': 0, 'greyscale': 0, 'grayscale': 0}
            ct_name = (args.png_color or extra.get('color_type', 'rgba')).lower()
            if ct_name.isdigit():
                color_type = int(ct_name)
            else:
                color_type = ct_map.get(ct_name, 6)
            bit_depth = args.png_depth or extra.get('bit_depth', 8)
            filter_name = (args.png_filter or extra.get('filter', 'paeth')).lower()
            f_map = {'none': 0, 'sub': 1, 'up': 2, 'avg': 3, 'average': 3, 'paeth': 4}
            filter_type = f_map.get(filter_name, 4)
            strategy = args.palette_strategy or extra.get('strategy', 'quantize')
            split = extra.get('idat_split', None)

            out_data = img.to_png(color_type=color_type, bit_depth=bit_depth,
                                  filter_type=filter_type, palette_strategy=strategy,
                                  idat_split=split)

            print()
            print("--- PNG encoder ---")
            print(f"  Color type  : {color_type} ({COLOR_TYPE_NAMES.get(color_type, 'Unknown')})")
            print(f"  Bit depth   : {bit_depth}")
            print(f"  Filter      : {filter_type} ({FILTER_NAMES.get(filter_type, '?')})")

            if color_type == 3:
                qi = img.last_quantization_info
                if qi:
                    print(f"  Palette     : {qi.palette_colors} colors")
                    if qi.is_lossy:
                        exact_pct = 100 * qi.exact_match_count / qi.total_pixels
                        print(f"  Quantized   : {qi.original_colors} -> {qi.palette_colors} colors")
                        print(f"  Exact match : {exact_pct:.1f}%")

        with open(out_path, 'wb') as f:
            f.write(out_data)

        print()
        print("--- Output written ---")
        print(f"  File       : {out_path}")
        print(f"  Bytes      : {len(out_data):,}")
        if in_size > 0:
            ratio = len(out_data) / in_size
            direction = "smaller" if ratio < 1 else "larger"
            print(f"  Ratio      : {ratio:.2f}x ({direction})")

        if args.verify:
            print()
            print("--- Round-trip verification ---")
            try:
                with open(out_path, 'rb') as f:
                    back = f.read()
                img2 = Image.from_file(back, fmt=out_fmt)
                diff_count, max_diff = img.count_differences(img2, ignore_alpha=True)
                if diff_count == 0:
                    print(f"  Result: LOSSLESS (0 channel differences)")
                else:
                    total = img.width * img.height * 3
                    pct = 100 * diff_count / total
                    print(f"  Result: LOSSY ({diff_count:,}/{total:,} channel diffs = {pct:.3f}%)")
                    print(f"  Max per-channel difference: {max_diff}")
            except Exception as e:
                print(f"  Verification failed: {e}")

    except (BMPError, PNGError, PaletteError) as e:
        print_error(e, in_fmt)
        return 1
    except Exception as e:
        print_error(e, in_fmt)
        return 1

    return 0


def cmd_decode(args):
    path = args.input
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return 1

    fmt = detect_format(path) or args.format

    with open(path, 'rb') as f:
        data = f.read()

    try:
        img = Image.from_file(data, fmt=fmt)
        w, h = img.width, img.height
        n = min(args.sample, min(w, h) if args.sample <= 4 else w)

        print()
        print(f"--- Decoded: {path} ({w}x{h}) ---")
        print(f"  Format: {fmt.upper()}")
        print()

        if args.pixel_dump:
            print("Pixel dump (top-left corner):")
            for y in range(min(n, h)):
                row_strs = []
                for x in range(min(n, w)):
                    p = img.pixels[y][x]
                    if len(p) == 4:
                        row_strs.append(f"({p[0]:3d},{p[1]:3d},{p[2]:3d},{p[3]:3d})")
                    else:
                        row_strs.append(f"({p[0]:3d},{p[1]:3d},{p[2]:3d})")
                print(f"  y={y:2d}: " + " ".join(row_strs))
        else:
            corners = [
                ("top-left", 0, 0),
                ("top-right", w - 1, 0),
                ("bottom-left", 0, h - 1),
                ("bottom-right", w - 1, h - 1),
                ("center", w // 2, h // 2),
            ]
            print("Corner & center samples:")
            for name, x, y in corners:
                p = img.pixels[y][x]
                print(f"  {name:13s} ({x:4d},{y:4d}): {p}")
    except (BMPError, PNGError, PaletteError) as e:
        print_error(e, fmt)
        return 1
    except Exception as e:
        print_error(e, fmt)
        return 1
    return 0


def cmd_encode(args):
    width = args.width
    height = args.height
    out_path = args.output
    out_fmt = detect_format(out_path) or args.format
    if not out_fmt:
        print(f"Cannot detect output format, use --format png|bmp")
        return 1

    img = Image(width, height)
    pattern = (args.pattern or 'gradient').lower()

    if pattern == 'gradient':
        for y in range(height):
            for x in range(width):
                r = int(255 * x / max(width - 1, 1))
                g = int(255 * y / max(height - 1, 1))
                b = int(255 * (x + y) / max(width + height - 2, 1))
                img.pixels[y][x] = (r, g, b, 255)
    elif pattern == 'checker':
        size = 8
        for y in range(height):
            for x in range(width):
                black = ((x // size) + (y // size)) % 2 == 0
                v = 0 if black else 255
                img.pixels[y][x] = (v, v, v, 255)
    elif pattern == 'rainbow':
        import math
        for y in range(height):
            for x in range(width):
                h_val = x / max(width - 1, 1)
                i = int(h_val * 6)
                f = h_val * 6 - i
                p = 0
                q = int(255 * (1 - f))
                t = int(255 * f)
                i = i % 6
                colors = [(255, t, p), (q, 255, p), (p, 255, t),
                          (p, q, 255), (t, p, 255), (255, p, q)]
                r, g, b = colors[i]
                img.pixels[y][x] = (r, g, b, 255)
    elif pattern == 'random':
        import random
        random.seed(args.seed or 42)
        for y in range(height):
            for x in range(width):
                img.pixels[y][x] = (
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255),
                )
    elif pattern == 'noise_alpha':
        import random
        random.seed(args.seed or 7)
        for y in range(height):
            for x in range(width):
                img.pixels[y][x] = (
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255),
                )
    elif pattern == 'stripes':
        stripe_w = max(1, width // 8)
        palette = [
            (255, 0, 0), (255, 127, 0), (255, 255, 0),
            (0, 255, 0), (0, 0, 255), (75, 0, 130),
            (148, 0, 211), (255, 255, 255),
        ]
        for y in range(height):
            for x in range(width):
                idx = (x // stripe_w) % len(palette)
                r, g, b = palette[idx]
                img.pixels[y][x] = (r, g, b, 255)
    else:
        print(f"Unknown pattern: {pattern}")
        return 1

    print(f"\n--- Encoding {width}x{height} '{pattern}' -> {out_path} ({out_fmt.upper()}) ---")

    try:
        out_data = img.to_file(out_fmt)
        with open(out_path, 'wb') as f:
            f.write(out_data)
        print(f"  Written {len(out_data):,} bytes")
    except (BMPError, PNGError, PaletteError) as e:
        print_error(e, out_fmt)
        return 1
    except Exception as e:
        print_error(e, out_fmt)
        return 1
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="imcodec",
        description=banner(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  imcodec inspect photo.png
  imcodec inspect photo.bmp --decode
  imcodec convert in.bmp out.png --to png --color-type rgba --filter paeth --verify
  imcodec convert in.png out_256color.bmp --to bmp --bmp-bpp 8 --strategy quantize
  imcodec convert in.png out_indexed.png --color-type indexed --palette-strategy error
  imcodec decode weird_file.bin --format bmp --dump 8
  imcodec encode --width 256 --height 256 --pattern gradient test.png
        """,
    )

    sub = parser.add_subparsers(dest='command', required=True,
                                help='Available commands')

    p_inspect = sub.add_parser('inspect', help='Analyze file format/structure without full decode')
    p_inspect.add_argument('input', help='Input file path')
    p_inspect.add_argument('--format', choices=['bmp', 'png'], help='Force format (detect by extension otherwise)')
    p_inspect.add_argument('--ignore-crc', action='store_true', help='Skip PNG CRC checks')
    p_inspect.add_argument('--decode-and-check', '-d', action='store_true', help='Also run full decode')

    p_conv = sub.add_parser('convert', help='Convert between BMP and PNG')
    p_conv.add_argument('input', help='Input file')
    p_conv.add_argument('output', help='Output file')
    p_conv.add_argument('--from', dest='from_fmt', choices=['bmp', 'png'])
    p_conv.add_argument('--to', dest='to_fmt', choices=['bmp', 'png'])
    p_conv.add_argument('--verify', action='store_true', help='Re-decode output and report lossiness')
    p_conv.add_argument('--bmp-bpp', type=int, choices=[1, 2, 4, 8, 16, 24, 32], help='BMP bits per pixel')
    p_conv.add_argument('--png-color', help='PNG color type: rgb|rgba|indexed|gray or number')
    p_conv.add_argument('--png-depth', type=int, choices=[1, 2, 4, 8], help='PNG bit depth')
    p_conv.add_argument('--png-filter', help='PNG filter: none|sub|up|avg|average|paeth')
    p_conv.add_argument('--palette-strategy', choices=['auto', 'quantize', 'error', 'exact'],
                        help='How to handle too many colors for indexed mode')
    p_conv.add_argument('--options', '-O', help='Comma-separated key=value: idat_split=8192 etc.')

    p_dec = sub.add_parser('decode', help='Decode and sample pixel values')
    p_dec.add_argument('input')
    p_dec.add_argument('--format', choices=['bmp', 'png'])
    p_dec.add_argument('--sample', type=int, default=4, help='Sample window size')
    p_dec.add_argument('--dump', dest='pixel_dump', action='store_true', help='Full NxN dump instead of corners')

    p_enc = sub.add_parser('encode', help='Generate synthetic image to file')
    p_enc.add_argument('--width', type=int, required=True)
    p_enc.add_argument('--height', type=int, required=True)
    p_enc.add_argument('--output', '-o', required=True)
    p_enc.add_argument('--format', choices=['bmp', 'png'])
    p_enc.add_argument('--pattern',
                       help='gradient|checker|rainbow|random|stripes|noise_alpha')
    p_enc.add_argument('--seed', type=int)

    args = parser.parse_args(argv)

    try:
        if args.command == 'inspect':
            return cmd_inspect(args)
        elif args.command == 'convert':
            return cmd_convert(args)
        elif args.command == 'decode':
            return cmd_decode(args)
        elif args.command == 'encode':
            return cmd_encode(args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130

    return 0


if __name__ == '__main__':
    sys.exit(main())
