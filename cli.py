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


def _image_has_alpha(img):
    """Best-effort check: does an Image have any non-opaque alpha?"""
    try:
        src_ft = img.get_meta("source_format")
        if src_ft == "png":
            sct = img.get_meta("source_color_type")
            if sct in (4, 6):
                return True
            if sct == 3 and img.get_meta("source_transparency") is not None:
                trans = img.get_meta("source_transparency")
                if isinstance(trans, list) and any(a < 255 for a in trans):
                    return True
        if src_ft == "bmp":
            if img.get_meta("source_bpp") == 32:
                return True
    except Exception:
        pass
    # Scan a few pixels to be sure
    try:
        for y in range(min(img.height, 8)):
            for x in range(min(img.width, 8)):
                p = img.pixels[y][x]
                if len(p) >= 4 and p[3] < 255:
                    return True
    except Exception:
        pass
    return False


def _output_has_alpha(out_fmt, args):
    """Heuristic: does the target format+args produce alpha?"""
    if out_fmt == 'bmp':
        bpp = getattr(args, 'bmp_bpp', None) or 24
        return bpp == 32
    else:  # png
        ct_map = {'rgb': 2, 'rgba': 6, 'indexed': 3, 'palette': 3, 'gray': 0, 'greyscale': 0, 'grayscale': 0}
        ct_name = (getattr(args, 'png_color', None) or 'rgba').lower()
        if ct_name.isdigit():
            color_type = int(ct_name)
        else:
            color_type = ct_map.get(ct_name, 6)
        return color_type in (4, 6)  # GrayAlpha or RGBA


def _resolve_overwrite(target_path, policy):
    """Given an output path and overwrite policy, return (final_path, should_write_bool).

    Policies: always / never / rename / ask.
    ask -> default to always (non-interactive).
    """
    if not os.path.exists(target_path):
        return target_path, True
    if policy == 'always' or policy == 'ask':
        return target_path, True
    if policy == 'never':
        return target_path, False
    if policy == 'rename':
        base, ext = os.path.splitext(target_path)
        i = 1
        while True:
            cand = f"{base}_{i}{ext}"
            if not os.path.exists(cand):
                return cand, True
            i += 1
    return target_path, True


def _write_report(report_path, summary, per_file):
    """Write a JSON or CSV batch report."""
    ext = os.path.splitext(report_path)[1].lower()
    rdir = os.path.dirname(report_path)
    if rdir and not os.path.exists(rdir):
        os.makedirs(rdir, exist_ok=True)

    if ext == '.csv':
        import csv
        fields = ["input", "output", "ok", "in_fmt", "out_fmt", "in_size", "out_size",
                  "lossiness", "diff_count", "max_diff", "note", "error"]
        with open(report_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            w.writeheader()
            for r in per_file:
                w.writerow(r)
            w.writerow({
                "input": "__SUMMARY__",
                "output": "",
                "ok": summary["failed"] == 0,
                "in_fmt": "",
                "out_fmt": "",
                "in_size": summary["total_in"],
                "out_size": summary["total_out"],
                "lossiness": "",
                "diff_count": "",
                "max_diff": "",
                "note": f"succeeded={summary['succeeded']}, failed={summary['failed']}, "
                        f"lossless={summary.get('lossless_cnt', 0)}, lossy={summary.get('lossy_cnt', 0)}, "
                        f"unchecked={summary.get('unchecked_cnt', 0)}",
                "error": "",
            })
    else:
        # Default: JSON
        import json
        payload = {
            "summary": summary,
            "files": per_file,
        }
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)


def _convert_single(in_path, out_path, args, verbose=True):
    """Convert a single file in_path -> out_path.
    Returns dict with many fields for reporting.
    """
    result = {
        "ok": False,
        "input": in_path,
        "output": out_path,
        "in_size": 0,
        "out_size": 0,
        "in_fmt": None,
        "out_fmt": None,
        "lossiness": "unchecked",
        "diff_count": None,
        "max_diff": None,
        "error": None,
        "note": None,
    }

    extra = _parse_extra_options(args.options)

    in_fmt = detect_format(in_path) or args.from_fmt
    out_fmt = detect_format(out_path) or args.to_fmt
    if not in_fmt:
        try:
            with open(in_path, 'rb') as f:
                sig = f.read(16)
            if sig[:2] == b'BM':
                in_fmt = 'bmp'
            elif sig[:8] == b'\x89PNG\r\n\x1a\n':
                in_fmt = 'png'
        except Exception:
            pass
    if not in_fmt or not out_fmt:
        result["error"] = f"Cannot detect format for {'input' if not in_fmt else 'output'}"
        result["note"] = "format detection failed"
        return result

    result["in_fmt"] = in_fmt
    result["out_fmt"] = out_fmt

    try:
        in_size = os.path.getsize(in_path)
        result["in_size"] = in_size
        with open(in_path, 'rb') as f:
            in_data = f.read()
    except Exception as e:
        result["error"] = f"Read failed: {e}"
        return result

    if verbose:
        print(f"\n--- Converting: {in_path} ({in_fmt.upper()}) -> {out_path} ({out_fmt.upper()}) ---")
        print(f"  Input size : {in_size:,} bytes")

    try:
        img = Image.from_file(in_data, fmt=in_fmt)
        if verbose:
            print()
            print("--- Input decoded ---")
            print(img.describe())

        if out_fmt == 'bmp':
            bpp = args.bmp_bpp or extra.get('bpp', 24)
            strategy = args.palette_strategy or extra.get('strategy', 'quantize')
            out_data = img.to_bmp(bits_per_pixel=bpp, palette_strategy=strategy)
            if verbose and bpp <= 8:
                qi = img.last_quantization_info
                if qi and qi.is_lossy:
                    print()
                    print("--- Palette quantization (lossy) ---")
                    print(f"  Original unique colors : {qi.original_colors}")
                    print(f"  Palette size           : {qi.palette_colors} ({bpp}-bit = {1 << bpp} slots)")
                    exact_pct = 100 * qi.exact_match_count / qi.total_pixels if qi.total_pixels else 0
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

            if verbose:
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
                            exact_pct = 100 * qi.exact_match_count / qi.total_pixels if qi.total_pixels else 0
                            print(f"  Quantized   : {qi.original_colors} -> {qi.palette_colors} colors")
                            print(f"  Exact match : {exact_pct:.1f}%")

        out_dir = os.path.dirname(out_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        with open(out_path, 'wb') as f:
            f.write(out_data)
        result["out_size"] = len(out_data)

        if verbose:
            print()
            print("--- Output written ---")
            print(f"  File       : {out_path}")
            print(f"  Bytes      : {len(out_data):,}")
            if in_size > 0:
                ratio = len(out_data) / in_size
                direction = "smaller" if ratio < 1 else "larger"
                print(f"  Ratio      : {ratio:.2f}x ({direction})")

        # --- Decide lossiness (alpha-aware) + heuristic when no --verify ---
        source_has_alpha = _image_has_alpha(img)
        target_has_alpha = _output_has_alpha(out_fmt, args)
        qi = img.last_quantization_info
        heur_lossy_reasons = []
        if qi and qi.is_lossy:
            heur_lossy_reasons.append(f"quantized {qi.original_colors}->{qi.palette_colors} colors")
        if source_has_alpha and not target_has_alpha:
            heur_lossy_reasons.append("alpha discarded")

        if args.verify:
            if verbose:
                print()
                print("--- Round-trip verification (alpha-aware) ---")
            try:
                with open(out_path, 'rb') as f:
                    back = f.read()
                img2 = Image.from_file(back, fmt=out_fmt)
                # Compare alpha only if both sides can carry it:
                # If source had alpha but target doesn't support it, we still compare alpha
                # to detect the alpha loss as "lossy".
                compare_alpha = source_has_alpha
                if compare_alpha and not target_has_alpha:
                    # Alpha will be discarded; we expect 255 on target side.
                    # Compare ignoring alpha for RGB channels, then separately count alpha diffs.
                    diff_rgb, max_rgb = img.count_differences(img2, ignore_alpha=True)
                    alpha_diff = 0
                    max_ad = 0
                    for y in range(img.height):
                        for x in range(img.width):
                            src_a = img.pixels[y][x][3]
                            tgt_a = img2.pixels[y][x][3]
                            if src_a != tgt_a:
                                alpha_diff += 1
                                max_ad = max(max_ad, abs(src_a - tgt_a))
                    diff_count = diff_rgb + alpha_diff
                    max_diff = max(max_rgb, max_ad)
                    if verbose:
                        print(f"  Note: source had alpha but target has none; counting alpha differences")
                else:
                    diff_count, max_diff = img.count_differences(img2, ignore_alpha=not compare_alpha)
                result["diff_count"] = diff_count
                result["max_diff"] = max_diff
                total = img.width * img.height * (4 if compare_alpha else 3)
                if diff_count == 0:
                    result["lossiness"] = "lossless"
                    if verbose:
                        print(f"  Result: LOSSLESS (0 channel differences, "
                              f"alpha={'ON' if compare_alpha else 'OFF'})")
                else:
                    result["lossiness"] = "lossy"
                    pct = 100 * diff_count / total if total else 0
                    if verbose:
                        print(f"  Result: LOSSY ({diff_count:,}/{total:,} channel diffs = {pct:.3f}%)")
                        print(f"  Max per-channel difference: {max_diff}")
                        print(f"  Alpha compare: {'ON (alpha-preserving)' if compare_alpha else 'OFF (RGB only)'}")
            except Exception as e:
                result["error"] = f"Verification failed: {e}"
                result["lossiness"] = "unchecked"
                if verbose:
                    print(f"  Verification failed: {e}")
        else:
            if heur_lossy_reasons:
                result["lossiness"] = "likely-lossy"
                result["note"] = "; ".join(heur_lossy_reasons)
                if verbose:
                    print()
                    print(f"  Heuristic: LIKELY LOSSY ({result['note']})")
            else:
                result["lossiness"] = "likely-lossless"
                result["note"] = "no obvious lossy steps detected"
                if verbose:
                    print()
                    print("  Heuristic: LIKELY LOSSLESS (no --verify; re-check with --verify to confirm)")

        result["ok"] = True

    except (BMPError, PNGError, PaletteError) as e:
        result["error"] = f"{type(e).__name__}: {e}"
        if verbose:
            print_error(e, in_fmt)
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        if verbose:
            print_error(e, in_fmt)

    return result


def cmd_convert(args):
    in_path = args.input
    out_path = args.output

    in_is_dir = os.path.isdir(in_path)
    out_ext = os.path.splitext(out_path)[1]
    out_exists_dir = os.path.isdir(out_path) if os.path.exists(out_path) else False
    out_looks_like_dir = (
        out_exists_dir or
        out_path.endswith(os.sep) or
        (not os.path.exists(out_path) and out_ext == '')
    )
    out_is_dir = in_is_dir or out_looks_like_dir  # batch when either side is a directory

    if not in_is_dir and not out_is_dir:
        # Single file mode
        if not os.path.exists(in_path):
            print(f"Input file not found: {in_path}")
            return 1
        # Handle overwrite policy for single file too
        final_path, should_write = _resolve_overwrite(out_path, args.overwrite or 'always')
        if not should_write:
            print(f"Skipping (--overwrite=never): {out_path} exists")
            return 0
        r = _convert_single(in_path, final_path, args, verbose=True)
        if args.report:
            per = [r]
            summary = {
                "files_processed": 1,
                "succeeded": 1 if r["ok"] else 0,
                "failed": 0 if r["ok"] else 1,
                "lossless_cnt": 1 if r["lossiness"] == "lossless" else 0,
                "lossy_cnt": 1 if r["lossiness"] == "lossy" else 0,
                "unchecked_cnt": 1 if r["lossiness"] in ("unchecked", "likely-lossy", "likely-lossless") else 0,
                "total_in": r["in_size"],
                "total_out": r["out_size"],
            }
            try:
                _write_report(args.report, summary, per)
                print(f"  Report written: {args.report}")
            except Exception as e:
                print(f"  Failed to write report: {e}")
        return 0 if r["ok"] else 1

    # Batch (directory) mode
    if not os.path.isdir(in_path):
        print(f"Input directory not found: {in_path}")
        return 1

    out_dir = out_path
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    if not os.path.isdir(out_dir):
        print(f"Output must be a directory when input is a directory. Got: {out_dir}")
        return 1

    out_fmt = (args.to_fmt or '').lower()
    if out_fmt not in ('bmp', 'png'):
        if args.bmp_bpp:
            out_fmt = 'bmp'
        else:
            out_fmt = 'png'

    # Parse include extensions
    include_ext_set = {'.bmp', '.png', '.dib'}
    if args.include_ext:
        include_ext_set = set()
        for ext in args.include_ext.split(','):
            ext = ext.strip().lower()
            if not ext.startswith('.'):
                ext = '.' + ext
            include_ext_set.add(ext)

    # Gather candidate input files (respect --recursive / --no-recursive)
    candidates = []
    recursive = getattr(args, 'recursive', True)
    keep_structure = getattr(args, 'keep_structure', True)
    overwrite_policy = getattr(args, 'overwrite', 'always')

    if recursive:
        for root, dirs, files in os.walk(in_path):
            for fn in files:
                lower = fn.lower()
                _, ext = os.path.splitext(lower)
                if ext in include_ext_set:
                    candidates.append(os.path.join(root, fn))
    else:
        for fn in os.listdir(in_path):
            full = os.path.join(in_path, fn)
            if os.path.isfile(full):
                lower = fn.lower()
                _, ext = os.path.splitext(lower)
                if ext in include_ext_set:
                    candidates.append(full)
    candidates.sort()

    print()
    print(f"=================================================================")
    print(f"BATCH CONVERT: {len(candidates)} files found")
    print(f"  Input dir  : {in_path}")
    print(f"  Output dir : {out_dir}")
    print(f"  Target fmt : {out_fmt.upper()}")
    print(f"  Recursive  : {'yes' if recursive else 'no'}")
    print(f"  Keep dirs  : {'yes' if keep_structure else 'no (flatten)'}")
    print(f"  Overwrite  : {overwrite_policy}")
    print(f"  Extensions : {', '.join(sorted(include_ext_set))}")
    if args.verify:
        print(f"  Verify     : enabled (lossy/lossless check per file)")
    if args.report:
        print(f"  Report     : {args.report}")
    print(f"=================================================================")

    total_in = 0
    total_out = 0
    succeeded = 0
    failed = 0
    skipped = 0
    lossless_cnt = 0
    lossy_cnt = 0
    unchecked_cnt = 0
    per_file_results = []
    per_file_failures = []

    for idx, src in enumerate(candidates, 1):
        rel = os.path.relpath(src, in_path)
        base = os.path.splitext(os.path.basename(src))[0]
        rel_dir = os.path.dirname(rel)
        if keep_structure and rel_dir:
            dst_dir = os.path.join(out_dir, rel_dir)
        else:
            dst_dir = out_dir
        dst = os.path.join(dst_dir, base + '.' + out_fmt)

        final_dst, should_write = _resolve_overwrite(dst, overwrite_policy)
        if not should_write:
            skipped += 1
            skipped_r = {
                "ok": False, "input": src, "output": dst,
                "in_size": 0, "out_size": 0, "in_fmt": None, "out_fmt": None,
                "lossiness": "unchecked", "diff_count": None, "max_diff": None,
                "error": None, "note": f"skipped (exists, --overwrite={overwrite_policy})",
            }
            per_file_results.append(skipped_r)
            if not args.quiet:
                print(f"\n[{idx}/{len(candidates)}] {rel}")
                print(f"  [SKIP] exists (--overwrite={overwrite_policy})")
            continue

        if not args.quiet:
            print(f"\n[{idx}/{len(candidates)}] {rel}")
        r = _convert_single(src, final_dst, args, verbose=not args.quiet)
        per_file_results.append(r)
        total_in += r["in_size"]
        total_out += r["out_size"]

        if r["ok"]:
            succeeded += 1
            if r["lossiness"] == "lossless":
                lossless_cnt += 1
            elif r["lossiness"] == "lossy":
                lossy_cnt += 1
            else:
                unchecked_cnt += 1
        else:
            failed += 1
            per_file_failures.append((rel, r["error"]))
            if args.quiet:
                print(f"  [FAIL] {rel}: {r['error']}")

    # Final summary
    print()
    print(f"=================================================================")
    print(f"BATCH SUMMARY")
    print(f"=================================================================")
    print(f"  Files processed : {len(candidates)}")
    print(f"  Succeeded       : {succeeded}")
    print(f"  Skipped         : {skipped}")
    print(f"  Failed          : {failed}")
    print(f"  Lossless        : {lossless_cnt}")
    print(f"  Lossy           : {lossy_cnt}")
    print(f"  Unchecked       : {unchecked_cnt}  (no --verify, or heuristic only)")
    print(f"  Total input     : {total_in:,} bytes")
    print(f"  Total output    : {total_out:,} bytes")
    if total_in > 0:
        ratio = total_out / total_in
        direction = "smaller" if ratio < 1 else "larger"
        print(f"  Overall ratio   : {ratio:.2f}x ({direction})")

    if per_file_failures:
        print()
        print(f"  Failures:")
        for rel, err in per_file_failures:
            print(f"    - {rel}: {err}")

    if args.report:
        summary = {
            "files_processed": len(candidates),
            "succeeded": succeeded,
            "failed": failed,
            "skipped": skipped,
            "lossless_cnt": lossless_cnt,
            "lossy_cnt": lossy_cnt,
            "unchecked_cnt": unchecked_cnt,
            "total_in": total_in,
            "total_out": total_out,
            "overall_ratio": (total_out / total_in) if total_in > 0 else None,
        }
        try:
            _write_report(args.report, summary, per_file_results)
            print()
            print(f"  Report written: {args.report}")
        except Exception as e:
            print(f"  Failed to write report: {e}")

    return 0 if failed == 0 else 2


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

    p_conv = sub.add_parser('convert', help='Convert between BMP and PNG (file or directory batch)')
    p_conv.add_argument('input', help='Input file or directory')
    p_conv.add_argument('output', help='Output file or directory')
    p_conv.add_argument('--from', dest='from_fmt', choices=['bmp', 'png'])
    p_conv.add_argument('--to', dest='to_fmt', choices=['bmp', 'png'])
    p_conv.add_argument('--verify', action='store_true', help='Re-decode output and report lossiness')
    p_conv.add_argument('--quiet', action='store_true', help='Batch mode: only print per-file failures + summary')

    # Batch controls
    p_conv.add_argument('--no-recursive', dest='recursive', action='store_false',
                        help='Do not descend into subdirectories (default: recursive)')
    p_conv.add_argument('--keep-structure', dest='keep_structure', action='store_true', default=True,
                        help='Preserve relative directory structure in output (default: on)')
    p_conv.add_argument('--flatten', dest='keep_structure', action='store_false',
                        help='Dump all output files into a single output directory')
    p_conv.add_argument('--overwrite', choices=['ask', 'always', 'never', 'rename'], default='always',
                        help='How to handle existing output files (default: always)')
    p_conv.add_argument('--include-ext', dest='include_ext',
                        help='Comma-separated list of input extensions to process (default: .bmp,.png,.dib)')

    # Reports
    p_conv.add_argument('--report', help='Write a batch report to this path (.json or .csv inferred from extension)')

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
