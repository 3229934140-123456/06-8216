import random
import sys
from image import Image
from bmp_codec import bmp_bytes_per_row, bmp_row_padding
from bmp_codec import BMPImage, read_bmp, write_bmp


def create_test_image(width, height, seed=42):
    random.seed(seed)
    img = Image(width, height)
    for y in range(height):
        for x in range(width):
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            a = random.randint(0, 255)
            img.pixels[y][x] = (r, g, b, a)
    return img


def create_gradient_image(width, height):
    img = Image(width, height)
    for y in range(height):
        for x in range(width):
            r = int(255 * x / max(width - 1, 1))
            g = int(255 * y / max(height - 1, 1))
            b = int(255 * (x + y) / max(width + height - 2, 1))
            img.pixels[y][x] = (r, g, b, 255)
    return img


def test_bmp_roundtrip():
    print("=" * 60)
    print("Testing BMP Roundtrip (various bit depths)")
    print("=" * 60)

    lossy_bpp = [1, 4, 8, 16]
    lossless_bpp = [24, 32]

    all_pass = True

    for bpp in lossy_bpp:
        width, height = 11, 7
        bmp = BMPImage()
        bmp.width = width
        bmp.height = height
        bmp.bits_per_pixel = bpp

        if bpp <= 8:
            palette_size = 1 << bpp
            palette = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(palette_size)]
            bmp.palette = palette
            bmp.pixels = [[random.randint(0, palette_size - 1) for _ in range(width)] for _ in range(height)]
        else:
            row_list = []
            for _ in range(height):
                row = []
                for _ in range(width):
                    r = random.randint(0, 31) << 3
                    g = random.randint(0, 31) << 3
                    b = random.randint(0, 31) << 3
                    row.append((r, g, b))
                row_list.append(row)
            bmp.pixels = row_list

        try:
            data = write_bmp(bmp)
            decoded = read_bmp(data)

            if decoded.width != width or decoded.height != height or decoded.bits_per_pixel != bpp:
                print(f"  [FAIL] {bpp}bpp: dimension mismatch {decoded.width}x{decoded.height} @ {decoded.bits_per_pixel}bpp")
                all_pass = False
                continue

            mismatch_count = 0
            for y in range(height):
                for x in range(width):
                    if bpp <= 8:
                        if bmp.pixels[y][x] != decoded.pixels[y][x]:
                            mismatch_count += 1
                    else:
                        o = bmp.pixels[y][x]
                        d = decoded.pixels[y][x]
                        if o[:3] != d[:3]:
                            mismatch_count += 1

            row_bytes = bmp_bytes_per_row(width, bpp)
            pad = bmp_row_padding(width, bpp)

            if mismatch_count == 0:
                print(f"  [PASS] {bpp:>2d}bpp native: {width}x{height}, size={len(data)}B, row={row_bytes}B, pad={pad}B")
            else:
                print(f"  [FAIL] {bpp}bpp: {mismatch_count} pixel mismatches")
                all_pass = False
        except Exception as e:
            print(f"  [ERROR] {bpp}bpp: {e}")
            all_pass = False

    for bpp in lossless_bpp:
        width, height = 7, 5
        orig = create_test_image(width, height, seed=width * 1000 + height)
        try:
            data = orig.to_bmp(bits_per_pixel=bpp)
            decoded = Image.from_bmp(data)

            if decoded.width != width or decoded.height != height:
                print(f"  [FAIL] {bpp}bpp Image API: dimension mismatch")
                all_pass = False
                continue

            mismatch_count = 0
            for y in range(height):
                for x in range(width):
                    o = orig.pixels[y][x]
                    d = decoded.pixels[y][x]
                    if o[:3] != d[:3]:
                        mismatch_count += 1

            if mismatch_count == 0:
                print(f"  [PASS] {bpp:>2d}bpp Image API: {width}x{height}, size={len(data)}B")
            else:
                print(f"  [FAIL] {bpp}bpp Image API: {mismatch_count} pixel mismatches")
                all_pass = False
        except Exception as e:
            print(f"  [ERROR] {bpp}bpp Image API: {e}")
            all_pass = False

    return all_pass


def test_png_roundtrip():
    print("\n" + "=" * 60)
    print("Testing PNG Roundtrip (all filters and color types)")
    print("=" * 60)

    all_pass = True

    for color_type, ct_name in [(2, "RGB"), (6, "RGBA"), (3, "Indexed")]:
        for filter_type in range(5):
            filter_names = ["None", "Sub", "Up", "Average", "Paeth"]
            width, height = 11, 7

            orig = create_gradient_image(width, height)
            try:
                data = orig.to_png(color_type=color_type, bit_depth=8, filter_type=filter_type)
                decoded = Image.from_png(data)

                if decoded.width != width or decoded.height != height:
                    print(f"  [FAIL] {ct_name} + {filter_names[filter_type]}: dimension mismatch")
                    all_pass = False
                    continue

                mismatch_count = 0
                for y in range(height):
                    for x in range(width):
                        o = orig.pixels[y][x]
                        d = decoded.pixels[y][x]
                        if color_type == 6:
                            if o != d:
                                mismatch_count += 1
                        else:
                            if o[:3] != d[:3]:
                                mismatch_count += 1

                if mismatch_count == 0:
                    print(f"  [PASS] {ct_name:9s} + {filter_names[filter_type]:8s}: {len(data):>5d}B")
                else:
                    print(f"  [FAIL] {ct_name} + {filter_names[filter_type]}: {mismatch_count} pixel mismatches")
                    all_pass = False
            except Exception as e:
                print(f"  [ERROR] {ct_name} + {filter_names[filter_type]}: {e}")
                all_pass = False

    print("\n  Testing Grayscale (lossy conversion):")
    for filter_type in range(5):
        filter_names = ["None", "Sub", "Up", "Average", "Paeth"]
        width, height = 11, 7
        orig = create_gradient_image(width, height)
        try:
            data = orig.to_png(color_type=0, bit_depth=8, filter_type=filter_type)
            decoded = Image.from_png(data)

            if decoded.width != width or decoded.height != height:
                print(f"    [FAIL] Grayscale + {filter_names[filter_type]}: dimension mismatch")
                all_pass = False
                continue

            mismatch_count = 0
            for y in range(height):
                for x in range(width):
                    o = orig.pixels[y][x]
                    d = decoded.pixels[y][x]
                    orig_gray = int(0.299 * o[0] + 0.587 * o[1] + 0.114 * o[2])
                    if (orig_gray, orig_gray, orig_gray) != d[:3]:
                        mismatch_count += 1

            if mismatch_count == 0:
                print(f"    [PASS] Grayscale + {filter_names[filter_type]:8s}: {len(data):>5d}B")
            else:
                print(f"    [FAIL] Grayscale + {filter_names[filter_type]}: {mismatch_count} pixel mismatches")
                all_pass = False
        except Exception as e:
            print(f"    [ERROR] Grayscale + {filter_names[filter_type]}: {e}")
            all_pass = False

    return all_pass


def test_format_conversion():
    print("\n" + "=" * 60)
    print("Testing Cross-Format Conversion (BMP <-> PNG)")
    print("=" * 60)

    all_pass = True
    width, height = 9, 6

    orig = create_test_image(width, height, seed=12345)

    conversions = [
        ("BMP24 -> PNG_RGB_Paeth", lambda: orig.to_bmp(24), lambda d: Image.from_bmp(d).to_png(color_type=2, filter_type=4)),
        ("PNG_RGBA -> BMP32", lambda: orig.to_png(color_type=6, filter_type=4), lambda d: Image.from_png(d).to_bmp(32)),
        ("BMP8 -> PNG_Indexed_Sub", lambda: orig.to_bmp(8), lambda d: Image.from_bmp(d).to_png(color_type=3, filter_type=1)),
        ("PNG_RGB_Average -> BMP24", lambda: orig.to_png(color_type=2, filter_type=3), lambda d: Image.from_png(d).to_bmp(24)),
    ]

    for name, step1, step2 in conversions:
        try:
            data1 = step1()
            img_mid = Image.from_file(data1)
            data2 = step2(data1)
            img_final = Image.from_file(data2)

            mismatch_count = 0
            for y in range(height):
                for x in range(width):
                    o = orig.pixels[y][x]
                    f = img_final.pixels[y][x]
                    if o[:3] != f[:3]:
                        mismatch_count += 1

            if mismatch_count == 0:
                print(f"  [PASS] {name}: intermediate={len(data1)}B, final={len(data2)}B")
            else:
                print(f"  [FAIL] {name}: {mismatch_count} pixel mismatches")
                all_pass = False
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")
            all_pass = False

    return all_pass


def test_filter_efficiency():
    print("\n" + "=" * 60)
    print("Filter Efficiency Comparison (gradient image)")
    print("=" * 60)

    filter_names = ["None", "Sub", "Up", "Average", "Paeth"]
    width, height = 64, 64
    img = create_gradient_image(width, height)

    sizes = []
    for ft in range(5):
        data = img.to_png(color_type=2, filter_type=ft)
        sizes.append(len(data))

    best = min(sizes)
    for ft in range(5):
        ratio = sizes[ft] / best
        marker = " <-- best" if sizes[ft] == best else ""
        print(f"  {filter_names[ft]:8s}: {sizes[ft]:>6d}B  ({ratio:.2f}x){marker}")

    return True


def main():
    print("\n" + "#" * 60)
    print("#" + " " * 18 + "Image Codec Test Suite" + " " * 18 + "#")
    print("#" * 60)

    results = []
    results.append(("BMP Roundtrip", test_bmp_roundtrip()))
    results.append(("PNG Roundtrip", test_png_roundtrip()))
    results.append(("Format Conversion", test_format_conversion()))
    results.append(("Filter Efficiency", test_filter_efficiency()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name:25s}: {status}")
        if not passed:
            all_pass = False

    print("\n" + "=" * 60)
    if all_pass:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")
        sys.exit(1)
    print("=" * 60)


if __name__ == '__main__':
    main()
