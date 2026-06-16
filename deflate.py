import zlib as _zlib
import struct


def adler32(data):
    return _zlib.adler32(data) & 0xFFFFFFFF


def zlib_compress(data):
    return _zlib.compress(data)


def zlib_decompress(data):
    return _zlib.decompress(data)


class BitReader:
    def __init__(self, data):
        self.data = data
        self.pos = 0
        self.bit_pos = 0

    def read_bit(self):
        if self.pos >= len(self.data):
            return 0
        byte = self.data[self.pos]
        bit = (byte >> self.bit_pos) & 1
        self.bit_pos += 1
        if self.bit_pos == 8:
            self.bit_pos = 0
            self.pos += 1
        return bit

    def read_bits(self, n):
        val = 0
        for i in range(n):
            val |= (self.read_bit() << i)
        return val

    def read_bytes(self, n):
        if self.bit_pos != 0:
            self.bit_pos = 0
            self.pos += 1
        result = self.data[self.pos:self.pos + n]
        self.pos += n
        return bytes(result)

    def align(self):
        if self.bit_pos != 0:
            self.bit_pos = 0
            self.pos += 1


class BitWriter:
    def __init__(self):
        self.data = bytearray()
        self.current_byte = 0
        self.bit_pos = 0

    def write_bit(self, bit):
        self.current_byte |= (bit & 1) << self.bit_pos
        self.bit_pos += 1
        if self.bit_pos == 8:
            self.data.append(self.current_byte)
            self.current_byte = 0
            self.bit_pos = 0

    def write_bits(self, value, n):
        for i in range(n):
            self.write_bit((value >> i) & 1)

    def write_bytes(self, data):
        if self.bit_pos != 0:
            self.data.append(self.current_byte)
            self.current_byte = 0
            self.bit_pos = 0
        self.data.extend(data)

    def align(self):
        if self.bit_pos != 0:
            self.data.append(self.current_byte)
            self.current_byte = 0
            self.bit_pos = 0

    def get_bytes(self):
        self.align()
        return bytes(self.data)


if __name__ == '__main__':
    test_data = b'Hello, World! This is a test of the deflate compression algorithm. ' * 10
    compressed = zlib_compress(test_data)
    decompressed = zlib_decompress(compressed)
    assert decompressed == test_data
    print(f"Original: {len(test_data)}, Compressed: {len(compressed)}, Ratio: {len(compressed)/len(test_data):.2f}")
    print("Deflate/Inflate test passed!")
