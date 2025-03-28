import struct
from PIL import Image
import os


def rle_encode(data: bytes, m: int) -> bytes:
    result = bytearray()
    i = 0
    n = len(data)
    symbol_bytes = m // 8

    while i < n:
        current_symbol = data[i:i + symbol_bytes]
        count = 1

        while i + symbol_bytes < n and data[i + symbol_bytes:i + symbol_bytes + symbol_bytes] == current_symbol:
            count += 1
            i += symbol_bytes

        if count > 1:
            if count <= 255:
                result.append(count)
                result.extend(current_symbol)
            else:
                result.append(0xFF)
                result.append(count & 0xFF)
                result.append((count >> 8) & 0xFF)
                result.extend(current_symbol)
        else:
            result.append(0x80)
            result.extend(current_symbol)

        i += symbol_bytes

    return bytes(result)


def rle_decode(data: bytes, m: int) -> bytes:
    result = bytearray()
    i = 0
    n = len(data)
    symbol_bytes = m // 8

    while i < n:
        control_byte = data[i]
        i += 1
        if control_byte == 0xFF:
            count = data[i] + (data[i + 1] << 8)
            i += 2
        elif control_byte & 0x80:
            count = 1
        else:
            count = control_byte

        current_symbol = data[i:i + symbol_bytes]
        result.extend(current_symbol * count)
        i += symbol_bytes

    return bytes(result)



class BlockProcessor:
    BLOCK_HEADER = struct.Struct('>I')
    use_header = True

    @classmethod
    def split_blocks(cls, data: bytes, block_size: int) -> list:
        if cls.use_header:
            return [data[i:i + block_size] for i in range(0, len(data), block_size)]
        else:
            return [data]

    @classmethod
    def add_block_header(cls, block: bytes) -> bytes:
        if cls.use_header:
            return cls.BLOCK_HEADER.pack(len(block)) + block
        else:
            return block

    @classmethod
    def read_block(cls, data: bytes, ptr: int) -> tuple:
        if cls.use_header:
            if ptr + cls.BLOCK_HEADER.size > len(data):
                return None, ptr
            block_len = cls.BLOCK_HEADER.unpack_from(data, ptr)[0]
            ptr += cls.BLOCK_HEADER.size
            return data[ptr:ptr + block_len], ptr + block_len
        else:
            return data[ptr:], len(data)



class _Sorting:
    def init(self):
        pass

    def sort_indices(self, rotations):
        indices = list(range(len(rotations)))
        return self.merge_sort(rotations, indices)

    def merge_sort(self, rotations, indices):
        if len(indices) <= 1:
            return indices
        mid = len(indices) // 2
        left = self.merge_sort(rotations, indices[:mid])
        right = self.merge_sort(rotations, indices[mid:])
        return self.merge(rotations, left, right)

    def merge(self, rotations, left, right):
        result = []
        i = j = 0
        while i < len(left) and j < len(right):
            # При равных ротациях сравниваем исходные индексы
            if (rotations[left[i]] < rotations[right[j]] or
                    (rotations[left[i]] == rotations[right[j]] and left[i] < right[j])):
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        result.extend(left[i:])
        result.extend(right[j:])
        return result


class BWT:

    def __init__(self, block_size):
        self.block_size = block_size

    def encode(self, data: bytes) -> bytes:

        encoded = bytearray()
        for block in BlockProcessor.split_blocks(data, self.block_size):
            if not block:
                continue

            rotations = [block[i:] + block[:i] for i in range(len(block))]
            sa = _Sorting().sort_indices(rotations)
            last_col = bytes(rotations[i][-1] for i in sa)
            orig_idx = sa.index(0)

            encoded_block = struct.pack('>II', orig_idx, len(block)) + last_col
            encoded.extend(BlockProcessor.add_block_header(encoded_block))

        return bytes(encoded)

    def decode(self, data: bytes) -> bytes:
        decoded = bytearray()
        ptr = 0

        while ptr <= len(data):
            block, ptr = BlockProcessor.read_block(data, ptr)
            if not block:
                break

            if len(block) < 8:
                continue

            try:
                orig_idx, blen = struct.unpack('>II', block[:8])
            except struct.error:
                continue

            last_col = block[8:8 + blen]

            if len(last_col) != blen:
                continue

            tuples = [(last_col[i], i) for i in range(blen)]
            sorted_tuples = sorted(tuples, key=lambda x: (x[0], x[1]))
            LF = [t[1] for t in sorted_tuples]
            first_col = [t[0] for t in sorted_tuples]

            current_idx = orig_idx
            result = bytearray()
            for _ in range(blen):
                result.append(first_col[current_idx])
                current_idx = LF[current_idx]

            decoded.extend(result)

        return bytes(decoded)



def lz77_encode(data: bytes, buffer_size: int) -> bytes:
    encoded_data = bytearray()
    i = 0
    while i < len(data):
        search_start = max(0, i - buffer_size)
        search_end = i
        search_buffer = data[search_start:search_end]
        max_length = 0
        max_offset = 0
        for length in range(1, min(len(data) - i, buffer_size) + 1):
            substring = data[i:i + length]
            offset = search_buffer.rfind(substring)
            if offset != -1:
                max_length = length
                max_offset = len(search_buffer) - offset
        if max_length > 0:
            encoded_data.append(max_offset)
            encoded_data.append(max_length)
            i += max_length
        else:
            encoded_data.append(0)
            encoded_data.append(0)
            encoded_data.append(data[i])
            i += 1

    return bytes(encoded_data)

def lz77_decode(encoded_data: bytes) -> bytes:
    decoded_data = bytearray()
    i = 0
    while i < len(encoded_data):
        try:
            if i + 2 > len(encoded_data):
                raise ValueError("Недостаточно данных для чтения offset и length.")
            offset = encoded_data[i]
            length = encoded_data[i + 1]
            i += 2

            if offset == 0 and length == 0:
                if i >= len(encoded_data):
                    raise ValueError("Недостаточно данных для чтения символа.")
                decoded_data.append(encoded_data[i])
                i += 1
            else:
                start = len(decoded_data) - offset
                if start < 0 or start + length > len(decoded_data):
                    raise ValueError(f"Некорректные offset или length: offset={offset}, length={length}.")

                for _ in range(length):
                    decoded_data.append(decoded_data[start])
                    start += 1

        except Exception as e:
            print(f"Ошибка декодирования на шаге {i}: {e}")
            break

    return bytes(decoded_data)



def lz78_encode(data: bytes) -> bytes:
    dictionary = {b"": 0}
    current_string = b""
    encoded_data = bytearray()

    for byte in data:
        new_string = current_string + bytes([byte])
        if new_string in dictionary:
            current_string = new_string
        else:
            dictionary[new_string] = len(dictionary)
            encoded_data.extend(dictionary[current_string].to_bytes(4, "big"))
            encoded_data.append(byte)
            current_string = b""

    if current_string:
        encoded_data.extend(dictionary[current_string].to_bytes(4, "big"))
        encoded_data.append(0)

    return bytes(encoded_data)

def lz78_decode(encoded_data: bytes) -> bytes:
    dictionary = {0: b""}
    decoded_data = bytearray()
    i = 0

    while i < len(encoded_data):
        if i + 4 > len(encoded_data):
            raise ValueError("Недостаточно данных для чтения индекса.")
        index = int.from_bytes(encoded_data[i:i + 4], "big")
        i += 4

        if i >= len(encoded_data):
            raise ValueError("Недостаточно данных для чтения символа.")
        byte = encoded_data[i]
        i += 1

        if index not in dictionary:
            raise ValueError(f"Индекс {index} отсутствует в словаре.")

        new_string = dictionary[index] + bytes([byte])
        decoded_data.extend(new_string)
        dictionary[len(dictionary)] = new_string

    return bytes(decoded_data)

def mtf_encode(data: bytes):
    alphabet = list(range(256))
    encoded_data = []

    for byte in data:
        index = alphabet.index(byte)
        encoded_data.append(index)
        alphabet.pop(index)
        alphabet.insert(0, byte)

    return encoded_data

def mtf_decode(encoded_data):
    alphabet = list(range(256))
    decoded_data = []
    for index in encoded_data:
        byte = alphabet[index]
        decoded_data.append(byte)
        alphabet.pop(index)
        alphabet.insert(0, byte)

    return bytes(decoded_data)


def replace_stars(symbols):
    symbols.sort(key=lambda x: len(x[1]))
    codes = {}
    for symbol, code in symbols:
        codes[symbol] = code.replace('*', '')
    return codes


def HAF(S):
    from collections import defaultdict
    freq = defaultdict(int)
    for byte in S:
        freq[byte] += 1
    W = [[weight, [symbol, ""]] for symbol, weight in freq.items()]
    W.sort()
    while len(W) > 1:
        W.sort()
        lo = W.pop(0)
        hi = W.pop(0)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        W.append([lo[0] + hi[0]] + lo[1:] + hi[1:])

    codes = replace_stars(W[0][1:])
    bit_array = []
    for byte in S:
        code = codes[byte]
        bit_array.extend([int(bit) for bit in code])
    return codes, bit_array


def deHAF(bit_array, codes):
    reverse_codes = {code: symbol for symbol, code in codes.items()}
    decoded_string = b""
    current_code = ""
    for bit in bit_array:
        current_code += str(bit)
        if current_code in reverse_codes:
            decoded_string += bytes([reverse_codes[current_code]])
            current_code = ""

    return decoded_string



def bits_to_bytes(bit_array):
    bit_string = ''.join(map(str, bit_array))
    padding = 8 - len(bit_string) % 8
    bit_string += '0' * padding
    encoded_bytes = bytearray()
    for i in range(0, len(bit_string), 8):
        byte = bit_string[i:i+8]
        encoded_bytes.append(int(byte, 2))
    return encoded_bytes, padding

def bytes_to_bits(encoded_bytes, padding):
    bit_string = ''.join(f'{byte:08b}' for byte in encoded_bytes)
    bit_string = bit_string[:-padding] if padding > 0 else bit_string
    return [int(bit) for bit in bit_string]


def compress_HAF(file_path):
    with open(file_path, 'rb') as f:
        data = f.read()

    codes, bit_array = HAF(data)
    encoded_bytes, padding = bits_to_bytes(bit_array)
    compressed_file_path = os.path.splitext(file_path)[0] + '_compressedHAF.bin'
    with open(compressed_file_path, 'wb') as f:

        f.write(struct.pack('>H', len(codes)))
        for symbol, code in codes.items():
            f.write(struct.pack('>B', symbol))
            f.write(struct.pack('>H', len(code)))
            f.write(code.encode('utf-8'))
        f.write(struct.pack('>B', padding))
        f.write(encoded_bytes)

    original_size = len(data)
    compressed_size = os.path.getsize(compressed_file_path)
    compression_ratio = original_size / compressed_size
    print(f"Коэффициент сжатия: {compression_ratio:.2f}")

    with open(compressed_file_path, 'rb') as f:
        num_symbols = struct.unpack('>H', f.read(2))[0]
        codes = {}
        for _ in range(num_symbols):
            symbol = struct.unpack('>B', f.read(1))[0]
            code_length = struct.unpack('>H', f.read(2))[0]
            code = f.read(code_length).decode('utf-8')
            codes[symbol] = code
        padding = struct.unpack('>B', f.read(1))[0]
        encoded_bytes = f.read()

    bit_array = bytes_to_bits(encoded_bytes, padding)
    decoded_data = deHAF(bit_array, codes)
    result_file_path = os.path.splitext(file_path)[0] + '_ResultHAF' + os.path.splitext(file_path)[1]
    with open(result_file_path, 'wb') as f:
        f.write(decoded_data)

    print(f"Декомпрессия завершена. Результат записан в файл: {result_file_path}")

def compress_RLE(file_path, m):
    with open(file_path, 'rb') as f:
        data = f.read()

    encoded_data = rle_encode(data, m)

    compressed_file_path = os.path.splitext(file_path)[0] + '_compressedRLE.bin'
    with open(compressed_file_path, 'wb') as f:
        f.write(struct.pack('>H', m))
        f.write(encoded_data)

    original_size = len(data)
    compressed_size = os.path.getsize(compressed_file_path)
    compression_ratio = original_size / compressed_size
    print(f"Коэффициент сжатия: {compression_ratio:.2f}")

    with open(compressed_file_path, 'rb') as f:
        m = struct.unpack('>H', f.read(2))[0]
        encoded_data = f.read()

    decoded_data = rle_decode(encoded_data, m)
    result_file_path = os.path.splitext(file_path)[0] + '_ResultRLE' + os.path.splitext(file_path)[1]
    with open(result_file_path, 'wb') as f:
        f.write(decoded_data)

    print(f"Декомпрессия завершена. Результат записан в файл: {result_file_path}")





def compress_LZ77(file_path, buffer_size=128):
    with open(file_path, 'rb') as f:
        data = f.read()

    encoded_data = lz77_encode(data, buffer_size)

    compressed_file_path = os.path.splitext(file_path)[0] + '_compressedLZ77.bin'
    with open(compressed_file_path, 'wb') as f:
        f.write(struct.pack('>I', buffer_size))
        f.write(encoded_data)

    original_size = len(data)
    compressed_size = os.path.getsize(compressed_file_path)
    compression_ratio = original_size / compressed_size
    print(f"Коэффициент сжатия: {compression_ratio:.2f}")

    with open(compressed_file_path, 'rb') as f:
        buffer_size = struct.unpack('>I', f.read(4))[0]
        encoded_data = f.read()

    decoded_data = lz77_decode(encoded_data)
    result_file_path = os.path.splitext(file_path)[0] + '_ResultLZ77' + os.path.splitext(file_path)[1]
    with open(result_file_path, 'wb') as f:
        f.write(decoded_data)

    print(f"Декомпрессия завершена. Результат записан в файл: {result_file_path}")



def compress_LZ77_HA(file_path, buffer_size=128):
    with open(file_path, 'rb') as f:
        data = f.read()

    lz77_encoded_data = lz77_encode(data, buffer_size)
    codes, bit_array = HAF(lz77_encoded_data)
    def bits_to_bytes(bit_array):
        bit_string = ''.join(map(str, bit_array))
        padding = 8 - len(bit_string) % 8
        bit_string += '0' * padding
        encoded_bytes = bytearray()
        for i in range(0, len(bit_string), 8):
            byte = bit_string[i:i+8]
            encoded_bytes.append(int(byte, 2))
        return encoded_bytes, padding

    ha_encoded_bytes, padding = bits_to_bytes(bit_array)
    compressed_file_path = os.path.splitext(file_path)[0] + '_compressedLZ77_HA.bin'
    with open(compressed_file_path, 'wb') as f:
        f.write(struct.pack('>I', buffer_size))
        f.write(struct.pack('>H', len(codes)))
        for symbol, code in codes.items():
            f.write(struct.pack('>B', symbol))
            f.write(struct.pack('>H', len(code)))
            f.write(code.encode('utf-8'))
        f.write(struct.pack('>B', padding))
        f.write(ha_encoded_bytes)

    original_size = len(data)
    compressed_size = os.path.getsize(compressed_file_path)
    compression_ratio = original_size / compressed_size
    print(f"Коэффициент сжатия: {compression_ratio:.2f}")

    with open(compressed_file_path, 'rb') as f:
        buffer_size = struct.unpack('>I', f.read(4))[0]
        num_symbols = struct.unpack('>H', f.read(2))[0]
        codes = {}
        for _ in range(num_symbols):
            symbol = struct.unpack('>B', f.read(1))[0]
            code_length = struct.unpack('>H', f.read(2))[0]
            code = f.read(code_length).decode('utf-8')
            codes[symbol] = code
        padding = struct.unpack('>B', f.read(1))[0]
        ha_encoded_bytes = f.read()

    def bytes_to_bits(encoded_bytes, padding):
        bit_string = ''.join(f'{byte:08b}' for byte in encoded_bytes)
        bit_string = bit_string[:-padding] if padding > 0 else bit_string
        return [int(bit) for bit in bit_string]

    bit_array = bytes_to_bits(ha_encoded_bytes, padding)
    lz77_encoded_data = deHAF(bit_array, codes)
    decoded_data = lz77_decode(lz77_encoded_data)
    result_file_path = os.path.splitext(file_path)[0] + '_ResultLZ77_HA' + os.path.splitext(file_path)[1]
    with open(result_file_path, 'wb') as f:
        f.write(decoded_data)
    print(f"Декомпрессия завершена. Результат записан в файл: {result_file_path}")

def compress_LZ78(file_path):

    with open(file_path, 'rb') as f:
        data = f.read()
    encoded_data = lz78_encode(data)
    compressed_file_path = os.path.splitext(file_path)[0] + '_compressedLZ78.bin'
    with open(compressed_file_path, 'wb') as f:
        f.write(encoded_data)
    original_size = len(data)
    compressed_size = os.path.getsize(compressed_file_path)
    compression_ratio = original_size / compressed_size
    print(f"Коэффициент сжатия: {compression_ratio:.2f}")
    with open(compressed_file_path, 'rb') as f:
        encoded_data = f.read()
    decoded_data = lz78_decode(encoded_data)
    result_file_path = os.path.splitext(file_path)[0] + '_ResultLZ78' + os.path.splitext(file_path)[1]
    with open(result_file_path, 'wb') as f:
        f.write(decoded_data)

    print(f"Декомпрессия завершена. Результат записан в файл: {result_file_path}")

def compress_LZ78_HA(file_path):
    with open(file_path, 'rb') as f:
        data = f.read()

    lz78_encoded_data = lz78_encode(data)
    codes, bit_array = HAF(lz78_encoded_data)

    ha_encoded_bytes, padding = bits_to_bytes(bit_array)
    compressed_file_path = os.path.splitext(file_path)[0] + '_compressedLZ78_HA.bin'
    with open(compressed_file_path, 'wb') as f:
        f.write(struct.pack('>H', len(codes)))
        for symbol, code in codes.items():
            f.write(struct.pack('>B', symbol))
            f.write(struct.pack('>H', len(code)))
            f.write(code.encode('utf-8'))
        f.write(struct.pack('>B', padding))
        f.write(ha_encoded_bytes)
    original_size = len(data)
    compressed_size = os.path.getsize(compressed_file_path)
    compression_ratio = original_size / compressed_size
    print(f"Коэффициент сжатия: {compression_ratio:.2f}")
    with open(compressed_file_path, 'rb') as f:
        num_symbols = struct.unpack('>H', f.read(2))[0]
        codes = {}
        for _ in range(num_symbols):
            symbol = struct.unpack('>B', f.read(1))[0]
            code_length = struct.unpack('>H', f.read(2))[0]
            code = f.read(code_length).decode('utf-8')
            codes[symbol] = code
        padding = struct.unpack('>B', f.read(1))[0]
        ha_encoded_bytes = f.read()

    bit_array = bytes_to_bits(ha_encoded_bytes, padding)
    lz78_encoded_data = deHAF(bit_array, codes)
    decoded_data = lz78_decode(lz78_encoded_data)
    result_file_path = os.path.splitext(file_path)[0] + '_ResultLZ78_HA' + os.path.splitext(file_path)[1]
    with open(result_file_path, 'wb') as f:
        f.write(decoded_data)
    print(f"Декомпрессия завершена. Результат записан в файл: {result_file_path}")

def compress_BWT_MTF_HA(file_path):
    with open(file_path, 'rb') as f:
        data = f.read()

    bwt = BWT(block_size=1024)
    bwtEncoded_data = bwt.encode(data)
    mtf_encoded_data = mtf_encode(bwtEncoded_data)
    codes, bit_array = HAF(mtf_encoded_data)
    ha_encoded_bytes, padding = bits_to_bytes(bit_array)
    compressed_file_path = os.path.splitext(file_path)[0] + '_compressed_BWT_MTF_HA.bin'
    with open(compressed_file_path, 'wb') as f:
        f.write(struct.pack('>H', len(codes)))
        for symbol, code in codes.items():
            f.write(struct.pack('>B', symbol))
            f.write(struct.pack('>H', len(code)))
            f.write(code.encode('utf-8'))
        f.write(struct.pack('>B', padding))
        f.write(ha_encoded_bytes)

    original_size = len(data)
    compressed_size = os.path.getsize(compressed_file_path)
    compression_ratio = original_size / compressed_size
    print(f"Коэффициент сжатия: {compression_ratio:.2f}")
    with open(compressed_file_path, 'rb') as f:
        num_symbols = struct.unpack('>H', f.read(2))[0]
        codes = {}
        for _ in range(num_symbols):
            symbol = struct.unpack('>B', f.read(1))[0]
            code_length = struct.unpack('>H', f.read(2))[0]
            code = f.read(code_length).decode('utf-8')
            codes[symbol] = code
        padding = struct.unpack('>B', f.read(1))[0]
        ha_encoded_bytes = f.read()

    bit_array = bytes_to_bits(ha_encoded_bytes, padding)
    mtf_decoded_data = mtf_decode(deHAF(bit_array, codes))
    decoded_data = bwt.decode(mtf_decoded_data)

    result_file_path = os.path.splitext(file_path)[0] + '_Result_BWT_MTF_HA' + os.path.splitext(file_path)[1]
    with open(result_file_path, 'wb') as f:
        f.write(decoded_data)

    print(f"Декомпрессия завершена. Результат записан в файл: {result_file_path}")



def compress_BWT_MTF_RLE_HA(file_path, m):
    with open(file_path, 'rb') as f:
        data = f.read()

    bwt = BWT(block_size=1024)
    bwtEncoded_data = bwt.encode(data)
    mtf_encoded_data = mtf_encode(bwtEncoded_data)
    rle_encode_data = rle_encode(mtf_encoded_data, m)
    codes, bit_array = HAF(rle_encode_data)
    ha_encoded_bytes, padding = bits_to_bytes(bit_array)
    compressed_file_path = os.path.splitext(file_path)[0] + '_compressed_BWT_MTF_RLE_HA.bin'
    with open(compressed_file_path, 'wb') as f:
        f.write(struct.pack('>H', len(codes)))
        for symbol, code in codes.items():
            f.write(struct.pack('>B', symbol))
            f.write(struct.pack('>H', len(code)))
            f.write(code.encode('utf-8'))
        f.write(struct.pack('>B', padding))
        f.write(ha_encoded_bytes)

    original_size = len(data)
    compressed_size = os.path.getsize(compressed_file_path)
    compression_ratio = original_size / compressed_size
    print(f"Коэффициент сжатия: {compression_ratio:.2f}")
    with open(compressed_file_path, 'rb') as f:
        num_symbols = struct.unpack('>H', f.read(2))[0]
        codes = {}
        for _ in range(num_symbols):
            symbol = struct.unpack('>B', f.read(1))[0]
            code_length = struct.unpack('>H', f.read(2))[0]
            code = f.read(code_length).decode('utf-8')
            codes[symbol] = code
        padding = struct.unpack('>B', f.read(1))[0]
        ha_encoded_bytes = f.read()

    bit_array = bytes_to_bits(ha_encoded_bytes, padding)
    rle_decoded_data = rle_decode(deHAF(bit_array, codes), m)
    mtf_decoded_data = mtf_decode(rle_decoded_data)
    decoded_data = bwt.decode(mtf_decoded_data)
    result_file_path = os.path.splitext(file_path)[0] + '_Result_BWT_MTF_RLE_HA' + os.path.splitext(file_path)[1]
    with open(result_file_path, 'wb') as f:
        f.write(decoded_data)

    print(f"Декомпрессия завершена. Результат записан в файл: {result_file_path}")

def compress_BWT_RLE(file_path, m):
    with open(file_path, 'rb') as f:
        data = f.read()

    bwt = BWT(block_size=1024)
    bwtEncoded_data = bwt.encode(data)
    rle_encoded_data = rle_encode(bwtEncoded_data, m)
    compressed_file_path = os.path.splitext(file_path)[0] + '_compressedBWT_RLE.bin'
    with open(compressed_file_path, 'wb') as f:
        f.write(struct.pack('>H', m))
        f.write(rle_encoded_data)

    original_size = len(data)
    compressed_size = os.path.getsize(compressed_file_path)
    compression_ratio = original_size / compressed_size
    print(f"Коэффициент сжатия: {compression_ratio:.2f}")
    with open(compressed_file_path, "rb") as f:
        m = struct.unpack('>H', f.read(2))[0]
        rle_encoded_data = f.read()

    bwt_restore = rle_decode(rle_encoded_data, m)
    decoded_data = bwt.decode(bwt_restore)
    result_file_path = os.path.splitext(file_path)[0] + '_ResultBWT_RLE' + os.path.splitext(file_path)[1]
    with open(result_file_path, 'wb') as f:
        f.write(decoded_data)
    print(f"Декомпрессия завершена. Результат записан в файл: {result_file_path}")

# def openpicture(image_path: str, marker: int) -> str:
#     if not os.path.exists(image_path):
#         raise FileNotFoundError(f"Файл '{image_path}' не найден.")
#     if marker not in {1, 2, 3}:
#         raise ValueError("Маркер должен быть 1, 2 или 3.")
#     img = Image.open(image_path)
#     output_dir = "converted_images"
#     os.makedirs(output_dir, exist_ok=True)
#     base_name = os.path.splitext(os.path.basename(image_path))[0]
#     if marker == 1:
#         output_path = os.path.join(output_dir, f"{base_name}_bw.jpg")
#         img = img.convert('1')
#     elif marker == 2:
#         output_path = os.path.join(output_dir, f"{base_name}_grayscale.jpg")
#         img = img.convert('L')
#     elif marker == 3:
#         output_path = os.path.join(output_dir, f"{base_name}_color.jpg")
#         img = img.convert('RGB')
#     img.save(output_path)
#     print(f"Изображение сохранено как '{output_path}'")
#     return output_path


def convert_to_raw(image_path: str, marker: int) -> str:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Файл '{image_path}' не найден.")
    if marker not in {1, 2, 3}:
        raise ValueError("Маркер должен быть 1, 2 или 3.")
    img = Image.open(image_path)
    output_dir = "converted_raw"
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    if marker == 1:
        img = img.convert("L")
        img = img.point(lambda x: 0 if x < 128 else 255, "1")
        output_path = os.path.join(output_dir, f"{base_name}_bw.raw")
    elif marker == 2:
        img = img.convert("L")
        output_path = os.path.join(output_dir, f"{base_name}_grayscale.raw")
    elif marker == 3:
        img = img.convert("RGB")
        output_path = os.path.join(output_dir, f"{base_name}_color.raw")
    with open(output_path, "wb") as f:
        f.write(img.tobytes())

    print(f"Изображение сохранено в RAW: '{output_path}'")
    return output_path



def main():
    print("choose the test for compression: ")
    print("1.  russian text, size >= 200 kb")
    print("2. bin file")
    print("3. ewnwik7 file (10**7 of enwik9)")
    print("4. black white image")
    print("5. grayscale image")
    print("6. color image")
    print("0. exit")
    y = int(input())
    if y == 1:
        path = 'Esenin.txt'
    elif y == 2:
        path = "AfterFX.bin"
    elif y == 3:
        path = 'enwik7.pmd'
    elif y == 4 or y == 5 or y ==6:
        path = "image.jpg"
        if y == 4:
            # path = openpicture(path,1)
            path = convert_to_raw(path,1)
        elif y == 5:
            # path = openpicture(path,2)
            path = convert_to_raw(path, 2)
        else:
            # path = openpicture(path,3)
            path = convert_to_raw(path, 3)
    else:
        print("Incorrect input!")
    print("choose the compressor: ")
    print("1 - HA\n2 - RLE\n3 - BWT + RLE\n4 - BWT + MTF + HA\n5 - BWT + MTF + RLE + HA\n6 - LZ77\n7 - LZ77 + HA\n8 - LZ78\n9 - LZ78 + HA")
    x = int(input())
    if x == 1:
        compress_HAF(path)
    elif x == 2:
        print("enter 'M' in bits: ")
        m = int(input())
        compress_RLE(path, m)
    elif x == 3:
        print("enter 'M' in bits: ")
        m = int(input())
        compress_BWT_RLE(path, m)
    elif x == 4:
        compress_BWT_MTF_HA(path)
    elif x == 5:
        print("enter 'M' in bits: ")
        m = int(input())
        compress_BWT_MTF_RLE_HA(path, m)
    elif x == 6:
        compress_LZ77(path)
    elif x == 7:
        compress_LZ77_HA(path)
    elif x == 8:
        compress_LZ78(path)
    elif x == 9:
        compress_LZ78_HA(path)
    else:
        print("Incorrect input!")


main()

