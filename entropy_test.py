import math
import struct
from collections import Counter
import matplotlib.pyplot as plt



def calculate_entropy(data):
    """Вычисляет энтропию данных в битах на символ"""
    if not data:
        return 0.0

    counts = Counter(data)
    length = len(data)
    entropy = 0.0

    for count in counts.values():
        probability = count / length
        entropy -= probability * math.log2(probability)

    return entropy


def process_bwt_mtf_entropy(data, block_sizes):
    """Вычисляет энтропию после BWT+MTF для разных размеров блоков"""
    results = []

    for block_size in block_sizes:
        # Применяем BWT
        bwt = BWT(block_size)
        bwt_encoded = bwt.encode(data)

        # Применяем MTF
        mtf_encoded = BWT.mtf_encode(bwt_encoded)

        # Вычисляем энтропию
        entropy = calculate_entropy(mtf_encoded)
        results.append(entropy)

        print(f"Block size: {block_size}, Entropy: {entropy:.4f} bits/symbol")

    return results


def plot_results(block_sizes, entropies):
    """Визуализирует результаты"""
    plt.figure(figsize=(10, 6))
    plt.plot(block_sizes, entropies, marker='o')
    plt.xlabel('Block Size')
    plt.ylabel('Entropy (bits/symbol)')
    plt.title('Entropy after BWT+MTF vs Block Size (enwik7)')
    plt.grid(True)
    plt.xscale('log')
    plt.show()





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

def main():
    # Загрузка enwik7 (предполагается, что файл находится в той же директории)
    with open('enwik7.pmd', 'rb') as f:
        data = f.read()

    # Выбираем диапазон размеров блоков для исследования
    block_sizes = [128, 256, 512, 1024, 2048, 4096, 8192]
    entropies = process_bwt_mtf_entropy(data, block_sizes)

    # Строим график
    plot_results(block_sizes, entropies)


if __name__ == "__main__":
    main()

