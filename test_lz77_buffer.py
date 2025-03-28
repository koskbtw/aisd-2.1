import os
import struct
import matplotlib.pyplot as plt

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
                # Это символ
                if i >= len(encoded_data):
                    raise ValueError("Недостаточно данных для чтения символа.")
                decoded_data.append(encoded_data[i])
                i += 1
            else:
                # Это ссылка
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

def compress_LZ77(file_path, buffer_size):
    # Чтение данных из файла
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

    return compression_ratio

def investigate_compression_ratio(file_path, buffer_sizes):
    compression_ratios = []
    for buffer_size in buffer_sizes:
        print(f"Исследуем размер буфера: {buffer_size}")
        compression_ratio = compress_LZ77(file_path, buffer_size)
        compression_ratios.append(compression_ratio)
        print(f"Коэффициент сжатия: {compression_ratio:.2f}")

    # Построение графика
    plt.plot(buffer_sizes, compression_ratios, marker='o')
    plt.xlabel('Размер буфера')
    plt.ylabel('Коэффициент сжатия')
    plt.title('Зависимость коэффициента сжатия от размера буфера')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    file_path = 'AfterFX.bin'
    buffer_sizes = [2**i - 1 for i in range(1,9)]
    investigate_compression_ratio(file_path, buffer_sizes)