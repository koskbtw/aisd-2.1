import math

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


def calculate_entropy(file_path):
    try:
        # Читаем файл как последовательность байтов
        with open(file_path, 'rb') as file:
            data = file.read()

        if not data:
            print("Файл пуст")
            return 0.0

        # Считаем частоту каждого байта
        byte_counts = {}
        total_bytes = len(data)

        for byte in data:
            if byte in byte_counts:
                byte_counts[byte] += 1
            else:
                byte_counts[byte] = 1

        # Вычисляем энтропию
        entropy = 0.0
        for count in byte_counts.values():
            probability = count / total_bytes
            entropy -= probability * math.log2(probability)

        return entropy

    except FileNotFoundError:
        print(f"Ошибка: Файл '{file_path}' не найден")
        return None
    except Exception as e:
        print(f"Произошла ошибка: {e}")
        return None


def entropy_haf(filePath):
    code_sum = 0

    with open(filePath, 'rb') as f:
        data = f.read() [:10 ** 7]
    codes, _ = HAF(data)
    reverse_codes = {code: symbol for symbol, code in codes.items()}

    codes = dict(sorted(codes.items()))
    for i in range(256):
        try:
            code_sum += len(codes[i])

        except KeyError: code_sum += 0
    haf_entropy = code_sum / len(codes)
    print(haf_entropy)
    print(codes)


file_path = "../PythonProject/testtt.txt"
entropy = calculate_entropy(file_path)
print(f"Энтропия файла '{file_path}': {entropy:.4f} бит/байт")
print(f"Максимально возможная энтропия для файла такого размера: 8.0000 бит/байт")
entropy_haf(file_path)