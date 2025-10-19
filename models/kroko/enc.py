import json
import os
import struct
import glob

def convert_kroko_model_final_v9(input_file_path: str, output_dir: str):
    """
    Конвертирует .data файл в onnx
    """
    print(f"Открываю файл: {input_file_path}")
    
    try:
        with open(input_file_path, 'rb') as f:
            # --- Шаг 1: Найти и прочитать JSON-заголовок ---
            header_bytes = bytearray()
            brace_count = 0
            found_start = False
            while True:
                byte = f.read(1)
                if not byte:
                    raise ValueError("Не удалось найти JSON-заголовок в файле.")
                
                char = byte.decode('utf-8', errors='ignore')
                header_bytes.append(byte[0])

                if char == '{':
                    brace_count += 1
                    found_start = True
                elif char == '}':
                    brace_count -= 1
                
                if found_start and brace_count == 0:
                    break
            
            print("JSON-заголовок найден и прочитан.")
            
            # --- Шаг 2: Последовательно извлечь каждый файл ---
            os.makedirs(output_dir, exist_ok=True)
            print(f"\nНачинаю извлечение файлов в: {output_dir}")

            # Порядок файлов важен!
            file_names = ["encoder.onnx", "decoder.onnx", "joiner.onnx", "tokens.txt"]
            size_format = '<I'  # 4-байтовое беззнаковое целое (unsigned int)
            size_len = struct.calcsize(size_format)

            for file_name in file_names:
                # 1. Прочитать 4 байта - размер следующего файла
                size_bytes = f.read(size_len)
                if len(size_bytes) != size_len:
                    raise EOFError(f"Не удалось прочитать размер для файла '{file_name}'. Файл закончился раньше времени.")
                
                file_size = struct.unpack(size_format, size_bytes)[0]
                
                print(f"  - Найден файл '{file_name}', размер: {file_size} байт")

                if file_size <= 0:
                    print(f"    Предупреждение: размер файла 0 или меньше. Файл будет пропущен.")
                    continue

                # 2. Прочитать данные файла (ровно file_size байт)
                content = f.read(file_size)
                if len(content) != file_size:
                    raise EOFError(f"Не удалось прочитать содержимое файла '{file_name}'. Ожидалось {file_size} байт, получено {len(content)}.")

                # 3. Сохранить чистые данные на диск
                output_path = os.path.join(output_dir, file_name)
                with open(output_path, 'wb') as out_f:
                    out_f.write(content)
                
                print(f"    -> Успешно сохранен: {output_path}")

            print("\nКонвертация успешно завершена!")

    except FileNotFoundError:
        print(f"Ошибка: файл не найден по пути {input_file_path}")
    except Exception as e:
        print(f"Произошла непредвиденная ошибка: {e}")

if __name__ == '__main__':
    # Ищем файлы .data в текущей директории
    current_dir = os.getcwd()
    data_files = glob.glob('*.data')

    if len(data_files) == 0:
        print("Ошибка: в текущей папке не найдены файлы с расширением .data")
    elif len(data_files) > 1:
        print("Ошибка: в текущей папке найдено несколько .data файлов. Оставьте только один.")
        for f in data_files:
            print(f" - {f}")
    else:
        # Найден ровно один файл, запускаем конвертацию
        input_file = data_files[0]
        # Распаковываем в ту же папку, где лежит .data файл
        output_directory = os.path.dirname(os.path.abspath(input_file))
        
        convert_kroko_model_final_v9(input_file, output_directory)