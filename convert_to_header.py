# convert_to_header.py

input_file = "tiny_cnn_esp32_int8.tflite"
output_file = "tiny_cnn_model.h"

with open(input_file, "rb") as f:
    data = f.read()

with open(output_file, "w") as f:
    f.write("#ifndef TINY_CNN_MODEL_H\n")
    f.write("#define TINY_CNN_MODEL_H\n\n")

    f.write("const unsigned char tiny_cnn_esp32_int8_tflite[] = {\n")

    for i, byte in enumerate(data):
        if i % 12 == 0:
            f.write("\n")
        f.write(f"0x{byte:02x}, ")

    f.write("\n};\n\n")
    f.write(f"const int tiny_cnn_esp32_int8_tflite_len = {len(data)};\n\n")

    f.write("#endif\n")

print("✅ Header file created: tiny_cnn_model.h")
