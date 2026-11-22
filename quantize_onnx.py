import os
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

ONNX_DIR = "./onnx_model"
QUANT_DIR = "./onnx_model/quantized"

def main():
    print("=" * 60)
    print("Step 3: Quantizing ONNX models to INT8 for mobile deployment...")
    print("=" * 60)
    
    os.makedirs(QUANT_DIR, exist_ok=True)
    
    enc_in = os.path.join(ONNX_DIR, "encoder_model.onnx")
    enc_out = os.path.join(QUANT_DIR, "encoder_model_int8.onnx")
    print(f"Quantizing Encoder: {enc_in} -> {enc_out}")
    quantize_dynamic(
        model_input=enc_in,
        model_output=enc_out,
        weight_type=QuantType.QInt8,
    )
    print(f"Encoder quantized! Size: {os.path.getsize(enc_out) / (1024*1024):.2f} MB")

    dec_in = os.path.join(ONNX_DIR, "decoder_model.onnx")
    dec_out = os.path.join(QUANT_DIR, "decoder_model_int8.onnx")
    print(f"Quantizing Decoder: {dec_in} -> {dec_out}")
    quantize_dynamic(
        model_input=dec_in,
        model_output=dec_out,
        weight_type=QuantType.QInt8,
    )
    print(f"Decoder quantized! Size: {os.path.getsize(dec_out) / (1024*1024):.2f} MB")
    print("Quantization complete!")

if __name__ == "__main__":
    main()
