import os
import json
import zipfile
from sentencepiece import sentencepiece_model_pb2

BASE_DIR = r"E:\vscode\vibeCode\program\stt-tuning\IndicTrans2-SimpleHindi"
QUANT_DIR = os.path.join(BASE_DIR, "onnx_model", "quantized")
SNAPSHOT_DIR = r"C:\Users\priyanshu\.cache\huggingface\hub\models--ai4bharat--indictrans2-en-indic-dist-200M\snapshots\2cfd0a20057756d756bf3e53528f492afc52f91a"
PACK_DIR = os.path.join(BASE_DIR, "pack")
os.makedirs(PACK_DIR, exist_ok=True)

# 1. Generate src_vocab.tsv
print("Generating src_vocab.tsv...")
model_proto = sentencepiece_model_pb2.ModelProto()
with open(os.path.join(SNAPSHOT_DIR, "model.SRC"), "rb") as f:
    model_proto.ParseFromString(f.read())
piece_scores = {p.piece: p.score for p in model_proto.pieces}

with open(os.path.join(SNAPSHOT_DIR, "dict.SRC.json"), "r", encoding="utf-8") as f:
    dict_src = json.load(f)

src_tsv_path = os.path.join(PACK_DIR, "src_vocab.tsv")
with open(src_tsv_path, "w", encoding="utf-8") as f:
    for piece, token_id in dict_src.items():
        score = piece_scores.get(piece, -100.0)
        f.write(f"{token_id}\t{score}\t{piece}\n")

# 2. Generate tgt_vocab.txt
print("Generating tgt_vocab.txt...")
with open(os.path.join(SNAPSHOT_DIR, "dict.TGT.json"), "r", encoding="utf-8") as f:
    dict_tgt = json.load(f)

tgt_pieces = [""] * len(dict_tgt)
for piece, idx in dict_tgt.items():
    if idx < len(tgt_pieces):
        tgt_pieces[idx] = piece

tgt_txt_path = os.path.join(PACK_DIR, "tgt_vocab.txt")
with open(tgt_txt_path, "w", encoding="utf-8") as f:
    for p in tgt_pieces:
        f.write(f"{p}\n")

print(f"SRC TSV size: {os.path.getsize(src_tsv_path)} bytes")
print(f"TGT TXT size: {os.path.getsize(tgt_txt_path)} bytes")

# 3. Create ZIP archive
zip_path = os.path.join(BASE_DIR, "indictrans2_simple_hindi.zip")
print(f"Creating {zip_path}...")
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
    z.write(os.path.join(QUANT_DIR, "encoder_model_int8.onnx"), "encoder_model.onnx")
    z.write(os.path.join(QUANT_DIR, "decoder_model_int8.onnx"), "decoder_model.onnx")
    z.write(src_tsv_path, "src_vocab.tsv")
    z.write(tgt_txt_path, "tgt_vocab.txt")

print(f"Zip created! Size: {os.path.getsize(zip_path) / 1024 / 1024:.2f} MB")
