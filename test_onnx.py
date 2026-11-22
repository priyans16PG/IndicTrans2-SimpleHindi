import os
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor

QUANT_DIR = "./onnx_model/quantized"
MERGED_DIR = "./merged_model"

def main():
    print("=" * 60)
    print("Step 4: Testing ONNX INT8 Inference on CPU...")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indictrans2-en-indic-dist-200M", trust_remote_code=True)
    ip = IndicProcessor(inference=True)

    enc_session = ort.InferenceSession(os.path.join(QUANT_DIR, "encoder_model_int8.onnx"))
    dec_session = ort.InferenceSession(os.path.join(QUANT_DIR, "decoder_model_int8.onnx"))

    sentences = [
        "Don't worry about it, I will handle it.",
        "Where are you going right now?"
    ]

    preprocessed = ip.preprocess_batch(sentences, src_lang="eng_Latn", tgt_lang="hin_Deva")
    enc_inputs = tokenizer(preprocessed, padding=True, truncation=True, return_tensors="np")

    # Run encoder
    enc_outs = enc_session.run(
        None,
        {
            "input_ids": enc_inputs["input_ids"].astype(np.int64),
            "attention_mask": enc_inputs["attention_mask"].astype(np.int64)
        }
    )
    last_hidden_state = enc_outs[0]
    print(f"Encoder output shape: {last_hidden_state.shape}")

    # Run greedy decoding
    batch_size = len(sentences)
    # IndicTrans2 starts decoder with: [<s>, __hin_Deva__]
    tgt_lang_id = tokenizer.convert_tokens_to_ids("__hin_Deva__")  # token ID 3
    print(f"Target lang token '__hin_Deva__': {tgt_lang_id}")
    cur_ids = np.array([[tokenizer.bos_token_id, tgt_lang_id]] * batch_size, dtype=np.int64)

    for step in range(32):
        dec_outs = dec_session.run(
            None,
            {
                "decoder_input_ids": cur_ids,
                "encoder_hidden_states": last_hidden_state,
                "encoder_attention_mask": enc_inputs["attention_mask"].astype(np.int64)
            }
        )
        logits = dec_outs[0]  # [batch_size, seq_len, vocab_size]
        next_token = np.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        cur_ids = np.concatenate([cur_ids, next_token], axis=-1)
        if np.all(next_token.squeeze(-1) == tokenizer.eos_token_id):
            break

    decoded = tokenizer.batch_decode(cur_ids, skip_special_tokens=True)
    postprocessed = ip.postprocess_batch(decoded, lang="hin_Deva")

    for s, t in zip(sentences, postprocessed):
        print(f"EN: {s}")
        print(f"HI: {t}\n")

if __name__ == "__main__":
    main()
