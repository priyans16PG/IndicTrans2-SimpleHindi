import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from safetensors.torch import load_file

MERGED_DIR = "./merged_model"
ONNX_DIR = "./onnx_model"

class IndicTransEncoderWrapper(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state

def main():
    print("=" * 60)
    print("Step 2: Exporting IndicTrans2 Encoder & Decoder to ONNX...")
    print("=" * 60)
    
    os.makedirs(ONNX_DIR, exist_ok=True)
    
    # 1. Load model with safetensors
    print("Loading merged model...")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MERGED_DIR,
        trust_remote_code=True,
        dtype=torch.float32,
    )
    model.eval()

    # 2. Export Encoder
    print("Exporting Encoder to ONNX...")
    encoder_wrapper = IndicTransEncoderWrapper(model.model.encoder)
    encoder_wrapper.eval()

    dummy_input_ids = torch.randint(0, 1000, (1, 16), dtype=torch.long)
    dummy_attention_mask = torch.ones((1, 16), dtype=torch.long)

    encoder_path = os.path.join(ONNX_DIR, "encoder_model.onnx")
    torch.onnx.export(
        encoder_wrapper,
        (dummy_input_ids, dummy_attention_mask),
        encoder_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "last_hidden_state": {0: "batch_size", 1: "sequence_length"},
        }
    )
    print(f"Encoder exported to {encoder_path}")

    # 3. Export Decoder Step (Autoregressive step)
    class IndicTransDecoderWrapper(nn.Module):
        def __init__(self, decoder, lm_head):
            super().__init__()
            self.decoder = decoder
            self.lm_head = lm_head

        def forward(self, decoder_input_ids, encoder_hidden_states, encoder_attention_mask):
            decoder_out = self.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
            )
            logits = self.lm_head(decoder_out.last_hidden_state)
            return logits

    print("Exporting Decoder to ONNX...")
    decoder_wrapper = IndicTransDecoderWrapper(model.model.decoder, model.lm_head)
    decoder_wrapper.eval()

    dummy_decoder_input_ids = torch.randint(0, 1000, (1, 4), dtype=torch.long)
    dummy_encoder_hidden_states = torch.randn(1, 16, model.config.encoder_embed_dim, dtype=torch.float32)
    dummy_encoder_attention_mask = torch.ones((1, 16), dtype=torch.long)

    decoder_path = os.path.join(ONNX_DIR, "decoder_model.onnx")
    torch.onnx.export(
        decoder_wrapper,
        (dummy_decoder_input_ids, dummy_encoder_hidden_states, dummy_encoder_attention_mask),
        decoder_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["decoder_input_ids", "encoder_hidden_states", "encoder_attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "decoder_input_ids": {0: "batch_size", 1: "decoder_sequence_length"},
            "encoder_hidden_states": {0: "batch_size", 1: "encoder_sequence_length"},
            "encoder_attention_mask": {0: "batch_size", 1: "encoder_sequence_length"},
            "logits": {0: "batch_size", 1: "decoder_sequence_length"},
        }
    )
    print(f"Decoder exported to {decoder_path}")
    print("All ONNX models exported successfully!")

if __name__ == "__main__":
    main()
