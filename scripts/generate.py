"""LFM2 Inference Script.

Simple CLI for generating text using LFM2 models.
"""

import argparse
import torch
from lfm2.model.backbone import LFM2Model
from lfm2.model.configs import get_config

def generate(model_name: str, prompt: str, max_new_tokens: int = 50):
    config = get_config(model_name)
    model = LFM2Model(config)
    
    # [Tokenizer and real encoding would go here]
    # For now, using dummy tokens
    input_ids = torch.randint(0, config.vocab_size, (1, 10))
    
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids).logits
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
    print(f"Generated tokens: {input_ids.tolist()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="lfm2-350m")
    parser.add_argument("--prompt", type=str, default="Hello, LFM2!")
    args = parser.parse_args()
    
    generate(args.model, args.prompt)
