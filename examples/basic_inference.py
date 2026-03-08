"""Example: LFM2 Text Generation.

Shows how to initialize the LFM2-350M model and generate text.
"""

import torch
from lfm2.model.backbone import LFM2Model
from lfm2.model.configs import LFM2_350M

def main():
    # 1. Initialize Model from Config
    config = LFM2_350M
    model = LFM2Model(config)
    print(f"Initialized LFM2-350M with {model.count_parameters():,} parameters.")

    # 2. Dummy Input
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # "The liquid foundation model is" encoded
    prompt_ids = torch.tensor([[1, 2, 3, 4, 5]], device=device)
    
    # 3. Simple Greedy Generation
    model.eval()
    with torch.no_grad():
        for _ in range(10):
            output = model(prompt_ids)
            next_token = torch.argmax(output.logits[:, -1, :], dim=-1, keepdim=True)
            prompt_ids = torch.cat([prompt_ids, next_token], dim=1)
            
    print(f"Generated token sequence: {prompt_ids.tolist()}")

if __name__ == "__main__":
    main()
