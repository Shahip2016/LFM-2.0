import torch
def clip_gradients(model, max_norm):
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
