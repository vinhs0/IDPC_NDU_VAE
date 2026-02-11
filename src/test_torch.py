import torch
print("Torch version:", torch.__version__)
print("CUDA version built with PyTorch:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
