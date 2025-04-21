import torch

if __name__ == "__main__":
    print("Cuda support:", torch.cuda.is_available(),":", torch.cuda.device_count(), "devices")