import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create a tensor and move it to GPU
x = torch.rand(1000, 1000).to(device)
y = torch.rand(1000, 1000).to(device)

# Perform a matrix multiplication on GPU
z = torch.matmul(x, y)

print(x,y,z)
print("Computation successful on:", z.device)
