import torch

# Check if CUDA is available
if torch.cuda.is_available():
    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")

    # List all GPUs
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    # Select the second GPU (index 1)
    if num_gpus > 1:
        torch.cuda.set_device(1)
        print(f"\nSelected GPU: {torch.cuda.get_device_name(1)}")
    else:
        print("\nThere is no second GPU available.")
else:
    print("CUDA is not available on this system.")
