import numpy as np
from pprint import pprint
import os

def inspect_npz_file(file_path):
    data = np.load(file_path)
    print("Keys in the npz file:")
    print(data.keys())
    
    for key in data.keys():
        print(f"\nData in '{key}':")
        print(data[key])
    
    pprint(data)

    # convert to dict
    data = dict(data)
    
    if "mean_array" in data.keys():
        print("\nMean:")
        print(data["mean_array"])
        print(len(data["mean_array"]))
    
    if "std_array" in data.keys():
        print("\nStd:")
        print(data["mean_array"])
        print(len(data["std_array"]))




if __name__ == "__main__":
    file_path = os.path.join(os.getcwd(), "mean_std.npz")
    inspect_npz_file(file_path=file_path)