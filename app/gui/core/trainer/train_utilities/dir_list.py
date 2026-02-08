import os
import pandas as pd
# Directory to search for subfolders
directory = r"C:\Users\pod44433\Downloads\february\splits"

# List for file names
file_names = []
folder_names  = []

# Traverse all subfolders and collect file names
for folder_name in os.listdir(directory):
    folder_names.append(folder_name)

# Create a DataFrame with the file names
folder_names = sorted(folder_names)
df = pd.DataFrame(folder_names, columns=["Folder Names"])

# Save the DataFrame to an Excel file
excel_file = "subfolder_names.xlsx"
df.to_excel(excel_file, index=False)

print(f"The file names have been saved to '{excel_file}'.")
