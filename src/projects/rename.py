import os

folder_path = "/home/clemi/catkin_ws/src/projects/images/bedside_table/depth"  # Replace with the actual path to your folder

# Get a list of all files in the folder
files = os.listdir(folder_path)

# Rename the files with a running index
for i, file_name in enumerate(files):
    # Construct the full path to the file
    file_path = os.path.join(folder_path, file_name)
    
    # Generate the new file name with the running index
    new_file_name = f"{i+1}.jpg"  # Replace ".jpg" with the desired file extension
    
    # Construct the full path to the new file
    new_file_path = os.path.join(folder_path, new_file_name)
    
    # Rename the file
    os.rename(file_path, new_file_path)
    print(f"Renamed {file_name} to {new_file_name}")