import os


def rm_files(directory):
    # Get list of all files in the directory
    files = os.listdir(directory)
    # Iterate over the files and remove each one
    for file in files:
        file_path = os.path.join(directory, file)
        # Check if the path is a file (not a directory)
        if os.path.isfile(file_path):
            os.remove(file_path)
