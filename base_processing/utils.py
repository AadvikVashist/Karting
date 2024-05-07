import os

cwd = os.getcwd()


def get_base_file_path(file):
    return os.path.join(cwd, file)