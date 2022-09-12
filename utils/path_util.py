import os


def look_up(path):
    if os.path.exists(path):
        return path

    upper = "." + path
    if os.path.exists(upper):
        # print("switch", path, "==>", upper)
        return upper

    return path
