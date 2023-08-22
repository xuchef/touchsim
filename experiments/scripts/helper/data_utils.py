import lzma
import pickle
import json


def add_ext(path, ext):
    if path[-len(ext):] != ext:
        return path + ext
    return path


def save_pkl(obj, path):
    path = add_ext(path, ".pkl.xz")
    with lzma.open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pkl(path):
    path = add_ext(path, ".pkl.xz")
    with lzma.open(path, "rb") as f:
        data = pickle.load(f)
    return data


def save_json(obj, path):
    path = add_ext(path, ".json")
    with open(path, "w") as f:
        json.dump(obj, f)


def load_json(path):
    path = add_ext(path, ".json")
    print(path)
    with open(path, "r") as f:
        data = json.load(f)
    return data