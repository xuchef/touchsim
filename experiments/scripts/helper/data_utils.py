import lzma
import pickle


def save_pkl(obj, path):
    with lzma.open(path+".pkl.xz", "wb") as f:
        pickle.dump(obj, f)


def load_pkl(path):
    with lzma.open(path, "rb") as f:
        data = pickle.load(f)
    return data