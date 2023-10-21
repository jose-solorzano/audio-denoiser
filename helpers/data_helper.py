import io
import logging
import os
import glob
from typing import Any, Callable
import pickle

import torch

from config import CACHE_DIR
_cache_dir: str = CACHE_DIR


class _CPUUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


def string_hash(s: str):
    hash_value = 0
    for c in s:
        hash_value = hash_value * 31 + ord(c)
    return hash_value % 1000000000


def save_model(model: Any, path: str):
    file_dir = os.path.dirname(path)
    os.makedirs(file_dir, exist_ok=True)
    with open(path, 'wb') as fd:
        pickle.dump(model, fd)


def load_model(path: str) -> Any:
    with open(path, 'rb') as fd:
        return _CPUUnpickler(fd).load()


def get_object_from_cache(filename: str, create_function: Callable, **params) -> Any:
    file_path: str = os.path.join(_cache_dir, filename)
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            obj: Any = pickle.load(f)
    else:
        logging.warning(f'Did not find {filename} in cache.')
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        obj: Any = create_function(**params)
        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)
    return obj


def get_file_paths(directory: str, file_pattern: str):
    pattern = os.path.join(directory, '**', file_pattern)
    return glob.glob(pattern, recursive=True)
