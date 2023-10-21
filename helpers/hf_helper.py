import codecs
import os

from config import HF_KEY_FILE


def get_hf_token() -> str:
    hf_key_file = os.path.join(os.path.dirname(__file__), '..', 'cred', HF_KEY_FILE)
    with codecs.open(hf_key_file, 'r', 'utf-8') as fd:
        return fd.read().strip()
