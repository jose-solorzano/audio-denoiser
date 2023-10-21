import time


def new_exp_id() -> str:
    return str(round(time.time() / 60.0) - 28183202)
