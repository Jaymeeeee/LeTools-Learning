"""The lossless background observation recorder used in round-three tests."""
import atexit
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import imageio
import numpy as np

_installed = False


def install_async_recorder():
    global _installed
    if _installed:
        return
    _installed = True
    original_write, original_read = imageio.imwrite, imageio.imread
    pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix='evidence-png')
    pending = deque()

    def flush():
        while pending:
            pending.popleft().result()

    def write(path, pixels, **kwargs):
        if 'temp_frames_' not in str(path) or Path(path).suffix.lower() != '.png':
            return original_write(path, pixels, **kwargs)
        while pending and pending[0].done():
            pending.popleft().result()
        if len(pending) >= 64:
            pending.popleft().result()
        kwargs['compress_level'] = 1
        pending.append(pool.submit(original_write, path, np.asarray(pixels).copy(), **kwargs))

    def read(*args, **kwargs):
        flush()
        return original_read(*args, **kwargs)

    def close():
        flush()
        pool.shutdown(wait=True)

    imageio.imwrite, imageio.imread = write, read
    atexit.register(close)
