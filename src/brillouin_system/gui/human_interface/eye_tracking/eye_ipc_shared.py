# eye_ipc_shared.py
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
import numpy as np

@dataclass
class ShmFrameSpec:
    name: str
    shape: tuple   # (H, W) or (H, W, C)
    dtype: str     # 'uint8', etc.
    slots: int

class ShmRing:
    """
    Simple fixed-size ring buffer in shared memory for frames.
    Each slot is a contiguous block sized by shape*dtype.
    """
    def __init__(self, spec: ShmFrameSpec, create: bool):
        self.spec = spec
        self.item_nbytes = int(np.prod(spec.shape)) * np.dtype(spec.dtype).itemsize
        self.total_nbytes = self.item_nbytes * spec.slots
        if create:
            self.shm = SharedMemory(create=True, size=self.total_nbytes, name=spec.name)
        else:
            self.shm = SharedMemory(name=spec.name, create=False)
        self.buf = self.shm.buf  # memoryview

    def write_slot(self, idx: int, arr: np.ndarray):
        assert arr.shape == self.spec.shape, f"Shape mismatch: arr {arr.shape}, spec {self.spec.shape}"
        start = idx * self.item_nbytes
        mv = memoryview(self.buf)[start:start + self.item_nbytes]
        mv[:] = arr.tobytes()

    def read_slot(self, idx: int) -> np.ndarray:
        assert 0 <= idx < self.spec.slots
        start = idx * self.item_nbytes
        mv = memoryview(self.buf)[start:start + self.item_nbytes]
        out = np.frombuffer(mv, dtype=self.spec.dtype).reshape(self.spec.shape)
        # NOTE: return a COPY so SM can be reused safely; if you want zero-copy, keep the view.
        return out.copy()

    def close(self):
        self.shm.close()

    def unlink(self):
        self.shm.unlink()
