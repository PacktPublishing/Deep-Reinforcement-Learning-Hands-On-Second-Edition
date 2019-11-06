# reproduce of PyTorch performance regression with tensors concatenation
import torch
import time
import numpy as np

ITERS = 100
BATCH = 128
SHAPE = (4, 84, 84)


def test_1(device):
    ts = time.time()
    for _ in range(ITERS):
        batch = []
        for _ in range(BATCH):
            batch.append(np.zeros(SHAPE, dtype=np.float32))
        batch_t = torch.FloatTensor(batch).to(device)
        torch.cuda.synchronize()
    dt = time.time() - ts
    print("1: Done %d iters in %.3f seconds = %.3f it/s" % (
        ITERS, dt, ITERS/dt
    ))


def test_2(device):
    ts = time.time()
    for _ in range(ITERS):
        batch = []
        for _ in range(BATCH):
            batch.append(torch.FloatTensor(np.zeros(SHAPE, dtype=np.float32)))
        batch_t = torch.stack(batch).to(device)
        torch.cuda.synchronize()
    dt = time.time() - ts
    print("2: Done %d iters in %.3f seconds = %.3f it/s" % (
        ITERS, dt, ITERS/dt
    ))


def test_0(device):
    ts = time.time()
    for _ in range(ITERS):
        batch = []
        for _ in range(BATCH):
            batch.append(np.zeros(SHAPE, dtype=np.float32))
        batch_t = torch.FloatTensor(np.array(batch, copy=False)).to(device)
        torch.cuda.synchronize()
    dt = time.time() - ts
    print("0: Done %d iters in %.3f seconds = %.3f it/s" % (
        ITERS, dt, ITERS/dt
    ))


# GTX 1080Ti, Ubuntu, Drivers 430.26

# PyTorch 1.3, CUDA 10.2
# 0: Done 100 iters in 2.980 seconds = 33.562 it/s
# 1: Done 100 iters in 28.654 seconds = 3.490 it/s
# 2: Done 100 iters in 0.409 seconds = 244.373 it/s

# 0: Done 100 iters in 0.369 seconds = 271.093 it/s
# 1: Done 100 iters in 28.663 seconds = 3.489 it/s
# 2: Done 100 iters in 0.410 seconds = 243.695 it/s

# PyTorch 0.4.1, CUDA 9.2
# Done 100 iters in 30.947 seconds = 3.231 it/s
# Done 100 iters in 0.497 seconds = 201.295 it/s

# In fact, that's a known bug: https://github.com/pytorch/pytorch/issues/13918

if __name__ == "__main__":
    device = torch.device("cuda")

    for _ in range(2):
        test_0(device)
        test_1(device)
        test_2(device)
