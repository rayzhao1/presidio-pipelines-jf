import sys
import os
from glob import glob
import numpy as np
from scipy import signal
from tqdm import tqdm

SAMPLING_RATE = 1000

if __name__ == '__main__':
    print("Starting...", flush=True)

    # Collect command-line arguments
    assert len(sys.argv) == 4, f'Expected 3, got {sys.argv}'
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    night_idx = int(sys.argv[3])

    arr = np.swapaxes(np.load(os.path.join(input_dir, f'output-n{night_idx}.npz'))['ieeg_data'], 0, 1)

    same_freq = True
    last_f, _ = signal.coherence(arr[0], arr[0], fs=SAMPLING_RATE)
    res = np.zeros((n, n, len(last_f)), dtype=np.float32)

    for i in tqdm(range(n)):
        for j in range(i, n):
            f, Cxy = signal.coherence(arr[i], arr[j], fs=SAMPLING_RATE)
            same_freq = same_freq and np.array_equal(f, last_f)
            res[i][j] = Cxy
            last_f = f
    assert same_freq == True
    print(res.shape)

    fn = os.path.join(output_dir, f'coherence-n{night_idx}')
    np.savez(fn, coherence=res, freq=last_f)
