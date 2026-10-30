import argparse, numpy as np
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=str, required=True)
    ap.add_argument('--n', type=int, default=8)
    args = ap.parse_args()
    root = Path(args.root)
    (root/'train').mkdir(parents=True, exist_ok=True)
    (root/'val').mkdir(parents=True, exist_ok=True)
    # create tiny 4‑channel blobs with 3 nested spheres as classes 1..3
    def make_case(seed):
        rng = np.random.default_rng(seed)
        D=64; H=64; W=64; C=4
        grid = np.indices((D,H,W)).astype('float32')
        center = np.array([D/2,H/2,W/2]) + rng.normal(0,2,3)
        r1, r2, r3 = 20+rng.integers(-3,3), 12+rng.integers(-2,2), 6+rng.integers(-1,1)
        dist = np.sqrt(((grid[0]-center[0])**2 + (grid[1]-center[1])**2 + (grid[2]-center[2])**2))
        y = np.zeros((1,D,H,W), dtype='int16')
        y[(0,)+np.where(dist<=r1)] = 1  # WT
        y[(0,)+np.where(dist<=r2)] = 2  # TC
        y[(0,)+np.where(dist<=r3)] = 3  # ET
        x = rng.normal(0,1,(C,D,H,W)).astype('float32')
        for c in range(C):
            x[c] += (c+1)*0.1*(y[0]>0)
        return x, y
    for i in range(args.n):
        x, y = make_case(i)
        split = 'val' if i%4==0 else 'train'
        np.save(root/split/f"case{i:03d}_x.npy", x)
        np.save(root/split/f"case{i:03d}_y.npy", y)
    print('Wrote synthetic cases to', root)

if __name__ == "__main__":
    main()
