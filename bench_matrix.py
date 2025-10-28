import argparse, time, csv, numpy as np
parser = argparse.ArgumentParser()
parser.add_argument("--sizes", nargs="+", type=int, default=[256,512,1024])
parser.add_argument("--iters", type=int, default=8)
parser.add_argument("--out", type=str, default="matrix_results.csv")
args = parser.parse_args()

try:
    import cupy as cp
    has_cupy = True
except Exception as e:
    print("CuPy not available:", e)
    has_cupy = False

def time_numpy(N, iters):
    results=[]
    for run in range(iters+2):
        A = np.random.rand(N,N).astype(np.float32)
        B = np.random.rand(N,N).astype(np.float32)
        if run < 2: 
            _ = A.dot(B); continue
        t0 = time.time()
        C = A.dot(B)
        t1 = time.time()
        results.append(( "numpy", N, run-2, 0.0, (t1-t0)*1000.0, 0.0, (t1-t0)*1000.0))
    return results

def time_cupy(N, iters):
    if not has_cupy: return []
    results=[]
    for run in range(iters+2):
        Ah = np.random.rand(N,N).astype(np.float32)
        Bh = np.random.rand(N,N).astype(np.float32)
        if run < 2:
            A=cp.asarray(Ah); B=cp.asarray(Bh); _ = A.dot(B); cp.cuda.Stream.null.synchronize(); continue
        t0 = time.time()
        A = cp.asarray(Ah)
        t1 = time.time(); h2d=(t1-t0)*1000.0
        t0c=time.time()
        C = A.dot(cp.asarray(Bh))
        cp.cuda.Stream.null.synchronize()
        t1c=time.time(); compute=(t1c-t0c)*1000.0
        t0d=time.time()
        _ = cp.asnumpy(C)
        t1d=time.time(); d2h=(t1d-t0d)*1000.0
        results.append(("cupy", N, run-2, h2d, compute, d2h, h2d+compute+d2h))
    return results

with open(args.out,"w",newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["backend","size","run","transfer_h2d_ms","compute_ms","transfer_d2h_ms","total_ms"])
    for N in args.sizes:
        print("Matrix N=",N)
        for row in time_numpy(N,args.iters): writer.writerow(row)
        for row in time_cupy(N,args.iters): writer.writerow(row)
print("Wrote", args.out)
