import argparse, time, csv, numpy as np, cv2
parser = argparse.ArgumentParser()
parser.add_argument("--sizes", nargs="+", type=int, default=[512,1024])
parser.add_argument("--iters", type=int, default=8)
parser.add_argument("--out", default="image_results.csv")
parser.add_argument("--image", default=None)
args = parser.parse_args()

try:
    import cupy as cp
    import cupyx.scipy.ndimage as ndi
    has_cupy = True
except Exception as e:
    print("CuPy not available:", e)
    has_cupy = False

with open(args.out,"w",newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["backend","size","run","compute_ms","total_ms"])
    for size in args.sizes:
        if args.image:
            im = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
            im = cv2.resize(im, (size,size))
        else:
            im = (np.random.rand(size,size)*255).astype(np.uint8)
        # CPU warmup + runs
        for run in range(args.iters+2):
            if run < 2:
                _ = cv2.GaussianBlur(im,(5,5),1.0); continue
            t0=time.time()
            g = cv2.GaussianBlur(im,(5,5),1.0)
            sx = cv2.Sobel(g,cv2.CV_32F,1,0,ksize=3)
            sy = cv2.Sobel(g,cv2.CV_32F,0,1,ksize=3)
            mag = cv2.magnitude(sx,sy)
            t1=time.time()
            writer.writerow(["cpu_opencv", size, run-2, (t1-t0)*1000.0, (t1-t0)*1000.0])
        # GPU with CuPy
        if has_cupy:
            for run in range(args.iters+2):
                if run < 2:
                    _ = ndi.gaussian_filter(cp.asarray(im), sigma=1.0); cp.cuda.Stream.null.synchronize(); continue
                t0=time.time()
                d = cp.asarray(im)
                g = ndi.gaussian_filter(d, sigma=1.0)
                kx = cp.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=cp.float32)
                ky = cp.array([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=cp.float32)
                sx = ndi.convolve(g, kx)
                sy = ndi.convolve(g, ky)
                mag = cp.sqrt(sx*sx + sy*sy)
                cp.cuda.Stream.null.synchronize()
                t1=time.time()
                writer.writerow(["gpu_cupy", size, run-2, (t1-t0)*1000.0, (t1-t0)*1000.0])
print("Wrote", args.out)
