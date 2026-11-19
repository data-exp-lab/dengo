"""Compare two Enzo run directories (dengo-grackle vs real Grackle) dump by dump."""
import glob, os, sys, re
import h5py, numpy as np

def dumps(run):
    out = []
    for p in sorted(glob.glob(os.path.join(run, "DD*", "DD????"))):
        t = float(re.search(r"InitialTime\s*=\s*(\S+)", open(p).read()).group(1))
        cyc = int(re.search(r"InitialCycleNumber\s*=\s*(\S+)", open(p).read()).group(1))
        with h5py.File(p + ".cpu0000", "r") as f:
            g = f["Grid00000001"]
            data = {k: np.array(g[k]) for k in g.keys()}
        out.append((os.path.basename(p), t, cyc, data))
    return out

a, b = dumps(sys.argv[1]), dumps(sys.argv[2])
fields = sys.argv[3].split(",") if len(sys.argv) > 3 else ["Density", "Temperature", "H2I_Density", "Electron_Density", "Cooling_Time"]
print("dumps: %d vs %d" % (len(a), len(b)))
for (na, ta, ca, da), (nb, tb, cb, db) in zip(a, b):
    line = "%s t=%.5g/%.5g cyc=%d/%d" % (na, ta, tb, ca, cb)
    for k in fields:
        if k in da and k in db:
            x, y = da[k].astype(float), db[k].astype(float)
            rel = np.abs(x - y) / np.maximum(np.abs(y), 1e-300)
            line += " | %s med %.3g max %.3g" % (k, np.median(rel), rel.max())
    print(line)
