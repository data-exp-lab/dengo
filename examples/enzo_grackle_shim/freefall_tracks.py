"""Collapse tracks (T and H2 fraction vs n_H) from one-zone free-fall runs.

    # an Enzo ProblemType 63 run directory -> track JSON
    python freefall_tracks.py extract RUN_DIR OUT.json
    # the run's initial state (cell 1,1 of DD0000) as onezone_driver.py input
    python freefall_tracks.py init RUN_DIR OUT.json
    # tabulate several tracks (from `extract` or onezone_driver.py) side by side
    python freefall_tracks.py compare NAME=TRACK.json [NAME=TRACK.json ...]

A track is a list of (t, n_H, T, f_H2, x_e), with f_H2 and x_e as mass
fractions of the total density.
"""
import glob
import json
import os
import re
import sys

import numpy as np


def extract(run_dir, out, cell=(1, 1)):
    import h5py
    rows = []
    for p in sorted(glob.glob(os.path.join(run_dir, "DD????", "DD????"))):
        t = float(re.search(r"InitialTime\s*=\s*(\S+)", open(p).read()).group(1))
        with h5py.File(p + ".cpu0000", "r") as f:
            g = f["Grid00000001"]
            rho = np.array(g["Density"])[cell]
            rows.append((t,
                         rho * 1.67e-24 / 1.6726e-24 * 0.76,  # DensityUnits = 1.67e-24
                         float(np.array(g["Temperature"])[cell]),
                         float(np.array(g["H2I_Density"])[cell] / rho),
                         float(np.array(g["Electron_Density"])[cell] / rho)))
    with open(out, "w") as f:
        json.dump(rows, f)
    print("%d dumps -> %s" % (len(rows), out))


ENZO_TO_GRACKLE = {
    "Density": "density", "HI_Density": "HI_density", "HII_Density": "HII_density",
    "HeI_Density": "HeI_density", "HeII_Density": "HeII_density",
    "HeIII_Density": "HeIII_density", "HM_Density": "HM_density",
    "H2I_Density": "H2I_density", "H2II_Density": "H2II_density",
    "Electron_Density": "e_density", "GasEnergy": "internal_energy",
}


def initial_state(run_dir, out, cell=(1, 1)):
    import h5py
    with h5py.File(os.path.join(run_dir, "DD0000", "DD0000.cpu0000"), "r") as f:
        g = f["Grid00000001"]
        init = {v: float(np.array(g[k])[cell]) for k, v in ENZO_TO_GRACKLE.items()}
    with open(out, "w") as f:
        json.dump(init, f)
    print(init)


def compare(named):
    tracks = {}
    for item in named:
        name, path = item.split("=", 1)
        tracks[name] = np.array(json.load(open(path)))

    def at(a, n, col):
        return np.interp(np.log10(n), np.log10(a[:, 1]), a[:, col])

    print("     n_H  | " + " | ".join("%-20s" % k for k in tracks))
    print("          | " + " | ".join("%-20s" % "T [K]      f_H2" for _ in tracks))
    for ln in range(0, 18):
        n = 10.0 ** ln
        cells = []
        for a in tracks.values():
            if n > a[:, 1].max():
                cells.append("%-20s" % "")
            else:
                cells.append("%7.0f  %10.3e " % (at(a, n, 2), at(a, n, 3)))
        print("%9.0e | " % n + " | ".join(cells))


if __name__ == "__main__":
    if sys.argv[1] == "extract":
        extract(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "init":
        initial_state(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "compare":
        compare(sys.argv[2:])
    else:
        sys.exit(__doc__)
