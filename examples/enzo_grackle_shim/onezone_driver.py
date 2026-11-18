"""Enzo's ProblemType 63 one-zone free-fall step (Grid_SolveOneZoneFreefall.C,
Grid_ComputeOneZoneCollapseFactor.C, Grid_ComputeTimeStep.C), driven through
gracklepy. Run as-is for real Grackle, or with LD_PRELOAD=<dengo libgrackle.so>
to drive dengo-grackle through the same (unmodified) Python API.

    python onezone_driver.py OUT_TRACK.json "$(cat INIT.json)"

INIT.json: initial fields in code units, e.g. from
`freefall_tracks.py init RUN_DIR INIT.json` on an Enzo run of
OneZoneFreefall_Primordial.enzo. Units and parameters match that file."""
import json, sys
import numpy as np
from gracklepy import chemistry_data, FluidContainer

out = sys.argv[1]
init = json.loads(sys.argv[2])          # initial fields (code units), from Enzo's DD0000
G = 1.394833e-3
FF = np.sqrt(32 * G / (3 * np.pi))
gamma = float(np.float32(5.0 / 3.0))

c = chemistry_data()
c.use_grackle = 1
c.with_radiative_cooling = 1
c.primordial_chemistry = 2
c.metal_cooling = 0
c.UVbackground = 0
c.three_body_rate = 4
c.cie_cooling = 1
c.h2_optical_depth_approximation = 1
c.cmb_temperature_floor = 0
c.Gamma = gamma
c.grackle_data_file = b""
c.comoving_coordinates = 0
c.a_units = 1.0
c.a_value = 1.0
c.density_units = 1.67e-24
c.length_units = 3.0857e18
c.time_units = 3.1557e13
c.set_velocity_units()
assert c.initialize() == 1

fc = FluidContainer(c, 1)
for k, v in init.items():
    fc[k][:] = v
names = ["HI_density", "HII_density", "HeI_density", "HeII_density", "HeIII_density",
         "HM_density", "H2I_density", "H2II_density", "e_density"]

def pressure():
    fc.calculate_temperature()
    ntot = (fc["HI_density"] + fc["HII_density"] + fc["HM_density"] + fc["e_density"]
            + (fc["H2I_density"] + fc["H2II_density"]) / 2
            + (fc["HeI_density"] + fc["HeII_density"] + fc["HeIII_density"]) / 4)
    return float(ntot[0] * fc["temperature"][0])   # proportional to P; only ratios matter

def force_factor(hist):
    if len(hist) < 2:
        return 0.0
    (r0, p0), (r1, p1) = hist[0], hist[1]
    ge = np.log10(p0 / p1) / np.log10(r0 / r1)
    if len(hist) > 2:
        r2, p2 = hist[2]
        ge += 0.5 * (np.log10(p1 / p2) / np.log10(r1 / r2) - ge)
    ge = min(ge, 4.0 / 3.0)
    if ge < 0.83:
        f = 0.0
    elif ge < 1.0:
        f = 0.6 + 2.5 * (ge - 1) - 6.0 * (ge - 1.0) ** 2
    else:
        f = 1.0 + 0.2 * (ge - 4.0 / 3.0) - 2.9 * (ge - 4.0 / 3.0) ** 2
    return min(max(f, 0.0), 0.95)

hist = []
track = []
t = 0.0
for cycle in range(5000):
    rho = float(fc["density"][0])
    fc.calculate_temperature()
    track.append((t, rho * 1.67e-24 / 1.6726e-24 * 0.76, float(fc["temperature"][0]),
                  float(fc["H2I_density"][0] / rho), float(fc["e_density"][0] / rho)))
    if track[-1][1] > 1e17:
        break
    f_dt = force_factor(hist)
    dt = 1e-2 * np.sqrt(3 * np.pi / (32 * G * rho * (1 - f_dt)))
    hist.insert(0, (rho, pressure()))
    hist = hist[:3]
    f = force_factor(hist)
    new = (rho ** -0.5 - 0.5 * FF * dt * np.sqrt(1 - f)) ** -2.0
    ratio = new / rho
    fc["internal_energy"][:] += (gamma - 1) * fc["internal_energy"] * FF * np.sqrt(rho) * dt
    fc["density"][:] = new
    for k in names:
        fc[k][:] *= ratio
    fc.solve_chemistry(dt)
    t += dt
json.dump(track, open(out, "w"))
print("cycles", len(track), "final nH %.3g T %.0f" % (track[-1][1], track[-1][2]))
