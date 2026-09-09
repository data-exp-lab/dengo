/* Pure-C++ benchmark of dengo's actual generated solver entry points --
 * primordial_setup_data()/BE_chem_solve() -- called directly against the
 * generated primordial_solver.C/BE_chem_solve.C, exactly mirroring the
 * adaptive-dt loop Solver.step_inplace()/_advance() runs in
 * cython_solver_run.pyx.template, but with no Cython/Python anywhere in
 * the timed path -- the same C-level entry points a real HPC code
 * embedding dengo directly (not through Python) would call.
 *
 * Usage: ./dengo_bench <dims> <n_timed>
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>

#include "primordial_solver.h"

#define MH 1.67e-24
#define KBOLTZ 1.3806504e-16
#define G_GRAV 6.674e-8

/* sorted species order, matching SPECIES_NAMES in the generated
   .pyx (Python's sorted() over the species name strings) */
enum { I_H2_1 = 0, I_H2_2, I_H_1, I_H_2, I_H_m0, I_He_1, I_He_2, I_He_3, I_de, I_ge };

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

int main(int argc, char *argv[]) {
    int dims = argc > 1 ? atoi(argv[1]) : 100000;
    int n_timed = argc > 2 ? atoi(argv[2]) : 20;
    int n = NSPECIES;

    double t0 = now();
    primordial_data *data = primordial_setup_data("primordial_tables.bin", dims);
    double t1 = now();
    fprintf(stdout, "one-time: dengo primordial_setup_data (tables.bin read): %.4f s\n", t1 - t0);

    /* --- per-dims: allocate the same scratch buffers Solver.__cinit__ does --- */
    double t_handle0 = now();
    double *input = (double *) malloc(dims * n * sizeof(double));
    double *prev  = (double *) malloc(dims * n * sizeof(double));
    double *scale = (double *) malloc(dims * n * sizeof(double));
    double *atol  = (double *) malloc(dims * n * sizeof(double));
    double *rtol  = (double *) malloc(dims * n * sizeof(double));
    double *u0    = (double *) malloc(dims * n * sizeof(double));
    double *s     = (double *) malloc(dims * n * sizeof(double));
    double *gu    = (double *) malloc(dims * n * sizeof(double));
    double *Ju    = (double *) malloc(dims * n * n * sizeof(double));
    double t_handle1 = now();

    /* --- same synthetic state as the grackle side: n=1e13 cm^-3,
     * T=1500K, 10% molecular by number --- */
    double NH = 1.0e13, T0 = 1500.0;
    double ge0 = 1.5 * KBOLTZ * T0 / MH;
    for (int i = 0; i < dims; i++) {
        int j = i * n;
        input[j + I_H_1]  = NH * 0.4;
        input[j + I_H_2]  = NH * 1e-6;
        input[j + I_He_1] = NH * 0.24 / 4;
        input[j + I_He_2] = NH * 1e-8;
        input[j + I_He_3] = NH * 1e-10;
        input[j + I_H_m0] = NH * 1e-10;
        input[j + I_H2_1] = NH * 0.1;
        input[j + I_H2_2] = NH * 1e-9;
        input[j + I_de]   = NH * 1e-6;
        input[j + I_ge]   = ge0;
    }

    double rho = NH * MH;
    double t_ff = sqrt(3.0 * M_PI / (32.0 * G_GRAV * rho));
    double dtf = 1.0e-3 * t_ff;
    double reltol = 1.0e-5;
    double floor_value = 1.0e-20;
    int niter = 200;

    auto run_one_step_inplace = [&]() -> bool {
        ensure_electron_consistency(input, dims, n);
        for (int i = 0; i < dims * n; i++) {
            atol[i] = floor_value * reltol;
            rtol[i] = reltol;
            scale[i] = fmax(fabs(input[i]), floor_value);
            prev[i] = input[i];
        }

        double dt = dtf / niter;
        double ttot = 0.0;
        int it = 0;
        while (it < niter && ttot < dtf) {
            int rv = BE_chem_solve(calculate_rhs_primordial, calculate_jacobian_primordial,
                                    input, dt, rtol, atol, dims, n, scale,
                                    (void *) data, u0, s, gu, Ju);
            for (int i = 0; i < dims * n; i++) {
                if (input[i] < 0) { rv = 1; break; }
            }
            if (rv == 0) {
                ttot += dt;
                for (int i = 0; i < dims * n; i++) {
                    if (input[i] < floor_value) input[i] = floor_value;
                    prev[i] = input[i];
                    scale[i] = fmax(fabs(input[i]), floor_value);
                }
                if (ttot < dtf) dt = fmin(dt * 2.0, dtf - ttot);
                it += 1;
            } else {
                dt /= 2.0;
                for (int i = 0; i < dims * n; i++) {
                    input[i] = prev[i];
                    scale[i] = fmax(fabs(prev[i]), floor_value);
                }
                if (dt < 1.0e-15 * dtf) break;
            }
        }
        return ttot >= dtf;
    };

    if (!run_one_step_inplace()) {
        fprintf(stderr, "warmup step did not converge\n");
        return EXIT_FAILURE;
    }

    double tt0 = now();
    for (int step = 0; step < n_timed; step++) {
        if (!run_one_step_inplace()) {
            fprintf(stderr, "step %d did not converge\n", step);
            return EXIT_FAILURE;
        }
    }
    double tt1 = now();
    double per_call = (tt1 - tt0) / n_timed;

    fprintf(stdout, "dims=%d handle_setup=%.6f s  us/call=%.3f  us/call/cell=%.5f\n",
            dims, t_handle1 - t_handle0, 1e6 * per_call, 1e6 * per_call / dims);

    primordial_free_data(data);
    return EXIT_SUCCESS;
}
