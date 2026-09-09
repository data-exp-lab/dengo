/* Exposes dengo's generated primordial solver to JavaScript via a
 * small C API, compiled with Emscripten -- no Python, no Cython, in
 * the browser at all. Same idea as .grackle_compare/c_bench/
 * dengo_bench.cpp (calls primordial_setup_data()/BE_chem_solve()
 * directly), but as a reusable exported API instead of a throwaway
 * benchmark loop. See NOTES.md for the feasibility evaluation this was
 * built to answer (verified bit-for-bit identical to the native
 * Python/Cython solver on the same initial conditions).
 *
 * Hardcoded to the primordial network specifically for now (species
 * order below matches Python's SPECIES_NAMES, i.e. sorted species
 * names) -- see wasm/README.md for the plan to generate this file from
 * a Jinja template instead, the way cython_solver_run.pyx.template
 * does for the Cython wrapper, so any dengo network can target wasm.
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>

#include <emscripten/emscripten.h>

#include "primordial_solver.h"

#define MH 1.67e-24
#define KBOLTZ 1.3806504e-16

static primordial_data *g_data = NULL;
static int g_n = NSPECIES;
static double *g_input, *g_prev, *g_scale, *g_atol, *g_rtol, *g_u0, *g_s, *g_gu, *g_Ju;
static double *g_rhs_diag;

extern "C" {

EMSCRIPTEN_KEEPALIVE
int dengo_wasm_nspecies() { return NSPECIES; }

/* Comma-joined species names, in the exact order dengo_wasm_state_ptr()'s
   buffer uses -- one source of truth the JS side reads instead of
   independently hardcoding the same order (sorted species names, same
   as Python's SPECIES_NAMES/SPECIES_INDEX). */
EMSCRIPTEN_KEEPALIVE
const char *dengo_wasm_species_names() {
    return "H2_1,H2_2,H_1,H_2,H_m0,He_1,He_2,He_3,de,ge";
}

EMSCRIPTEN_KEEPALIVE
void dengo_wasm_init() {
    g_data = primordial_setup_data("primordial_tables.bin", 1);
    g_input = (double *) malloc(g_n * sizeof(double));
    g_prev  = (double *) malloc(g_n * sizeof(double));
    g_scale = (double *) malloc(g_n * sizeof(double));
    g_atol  = (double *) malloc(g_n * sizeof(double));
    g_rtol  = (double *) malloc(g_n * sizeof(double));
    g_u0    = (double *) malloc(g_n * sizeof(double));
    g_s     = (double *) malloc(g_n * sizeof(double));
    g_gu    = (double *) malloc(g_n * sizeof(double));
    g_Ju    = (double *) malloc(g_n * g_n * sizeof(double));
    g_rhs_diag = (double *) malloc(g_n * sizeof(double));
}

/* Direct access to the state buffer -- JS writes/reads this array
   in place (via a Float64Array view over WASM memory), same spirit
   as Solver.state's zero-copy view in the Python API. */
EMSCRIPTEN_KEEPALIVE
double *dengo_wasm_state_ptr() { return g_input; }

EMSCRIPTEN_KEEPALIVE
double dengo_wasm_temperature() { return g_data->Ts[0]; }

/* d(state)/dt at the *current* g_input, without advancing anything --
   the wasm counterpart of Solver.evaluate_rhs(). Used by the JS driver
   to estimate a cooling-time-limited dt for the constant-density mode,
   the same recipe run_dengo.py/the notebook use (safety_factor *
   |ge / d(ge)/dt|) -- self-contained: calculate_rhs_primordial()
   computes its own temperature/rate-table interpolation from g_input,
   it doesn't need dengo_wasm_step() to have run first. */
EMSCRIPTEN_KEEPALIVE
double *dengo_wasm_rhs_ptr() {
    calculate_rhs_primordial(g_input, g_rhs_diag, 1, g_n, (void *) g_data);
    return g_rhs_diag;
}

/* Evolve g_input forward by dtf seconds, in place. Returns 1 if
   converged (reached dtf), 0 otherwise. Mirrors _advance() in
   cython_solver_run.pyx.template exactly (including the growth=2.0
   step-size fix -- see NOTES.md). */
EMSCRIPTEN_KEEPALIVE
int dengo_wasm_step(double dtf, int niter, double reltol) {
    double floor_value = 1.0e-20;
    ensure_electron_consistency(g_input, 1, g_n);
    for (int i = 0; i < g_n; i++) {
        g_atol[i] = floor_value * reltol;
        g_rtol[i] = reltol;
        g_scale[i] = fmax(fabs(g_input[i]), floor_value);
        g_prev[i] = g_input[i];
    }

    double dt = dtf / niter;
    double ttot = 0.0;
    int it = 0;
    while (it < niter && ttot < dtf) {
        int rv = BE_chem_solve(calculate_rhs_primordial, calculate_jacobian_primordial,
                                g_input, dt, g_rtol, g_atol, 1, g_n, g_scale,
                                (void *) g_data, g_u0, g_s, g_gu, g_Ju);
        for (int i = 0; i < g_n; i++) {
            if (g_input[i] < 0) { rv = 1; break; }
        }
        if (rv == 0) {
            ttot += dt;
            for (int i = 0; i < g_n; i++) {
                if (g_input[i] < floor_value) g_input[i] = floor_value;
                g_prev[i] = g_input[i];
                g_scale[i] = fmax(fabs(g_input[i]), floor_value);
            }
            if (ttot < dtf) dt = fmin(dt * 2.0, dtf - ttot);
            it += 1;
        } else {
            dt /= 2.0;
            for (int i = 0; i < g_n; i++) {
                g_input[i] = g_prev[i];
                g_scale[i] = fmax(fabs(g_prev[i]), floor_value);
            }
            if (dt < 1.0e-15 * dtf) break;
        }
    }
    return ttot >= dtf;
}

} // extern "C"
