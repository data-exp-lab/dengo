/* Prototype: a network-agnostic wasm build of dengo's stiff-chemistry
 * Newton solver (src/dengo/solvers/BE_chem_solve.C, vendored into this
 * directory unmodified -- see wasm/README-generic.md), compiled *once*
 * and never rebuilt when a caller's species/reaction selection changes.
 *
 * Contrast with the production path (dengo_wasm.cpp.template): there,
 * `calculate_rhs_<name>`/`calculate_jacobian_<name>` are per-network
 * generated C++, compiled together with BE_chem_solve.C into one
 * per-network .wasm -- picking a different species/reaction subset
 * means a different network means a new em++ invocation. Here, f/J are
 * ordinary C function pointers BE_chem_solve.C already took (it never
 * knew or cared how they were implemented) -- so a caller can instead
 * hand in JS functions registered via Emscripten's Module.addFunction(),
 * built at runtime from wasm/generic_kinetics.js's generic mass-action
 * assembler (rate(T) * product of reactant densities, standard
 * chemical-kinetics math -- no per-reaction code generation needed,
 * just the reaction database wasm/generate_reaction_db.py exports) over
 * whatever subset of reactions is currently checked. This module never
 * needs to know nchem or which species/reactions exist ahead of time.
 *
 * dengo_generic_step()'s adaptive-substep/floor-clamp policy mirrors
 * dengo_wasm.cpp.template's dengo_wasm_step() exactly, generalized from
 * a compile-time NSPECIES to a runtime `nchem` -- the one deliberate
 * omission is per-network ensure_electron_consistency() (a numerical-
 * robustness nicety, not a correctness requirement: this prototype
 * tracks electron density as an ordinary reaction-driven species like
 * any other, since every reaction's own left/right side already
 * balances charge -- see NOTES.md).
 */
#include <cstdlib>
#include <cmath>

#include <emscripten/emscripten.h>

typedef int (*rhs_f)(double *, double *, int, int, void *);
typedef int (*jac_f)(double *, double *, int, int, void *);

// Plain C++ linkage, not extern "C" -- BE_chem_solve.C's own definition
// isn't extern "C" either (it's vendored unmodified from src/dengo/
// solvers/, which the production Cython/wasm builds also link against
// as ordinary, name-mangled C++), so this declaration has to match that
// or the two objects' symbol names won't agree at link time.
int BE_chem_solve(rhs_f f, jac_f J, double *u, double dt, double *rtol,
                   double *atol, int nstrip, int nchem, double *scaling,
                   void *sdata, double *u0, double *s, double *gu, double *Ju);

extern "C" {

EMSCRIPTEN_KEEPALIVE
double *dengo_generic_alloc(int n) { return (double *) malloc(n * sizeof(double)); }

EMSCRIPTEN_KEEPALIVE
void dengo_generic_free(double *p) { free(p); }

EMSCRIPTEN_KEEPALIVE
int dengo_generic_step(rhs_f f, jac_f J, double *state, double dtf, int niter, double reltol, int nchem) {
    double floor_value = 1.0e-20;
    double *prev  = (double *) malloc(nchem * sizeof(double));
    double *scale = (double *) malloc(nchem * sizeof(double));
    double *atol  = (double *) malloc(nchem * sizeof(double));
    double *rtolv = (double *) malloc(nchem * sizeof(double));
    double *u0    = (double *) malloc(nchem * sizeof(double));
    double *s     = (double *) malloc(nchem * sizeof(double));
    double *gu    = (double *) malloc(nchem * sizeof(double));
    double *Ju    = (double *) malloc(nchem * nchem * sizeof(double));

    for (int i = 0; i < nchem; i++) {
        atol[i]  = floor_value * reltol;
        rtolv[i] = reltol;
        scale[i] = fmax(fabs(state[i]), floor_value);
        prev[i]  = state[i];
    }

    double dt = dtf / niter;
    double ttot = 0.0;
    int it = 0;
    while (it < niter && ttot < dtf) {
        int rv = BE_chem_solve(f, J, state, dt, rtolv, atol, 1, nchem, scale,
                                NULL, u0, s, gu, Ju);
        for (int i = 0; i < nchem; i++) {
            if (state[i] < 0) { rv = 1; break; }
        }
        if (rv == 0) {
            ttot += dt;
            for (int i = 0; i < nchem; i++) {
                if (state[i] < floor_value) state[i] = floor_value;
                prev[i] = state[i];
                scale[i] = fmax(fabs(state[i]), floor_value);
            }
            if (ttot < dtf) dt = fmin(dt * 2.0, dtf - ttot);
            it += 1;
        } else {
            dt /= 2.0;
            for (int i = 0; i < nchem; i++) {
                state[i] = prev[i];
                scale[i] = fmax(fabs(prev[i]), floor_value);
            }
            if (dt < 1.0e-15 * dtf) break;
        }
    }

    free(prev); free(scale); free(atol); free(rtolv);
    free(u0); free(s); free(gu); free(Ju);
    return ttot >= dtf;
}

} // extern "C"
