/*****************************************************************************
 *                                                                           *
 * Copyright 2011 Daniel R. Reynolds                                         *
 *                                                                           *
 * This software is released under the terms of the "Enzo Public License"    *
 * in the accompanying LICENSE file.                                         *
 *                                                                           *
 *****************************************************************************/
/***********************************************************************
/
/  Generic rate equation solver
/
/  written by: Daniel Reynolds
/  date:       October 2011
/
/  PURPOSE: This routine solves the coupled equations,
/               du/dt = f(u),
/           using an implicit backward Euler method with stopping criteria 
/               ||(xnew - xold)/(atol + rtol*xnew)||_RMS < 1
/
/ Solver API: 
/ int BE_chem_solve(int (*f)(double *, double *, int, int), 
/                   int (*J)(double *, double *, int, int), 
/                   double *u, double dt, double *rtol, 
/                   double *atol, int nstrip, int nchem)
/
/ output: integer flag denoting success (0) or failure (1)
/
/ inputs:
/
/   int *f -- function pointer that has the form
/             int f(double *u, double *fu, int nstrip, int nchem)
/       Here, the set of unknowns *u is defined over a strip of length
/       nstrip that contains nchem species per cell, and outputs an array
/       *fu of the same size/shape as *u that gives the ODE RHS
/       corresponding to  du/dt = f(u).  The integer return value should
/       denote success (0) or failure (1). 
/
/   int *J -- function pointer that has the form
/             int J(double *u, double *Ju, int nstrip, int nchem)
/       Here the Jacobian Ju should be a 1D array of length
/       nchem*nchem*nstrip.  Here, for spatial location k, with Jacobian
/       matrix row i and column j, the entries should be ordered as i
/       (fastest) then j (middle) then k (slowest), i.e. the Jacobian 
/       matrix for each cell is stored in a contiguous block, in 
/       column-major (Fortran) ordering.
/
/   double *u -- initial conditions, stored in the form u[nstrip*nchem], 
/       with the nchem variables in a given cell stored contiguously.
/
/   double dt -- desired time step size
/
/   double *rtol -- relative tolerance in each equation, of same size 
/       and ordering as u.
/
/   double *atol -- absolute tolerance in each equation, of the same 
/       size and ordering as u.
/
/   int nstrip, int nchem -- inputs denoting the size of the spatial
/       strip and the number of species per cell. 
/
************************************************************************/

#include <stdio.h>
#include <math.h>

/* See the same constant in cython_solver.h.template's comment: below
   this many cells in a batch, OpenMP's thread-team spawn/join overhead
   costs more than the parallel work saves, so the #pragma below is
   guarded with `if (nstrip > DENGO_OMP_MIN_CELLS)`. Redefined here
   (rather than shared via a header) because this file is compiled
   standalone, with no dependency on any generated solver's headers. */
#ifndef DENGO_OMP_MIN_CELLS
#define DENGO_OMP_MIN_CELLS 2048
#endif

typedef int(*rhs_f)(double *, double *, int, int, void *);
typedef int(*jac_f)(double *, double *, int, int, void *);

// Diagnostic info for the most recent BE_chem_solve() call's failure (if
// any), so a caller can report *why* a step didn't converge instead of
// just that it didn't -- the underlying numbers were already computed
// by the existing per-cell convergence check below, just never kept
// around past a debug fprintf(). Reset at the top of every call, so
// after _advance()'s adaptive-dt retry loop gives up, this reflects the
// most recent (smallest-dt) attempt. reason: 0 = none (last call
// converged), 1 = a species' Newton update exceeded its tolerance
// (species_index/ratio/etc. meaningful), 2 = NaN encountered
// (species_index meaningful, ratio is not), 3 = Gauss_Elim hit a
// singular Jacobian (only cell is meaningful). See NOTES.md.
typedef struct {
  int occurred;
  int reason;
  int cell;
  int species_index;
  double value;   // current normalized species value u
  double change;  // Newton update s (how far it wanted to move)
  double atol;
  double rtol;
  double ratio;   // |change| / (atol + rtol*|value|); reason==1 only
} BE_chem_solve_diag;

static BE_chem_solve_diag g_be_chem_solve_last_failure;

BE_chem_solve_diag BE_chem_solve_last_failure() {
  return g_be_chem_solve_last_failure;
}

// function prototypes
int BE_Resid_FunJac(rhs_f, jac_f, double *u, double *u0, double *gu, double *Ju, double dt,
                     int nstrip, int nchem, double *scaling, double *inv_scaling, void *sdata);
int Gauss_Elim(double *A, double *x, double *b, int n);


// solver function
//
// The per-cell Newton update loop below (and calculate_rhs/
// calculate_jacobian/calculate_temperature/interpolate_rates/
// ensure_electron_consistency in the generated solver) are parallelized
// with OpenMP: every cell in the strip is completely independent (no
// cross-cell terms anywhere in this network), so this is embarrassingly
// parallel over cells. Built without -fopenmp, the #pragma lines are
// just ignored and everything runs serially, identically to before.
//
// A plain `return 1` isn't allowed from inside an OpenMP-parallelized
// loop (it has to run to completion across all threads), so the two
// fatal-error cases that used to return immediately (a singular Jacobian
// in Gauss_Elim, a NaN in the updated state) instead set a shared
// `fatal_error` flag and `continue` past that one cell; the flag is
// checked once the parallel loop finishes and every thread has reported
// in. Same idea for `unsolved`, just as a count instead of a flag (`sum
// > 0` means "not converged", same as the old boolean check).
int BE_chem_solve(rhs_f f, jac_f J,
		  double *u, double dt, double *rtol,
                  double *atol, int nstrip, int nchem,
		  double *scaling, void *sdata,
          double *u0, double *s, double *gu, double *Ju) {

  // local variables
  int i, isweep;
  int sweeps=10;
  double lam=1.0;
  int unsolved;

  // Reset the *exposed* diagnostic now; it's only overwritten (from
  // local_diag, below) at a point where this call is actually about to
  // return failure. That's deliberate: early Newton sweeps routinely
  // violate tolerance before converging on a later one -- that's normal
  // iteration, not a failure -- so capturing unconditionally on every
  // sweep (an earlier version of this) left stale "failure" info
  // exposed even after a call that ultimately succeeded. local_diag is
  // reset every sweep (see below) so on return it reflects only the
  // *last* sweep's worst offender, not the worst across all sweeps
  // (typically the least converged, most misleading choice, since
  // Newton sweeps generally improve monotonically).
  g_be_chem_solve_last_failure.occurred = 0;
  g_be_chem_solve_last_failure.reason = 0;
  g_be_chem_solve_last_failure.ratio = 0.0;
  BE_chem_solve_diag local_diag;

  //create an array to store 1/scaling
  double *inv_scaling = new double[nchem*nstrip];
  for (i=0; i<nstrip*nchem; i++)  inv_scaling[i] = 1.0 / scaling[i];

  ///*
  // rescale input to normalized variables
  for (i=0; i<nstrip*nchem; i++)  u[i] *= inv_scaling[i];
  // also rescale the absolute tolerances
  for (i=0; i<nstrip*nchem; i++)  atol[i] *= inv_scaling[i];
  //*/

  //fprintf(stderr, "nchem = %d, nstrip = %d\n", nchem, nstrip);

  // create/initialize temporary arrays
  //double *u0 = new double[nchem*nstrip];        // initial state
  //double *s  = new double[nchem*nstrip];        // Newton update (each cell)
  //double *gu = new double[nchem*nstrip];        // nonlinear residual
  //double *Ju = new double[nchem*nchem*nstrip];  // Jacobian

  for (i=0; i<nstrip*nchem; i++) {
    u0[i] = u[i];
    //fprintf(stderr, "u[i]: %0.6g (for %d)\n", u[i], i);
  }
  for (i=0; i<nstrip*nchem; i++)         s[i] = 0.0;
  for (i=0; i<nstrip*nchem; i++)        gu[i] = 0.0;
  for (i=0; i<nstrip*nchem*nchem; i++)  Ju[i] = 0.0;

  // perform Newton iterations
  for (isweep=0; isweep<sweeps; isweep++) {
    local_diag.occurred = 0;
    local_diag.reason = 0;
    local_diag.ratio = 0.0;

    // compute nonlinear residual and Jacobian -- f() and J() are always
    // evaluated at the same u within one sweep, so this rescales u to
    // physical units once and calls both, rather than the old separate
    // BE_Resid_Fun/BE_Resid_Jac each independently rescaling u to
    // physical and back for the identical, unchanged values (see
    // NOTES.md: a real, measured redundant-work finding, not just a
    // style cleanup -- it also used to spend two avoidable floating-
    // point round-trips on every Newton sweep instead of zero).
    if (BE_Resid_FunJac(f, J, u, u0, gu, Ju, dt, nstrip, nchem, scaling, inv_scaling, sdata) != 0) {
      ///*
      // rescale back to input variables
      for (i=0; i<nstrip*nchem; i++)  u[i] *= scaling[i];
      // also rescale the absolute tolerances back
      for (i=0; i<nstrip*nchem; i++)  atol[i] *= scaling[i];
      //*/

      //fprintf(stderr, "Error in BE_Resid_FunJac \n");
      // f()/J() themselves failed (e.g. a generated calculate_rhs/
      // calculate_jacobian rejected a negative species density) --
      // a different failure mode than the three below, and one that
      // doesn't currently report which species/cell triggered it (that
      // detail isn't threaded back through f()/J()'s own return code),
      // so this is deliberately generic.
      g_be_chem_solve_last_failure.occurred = 1;
      g_be_chem_solve_last_failure.reason = 4;
      g_be_chem_solve_last_failure.cell = -1;
      g_be_chem_solve_last_failure.species_index = -1;
      delete[] inv_scaling;
      return 1;
    }

    // Newton update for each cell in strip, accumulate convergence check
    unsolved = 0;
    int fatal_error = 0;
    #pragma omp parallel for if(nstrip > DENGO_OMP_MIN_CELLS) reduction(+:unsolved,fatal_error)
    for (int ix=0; ix<nstrip; ix++) {
      // set offset
      int ioff = ix*nchem;
      int cell_unsolved = 0;

      // solve for Newton update
      if (Gauss_Elim(&(Ju[ix*nchem*nchem]), &(s[ioff]), &(gu[ioff]), nchem) != 0) {
          fprintf(stderr, "There was an unsolved case in Gauss_Elim! \n");
          #pragma omp critical
          {
            local_diag.occurred = 1;
            local_diag.reason = 3;
            local_diag.cell = ix;
            local_diag.species_index = -1;
          }
          fatal_error = 1;
          continue;
      }

      // update solution in this cell
      for (int ii=0; ii<nchem; ii++)  u[ioff+ii] -= lam*s[ioff+ii];

      // check error in this cell (max norm)
      for (int ii=0; ii<nchem; ii++) {
          double tol = atol[ioff+ii] + rtol[ioff+ii] * fabs(u[ioff+ii]);
          if ( fabs(s[ioff+ii]) > tol) {
              if (dt < 1.0) {
	              fprintf(stdout, "dt %0.5g, Sweep %d, Unsolved[%d]: nchem: %d change: % 0.8g sum tol: % 0.5g atol: % 0.5g rtol: % 0.5g value: % 0.5g\n",
		                  dt, isweep, ix, ii, s[ioff+ii], atol[ioff+ii] + rtol[ioff+ii] * fabs(u[ioff+ii]), atol[ioff+ii], rtol[ioff+ii], u[ioff+ii]);
              }
              // Record this as the current worst-known violation (by
              // normalized ratio) *within this sweep* (local_diag is
              // reset every sweep -- see above), so a caller that ends
              // up giving up can report which species/cell was hardest
              // to converge on the last attempt, not just "didn't
              // converge". reason 2/3 (NaN, singular Jacobian) take
              // priority and are never overwritten by a mere tolerance
              // miss.
              double ratio = fabs(s[ioff+ii]) / tol;
              if (local_diag.reason < 2 && ratio > local_diag.ratio) {
                #pragma omp critical
                {
                  if (local_diag.reason < 2 && ratio > local_diag.ratio) {
                    local_diag.occurred = 1;
                    local_diag.reason = 1;
                    local_diag.cell = ix;
                    local_diag.species_index = ii;
                    local_diag.value = u[ioff+ii];
                    local_diag.change = s[ioff+ii];
                    local_diag.atol = atol[ioff+ii];
                    local_diag.rtol = rtol[ioff+ii];
                    local_diag.ratio = ratio;
                  }
                }
              }
              cell_unsolved = 1;
              break;
          }
          if ( u[ioff+ii] != u[ioff+ii] ) {  // NaN encountered!!
            printf("BE_chem_solve ERROR: NaN in iteration %i (cell %i, species %i); dt = %0.5g, atol = %0.5g\n",
                   isweep,ix,ii, dt, atol[ioff+ii]);
            #pragma omp critical
            {
              local_diag.occurred = 1;
              local_diag.reason = 2;
              local_diag.cell = ix;
              local_diag.species_index = ii;
              local_diag.value = u[ioff+ii];
              local_diag.change = s[ioff+ii];
              local_diag.atol = atol[ioff+ii];
              local_diag.rtol = rtol[ioff+ii];
            }
            #ifdef DENGO_DEBUG
            for (int jj = 0; jj < nchem; jj++){
                printf("u[%d+%d] = %0.5g\n", ioff, jj, u[ioff+jj]);
            }
            printf("\n");
            #endif
            if (dt < 1.0) {
	              fprintf(stdout, "dt %0.5g, Sweep %d, Unsolved[%d]: nchem: %d change: % 0.8g sum tol: % 0.5g atol: % 0.5g rtol: % 0.5g value: % 0.5g\n",
		                  dt, isweep, ix, ii, s[ioff+ii], atol[ioff+ii] + rtol[ioff+ii] * fabs(u[ioff+ii]), atol[ioff+ii], rtol[ioff+ii], u[ioff+ii]);
            }
            fatal_error = 1;
            break;
          }
      } // ii loop
      unsolved += cell_unsolved;

    } // ix loop

    if (fatal_error) {
      g_be_chem_solve_last_failure = local_diag;
      ///*
      // rescale back to input variables
      for (i=0; i<nstrip*nchem; i++)  u[i] *= scaling[i];
      // also rescale the absolute tolerances back
      for (i=0; i<nstrip*nchem; i++)  atol[i] *= scaling[i];
      //*/
      delete[] inv_scaling;
      return 1;
    }

    // check for convergence
    if (!unsolved)  break;

  } // end newton iterations

  // free temporary arrays
  //delete[] u0;
  //delete[] s;
  //delete[] gu;
  //delete[] Ju;
  delete[] inv_scaling;

  ///*
  // rescale back to input variables
  for (i=0; i<nstrip*nchem; i++)  u[i] *= scaling[i];
  // also rescale the absolute tolerances back
  for (i=0; i<nstrip*nchem; i++)  atol[i] *= scaling[i];
  //*/

  // final check, diagnostics output
  if (unsolved) {
    g_be_chem_solve_last_failure = local_diag;
    #ifdef DENGO_DEBUG
    printf("BE_chem_solve WARNING: unsolved after %i iterations\n",isweep);
    #endif
    return 1;
  } else {
    #ifdef DENGI_DEBUG
    printf("BE_chem_solve: solved with %i total iterations\n",isweep);
    #endif
    return 0;
  }

}


// Combined residual + Jacobian evaluation, forming the backward-Euler
// residual g(u) = u - u0 - dt*f(u) and Jacobian J = I - dt*Jf(u), using
// the user-provided f/J. f() and J() are always evaluated at the same
// u within one Newton sweep (BE_chem_solve never updates u between the
// two calls), so u is rescaled to physical units *once* here, not once
// per function the way the old separate BE_Resid_Fun/BE_Resid_Jac did
// -- that repeated the identical rescale (and its rounding error) for
// no reason. Correspondingly, this is also where the Jacobian's
// normalization rescale happens; the two loops the old BE_Resid_Jac
// used (rows, then columns, as separate passes over the whole
// nstrip*nchem*nchem array) are fused into one pass below, with `ivar`
// innermost -- matching Ju's fastest-varying index (contiguous in
// memory) -- rather than the old column-rescale loop's `jvar`
// innermost, which strided by nchem (a fresh, non-contiguous cache
// line on every access). See NOTES.md.
int BE_Resid_FunJac(rhs_f f, jac_f J, double *u, double *u0, double *gu, double *Ju, double dt,
                     int nstrip, int nchem, double *scaling, double *inv_scaling, void *sdata)
{
  int i, ix, ivar, jvar;

  // rescale to physical units -- shared by both f() and J()
  for (i=0; i<nstrip*nchem; i++)  u[i] *= scaling[i];

  // call user-supplied RHS function at current guess
  if (f(u, gu, nstrip, nchem, sdata) != 0)
    /*ENZO_FAIL("Error in user-supplied ODE RHS function f(u)");*/
    return 1;   // u intentionally left in physical units here: the caller
                // discards/overwrites it from a known-good state on any
                // failure return, so nothing depends on its value.

  // call user-supplied Jacobian function at the same (physical) guess
  if (J(u, Ju, nstrip, nchem, sdata) != 0)
    /*ENZO_FAIL("Error in user-supplied ODE Jacobian function J(u)");*/
    return 1;

  // rescale u and the RHS back to normalized variables
  for (i=0; i<nstrip*nchem; i++)  u[i] *= inv_scaling[i];
  for (i=0; i<nstrip*nchem; i++)  gu[i] *= inv_scaling[i];

  // update RHS function to additionally include remaining terms for residual,
  //   g(u) = u - u0 - dt*f(u)
  for (i=0; i<nstrip*nchem; i++)  gu[i] = u[i] - u0[i] - dt*gu[i];

  #ifdef DENGO_DEBUG
  for (i=0; i<nstrip*nchem; i++){
    if ( gu[i] != gu[i] ) {  // NaN encountered!!
      printf("[RHS] NaN encountered at gu[%d] = %0.5g\n", i, gu[i]);
    }
  }
  #endif

  // rescale Jacobian for normalization -- fused row+column rescale,
  // ivar innermost (contiguous), one pass over the whole array instead
  // of two
  for (ix=0; ix<nstrip; ix++)
    for (jvar=0; jvar<nchem; jvar++)
      for (ivar=0; ivar<nchem; ivar++)
        Ju[(ix*nchem+jvar)*nchem+ivar] *= inv_scaling[ix*nchem+ivar] * scaling[ix*nchem+jvar];

  // update Jacobian to additionally include remaining terms,
  //   J = I - dt*Jf(u)
  for (ix=0; ix<nstrip*nchem*nchem; ix++)   Ju[ix] = -dt*Ju[ix];
  for (ix=0; ix<nstrip; ix++)
    for (ivar=0; ivar<nchem; ivar++)
      Ju[ix*nchem*nchem + ivar*nchem + ivar] += 1.0;

  #ifdef DENGO_DEBUG
  for (ix=0; ix<nstrip*nchem*nchem; ix++){
    if ( Ju[ix] != Ju[ix] ) {  // NaN encountered!!
      printf("[JAC] NaN encountered at Jac[%d] = %0.5g\n", ix, Ju[ix]);
    }
  }
  #endif
  return 0;
}



// Gaussian Elimination with partial pivoting, followed by backwards 
// substitution, to solve a linear system Ax=b, where A is an n*n matrix, 
// stored in column-major (Fortran) ordering, and where x and b are vectors 
// of length n.
#define idx(i,j,n) ( j*n + i )
int Gauss_Elim(double *A, double *x, double *b, int n)
{
  // local variables
  int i, j, k, p;
  double m, dtmp;

  // copy rhs into solution
  for (i=0; i<n; i++)  x[i] = b[i];

  // forwared elimination stage:
  for (k=0; k<n-1; k++) {
    // search for pivot row
    p = k;
    for (i=k+1; i<n; i++)
      if (fabs(A[idx(i,k,n)]) > fabs(A[idx(p,k,n)]))  p = i;
    
    // perform row swap
    for (j=k; j<n; j++)  {
      dtmp = A[idx(k,j,n)];
      A[idx(k,j,n)] = A[idx(p,j,n)];
      A[idx(p,j,n)] = dtmp;
    }
    dtmp = x[k];
    x[k] = x[p];
    x[p] = dtmp;

    // check for singular matrix
    //if (fabs(A[idx(k,k,n)]) < 1.e-14*fabs(A[0]))
      //fprintf(stderr,"Gauss Elim warning: singular matrix, results may be inaccurate\n");
    
    // elimination of submatrix (column-major ordering)
    for (i=k+1; i<n; i++) 
      A[idx(i,k,n)] /= A[idx(k,k,n)];
    for (j=k+1; j<n; j++)
      for (i=k+1; i<n; i++) 
	A[idx(i,j,n)] -= A[idx(i,k,n)]*A[idx(k,j,n)];
    for (i=k+1; i<n; i++) 
      x[i] -= A[idx(i,k,n)]*x[k];
  } // k loop
  
  // check for singular matrix in last row
  //if (fabs(A[idx(n-1,n-1,n)]) < 1.e-14*fabs(A[0]))
    //fprintf(stderr,"Gauss Elim warning: singular matrix, results may be inaccurate (in last row)\n");
  
  // backwards substitution stage:
  for (i=n-1; i>=0; i--) {
    for (j=i+1; j<n; j++)
      x[i] -= A[idx(i,j,n)]*x[j];
    x[i] /= A[idx(i,i,n)];
  }

  return 0;
}
