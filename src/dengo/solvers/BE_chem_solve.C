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

// function prototypes
int BE_Resid_Fun(rhs_f, double *u, double *u0, double *gu, double dt, 
                 int nstrip, int nchem, double *scaling, double *inv_scaling, void *sdata);
int BE_Resid_Jac(jac_f, double *u, double *Ju, double dt, 
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

    // compute nonlinear residual and Jacobian
    if (BE_Resid_Fun(f, u, u0, gu, dt, nstrip, nchem, scaling, inv_scaling, sdata) != 0) {
      ///*
      // rescale back to input variables
      for (i=0; i<nstrip*nchem; i++)  u[i] *= scaling[i];
      // also rescale the absolute tolerances back
      for (i=0; i<nstrip*nchem; i++)  atol[i] *= scaling[i];
      //*/

      //fprintf(stderr, "Error in BE_Resid_Fun \n");
      delete[] inv_scaling;
      return 1;
    }

    if (BE_Resid_Jac(J, u, Ju, dt, nstrip, nchem, scaling, inv_scaling, sdata) != 0) {
      ///*
      // rescale back to input variables
      for (i=0; i<nstrip*nchem; i++)  u[i] *= scaling[i];
      // also rescale the absolute tolerances back
      for (i=0; i<nstrip*nchem; i++)  atol[i] *= scaling[i];
      //*/

      //fprintf(stderr, "Error in BE_Resid_Jac \n");
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
          fatal_error = 1;
          continue;
      }

      // update solution in this cell
      for (int ii=0; ii<nchem; ii++)  u[ioff+ii] -= lam*s[ioff+ii];

      // check error in this cell (max norm)
      for (int ii=0; ii<nchem; ii++) {
          if ( fabs(s[ioff+ii]) > (atol[ioff+ii] + rtol[ioff+ii] * fabs(u[ioff+ii]))) {
              if (dt < 1.0) {
	              fprintf(stderr, "dt %0.5g, Sweep %d, Unsolved[%d]: nchem: %d change: % 0.8g sum tol: % 0.5g atol: % 0.5g rtol: % 0.5g value: % 0.5g\n",
		                  dt, isweep, ix, ii, s[ioff+ii], atol[ioff+ii] + rtol[ioff+ii] * fabs(u[ioff+ii]), atol[ioff+ii], rtol[ioff+ii], u[ioff+ii]);
              }
              cell_unsolved = 1;
              break;
          }
          if ( u[ioff+ii] != u[ioff+ii] ) {  // NaN encountered!!
            printf("BE_chem_solve ERROR: NaN in iteration %i (cell %i, species %i); dt = %0.5g, atol = %0.5g\n",
                   isweep,ix,ii, dt, atol[ioff+ii]);
            #ifdef DENGO_DEBUG
            for (int jj = 0; jj < nchem; jj++){
                printf("u[%d+%d] = %0.5g\n", ioff, jj, u[ioff+jj]);
            }
            printf("\n");
            #endif
            if (dt < 1.0) {
	              fprintf(stderr, "dt %0.5g, Sweep %d, Unsolved[%d]: nchem: %d change: % 0.8g sum tol: % 0.5g atol: % 0.5g rtol: % 0.5g value: % 0.5g\n",
		                  dt, isweep, ix, ii, s[ioff+ii], atol[ioff+ii] + rtol[ioff+ii] * fabs(u[ioff+ii]), atol[ioff+ii], rtol[ioff+ii], u[ioff+ii]);
            }
            fatal_error = 1;
            break;
          }
      } // ii loop
      unsolved += cell_unsolved;

    } // ix loop

    if (fatal_error) {
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


// nonlinear residual calculation function, forms nonlinear residual defined 
// by backwards Euler discretization, using user-provided RHS function f.
int BE_Resid_Fun(rhs_f f, double *u, double *u0, double *gu, double dt, 
                 int nstrip, int nchem, double *scaling, double*inv_scaling, void *sdata) 
{
  // local variables
  int i;

  ///*
  // rescale back to input variables
  for (i=0; i<nstrip*nchem; i++)  u[i] *= scaling[i];
  //*/

  // call user-supplied RHS function at current guess
  if (f(u, gu, nstrip, nchem, sdata) != 0)
    /*ENZO_FAIL("Error in user-supplied ODE RHS function f(u)");*/
    return 1;

  ///*
  // rescale u to scaled variables
  for (i=0; i<nstrip*nchem; i++)  u[i] *= inv_scaling[i];

  // rescale rhs to normalized variables variables
  for (i=0; i<nstrip*nchem; i++)  gu[i] *= inv_scaling[i];
  //*/

  // update RHS function to additionally include remaining terms for residual,
  //   g(u) = u - u0 - dt*f(u)
  for (i=0; i<nstrip*nchem; i++)  gu[i] = u[i] - u0[i] - dt*gu[i];

  for (i=0; i<nstrip*nchem; i++){

  #ifdef DENGO_DEBUG
  if ( gu[i] != gu[i] ) {  // NaN encountered!!
    printf("[RHS] NaN encountered at gu[%d] = %0.5g\n", i, gu[i]);
  }
  #endif
  }
  return 0;
}


// nonlinear residual Jacobian function, forms Jacobian defined by backwards
//  Euler discretization, using user-provided Jacobian function J.
int BE_Resid_Jac(jac_f J, double *u, double *Ju, double dt, 
		 int nstrip, int nchem, double *scaling, double*inv_scaling, void *sdata)
{
  // local variables
  int ix, ivar, jvar, i;

  ///*
  // rescale back to input variables
  for (i=0; i<nstrip*nchem; i++)  u[i] *= scaling[i];
  //*/

  // call user-supplied Jacobian function at current guess
  if (J(u, Ju, nstrip, nchem, sdata) != 0)
    /*ENZO_FAIL("Error in user-supplied ODE Jacobian function J(u)");*/
    return 1;

  ///*
  // rescale u to scaled variables
  for (i=0; i<nstrip*nchem; i++)  u[i] *= inv_scaling[i];

  // rescale Jacobian rows to use normalization
  for (ix=0; ix<nstrip; ix++)
    for (jvar=0; jvar<nchem; jvar++) 
      for (ivar=0; ivar<nchem; ivar++) 
	Ju[(ix*nchem+jvar)*nchem+ivar] *= inv_scaling[ix*nchem+ivar];

  // rescale Jacobian columns to account for normalization
  for (ix=0; ix<nstrip; ix++)
    for (ivar=0; ivar<nchem; ivar++) 
      for (jvar=0; jvar<nchem; jvar++) 
	Ju[(ix*nchem+jvar)*nchem+ivar] *= scaling[ix*nchem+jvar];
  //*/

  // update Jacobian to additionally include remaining terms,
  //   J = I - dt*Jf(u)
  for (ix=0; ix<nstrip*nchem*nchem; ix++)   Ju[ix] = -dt*Ju[ix];
  for (ix=0; ix<nstrip; ix++)
    for (ivar=0; ivar<nchem; ivar++)
      Ju[ix*nchem*nchem + ivar*nchem + ivar] += 1.0;
  
  for (ix=0; ix<nstrip*nchem*nchem; ix++){

  #ifdef DENGO_DEBUG
  if ( Ju[ix] != Ju[ix] ) {  // NaN encountered!!
    printf("[JAC] NaN encountered at Jac[%d] = %0.5g\n", ix, Ju[ix]);
  }
  #endif
  }
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
