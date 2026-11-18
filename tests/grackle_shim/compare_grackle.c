/* Exercise Grackle's C API the way Enzo does (Grid_GrackleWrapper.C,
   Grid_ComputeCoolingTime.C, GrackleReadParameters.C) on a ghost-zone-
   padded 3D grid, and print the results as JSON. Built once against
   dengo-grackle and once against the real libgrackle, so the two can be
   compared field by field.

   usage: compare_grackle OUT.json COMOVING THREE_BODY CIE H2OPT DT_SECONDS
                          [reject PARAM VALUE | negative]

   "negative" makes HII and HeIII slightly negative in every interior cell,
   as Enzo's own CoolingTest initializer does.

   (JSON goes to a file: real Grackle prints its own warnings to stdout.)
*/
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "grackle.h"

#define MH 1.67262171e-24
#define KB 1.3806504e-16
#define NX 8
#define NY 6
#define NZ 5
#define NFLD 11
static const int GS[3] = {2, 1, 1};            /* ghost widths */
static const char *NAMES[NFLD] = {"density", "HI_density", "HII_density", "HeI_density",
    "HeII_density", "HeIII_density", "HM_density", "H2I_density", "H2II_density",
    "e_density", "internal_energy"};

static double *fld[NFLD];
static FILE *out;

static int interior(int i, int j, int k)
{
    return i >= GS[0] && i < NX - GS[0] && j >= GS[1] && j < NY - GS[1] &&
           k >= GS[2] && k < NZ - GS[2];
}

static void print_interior(const char *name, const double *a, int last)
{
    fprintf(out, "  \"%s\": [", name);
    int first = 1;
    for (int k = 0; k < NZ; k++) for (int j = 0; j < NY; j++) for (int i = 0; i < NX; i++) {
        if (!interior(i, j, k)) continue;
        fprintf(out, "%s%.17g", first ? "" : ", ", a[i + NX * (j + NY * k)]);
        first = 0;
    }
    fprintf(out, "]%s\n", last ? "" : ",");
}

int main(int argc, char **argv)
{
    if (argc < 7) { fprintf(stderr, "usage\n"); return 2; }
    out = fopen(argv[1], "w");
    if (out == NULL) return 2;
    argv++; argc--;
    int comoving = atoi(argv[1]);
    int three_body = atoi(argv[2]);
    int cie = atoi(argv[3]);
    int h2opt = atoi(argv[4]);
    double dt_seconds = atof(argv[5]);

    if (grimpl_check_consistency_((int) sizeof(gr_float)) != 1) return 3;

    chemistry_data *chem = (chemistry_data *) malloc(sizeof(chemistry_data));
    if (set_default_chemistry_parameters(chem) != 1) return 4;
    chem->use_grackle = 1;
    chem->with_radiative_cooling = 1;
    chem->primordial_chemistry = 2;
    chem->metal_cooling = 0;
    chem->UVbackground = 0;
    chem->three_body_rate = three_body;
    chem->cie_cooling = cie;
    chem->h2_optical_depth_approximation = h2opt;
    chem->grackle_data_file = "";
    chem->Gamma = (double) (float) (5.0 / 3.0);   /* as Enzo passes it */
    if (argc >= 9 && strcmp(argv[6], "reject") == 0) {
        int *ip = local_chemistry_data_access_int(chem, argv[7]);
        double *dp = local_chemistry_data_access_double(chem, argv[7]);
        if (ip) *ip = atoi(argv[8]);
        else if (dp) *dp = atof(argv[8]);
        else return 5;
    }

    /* Enzo-like units. Proper: 1 code density = m_H/cm^3, pc, Myr.
       Comoving: Enzo's cosmology units at z = 20 for a 1 Mpc/h box
       started at z_init = 100 (CosmologyGetUnits.C). */
    code_units u;
    u.comoving_coordinates = comoving;
    if (!comoving) {
        u.density_units = MH;
        u.length_units = 3.0857e18;
        u.time_units = 3.15576e13;
        u.a_units = 1.0;
        u.a_value = 1.0;
    } else {
        double h = 0.7, Om = 0.3, zi = 100.0, z = 20.0, box = 1.0;
        u.density_units = 1.8788e-29 * Om * h * h * pow(1 + z, 3);
        u.length_units = 3.0857e24 * box / h / (1 + z);
        u.time_units = 2.519445e17 / sqrt(Om) / h / pow(1 + zi, 1.5);
        u.a_units = 1.0 / (1 + zi);
        u.a_value = (1.0 / (1 + z)) / u.a_units;
    }
    set_velocity_units(&u);
    double v = get_velocity_units(&u);

    if (initialize_chemistry_data(&u) != 1) { fprintf(out, "{\"initialized\": false}\n"); fclose(out); return 0; }

    int size = NX * NY * NZ;
    for (int f = 0; f < NFLD; f++) fld[f] = (double *) malloc(size * sizeof(double));
    double *tcool = (double *) malloc(size * sizeof(double));
    double *temp = (double *) malloc(size * sizeof(double));

    /* every cell (ghosts too) gets a plausible state; interior cells span
       n = 1e-2 .. 1e13 cm^-3, T = 30 .. 2e4 K */
    int ncell = 0;
    for (int k = 0; k < NZ; k++) for (int j = 0; j < NY; j++) for (int i = 0; i < NX; i++) {
        int x = i + NX * (j + NY * k);
        int c = interior(i, j, k) ? ncell++ : 7;
        double fc = (c % 48) / 47.0;
        double n = pow(10.0, -2.0 + 15.0 * fc);
        double T = pow(10.0, 1.5 + 2.8 * fmod(fc * 7.3, 1.0));
        double xe = pow(10.0, -4.0 + 3.5 * fmod(fc * 3.1, 1.0));
        double fH2 = pow(10.0, -6.0 + 5.0 * fmod(fc * 5.7, 1.0));
        double rho = n * MH / 0.76 / u.density_units;     /* code density */
        double X = 0.76;
        fld[0][x] = rho;
        fld[7][x] = X * rho * fH2;                    /* H2I (mass) */
        fld[2][x] = X * rho * xe;                     /* HII */
        fld[6][x] = X * rho * 1e-9;                   /* HM */
        fld[8][x] = X * rho * 1e-10;                  /* H2II */
        fld[1][x] = X * rho - fld[7][x] - fld[2][x] - fld[6][x] - fld[8][x];
        fld[3][x] = (1 - X) * rho * (1 - 1e-6);       /* HeI */
        fld[4][x] = (1 - X) * rho * 1e-6;             /* HeII */
        fld[5][x] = (1 - X) * rho * 1e-12;            /* HeIII */
        fld[9][x] = fld[2][x] + fld[4][x] / 4 + fld[5][x] / 2 - fld[6][x] + fld[8][x] / 2;
        double ntot = fld[1][x] + fld[2][x] + fld[6][x] + fld[7][x] / 2 + fld[8][x] / 2
                    + (fld[3][x] + fld[4][x] + fld[5][x]) / 4 + fld[9][x];
        double mu = rho / ntot;
        fld[10][x] = KB * T / ((5.0 / 3.0 - 1.0) * mu * MH) / (v * v);
    }

    int negative = (argc >= 7 && strcmp(argv[6], "negative") == 0);
    if (negative) {
        for (int k = 0; k < NZ; k++) for (int j = 0; j < NY; j++) for (int i = 0; i < NX; i++) {
            if (!interior(i, j, k)) continue;
            int x = i + NX * (j + NY * k);
            /* move the ionized mass back to the neutral species so the
               species still sum to the gas density */
            fld[1][x] += fld[2][x] + 5e-20 * fld[0][x];
            fld[3][x] += fld[5][x] + 3e-21 * fld[0][x];
            fld[2][x] = -5e-20 * fld[0][x];
            fld[5][x] = -3e-21 * fld[0][x];
            fld[9][x] = fld[4][x] / 4 - fld[6][x] + fld[8][x] / 2;
        }
    }

    double *ghost_before = (double *) malloc(NFLD * size * sizeof(double));
    for (int f = 0; f < NFLD; f++) memcpy(ghost_before + f * size, fld[f], size * sizeof(double));

    int dim[3] = {NX, NY, NZ}, start[3], end[3];
    for (int d = 0; d < 3; d++) { start[d] = GS[d]; end[d] = dim[d] - 1 - GS[d]; }
    grackle_field_data my;
    gr_initialize_field_data(&my);
    my.grid_rank = 3; my.grid_dimension = dim; my.grid_start = start; my.grid_end = end;
    my.grid_dx = 1.0;
    my.density = fld[0]; my.HI_density = fld[1]; my.HII_density = fld[2];
    my.HeI_density = fld[3]; my.HeII_density = fld[4]; my.HeIII_density = fld[5];
    my.HM_density = fld[6]; my.H2I_density = fld[7]; my.H2II_density = fld[8];
    my.e_density = fld[9]; my.internal_energy = fld[10];

    for (int x = 0; x < size; x++) { temp[x] = -1; tcool[x] = -1; }
    int rt = calculate_temperature(&u, &my, temp);
    int rc = calculate_cooling_time(&u, &my, tcool);
    fprintf(out, "{\n  \"initialized\": true,\n  \"ncell\": %d,\n  \"velocity_units\": %.17g,\n"
           "  \"time_units\": %.17g,\n  \"density_units\": %.17g,\n  \"temperature_rv\": %d,\n  \"cooling_time_rv\": %d,\n",
           ncell, v, u.time_units, u.density_units, rt, rc);
    print_interior("T0", temp, 0);
    print_interior("tcool0", tcool, 0);
    for (int f = 0; f < NFLD; f++) {
        char name[64];
        snprintf(name, sizeof name, "%s0", NAMES[f]);
        print_interior(name, fld[f], 0);
    }

    /* Callers may copy chemistry_data_storage by value between calls
       (gracklepy does); the copy must work like the original. */
    chemistry_data_storage rates_copy = grackle_rates;
    double *tcopy = (double *) malloc(size * sizeof(double));
    int rcopy = local_calculate_temperature(chem, &rates_copy, &u, &my, tcopy);
    int copy_ok = (rcopy == 1);
    for (int x = 0; x < size; x++) {
        int i = x % NX, j = (x / NX) % NY, k = x / (NX * NY);
        if (interior(i, j, k) && tcopy[x] != temp[x]) copy_ok = 0;
    }
    fprintf(out, "  \"copied_rates_ok\": %s,\n", copy_ok ? "true" : "false");

    int rs = solve_chemistry(&u, &my, dt_seconds / u.time_units);
    for (int x = 0; x < size; x++) temp[x] = -1;
    calculate_temperature(&u, &my, temp);
    fprintf(out, "  \"solve_rv\": %d,\n", rs);
    print_interior("T1", temp, 0);
    for (int f = 0; f < NFLD; f++) {
        char name[64];
        snprintf(name, sizeof name, "%s1", NAMES[f]);
        print_interior(name, fld[f], 0);
    }
    int ghost_ok = 1;
    for (int f = 0; f < NFLD; f++)
        for (int k = 0; k < NZ; k++) for (int j = 0; j < NY; j++) for (int i = 0; i < NX; i++) {
            int x = i + NX * (j + NY * k);
            if (!interior(i, j, k) && fld[f][x] != ghost_before[f * size + x]) ghost_ok = 0;
        }
    fprintf(out, "  \"ghosts_untouched\": %s,\n", ghost_ok ? "true" : "false");
    fprintf(out, "  \"HydrogenFractionByMass\": %.17g,\n", chem->HydrogenFractionByMass);
    fprintf(out, "  \"k1_0\": %.17g, \"k1_mid\": %.17g, \"k2_mid\": %.17g\n}\n",
           grackle_rates.k1[0], grackle_rates.k1[chem->NumberOfTemperatureBins / 2],
           grackle_rates.k2[chem->NumberOfTemperatureBins / 2]);
    free_chemistry_data();
    fclose(out);
    return 0;
}
