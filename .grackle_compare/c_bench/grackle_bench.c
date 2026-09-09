/* Pure-C benchmark of Grackle's actual public C API -- the same
 * local_initialize_chemistry_data()/local_solve_chemistry() entry
 * points a real HPC code (Enzo, etc.) calls, adapted directly from
 * grackle's own src/example/c_local_example.c. No Python, no Cython,
 * anywhere in the timed path.
 *
 * Usage: ./grackle_bench <dims> <n_timed>
 */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>

#include <grackle.h>

#define MH     1.67e-24
#define KBOLTZ 1.3806504e-16
#define G_GRAV 6.674e-8

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

int main(int argc, char *argv[]) {
    int dims = argc > 1 ? atoi(argv[1]) : 100000;
    int n_timed = argc > 2 ? atoi(argv[2]) : 20;

    grackle_verbose = 0;

    double t_setup0 = now();

    code_units my_units;
    my_units.comoving_coordinates = 0;
    my_units.density_units = MH;             /* mass_hydrogen_cgs */
    my_units.length_units = 3.0857e24;        /* ~1 Mpc */
    my_units.time_units = 3.15e13;            /* ~1 Myr */
    my_units.a_units = 1.0;
    my_units.a_value = 1.0;
    set_velocity_units(&my_units);

    chemistry_data *my_grackle_data = malloc(sizeof(chemistry_data));
    if (set_default_chemistry_parameters(my_grackle_data) == 0) {
        fprintf(stderr, "Error in set_default_chemistry_parameters.\n");
        return EXIT_FAILURE;
    }
    my_grackle_data->use_grackle = 1;
    my_grackle_data->with_radiative_cooling = 1;
    my_grackle_data->primordial_chemistry = 2;
    my_grackle_data->metal_cooling = 0;
    my_grackle_data->dust_chemistry = 0;
    my_grackle_data->UVbackground = 0;
    my_grackle_data->CaseBRecombination = 1;
    my_grackle_data->cie_cooling = 1;
    my_grackle_data->h2_optical_depth_approximation = 1;
    my_grackle_data->three_body_rate = 4;
    my_grackle_data->grackle_data_file = "../../.grackle/grackle_data_files/input/CloudyData_noUVB.h5";

    chemistry_data_storage my_grackle_rates;
    if (local_initialize_chemistry_data(my_grackle_data, &my_grackle_rates, &my_units) == 0) {
        fprintf(stderr, "Error in local_initialize_chemistry_data.\n");
        return EXIT_FAILURE;
    }
    double t_setup1 = now();
    fprintf(stdout, "one-time: grackle local_initialize_chemistry_data: %.4f s\n", t_setup1 - t_setup0);

    /* --- per-dims: allocate the field_data handle --- */
    double t_handle0 = now();
    grackle_field_data my_fields;
    gr_initialize_field_data(&my_fields);
    my_fields.grid_rank = 1;
    my_fields.grid_dimension = malloc(sizeof(int));
    my_fields.grid_start = malloc(sizeof(int));
    my_fields.grid_end = malloc(sizeof(int));
    my_fields.grid_dimension[0] = dims;
    my_fields.grid_start[0] = 0;
    my_fields.grid_end[0] = dims - 1;
    my_fields.grid_dx = 0.0;

    my_fields.density         = malloc(dims * sizeof(gr_float));
    my_fields.internal_energy = malloc(dims * sizeof(gr_float));
    my_fields.x_velocity      = malloc(dims * sizeof(gr_float));
    my_fields.y_velocity      = malloc(dims * sizeof(gr_float));
    my_fields.z_velocity      = malloc(dims * sizeof(gr_float));
    my_fields.HI_density      = malloc(dims * sizeof(gr_float));
    my_fields.HII_density     = malloc(dims * sizeof(gr_float));
    my_fields.HeI_density     = malloc(dims * sizeof(gr_float));
    my_fields.HeII_density    = malloc(dims * sizeof(gr_float));
    my_fields.HeIII_density   = malloc(dims * sizeof(gr_float));
    my_fields.e_density       = malloc(dims * sizeof(gr_float));
    my_fields.HM_density      = malloc(dims * sizeof(gr_float));
    my_fields.H2I_density     = malloc(dims * sizeof(gr_float));
    my_fields.H2II_density    = malloc(dims * sizeof(gr_float));
    double t_handle1 = now();

    /* --- fill with the same synthetic state used on the dengo side:
     * n=1e13 cm^-3, T=1500K, 10% molecular by number --- */
    double NH = 1.0e13, T0 = 1500.0;
    double temperature_units = get_temperature_units(&my_units);
    for (int i = 0; i < dims; i++) {
        double n_H1 = NH * 0.4, n_H2 = NH * 1e-6;
        double n_He1 = NH * 0.24 / 4, n_He2 = NH * 1e-8, n_He3 = NH * 1e-10;
        double n_Hm = NH * 1e-10, n_H2I = NH * 0.1, n_H2II = NH * 1e-9;
        double n_e = NH * 1e-6;
        double rho_cgs = MH * (n_H1 + n_H2 + 4 * n_He1 + 4 * n_He2 + 4 * n_He3
                                + n_Hm + 2 * n_H2I + 2 * n_H2II);

        my_fields.density[i] = rho_cgs / my_units.density_units;
        my_fields.HI_density[i] = n_H1 * MH / my_units.density_units;
        my_fields.HII_density[i] = n_H2 * MH / my_units.density_units;
        my_fields.HeI_density[i] = n_He1 * 4 * MH / my_units.density_units;
        my_fields.HeII_density[i] = n_He2 * 4 * MH / my_units.density_units;
        my_fields.HeIII_density[i] = n_He3 * 4 * MH / my_units.density_units;
        my_fields.HM_density[i] = n_Hm * MH / my_units.density_units;
        my_fields.H2I_density[i] = n_H2I * 2 * MH / my_units.density_units;
        my_fields.H2II_density[i] = n_H2II * 2 * MH / my_units.density_units;
        my_fields.e_density[i] = n_e * MH / my_units.density_units;
        my_fields.x_velocity[i] = 0.0;
        my_fields.y_velocity[i] = 0.0;
        my_fields.z_velocity[i] = 0.0;
        my_fields.internal_energy[i] = (1.5 * KBOLTZ * T0 / MH) / (my_units.velocity_units * my_units.velocity_units);
    }

    double rho = NH * MH;
    double t_ff = sqrt(3.0 * M_PI / (32.0 * G_GRAV * rho));
    double dt_seconds = 1.0e-3 * t_ff;
    double dt_code = dt_seconds / my_units.time_units;

    /* warmup */
    if (local_solve_chemistry(my_grackle_data, &my_grackle_rates, &my_units, &my_fields, dt_code) == 0) {
        fprintf(stderr, "Error in local_solve_chemistry (warmup).\n");
        return EXIT_FAILURE;
    }

    double t0 = now();
    for (int step = 0; step < n_timed; step++) {
        if (local_solve_chemistry(my_grackle_data, &my_grackle_rates, &my_units, &my_fields, dt_code) == 0) {
            fprintf(stderr, "Error in local_solve_chemistry (step %d).\n", step);
            return EXIT_FAILURE;
        }
    }
    double t1 = now();
    double per_call = (t1 - t0) / n_timed;

    fprintf(stdout, "dims=%d handle_setup=%.6f s  us/call=%.3f  us/call/cell=%.5f\n",
            dims, t_handle1 - t_handle0, 1e6 * per_call, 1e6 * per_call / dims);

    local_free_chemistry_data(my_grackle_data, &my_grackle_rates);
    return EXIT_SUCCESS;
}
