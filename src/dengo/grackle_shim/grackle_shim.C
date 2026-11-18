/***********************************************************************
 dengo-grackle: Grackle's public C API (as of Grackle 3.4.1) implemented
 on top of dengo's generated primordial H/He/H2 solver.

 Build this (with `python -m dengo.grackle_shim --prefix DIR`) into
 DIR/lib/libgrackle.{a,so} + DIR/include/grackle.h, and a code written
 against Grackle -- Enzo, unmodified -- links it in place of the real
 libgrackle.

 Scope: Grackle's primordial_chemistry = 2 network (HI, HII, HeI, HeII,
 HeIII, HM, H2I, H2II, e-) with radiative cooling, no metals, dust, UV
 background, self-shielding or radiative transfer. initialize_chemistry_
 data() refuses (returns FAIL with a message) any setting outside that,
 rather than silently computing different physics. See README.md next to
 this file.

 Unit conventions are Grackle's own (solve_chemistry.c, grackle_units.c,
 solve_rate_cool_g.F, initialize_rates.c in Grackle 3.4.1):
   n_i [cm^-3]  = field_i * density_units / (A_i * m_H), A = 1 (H, H-, e-
                  -- e_density is n_e * m_H), 4 (He), 2 (H2, H2+);
                  the comoving rescaling by a^-3 and Grackle's a^3 in its
                  density conversion cancel, so this holds either way.
   e [erg/g]    = internal_energy * velocity_units^2,
                  velocity_units = length_units / time_units (/ a_value
                  when comoving) -- recomputed, not read from my_units.
   t [s]        = dt * time_units
   z            = 1 / (a_value * a_units) - 1
************************************************************************/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "grackle.h"
#include "primordial_solver.h"
#include "primordial_c_api.h"
#include "primordial_grackle_tables.h"

#ifndef FAIL
#define FAIL 0
#endif
#ifndef SUCCESS
#define SUCCESS 1
#endif
#ifndef TRUE
#define TRUE 1
#endif
#ifndef FALSE
#define FALSE 0
#endif
#define FLOAT_UNDEFINED -99999.0

#define GR_MH 1.67262171e-24      /* Grackle's mh (phys_constants.h) */
#define GR_KBOLTZ 1.3806504e-16   /* Grackle's kboltz */

/* ------------------------------------------------------------------ */
/* Global state Grackle's "global" API uses.                          */
/* ------------------------------------------------------------------ */

int grackle_verbose = FALSE;
chemistry_data *grackle_data = NULL;
chemistry_data_storage grackle_rates;

/* A chemistry_data_storage initialized by us, and the dengo context it
   stands for (the real struct has no room for our own pointer). */
#define MAX_REGISTERED 64
/* Two contexts per initialization: `ctx` for the chemistry solve, and
   `cool` -- the same tables minus the chemical-energy (H2 formation
   heating / dissociation cooling) terms -- for calculate_cooling_time,
   which in Grackle covers radiative processes only. */
typedef struct {
    chemistry_data_storage *rates;
    primordial_context *ctx;
    primordial_context *cool;
} registry_entry;
static registry_entry g_registry[MAX_REGISTERED];

static registry_entry *lookup_entry(chemistry_data_storage *rates)
{
    for (int i = 0; i < MAX_REGISTERED; i++)
        if (rates != NULL && g_registry[i].rates == rates) return &g_registry[i];
    return NULL;
}

static int register_contexts(chemistry_data_storage *rates, primordial_context *ctx,
                             primordial_context *cool)
{
    registry_entry *e = lookup_entry(rates);
    if (e == NULL) {
        for (int i = 0; i < MAX_REGISTERED && e == NULL; i++)
            if (g_registry[i].rates == NULL) e = &g_registry[i];
        if (e == NULL) return FAIL;
    } else {
        primordial_destroy(e->ctx);
        primordial_destroy(e->cool);
    }
    e->rates = rates;
    e->ctx = ctx;
    e->cool = cool;
    return SUCCESS;
}

static void unregister_contexts(chemistry_data_storage *rates)
{
    registry_entry *e = lookup_entry(rates);
    if (e == NULL) return;
    primordial_destroy(e->ctx);
    primordial_destroy(e->cool);
    e->rates = NULL;
    e->ctx = e->cool = NULL;
}

/* ------------------------------------------------------------------ */
/* Units                                                              */
/* ------------------------------------------------------------------ */

double get_velocity_units(const code_units *my_units)
{
    double velocity_units = my_units->length_units / my_units->time_units;
    if (my_units->comoving_coordinates == 1) velocity_units /= my_units->a_value;
    return velocity_units;
}

void set_velocity_units(code_units *my_units)
{
    my_units->velocity_units = get_velocity_units(my_units);
}

double get_temperature_units(const code_units *my_units)
{
    double v = get_velocity_units(my_units);
    return GR_MH * v * v / GR_KBOLTZ;
}

double gr_query_units(const chemistry_data_storage *my_rates,
                      const char *units_name, double current_a_value)
{
    (void) my_rates; (void) current_a_value;
    fprintf(stderr, "dengo-grackle: gr_query_units(\"%s\") is not implemented.\n",
            units_name ? units_name : "(null)");
    return FLOAT_UNDEFINED;
}

/* ------------------------------------------------------------------ */
/* Parameters                                                         */
/* ------------------------------------------------------------------ */

int local_initialize_chemistry_parameters(chemistry_data *my_chemistry)
{
    if (my_chemistry == NULL) return FAIL;
#define ENTRY(FIELD, TYPE, DEFAULT_VAL) my_chemistry->FIELD = DEFAULT_VAL;
#include "grackle_chemistry_data_fields.def"
#undef ENTRY
    return SUCCESS;
}

int set_default_chemistry_parameters(chemistry_data *my_grackle)
{
    grackle_data = my_grackle;
    return local_initialize_chemistry_parameters(my_grackle);
}

/* Dynamic parameter access, same semantics as Grackle's dynamic_api.c. */
#define TYPE_IS_INT(T) (strcmp(#T, "INT") == 0)
#define TYPE_IS_DOUBLE(T) (strcmp(#T, "DOUBLE") == 0)
#define TYPE_IS_STRING(T) (strcmp(#T, "STRING") == 0)

int *local_chemistry_data_access_int(chemistry_data *my_chemistry, const char *param_name)
{
#define ENTRY(FIELD, TYPE, DEFAULT_VAL)                                  \
    if (TYPE_IS_INT(TYPE) && strcmp(param_name, #FIELD) == 0)            \
        return (int *) (void *) &my_chemistry->FIELD;
#include "grackle_chemistry_data_fields.def"
#undef ENTRY
    return NULL;
}

double *local_chemistry_data_access_double(chemistry_data *my_chemistry, const char *param_name)
{
#define ENTRY(FIELD, TYPE, DEFAULT_VAL)                                  \
    if (TYPE_IS_DOUBLE(TYPE) && strcmp(param_name, #FIELD) == 0)         \
        return (double *) (void *) &my_chemistry->FIELD;
#include "grackle_chemistry_data_fields.def"
#undef ENTRY
    return NULL;
}

char **local_chemistry_data_access_string(chemistry_data *my_chemistry, const char *param_name)
{
#define ENTRY(FIELD, TYPE, DEFAULT_VAL)                                  \
    if (TYPE_IS_STRING(TYPE) && strcmp(param_name, #FIELD) == 0)         \
        return (char **) (void *) &my_chemistry->FIELD;
#include "grackle_chemistry_data_fields.def"
#undef ENTRY
    return NULL;
}

static const char *param_name_of_type(const char *type_name, unsigned int i)
{
    unsigned int count = 0;
#define ENTRY(FIELD, TYPE, DEFAULT_VAL)                                  \
    if (strcmp(#TYPE, type_name) == 0) {                                 \
        if (count == i) return #FIELD;                                   \
        count++;                                                         \
    }
#include "grackle_chemistry_data_fields.def"
#undef ENTRY
    return NULL;
}

const char *param_name_int(unsigned int i) { return param_name_of_type("INT", i); }
const char *param_name_double(unsigned int i) { return param_name_of_type("DOUBLE", i); }
const char *param_name_string(unsigned int i) { return param_name_of_type("STRING", i); }

unsigned int grackle_num_params(const char *type_name)
{
    unsigned int count = 0;
#define ENTRY(FIELD, TYPE, DEFAULT_VAL) if (strcmp(#TYPE, type_name) == 0) count++;
#include "grackle_chemistry_data_fields.def"
#undef ENTRY
    return count;
}

/* ------------------------------------------------------------------ */
/* Initialization                                                     */
/* ------------------------------------------------------------------ */

static int reject_int(const char *name, int value, int required)
{
    if (value == required) return 0;
    fprintf(stderr,
            "dengo-grackle: %s = %d is not supported (this library implements "
            "Grackle's primordial_chemistry = 2 network only and requires "
            "%s = %d). Use the real Grackle for this configuration.\n",
            name, value, name, required);
    return 1;
}

static int check_supported(const chemistry_data *c, const code_units *u)
{
    int bad = 0;
    bad += reject_int("primordial_chemistry", c->primordial_chemistry, 2);
    bad += reject_int("with_radiative_cooling", c->with_radiative_cooling, 1);
    bad += reject_int("metal_cooling", c->metal_cooling, 0);
    bad += reject_int("dust_chemistry", c->dust_chemistry, 0);
    bad += reject_int("h2_on_dust", c->h2_on_dust, 0);
    bad += reject_int("use_dust_density_field", c->use_dust_density_field, 0);
    if (c->dust_recombination_cooling > 0)
        bad += reject_int("dust_recombination_cooling", c->dust_recombination_cooling, 0);
    if (c->photoelectric_heating > 0)
        bad += reject_int("photoelectric_heating", c->photoelectric_heating, 0);
    bad += reject_int("use_isrf_field", c->use_isrf_field, 0);
    bad += reject_int("UVbackground", c->UVbackground, 0);
    bad += reject_int("Compton_xray_heating", c->Compton_xray_heating, 0);
    bad += reject_int("use_volumetric_heating_rate", c->use_volumetric_heating_rate, 0);
    bad += reject_int("use_specific_heating_rate", c->use_specific_heating_rate, 0);
    bad += reject_int("use_temperature_floor", c->use_temperature_floor, 0);
    bad += reject_int("use_radiative_transfer", c->use_radiative_transfer, 0);
    bad += reject_int("self_shielding_method", c->self_shielding_method, 0);
    bad += reject_int("H2_self_shielding", c->H2_self_shielding, 0);
    bad += reject_int("H2_custom_shielding", c->H2_custom_shielding, 0);
    bad += reject_int("CaseBRecombination", c->CaseBRecombination, 0);
    bad += reject_int("ih2co", c->ih2co, 1);
    /* Rate-selection switches: dengo has one fixed set of fits for each
       of these processes, always on. */
    bad += reject_int("h2_charge_exchange_rate", c->h2_charge_exchange_rate, 1);
    bad += reject_int("h2_h_cooling_rate", c->h2_h_cooling_rate, 1);
    bad += reject_int("collisional_excitation_rates", c->collisional_excitation_rates, 1);
    bad += reject_int("collisional_ionisation_rates", c->collisional_ionisation_rates, 1);
    bad += reject_int("recombination_cooling_rates", c->recombination_cooling_rates, 1);
    bad += reject_int("bremsstrahlung_cooling_rates", c->bremsstrahlung_cooling_rates, 1);
    if (c->LWbackground_intensity != 0.0) {
        fprintf(stderr, "dengo-grackle: LWbackground_intensity = %g is not supported "
                "(no Lyman-Werner background; requires 0).\n", c->LWbackground_intensity);
        bad++;
    }
    if (fabs(c->Gamma - 5.0 / 3.0) > 1.0e-10) {
        fprintf(stderr, "dengo-grackle: Gamma = %.17g is not supported (dengo's "
                "equation of state uses 5/3 for every monatomic species).\n", c->Gamma);
        bad++;
    }
    if (c->three_body_rate < 0 || c->three_body_rate >= DENGO_GR_NVARIANTS) {
        fprintf(stderr, "dengo-grackle: three_body_rate = %d is not supported "
                "(0-%d).\n", c->three_body_rate, DENGO_GR_NVARIANTS - 1);
        bad++;
    }
    if (u->comoving_coordinates == FALSE && u->a_units != 1.0) {
        fprintf(stderr, "ERROR: a_units must be 1.0 if comoving_coordinates is 0.\n");
        bad++;
    }
    return bad;
}

/* dengo's tables for the requested three_body_rate (only a few tables
   differ between variants; the rest are shared). */
static double *assemble_tables(int three_body_rate)
{
    double *tables = (double *) malloc(PRIMORDIAL_TABLE_DOUBLES * sizeof(double));
    if (tables == NULL) return NULL;
    memcpy(tables, dengo_gr_base_tables, PRIMORDIAL_TABLE_DOUBLES * sizeof(double));
    for (int v = 0; v < DENGO_GR_NVARYING; v++) {
        memcpy(tables + (size_t) dengo_gr_varying_tables[v] * PRIMORDIAL_NBINS,
               dengo_gr_variant_tables[three_body_rate][v],
               PRIMORDIAL_NBINS * sizeof(double));
    }
    return tables;
}

/* Linear-in-value, log-T-binned lookup into one dengo table -- the same
   interpolation the solver does. */
static double dengo_table_value(const double *tables, int table, double T)
{
    const double *t = tables + (size_t) table * PRIMORDIAL_NBINS;
    double lb = log(DENGO_GR_T_START);
    int nbins = PRIMORDIAL_NBINS - 1;
    double dbin = (log(DENGO_GR_T_END) - lb) / nbins;
    double x = (log(T) - lb) / dbin;
    int bin = (int) x;
    if (bin <= 0) bin = 0;
    else if (bin >= nbins) bin = nbins - 1;
    double frac = x - bin;
    if (frac < 0.0) frac = 0.0;
    if (frac > 1.0) frac = 1.0;
    return t[bin] + frac * (t[bin + 1] - t[bin]);
}

/* k1/k2 on Grackle's own temperature grid and in Grackle's code units,
   for codes that read grackle_rates directly (Enzo's photon timestep). */
static int fill_k1_k2(const chemistry_data *c, chemistry_data_storage *rates,
                      const code_units *u, const double *tables)
{
    int n = c->NumberOfTemperatureBins;
    if (n < 2) return FAIL;
    double co_density = (u->comoving_coordinates == TRUE)
        ? u->density_units
        : u->density_units / pow(u->a_value * u->a_units, 3);
    double density_base = co_density * pow(u->a_value * u->a_units, 3);
    double kunit = pow(u->a_units, 3) * GR_MH / (density_base * u->time_units);
    double log_start = log(c->TemperatureStart);
    double dlog = (log(c->TemperatureEnd) - log_start) / (n - 1);
    free(rates->k1);
    free(rates->k2);
    rates->k1 = (double *) malloc(n * sizeof(double));
    rates->k2 = (double *) malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) {
        double T = exp(log_start + i * dlog);
        rates->k1[i] = dengo_table_value(tables, DENGO_GR_TABLE_K01, T) / kunit;
        rates->k2[i] = dengo_table_value(tables, DENGO_GR_TABLE_K02, T) / kunit;
    }
    return SUCCESS;
}

int local_initialize_chemistry_data(chemistry_data *my_chemistry,
                                    chemistry_data_storage *my_rates,
                                    code_units *my_units)
{
    if (my_chemistry == NULL || my_rates == NULL || my_units == NULL) return FAIL;

    if (grackle_verbose) {
        fprintf(stdout, "dengo-grackle: Grackle 3.4.1 API backed by dengo's "
                "primordial network (%s)\n", DENGO_GR_VERSION);
    }

#ifdef _OPENMP
    if (my_chemistry->omp_nthreads < 1) my_chemistry->omp_nthreads = omp_get_max_threads();
    omp_set_num_threads(my_chemistry->omp_nthreads);
#else
    if (my_chemistry->omp_nthreads > 1) {
        fprintf(stdout, "omp_nthreads can't be set when Grackle isn't compiled with OPENMP\n");
        return FAIL;
    }
#endif

    if (my_chemistry->HydrogenFractionByMass > 1) {
        fprintf(stderr, "ERROR: HydrogenFractionByMass cannot exceed 1.0\n");
        return FAIL;
    }
    if (my_chemistry->HydrogenFractionByMass < 0) my_chemistry->HydrogenFractionByMass = 0.76;

    if (check_supported(my_chemistry, my_units) != 0) return FAIL;

    double *tables = assemble_tables(my_chemistry->three_body_rate);
    if (tables == NULL) return FAIL;

    primordial_options opts;
    primordial_default_options(&opts);
    opts.niter = my_chemistry->max_iterations;
    /* Grackle's cie_cooling switches on both the CIE cooling term and the
       continuum optical-depth suppression that goes with it. */
    opts.use_cie_optical_depth = my_chemistry->cie_cooling ? 1 : 0;
    opts.use_h2_optical_depth = my_chemistry->h2_optical_depth_approximation ? 1 : 0;
    if (!my_chemistry->cie_cooling) {
        for (int i = 0; i < PRIMORDIAL_NBINS; i++)
            tables[(size_t) DENGO_GR_TABLE_CIECO * PRIMORDIAL_NBINS + i] = 0.0;
    }

    memset(my_rates, 0, sizeof(chemistry_data_storage));
    if (fill_k1_k2(my_chemistry, my_rates, my_units, tables) != SUCCESS) {
        free(tables);
        return FAIL;
    }
    primordial_context *ctx = primordial_create(tables, &opts);
    for (int t = 0; t < DENGO_GR_NCHEMHEAT; t++)
        for (int i = 0; i < PRIMORDIAL_NBINS; i++)
            tables[(size_t) dengo_gr_chemheat_tables[t] * PRIMORDIAL_NBINS + i] = 0.0;
    primordial_context *cool = primordial_create(tables, &opts);
    free(tables);
    if (ctx == NULL || cool == NULL || register_contexts(my_rates, ctx, cool) != SUCCESS) {
        primordial_destroy(ctx);
        primordial_destroy(cool);
        return FAIL;
    }
    return SUCCESS;
}

int initialize_chemistry_data(code_units *my_units)
{
    if (grackle_data == NULL) {
        fprintf(stderr, "dengo-grackle: call set_default_chemistry_parameters() first.\n");
        return FAIL;
    }
    return local_initialize_chemistry_data(grackle_data, &grackle_rates, my_units);
}

int local_free_chemistry_data(chemistry_data *my_chemistry, chemistry_data_storage *my_rates)
{
    (void) my_chemistry;
    if (my_rates == NULL) return FAIL;
    unregister_contexts(my_rates);
    free(my_rates->k1);
    free(my_rates->k2);
    my_rates->k1 = my_rates->k2 = NULL;
    return SUCCESS;
}

int free_chemistry_data(void)
{
    return local_free_chemistry_data(grackle_data, &grackle_rates);
}

/* ------------------------------------------------------------------ */
/* Field mapping                                                      */
/* ------------------------------------------------------------------ */

static int setup_grid(const chemistry_data *c, const code_units *u,
                      grackle_field_data *f, primordial_grid *g)
{
    if (f->grid_rank < 1 || f->grid_rank > 3 || f->grid_dimension == NULL ||
        f->grid_start == NULL || f->grid_end == NULL) {
        fprintf(stderr, "dengo-grackle: invalid grid description.\n");
        return FAIL;
    }
    g->grid_rank = f->grid_rank;
    g->grid_dimension = f->grid_dimension;
    g->grid_start = f->grid_start;
    g->grid_end = f->grid_end;

    double n_unit = u->density_units / GR_MH;
    double v = get_velocity_units(u);
    struct { int field; gr_float *ptr; double to_cgs; const char *name; } map[] = {
        {PRIMORDIAL_FIELD_H_1,  f->HI_density,      n_unit,       "HI_density"},
        {PRIMORDIAL_FIELD_H_2,  f->HII_density,     n_unit,       "HII_density"},
        {PRIMORDIAL_FIELD_He_1, f->HeI_density,     n_unit / 4.0, "HeI_density"},
        {PRIMORDIAL_FIELD_He_2, f->HeII_density,    n_unit / 4.0, "HeII_density"},
        {PRIMORDIAL_FIELD_He_3, f->HeIII_density,   n_unit / 4.0, "HeIII_density"},
        {PRIMORDIAL_FIELD_H_m0, f->HM_density,      n_unit,       "HM_density"},
        {PRIMORDIAL_FIELD_H2_1, f->H2I_density,     n_unit / 2.0, "H2I_density"},
        {PRIMORDIAL_FIELD_H2_2, f->H2II_density,    n_unit / 2.0, "H2II_density"},
        {PRIMORDIAL_FIELD_de,   f->e_density,       n_unit,       "e_density"},
        {PRIMORDIAL_FIELD_ge,   f->internal_energy, v * v,        "internal_energy"},
    };
    int nmap = (int) (sizeof(map) / sizeof(map[0]));
    if (nmap != PRIMORDIAL_NFIELDS) return FAIL;
    for (int i = 0; i < nmap; i++) {
        if (map[i].ptr == NULL) {
            fprintf(stderr, "dengo-grackle: required field %s is NULL.\n", map[i].name);
            return FAIL;
        }
        g->fields[map[i].field] = map[i].ptr;
        g->to_cgs[map[i].field] = map[i].to_cgs;
    }
    (void) c;
    return SUCCESS;
}

static double redshift_of(const code_units *u)
{
    return 1.0 / (u->a_value * u->a_units) - 1.0;
}

/* Grackle's make_consistent_g: rescale H and He species to the
   (metal-free) gas density, then set e- from charge neutrality.
   Interior cells only, same as Grackle. */
static void make_consistent(const chemistry_data *c, grackle_field_data *f)
{
    long dim[3] = {1, 1, 1}, st[3] = {0, 0, 0}, en[3] = {0, 0, 0};
    for (int d = 0; d < f->grid_rank; d++) {
        dim[d] = f->grid_dimension[d];
        st[d] = f->grid_start[d];
        en[d] = f->grid_end[d];
    }
    double fh = c->HydrogenFractionByMass;
    for (long k = st[2]; k <= en[2]; k++)
    for (long j = st[1]; j <= en[1]; j++)
    for (long i = st[0]; i <= en[0]; i++) {
        long x = i + dim[0] * (j + dim[1] * k);
        double metalfree = f->density[x];
        if (f->metal_density != NULL) metalfree -= f->metal_density[x];
        f->HI_density[x] = fabs(f->HI_density[x]);
        f->HII_density[x] = fabs(f->HII_density[x]);
        f->HeI_density[x] = fabs(f->HeI_density[x]);
        f->HeII_density[x] = fabs(f->HeII_density[x]);
        f->HeIII_density[x] = fabs(f->HeIII_density[x]);
        f->HM_density[x] = fabs(f->HM_density[x]);
        f->H2I_density[x] = fabs(f->H2I_density[x]);
        f->H2II_density[x] = fabs(f->H2II_density[x]);
        double totalH = f->HI_density[x] + f->HII_density[x] + f->HM_density[x]
                      + f->H2I_density[x] + f->H2II_density[x];
        double totalHe = f->HeI_density[x] + f->HeII_density[x] + f->HeIII_density[x];
        double correctH = fh * metalfree / totalH;
        double correctHe = (1.0 - fh) * metalfree / totalHe;
        f->HI_density[x] *= correctH;
        f->HII_density[x] *= correctH;
        f->HM_density[x] *= correctH;
        f->H2I_density[x] *= correctH;
        f->H2II_density[x] *= correctH;
        f->HeI_density[x] *= correctHe;
        f->HeII_density[x] *= correctHe;
        f->HeIII_density[x] *= correctHe;
        f->e_density[x] = f->HII_density[x] + f->HeII_density[x] / 4.0
                        + f->HeIII_density[x] / 2.0
                        - f->HM_density[x] + f->H2II_density[x] / 2.0;
    }
}

/* ------------------------------------------------------------------ */
/* Solve / derived quantities                                         */
/* ------------------------------------------------------------------ */

static registry_entry *entry_or_complain(chemistry_data_storage *my_rates)
{
    registry_entry *e = lookup_entry(my_rates);
    if (e == NULL)
        fprintf(stderr, "dengo-grackle: chemistry data not initialized "
                "(call initialize_chemistry_data first).\n");
    return e;
}

static primordial_context *context_or_complain(chemistry_data_storage *my_rates)
{
    registry_entry *e = entry_or_complain(my_rates);
    return e ? e->ctx : NULL;
}

int local_solve_chemistry(chemistry_data *my_chemistry,
                          chemistry_data_storage *my_rates,
                          code_units *my_units,
                          grackle_field_data *my_fields,
                          double dt_value)
{
    if (!my_chemistry->use_grackle) return SUCCESS;
    primordial_context *ctx = context_or_complain(my_rates);
    if (ctx == NULL) return FAIL;
    if (my_fields->density == NULL) {
        fprintf(stderr, "dengo-grackle: required field density is NULL.\n");
        return FAIL;
    }
    primordial_grid g;
    if (setup_grid(my_chemistry, my_units, my_fields, &g) != SUCCESS) return FAIL;

    double dt = dt_value * my_units->time_units;
    long n_failed = 0;
    int rv = primordial_solve_grid(ctx, &g, dt, redshift_of(my_units), &n_failed);
    if (rv < 0) {
        fprintf(stderr, "dengo-grackle: invalid arguments to solve_chemistry.\n");
        return FAIL;
    }
    make_consistent(my_chemistry, my_fields);
    if (n_failed > 0) {
        /* Same policy as Grackle when max_iterations is exceeded: warn,
           keep going, unless exit_after_iterations_exceeded is set. */
        fprintf(stderr, "dengo-grackle WARNING: %ld cell(s) did not reach "
                "dt = %g s within max_iterations = %d; they keep their last "
                "converged state.\n", n_failed, dt, my_chemistry->max_iterations);
        if (my_chemistry->exit_after_iterations_exceeded) return FAIL;
    }
    return SUCCESS;
}

int solve_chemistry(code_units *my_units, grackle_field_data *my_fields, double dt_value)
{
    if (grackle_data == NULL) return FAIL;
    return local_solve_chemistry(grackle_data, &grackle_rates, my_units, my_fields, dt_value);
}

int local_calculate_cooling_time(chemistry_data *my_chemistry,
                                 chemistry_data_storage *my_rates,
                                 code_units *my_units,
                                 grackle_field_data *my_fields,
                                 gr_float *cooling_time)
{
    if (!my_chemistry->use_grackle) return SUCCESS;
    registry_entry *e = entry_or_complain(my_rates);
    if (e == NULL || cooling_time == NULL) return FAIL;
    primordial_context *ctx = e->cool;
    primordial_grid g;
    if (setup_grid(my_chemistry, my_units, my_fields, &g) != SUCCESS) return FAIL;
    int rv = primordial_cooling_time_grid(ctx, &g, redshift_of(my_units), cooling_time);
    if (rv < 0) return FAIL;
    /* seconds -> code time units, interior only (as Grackle) */
    long dim[3] = {1, 1, 1}, st[3] = {0, 0, 0}, en[3] = {0, 0, 0};
    for (int d = 0; d < my_fields->grid_rank; d++) {
        dim[d] = my_fields->grid_dimension[d];
        st[d] = my_fields->grid_start[d];
        en[d] = my_fields->grid_end[d];
    }
    for (long k = st[2]; k <= en[2]; k++)
    for (long j = st[1]; j <= en[1]; j++)
    for (long i = st[0]; i <= en[0]; i++)
        cooling_time[i + dim[0] * (j + dim[1] * k)] /= my_units->time_units;
    if (rv != 0) fprintf(stderr, "dengo-grackle WARNING: cooling time could not be "
                         "evaluated in some cells (set to 0).\n");
    return SUCCESS;
}

int calculate_cooling_time(code_units *my_units, grackle_field_data *my_fields,
                           gr_float *cooling_time)
{
    if (grackle_data == NULL) return FAIL;
    return local_calculate_cooling_time(grackle_data, &grackle_rates, my_units,
                                        my_fields, cooling_time);
}

int local_calculate_temperature(chemistry_data *my_chemistry,
                                chemistry_data_storage *my_rates,
                                code_units *my_units,
                                grackle_field_data *my_fields,
                                gr_float *temperature)
{
    if (!my_chemistry->use_grackle) return SUCCESS;
    primordial_context *ctx = context_or_complain(my_rates);
    if (ctx == NULL || temperature == NULL) return FAIL;
    primordial_grid g;
    if (setup_grid(my_chemistry, my_units, my_fields, &g) != SUCCESS) return FAIL;
    int rv = primordial_temperature_grid(ctx, &g, temperature);
    return rv < 0 ? FAIL : SUCCESS;
}

int calculate_temperature(code_units *my_units, grackle_field_data *my_fields,
                          gr_float *temperature)
{
    if (grackle_data == NULL) return FAIL;
    return local_calculate_temperature(grackle_data, &grackle_rates, my_units,
                                       my_fields, temperature);
}

static int not_implemented(const char *what)
{
    fprintf(stderr, "dengo-grackle: %s is not implemented.\n", what);
    return FAIL;
}

int local_calculate_gamma(chemistry_data *c, chemistry_data_storage *r, code_units *u,
                          grackle_field_data *f, gr_float *g)
{ (void) c; (void) r; (void) u; (void) f; (void) g; return not_implemented("calculate_gamma"); }

int calculate_gamma(code_units *u, grackle_field_data *f, gr_float *g)
{ return local_calculate_gamma(grackle_data, &grackle_rates, u, f, g); }

int local_calculate_pressure(chemistry_data *c, chemistry_data_storage *r, code_units *u,
                             grackle_field_data *f, gr_float *p)
{ (void) c; (void) r; (void) u; (void) f; (void) p; return not_implemented("calculate_pressure"); }

int calculate_pressure(code_units *u, grackle_field_data *f, gr_float *p)
{ return local_calculate_pressure(grackle_data, &grackle_rates, u, f, p); }

int local_calculate_dust_temperature(chemistry_data *c, chemistry_data_storage *r, code_units *u,
                                     grackle_field_data *f, gr_float *t)
{ (void) c; (void) r; (void) u; (void) f; (void) t; return not_implemented("calculate_dust_temperature"); }

int calculate_dust_temperature(code_units *u, grackle_field_data *f, gr_float *t)
{ return local_calculate_dust_temperature(grackle_data, &grackle_rates, u, f, t); }

/* ------------------------------------------------------------------ */
/* Misc                                                               */
/* ------------------------------------------------------------------ */

grackle_version get_grackle_version(void)
{
    grackle_version v;
    v.version = "3.4.1";          /* the API level implemented */
    v.branch = "dengo-grackle";
    v.revision = DENGO_GR_VERSION;
    return v;
}

int grimpl_check_consistency_(int gr_float_hdrsize)
{
    if (gr_float_hdrsize != (int) sizeof(gr_float)) {
        fprintf(stderr, "dengo-grackle: gr_float size mismatch (%d vs %d).\n",
                gr_float_hdrsize, (int) sizeof(gr_float));
        return FAIL;
    }
    return SUCCESS;
}

int gr_initialize_field_data(grackle_field_data *my_fields)
{
    if (my_fields == NULL) {
        fprintf(stderr, "gr_initial_field_data was passed a NULL pointer\n");
        return FAIL;
    }
    my_fields->grid_rank = -1;
    my_fields->grid_dimension = NULL;
    my_fields->grid_start = NULL;
    my_fields->grid_end = NULL;
    my_fields->grid_dx = -1.0;
#define ENTRY(MEMBER_NAME) my_fields->MEMBER_NAME = NULL;
#include "grackle_field_data_fdatamembers.def"
#undef ENTRY
    return SUCCESS;
}
