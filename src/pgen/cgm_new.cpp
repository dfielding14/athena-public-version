// Problem generator for realistic cooling

// C++ headers
#include <algorithm>  // min()
#include <cmath>      // abs(), pow(), sqrt()
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // string

// Athena++ headers
#include "../mesh/mesh.hpp"
#include "../athena.hpp"                   // macros, enums, declarations
#include "../athena_arrays.hpp"            // AthenaArray
#include "../globals.hpp"                  // Globals
#include "../parameter_input.hpp"          // ParameterInput
#include "../coordinates/coordinates.hpp"  // Coordinates
#include "../eos/eos.hpp"                  // EquationOfState
#include "../field/field.hpp"              // Field
#include "../hydro/hydro.hpp"              // Hydro
#include "../fft/turbulence.hpp"           // Turbulence

// External library headers
#include <hdf5.h>  // H5*, hid_t, hsize_t, H5*()

// Configuration checking
#if MAGNETIC_FIELDS_ENABLED
#error "This problem generator cannot be used with MHD (yet)"
#endif

// Declarations
void Cooling_Conduction_TurbDriving(MeshBlock *pmb, const Real t, const Real dt,
    const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc, AthenaArray<Real> &cons, int stage);
Real CoolingLosses(MeshBlock *pmb, int iout);
static void FindIndex(const AthenaArray<double> &array, Real value, int *p_index,
    Real *p_fraction);
static Real Interpolate2D(const AthenaArray<double> &table, int j, int i, Real g, Real f);
static Real Interpolate3D(const AthenaArray<double> &table, int k, int j, int i, Real h,
    Real g, Real f);

// Global variables
static const Real mu_m_h = 1.008 * 1.660539040e-24;
static Real gamma_adi;
static Real rho_0, pgas_0;
static Real overdensity_factor, overdensity_radius, overdensity_smoothing_length;
static Real temperature_max;
static Real rho_table_min, rho_table_max, rho_table_n;
static Real pgas_table_min, pgas_table_max, pgas_table_n;
static Real T_cool_start, cfl_cool;
static bool heat_redistribute, rho_dependent_heat_redistribute, constant_energy, cooling_on;
static bool adaptive_driving, tophat_driving;
static Real drive_duration, drive_separation, dedt_on;

static Real cooling_timestep(MeshBlock *pmb);
static int turb_grid_size;
static Real kappa;
static bool conduction_on;
static Real dt_cutoff;

static Real grav_accel;
static bool gravity_on;

// static Real dt_drive; // the duration of driving
// static Real deltat_drive; // the spacing between driving windows

//----------------------------------------------------------------------------------------
// Function for preparing Mesh
// Inputs:
//   pin: input parameters (unused)
// Outputs: (none)

void Mesh::InitUserMeshData(ParameterInput *pin)
{
// turb_flag is initialzed in the Mesh constructor to 0 by default;
// turb_flag = 1 for decaying turbulence
// turb_flag = 2 for driven turbulence
  turb_flag = pin->GetInteger("problem","turb_flag");
  if(turb_flag != 0){
#ifndef FFT
    std::stringstream msg;
    msg << "### FATAL ERROR in TurbulenceDriver::TurbulenceDriver" << std::endl
        << "non zero Turbulence flag is set without FFT!" << std::endl;
    throw std::runtime_error(msg.str().c_str());
    return;
#endif
  }

  // Read general parameters from input file
  gamma_adi = pin->GetReal("hydro", "gamma");
  Real length_scale = pin->GetReal("problem", "length_scale");
  Real rho_scale = pin->GetReal("problem", "rho_scale");
  Real pgas_scale = pin->GetReal("problem", "pgas_scale");
  Real temperature_scale = pin->GetReal("problem", "temperature_scale");
  Real vel_scale = std::sqrt(pgas_scale / rho_scale);
  rho_0 = pin->GetReal("problem", "rho_0");
  pgas_0 = pin->GetReal("problem", "pgas_0");
  overdensity_factor = pin->GetReal("problem", "overdensity_factor");
  overdensity_radius = pin->GetReal("problem", "overdensity_radius");
  overdensity_smoothing_length = pin->GetReal("problem", "overdensity_smoothing_length");
  temperature_max = pin->GetReal("hydro", "tceil");

  // Read cooling-table-related parameters from input file
  rho_table_min = pin->GetReal("problem", "rho_table_min");
  rho_table_max = pin->GetReal("problem", "rho_table_max");
  rho_table_n = pin->GetReal("problem", "rho_table_n");
  pgas_table_min = pin->GetReal("problem", "pgas_table_min");
  pgas_table_max = pin->GetReal("problem", "pgas_table_max");
  pgas_table_n = pin->GetReal("problem", "pgas_table_n");
  std::string cooling_file = pin->GetString("problem", "cooling_file");
  Real z_z_solar = pin->GetReal("problem", "relative_metallicity");
  Real chi_he = pin->GetReal("problem", "helium_mass_fraction");
  int num_helium_fractions =
      pin->GetOrAddInteger("problem", "num_helium_fractions_override", 0);
  int num_hydrogen_densities =
      pin->GetOrAddInteger("problem", "num_hydrogen_densities_override", 0);
  int num_temperatures = pin->GetOrAddInteger("problem", "num_temperatures_override", 0);
  heat_redistribute = pin->GetBoolean("problem", "heat_redistribute");
  rho_dependent_heat_redistribute = pin->GetBoolean("problem", "rho_dependent_heat_redistribute");
  constant_energy = pin->GetBoolean("problem", "constant_energy");
  adaptive_driving = pin->GetBoolean("problem", "adaptive_driving");
  tophat_driving = pin->GetBoolean("problem", "tophat_driving");

  T_cool_start = pin->GetReal("problem", "T_cool_start");
  cfl_cool = pin->GetOrAddReal("problem", "cfl_cool", 0.1);
  cooling_on = false;

  drive_duration   = pin->GetReal("problem", "drive_duration");
  drive_separation = pin->GetReal("problem", "drive_separation");
  dedt_on = pin->GetReal("problem", "dedt");


  // conduction
  kappa = pin->GetReal("problem", "kappa");
  conduction_on = pin->GetBoolean("problem", "conduction_on");
  dt_cutoff = pin->GetOrAddReal("problem", "dt_cutoff", 3.0e-5);

  // gravity
  grav_accel = pin->GetReal("problem", "grav_accel");
  gravity_on = pin->GetBoolean("problem", "gravity_on");


  // Open cooling data file
  hid_t property_list_file = H5Pcreate(H5P_FILE_ACCESS);
  #ifdef MPI_PARALLEL
  {
    H5Pset_fapl_mpio(property_list_file, MPI_COMM_WORLD, MPI_INFO_NULL);
  }
  #endif
  hid_t file = H5Fopen(cooling_file.c_str(), H5F_ACC_RDONLY, property_list_file);
  H5Pclose(property_list_file);
  if (file < 0) {
    std::stringstream message;
    message << "### FATAL ERROR in problem generator\nCould not open " << cooling_file
        << "\n";
    throw std::runtime_error(message.str().c_str());
  }
  hid_t property_list_transfer = H5Pcreate(H5P_DATASET_XFER);
  #ifdef MPI_PARALLEL
  {
    H5Pset_dxpl_mpio(property_list_transfer, H5FD_MPIO_COLLECTIVE);
  }
  #endif

  // Read solar abundances
  double chi_vals[2];
  hsize_t dims[1];
  dims[0] = 2;
  hid_t dataset = H5Dopen(file, "Header/Abundances/Solar_mass_fractions", H5P_DEFAULT);
  hid_t dataspace = H5Screate_simple(1, dims, NULL);
  H5Dread(dataset, H5T_NATIVE_DOUBLE, dataspace, dataspace, property_list_transfer,
      chi_vals);
  H5Dclose(dataset);
  H5Sclose(dataspace);
  Real chi_h_solar = chi_vals[0];
  Real chi_he_solar = chi_vals[1];
  Real chi_z_solar = 1.0 - chi_h_solar - chi_he_solar;
  Real chi_h = 1.0 - chi_he - z_z_solar * chi_z_solar;

  // Read sizes of tables
  if (num_helium_fractions <= 0) {
    dataset = H5Dopen(file, "Header/Number_of_helium_fractions", H5P_DEFAULT);
    H5Dread(dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, property_list_transfer,
        &num_helium_fractions);
    H5Dclose(dataset);
  }
  if (num_hydrogen_densities <= 0) {
    dataset = H5Dopen(file, "Header/Number_of_density_bins", H5P_DEFAULT);
    H5Dread(dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, property_list_transfer,
        &num_hydrogen_densities);
    H5Dclose(dataset);
  }
  if (num_temperatures <= 0) {
    dataset = H5Dopen(file, "Header/Number_of_temperature_bins", H5P_DEFAULT);
    H5Dread(dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, property_list_transfer,
        &num_temperatures);
    H5Dclose(dataset);
  }

  // Read sample data from file
  AthenaArray<double> helium_fraction_samples, hydrogen_density_samples,
      temperature_samples, energy_samples;
  helium_fraction_samples.NewAthenaArray(num_helium_fractions);
  hydrogen_density_samples.NewAthenaArray(num_hydrogen_densities);
  temperature_samples.NewAthenaArray(num_temperatures);
  energy_samples.NewAthenaArray(num_temperatures);
  dataset = H5Dopen(file, "Metal_free/Helium_mass_fraction_bins", H5P_DEFAULT);
  H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, property_list_transfer,
      helium_fraction_samples.data());
  H5Dclose(dataset);
  dataset = H5Dopen(file, "Metal_free/Hydrogen_density_bins", H5P_DEFAULT);
  H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, property_list_transfer,
      hydrogen_density_samples.data());
  H5Dclose(dataset);
  dataset = H5Dopen(file, "Metal_free/Temperature_bins", H5P_DEFAULT);
  H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, property_list_transfer,
      temperature_samples.data());
  H5Dclose(dataset);
  dataset = H5Dopen(file, "Metal_free/Temperature/Energy_density_bins", H5P_DEFAULT);
  H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, property_list_transfer,
      energy_samples.data());
  H5Dclose(dataset);

  // Read temperature data from file
  AthenaArray<double> temperature_table;
  temperature_table.NewAthenaArray(num_helium_fractions, num_temperatures,
      num_hydrogen_densities);
  dataset = H5Dopen(file, "Metal_free/Temperature/Temperature", H5P_DEFAULT);
  H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, property_list_transfer,
      temperature_table.data());
  H5Dclose(dataset);

  // Read electron density data from file
  AthenaArray<double> electron_density_table, electron_density_solar_table;
  electron_density_table.NewAthenaArray(num_helium_fractions, num_temperatures,
      num_hydrogen_densities);
  electron_density_solar_table.NewAthenaArray(num_temperatures, num_hydrogen_densities);
  dataset = H5Dopen(file, "Metal_free/Electron_density_over_n_h", H5P_DEFAULT);
  H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, property_list_transfer,
      electron_density_table.data());
  H5Dclose(dataset);
  dataset = H5Dopen(file, "Solar/Electron_density_over_n_h", H5P_DEFAULT);
  H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, property_list_transfer,
      electron_density_solar_table.data());
  H5Dclose(dataset);

  // Read cooling data from file
  AthenaArray<double> cooling_no_metals_table, cooling_metals_table;
  cooling_no_metals_table.NewAthenaArray(num_helium_fractions, num_temperatures,
      num_hydrogen_densities);
  cooling_metals_table.NewAthenaArray(num_temperatures, num_hydrogen_densities);
  dataset = H5Dopen(file, "Metal_free/Net_Cooling", H5P_DEFAULT);
  H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, property_list_transfer,
      cooling_no_metals_table.data());
  H5Dclose(dataset);
  dataset = H5Dopen(file, "Solar/Net_cooling", H5P_DEFAULT);
  H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, property_list_transfer,
      cooling_metals_table.data());
  H5Dclose(dataset);

  // Close cooling data file
  H5Pclose(property_list_transfer);
  H5Fclose(file);

  // Allocate fixed cooling table
  AllocateRealUserMeshDataField(5);
  ruser_mesh_data[0].NewAthenaArray(rho_table_n);
  ruser_mesh_data[1].NewAthenaArray(pgas_table_n);
  ruser_mesh_data[2].NewAthenaArray(rho_table_n, pgas_table_n);
  ruser_mesh_data[3].NewAthenaArray(rho_table_n, pgas_table_n);
  ruser_mesh_data[4].NewAthenaArray(3); // delta_e_tot, vol_tot, vol_cell

  Real vol_tot = (mesh_size.x1max - mesh_size.x1min)
               * (mesh_size.x2max - mesh_size.x2min) 
               * (mesh_size.x3max - mesh_size.x3min);
  Real vol_cell = vol_tot / mesh_size.nx1 / mesh_size.nx2 / mesh_size.nx3;
  ruser_mesh_data[4](1) = vol_tot;
  ruser_mesh_data[4](2) = vol_cell;


  // Tabulate sample points
  for (int i = 0; i < rho_table_n; ++i) {
    ruser_mesh_data[0](i) = rho_table_min * std::pow(rho_table_max/rho_table_min,
        static_cast<Real>(i)/static_cast<Real>(rho_table_n-1));
  }
  for (int i = 0; i < pgas_table_n; ++i) {
    ruser_mesh_data[1](i) = pgas_table_min * std::pow(pgas_table_max/pgas_table_min,
        static_cast<Real>(i)/static_cast<Real>(pgas_table_n-1));
  }

  // Locate helium fraction in tables
  int helium_fraction_index;
  Real helium_fraction_fraction;
  FindIndex(helium_fraction_samples, chi_he, &helium_fraction_index,
      &helium_fraction_fraction);

  // Tabulate cooling rate
  for (int j = 0; j < rho_table_n; ++j) {
    for (int i = 0; i < pgas_table_n; ++i) {

      // Calculate gas properties
      Real rho = ruser_mesh_data[0](j) * rho_scale;
      Real pgas = ruser_mesh_data[1](i) * pgas_scale;
      Real n_h =
          rho / mu_m_h / (1.0 + chi_he / chi_h + z_z_solar * chi_z_solar / chi_h_solar);
      Real u = pgas / (gamma_adi-1.0);

      // Locate hydrogen density in tables
      int hydrogen_density_index;
      Real hydrogen_density_fraction;
      FindIndex(hydrogen_density_samples, n_h, &hydrogen_density_index,
          &hydrogen_density_fraction);

      // Locate energy in tables
      int energy_index;
      Real energy_fraction;
      FindIndex(energy_samples, u/rho, &energy_index, &energy_fraction);

      // Interpolate temperature from table
      Real temperature = Interpolate3D(temperature_table, helium_fraction_index,
          energy_index, hydrogen_density_index, helium_fraction_fraction, energy_fraction,
          hydrogen_density_fraction);
      ruser_mesh_data[3](j,i) = temperature / temperature_scale;

      // Locate temperature in tables
      int temperature_index;
      Real temperature_fraction;
      FindIndex(temperature_samples, temperature, &temperature_index,
          &temperature_fraction);

      // Interpolate electron densities from tables
      Real n_e_n_h = Interpolate3D(electron_density_table, helium_fraction_index,
          temperature_index, hydrogen_density_index, helium_fraction_fraction,
          temperature_fraction, hydrogen_density_fraction);
      Real n_e_n_h_solar = Interpolate2D(electron_density_solar_table, temperature_index,
          hydrogen_density_index, temperature_fraction, hydrogen_density_fraction);

      // Interpolate cooling from tables
      Real lambda_hhe_n_h_sq = Interpolate3D(cooling_no_metals_table,
          helium_fraction_index, temperature_index, hydrogen_density_index,
          helium_fraction_fraction, temperature_fraction, hydrogen_density_fraction);
      Real lambda_zsolar_n_h_sq = Interpolate2D(cooling_metals_table, temperature_index,
          hydrogen_density_index, temperature_fraction, hydrogen_density_fraction);

      // Calculate cooling rate
      Real edot_cool = SQR(n_h) * (lambda_hhe_n_h_sq
          + lambda_zsolar_n_h_sq * n_e_n_h / n_e_n_h_solar * z_z_solar);
      ruser_mesh_data[2](j,i) = edot_cool * length_scale / (pgas_scale * vel_scale);
    }
  }

  // Delete intermediate tables
  helium_fraction_samples.DeleteAthenaArray();
  hydrogen_density_samples.DeleteAthenaArray();
  temperature_samples.DeleteAthenaArray();
  energy_samples.DeleteAthenaArray();
  temperature_table.DeleteAthenaArray();
  electron_density_table.DeleteAthenaArray();
  electron_density_solar_table.DeleteAthenaArray();
  cooling_no_metals_table.DeleteAthenaArray();
  cooling_metals_table.DeleteAthenaArray();

  // Enroll user-defined functions
  EnrollUserExplicitSourceFunction(Cooling_Conduction_TurbDriving);
  AllocateUserHistoryOutput(3);
  EnrollUserHistoryOutput(0, CoolingLosses, "e_cool");
  EnrollUserHistoryOutput(1, CoolingLosses, "e_ceil");
  EnrollUserHistoryOutput(2, CoolingLosses, "T_avg");
  EnrollUserTimeStepFunction(cooling_timestep);

  return;
}

//----------------------------------------------------------------------------------------
// Function for preparing MeshBlock
// Inputs:
//   pin: input parameters (unused)
// Outputs: (none)

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin)
{
  // Allocate storage for keeping track of cooling
  AllocateRealUserMeshBlockDataField(1);
  ruser_meshblock_data[0].NewAthenaArray(3);
  ruser_meshblock_data[0](0) = 0.0;
  ruser_meshblock_data[0](1) = 0.0;
  ruser_meshblock_data[0](2) = 0.0;

  // Set output variables
  AllocateUserOutputVariables(2);
  SetUserOutputVariableName(0, "edot_cool");
  SetUserOutputVariableName(1, "T");
  return;
}

//----------------------------------------------------------------------------------------
// Function for setting initial conditions
// Inputs:
//   pin: parameters
// Outputs: (none)

void MeshBlock::ProblemGenerator(ParameterInput *pin)
{
  // Prepare index bounds
  int il = is - NGHOST;
  int iu = ie + NGHOST;
  int jl = js;
  int ju = je;
  if (block_size.nx2 > 1) {
    jl -= (NGHOST);
    ju += (NGHOST);
  }
  int kl = ks;
  int ku = ke;
  if (block_size.nx3 > 1) {
    kl -= (NGHOST);
    ku += (NGHOST);
  }

  Real cs_iso_squared = pgas_0/rho_0;
  Real grav_z_offset = pin->GetReal("problem", "grav_z_offset");
  // Initialize primitive values
  for (int k = kl; k <= ku; ++k) {
    Real z = pcoord->x3v(k);
    for (int j = jl; j <= ju; ++j) {
      Real y = pcoord->x2v(j);
      for (int i = il; i <= iu; ++i) {
        Real x = pcoord->x1v(i);
        phydro->w(IDN,k,j,i) = rho_0;
        if (gravity_on){
          phydro->w(IDN,k,j,i) *= std::exp(-1.*grav_accel*std::abs(z-grav_z_offset)/cs_iso_squared);
        }
        phydro->w(IPR,k,j,i) = cs_iso_squared * phydro->w(IDN,k,j,i);
        phydro->w(IVX,k,j,i) = 0.0;
        phydro->w(IVY,k,j,i) = 0.0;
        phydro->w(IVZ,k,j,i) = 0.0;
        if (overdensity_factor > 0.0){
          Real r = sqrt(x*x + y*y + z*z); 
          phydro->w(IDN,k,j,i) = rho_0 * ((overdensity_factor-1) * 0.5 * (1.0 + std::tanh((overdensity_radius - r)/overdensity_smoothing_length))+1.0);
        } 
      }
    }
  }

  // Initialize conserved values
  AthenaArray<Real> b;
  peos->PrimitiveToConserved(phydro->w, b, phydro->u, pcoord, il, iu, jl, ju, kl, ku);
  return;
}


//----------------------------------------------------------------------------------------
// Function to calculate the timestep required to resolve cooling (and later conduction)
//          tcool = 5/2 P/Edot_cool
//          tcond = (dx)^2 / kappa (?)
// Inputs:
//   pmb: pointer to MeshBlock
Real cooling_timestep(MeshBlock *pmb)
{
  if (pmb->pmy_mesh->time == 0.0){
    return 1.0e-6;
  } else {
    Real min_dt=1.0e10;
    AthenaArray<Real> edot_cool, temperature;
    edot_cool.InitWithShallowSlice(pmb->user_out_var, 4, 0, 1);
    temperature.InitWithShallowSlice(pmb->user_out_var, 4, 1, 1);
    for (int k=pmb->ks; k<=pmb->ke; ++k) {
      for (int j=pmb->js; j<=pmb->je; ++j) {
        for (int i=pmb->is; i<=pmb->ie; ++i) {
          if (temperature(k,j,i)>1.0e3){
            Real dt;
            Real edot = fabs(edot_cool(k,j,i));
            Real press = pmb->phydro->w(IPR,k,j,i);
            if (conduction_on){
              dt = cfl_cool * std::min( SQR(pmb->pcoord->dx1f(i))/kappa , 1.5*press/edot); // experiment with that cfl_cool
            } else {
              dt = cfl_cool * 1.5*press/edot;
            }
            dt = std::max( dt , dt_cutoff );
            min_dt = std::min(min_dt, dt);
          }
        }
      }
    }
    edot_cool.DeleteAthenaArray();
    return min_dt;
  }
}


//----------------------------------------------------------------------------------------
// Source function for cooling (with heat redistribution) and turbulent driving
// Inputs:
//   pmb: pointer to MeshBlock
//   t,dt: time (not used) and timestep
//   prim: primitives
//   bcc: cell-centered magnetic fields (not used)
// Outputs:
//   cons: conserved variables updated
// Notes:
//   writes to user_out_var array the following quantities:
//     0: edot_cool
//     1: T
//     3: edot_cond

void Cooling_Conduction_TurbDriving(MeshBlock *pmb, const Real t, const Real dt,
    const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc, AthenaArray<Real> &cons, int stage)
{
  // Prepare values to aggregate
  // cooling
  Real delta_e_tot;
  Real Average_T;
  Real m[3] = {0}, gm[3];
  Real vol_tot  = pmb->pmy_mesh->ruser_mesh_data[4](1);
  Real vol_cell = pmb->pmy_mesh->ruser_mesh_data[4](2);
 

  // Extract data tables
  AthenaArray<Real> rho_table, pgas_table, edot_cool_table, temperature_table;
  rho_table.InitWithShallowCopy(pmb->pmy_mesh->ruser_mesh_data[0]);
  pgas_table.InitWithShallowCopy(pmb->pmy_mesh->ruser_mesh_data[1]);
  edot_cool_table.InitWithShallowCopy(pmb->pmy_mesh->ruser_mesh_data[2]);
  temperature_table.InitWithShallowCopy(pmb->pmy_mesh->ruser_mesh_data[3]);
  
  // Calculate cooling on all blocks
  if (pmb->lid == 0) {

    // Go through all blocks on mesh
    int num_blocks = pmb->pmy_mesh->GetNumMeshBlocksThisRank(Globals::my_rank);
    MeshBlock *pblock = pmb;
    for (int n = 0; n < num_blocks; ++n) {

      // Extract indices
      int is = pblock->is;
      int ie = pblock->ie;
      int js = pblock->js;
      int je = pblock->je;
      int ks = pblock->ks;
      int ke = pblock->ke;

      // Extract arrays
      AthenaArray<Real> prim_local, cons_local, user_out_var_local;
      if (stage == 1) {
        prim_local.InitWithShallowCopy(pblock->phydro->w);
        cons_local.InitWithShallowCopy(pblock->phydro->u1);
      } else {
        prim_local.InitWithShallowCopy(pblock->phydro->w1);
        cons_local.InitWithShallowCopy(pblock->phydro->u);
      }
      user_out_var_local.InitWithShallowCopy(pblock->user_out_var);

      // Calculate cooling
      for (int k = ks; k <= ke; ++k) {
        for (int j = js; j <= je; ++j) {
          for (int i = is; i <= ie; ++i) {

            // Extract density and pressure
            Real rho = prim_local(IDN,k,j,i);
            Real pgas = prim_local(IPR,k,j,i);

            // Locate density and pressure in table
            int rho_index;
            Real rho_fraction;
            FindIndex(rho_table, rho, &rho_index, &rho_fraction);
            int pgas_index;
            Real pgas_fraction;
            FindIndex(pgas_table, pgas, &pgas_index, &pgas_fraction);

            // Interpolate and store cooling
            Real edot_cool = Interpolate2D(edot_cool_table, rho_index, pgas_index, rho_fraction,
                pgas_fraction);
            Real u = cons_local(IEN,k,j,i) - (SQR(cons_local(IM1,k,j,i))
                + SQR(cons_local(IM2,k,j,i)) + SQR(cons_local(IM3,k,j,i)))
                / (2.0 * cons_local(IDN,k,j,i));
            Real delta_e = std::min(edot_cool * dt, u);
            user_out_var_local(0,k,j,i) = edot_cool;
            m[0] += delta_e * vol_cell;
            m[1] += Interpolate2D(temperature_table, rho_index, pgas_index, rho_fraction, pgas_fraction);
            m[2] += 1.0;
          }
        }
      }
      // Prepare for next block on mesh
      prim_local.DeleteAthenaArray();
      cons_local.DeleteAthenaArray();
      user_out_var_local.DeleteAthenaArray();
      pblock = pblock->next;
    }
#ifdef MPI_PARALLEL
    MPI_Allreduce(m, gm, 3, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#else
    gm[0] = m[0];
    gm[1] = m[1];
    gm[2] = m[2];
#endif
    delta_e_tot = gm[0];
    Average_T = gm[1]/gm[2];
  }

  // determine if cooling should start
  if ((not cooling_on) and (Average_T > T_cool_start)){
    cooling_on = true;
    if(Globals::my_rank==0) {
      std::cout << "Time to start cooling!! Average_T = " << Average_T << "\n";
    }
  }

  // Store or extract redistributed heating
  if (pmb->lid == 0) {
    pmb->pmy_mesh->ruser_mesh_data[4](0) = delta_e_tot;
  } else {
    delta_e_tot = pmb->pmy_mesh->ruser_mesh_data[4](0);
  }


  Real &dedt = pmb->pmy_mesh->ptrbd->dedt;

  // Calculate amount of heat to redistribute
  Real delta_e_redist = 0.0;
  if (heat_redistribute) {
    delta_e_redist = delta_e_tot / vol_tot;
    if (constant_energy) {
      delta_e_redist = std::max(delta_e_redist - dedt/vol_tot * dt, 0.0);
    }
    if(Globals::my_rank==0) {
      std::cout << "heat_redistribute delta_e_tot / vol_tot = " << delta_e_tot / vol_tot << " dedt * dt " << dedt/ vol_tot * dt << "\n";
    }
  }

  if (adaptive_driving) {
    dedt = std::max(delta_e_tot / dt, 0.0);
    if(Globals::my_rank==0) {
      std::cout << "adaptive_driving dedt " << dedt << "\n";
    }
  }

  // if (t%drive_separation <= drive_duration){  }

  if (tophat_driving){
    dedt = dedt_on * (0.5*(1.0+std::tanh((fmod(t,drive_separation)-((drive_separation-drive_duration)/2.))/(drive_separation/100.))) 
                    - 0.5*(1.0+std::tanh((fmod(t,drive_separation)-((drive_separation+drive_duration)/2.))/(drive_separation/100.))));
    if(Globals::my_rank==0) {
      std::cout << "tophat_driving dedt " << dedt << "\n";
    }
  }

  // Extract indices
  int is = pmb->is;
  int ie = pmb->ie;
  int js = pmb->js;
  int je = pmb->je;
  int ks = pmb->ks;
  int ke = pmb->ke;

  // Extract arrays
  AthenaArray<Real> edot_cool, temperature;
  edot_cool.InitWithShallowSlice(pmb->user_out_var, 4, 0, 1);
  temperature.InitWithShallowSlice(pmb->user_out_var, 4, 1, 1);

  // Apply all source terms
  Real delta_e_block = 0.0;
  Real delta_e_ceil = 0.0;
  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is; i <= ie; ++i) {

        // Extract primitive and conserved quantities
        const Real &rho_half = prim(IDN,k,j,i);
        const Real &pgas_half = prim(IPR,k,j,i);
        const Real &rho = cons(IDN,k,j,i);
        Real &e = cons(IEN,k,j,i);
        Real &m1 = cons(IM1,k,j,i);
        Real &m2 = cons(IM2,k,j,i);
        Real &m3 = cons(IM3,k,j,i);

        // Locate density and pressure in table
        int rho_index;
        Real rho_fraction;
        FindIndex(rho_table, rho_half, &rho_index, &rho_fraction);
        int pgas_index;
        Real pgas_fraction;
        FindIndex(pgas_table, pgas_half, &pgas_index, &pgas_fraction);

        // Interpolate and store temperature
        temperature(k,j,i) = Interpolate2D(temperature_table, rho_index, pgas_index,
            rho_fraction, pgas_fraction);

        // Apply cooling and heating
        Real delta_e = -edot_cool(k,j,i) * dt;
        if (rho_dependent_heat_redistribute){
          delta_e += delta_e_redist * rho/rho_0;
        } else {
          delta_e += delta_e_redist;
        }
        Real kinetic = (SQR(m1) + SQR(m2) + SQR(m3)) / (2.0 * rho);
        Real u = e - kinetic;
        delta_e = std::max(delta_e, -u);
        if (cooling_on){
          e += delta_e;
        }
        if (stage == 2) {
          delta_e_block += delta_e;
        }

        // Apply temperature ceiling
        Real u_max = temperature_max * rho / (gamma_adi-1.0);
        kinetic = (SQR(m1) + SQR(m2) + SQR(m3)) / (2.0 * rho);
        u = e - kinetic;
        if (u > u_max) {
          e = kinetic + u_max;
          if (stage == 2) {
            delta_e_ceil += (u - u_max) * vol_cell;
          }
        }
      }
    }
  }
  pmb->ruser_meshblock_data[0](0) += delta_e_block;
  pmb->ruser_meshblock_data[0](1) += delta_e_ceil;
  pmb->ruser_meshblock_data[0](2) = Average_T;

  // Free arrays
  rho_table.DeleteAthenaArray();
  pgas_table.DeleteAthenaArray();
  edot_cool_table.DeleteAthenaArray();
  temperature_table.DeleteAthenaArray();
  edot_cool.DeleteAthenaArray();
  temperature.DeleteAthenaArray();

  // gravity 
  if (gravity_on){
    for (int k=pmb->ks; k<=pmb->ke; ++k) {
      for (int j=pmb->js; j<=pmb->je; ++j) {
        for (int i=pmb->is; i<=pmb->ie; ++i) {
          Real delta_vz = dt*grav_accel;
          cons(IM3,k,j,i) -= prim(IDN,k,j,i) * delta_vz;
          if (NON_BAROTROPIC_EOS) cons(IEN,k,j,i) -= 0.5*prim(IDN,k,j,i)*(2.*delta_vz*prim(IVZ,k,j,i) - SQR(delta_vz));
        }
      }
    }
  }

  return;
}

//----------------------------------------------------------------------------------------
// Cooling losses
// Inputs:
//   pmb: pointer to MeshBlock
//   iout: index of history output
// Outputs:
//   returned value: sum of all energy losses due to different cooling mechanisms
// Notes:
//   resets time-integrated values to 0
//   cooling mechanisms are:
//     0: physical radiative losses
//     1: numerical temperature ceiling
//     2: average temperature

Real CoolingLosses(MeshBlock *pmb, int iout)
{
  Real delta_e = pmb->ruser_meshblock_data[0](iout);
  pmb->ruser_meshblock_data[0](iout) = 0.0;
  return delta_e;
}

//----------------------------------------------------------------------------------------
// 1D index location function
// Inputs:
//   array: 1D array of monotonically increasing values of length at least 2
//   value: value to locate in array
// Outputs:
//   index,fraction: value is located at fraction of the way from array(index) to
//       array(index+1)

static void FindIndex(const AthenaArray<double> &array, Real value, int *p_index,
    Real *p_fraction)
{
  int array_length = array.GetDim1();
  if (value <= array(0)) {
    *p_index = 0;
    *p_fraction = 0.0;
  } else if (value >= array(array_length-1)) {
    *p_index = array_length - 2;
    *p_fraction = 1.0;
  } else {
    for (int i = 1; i < array_length; ++i) {
      if (array(i) > value) {
        *p_index = i - 1;
        *p_fraction = (value - array(i-1)) / (array(i) - array(i-1));
        break;
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
// 2D table interpolation function
// Inputs:
//   table: 2D array data
//   j,i: indices on low side of desired value
//   g,f: corresponding fractions desired value is toward next index
// Outputs:
//   returned value: linearly interpolated value from table

static Real Interpolate2D(const AthenaArray<double> &table, int j, int i, Real g, Real f)
{
  Real val_00 = table(j,i);
  Real val_01 = table(j,i+1);
  Real val_10 = table(j+1,i);
  Real val_11 = table(j+1,i+1);
  Real val = (1.0-g)*(1.0-f)*val_00 + (1.0-g)*f*val_01 + g*(1.0-f)*val_10 + g*f*val_11;
  return val;
}

//----------------------------------------------------------------------------------------
// 3D table interpolation function
// Inputs:
//   table: 3D array data
//   k,j,i: indices on low side of desired value
//   h,g,f: corresponding fractions desired value is toward next index
// Outputs:
//   returned value: linearly interpolated value from table

static Real Interpolate3D(const AthenaArray<double> &table, int k, int j, int i, Real h,
    Real g, Real f)
{
  Real val_000 = table(k,j,i);
  Real val_001 = table(k,j,i+1);
  Real val_010 = table(k,j+1,i);
  Real val_011 = table(k,j+1,i+1);
  Real val_100 = table(k+1,j,i);
  Real val_101 = table(k+1,j,i+1);
  Real val_110 = table(k+1,j+1,i);
  Real val_111 = table(k+1,j+1,i+1);
  Real val = (1.0-h)*(1.0-g)*(1.0-f)*val_000 + (1.0-h)*(1.0-g)*f*val_001
      + (1.0-h)*g*(1.0-f)*val_010 + (1.0-h)*g*f*val_011 + h*(1.0-g)*(1.0-f)*val_100
      + h*(1.0-g)*f*val_101 + h*g*(1.0-f)*val_110 + h*g*f*val_111;
  return val;
}
