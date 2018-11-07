// Problem generator for realistic cooling

// C++ headers
#include <algorithm>  // min()
#include <cmath>      // abs(), pow(), sqrt()
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // string
#include <fstream>

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

// Declarations
void SourceFunction(MeshBlock *pmb, const Real t, const Real dt,
    const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc, AthenaArray<Real> &cons);
Real CoolingLosses(MeshBlock *pmb, int iout);
static void FindIndex(const AthenaArray<double> &array, Real value, int *p_index,
    Real *p_fraction);
static Real Interpolate1D(const AthenaArray<double> &table, int i, Real f);
static Real Interpolate2D(const AthenaArray<double> &table, int j, int i, Real g, Real f);
static Real Interpolate3D(const AthenaArray<double> &table, int k, int j, int i, Real h,
    Real g, Real f);
static Real grav_accel(Real r);
static Real grav_pot(Real r);

// Global variables
static const Real mu_m_h = 1.008 * 1.660539040e-24;
static Real gamma_adi;
static Real temperature_max;
static Real rho_table_min, rho_table_max, rho_table_n;
static Real pgas_table_min, pgas_table_max, pgas_table_n;
static Real t_cool_start;
static bool heat_redistribute, constant_energy;
static bool adaptive_driving, tophat_driving;
static Real drive_duration, drive_separation, dedt_on;
static Real beta;

static Real cooling_timestep(MeshBlock *pmb);
static int turb_grid_size;
static Real kappa;
static bool conduction_on;
static Real dt_cutoff, cfl_cool;

static Real Mhalo, cnfw, GMhalo, rvir;
static Real rho_wind, v_wind, cs_wind;
static Real rho_igm, v_igm, cs_igm, Mdot_igm;
static Real r_inner, r_outer;
static Real f_shock, f_core;

void WindX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke);
void NoInflowOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke);
void IGMAccretionOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke);

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
// turb_flag = 3 for density perturbations
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

  if (COORDINATE_SYSTEM != "spherical_polar"){
    std::stringstream msg;
    msg << "### FATAL ERROR can only be used with spherical_polar coordiantes " << std::endl;
    throw std::runtime_error(msg.str().c_str());
    return;
  }


  // Read general parameters from input file
  gamma_adi = pin->GetReal("hydro", "gamma");
  Real length_scale = pin->GetReal("problem", "length_scale");
  Real rho_scale = pin->GetReal("problem", "rho_scale");
  Real pgas_scale = pin->GetReal("problem", "pgas_scale");
  Real temperature_scale = pin->GetReal("problem", "temperature_scale");
  Real vel_scale = std::sqrt(pgas_scale / rho_scale);
  Real time_scale = length_scale/vel_scale;
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

  // cooling 
  t_cool_start = pin->GetReal("problem", "t_cool_start");
  dt_cutoff = pin->GetOrAddReal("problem", "dt_cutoff", 1.0e-6);
  cfl_cool = pin->GetOrAddReal("problem", "cfl_cool", 0.1);

  Real kpc  = 3.08567758096e+21;
  Real G    = 6.67384e-08;
  Real mp   = 1.67373522381e-24; 
  Real Msun = 2.0e33;
  Real kb   = 1.3806488e-16;
  Real yr   = 31557600.0;
  // Gravity
  Mhalo        = pin->GetReal("problem", "Mhalo"); // in Msun
  cnfw         = pin->GetReal("problem", "cnfw"); 
  rvir         = pin->GetReal("problem", "rvir"); // in kpc
  rvir         = rvir * kpc/length_scale;
  GMhalo       = (G * Mhalo * Msun) / (pow(length_scale,3)/SQR(time_scale));

  // ICs
  f_shock      = pin->GetReal("problem", "f_shock");  //transition from free fall outside to shocked inside 
  f_core       = pin->GetReal("problem", "f_core");   //transition from isentropic inside to isothermal outside
  r_inner      = mesh_size.x1min;
  r_outer      = mesh_size.x1max;
  // Outflows
  Real Mdot_out = pin->GetReal("problem", "Mdot_out")*(Msun/yr)/(rho_scale*pow(length_scale,3)/time_scale); // in Msun/yr
  v_wind        = pin->GetReal("problem", "v_wind")*1.0e5/vel_scale;  // in km/s
  rho_wind      = Mdot_out/(4*PI*SQR(r_inner)*v_wind);
  Real T_wind   = pin->GetReal("problem", "T_wind");
  cs_wind       = sqrt(kb*T_wind/(0.62*mp))/vel_scale;
  // IGM accretion
  Mdot_igm      = pin->GetReal("problem", "Mdot_igm")*(Msun/yr)/(rho_scale*pow(length_scale,3)/time_scale); // in Msun/yr
  v_igm         = pin->GetReal("problem", "v_igm")*1.0e5/vel_scale;  // in km/s
  rho_igm       = Mdot_igm/(4*PI*SQR(r_outer)*v_igm);
  Real T_igm    = pin->GetReal("problem", "T_igm");
  cs_igm        = sqrt(kb*T_igm/(0.62*mp))/vel_scale;

  if(Globals::my_rank==0) {
    std::cout << " Mhalo = " << Mhalo << "\n";
    std::cout << " cnfw = " << cnfw << "\n";
    std::cout << " rvir = " << rvir << "\n";
    std::cout << " GMhalo = " << GMhalo << "\n";
    std::cout << " f_core = " << f_core << "\n";
    std::cout << " r_inner = " << r_inner << "\n";
    std::cout << " r_outer = " << r_outer << "\n";
    std::cout << " Mdot_out = " << Mdot_out << "\n";
    std::cout << " rho_wind = " << rho_wind << "\n";
    std::cout << " T_wind = " << T_wind << "\n";
    std::cout << " cs_wind = " << cs_wind << "\n";
    std::cout << " Mdot_igm = " << Mdot_igm << "\n";
    std::cout << " v_igm = " << v_igm << "\n";
    std::cout << " rho_igm = " << rho_igm << "\n";
    std::cout << " T_igm = " << T_igm << "\n";
    std::cout << " cs_igm = " << cs_igm << "\n";
  }




  if (MAGNETIC_FIELDS_ENABLED) beta = pin->GetOrAddReal("problem", "beta", 1.0e10);

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
  ruser_mesh_data[4].NewAthenaArray(2); // delta_e_tot, vol_tot

  Real vol_tot = (mesh_size.x3max - mesh_size.x3min)
                 * (std::cos(mesh_size.x2max) - std::cos(mesh_size.x2min))
                 * (pow(mesh_size.x1max,3) - pow(mesh_size.x1min,3))/3.;
  ruser_mesh_data[4](1) = vol_tot;



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
  EnrollUserExplicitSourceFunction(SourceFunction);
  AllocateUserHistoryOutput(2);
  EnrollUserHistoryOutput(0, CoolingLosses, "e_cool");
  EnrollUserHistoryOutput(1, CoolingLosses, "e_ceil");
  EnrollUserTimeStepFunction(cooling_timestep);

  // Enroll no inflow boundary condition but only if it is turned on
  if(mesh_bcs[OUTER_X1] == GetBoundaryFlag("user")) {
    if (Mdot_igm <= 0.0) {
      EnrollUserBoundaryFunction(OUTER_X1, NoInflowOuterX1);
    } else {
      EnrollUserBoundaryFunction(OUTER_X1, IGMAccretionOuterX1);
    }
  }
  if(mesh_bcs[INNER_X1] == GetBoundaryFlag("user")) {
    if (Mdot_out > 0.){
      EnrollUserBoundaryFunction(INNER_X1, WindX1);
    } // else {
      // DO NOTHING
    // }
  }
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
  ruser_meshblock_data[0].NewAthenaArray(2);
  ruser_meshblock_data[0](0) = 0.0;
  ruser_meshblock_data[0](1) = 0.0;

  // Set output variables
  AllocateUserOutputVariables(2);
  // SetUserOutputVariableName(0, "edot_cool");
  // SetUserOutputVariableName(1, "T");
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


  Real length_scale = pin->GetReal("problem", "length_scale");
  Real rho_scale = pin->GetReal("problem", "rho_scale");
  Real pgas_scale = pin->GetReal("problem", "pgas_scale");
  Real temperature_scale = pin->GetReal("problem", "temperature_scale");
  Real vel_scale = std::sqrt(pgas_scale / rho_scale);


  AthenaArray<Real> interp_r_rvir, interp_v, interp_rho, interp_press;  // table for analytic solution
  
  // Allocate array for interpolation points
  int num_lines = pin->GetInteger("problem", "num_data_lines");
  interp_r_rvir.NewAthenaArray(num_lines);
  interp_v.NewAthenaArray(num_lines);
  interp_rho.NewAthenaArray(num_lines);
  interp_press.NewAthenaArray(num_lines);
  
  // Read interpolation data from file
  std::string filename = pin->GetString("problem", "IC_file");
  std::ifstream file(filename.c_str());
  if (not file.is_open()) {
    std::stringstream msg;
    msg << "### FATAL ERROR in Problem Generator\n"
        << "file " << filename << " cannot be opened" << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }
  Real r_rvir, v, rho, press;
  for (int n = 0; n < num_lines; ++n) {
    file >> r_rvir >> v >> rho >> press;
    interp_r_rvir(n) = r_rvir;
    interp_v(n) = v/vel_scale;
    interp_rho(n) = rho/rho_scale;
    interp_press(n) = press/pgas_scale;
  }


  Real rshock      = interp_r_rvir(num_lines-1) * rvir;
  Real phishock    = fabs(grav_pot(rshock));
  Real phita       = fabs(grav_pot(r_outer));
   
  // -- pre-shock, and... 
  Real v_ff        = sqrt(2.0*(phishock-phita) + SQR(v_igm));
  Real rho_inflow  = Mdot_igm/(4.*PI*SQR(rshock)*v_ff);
  
  // -- ...post-shock 
  Real v_post_shock = (2./3.) * v_ff * ( 1. + sqrt(1. + (15./4.)*(SQR(cs_igm)/SQR(v_ff))));
  Real rhoshock     = 4./(1. + (5.*SQR(cs_igm)/SQR(v_post_shock))) * rho_inflow; 
  Real Pshock       = ((3./4.)*SQR(v_post_shock) - SQR(cs_igm)/4.) * rho_inflow;
  
  if(Globals::my_rank==0) {
    std::cout << " rshock = " << rshock << " v_post_shock = " << v_post_shock << 
                 " rhoshock = " << rhoshock << " Pshock = " << Pshock << 
    std::cout << " from IC file, v_post_shock = " << interp_v(num_lines-1) << 
                 " rhoshock = " << interp_rho(num_lines-1) << " Pshock = " << interp_press(num_lines-1) << "\n";
  }

  Real velocity_factor = pin->GetOrAddReal("problem", "velocity_factor", 1.0);

  // Initialize primitive values
  for (int k = kl; k <= ku; ++k) {
    // Real phi = pcoord->x3v(k);
    for (int j = jl; j <= ju; ++j) {
      // Real theta = pcoord->x2v(j);
      for (int i = il; i <= iu; ++i) {
        Real r = pcoord->x1v(i);
        Real phi = fabs(grav_pot(r));
        if (r > rshock){
          v       = sqrt(2.0*(phi-phita) + SQR(v_igm));
          rho     = Mdot_igm/(4.*PI*SQR(r)*v);
          press   = SQR(cs_igm) * rho; 
        } else {
          // Locate r/rvir in tables
          int r_rvir_index;
          Real r_rvir_fraction;
          FindIndex(interp_r_rvir, r/rvir, &r_rvir_index,&r_rvir_fraction);
          // interpolate
          v       = velocity_factor * Interpolate1D(interp_v, r_rvir_index, r_rvir_fraction);
          rho     = Interpolate1D(interp_rho, r_rvir_index, r_rvir_fraction);
          press   = Interpolate1D(interp_press, r_rvir_index, r_rvir_fraction);
        }
        phydro->w(IDN,k,j,i) = rho;
        phydro->w(IPR,k,j,i) = press;
        phydro->w(IVX,k,j,i) = -1.*v;
        phydro->w(IVY,k,j,i) = 0.0;
        phydro->w(IVZ,k,j,i) = 0.0;
// Configuration checking
#if MAGNETIC_FIELDS_ENABLED
        pfield->b.x1f(k,j,i) = 0.0;
        pfield->b.x2f(k,j,i) = 0.0;
        pfield->b.x3f(k,j,i) = sqrt(8*PI*P/beta); // beta = P_Th/P_Mag ==> P_Mag = P_Th / beta ==> B = sqrt(8 pi P_th / beta )
#endif
      }
    }
  }


  // Calculate cell-centered magnetic field
  AthenaArray<Real> bb;
  bb.NewAthenaArray(3, ku+1, ju+1, iu+1);
  if (MAGNETIC_FIELDS_ENABLED) {
    pfield->CalculateCellCenteredField(pfield->b, bb, pcoord, il, iu, jl, ju, kl, ku);
  }
  // Initialize conserved values
  peos->PrimitiveToConserved(phydro->w, bb, phydro->u, pcoord, il, iu, jl, ju, kl, ku);
  bb.DeleteAthenaArray();

  interp_r_rvir.DeleteAthenaArray();
  interp_v.DeleteAthenaArray();
  interp_rho.DeleteAthenaArray();
  interp_press.DeleteAthenaArray();

  if(Globals::my_rank==0) {
    std::cout << " problem generated \n";
  }

  return;
}


//----------------------------------------------------------------------------------------
// Function to calculate the timestep required to resolve cooling (and later conduction)
//          tcool = 3/2 P/Edot_cool
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
//   bcc: cell-centered magnetic fields
// Outputs:
//   cons: conserved variables updated
// Notes:
//   writes to user_out_var array the following quantities:
//     0: edot_cool
//     1: T
//     3: edot_cond

void SourceFunction(MeshBlock *pmb, const Real t, const Real dt,
    const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc, AthenaArray<Real> &cons)
{
  // Prepare values to aggregate
  // cooling
  Real delta_e_mesh = 0.0;
  Real delta_e_tot;
  Real vol_tot  = pmb->pmy_mesh->ruser_mesh_data[4](1);
 
  // gravity 
  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        Real r = pmb->pcoord->x1v(i);
        Real delta_vr = dt*grav_accel(r);
        cons(IM1,k,j,i) -= prim(IDN,k,j,i) * delta_vr;
        if (NON_BAROTROPIC_EOS) cons(IEN,k,j,i) -= 0.5*prim(IDN,k,j,i)*(2.*delta_vr*prim(IVX,k,j,i) - SQR(delta_vr));
      }
    }
  }


  // Extract data tables
  AthenaArray<Real> rho_table, pgas_table, edot_cool_table, temperature_table;
  rho_table.InitWithShallowCopy(pmb->pmy_mesh->ruser_mesh_data[0]);
  pgas_table.InitWithShallowCopy(pmb->pmy_mesh->ruser_mesh_data[1]);
  edot_cool_table.InitWithShallowCopy(pmb->pmy_mesh->ruser_mesh_data[2]);
  temperature_table.InitWithShallowCopy(pmb->pmy_mesh->ruser_mesh_data[3]);
  
  // Determine which part of step this is
  bool predict_step = prim.data() == pmb->phydro->w.data();

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
      if (predict_step) {
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

            // Interpolate and store cooling ------- Should this be done with RK2 or RK4????
            Real edot_cool = Interpolate2D(edot_cool_table, rho_index, pgas_index, rho_fraction,
                pgas_fraction);
            Real u = cons_local(IEN,k,j,i) - (SQR(cons_local(IM1,k,j,i))
                + SQR(cons_local(IM2,k,j,i)) + SQR(cons_local(IM3,k,j,i)))
                / (2.0 * cons_local(IDN,k,j,i));
            Real delta_e = std::min(edot_cool * dt, u);
            user_out_var_local(0,k,j,i) = edot_cool;
            // Real vol_cell = (pcoord->x3f(k+1)-pcoord->x3f(k))
            //                 * (std::cos(pcoord->x2f(j+1)) - std::cos(pcoord->x2f(j)))
            //                 * (pow(pcoord->x1f(i+1),3) - pow(pcoord->x1f(i),3))/3.;
            Real vol_cell = pmb->pcoord->GetCellVolume(k,j,i);
            delta_e_mesh += delta_e * vol_cell;
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
    MPI_Allreduce(&delta_e_mesh, &delta_e_tot, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#else
    delta_e_tot = delta_e_mesh;
#endif
  }

  // Store or extract redistributed heating
  if (pmb->lid == 0) {
    pmb->pmy_mesh->ruser_mesh_data[4](0) = delta_e_tot;
  } else {
    delta_e_tot = pmb->pmy_mesh->ruser_mesh_data[4](0);
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
        Real delta_e = -edot_cool(k,j,i) * dt; // <------- Should this be done with RK2 or RK4????
        Real kinetic = (SQR(m1) + SQR(m2) + SQR(m3)) / (2.0 * rho);
        Real u = e - kinetic;
        Real magnetic;
        if (MAGNETIC_FIELDS_ENABLED) {
          const Real &bcc1 = bcc(IB1,k,j,i);
          const Real &bcc2 = bcc(IB2,k,j,i);
          const Real &bcc3 = bcc(IB3,k,j,i);

          magnetic = 0.5*(SQR(bcc1) + SQR(bcc2) + SQR(bcc3));
          u -= magnetic; 
        }
        delta_e = std::max(delta_e, -u);
        if (t > t_cool_start){
          e += delta_e;
        }
        if (not predict_step) {
          delta_e_block += delta_e;
        }

        // Apply temperature ceiling
        Real u_max = temperature_max * rho / (gamma_adi-1.0);
        kinetic = (SQR(m1) + SQR(m2) + SQR(m3)) / (2.0 * rho);
        u = e - kinetic;
        if (MAGNETIC_FIELDS_ENABLED) {
          u -= magnetic; 
        }
        if (u > u_max) {
          e = kinetic + u_max;
          if (MAGNETIC_FIELDS_ENABLED) e += magnetic;
          if (not predict_step) {
            Real vol_cell = pmb->pcoord->GetCellVolume(k,j,i);
            delta_e_ceil += (u - u_max);
          }
        }
      }
    }
  }
  pmb->ruser_meshblock_data[0](0) += delta_e_block;
  pmb->ruser_meshblock_data[0](1) += delta_e_ceil;

  // Free arrays
  rho_table.DeleteAthenaArray();
  pgas_table.DeleteAthenaArray();
  edot_cool_table.DeleteAthenaArray();
  temperature_table.DeleteAthenaArray();
  edot_cool.DeleteAthenaArray();
  temperature.DeleteAthenaArray();

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

Real CoolingLosses(MeshBlock *pmb, int iout)
{
  Real delta_e = pmb->ruser_meshblock_data[0](iout);
  pmb->ruser_meshblock_data[0](iout) = 0.0;
  return delta_e;
}



//----------------------------------------------------------------------------------------
// gravitational acceleration
// maybe I will need to put in a smooth transition to 0 at the outer boundary
static Real grav_accel(Real r)
{
  Real f = 1.0/(log(1.0+cnfw)/cnfw - 1.0/(1.0+cnfw));
  Real x = (r/rvir); 
  return -1.0*f*GMhalo/SQR(rvir) * (1.0/(x*(cnfw*x+1.0)) - std::log(cnfw*x+1.0)/(cnfw*SQR(x)));
}


//----------------------------------------------------------------------------------------
// gravitational potential
// maybe I will need to put in a smooth transition to 0 at the outer boundary
static Real grav_pot(Real r)
{
  Real f = 1.0/(log(1.0+cnfw)/cnfw - 1.0/(1.0+cnfw));
  return -1.0 * GMhalo/rvir * f * std::log(1.0+(cnfw*r/rvir))/(cnfw*r/rvir);
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
// 1D table interpolation function
// Inputs:
//   table: 1D array data
//   j,i: indices on low side of desired value
//   g,f: corresponding fractions desired value is toward next index
// Outputs:
//   returned value: linearly interpolated value from table

static Real Interpolate1D(const AthenaArray<double> &table, int i, Real f)
{
  Real val_0 = table(i);
  Real val_1 = table(i+1);
  Real val = (1.0-f)*val_0 + f*val_1;
  return val;
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














//----------------------------------------------------------------------------------------
//! \fn void WindX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
//                          FaceField &b, Real time, Real dt,
//                          int is, int ie, int js, int je, int ks, int ke)
//  \brief Wind boundary conditions with no inflow, inner x1 boundary

void WindX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke)
{
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=1; i<=(NGHOST); ++i) {
        prim(IDN,k,j,is-i) = rho_wind;
        prim(IVX,k,j,is-i) = v_wind;
        prim(IVY,k,j,is-i) = 0.0;
        prim(IVZ,k,j,is-i) = 0.0;
        prim(IPR,k,j,is-i) = rho_wind*SQR(cs_wind);
#if MAGNETIC_FIELDS_ENABLED
        b.x1f(k,j,is-i) = 0.0;
        b.x2f(k,j,is-i) = 0.0;
        b.x3f(k,j,is-i) = sqrt(8*PI*rho_wind*SQR(cs_wind)/beta); // beta = P_Th/P_Mag ==> P_Mag = P_Th / beta ==> B = sqrt(8 pi P_th / beta )
#endif
      }
    }
  }
  return;
}


//----------------------------------------------------------------------------------------
//! \fn void NoInflowOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
//                          FaceField &b, Real time, Real dt,
//                          int is, int ie, int js, int je, int ks, int ke)
//  \brief OUTFLOW boundary conditions with no inflow, outer x3 boundary

void NoInflowOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke)
{
  // copy hydro variables into ghost zones
  for (int n=0; n<(NHYDRO); ++n) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=1; i<=(NGHOST); ++i) {
          prim(n,k,j,ie+i) = prim(n,k,j,ie);  
          if ( n == IVX ){
            if ( prim(n,k,j,ie+i) < 0.0 ){
              prim(n,k,j,ie+i) = 0.0;
            }
          } 
        }
      }
    }
  }

  // copy face-centered magnetic fields into ghost zones
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma simd
      for (int i=1; i<=(NGHOST); ++i) {
        b.x1f(k,j,(ie+i+1)) = b.x1f(k,j,(ie+1));
      }
    }}

    for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je+1; ++j) {
#pragma simd
      for (int i=1; i<=(NGHOST); ++i) {
        b.x2f(k,j,(ie+i)) = b.x2f(k,j,ie);
      }
    }}

    for (int k=ks; k<=ke+1; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma simd
      for (int i=1; i<=(NGHOST); ++i) {
        b.x3f(k,j,(ie+i)) = b.x3f(k,j,ie);
      }
    }}
  }


  return;
}


//----------------------------------------------------------------------------------------
//! \fn void NoInflowOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
//                          FaceField &b, Real time, Real dt,
//                          int is, int ie, int js, int je, int ks, int ke)
//  \brief OUTFLOW boundary conditions with no inflow, outer x3 boundary

void IGMAccretionOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke)
{
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=1; i<=(NGHOST); ++i) {
        prim(IDN,k,j,ie+i) = rho_igm;
        prim(IVX,k,j,ie+i) = -v_igm;
        prim(IVY,k,j,ie+i) = 0.0;
        prim(IVZ,k,j,ie+i) = 0.0;
        prim(IPR,k,j,ie+i) = rho_igm*SQR(cs_igm);
#if MAGNETIC_FIELDS_ENABLED
        b.x1f(k,j,ie+i) = 0.0;
        b.x2f(k,j,ie+i) = 0.0;
        b.x3f(k,j,ie+i) = sqrt(8*PI*rho_igm*SQR(cs_igm)/beta); // beta = P_Th/P_Mag ==> P_Mag = P_Th / beta ==> B = sqrt(8 pi P_th / beta )
#endif
      }
    }
  }
  return;
}
