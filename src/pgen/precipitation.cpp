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
#include "../hydro/hydro_diffusion/hydro_diffusion.hpp" // diffusion

// External library headers
#include <hdf5.h>  // H5*, hid_t, hsize_t, H5*()

// Declarations
void SourceFunction(MeshBlock *pmb, const Real t, const Real dt,
    const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc, AthenaArray<Real> &cons, int stage);
Real history(MeshBlock *pmb, int iout);
static void FindIndex(const AthenaArray<double> &array, Real value, int *p_index,
    Real *p_fraction);
static Real Interpolate1D(const AthenaArray<double> &table, int i, Real f);
static Real Interpolate2D(const AthenaArray<double> &table, int j, int i, Real g, Real f);
static Real Interpolate3D(const AthenaArray<double> &table, int k, int j, int i, Real h,
    Real g, Real f);
static Real grav_accel(Real r);
static Real grav_pot(Real r);

// User defined conduction and viscosity
void SmagorinskyConduction(HydroDiffusion *phdif, MeshBlock *pmb, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &bcc, int is, int ie, int js, int je, int ks, int ke);
void SmagorinskyViscosity(HydroDiffusion *phdif, MeshBlock *pmb, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &bcc, int is, int ie, int js, int je, int ks, int ke);
void DeviatoricSmagorinskyConduction(HydroDiffusion *phdif, MeshBlock *pmb, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &bcc, int is, int ie, int js, int je, int ks, int ke);
void DeviatoricSmagorinskyViscosity(HydroDiffusion *phdif, MeshBlock *pmb, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &bcc, int is, int ie, int js, int je, int ks, int ke);

// Global variables
static const Real mu_m_h = 1.008 * 1.660539040e-24;
static Real gamma_adi;
static Real temperature_max;
static Real rho_table_min, rho_table_max, rho_table_n;
static Real pgas_table_min, pgas_table_max, pgas_table_n;
static Real t_cool_start;
static Real beta;
static Real grav_scale_inner, aaa, rs_rt, rhom, rho0;

static Real cooling_timestep(MeshBlock *pmb);
static Real dt_cutoff, cfl_cool;
static Real pfloor, dfloor, vceil;

static Real Mhalo, cnfw, GMhalo, rvir, r200m, Mgal, GMgal, Rgal;
static Real rho_wind, v_wind, cs_wind, eta, opening_angle;
static Real cs, rho_ta, f_cs;
static Real Mdot_in, Mdot_out;

static Real r_inner, r_outer;
static Real alpha_wind;
static bool density_weighted_winds, volume_weighted_winds;
static Real t_feedback_start;

void ExtrapInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
void AdaptiveWindX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
void ExtrapOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
void ConstantOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);

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
  dt_cutoff = pin->GetOrAddReal("problem", "dt_cutoff", 0.0);
  cfl_cool = pin->GetOrAddReal("problem", "cfl_cool", 0.1);

  Real kpc  = 3.08567758096e+21;
  Real G    = 6.67384e-08;
  Real mp   = 1.67373522381e-24; 
  Real Msun = 2.0e33;
  Real kb   = 1.3806488e-16;
  Real yr   = 31557600.0;
  // ICs
  r_inner      = mesh_size.x1min;
  r_outer      = mesh_size.x1max;
  f_cs         = pin->GetOrAddReal("problem","f_cs",0.6);
  rho_ta       = pin->GetReal("problem", "rho_ta")/rho_scale;
  
  // floor
  dfloor       = pin->GetReal("hydro", "dfloor");
  pfloor       = pin->GetReal("hydro", "pfloor");
  vceil       = pin->GetReal("hydro", "vceil");
  
  // Outflows
  v_wind        = pin->GetOrAddReal("problem", "v_wind", 300.0)*1.0e5/vel_scale;  // in km/s
  Real T_wind   = pin->GetOrAddReal("problem", "T_wind", 1.0e4);
  cs_wind       = sqrt(kb*T_wind/(0.62*mp))/vel_scale;
  eta           = pin->GetOrAddReal("problem", "eta", 0.0);
  opening_angle = pin->GetOrAddReal("problem", "opening_angle", PI/2.); // half opening angle, symmetric about equator
  alpha_wind    = pin->GetOrAddReal("problem", "alpha_wind", -2.0); 
  t_feedback_start = pin->GetReal("problem", "t_feedback_start");

  density_weighted_winds = pin->GetOrAddBoolean("problem", "density_weighted_winds", false);
  volume_weighted_winds = pin->GetOrAddBoolean("problem", "volume_weighted_winds", false);

  // Gravity
  Mhalo        = pin->GetReal("problem", "Mhalo"); // in Msun
  Mgal         = pin->GetReal("problem", "Mgal"); // in Msun
  Rgal         = pin->GetReal("problem", "Rgal") * kpc/length_scale; // effective radius entered in kpc and converted to code units and hernquist length
  cnfw         = pin->GetReal("problem", "cnfw"); 
  rvir         = pin->GetReal("problem", "rvir"); // in kpc
  rvir         = rvir * kpc/length_scale;
  r200m        = pin->GetReal("problem", "r200m"); // in kpc
  r200m        = r200m * kpc/length_scale;
  GMhalo       = (G * Mhalo * Msun) / (pow(length_scale,3)/SQR(time_scale));
  GMgal        = (G * Mgal * Msun) / (pow(length_scale,3)/SQR(time_scale));
  Real FourPiG = 8.385539110961876e-07;
  Real H0      = 2.1950745301360983e-18; // 2.268308489954634e-18;
  Real Om      = 0.3075; // 0.27;
  Real redshift= 0.0;
  rhom         = (3. * SQR(H0) * Om * pow(1.+redshift,3))/(2.*FourPiG) / rho_scale;
  Real rhoc    = (3. * SQR(H0))/(2.*FourPiG) / rho_scale;
  Real rs      = rvir/cnfw;
  rho0         = Mhalo * Msun / (4. * PI * pow(rs,3) * ( std::log(1.+cnfw) - cnfw/(1.+cnfw) )) / (rho_scale * pow(length_scale,3));
  Real nu      = pin->GetReal("problem", "nu"); // Diemer+14 figure 1
  Real rt      = (1.9-0.18*nu)*r200m;

  grav_scale_inner = FourPiG*rs*(SQR(time_scale)*rho_scale);

  aaa = 5. * cnfw * r200m / rvir;
  rs_rt = rs/rt;

  Real x_outer = r_outer/rvir; 

  if(Globals::my_rank==0) {
    std::cout << " Mhalo = " << Mhalo << "\n";
    std::cout << " Mgal = " << Mgal << "\n";
    std::cout << " cnfw = " << cnfw << "\n";
    std::cout << " rvir = " << rvir << "\n";
    std::cout << " Rgal = " << Rgal << "\n";
    std::cout << " GMhalo = " << GMhalo << "\n";
    std::cout << " GMgal = " << GMgal << "\n";
    std::cout << " r_inner = " << r_inner << "\n";
    std::cout << " r_outer = " << r_outer << "\n";
    std::cout << " grav_scale_inner = " << grav_scale_inner << "\n";
    std::cout << " rho0 = " << rho0 * rho_scale << "\n";
    std::cout << " rhom = " << rhom * rho_scale << "\n";
    std::cout << " aaa = " << aaa << "\n";
    std::cout << " rs_rt = " << rs_rt << "\n";
    std::cout << " g_in = " << grav_accel(r_inner) << "\n";
    std::cout << " g_vir = " << grav_accel(rvir) << "\n";
    std::cout << " g_out = " << grav_accel(r_outer) << "\n";
    std::cout << " n_ta = " << rho_ta * rho_scale/0.62/mp << "\n";
    std::cout << " vc_ta = " << sqrt(grav_accel(r_outer) * r_outer )*length_scale/time_scale / 1e5 << "\n";
    std::cout << " P_ta = " << rho_ta * rho_scale/kb * f_cs * (grav_accel(r_outer) * r_outer )*SQR(length_scale/time_scale) << "\n";
    std::cout << " dt_cutoff = " << dt_cutoff << "\n";
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
  AllocateRealUserMeshDataField(4);
  ruser_mesh_data[0].NewAthenaArray(rho_table_n);
  ruser_mesh_data[1].NewAthenaArray(pgas_table_n);
  ruser_mesh_data[2].NewAthenaArray(rho_table_n, pgas_table_n);
  ruser_mesh_data[3].NewAthenaArray(rho_table_n, pgas_table_n);

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


  // Enroll user-defined conduction and viscosity
  bool SmagorinskyViscosity_on = pin->GetOrAddBoolean("problem", "SmagorinskyViscosity_on", false);
  bool SmagorinskyConduction_on = pin->GetOrAddBoolean("problem", "SmagorinskyConduction_on", false);

  if (SmagorinskyViscosity_on){
    EnrollViscosityCoefficient(SmagorinskyViscosity);
  }
  if (SmagorinskyConduction_on){
    EnrollConductionCoefficient(SmagorinskyConduction);
  }

  bool DeviatoricSmagorinskyViscosity_on = pin->GetOrAddBoolean("problem", "DeviatoricSmagorinskyViscosity_on", false);
  bool DeviatoricSmagorinskyConduction_on = pin->GetOrAddBoolean("problem", "DeviatoricSmagorinskyConduction_on", false);

  if (DeviatoricSmagorinskyViscosity_on){
    EnrollViscosityCoefficient(DeviatoricSmagorinskyViscosity);
  }
  if (DeviatoricSmagorinskyConduction_on){
    EnrollConductionCoefficient(DeviatoricSmagorinskyConduction);
  }

 
  // Enroll user-defined functions
  EnrollUserExplicitSourceFunction(SourceFunction);
  AllocateUserHistoryOutput(4);
  EnrollUserHistoryOutput(0 , history, "e_cool");
  EnrollUserHistoryOutput(1 , history, "e_ceil");
  EnrollUserHistoryOutput(2 , history, "Migal");
  EnrollUserHistoryOutput(3 , history, "Mogal");
  
  // Enroll user-defined time step constraint
  EnrollUserTimeStepFunction(cooling_timestep);

  // Enroll no inflow boundary condition but only if it is turned on
  if(mesh_bcs[OUTER_X1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(OUTER_X1, ConstantOuterX1);
  }
  if(mesh_bcs[INNER_X1] == GetBoundaryFlag("user")) {
    if (eta > 0.){
      if(Globals::my_rank==0) std::cout << " turning on AdaptiveWindX1 \n";
      EnrollUserBoundaryFunction(INNER_X1, AdaptiveWindX1);
    } else {
      EnrollUserBoundaryFunction(INNER_X1, ExtrapInnerX1);
    }
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
  ruser_meshblock_data[0].NewAthenaArray(4);
  ruser_meshblock_data[0](0) = 0.0; // edot cool
  ruser_meshblock_data[0](1) = 0.0; // edot ceiling
  ruser_meshblock_data[0](2) = 0.0; // mdot in 
  ruser_meshblock_data[0](3) = 0.0; // mdot out 
  

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


  Real vc_ta  = sqrt( grav_accel(r_outer) * r_outer );
  // Initialize primitive values
  for (int k = kl; k <= ku; ++k) {
    for (int j = jl; j <= ju; ++j) {
      for (int i = il; i <= iu; ++i) {
        Real r = pcoord->x1v(i);
        Real rho, press;
        Real vc = sqrt(grav_accel(r) * r );
        rho     = rho_ta * SQR(vc_ta/vc) * pow(r/r_outer,-1./f_cs);
        press   = f_cs * SQR(vc) * rho; 
        phydro->w(IDN,k,j,i) = rho;
        phydro->w(IPR,k,j,i) = press;
        phydro->w(IVX,k,j,i) = 0.0;
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
  return;
}


//----------------------------------------------------------------------------------------
// Function to calculate the timestep required to resolve cooling 
//          tcool = 3/2 P/Edot_cool
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
          if (temperature(k,j,i)>1.0e4){
            Real dt;
            Real edot = fabs(edot_cool(k,j,i));
            Real press = pmb->phydro->w(IPR,k,j,i);
            dt = cfl_cool * 1.5*press/edot;
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

void SourceFunction(MeshBlock *pmb, const Real t, const Real dt,
    const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc, AthenaArray<Real> &cons, int stage)
{
  // Extract indices
  int is = pmb->is;
  int ie = pmb->ie;
  int js = pmb->js;
  int je = pmb->je;
  int ks = pmb->ks;
  int ke = pmb->ke;

  // Extract data tables
  AthenaArray<Real> rho_table, pgas_table, edot_cool_table, temperature_table;
  rho_table.InitWithShallowCopy(pmb->pmy_mesh->ruser_mesh_data[0]);
  pgas_table.InitWithShallowCopy(pmb->pmy_mesh->ruser_mesh_data[1]);
  edot_cool_table.InitWithShallowCopy(pmb->pmy_mesh->ruser_mesh_data[2]);
  temperature_table.InitWithShallowCopy(pmb->pmy_mesh->ruser_mesh_data[3]);
  
  // Extract arrays
  AthenaArray<Real> edot_cool, temperature;
  edot_cool.InitWithShallowSlice(pmb->user_out_var, 4, 0, 1);
  temperature.InitWithShallowSlice(pmb->user_out_var, 4, 1, 1);

  // Apply all source terms
  Real delta_e_block = 0.0;
  Real delta_e_ceil = 0.0;
  Real mdot_local[2] = {0}, mdot_global[2];
  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is; i <= ie; ++i) {
        // gravitational acceleration
        Real r = pmb->pcoord->x1v(i);
        Real delta_vr = dt*grav_accel(r);
        cons(IM1,k,j,i) -= prim(IDN,k,j,i) * delta_vr;
        if (NON_BAROTROPIC_EOS) cons(IEN,k,j,i) -= 0.5*prim(IDN,k,j,i)*(2.*delta_vr*prim(IVX,k,j,i) - SQR(delta_vr));

        // Extract primitive and conserved quantities
        const Real &rho_half = prim(IDN,k,j,i);
        const Real &pgas_half = prim(IPR,k,j,i);
        const Real &rho = cons(IDN,k,j,i);
        const Real &vr = prim(IVX,k,j,i);
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

        // Interpolate and store temperature and edot_cool
        temperature(k,j,i) = Interpolate2D(temperature_table, rho_index, pgas_index,
            rho_fraction, pgas_fraction);
        edot_cool(k,j,i) = Interpolate2D(edot_cool_table, rho_index, pgas_index, 
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
        // don't let cooling give you a negative thermal energy
        delta_e = std::max(delta_e, -u + 2.0*pfloor/(gamma_adi-1.0));
        if (t > t_cool_start){
          e += delta_e;
        }
        // store how much energy is being removed for history file
        if (stage == 2) {
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
          if (stage == 2) {
            // Real vol_cell = pmb->pcoord->GetCellVolume(k,j,i);
            delta_e_ceil += (u - u_max);
          }
        }
        // calculate the inward and outward mass fluxes at the inner boundary
        if (pmb->pcoord->x1f(i) == r_inner){
          if (vr<0.){
            mdot_local[0] += pmb->pcoord->GetFace1Area(k,j,i)*rho*vr;
          } else {
            mdot_local[1] += pmb->pcoord->GetFace1Area(k,j,i)*rho*vr;
          }
        } 
      }
    }
  }

#ifdef MPI_PARALLEL
  MPI_Allreduce(mdot_local, mdot_global, 2, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#else
  mdot_global[0] = mdot_local[0];
  mdot_global[1] = mdot_local[1];
#endif

  pmb->ruser_meshblock_data[0](0) += delta_e_block;
  pmb->ruser_meshblock_data[0](1) += delta_e_ceil;
  pmb->ruser_meshblock_data[0](2) = mdot_global[0];
  pmb->ruser_meshblock_data[0](3) = mdot_global[1];
  Mdot_in = mdot_global[0];
  Mdot_out = mdot_global[1];
  // std::cout << " mdot = " << Mdot_in << "\n";

  if (t > t_feedback_start){
    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je; ++j) {
        for (int i = is; i <= ie; ++i) {
          // heating
          Real r = pmb->pcoord->x1v(i);
          Real &e = cons(IEN,k,j,i);
          Real &rho = cons(IDN,k,j,i);
          if (density_weighted_winds){
            e += -1.*mdot_global[0] * SQR(cs_wind) * pow(r/r_inner, alpha_wind) * rho;
          } 
          if (volume_weighted_winds){
            e += -1.*mdot_global[0] * SQR(cs_wind) * pow(r/r_inner, alpha_wind) / pmb->pcoord->GetCellVolume(k,j,i);
          }
        }
      }
    }
  }




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
//     2: inward  mass flux through inner boundary
//     3: outward mass flux through inner boundary

Real history(MeshBlock *pmb, int iout)
{
  Real stored_parameter = pmb->ruser_meshblock_data[0](iout);
  pmb->ruser_meshblock_data[0](iout) = 0.0;
  if (iout < 2) {
    return stored_parameter;
  }
  else {
    return stored_parameter/pmb->pmy_mesh->nbtotal; // these are summed over in the hist output so I have to normalize by the number of meshblocks
  }
}



//----------------------------------------------------------------------------------------
// gravitational acceleration
// maybe I will need to put in a smooth transition to 0 at the outer boundary
static Real grav_accel(Real r)
{

  Real x = r/(rvir/cnfw);

  Real g = grav_scale_inner * ((64.*pow(aaa,1.5)*rhom*pow(x,1.5) + 32.*rhom*pow(x,3.) + (96.*rho0)/(pow(1. + pow(rs_rt,4.),2.)*(1. + x)) - 
     (24.*rho0*(-1. + pow(rs_rt,4.)*(3. + x*(-4. + x*(3. - 2.*x + pow(rs_rt,4.)*(-1. + 2.*x))))))/
      (pow(1. + pow(rs_rt,4.),2.)*(1. + pow(rs_rt,4.)*pow(x,4.))) + 
     (12.*rs_rt*(-5.*sqrt(2.) + rs_rt*(18. - 14.*sqrt(2.)*rs_rt + 12.*sqrt(2.)*pow(rs_rt,3.) - 16.*pow(rs_rt,4.) + 2.*sqrt(2.)*pow(rs_rt,5.) + sqrt(2.)*pow(rs_rt,7.) - 
             2.*pow(rs_rt,8.)))*rho0*std::atan(1. - sqrt(2.)*rs_rt*x))/pow(1. + pow(rs_rt,4.),3.) + 
     (6.*rho0*(4.*(1. + pow(rs_rt,4.))*(-5. + 3.*pow(rs_rt,4.)) + 2.*pow(rs_rt,2.)*(-9. + 8.*pow(rs_rt,4.) + pow(rs_rt,8.))*PI - 
          2.*rs_rt*(-5.*sqrt(2.) + rs_rt*(-18. - 14.*sqrt(2.)*rs_rt + 12.*sqrt(2.)*pow(rs_rt,3.) + 16.*pow(rs_rt,4.) + 2.*sqrt(2.)*pow(rs_rt,5.) + 
                sqrt(2.)*pow(rs_rt,7.) + 2.*pow(rs_rt,8.)))*std::atan(1. + sqrt(2.)*rs_rt*x) + 16.*(1. - 7.*pow(rs_rt,4.))*std::log(1. + x) + 
          4.*(-1. + 7.*pow(rs_rt,4.))*std::log(1. + pow(rs_rt,4.)*pow(x,4.)) - 
          sqrt(2.)*rs_rt*(-5. + 14.*pow(rs_rt,2.) + 12.*pow(rs_rt,4.) - 2.*pow(rs_rt,6.) + pow(rs_rt,8.))*
           (std::log(1. + rs_rt*x*(-sqrt(2.) + rs_rt*x)) - std::log(1. + rs_rt*x*(sqrt(2.) + rs_rt*x)))))/pow(1. + pow(rs_rt,4.),3.))/(96.*pow(x,2.)));

  g += GMgal / (r*(r+Rgal));
  return g ; 
}




//----------------------------------------------------------------------------------------
// gravitational potential
// maybe I will need to put in a smooth transition to 0 at the outer boundary
static Real grav_pot(Real r)
{
  Real f = 1.0/(std::log(1.0+cnfw)/cnfw - 1.0/(1.0+cnfw));
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
//! \fn void ConstantOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
//                          FaceField &b, Real time, Real dt,
//                          int is, int ie, int js, int je, int ks, int ke, int ngh)
//  \brief Wind boundary conditions with no inflow, inner x1 boundary

void ConstantOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh)
{
  Real vc_ta  = sqrt( grav_accel(r_outer) * r_outer );
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=1; i<=(NGHOST); ++i) {
        prim(IDN,k,j,ie+i) = rho_ta;
        prim(IVX,k,j,ie+i) = 0.0;
        prim(IVY,k,j,ie+i) = 0.0;
        prim(IVZ,k,j,ie+i) = 0.0;
        prim(IPR,k,j,ie+i) = rho_ta*f_cs*SQR(vc_ta);
#if MAGNETIC_FIELDS_ENABLED
        b.x1f(k,j,ie+i) = 0.0;
        b.x2f(k,j,ie+i) = 0.0;
        b.x3f(k,j,ie+i) = sqrt(8*PI*rho_wind*SQR(cs_wind)/beta); // beta = P_Th/P_Mag ==> P_Mag = P_Th / beta ==> B = sqrt(8 pi P_th / beta )
#endif
      }
    }
  }
  return;
}






//----------------------------------------------------------------------------------------
//! \fn void ExtrapOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
//                          FaceField &b, Real time, Real dt,
//                          int is, int ie, int js, int je, int ks, int ke, int ngh)
//
//  \brief extrapolate into ghost zones outer x1 boundary using second order derivative

void ExtrapOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh)
{
  // extrapolate hydro variables into ghost zones
  for (int n=0; n<(NHYDRO); ++n) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=1; i<=(NGHOST); ++i) {
          Real fp = prim(n,k,j,ie) - prim(n,k,j,ie-1);
          Real fpp = prim(n,k,j,ie) - 2*prim(n,k,j,ie-1) + prim(n,k,j,ie-2);
          prim(n,k,j,ie+i) = prim(n,k,j,ie) + i*fp + 0.5*SQR(i)*fpp;  
          if (n==IDN){
            prim(n,k,j,ie+i) = std::max(prim(n,k,j,ie+i),dfloor);
          }
          if (n==IPR){
            prim(n,k,j,ie+i) = std::max(prim(n,k,j,ie+i),pfloor);
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
//! \fn void ExtrapInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
//                          FaceField &b, Real time, Real dt,
//                          int is, int ie, int js, int je, int ks, int ke, int ngh)
//
//  \brief extrapolate into ghost zones Inner x1 boundary using second order derivative

void ExtrapInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh)
{
  // extrapolate hydro variables into ghost zones
  for (int n=0; n<(NHYDRO); ++n) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=1; i<=(NGHOST); ++i) {
          Real fp = prim(n,k,j,is) - prim(n,k,j,is+1);
          Real fpp = prim(n,k,j,is) - 2*prim(n,k,j,is+1) + prim(n,k,j,is+2);
          prim(n,k,j,is-i) = prim(n,k,j,is) + i*fp + 0.5*SQR(i)*fpp;  
          if (n==IDN){
            // prim(n,k,j,is-i) = std::max(prim(n,k,j,is-i),dfloor);
            if (prim(n,k,j,is-i) < dfloor){
              prim(n,k,j,is-i) = prim(n,k,j,is);
            }
          }
          if (n==IPR){
            // prim(n,k,j,is-i) = std::max(prim(n,k,j,is-i),pfloor);
            if (prim(n,k,j,is-i) < pfloor){
              prim(n,k,j,is-i) = prim(n,k,j,is);
            }
          }
          if ((n==IVX)||(n==IVY)||(n==IVZ)){
            // Real Sign = (prim(n,k,j,is-i) > 0) ? 1 : ((prim(n,k,j,is-i) < 0) ? -1 : 0);
            // prim(n,k,j,is-i) = std::min(std::fabs(prim(n,k,j,is-i)),vceil) * Sign;
            if (std::fabs(prim(n,k,j,is-i)) > vceil){
              prim(n,k,j,is-i) = prim(n,k,j,is-i);
            }
          }
          if (n==IVX){
            prim(n,k,j,is-i) = std::min(0., prim(n,k,j,is-i));
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
        b.x1f(k,j,(is-i)) = b.x1f(k,j,is);
      }
    }}

    for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je+1; ++j) {
#pragma simd
      for (int i=1; i<=(NGHOST); ++i) {
        b.x2f(k,j,(is-i)) = b.x2f(k,j,is);
      }
    }}

    for (int k=ks; k<=ke+1; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma simd
      for (int i=1; i<=(NGHOST); ++i) {
        b.x3f(k,j,(is-i)) = b.x3f(k,j,is);
      }
    }}
  }
  return;
}




//----------------------------------------------------------------------------------------
//! \fn void AdaptiveWindX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
//                          FaceField &b, Real time, Real dt,
//                          int is, int ie, int js, int je, int ks, int ke, int ngh)
//  \brief Wind boundary conditions with no inflow, inner x1 boundary

void AdaptiveWindX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh)
{
  Real rho_out, area;
  area = SQR(pmb->pmy_mesh->mesh_size.x1min)
          * (pmb->pmy_mesh->mesh_size.x3max - pmb->pmy_mesh->mesh_size.x3min)
          * 2.0*(1 - std::cos(opening_angle)); // area = r^2 dphi 2*(cos0 - cos theta)
  rho_out = -1.0*Mdot_in/area/v_wind * (eta/(1.0+eta));

  // if(Globals::my_rank==0) {
  // std::cout << " mdot = " << Mdot_in << " rho_out = " << rho_out << " area = " << area << "\n";
  // }

  // Maybe I can just calculate the Mdot here... or in user after work

  if (rho_out > 0.0) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        Real theta = pco->x2v(j);
        // std::cout << " theta = " << theta << " PI-theta = " << PI-theta << "\n";
        if ((theta <= opening_angle) || ((PI-theta) <= opening_angle)){
          for (int i=1; i<=(NGHOST); ++i) {
            std::cout << " mdot = " << Mdot_in << "\n";
            prim(IDN,k,j,is-i) = rho_out;
            prim(IVX,k,j,is-i) = v_wind;
            prim(IVY,k,j,is-i) = 0.0;
            prim(IVZ,k,j,is-i) = 0.0;
            prim(IPR,k,j,is-i) = rho_out*SQR(cs_wind);
#if MAGNETIC_FIELDS_ENABLED
            b.x1f(k,j,is-i) = 0.0;
            b.x2f(k,j,is-i) = 0.0;
            b.x3f(k,j,is-i) = 0.0;// sqrt(8*PI*rho_out*SQR(cs_wind)/beta); // need to be VERY careful about how to add magnetic fields beta = P_Th/P_Mag ==> P_Mag = P_Th / beta ==> B = sqrt(8 pi P_th / beta ) but in code units I think there is a 4 pi 
#endif
          }
        } else {
          for (int n=0; n<(NHYDRO); ++n) {
            for (int i=1; i<=(NGHOST); ++i) {
              Real fp = prim(n,k,j,is) - prim(n,k,j,is+1);
              Real fpp = prim(n,k,j,is) - 2*prim(n,k,j,is+1) + prim(n,k,j,is+2);
              prim(n,k,j,is-i) = prim(n,k,j,is) + i*fp + 0.5*SQR(i)*fpp;
            }  
          }
        }
      }
    }
  } else {
    // extrapolate hydro variables into ghost zones
    for (int n=0; n<(NHYDRO); ++n) {
      for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je; ++j) {
          for (int i=1; i<=(NGHOST); ++i) {
            Real fp = prim(n,k,j,is) - prim(n,k,j,is+1);
            Real fpp = prim(n,k,j,is) - 2*prim(n,k,j,is+1) + prim(n,k,j,is+2);
            prim(n,k,j,is-i) = prim(n,k,j,is) + i*fp + 0.5*SQR(i)*fpp;  
            if (n==IDN){
              // prim(n,k,j,is-i) = std::max(prim(n,k,j,is-i),dfloor);
              if (prim(n,k,j,is-i) < dfloor){
                prim(n,k,j,is-i) = prim(n,k,j,is);
              }
            }
            if (n==IPR){
              // prim(n,k,j,is-i) = std::max(prim(n,k,j,is-i),pfloor);
              if (prim(n,k,j,is-i) < pfloor){
                prim(n,k,j,is-i) = prim(n,k,j,is);
              }
            }
            if ((n==IVX)||(n==IVY)||(n==IVZ)){
              // Real Sign = (prim(n,k,j,is-i) > 0) ? 1 : ((prim(n,k,j,is-i) < 0) ? -1 : 0);
              // prim(n,k,j,is-i) = std::min(std::fabs(prim(n,k,j,is-i)),vceil) * Sign;
              if (std::fabs(prim(n,k,j,is-i)) > vceil){
                prim(n,k,j,is-i) = prim(n,k,j,is-i);
              }
            }
            if (n==IVX){
              prim(n,k,j,is-i) = std::min(0., prim(n,k,j,is-i));
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
        for (int i=1; i<=(NGHOST); ++i) {
          b.x1f(k,j,(is-i)) = b.x1f(k,j,is);
        }
      }}

      for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je+1; ++j) {
        for (int i=1; i<=(NGHOST); ++i) {
          b.x2f(k,j,(is-i)) = b.x2f(k,j,is);
        }
      }}

      for (int k=ks; k<=ke+1; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=1; i<=(NGHOST); ++i) {
          b.x3f(k,j,(is-i)) = b.x3f(k,j,is);
        }
      }}
    }
  return;
}






// ----------------------------------------------------------------------------------------
// SmagorinskyViscosity 
// nu = (C * dx)^2 * |S|
// S_ij = 0.5*(dvelj_dxi + dveli_dxj)
// |S| = sqrt(2 S_ij S_ij)
// 

void SmagorinskyViscosity(HydroDiffusion *phdif, MeshBlock *pmb, const AthenaArray<Real> &prim,
     const AthenaArray<Real> &bcc, int is, int ie, int js, int je, int ks, int ke) 
{
  Real dvel1_dx1, dvel2_dx1, dvel3_dx1, dvel1_dx2, dvel2_dx2, dvel3_dx2, dvel1_dx3, dvel2_dx3, dvel3_dx3;
  Real S_norm;
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      phdif->nu(ISO,k,j,is) = 0.0;
      for (int i=is+1; i<=ie; ++i) {

        dvel1_dx1 = (prim(IVX,k,j,i) - prim(IVX,k,j,i-1))/pmb->pcoord->dx1v(i-1);
        dvel2_dx1 = (prim(IVY,k,j,i) - prim(IVY,k,j,i-1))/pmb->pcoord->dx1v(i-1);
        dvel3_dx1 = (prim(IVZ,k,j,i) - prim(IVZ,k,j,i-1))/pmb->pcoord->dx1v(i-1);

        dvel1_dx2 = (prim(IVX,k,j,i) - prim(IVX,k,j-1,i))/pmb->pcoord->dx2v(j-1)/pmb->pcoord->h2v(i);
        dvel2_dx2 = (prim(IVY,k,j,i) - prim(IVY,k,j-1,i))/pmb->pcoord->dx2v(j-1)/pmb->pcoord->h2v(i);
        dvel3_dx2 = (prim(IVZ,k,j,i) - prim(IVZ,k,j-1,i))/pmb->pcoord->dx2v(j-1)/pmb->pcoord->h2v(i);

        dvel1_dx3 = (prim(IVX,k,j,i) - prim(IVX,k-1,j,i))/pmb->pcoord->dx3v(k-1)/pmb->pcoord->h31v(i)/pmb->pcoord->h32v(j);
        dvel2_dx3 = (prim(IVY,k,j,i) - prim(IVY,k-1,j,i))/pmb->pcoord->dx3v(k-1)/pmb->pcoord->h31v(i)/pmb->pcoord->h32v(j);
        dvel3_dx3 = (prim(IVZ,k,j,i) - prim(IVZ,k-1,j,i))/pmb->pcoord->dx3v(k-1)/pmb->pcoord->h31v(i)/pmb->pcoord->h32v(j);

        S_norm = sqrt(2.0*( SQR(dvel1_dx1) + SQR(dvel2_dx2) + SQR(dvel3_dx3) + 
          0.5*(SQR(dvel2_dx1)+SQR(dvel3_dx1)+SQR(dvel1_dx2)+SQR(dvel3_dx2)+SQR(dvel1_dx3)+SQR(dvel2_dx3)) +
          dvel2_dx1*dvel1_dx2 + dvel3_dx1*dvel1_dx3 + dvel2_dx3*dvel3_dx2));

        phdif->nu(ISO,k,j,i) = phdif->nu_iso*SQR(pmb->pcoord->dx1v(i)) * S_norm;
      }
    }
  }
  return;
}

// ----------------------------------------------------------------------------------------
// SmagorinskyConduction 
// kappa = (C * dx)^2 * |S| / Prandtl
// S_ij = 0.5*(dvelj_dxi + dveli_dxj)
// |S| = sqrt(2 S_ij S_ij)
// 

void SmagorinskyConduction(HydroDiffusion *phdif, MeshBlock *pmb, const AthenaArray<Real> &prim,
     const AthenaArray<Real> &bcc, int is, int ie, int js, int je, int ks, int ke) 
{
  Real dvel1_dx1, dvel2_dx1, dvel3_dx1, dvel1_dx2, dvel2_dx2, dvel3_dx2, dvel1_dx3, dvel2_dx3, dvel3_dx3;
  Real S_norm;
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      phdif->kappa(ISO,k,j,is) = 0.0;
      for (int i=is+1; i<=ie; ++i) {

        dvel1_dx1 = (prim(IVX,k,j,i) - prim(IVX,k,j,i-1))/pmb->pcoord->dx1v(i-1);
        dvel2_dx1 = (prim(IVY,k,j,i) - prim(IVY,k,j,i-1))/pmb->pcoord->dx1v(i-1);
        dvel3_dx1 = (prim(IVZ,k,j,i) - prim(IVZ,k,j,i-1))/pmb->pcoord->dx1v(i-1);

        dvel1_dx2 = (prim(IVX,k,j,i) - prim(IVX,k,j-1,i))/pmb->pcoord->dx2v(j-1)/pmb->pcoord->h2v(i);
        dvel2_dx2 = (prim(IVY,k,j,i) - prim(IVY,k,j-1,i))/pmb->pcoord->dx2v(j-1)/pmb->pcoord->h2v(i);
        dvel3_dx2 = (prim(IVZ,k,j,i) - prim(IVZ,k,j-1,i))/pmb->pcoord->dx2v(j-1)/pmb->pcoord->h2v(i);

        dvel1_dx3 = (prim(IVX,k,j,i) - prim(IVX,k-1,j,i))/pmb->pcoord->dx3v(k-1)/pmb->pcoord->h31v(i)/pmb->pcoord->h32v(j);
        dvel2_dx3 = (prim(IVY,k,j,i) - prim(IVY,k-1,j,i))/pmb->pcoord->dx3v(k-1)/pmb->pcoord->h31v(i)/pmb->pcoord->h32v(j);
        dvel3_dx3 = (prim(IVZ,k,j,i) - prim(IVZ,k-1,j,i))/pmb->pcoord->dx3v(k-1)/pmb->pcoord->h31v(i)/pmb->pcoord->h32v(j);

        S_norm = sqrt(2.0*( SQR(dvel1_dx1) + SQR(dvel2_dx2) + SQR(dvel3_dx3) + 
          0.5*(SQR(dvel2_dx1)+SQR(dvel3_dx1)+SQR(dvel1_dx2)+SQR(dvel3_dx2)+SQR(dvel1_dx3)+SQR(dvel2_dx3)) +
          dvel2_dx1*dvel1_dx2 + dvel3_dx1*dvel1_dx3 + dvel2_dx3*dvel3_dx2));

        phdif->kappa(ISO,k,j,i) = phdif->kappa_iso*SQR(pmb->pcoord->dx1v(i)) * S_norm;
      }
    }
  }
  return;
}




// ----------------------------------------------------------------------------------------
// DeviatoricSmagorinskyViscosity 
// nu = (C * dx)^2 * |Sd|
// Sd_ij = 0.5*(dvelj_dxi + dveli_dxj) - 1/3 delta_ij dvelk_dxk
// |Sd| = sqrt(2 Sd_ij Sd_ij)
// 

void DeviatoricSmagorinskyViscosity(HydroDiffusion *phdif, MeshBlock *pmb, const AthenaArray<Real> &prim,
     const AthenaArray<Real> &bcc, int is, int ie, int js, int je, int ks, int ke) 
{
  Real dvel1_dx1, dvel2_dx1, dvel3_dx1, dvel1_dx2, dvel2_dx2, dvel3_dx2, dvel1_dx3, dvel2_dx3, dvel3_dx3;
  Real S_norm;
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      phdif->nu(ISO,k,j,is) = 0.0;
      for (int i=is+1; i<=ie; ++i) {

        dvel1_dx1 = (prim(IVX,k,j,i) - prim(IVX,k,j,i-1))/pmb->pcoord->dx1v(i-1);
        dvel2_dx1 = (prim(IVY,k,j,i) - prim(IVY,k,j,i-1))/pmb->pcoord->dx1v(i-1);
        dvel3_dx1 = (prim(IVZ,k,j,i) - prim(IVZ,k,j,i-1))/pmb->pcoord->dx1v(i-1);

        dvel1_dx2 = (prim(IVX,k,j,i) - prim(IVX,k,j-1,i))/pmb->pcoord->dx2v(j-1)/pmb->pcoord->h2v(i);
        dvel2_dx2 = (prim(IVY,k,j,i) - prim(IVY,k,j-1,i))/pmb->pcoord->dx2v(j-1)/pmb->pcoord->h2v(i);
        dvel3_dx2 = (prim(IVZ,k,j,i) - prim(IVZ,k,j-1,i))/pmb->pcoord->dx2v(j-1)/pmb->pcoord->h2v(i);

        dvel1_dx3 = (prim(IVX,k,j,i) - prim(IVX,k-1,j,i))/pmb->pcoord->dx3v(k-1)/pmb->pcoord->h31v(i)/pmb->pcoord->h32v(j);
        dvel2_dx3 = (prim(IVY,k,j,i) - prim(IVY,k-1,j,i))/pmb->pcoord->dx3v(k-1)/pmb->pcoord->h31v(i)/pmb->pcoord->h32v(j);
        dvel3_dx3 = (prim(IVZ,k,j,i) - prim(IVZ,k-1,j,i))/pmb->pcoord->dx3v(k-1)/pmb->pcoord->h31v(i)/pmb->pcoord->h32v(j);

        S_norm = sqrt(pow(dvel2_dx1 + dvel1_dx2,2) + pow(dvel3_dx1 + dvel1_dx3,2) + pow(dvel3_dx2 + dvel2_dx3,2) + 
                      (4/3.*(pow(dvel1_dx1,2) + pow(dvel2_dx2,2) + pow(dvel3_dx3,2)
                             -dvel2_dx2*dvel3_dx3 - dvel1_dx1*dvel2_dx2 - dvel1_dx1*dvel3_dx3)));

        phdif->nu(ISO,k,j,i) = phdif->nu_iso*SQR(pmb->pcoord->dx1v(i)) * S_norm;
      }
    }
  }
  return;
}

// ----------------------------------------------------------------------------------------
// DeviatoricSmagorinskyConduction 
// nu = (C * dx)^2 * |Sd| / Pr
// Sd_ij = 0.5*(dvelj_dxi + dveli_dxj) - 1/3 delta_ij dvelk_dxk
// |Sd| = sqrt(2 Sd_ij Sd_ij)
// 

void DeviatoricSmagorinskyConduction(HydroDiffusion *phdif, MeshBlock *pmb, const AthenaArray<Real> &prim,
     const AthenaArray<Real> &bcc, int is, int ie, int js, int je, int ks, int ke) 
{
  Real dvel1_dx1, dvel2_dx1, dvel3_dx1, dvel1_dx2, dvel2_dx2, dvel3_dx2, dvel1_dx3, dvel2_dx3, dvel3_dx3;
  Real S_norm;
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      phdif->kappa(ISO,k,j,is) = 0.0;
      for (int i=is+1; i<=ie; ++i) {

        dvel1_dx1 = (prim(IVX,k,j,i) - prim(IVX,k,j,i-1))/pmb->pcoord->dx1v(i-1);
        dvel2_dx1 = (prim(IVY,k,j,i) - prim(IVY,k,j,i-1))/pmb->pcoord->dx1v(i-1);
        dvel3_dx1 = (prim(IVZ,k,j,i) - prim(IVZ,k,j,i-1))/pmb->pcoord->dx1v(i-1);

        dvel1_dx2 = (prim(IVX,k,j,i) - prim(IVX,k,j-1,i))/pmb->pcoord->dx2v(j-1)/pmb->pcoord->h2v(i);
        dvel2_dx2 = (prim(IVY,k,j,i) - prim(IVY,k,j-1,i))/pmb->pcoord->dx2v(j-1)/pmb->pcoord->h2v(i);
        dvel3_dx2 = (prim(IVZ,k,j,i) - prim(IVZ,k,j-1,i))/pmb->pcoord->dx2v(j-1)/pmb->pcoord->h2v(i);

        dvel1_dx3 = (prim(IVX,k,j,i) - prim(IVX,k-1,j,i))/pmb->pcoord->dx3v(k-1)/pmb->pcoord->h31v(i)/pmb->pcoord->h32v(j);
        dvel2_dx3 = (prim(IVY,k,j,i) - prim(IVY,k-1,j,i))/pmb->pcoord->dx3v(k-1)/pmb->pcoord->h31v(i)/pmb->pcoord->h32v(j);
        dvel3_dx3 = (prim(IVZ,k,j,i) - prim(IVZ,k-1,j,i))/pmb->pcoord->dx3v(k-1)/pmb->pcoord->h31v(i)/pmb->pcoord->h32v(j);

        S_norm = sqrt(pow(dvel2_dx1 + dvel1_dx2,2) + pow(dvel3_dx1 + dvel1_dx3,2) + pow(dvel3_dx2 + dvel2_dx3,2) + 
                      (4/3.*(pow(dvel1_dx1,2) + pow(dvel2_dx2,2) + pow(dvel3_dx3,2)
                             -dvel2_dx2*dvel3_dx3 - dvel1_dx1*dvel2_dx2 - dvel1_dx1*dvel3_dx3)));

        phdif->kappa(ISO,k,j,i) = phdif->kappa_iso*SQR(pmb->pcoord->dx1v(i)) * S_norm;
      }
    }
  }
  return;
}



