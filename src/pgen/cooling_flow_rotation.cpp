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
    const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc, AthenaArray<Real> &cons, int stage);
Real CoolingLosses(MeshBlock *pmb, int iout);
Real fluxes(MeshBlock *pmb, int iout);
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
static Real t_cool_start;
static Real beta;
static Real grav_scale_inner, aaa, rs_rt, rhom, rho0;

static Real cooling_timestep(MeshBlock *pmb);
static int turb_grid_size;
static Real kappa;
static bool conduction_on;
static Real dt_cutoff, cfl_cool;
static Real pfloor, dfloor, vceil;

static Real Mhalo, cnfw, GMhalo, rvir, r200m, Mgal, GMgal, Rgal;
static Real rho_wind, v_wind, cs_wind, eta;
static Real rho_igm, v_igm, cs_igm, Mdot_igm;
static Real cs, rho_ta, f_cs,f2;

static Real r_inner, r_outer;

static bool rotation;
static Real lambda, r_circ;


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

  // rotation

  rotation = pin->GetBoolean("problem", "rotation");
  lambda = pin->GetOrAddReal("problem", "lambda", 0.0);
  r_circ = lambda * rvir;
  f_cs         = pin->GetOrAddReal("problem","f_cs",1.0);
  f2           = pin->GetOrAddReal("problem","f2",1.0);



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

  // Read cooling-table-related parameters from input file
  std::string cooling_file = pin->GetString("problem", "cooling_file");
  Real z_z_solar = pin->GetReal("problem", "relative_metallicity");

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

  // Read sizes of tables
  int number_of_pressure_bins; 
  dataset = H5Dopen(file, "/number_of_pressure_bins", H5P_DEFAULT);
  H5Dread(dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, property_list_transfer,
      &number_of_pressure_bins);
  H5Dclose(dataset);

  int number_of_rho_bins; 
  dataset = H5Dopen(file, "/number_of_rho_bins", H5P_DEFAULT);
  H5Dread(dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, property_list_transfer,
      &number_of_rho_bins);
  H5Dclose(dataset);

  // Read sample data from file
  AthenaArray<double> pressure_samples, rho_samples;
  pressure_samples.NewAthenaArray(number_of_pressure_bins);
  rho_samples.NewAthenaArray(number_of_rho_bins);
  dataset = H5Dopen(file, "pressure", H5P_DEFAULT);
  H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, property_list_transfer,
      pressure_samples.data());
  H5Dclose(dataset);
  dataset = H5Dopen(file, "rho", H5P_DEFAULT);
  H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, property_list_transfer,
      rho_samples.data());
  H5Dclose(dataset);

  // Read temperature data from file
  AthenaArray<double> temperature_table;
  temperature_table.NewAthenaArray(number_of_pressure_bins, number_of_rho_bins);
  dataset = H5Dopen(file, "temperature", H5P_DEFAULT);
  H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, property_list_transfer,
      temperature_table.data());
  H5Dclose(dataset);

  // Read mu data from file
  AthenaArray<double> mu_table;
  mu_table.NewAthenaArray(number_of_pressure_bins, number_of_rho_bins);
  dataset = H5Dopen(file, "mu", H5P_DEFAULT);
  H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, property_list_transfer,
      mu_table.data());
  H5Dclose(dataset);

  // Read Metal_Cooling data from file
  AthenaArray<double> Metal_Cooling_table;
  Metal_Cooling_table.NewAthenaArray(number_of_pressure_bins, number_of_rho_bins);
  dataset = H5Dopen(file, "Metal_Cooling", H5P_DEFAULT);
  H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, property_list_transfer,
      Metal_Cooling_table.data());
  H5Dclose(dataset);

  // Read Primordial_Cooling data from file
  AthenaArray<double> Primordial_Cooling_table;
  Primordial_Cooling_table.NewAthenaArray(number_of_pressure_bins, number_of_rho_bins);
  dataset = H5Dopen(file, "Primordial_Cooling", H5P_DEFAULT);
  H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, property_list_transfer,
      Primordial_Cooling_table.data());
  H5Dclose(dataset);

  // Read Metal_Heating data from file
  AthenaArray<double> Metal_Heating_table;
  Metal_Heating_table.NewAthenaArray(number_of_pressure_bins, number_of_rho_bins);
  dataset = H5Dopen(file, "Metal_Heating", H5P_DEFAULT);
  H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, property_list_transfer,
      Metal_Heating_table.data());
  H5Dclose(dataset);

  // Read Primordial_Heating data from file
  AthenaArray<double> Primordial_Heating_table;
  Primordial_Heating_table.NewAthenaArray(number_of_pressure_bins, number_of_rho_bins);
  dataset = H5Dopen(file, "Primordial_Heating", H5P_DEFAULT);
  H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, property_list_transfer,
      Primordial_Heating_table.data());
  H5Dclose(dataset);

  // Close cooling data file
  H5Pclose(property_list_transfer);
  H5Fclose(file);




  // Allocate fixed cooling table
  AllocateRealUserMeshDataField(5);
  ruser_mesh_data[0].NewAthenaArray(number_of_pressure_bins);                      // pressure 
  ruser_mesh_data[1].NewAthenaArray(number_of_rho_bins);                           // density
  ruser_mesh_data[2].NewAthenaArray(number_of_pressure_bins, number_of_rho_bins);  // temperature
  ruser_mesh_data[3].NewAthenaArray(number_of_pressure_bins, number_of_rho_bins);  // total cooling
  ruser_mesh_data[4].NewAthenaArray(2); // delta_e_tot, vol_tot

  Real vol_tot = (mesh_size.x3max - mesh_size.x3min)
                 * (std::cos(mesh_size.x2max) - std::cos(mesh_size.x2min))
                 * (pow(mesh_size.x1max,3) - pow(mesh_size.x1min,3))/3.;
  ruser_mesh_data[4](1) = vol_tot;

  // Tabulate sample points
  for (int i = 0; i < number_of_pressure_bins; ++i) {
    ruser_mesh_data[0](i) = pressure_samples(i) / pgas_scale;
  }
  for (int i = 0; i < number_of_rho_bins; ++i) {
    ruser_mesh_data[1](i) = rho_samples(i) / rho_scale;
  }
  for (int i = 0; i < number_of_pressure_bins; ++i) {
    for (int j = 0; j < number_of_rho_bins; ++j) {
      ruser_mesh_data[2](i,j) = temperature_table(i,j);
      ruser_mesh_data[3](i,j)  = SQR(rho_samples(j)/(muH*mp)) * Primordial_Cooling_table(i,j) / (pgas_scale / time_scale);
      ruser_mesh_data[3](i,j) -= SQR(rho_samples(j)/(muH*mp)) * Primordial_Heating_table(i,j) / (pgas_scale / time_scale);
      ruser_mesh_data[3](i,j) += SQR(rho_samples(j)/(muH*mp)) * Metal_Cooling_table(i,j) * z_z_solar / (pgas_scale / time_scale);
      ruser_mesh_data[3](i,j) -= SQR(rho_samples(j)/(muH*mp)) * Metal_Heating_table(i,j) * z_z_solar / (pgas_scale / time_scale);
    }
  }


  // Delete intermediate tables
  pressure_samples.DeleteAthenaArray();
  rho_samples.DeleteAthenaArray();
  temperature_table.DeleteAthenaArray();
  mu_table.DeleteAthenaArray();
  Metal_Cooling_table.DeleteAthenaArray();
  Primordial_Cooling_table.DeleteAthenaArray();
  Metal_Heating_table.DeleteAthenaArray();
  Primordial_Heating_table.DeleteAthenaArray();

  // Enroll user-defined functions
  EnrollUserExplicitSourceFunction(SourceFunction);
  AllocateUserHistoryOutput(30);
  EnrollUserHistoryOutput(0 , CoolingLosses, "e_cool");
  EnrollUserHistoryOutput(1 , CoolingLosses, "e_ceil");
  EnrollUserHistoryOutput(2 , fluxes, "Migal");
  EnrollUserHistoryOutput(3 , fluxes, "Mi01");
  EnrollUserHistoryOutput(4 , fluxes, "Mi025");
  EnrollUserHistoryOutput(5 , fluxes, "Mi05");
  EnrollUserHistoryOutput(6 , fluxes, "Mi10");
  EnrollUserHistoryOutput(7 , fluxes, "Mi15");
  EnrollUserHistoryOutput(8 , fluxes, "Mita");
  EnrollUserHistoryOutput(9 , fluxes, "Mogal");
  EnrollUserHistoryOutput(10, fluxes, "Mo01");
  EnrollUserHistoryOutput(11, fluxes, "Mo025");
  EnrollUserHistoryOutput(12, fluxes, "Mo05");
  EnrollUserHistoryOutput(13, fluxes, "Mo10");
  EnrollUserHistoryOutput(14, fluxes, "Mo15");
  EnrollUserHistoryOutput(15, fluxes, "Mota");
  EnrollUserHistoryOutput(16, fluxes, "Eigal");
  EnrollUserHistoryOutput(17, fluxes, "Ei01");
  EnrollUserHistoryOutput(18, fluxes, "Ei025");
  EnrollUserHistoryOutput(19, fluxes, "Ei05");
  EnrollUserHistoryOutput(20, fluxes, "Ei10");
  EnrollUserHistoryOutput(21, fluxes, "Ei15");
  EnrollUserHistoryOutput(22, fluxes, "Eita");
  EnrollUserHistoryOutput(23, fluxes, "Eogal");
  EnrollUserHistoryOutput(24, fluxes, "Eo01");
  EnrollUserHistoryOutput(25, fluxes, "Eo025");
  EnrollUserHistoryOutput(26, fluxes, "Eo05");
  EnrollUserHistoryOutput(27, fluxes, "Eo10");
  EnrollUserHistoryOutput(28, fluxes, "Eo15");
  EnrollUserHistoryOutput(29, fluxes, "Eota");
  
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
  AllocateRealUserMeshBlockDataField(2);
  ruser_meshblock_data[0].NewAthenaArray(2);
  ruser_meshblock_data[0](0) = 0.0;
  ruser_meshblock_data[0](1) = 0.0;
  
  // Allocate storage for keeping track of fluxes
  ruser_meshblock_data[1].NewAthenaArray(28);
  for (int i = 0; i < 28; ++i) {
    ruser_meshblock_data[1](i) = 0.0; 
  }

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


  Real phi_ta = fabs(grav_pot(rvir));
  Real vc_ta  = sqrt( grav_accel(rvir) * rvir );
  // Initialize primitive values
  for (int k = kl; k <= ku; ++k) {
    for (int j = jl; j <= ju; ++j) {
      for (int i = il; i <= iu; ++i) {
        Real r = pcoord->x1v(i);
        Real theta = pcoord->x2v(j);
        Real R_cyl = r*sin(theta);
        Real rho, press;
        Real vc = sqrt(grav_accel(r) * r );
        Real v_phi;

        // if(Globals::my_rank==0) {
          // std::cout << "r, theta, R_cyl, r_circ = " << r << " " << theta << " " << R_cyl << " " << r_circ << " \n";
        // }

        if (R_cyl <= r_circ){
          rho = rho_ta * pow(rvir / r_circ, gamma_adi*f2) * exp(-0.5*gamma_adi*f_cs);
          rho *= SQR(vc_ta/vc) * pow(r/rvir,-gamma_adi*(f_cs-f2)) * pow(sin(theta),gamma_adi*f2);
          v_phi = sqrt(f2/f_cs) * vc;
        } else {
          rho = rho_ta * SQR(vc_ta/vc) * pow(r/rvir,-gamma_adi*f_cs) * exp(-0.5*gamma_adi*f_cs*SQR(r_circ/R_cyl));
          v_phi = vc * r_circ / R_cyl;
        }
        // if ((k==16)&&(j==16)){
        //   std::cout << "r, theta, R_cyl, r_circ, rho, v_phi = " << r << " " << theta << " " << R_cyl << " " << r_circ <<" " << rho <<" " << v_phi << " \n";
        // }

        press   = SQR(vc) * rho / (gamma_adi * f_cs); 
        phydro->w(IDN,k,j,i) = rho;
        phydro->w(IPR,k,j,i) = press;
        phydro->w(IVX,k,j,i) = 0.0;
        phydro->w(IVY,k,j,i) = 0.0;
        phydro->w(IVZ,k,j,i) = -v_phi; // negative to make net ang mom positive
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
          if (temperature(k,j,i)>1.0e4){
            Real dt;
            Real edot = fabs(edot_cool(k,j,i));
            Real press = pmb->phydro->w(IPR,k,j,i);
            if (conduction_on){
              dt = cfl_cool * std::min( SQR(pmb->pcoord->dx1f(i))/kappa , 1.5*press/edot); // experiment with that cfl_cool
            } else {
              dt = cfl_cool * 1.5*press/edot;
            }
            // if (dt < dt_cutoff){
            //   Real r = pmb->pcoord->x1v(i);
            //   Real theta = pmb->pcoord->x1v(j);
            //   Real phi = pmb->pcoord->x1v(k);
            //   std::cout << " dt_cutoff > dt = " << dt << " press "<< press << " temperature "<< temperature(k,j,i)  << 
            //                " density " << pmb->phydro->w(IDN,k,j,i)  << " edot "<< edot <<
            //                " r " << r  << " theta "<< theta << " phi "<< phi <<"\n";
            // }
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
    const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc, AthenaArray<Real> &cons, int stage)
{
  // Prepare values to aggregate
  // cooling
  Real delta_e_mesh = 0.0;
  Real delta_e_tot;
  Real vol_tot  = pmb->pmy_mesh->ruser_mesh_data[4](1);
  // mass fluxes
  Real flux_mesh[28] = {0.0};
  Real flux_tot[28] = {0.0};

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
  pgas_table.InitWithShallowCopy(pmb->pmy_mesh->ruser_mesh_data[0]);
  rho_table.InitWithShallowCopy(pmb->pmy_mesh->ruser_mesh_data[1]);
  temperature_table.InitWithShallowCopy(pmb->pmy_mesh->ruser_mesh_data[2]);
  edot_cool_table.InitWithShallowCopy(pmb->pmy_mesh->ruser_mesh_data[3]);
  
  Real phi_ta = grav_pot(rvir);

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
            Real r = pblock->pcoord->x1v(i);
            // Extract density, pressure, vr
            Real rho  = prim_local(IDN,k,j,i);
            Real pgas = prim_local(IPR,k,j,i);
            Real vr   = prim_local(IVX,k,j,i);
            Real vth  = prim_local(IVY,k,j,i);
            Real vph  = prim_local(IVZ,k,j,i);

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
            
            // store the local volumetric cooling rate in out_var_0
            user_out_var_local(0,k,j,i) = edot_cool;

            
            // sum up the total amount of cooling
            Real vol_cell = pmb->pcoord->GetCellVolume(k,j,i);
            delta_e_mesh += delta_e * vol_cell;
            
            // calculate mass flux and place in appropriate holder
            Real mdot = pblock->pcoord->GetFace1Area(k,j,i)*rho*vr;
            Real edot = mdot * (0.5*(SQR(vr)+SQR(vth)+SQR(vph)) + gamma_adi/(gamma_adi-1.0) * pgas/rho + (phi_ta - grav_pot(r)));
            if (pblock->pcoord->x1f(i) == r_inner){
              if (vr<0.){
                flux_mesh[0]  += mdot;
                flux_mesh[14] += edot;
              } else {
                flux_mesh[7]  += mdot;
                flux_mesh[21] += edot;
              }
            } else if ( ( r/rvir < 0.10 ) and ( 0.10 - r/rvir < pblock->pcoord->dx1v(i))){
              if (vr<0.){
                flux_mesh[1]  += mdot;
                flux_mesh[15] += edot;
              } else {
                flux_mesh[8]  += mdot;
                flux_mesh[22] += edot;
              }
            } else if ( ( r/rvir < 0.25 ) and ( 0.25 - r/rvir < pblock->pcoord->dx1v(i))){
              if (vr<0.){
                flux_mesh[2]  += mdot;
                flux_mesh[16] += edot;
              } else {
                flux_mesh[9]  += mdot;
                flux_mesh[23] += edot;
              }
            } else if ( ( r/rvir < 0.50 ) and ( 0.50 - r/rvir < pblock->pcoord->dx1v(i))){
              if (vr<0.){
                flux_mesh[3]  += mdot;
                flux_mesh[17] += edot;
              } else {
                flux_mesh[10]  += mdot;
                flux_mesh[24] += edot;
              }
            } else if ( ( r/rvir < 1.00 ) and ( 1.00 - r/rvir < pblock->pcoord->dx1v(i))){
              if (vr<0.){
                flux_mesh[4]  += mdot;
                flux_mesh[18] += edot;
              } else {
                flux_mesh[11]  += mdot;
                flux_mesh[25] += edot;
              }
            } else if ( ( r/rvir < 1.50 ) and ( 1.50 - r/rvir < pblock->pcoord->dx1v(i))){
              if (vr<0.){
                flux_mesh[5]  += mdot;
                flux_mesh[19] += edot;
              } else {
                flux_mesh[12]  += mdot;
                flux_mesh[26] += edot;
              }
            } else if ( pblock->pcoord->x1f(i+1) == r_outer ){
              if (vr<0.){
                flux_mesh[6]  += mdot;
                flux_mesh[20] += edot;
              } else {
                flux_mesh[13]  += mdot;
                flux_mesh[27] += edot;
              }
            }
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
    MPI_Allreduce(&flux_mesh, &flux_tot, 28, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#else
    delta_e_tot = delta_e_mesh;
    flux_tot = flux_mesh;
#endif
  }

  // Store or extract redistributed heating
  if (pmb->lid == 0) {
    pmb->pmy_mesh->ruser_mesh_data[4](0) = delta_e_tot;
  } else {
    delta_e_tot = pmb->pmy_mesh->ruser_mesh_data[4](0);
  }
  
  // Store or extract flux
  if (pmb->lid == 0) {
    for (int i = 0; i < 28; ++i) {
      pmb->ruser_meshblock_data[1](i) = flux_tot[i];
    }
  } else {
    for (int i = 0; i < 28; ++i) {
      flux_tot[i] = pmb->ruser_meshblock_data[1](i);
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
        delta_e = std::max(delta_e, -u + 2.0*pfloor/(gamma_adi-1.0));
        if (t > t_cool_start){
          e += delta_e;
        }
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
// fluxes
// Inputs:
//   pmb: pointer to MeshBlock
//   iout: index of history output
// Outputs:
//   returned value: mass fluxes at inner edge, 0.1, 0.25, 0.5, 1.0, 1.5, and outer edge

Real fluxes(MeshBlock *pmb, int iout)
{
  if(Globals::my_rank==0){
    Real flux = pmb->ruser_meshblock_data[1](iout-2);
    pmb->ruser_meshblock_data[1](iout-2) = 0.0;
    return flux;
  } else {
    return 0;
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
        Real r = pco->x1v(ie+i);
        Real theta = pco->x2v(j);
        Real R_cyl = r*sin(theta);
        Real rho, press;
        Real vc = sqrt(grav_accel(r) * r );
        Real v_phi;
        if (R_cyl <= r_circ){
          rho = rho_ta * pow(rvir / r_circ, gamma_adi*f2) * exp(-0.5*gamma_adi*f_cs);
          rho *= SQR(vc_ta/vc) * pow(r/rvir,-gamma_adi*(f_cs-f2)) * pow(sin(theta),gamma_adi*f2);
          v_phi = sqrt(f2/f_cs) * vc;
        } else {
          rho = rho_ta * SQR(vc_ta/vc) * pow(r/rvir,-gamma_adi*f_cs) * exp(-0.5*gamma_adi*f_cs*SQR(r_circ/R_cyl));
          v_phi = vc * r_circ / R_cyl;
        }
        press   = SQR(vc) * rho / (gamma_adi * f_cs); 
        prim(IDN,k,j,ie+i) = rho;
        prim(IVX,k,j,ie+i) = 0.0;
        prim(IVY,k,j,ie+i) = 0.0;
        prim(IVZ,k,j,ie+i) = -v_phi;
        prim(IPR,k,j,ie+i) = press; 
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
          * (std::cos(pmb->pmy_mesh->mesh_size.x2min) - std::cos(pmb->pmy_mesh->mesh_size.x2max)); // area = r^2 dphi dcostheta
  rho_out = -1.0*pmb->ruser_meshblock_data[1](0)/area/v_wind * (eta/(1.0+eta));

  // if(Globals::my_rank==0) {
  //   std::cout << " mdot = " << pmb->ruser_meshblock_data[1](0) << " rho_out = " << rho_out << " area = " << area << "\n";
  // }

  if (rho_out > 0.0) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=1; i<=(NGHOST); ++i) {
          prim(IDN,k,j,is-i) = rho_out;
          prim(IVX,k,j,is-i) = v_wind;
          prim(IVY,k,j,is-i) = 0.0;
          prim(IVZ,k,j,is-i) = 0.0;
          prim(IPR,k,j,is-i) = rho_out*SQR(cs_wind);
#if MAGNETIC_FIELDS_ENABLED
          b.x1f(k,j,is-i) = 0.0;
          b.x2f(k,j,is-i) = 0.0;
          b.x3f(k,j,is-i) = sqrt(8*PI*rho_out*SQR(cs_wind)/beta); // beta = P_Th/P_Mag ==> P_Mag = P_Th / beta ==> B = sqrt(8 pi P_th / beta )
#endif
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


