// Problem generator for testing exact cooling method

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
#include "../fft/athena_fft.hpp"
#include "../utils/utils.hpp"

// External library headers
#include <hdf5.h>  // H5*, hid_t, hsize_t, H5*()

// Global variables --- hydro & units
static Real gamma_adi;
static Real rho_0, pgas_0; // in code units
static Real rho_scale, pgas_scale, vel_scale, length_scale, time_scale; 
static const Real mu=0.62;
static const Real muH=1.4;
static const Real mp = 1.67373522381e-24;
static const Real kb = 1.3806488e-16;
static Real vc2o2r2; 
static Real Gamma; 
static Real t_start_cooling, T_floor, T_max, rho_floor;
static Real v_max, dt_cutoff;
static Real dt_SN, t_last_SN, t_start_SN, r_inj_sq;
static int i_SN;
static Real E_SN, P_SN, ejecta_mass;
static Real beta;

Real x_SN[300]={0.};
Real y_SN[300]={0.};
Real z_SN[300]={0.};

static Real grav_accel( Real z );
void NoInflowInnerX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
void NoInflowOuterX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
Real CoolingLosses(MeshBlock *pmb, int iout);
static Real cooling_timestep(MeshBlock *pmb);
Real fluxes(MeshBlock *pmb, int iout);

void SourceFunction(MeshBlock *pmb, const Real t, const Real dt,
    const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc, 
    AthenaArray<Real> &cons);
int RefinementCondition(MeshBlock *pmb);

// Global variables --- cooling
// all of this can be swapped for pre-tabulated tables of Y, Y_inv, and T
// which can be calculated beforehand for any cooling curve. 
static int nfit_cool = 12;
static Real T_cooling_curve[12] = 
  {0.99999999e2,
   1.0e+02, 6.0e+03, 1.75e+04, 
   4.0e+04, 8.7e+04, 2.30e+05, 
   3.6e+05, 1.5e+06, 3.50e+06, 
   2.6e+07, 1.0e+12};

static Real lambda_cooling_curve[12] = 
  { 3.720076376848256e-71,
    1.00e-27,   2.00e-26,   1.50e-22,
    1.20e-22,   5.25e-22,   5.20e-22,
    2.25e-22,   1.25e-22,   3.50e-23,
    2.10e-23,   4.12e-21};

static Real exponent_cooling_curve[12] = 
  {1e10,
   0.73167566,  8.33549431, -0.26992783,  
   1.89942352, -0.00984338, -1.8698263 , 
  -0.41187018, -1.50238273, -0.25473349,  
   0.5000359, 0.5 };


static Real Lambda_T(const Real T);
static Real tcool(const Real T, const Real nH);



// ============================================================================
// piecewise power-law fit to the cooling curve with
// temperature in K and L in erg cm^3 / s 
static Real Lambda_T(const Real T)
{
  int k, n=nfit_cool-1;
  // first find the temperature bin 
  for(k=n; k>=0; k--){
    if (T >= T_cooling_curve[k])
      break;
  }
  if (T > T_cooling_curve[0]){
    return (lambda_cooling_curve[k] * 
              pow(T/T_cooling_curve[k], exponent_cooling_curve[k]));
  } else {
    return 1.0e-50;
  }
}
// ============================================================================



// ============================================================================
static Real tcool(const Real T, const Real nH)
{
  if (T < 2.0e4){
    return (kb * T) / ( (gamma_adi-1.0) * (mu/muH) * (nH*Lambda_T(T) - Gamma) );
  } else {
    return (kb * T) / ( (gamma_adi-1.0) * (mu/muH) * (nH*Lambda_T(T)) );
  }
}
// ============================================================================


//-----------------------------------------------------------------------------
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
  gamma_adi               = pin->GetReal("hydro", "gamma");
  rho_scale               = pin->GetReal("problem", "rho_scale");
  pgas_scale              = pin->GetReal("problem", "pgas_scale");
  length_scale            = pin->GetReal("problem", "length_scale");
  vel_scale               = std::sqrt(pgas_scale / rho_scale);
  time_scale              = length_scale/std::sqrt(pgas_scale / rho_scale);

  Real mass_scale         = rho_scale*pow(length_scale,3);
  Real energy_scale       = rho_scale*pow(length_scale,3)*SQR(vel_scale);

  rho_0                   = pin->GetReal("problem", "rho_0");
  pgas_0                  = pin->GetReal("problem", "pgas_0");
  vc2o2r2                 = pin->GetReal("problem", "vc2o2r2")*SQR(time_scale); 
  t_start_cooling         = pin->GetOrAddReal("problem", "t_start_cooling", 0.0); 
  dt_cutoff               = pin->GetOrAddReal("problem", "dt_cutoff", 1.0e-7); 
  rho_floor               = pin->GetOrAddReal("problem", "rho_floor", 1e-6); 
  T_floor                 = pin->GetOrAddReal("problem", "T_floor", 100.0); 
  T_max                   = pin->GetOrAddReal("problem", "T_max", 1.0e10); 
  v_max                   = pin->GetOrAddReal("problem", "v_max", 1.0e4);  // faster than speed of light...
  Gamma            = pin->GetReal("problem", "Gamma"); //heating rate in ergs / sec


  // SNE
  Real M_cluster          = pin->GetReal("problem", "M_cluster")/mass_scale; 
  Real R_cluster          = pin->GetReal("problem", "R_cluster")/length_scale;
  Real t_SNe              = pin->GetReal("problem", "t_SNe")/time_scale; 
  Real m_star             = pin->GetReal("problem", "m_star")/mass_scale; 
  t_start_SN              = pin->GetReal("problem", "t_start_SN"); 
  r_inj_sq                = SQR(pin->GetReal("problem", "r_inj"));
  E_SN                    = 1.0e51 / energy_scale;
  P_SN                    = 3.479830e42 / mass_scale / vel_scale;
  ejecta_mass             = pin->GetReal("problem", "ejecta_mass") / mass_scale;
  Real x_SN_offset        = pin->GetOrAddReal("problem", "x_SN_offset", 0.0);
  Real y_SN_offset        = pin->GetOrAddReal("problem", "y_SN_offset", 0.0);
  Real z_SN_offset        = pin->GetOrAddReal("problem", "z_SN_offset", 0.0);


  if (MAGNETIC_FIELDS_ENABLED) {
    beta             = pin->GetReal("problem", "beta");
  }


  // Set up a list of SNe locations and times, same for all processors
  // total number of SNe = M_cluster/m_star
  // spread out randomly in time between t_start_SN and t_start_SN+t_SNe
  // and randomly in a sphere of radius R_cluster
  Real realN_SNe = M_cluster/m_star;
  int N_SNe = (int)realN_SNe;
  if(Globals::my_rank==0) {
    std::cout << "N_SNe " << N_SNe << "\n";
  }

  Real dx = (mesh_size.x1max - mesh_size.x1min) / mesh_size.nx1 ;
  Real dy = (mesh_size.x2max - mesh_size.x2min) / mesh_size.nx2 ;
  Real dz = (mesh_size.x3max - mesh_size.x3min) / mesh_size.nx3 ;

  for (int i = 0; i < 300; ++i) {
    if (mesh_size.nx3 % 2 == 0){
      z_SN[i] = (((Real) std::rand() / (RAND_MAX)) * 2. * R_cluster) - R_cluster;
      z_SN[i] = round(z_SN[i]/dz)*dz;
    } else {
      z_SN[i] = (((Real) std::rand() / (RAND_MAX)) * 2. * R_cluster) - R_cluster + dz/2.;
      z_SN[i] = round(z_SN[i]/dz)*dz - dz/2.;
    }
    y_SN[i] = (((Real) std::rand() / (RAND_MAX)) * 2. * std::sqrt(SQR(R_cluster) - SQR(z_SN[i]))) - std::sqrt(SQR(R_cluster) - SQR(z_SN[i]));
    y_SN[i] = round(y_SN[i]/dy)*dy;
    x_SN[i] = (((Real) std::rand() / (RAND_MAX)) * 2. * std::sqrt(SQR(R_cluster) - SQR(z_SN[i]) - SQR(y_SN[i]))) - std::sqrt(SQR(R_cluster) - SQR(z_SN[i]) - SQR(y_SN[i]));
    x_SN[i] = round(z_SN[i]/dx)*dx;
    z_SN[i] += z_SN_offset*dz;
    y_SN[i] += y_SN_offset*dy;
    x_SN[i] += x_SN_offset*dx;
    if(Globals::my_rank==0) {
      std::cout << "x y z " << x_SN[i] << " " << y_SN[i] << " " << z_SN[i] << "\n";
    }
  }
  // initialize the SN counter to 0
  i_SN = 0; 
  dt_SN = t_SNe/N_SNe;
  t_last_SN = -dt_SN;

  if(Globals::my_rank==0) {
    std::cout << "dt_SN in code units " << dt_SN << "\n";
    std::cout << "t_SNe in code units " << t_SNe << "\n";
  }

  //------------------------------------------------------------------------------------------//
  //------------------------------------------------------------------------------------------//
  //------------------------------------------------------------------------------------------//


  AllocateRealUserMeshDataField(1);

  // Allocate variable source term storage
  ruser_mesh_data[0].NewAthenaArray(4);


  // Enroll source function
  EnrollUserExplicitSourceFunction(SourceFunction);

  // Enroll no inflow boundary condition but only if it is turned on
  if(mesh_bcs[INNER_X3] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(INNER_X3, NoInflowInnerX3);
  }
  if(mesh_bcs[OUTER_X3] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(OUTER_X3, NoInflowOuterX3);
  }

  
  // Enroll user-defined functions
  AllocateUserHistoryOutput(35);
  EnrollUserHistoryOutput(0, CoolingLosses, "e_cool");
  EnrollUserHistoryOutput(1, CoolingLosses, "e_SN");
  EnrollUserHistoryOutput(2, CoolingLosses, "e_ceil");

  EnrollUserHistoryOutput(3 , fluxes, "Edt20");
  EnrollUserHistoryOutput(4 , fluxes, "Mdt20");
  EnrollUserHistoryOutput(5 , fluxes, "Edo20");
  EnrollUserHistoryOutput(6 , fluxes, "Mdo20");
  EnrollUserHistoryOutput(7 , fluxes, "Edt20c");
  EnrollUserHistoryOutput(8 , fluxes, "Mdt20c");
  EnrollUserHistoryOutput(9 , fluxes, "Edo20c");
  EnrollUserHistoryOutput(10, fluxes, "Mdo20c");
  EnrollUserHistoryOutput(11, fluxes, "Edt20w");
  EnrollUserHistoryOutput(12, fluxes, "Mdt20w");
  EnrollUserHistoryOutput(13, fluxes, "Edo20w");
  EnrollUserHistoryOutput(14, fluxes, "Mdo20w");
  EnrollUserHistoryOutput(15, fluxes, "Edt20h");
  EnrollUserHistoryOutput(16, fluxes, "Mdt20h");
  EnrollUserHistoryOutput(17, fluxes, "Edo20h");
  EnrollUserHistoryOutput(18, fluxes, "Mdo20h");
  EnrollUserHistoryOutput(19, fluxes, "Edt54");
  EnrollUserHistoryOutput(20, fluxes, "Mdt54");
  EnrollUserHistoryOutput(21, fluxes, "Edo54");
  EnrollUserHistoryOutput(22, fluxes, "Mdo54");
  EnrollUserHistoryOutput(23, fluxes, "Edt54c");
  EnrollUserHistoryOutput(24, fluxes, "Mdt54c");
  EnrollUserHistoryOutput(25, fluxes, "Edo54c");
  EnrollUserHistoryOutput(26, fluxes, "Mdo54c");
  EnrollUserHistoryOutput(27, fluxes, "Edt54w");
  EnrollUserHistoryOutput(28, fluxes, "Mdt54w");
  EnrollUserHistoryOutput(29, fluxes, "Edo54w");
  EnrollUserHistoryOutput(30, fluxes, "Mdo54w");
  EnrollUserHistoryOutput(31, fluxes, "Edt54h");
  EnrollUserHistoryOutput(32, fluxes, "Mdt54h");
  EnrollUserHistoryOutput(33, fluxes, "Edo54h");
  EnrollUserHistoryOutput(34, fluxes, "Mdo54h");

  // Enroll timestep so that dt <= min t_cool
  EnrollUserTimeStepFunction(cooling_timestep);

  // Enroll amr condition
  if(adaptive==true){
    std::cout << "turning on AMR \n";
    EnrollUserRefinementCondition(RefinementCondition);
  }

  if(Globals::my_rank==0) {
    std::cout << "Set it all up \n";
  }
}

//-----------------------------------------------------------------------------
// Function for preparing MeshBlock
// Inputs:
//   pin: input parameters (unused)
// Outputs: (none)

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin)
{

  // Allocate storage for keeping track of cooling
  AllocateRealUserMeshBlockDataField(2);
  ruser_meshblock_data[0].NewAthenaArray(3);
  ruser_meshblock_data[0](0) = 0.0; // e_cool
  ruser_meshblock_data[0](1) = 0.0; // e_SN
  ruser_meshblock_data[0](2) = 0.0; // e_ceil
  // ruser_meshblock_data[0](3) = 0.0; // e_floor
  
  ruser_meshblock_data[1].NewAthenaArray(32);
  for (int i = 0; i < 32; ++i) {
    ruser_meshblock_data[1](i) = 0.0; 
  }
  

  // Set output variables
  AllocateUserOutputVariables(1);
  // SetUserOutputVariableName(0, "edot");
  return;
}

//-----------------------------------------------------------------------------
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
  Real cs2 = pgas_0/rho_0;
  if(Globals::my_rank==0) {
        std::cout << " vc2o2r2 " << vc2o2r2 << "\n";
        std::cout << " cs2 " << cs2 << "\n";
        std::cout << " rho_0 " << rho_0 << "\n";
        std::cout << " rho_floor " << rho_floor << "\n";
  }
  // Initialize primitive values
  for (int k = kl; k <= ku; ++k) {
    Real z = pcoord->x3v(k);
    for (int j = jl; j <= ju; ++j) {
      Real y = pcoord->x2v(j);
      for (int i = il; i <= iu; ++i) {
        Real x = pcoord->x1v(i);
        Real RHO = rho_0 * std::exp(-1.0*vc2o2r2*SQR(z)/cs2);
        if (RHO > rho_floor){
          phydro->w(IDN,k,j,i) = rho_0 * std::exp(-1.0*vc2o2r2*SQR(z)/cs2);
          phydro->w(IPR,k,j,i) = cs2 * phydro->w(IDN,k,j,i); 
        } else {
          phydro->w(IDN,k,j,i) = rho_floor;
          phydro->w(IPR,k,j,i) = rho_floor * (cs2 * (1.0 + std::log(rho_0/rho_floor)) - vc2o2r2*SQR(z));
        }
        phydro->w(IVX,k,j,i) = 0.0;
        phydro->w(IVY,k,j,i) = 0.0;
        phydro->w(IVZ,k,j,i) = 0.0;
      }
    }
  }


  // initialize interface B, assuming vertical field only B=(0,0,1)
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke+1; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie+1; i++) {
          pfield->b.x1f(k,j,i) = 0.0;
          pfield->b.x2f(k,j,i) = 0.0;
          pfield->b.x3f(k,j,i) = std::sqrt(2.0 * phydro->w(IPR,k,j,i)/beta);
        }
      }
    }
    // initialize total energy
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie; i++) {
          phydro->u(IEN,k,j,i) += 0.5*SQR(pfield->b.x3f(k,j,i));
        }
      }
    }
    // Initialize conserved values
    peos->PrimitiveToConserved(phydro->w, pfield->bcc, phydro->u, pcoord, il, iu, jl, ju, kl, ku);
  } else {
    // Initialize conserved values
    AthenaArray<Real> b;
    peos->PrimitiveToConserved(phydro->w, b, phydro->u, pcoord, il, iu, jl, ju, kl, ku);
  }
  return;
}




//----------------------------------------------------------------------------------------
// Function to calculate the timestep required to resolve cooling 
//          tcool = 3/2 P/Edot_cool
// Inputs:
//   pmb: pointer to MeshBlock
Real cooling_timestep(MeshBlock *pmb)
{
  Real min_dt=1.0e10;
  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        Real &P = pmb->phydro->w(IPR,k,j,i);
        Real &rho = pmb->phydro->w(IDN,k,j,i);

        Real T_before = mu * mp * (P/rho)*(pgas_scale/rho_scale) / kb;
        Real nH = rho*rho_scale/(muH*mp);
        if (T_before > 1.01 * T_floor){
          min_dt = std::min(min_dt, 0.25 * std::abs(tcool(T_before,nH))/time_scale );
        }
        min_dt = std::max(dt_cutoff,min_dt);
      }
    }
  }
  return min_dt;
}




//----------------------------------------------------------------------------------------
// Source function for SourceFunction 
// Inputs:
//   pmb: pointer to MeshBlock
//   t,dt: time (not used) and timestep
//   prim: primitives
//   bcc: cell-centered magnetic fields (not used)
// Outputs:
//   cons: conserved variables updated

void SourceFunction(MeshBlock *pmb, const Real t, const Real dt,
    const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc, 
    AthenaArray<Real> &cons)
{

//_______________ GRAVITY source term first _______________//
  if (vc2o2r2 > 0.){
    for (int k=pmb->ks; k<=pmb->ke; ++k) {
      Real z = pmb->pcoord->x3v(k);
      for (int j=pmb->js; j<=pmb->je; ++j) {
        for (int i=pmb->is; i<=pmb->ie; ++i) {
          Real delta_vz = dt*grav_accel(z);
          cons(IM3,k,j,i) -= prim(IDN,k,j,i) * delta_vz;
          if (NON_BAROTROPIC_EOS) cons(IEN,k,j,i) -= 0.5*prim(IDN,k,j,i)*(2.*delta_vz*prim(IVZ,k,j,i) - SQR(delta_vz));
        }
      }
    }
  }

//_______________ COOLING source term next _______________//
  Real vol_cell = pmb->pcoord->dx1f(0)*pmb->pcoord->dx2f(0)*pmb->pcoord->dx3f(0);

  // Determine which part of step this is
  bool predict_step = prim.data() == pmb->phydro->w.data();

  AthenaArray<Real> edot;
  edot.InitWithShallowSlice(pmb->user_out_var, 4, 0, 1);

  Real delta_e_block = 0.0;
  Real delta_e_ceil_block  = 0.0;
  Real pc = 3.086e+18; 
  
  // zero out the fluxes
  if (pmb->lid == 0) {
    pmb->ruser_meshblock_data[1].NewAthenaArray(32); // is this supposed to be here???
    for (int i = 0; i < 32; ++i) {
      pmb->ruser_meshblock_data[1](i) = 0.0; 
    }
  }
  // Real delta_e_floor = 0.0;

  // Extract indices
  int is = pmb->is, ie = pmb->ie;
  int js = pmb->js, je = pmb->je;
  int ks = pmb->ks, ke = pmb->ke;

  int i_flux_z,i_flux_T;
  Real z_top = pmb->pmy_mesh->mesh_size.x3max;

  for (int k = ks; k <= ke; ++k) {
    i_flux_z = -1;
    Real z = pmb->pcoord->x3v(k);
    if ( ((std::abs(z) - 200.*pc/length_scale) > -1.0*pmb->pcoord->dx3v(k)) and ((std::abs(z) - 200.*pc/length_scale) <= 0.) ){
      i_flux_z=0;
    } 
    if ( ((std::abs(z) - 540.*pc/length_scale) > -1.0*pmb->pcoord->dx3v(k)) and ((std::abs(z) - 540.*pc/length_scale) <= 0.) ){
      i_flux_z=16;
    } 
    for (int j = js; j <= je; ++j) {
      Real y = pmb->pcoord->x2v(j);
      for (int i = is; i <= ie; ++i) {
        Real x = pmb->pcoord->x1v(i);
        // Extract primitive and conserved quantities
        const Real &rho_half = prim(IDN,k,j,i);
        const Real &pgas_half = prim(IPR,k,j,i);
        Real &rho = cons(IDN,k,j,i);
        Real &e = cons(IEN,k,j,i);
        Real &m1 = cons(IM1,k,j,i);
        Real &m2 = cons(IM2,k,j,i);
        Real &m3 = cons(IM3,k,j,i);
        const Real &v1 = prim(IVX,k,j,i);
        const Real &v2 = prim(IVY,k,j,i);
        const Real &v3 = prim(IVZ,k,j,i);
        Real dA = pmb->pcoord->dx1v(k)*pmb->pcoord->dx2v(k);
        Real kinetic = (SQR(m1) + SQR(m2) + SQR(m3)) / (2.0 * rho);
        Real u = e - kinetic; 
        if (MAGNETIC_FIELDS_ENABLED) {
          const Real &bcc1 = bcc(IB1,k,j,i);
          const Real &bcc2 = bcc(IB2,k,j,i);
          const Real &bcc3 = bcc(IB3,k,j,i);
          Real magnetic = 0.5*(SQR(bcc1) + SQR(bcc2) + SQR(bcc3));
          u -= magnetic; 
        } 
        Real P = u * (gamma_adi-1.0);

        // calculate temperature in physical units before cooling
        Real T_before = mu * mp * (P/rho)*(pgas_scale/rho_scale) / kb;
        Real nH = rho*rho_scale/(muH*mp);

        int n_subcycle = (int) ceil(dt / (0.1 * std::abs(tcool(T_before,nH))/time_scale) );
        n_subcycle = std::min(100, n_subcycle);
        Real dt_cool = dt*time_scale/n_subcycle;

        Real T_update = 0.;
        T_update += T_before;
        for (int i_subcycle = 0; i_subcycle < n_subcycle; ++i_subcycle) {
          // dT/dt = - T/tcool(T,nH)
          Real k1 = -1.0 * (T_update/tcool(T_update, nH));
          Real k2 = -1.0 * (T_update + 0.5*dt_cool * k1)/tcool(T_update + 0.5*dt_cool * k1, nH);
          Real k3 = -1.0 * (T_update + 0.5*dt_cool * k2)/tcool(T_update + 0.5*dt_cool * k2, nH);
          Real k4 = -1.0 * (T_update + dt_cool * k3)/tcool(T_update + dt_cool * k3, nH);
          T_update += (k1 + 2.*k2 + 2.*k3 + k4)/6.0 * dt_cool; 
        }
        
        Real u_after = (kb*std::max(T_update,T_floor) / (mu * mp) * rho * (rho_scale/pgas_scale))/(gamma_adi-1.0);

        Real delta_e_ceil = 0.0;
        if (T_update > T_max){
          delta_e_ceil -= u_after;
          u_after = (kb*std::min(T_update,T_max) / (mu * mp) * rho * (rho_scale/pgas_scale))/(gamma_adi-1.0);
          delta_e_ceil += u_after;
          // std::cout << " T_update > T_max, T_update = " << T_update << " delta_e_ceil = " << delta_e_ceil << "\n";
        }

        Real delta_e = u_after - u;

        if (t >= t_start_cooling){
          e += delta_e;  
        }

        if ((i_flux_z >= 0) and (not predict_step) and (vc2o2r2 > 0.)){  
          Real Mdot = z < 0.0 ? -1*rho*dA*v3 : rho*dA*v3 ;
          Real Edot = Mdot*( 0.5*(SQR(v1)+SQR(v2)+SQR(v3)) + 2.5*P/rho - vc2o2r2*(SQR(z_top)-SQR(z))); 
          Real T = std::min(T_max,std::max(T_update,T_floor));
          
          i_flux_T = 0;
          if (T < 1.0e4){
            i_flux_T = 4;
          } else if (T < 1.0e6){
            i_flux_T = 8;
          } else {
            i_flux_T = 12;
          }

          pmb->ruser_meshblock_data[1](i_flux_T+i_flux_z)   += Edot;
          pmb->ruser_meshblock_data[1](i_flux_T+i_flux_z+1) += Mdot;
          pmb->ruser_meshblock_data[1](i_flux_z)   += Edot;
          pmb->ruser_meshblock_data[1](i_flux_z+1) += Mdot;
          if (z*v3 > 0){
            pmb->ruser_meshblock_data[1](i_flux_T+i_flux_z+2) += Edot;
            pmb->ruser_meshblock_data[1](i_flux_T+i_flux_z+3) += Mdot;
            pmb->ruser_meshblock_data[1](i_flux_z+2) += Edot;
            pmb->ruser_meshblock_data[1](i_flux_z+3) += Mdot;
          } 
        }

        if ((vc2o2r2 <= 0.) and (T_update>1e5)){
          pmb->ruser_meshblock_data[1](0) += pmb->pcoord->dx3v(k)*pmb->pcoord->dx2v(j)*pmb->pcoord->dx1v(i);
          pmb->ruser_meshblock_data[1](1) += e * (pmb->pcoord->dx3v(k)*pmb->pcoord->dx2v(j)*pmb->pcoord->dx1v(i));
          pmb->ruser_meshblock_data[1](2) += (e - kinetic)* (pmb->pcoord->dx3v(k)*pmb->pcoord->dx2v(j)*pmb->pcoord->dx1v(i));
          pmb->ruser_meshblock_data[1](3) += rho * (pmb->pcoord->dx3v(k)*pmb->pcoord->dx2v(j)*pmb->pcoord->dx1v(i));
          pmb->ruser_meshblock_data[1](4) += T_update;
          pmb->ruser_meshblock_data[1](5) += rho;
        }


        edot(k,j,i) = delta_e / dt;

        if (not predict_step) {
          delta_e_block += delta_e;
          delta_e_ceil_block += delta_e_ceil;
        }
      }
    }
  }

  pmb->ruser_meshblock_data[0](0) += delta_e_block;
  pmb->ruser_meshblock_data[0](2) += delta_e_ceil_block;

  // Free arrays
  edot.DeleteAthenaArray();

//_______________ SN INJ. source term last _______________//
  if ((t - t_last_SN > dt_SN) and (t>=t_start_SN) and (not predict_step)) {
    Real N_cells         = 0.0;
    Real average_density = 0.0;
    Real total_m         = 0.0;
    Real total_e         = 0.0;
    Real total_ke        = 0.0;
    // Calculate average density and Ncells on all blocks
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
        AthenaArray<Real> cons_local;
        cons_local.InitWithShallowCopy(pblock->phydro->u);
  
        for (int k = ks; k <= ke; ++k) {
          Real z = pblock->pcoord->x3v(k);
          if (std::abs(z) < 50.*pc/length_scale){
            for (int j = js; j <= je; ++j) {
              Real y = pblock->pcoord->x2v(j);
              if (std::abs(y) < 50.*pc/length_scale){
                for (int i = is; i <= ie; ++i) {
                  Real x = pblock->pcoord->x1v(i);
                  if (std::abs(x) < 50.*pc/length_scale){
                    if ( SQR(z-z_SN[i_SN%300]) + SQR(y-y_SN[i_SN%300]) + SQR(x-x_SN[i_SN%300]) < r_inj_sq*SQR(pblock->pcoord->dx3f(k))){
                      N_cells += 1.0;
                      average_density += cons_local(IDN,k,j,i);
                      total_m += cons_local(IDN,k,j,i) * vol_cell;
                      total_e += cons_local(IEN,k,j,i);
                      total_ke += ( SQR(cons_local(IM1,k,j,i)) +
                                    SQR(cons_local(IM2,k,j,i)) +
                                    SQR(cons_local(IM3,k,j,i)) )
                                / (2. * cons_local(IDN,k,j,i));
                    }
                  }
                }
              }
            }
          }
        }
        // Prepare for next block on mesh
        cons_local.DeleteAthenaArray();
        pblock = pblock->next;
      }
    }


    // Aggregate across all ranks
    if (pmb->lid == 0) {
      Real aggregates_mesh[5];
      aggregates_mesh[0] = N_cells;
      aggregates_mesh[1] = average_density;
      aggregates_mesh[2] = total_m;
      aggregates_mesh[3] = total_e;
      aggregates_mesh[4] = total_ke;
      Real aggregates_tot[5];
      #ifdef MPI_PARALLEL
      {
        MPI_Allreduce(aggregates_mesh, aggregates_tot, 5, MPI_ATHENA_REAL, MPI_SUM,
            MPI_COMM_WORLD);
      }
      #else
      {
        for (int i = 0; i < 5; ++i) {
          aggregates_tot[i] = aggregates_mesh[i];
        }
      }
      #endif
      N_cells         = aggregates_tot[0];
      average_density = aggregates_tot[1]/N_cells;
      total_m         = aggregates_tot[2];
      total_e         = aggregates_tot[3];
      total_ke        = aggregates_tot[4];
      // std::cout << " Globals::my_rank " << Globals::my_rank << " N_cells = " << N_cells << " average_density = " << average_density
                // << " total_m = " << total_m << " total_e = " << total_e << " total_ke = " << total_ke << "\n";
    }

    Real E_SN_th, rho_ej, v_ej;
    // Store or extract average density
    if (pmb->lid == 0) {
      // Martizzi+15 fitting params
      Real average_n_H = average_density*rho_scale/(muH*mp);
      Real alpha       = -7.8 *pow(average_n_H/100.,0.03);
      Real r_cool      = 3.0  *pow(average_n_H/100.,-0.42) * pc / length_scale;
      Real r_rise      = 5.5  *pow(average_n_H/100.,-0.40) * pc / length_scale;
      Real r_0         = 0.97 *pow(average_n_H/100.,-0.33) * pc / length_scale;
      Real r_break     = 4.0  *pow(average_n_H/100.,-0.43) * pc / length_scale;
      // Martizzi+15 radial momentum
      Real P_SN_rad;
      if ( sqrt(r_inj_sq)*pmb->pcoord->dx3f(0) < r_break ){
        P_SN_rad= P_SN * pow(sqrt(r_inj_sq)*pmb->pcoord->dx3f(0)/r_0,1.5);
      } else {
        P_SN_rad= P_SN * pow(r_break/r_0,1.5);
      }
      // Martizzi+15 thermal energy
      if ( sqrt(r_inj_sq)*pmb->pcoord->dx3f(0) < r_cool ){
        E_SN_th = E_SN;
      } else {
        if ( sqrt(r_inj_sq)*pmb->pcoord->dx3f(0) < r_rise){
          E_SN_th = E_SN * pow( sqrt(r_inj_sq)*pmb->pcoord->dx3f(0) / r_cool , alpha);
        } else {
          E_SN_th = E_SN * pow(r_rise/r_cool,alpha);
        }
      }
      E_SN_th /= (N_cells*vol_cell);
      rho_ej = ejecta_mass/vol_cell/N_cells;
      v_ej   = P_SN_rad/(vol_cell*N_cells*(average_density + rho_ej)) * sqrt((average_density + rho_ej)/average_density);
      if(Globals::my_rank==0) {
        std::cout << " boom N_cells " << N_cells <<  " i_SN " << i_SN <<   
                     " x " << x_SN[i_SN%300] <<   " y " << y_SN[i_SN%300] <<   " z " << z_SN[i_SN%300] <<   
                     " average_n_H " << average_n_H <<  " average_density " << average_density << 
                     " P_SN_rad " << P_SN_rad <<  " E_SN_th " << E_SN_th <<  " v_ej " << v_ej << 
                     " alpha " << alpha <<  " r_cool " << r_cool <<  " r_rise " << r_rise <<  " r_0 " << r_0 <<  " r_break " << r_break <<  " r_inj " << sqrt(r_inj_sq)*pmb->pcoord->dx3f(0) <<  "\n";
      }

      // store
      pmb->pmy_mesh->ruser_mesh_data[0](0) = average_density;
      pmb->pmy_mesh->ruser_mesh_data[0](1) = E_SN_th;
      pmb->pmy_mesh->ruser_mesh_data[0](2) = v_ej;
      pmb->pmy_mesh->ruser_mesh_data[0](3) = rho_ej;
    } else {
      // extract
      average_density  = pmb->pmy_mesh->ruser_mesh_data[0](0);
      E_SN_th          = pmb->pmy_mesh->ruser_mesh_data[0](1);
      v_ej             = pmb->pmy_mesh->ruser_mesh_data[0](2);
      rho_ej           = pmb->pmy_mesh->ruser_mesh_data[0](3);
    }

    // std::cout << " Globals::my_rank " << Globals::my_rank <<  " average_density = " << average_density << " E_SN_th = " << E_SN_th << " v_ej = " << v_ej << " rho_ej = " << rho_ej << "\n";

    // do the damn thing
    Real delta_e  = 0.0;    // Real delta_ke = 0.0;    // Real delta_m  = 0.0;
    for (int k = ks; k <= ke; ++k) {
      Real z = pmb->pcoord->x3v(k);
      if (std::abs(z) < 50.*pc/length_scale){
        for (int j = js; j <= je; ++j) {
          Real y = pmb->pcoord->x2v(j);
          if (std::abs(y) < 50.*pc/length_scale){
            for (int i = is; i <= ie; ++i) {
              Real x = pmb->pcoord->x1v(i);
              if (std::abs(x) < 50.*pc/length_scale){
                Real distance_to_SNe = sqrt(SQR(z-z_SN[i_SN%300]) + SQR(y-y_SN[i_SN%300]) + SQR(x-x_SN[i_SN%300]));
                if ( distance_to_SNe <= sqrt(r_inj_sq)*pmb->pcoord->dx3f(k) ){
                  Real &rho = cons(IDN,k,j,i);
                  Real &e   = cons(IEN,k,j,i);
                  Real &m1  = cons(IM1,k,j,i);
                  Real &m2  = cons(IM2,k,j,i);
                  Real &m3  = cons(IM3,k,j,i);
                  Real KE_init = 0.5 * ( SQR(m1)+SQR(m2)+SQR(m3) ) / rho; 

                  // std::cout << "e_before " << total_e <<" ke_before " << total_ke << " m_before " << total_m << "\n";

                  delta_e  -= e;                   // delta_ke -= KE_init;                  // delta_m  -= rho*vol_cell;
                  rho = average_density + rho_ej;
                  e += E_SN_th;
                  m1 += (average_density + rho_ej) * v_ej * (x-x_SN[i_SN%300])/distance_to_SNe;
                  m2 += (average_density + rho_ej) * v_ej * (y-y_SN[i_SN%300])/distance_to_SNe;
                  m3 += (average_density + rho_ej) * v_ej * (z-z_SN[i_SN%300])/distance_to_SNe;
                  Real KE_final = 0.5 * ( SQR(m1)+SQR(m2)+SQR(m3) ) / rho; 
                  e += KE_final - KE_init; 
                  delta_e += e;          
                }
              }
            }
          }
        }
      }
    }    
    // if(Globals::my_rank==0) {
    //   std::cout << "e_after " << total_e_after <<" ke_after " << total_ke_after << " m_after " << total_m_after << "\n";
    // }
    pmb->ruser_meshblock_data[0](1) += delta_e; 
    if (pmb->lid == pmb->pmy_mesh->GetNumMeshBlocksThisRank(Globals::my_rank)-1){
      i_SN += 1;
      t_last_SN = t; 
    }
  }
  return;
}


//----------------------------------------------------------------------------------------
// Gravitational acceleration
// Inputs:
//   z: the vertical height in code units
// Outputs:
//   returned value: gravitational acceleration

static Real grav_accel( Real z )
{
  return 2.*vc2o2r2*z;
}




//----------------------------------------------------------------------------------------
//! \fn void NoInflowInnerX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
//                          FaceField &b, Real time, Real dt,
//                          int is, int ie, int js, int je, int ks, int ke, int ngh)
//  \brief OUTFLOW boundary conditions with no inflow, inner x3 boundary

void NoInflowInnerX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh)
{
  // copy hydro variables into ghost zones
  for (int n=0; n<(NHYDRO); ++n) {
    for (int k=1; k<=(ngh); ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie; ++i) {
          prim(n,ks-k,j,i) = prim(n,ks,j,i);
          if ( n == IVZ ){
            if ( prim(IVZ,ks-k,j,i) > 0.0 ){
              prim(IVZ,ks-k,j,i) = 0.0;
            }
          }
        }
      }
    }
  }

  // copy face-centered magnetic fields into ghost zones
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=1; k<=(ngh); ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie+1; ++i) {
        b.x1f((ks-k),j,i) = b.x1f(ks,j,i);
      }
    }}

    for (int k=1; k<=(ngh); ++k) {
    for (int j=js; j<=je+1; ++j) {
      for (int i=is; i<=ie; ++i) {
        b.x2f((ks-k),j,i) = b.x2f(ks,j,i);
      }
    }}

    for (int k=1; k<=(ngh); ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        b.x3f((ks-k),j,i) = b.x3f(ks,j,i);
      }
    }}
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void NoInflowOuterX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
//                          FaceField &b, Real time, Real dt,
//                          int is, int ie, int js, int je, int ks, int ke, int ngh)
//  \brief OUTFLOW boundary conditions with no inflow, outer x3 boundary

void NoInflowOuterX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
     FaceField &b, Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh)
{
  // copy hydro variables into ghost zones
  for (int n=0; n<(NHYDRO); ++n) {
    for (int k=1; k<=(ngh); ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie; ++i) {
          prim(n,ke+k,j,i) = prim(n,ke,j,i);
          if ( n == IVZ ){
            if ( prim(IVZ,ke+k,j,i) < 0.0 ){
              prim(IVZ,ke+k,j,i) = 0.0;
            }
          } 
        }
      }
    }
  }

  // copy face-centered magnetic fields into ghost zones
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=1; k<=(ngh); ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie+1; ++i) {
        b.x1f((ke+k  ),j,i) = b.x1f((ke  ),j,i);
      }
    }}

    for (int k=1; k<=(ngh); ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        b.x2f((ke+k  ),j,i) = b.x2f((ke  ),j,i);
      }
    }}

    for (int k=1; k<=(ngh); ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        b.x3f((ke+k+1),j,i) = b.x3f((ke+1),j,i);
      }
    }}
  }

  return;
}


int RefinementCondition(MeshBlock *pmb)
{
  return 1;
}


//----------------------------------------------------------------------------------------
// fluxes
// Inputs:
//   pmb: pointer to MeshBlock
//   iout: index of history output
// Outputs:
//   returned value: mass and energy fluxes at 200 pc and top of box 
//                   split by total and outflowing and by temperature
// Notes:
//   0-3  : 200pc:           Edot_tot Mdot_tot Edot_out Mdot_out
//   4-7  : 200pc: 2<logT<4: Edot_tot Mdot_tot Edot_out Mdot_out
//   8-11 : 200pc: 4<logT<6: Edot_tot Mdot_tot Edot_out Mdot_out
//   12-15: 200pc: 6<logT  : Edot_tot Mdot_tot Edot_out Mdot_out
//   16-19: 540pc:           Edot_tot Mdot_tot Edot_out Mdot_out
//   20-23: 540pc: 2<logT<4: Edot_tot Mdot_tot Edot_out Mdot_out
//   24-27: 540pc: 4<logT<6: Edot_tot Mdot_tot Edot_out Mdot_out
//   28-31: 540pc: 6<logT  : Edot_tot Mdot_tot Edot_out Mdot_out

Real fluxes(MeshBlock *pmb, int iout)
{
  Real flux = pmb->ruser_meshblock_data[1](iout-3);
  pmb->ruser_meshblock_data[1](iout-3) = 0.0;
  return flux;
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
//     1: SN injection
//     2: numerical temperature ceiling
//     3: numerical temperature floor

Real CoolingLosses(MeshBlock *pmb, int iout)
{
  Real delta_e = pmb->ruser_meshblock_data[0](iout);
  pmb->ruser_meshblock_data[0](iout) = 0.0;
  return delta_e;
}
