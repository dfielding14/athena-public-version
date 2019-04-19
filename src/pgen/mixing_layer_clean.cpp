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
#include "../hydro/hydro_diffusion/hydro_diffusion.hpp" // diffusion

// External library headers
#include <hdf5.h>  // H5*, hid_t, hsize_t, H5*()

// Declarations
void Cooling_Source_Function(MeshBlock *pmb, const Real t, const Real dt,
    const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc, AthenaArray<Real> &cons, int stage);
Real CoolingLosses(MeshBlock *pmb, int iout);
static Real edot_cool(Real press, Real dens);


// User defined conduction and viscosity
void SmagorinskyConduction(HydroDiffusion *phdif, MeshBlock *pmb, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &bcc, int is, int ie, int js, int je, int ks, int ke);
void SmagorinskyViscosity(HydroDiffusion *phdif, MeshBlock *pmb, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &bcc, int is, int ie, int js, int je, int ks, int ke);
void DeviatoricSmagorinskyConduction(HydroDiffusion *phdif, MeshBlock *pmb, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &bcc, int is, int ie, int js, int je, int ks, int ke);
void DeviatoricSmagorinskyViscosity(HydroDiffusion *phdif, MeshBlock *pmb, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &bcc, int is, int ie, int js, int je, int ks, int ke);


// Boundary Conditions
void ConstantShearInflowOuterX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                    FaceField &b, Real time, Real dt,
                    int is, int ie, int js, int je, int ks, int ke, int ngh);
void ConstantShearInflowInnerX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                    FaceField &b, Real time, Real dt,
                    int is, int ie, int js, int je, int ks, int ke, int ngh);

// Global variables
static Real gamma_adi;
static Real rho_0, pgas_0;
static Real density_contrast,velocity;
static Real Lambda_cool, s_Lambda, t_cool_start;

static Real cooling_timestep(MeshBlock *pmb);
static Real dt_cutoff, cfl_cool;
static Real smoothing_thickness, velocity_pert, lambda_pert, z_top, z_bot;

//----------------------------------------------------------------------------------------
// Function for preparing Mesh
// Inputs:
//   pin: input parameters (unused)
// Outputs: (none)

void Mesh::InitUserMeshData(ParameterInput *pin)
{
// turb_flag is initialzed in the Mesh constructor to 0 by default;
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
  gamma_adi              = pin->GetReal("hydro",   "gamma");
  Real length_scale      = pin->GetReal("problem", "length_scale");
  Real rho_scale         = pin->GetReal("problem", "rho_scale");
  Real pgas_scale        = pin->GetReal("problem", "pgas_scale");
  Real temperature_scale = pin->GetReal("problem", "temperature_scale");
  Real vel_scale         = std::sqrt(pgas_scale / rho_scale);
  rho_0                  = pin->GetReal("problem", "rho_0");
  pgas_0                 = pin->GetReal("problem", "pgas_0");
  density_contrast       = pin->GetReal("problem", "density_contrast");
  velocity               = pin->GetReal("problem", "velocity");

  // Read cooling-table-related parameters from input file
  Lambda_cool = pin->GetReal("problem", "Lambda_cool");
  s_Lambda = pin->GetReal("problem", "s_Lambda");

  // Initial conditions and Boundary values
  smoothing_thickness = pin->GetReal("problem", "smoothing_thickness");
  velocity_pert       = pin->GetReal("problem", "velocity_pert");
  lambda_pert         = pin->GetReal("problem", "lambda_pert");
  z_top               = pin->GetReal("problem", "z_top");
  z_bot               = pin->GetReal("problem", "z_bot");

  // Allocate fixed cooling table
  AllocateRealUserMeshDataField(1);
  ruser_mesh_data[0].NewAthenaArray(3); // delta_e_tot, vol_tot, vol_cell

  Real vol_tot = (mesh_size.x1max - mesh_size.x1min)
               * (mesh_size.x2max - mesh_size.x2min) 
               * (mesh_size.x3max - mesh_size.x3min);
  Real vol_cell = vol_tot / mesh_size.nx1 / mesh_size.nx2 / mesh_size.nx3;
  ruser_mesh_data[0](1) = vol_tot;
  ruser_mesh_data[0](2) = vol_cell;

  // Enroll user-defined functions
  EnrollUserExplicitSourceFunction(Cooling_Source_Function);
  AllocateUserHistoryOutput(2);
  EnrollUserHistoryOutput(0, CoolingLosses, "e_cool");
  EnrollUserHistoryOutput(1, CoolingLosses, "e_ceil");
  EnrollUserTimeStepFunction(cooling_timestep);

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

  // Enroll no inflow boundary condition but only if it is turned on
  if(mesh_bcs[INNER_X3] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(INNER_X3, ConstantShearInflowInnerX3);
  }
  if(mesh_bcs[OUTER_X3] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(OUTER_X3, ConstantShearInflowOuterX3);
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
  ruser_meshblock_data[0].NewAthenaArray(1);
  ruser_meshblock_data[0](0) = 0.0;
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

  Real beta = pin->GetOrAddReal("problem", "beta", 100.0);
  int B_direction = pin->GetOrAddInteger("problem", "B_direction", 0); // 0 = x, 1 = y, 2 = z

  // Initialize primitive values
  for (int k = kl; k <= ku; ++k) {
    Real z = pcoord->x3v(k);
    for (int j = jl; j <= ju; ++j) {
      Real y = pcoord->x2v(j);
      for (int i = il; i <= iu; ++i) {
        Real x = pcoord->x1v(i);
        if (block_size.nx3 > 1) {
          phydro->w(IDN,k,j,i) = rho_0 * (1.0 + (density_contrast-1.0) * 0.5 * ( std::tanh((z-z_bot)/smoothing_thickness) - std::tanh((z-z_top)/smoothing_thickness) ) );
          phydro->w(IPR,k,j,i) = pgas_0;
          phydro->w(IVX,k,j,i) = velocity * ( 0.5 - 0.5 * ( std::tanh((z-z_bot)/smoothing_thickness) - std::tanh((z-z_top)/smoothing_thickness) ));
          phydro->w(IVY,k,j,i) = 0.0;
          phydro->w(IVZ,k,j,i) = velocity_pert * (std::exp(-SQR((z-z_bot)/smoothing_thickness)) + std::exp(-SQR((z-z_top)/smoothing_thickness))) * std::sin(2*PI*x/lambda_pert) * std::sin(2*PI*y/lambda_pert) ;
        } else if (block_size.nx2 > 1) {
          phydro->w(IDN,k,j,i) = rho_0 * (1.0 + (density_contrast-1.0) * 0.5 * ( std::tanh((y-z_bot)/smoothing_thickness) - std::tanh((y-z_top)/smoothing_thickness) ) );
          phydro->w(IPR,k,j,i) = pgas_0;
          phydro->w(IVX,k,j,i) = velocity * ( 0.5 - 0.5 * ( std::tanh((y-z_bot)/smoothing_thickness) - std::tanh((y-z_top)/smoothing_thickness) ));
          phydro->w(IVY,k,j,i) = velocity_pert * (std::exp(-SQR((y-z_bot)/smoothing_thickness)) + std::exp(-SQR((y-z_top)/smoothing_thickness))) * std::sin(2*PI*x/lambda_pert);
          phydro->w(IVZ,k,j,i) = 0.0;
        } else {
          phydro->w(IDN,k,j,i) = rho_0 * (1.0 + (density_contrast-1.0) * 0.5 * ( std::tanh((x-z_bot)/smoothing_thickness) - std::tanh((x-z_top)/smoothing_thickness) ) );
          phydro->w(IPR,k,j,i) = pgas_0;
          phydro->w(IVX,k,j,i) = 0.0; 
          phydro->w(IVY,k,j,i) = velocity * ( 0.5 - 0.5 * ( std::tanh((x-z_bot)/smoothing_thickness) - std::tanh((x-z_top)/smoothing_thickness) ));
          phydro->w(IVZ,k,j,i) = 0.0;
        }
      }
    }
  }

  // initialize interface B, assuming vertical field only B=(0,0,1)
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke+1; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie+1; i++) {
          pfield->b.x1f(k,j,i) = (B_direction == 0) ?  sqrt(2*pgas_0/beta): 0.0;
          pfield->b.x2f(k,j,i) = (B_direction == 1) ?  sqrt(2*pgas_0/beta): 0.0;
          pfield->b.x3f(k,j,i) = (B_direction == 2) ?  sqrt(2*pgas_0/beta): 0.0; 
          phydro->u(IEN,k,j,i) += pgas_0/beta;
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
    for (int k=pmb->ks; k<=pmb->ke; ++k) {
      for (int j=pmb->js; j<=pmb->je; ++j) {
        for (int i=pmb->is; i<=pmb->ie; ++i) {
          Real dt;
          Real press = pmb->phydro->w(IPR,k,j,i);
          Real dens = pmb->phydro->w(IDN,k,j,i);
          Real edot = std::max(1e-4,fabs(edot_cool(press,dens)));
          dt = cfl_cool * 1.5*press/edot;
          if(Globals::my_rank==0) {
            std::cout << "dt " << dt << " edot " << edot << " press " << press << " dens " << dens << "\n";
          }
          dt = std::max( dt , dt_cutoff );
          min_dt = std::min(min_dt, dt);
        }
      }
    }
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

void Cooling_Source_Function(MeshBlock *pmb, const Real t, const Real dt,
    const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc, AthenaArray<Real> &cons, int stage)
{
  // Extract indices
  int is = pmb->is;
  int ie = pmb->ie;
  int js = pmb->js;
  int je = pmb->je;
  int ks = pmb->ks;
  int ke = pmb->ke;

  // Apply all source terms
  Real delta_e_block = 0.0;
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

        // Apply cooling and heating
        Real delta_e = -edot_cool(pgas_half, rho_half) * dt;
        Real kinetic = (SQR(m1) + SQR(m2) + SQR(m3)) / (2.0 * rho);
        Real u = e - kinetic;
if (MAGNETIC_FIELDS_ENABLED) {  
          const Real &bcc1 = bcc(IB1,k,j,i);
          const Real &bcc2 = bcc(IB2,k,j,i);
          const Real &bcc3 = bcc(IB3,k,j,i);
          Real magnetic = 0.5*(SQR(bcc1) + SQR(bcc2) + SQR(bcc3));
          u -= magnetic; 
        }
        delta_e = std::max(delta_e, -u);
        if (t > t_cool_start){
          e += delta_e;
        }
        if (stage == 2) {
          delta_e_block += delta_e;
        }
      }
    }
  }
  pmb->ruser_meshblock_data[0](0) += delta_e_block;
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
// calculated edot_cool 

static Real edot_cool(Real press, Real dens)
{
  Real Tmin = pgas_0/rho_0 / density_contrast;
  Real Tmax = pgas_0/rho_0;
  Real Tmix = sqrt(Tmin*Tmax);
  Real T = press/dens;
  Real M = std::log(Tmix) + SQR(s_Lambda);
  Real log_normal = std::exp(-SQR((std::log(T) - M)) /(2.*SQR(s_Lambda))) / (s_Lambda*T*sqrt(2.*PI)) ; 
  Real log_normal_min = std::exp(-SQR((std::log(Tmin) - M)) /(2.*SQR(s_Lambda))) / (s_Lambda*Tmin*sqrt(2.*PI)) ;
  return Lambda_cool * SQR(dens) * (log_normal-log_normal_min);
}


//----------------------------------------------------------------------------------------
//! \fn void ConstantShearInflowInnerX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
//                          FaceField &b, Real time, Real dt,
//                          int is, int ie, int js, int je, int ks, int ke, int ngh)
//  \brief ConstantShearInflow boundary conditions, inner x3 boundary

void ConstantShearInflowInnerX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                    FaceField &b, Real time, Real dt,
                    int is, int ie, int js, int je, int ks, int ke, int ngh) {
  // copy hydro variables into ghost zones
  for (int n=0; n<(NHYDRO); ++n) {
    for (int k=1; k<=ngh; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        Real z = pco->x3v(ks-k);
        prim(n,ks-k,j,i) = prim(n,ks,j,i);
        if ( n == IPR ){
          prim(IPR,ks-k,j,i) = pgas_0;
        } 
        if ( n == IDN ){
          prim(IDN,ks-k,j,i) = rho_0 * (1.0 + (density_contrast-1.0) * 0.5 * ( std::tanh((z-z_bot)/smoothing_thickness) - std::tanh((z-z_top)/smoothing_thickness) ) );
        } 
        if ( n == IVX ){
          prim(IVX,ks-k,j,i) = velocity * ( 0.5 - 0.5 * ( std::tanh((z-z_bot)/smoothing_thickness) - std::tanh((z-z_top)/smoothing_thickness) ));
        } 
      }
    }}
  }

  // copy face-centered magnetic fields into ghost zones
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=1; k<=ngh; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie+1; ++i) {
        b.x1f((ks-k),j,i) = b.x1f(ks,j,i);
      }
    }}

    for (int k=1; k<=ngh; ++k) {
    for (int j=js; j<=je+1; ++j) {
      for (int i=is; i<=ie; ++i) {
        b.x2f((ks-k),j,i) = b.x2f(ks,j,i);
      }
    }}

    for (int k=1; k<=ngh; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        b.x3f((ks-k),j,i) = b.x3f(ks,j,i);
      }
    }}
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ConstantShearInflowOuterX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
//                          FaceField &b, Real time, Real dt,
//                          int is, int ie, int js, int je, int ks, int ke, int ngh)
//  \brief ConstantShearInflow boundary conditions, outer x3 boundary

void ConstantShearInflowOuterX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                    FaceField &b, Real time, Real dt,
                    int is, int ie, int js, int je, int ks, int ke, int ngh) {
  // copy hydro variables into ghost zones
  for (int n=0; n<(NHYDRO); ++n) {
    for (int k=1; k<=ngh; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        Real z = pco->x3v(ke+k);
        prim(n,ke+k,j,i) = prim(n,ke,j,i);
        if ( n == IPR ){
          prim(IPR,ke+k,j,i) = pgas_0;
        } 
        if ( n == IDN ){
          prim(IDN,ke+k,j,i) = rho_0 * (1.0 + (density_contrast-1.0) * 0.5 * ( std::tanh((z-z_bot)/smoothing_thickness) - std::tanh((z-z_top)/smoothing_thickness) ) );
        } 
        if ( n == IVX ){
          prim(IVX,ke+k,j,i) = velocity * ( 0.5 - 0.5 * ( std::tanh((z-z_bot)/smoothing_thickness) - std::tanh((z-z_top)/smoothing_thickness) ));
        } 
      }
    }}
  }

  // copy face-centered magnetic fields into ghost zones
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=1; k<=ngh; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie+1; ++i) {
        b.x1f((ke+k  ),j,i) = b.x1f((ke  ),j,i);
      }
    }}

    for (int k=1; k<=ngh; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        b.x2f((ke+k  ),j,i) = b.x2f((ke  ),j,i);
      }
    }}

    for (int k=1; k<=ngh; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        b.x3f((ke+k+1),j,i) = b.x3f((ke+1),j,i);
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

        phdif->nu(ISO,k,j,i) = phdif->nu_iso * S_norm;
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

        phdif->kappa(ISO,k,j,i) = phdif->kappa_iso * S_norm;
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

        phdif->nu(ISO,k,j,i) = phdif->nu_iso * S_norm;
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

        phdif->kappa(ISO,k,j,i) = phdif->kappa_iso * S_norm;
      }
    }
  }
  return;
}
