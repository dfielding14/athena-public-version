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
Real history_recorder(MeshBlock *pmb, int iout);
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

void SpitzerConduction(HydroDiffusion *phdif, MeshBlock *pmb, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &bcc, int is, int ie, int js, int je, int ks, int ke);
void SpitzerViscosity(HydroDiffusion *phdif, MeshBlock *pmb, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &bcc, int is, int ie, int js, int je, int ks, int ke);


// Global variables
static Real gamma_adi;
static Real rho_0, pgas_0;
static Real density_contrast;
static Real Lambda_cool, s_Lambda, t_cool_start, Lambda_hot;
static Real Tmin,Tmax,Tmix,Tlow,Thigh,M;
static Real T_cond_max;
static Real dtdrive;

static Real cooling_timestep(MeshBlock *pmb);
static Real dt_cutoff, cfl_cool;

static int nstages;
static Real weights[4];
static bool adaptive_driving;
static bool Lambda_ramp_down;
static Real dt_Lambda_ramp_down, delta_Lambda_ramp_down, Lambda_cool_init ;

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
  Real vel_scale         = std::sqrt(pgas_scale / rho_scale);
  rho_0                  = pin->GetReal("problem", "rho_0");
  pgas_0                 = pin->GetReal("problem", "pgas_0");
  density_contrast       = pin->GetReal("problem", "density_contrast");
  T_cond_max             = pin->GetOrAddReal("problem", "T_cond_max", 1.0); // the value of P/rho where conduction saturates
  dtdrive                = pin->GetReal("problem", "dtdrive");
  adaptive_driving       = pin->GetBoolean("problem", "adaptive_driving");

  // Read cooling-table-related parameters from input file
  t_cool_start = pin->GetReal("problem", "t_cool_start");
  dt_cutoff    = pin->GetOrAddReal("problem", "dt_cutoff", 3.0e-5);
  cfl_cool     = pin->GetOrAddReal("problem", "cfl_cool", 0.1);
  Lambda_cool  = pin->GetReal("problem", "Lambda_cool");
  Lambda_hot   = pin->GetOrAddReal("problem", "Lambda_hot",1.0);
  s_Lambda     = pin->GetReal("problem", "s_Lambda");

  Lambda_cool_init = Lambda_cool;
  Lambda_ramp_down       = pin->GetOrAddBoolean("problem", "Lambda_ramp_down", false);
  dt_Lambda_ramp_down    = pin->GetOrAddReal("problem", "dt_Lambda_ramp_down",1.0);
  delta_Lambda_ramp_down = pin->GetOrAddReal("problem", "delta_Lambda_ramp_down",0.1);

  Tmin = pgas_0/rho_0 / density_contrast;
  Tmax = pgas_0/rho_0;
  Tmix = sqrt(Tmin*Tmax);
  Tlow = sqrt(Tmin*Tmix);
  Thigh = sqrt(Tmix*Tmax);
  M = std::log(Tmix) + SQR(s_Lambda);

  // Get the number of stages based on the integrator
  std::string integrator_name = pin->GetString("time", "integrator");

  std::string vl2 ("vl2");
  std::string rk2 ("rk2");
  std::string rk3 ("rk3");
  std::string rk4 ("rk4");
 
  if ((integrator_name.compare(rk2) == 0 )||(integrator_name.compare(vl2) == 0 )){
    nstages = 2;
    if (integrator_name.compare(rk2) == 0 ){
      weights[0]=0.5;
      weights[1]=1.0;
      weights[2]=0.0;
      weights[3]=0.0;
    } else {
      weights[0]=0.0;
      weights[1]=1.0;
      weights[2]=0.0;
      weights[3]=0.0;
    }
  }
  if (integrator_name.compare(rk3) == 0 ){
    nstages = 3;
    weights[0]=1./6.;
    weights[1]=2./3.;
    weights[2]=1.0;
    weights[3]=0.0;
  }

  if(Globals::my_rank==0) {
    std::cout << "nstages = " << nstages << "\n";
    std::cout << "Tmin = " << Tmin << "\n";
    std::cout << "Tmax = " << Tmax << "\n";
    std::cout << "Tmix = " << Tmix << "\n";
    std::cout << "Tlow = " << Tlow << "\n";
    std::cout << "Thigh = " << Thigh << "\n";
  }


  // Enroll user-defined functions
  EnrollUserExplicitSourceFunction(Cooling_Source_Function);
  AllocateUserHistoryOutput(20);
  EnrollUserHistoryOutput(0, history_recorder, "e_cool");
  EnrollUserHistoryOutput(1, history_recorder, "e_ceil");
  EnrollUserHistoryOutput(2, history_recorder, "M_h");
  EnrollUserHistoryOutput(3, history_recorder, "M_i");
  EnrollUserHistoryOutput(4, history_recorder, "M_c");
  EnrollUserHistoryOutput(5, history_recorder, "Px_h");
  EnrollUserHistoryOutput(6, history_recorder, "Px_i");
  EnrollUserHistoryOutput(7, history_recorder, "Px_c");
  EnrollUserHistoryOutput(8, history_recorder, "Py_h");
  EnrollUserHistoryOutput(9, history_recorder, "Py_i");
  EnrollUserHistoryOutput(10, history_recorder, "Py_c");
  EnrollUserHistoryOutput(11, history_recorder, "Pz_h");
  EnrollUserHistoryOutput(12, history_recorder, "Pz_i");
  EnrollUserHistoryOutput(13, history_recorder, "Pz_c");
  EnrollUserHistoryOutput(14, history_recorder, "Ek_h");
  EnrollUserHistoryOutput(15, history_recorder, "Ek_i");
  EnrollUserHistoryOutput(16, history_recorder, "Ek_c");
  EnrollUserHistoryOutput(17, history_recorder, "Eth_h");
  EnrollUserHistoryOutput(18, history_recorder, "Eth_i");
  EnrollUserHistoryOutput(19, history_recorder, "Eth_c");

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

  bool SpitzerViscosity_on = pin->GetOrAddBoolean("problem", "SpitzerViscosity_on", false);
  bool SpitzerConduction_on = pin->GetOrAddBoolean("problem", "SpitzerConduction_on", false);

  if (SpitzerViscosity_on){
    EnrollViscosityCoefficient(SpitzerViscosity);
  }
  if (SpitzerConduction_on){
    EnrollConductionCoefficient(SpitzerConduction);
  }

  AllocateRealUserMeshDataField(1);
  ruser_mesh_data[0].NewAthenaArray(1);
  ruser_mesh_data[0](0) = 0.0;
  return;
}

//----------------------------------------------------------------------------------------
// Function for preparing MeshBlock
// Inputs:
//   pin: input parameters (unused)
// Outputs: (none)

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin)
{
  // Allocate storage for keeping track of cooling and fluxes
  AllocateRealUserMeshBlockDataField(1);
  ruser_meshblock_data[0].NewAthenaArray(20);
  for (int i = 0; i < 20; ++i) {
    ruser_meshblock_data[0](i) = 0.0;
  }
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
  Real temperature_scale = pin->GetReal("problem", "temperature_scale");

  // Initialize primitive values
  for (int k = kl; k <= ku; ++k) {
    for (int j = jl; j <= ju; ++j) {
      for (int i = il; i <= iu; ++i) {
        phydro->w(IDN,k,j,i) = rho_0;
        phydro->w(IPR,k,j,i) = temperature_scale*pgas_0;
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
          Real edot = fabs(edot_cool(press,dens));
          dt = cfl_cool * 1.5*press/edot;
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
  // Readjust cooling if necessary
  if ( Lambda_ramp_down ){
    if (( fmod(t,dt_Lambda_ramp_down) < dt ) && (t > t_cool_start)){
      Lambda_cool = Lambda_cool_init - floor((t-t_cool_start)/dt_Lambda_ramp_down) * delta_Lambda_ramp_down
      Lambda_cool = Lambda_cool<0.0 ? 0.0 : Lambda_cool;
      if(Globals::my_rank==0) {
        std::cout << "decreased cooling, Lambda_cool = " << Lambda_cool << "\n";
      }
    }
  }

  // Extract indices
  int is = pmb->is;
  int ie = pmb->ie;
  int js = pmb->js;
  int je = pmb->je;
  int ks = pmb->ks;
  int ke = pmb->ke;

  // history file 
  Real e_cool = 0.0;
  Real M_h=0.0, M_i=0.0, M_c=0.0;
  Real Px_h=0.0, Px_i=0.0, Px_c=0.0;
  Real Py_h=0.0, Py_i=0.0, Py_c=0.0;
  Real Pz_h=0.0, Pz_i=0.0, Pz_c=0.0;
  Real Ek_h=0.0, Ek_i=0.0, Ek_c=0.0;
  Real Eth_h=0.0, Eth_i=0.0, Eth_c=0.0;

  // calculate average T
  // Real m[3] = {0}, gm[3];
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

        // M_h M_i M_c Px_h Px_i Px_c Py_h Py_i Py_c Pz_h Pz_i Pz_c Ek_h Ek_i Ek_c Eth_h Eth_i Eth_c
        // I am not exactly sure which variables at which point in the stage i should be using for the calculation of T
        Real T = (2./3.) * (u+delta_e) / rho;
        // m[0] += delta_e*weights[stage-1]
        // if (stage == nstages){
        //   m[1] += T;
        //   m[2] += 1.0;
        // }
        Real area_cell = pmb->pcoord->dx1f(i)*pmb->pcoord->dx2f(j);
        Real vol_cell = area_cell*pmb->pcoord->dx3f(k);
        if (T > Thigh){
          M_h += rho * vol_cell;
          Px_h += m1 * vol_cell;
          Py_h += m2 * vol_cell;
          Pz_h += m3 * vol_cell;
          Ek_h += kinetic * vol_cell;
          Eth_h += (u+delta_e) * vol_cell;
        } else if (T<Tlow){
          M_c += rho * vol_cell;
          Px_c += m1 * vol_cell;
          Py_c += m2 * vol_cell;
          Pz_c += m3 * vol_cell;
          Ek_c += kinetic * vol_cell;
          Eth_c += (u+delta_e) * vol_cell;
        } else {
          M_i += rho * vol_cell;
          Px_i += m1 * vol_cell;
          Py_i += m2 * vol_cell;
          Pz_i += m3 * vol_cell;
          Ek_i += kinetic * vol_cell;
          Eth_i += (u+delta_e) * vol_cell;
        }
        e_cool += delta_e*vol_cell;
      }
    }
  }

  pmb->ruser_meshblock_data[0](0) += e_cool*weights[stage-1];
  if (stage == nstages){
    pmb->ruser_meshblock_data[0](2) += M_h;
    pmb->ruser_meshblock_data[0](3) += M_i;
    pmb->ruser_meshblock_data[0](4) += M_c;
    pmb->ruser_meshblock_data[0](5) += Px_h;
    pmb->ruser_meshblock_data[0](6) += Px_i;
    pmb->ruser_meshblock_data[0](7) += Px_c;
    pmb->ruser_meshblock_data[0](8) += Py_h;
    pmb->ruser_meshblock_data[0](9) += Py_i;
    pmb->ruser_meshblock_data[0](10) += Py_c;
    pmb->ruser_meshblock_data[0](11) += Pz_h;
    pmb->ruser_meshblock_data[0](12) += Pz_i;
    pmb->ruser_meshblock_data[0](13) += Pz_c;
    pmb->ruser_meshblock_data[0](14) += Ek_h;
    pmb->ruser_meshblock_data[0](15) += Ek_i;
    pmb->ruser_meshblock_data[0](16) += Ek_c;
    pmb->ruser_meshblock_data[0](17) += Eth_h;
    pmb->ruser_meshblock_data[0](18) += Eth_i;
    pmb->ruser_meshblock_data[0](19) += Eth_c;
  }

  // If there is adaptive driving I need to add up and store the total amount
  // of energy that has been cooled away. If the driving is continuous then
  // there is no need to store since dedt can be set to match it exactly, but 
  // if there is impulsive turbulence driving then I need to save up all the E
  // to do this I am going to use ruser_mesh_data
  // I wonder if I even need to do an MPI call. 
  Real my_edotcool[1] = {0}, edotcool_tot[1];

  my_edotcool[0] += -e_cool/dt*weights[stage-1];
#ifdef MPI_PARALLEL
    MPI_Allreduce(my_edotcool, edotcool_tot, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#else
    edotcool_tot[0] = my_edotcool[0];
#endif
  // if(Globals::my_rank==0) {
  //   std::cout << "edotcool_tot = " << edotcool_tot[0] << " dt = " << dt << "\n";
  // }
  pmb->pmy_mesh->ruser_mesh_data[0](0) += edotcool_tot[0]*dt;
  Real &dedt = pmb->pmy_mesh->ptrbd->dedt;
  if ((stage == nstages)&&(adaptive_driving)){
    if ((pmb->pmy_mesh->turb_flag == 3)&&(t > t_cool_start)){
      dedt = edotcool_tot[0] < 0 ? 0.0 : edotcool_tot[0] ; 
    } else if ((pmb->pmy_mesh->turb_flag == 2)&&(t > t_cool_start)&&(t >= pmb->pmy_mesh->ptrbd->tdrive)){
      dedt = pmb->pmy_mesh->ruser_mesh_data[0](0)/dtdrive < 0 ? 0.0:pmb->pmy_mesh->ruser_mesh_data[0](0)/dtdrive;
      pmb->pmy_mesh->ruser_mesh_data[0](0) = 0.0; // is this going to mess things up for other processors or blocks??
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

Real history_recorder(MeshBlock *pmb, int iout)
{
  Real delta_e = pmb->ruser_meshblock_data[0](iout);
  pmb->ruser_meshblock_data[0](iout) = 0.0;
  return delta_e;
}


//----------------------------------------------------------------------------------------
// calculated edot_cool 
// static Real edot_cool(Real press, Real dens)
// {
//   Real T = press/dens;
//   Real log_normal = std::exp(-SQR((std::log(T) - M)) /(2.*SQR(s_Lambda))) / (s_Lambda*T*sqrt(2.*PI)) ; 
//   Real log_normal_min = std::exp(-SQR((std::log(Tmin) - M)) /(2.*SQR(s_Lambda))) / (s_Lambda*Tmin*sqrt(2.*PI)) ;
//   return Lambda_cool * SQR(dens) * std::max(log_normal-log_normal_min,0.0);
// }

//----------------------------------------------------------------------------------------
// calculated edot_cool 
static Real edot_cool(Real press, Real dens)
{
  Real T = press/dens;
  if (T>1.0){
    return Lambda_hot *SQR(dens)*(sqrt(T)-1.0);
  } else {
    Real log_normal = std::exp(-SQR((std::log(T) - M)) /(2.*SQR(s_Lambda))) / (s_Lambda*T*sqrt(2.*PI)); 
    Real log_normal_min = std::exp(-SQR((std::log(Tmin) - M)) /(2.*SQR(s_Lambda))) / (s_Lambda*Tmin*sqrt(2.*PI));
    return Lambda_cool * SQR(dens) * (log_normal-log_normal_min);
  }
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


// ----------------------------------------------------------------------------------------
// SpitzerViscosity 
// 
void SpitzerViscosity(HydroDiffusion *phdif, MeshBlock *pmb, const AthenaArray<Real> &prim,
     const AthenaArray<Real> &bcc, int is, int ie, int js, int je, int ks, int ke) 
{
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is+1; i<=ie; ++i) {
        Real T = prim(IPR,k,j,i)/prim(IDN,k,j,i);
        phdif->nu(ISO,k,j,i) = phdif->nu_iso/prim(IDN,k,j,i) * std::max(1.0 , pow( T/T_cond_max ,2.5));
      }
    }
  }
  return;
}

// ----------------------------------------------------------------------------------------
// SpitzerConduction 
// 
void SpitzerConduction(HydroDiffusion *phdif, MeshBlock *pmb, const AthenaArray<Real> &prim,
     const AthenaArray<Real> &bcc, int is, int ie, int js, int je, int ks, int ke) 
{
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is+1; i<=ie; ++i) {
        Real T = prim(IPR,k,j,i)/prim(IDN,k,j,i);
        phdif->kappa(ISO,k,j,i) = phdif->kappa_iso/prim(IDN,k,j,i) * std::max(1.0 , pow( T/T_cond_max ,2.5));
      }
    }
  }
  return;
}
