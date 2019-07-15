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

static Real Yk[12];




static void init_cooling();
static Real newtemp_townsend(const Real d, const Real P, const Real dt_hydro);
static Real Lambda_T(const Real T);
static Real Yinv(Real Y1);
static Real Y(const Real T);
static Real tcool(const Real d, const Real P);

void Cooling(MeshBlock *pmb, const Real t, const Real dt,
    const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc, 
    AthenaArray<Real> &cons);

static Real cooling_timestep(MeshBlock *pmb);
static Real cfl_cool;

//-----------------------------------------------------------------------------
// Function for preparing Mesh
// Inputs:
//   pin: input parameters (unused)
// Outputs: (none)

void Mesh::InitUserMeshData(ParameterInput *pin)
{
  // Read general parameters from input file
  gamma_adi               = pin->GetReal("hydro", "gamma");
  rho_scale               = pin->GetReal("problem", "rho_scale");
  pgas_scale              = pin->GetReal("problem", "pgas_scale");
  length_scale            = pin->GetReal("problem", "length_scale");
  vel_scale               = std::sqrt(pgas_scale / rho_scale);
  time_scale              = length_scale/std::sqrt(pgas_scale / rho_scale);
  rho_0                   = pin->GetReal("problem", "rho_0");
  pgas_0                  = pin->GetReal("problem", "pgas_0");
  // limit the hydro timestep to be some fraction of the cooling timestep
  cfl_cool = pin->GetOrAddReal("problem", "cfl_cool", 0.1);

  // set up the arrays for cooling
  init_cooling();


  // Enroll user-defined time step constraint
  EnrollUserTimeStepFunction(cooling_timestep);
  // Enroll user-defined source function
  EnrollUserExplicitSourceFunction(Cooling);

}

//-----------------------------------------------------------------------------
// Function for preparing MeshBlock
// Inputs:
//   pin: input parameters (unused)
// Outputs: (none)

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin)
{
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

  // Initialize primitive values
  for (int k = kl; k <= ku; ++k) {
    Real z = pcoord->x3v(k);
    for (int j = jl; j <= ju; ++j) {
      for (int i = il; i <= iu; ++i) {
        phydro->w(IDN,k,j,i) = rho_0;
        phydro->w(IPR,k,j,i) = pgas_0; 
        phydro->w(IVX,k,j,i) = 0.0;
        phydro->w(IVY,k,j,i) = 0.0;
        phydro->w(IVZ,k,j,i) = 0.0;
      }
    }
  }

  // Initialize conserved values
  AthenaArray<Real> b;
  peos->PrimitiveToConserved(phydro->w, b, phydro->u, pcoord, 
                              il, iu, jl, ju, kl, ku);
  return;
}



//----------------------------------------------------------------------------------------
// Function to calculate the timestep required to resolve cooling (and later conduction)
//          tcool = 3/2 P/Edot_cool
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
          if (temperature(k,j,i)>1.0e4){
            Real dt = tcool(pmb->phydro->w(IDN,k,j,i), pmb->phydro->w(IPR,k,j,i));
            min_dt = std::min(min_dt, dt);
          }
        }
      }
    }
    return min_dt;
  }
}
//----------------------------------------------------------------------------------------
// Source function for cooling 
// Inputs:
//   pmb: pointer to MeshBlock
//   t,dt: time (not used) and timestep
//   prim: primitives
//   bcc: cell-centered magnetic fields (not used)
// Outputs:
//   cons: conserved variables updated

void Cooling(MeshBlock *pmb, const Real t, const Real dt,
    const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc, 
    AthenaArray<Real> &cons)
{
  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        // Extract primitive and conserved quantities
        const Real &rho_half = prim(IDN,k,j,i);
        const Real &pgas_half = prim(IPR,k,j,i);
        const Real &rho = cons(IDN,k,j,i);
        Real &e = cons(IEN,k,j,i);
        Real &m1 = cons(IM1,k,j,i);
        Real &m2 = cons(IM2,k,j,i);
        Real &m3 = cons(IM3,k,j,i);

        Real kinetic = (SQR(m1) + SQR(m2) + SQR(m3)) / (2.0 * rho);
        Real u = e - kinetic;
        Real P = u * (gamma_adi-1.0);

        // calculate temperature in physical units before cooling
        Real T_before = mu * mp * (P/rho)*(pgas_scale/rho_scale) / kb;
        // calculate temperature in physical units after cooling
        Real T_after = newtemp_townsend(rho, P, dt);
        // calculate thermal energy in code units after cooling
        Real e_after = ((rho*rho_scale/(mu*mp))*kb*T_after / 
          (gamma_adi-1.0))/pgas_scale;
        // Apply cooling and heating
        Real delta_e = e_after-u; 

        e += delta_e;  

      }
    }
  }
  return;
}



// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// cooling routines 

// ============================================================================
static Real newtemp_townsend(const Real d, const Real P, const Real dt_hydro)
{
  Real term1, Tref;
  int n=nfit_cool-1;

  Tref = T_cooling_curve[n];

  Real T = mu * mp * (P/d) * (pgas_scale/rho_scale) / kb;

  term1 = (T/Tref) * (Lambda_T(Tref)/Lambda_T(T)) * 
                (dt_hydro*time_scale/tcool(d, P));

  return Yinv(Y(T) + term1);
}
// ============================================================================



// ============================================================================
static Real tcool(const Real d, const Real P)
{
  Real T = mu * mp * (P/d) * (pgas_scale/rho_scale) / kb;
  Real number_density = d * rho_scale / (mu * mp);
  return  (gamma_adi / (gamma_adi-1.0)) * (kb*T) * SQR(muH/mu) / 
  (number_density *Lambda_T(T)); 
}
// ============================================================================


// ============================================================================
static void init_cooling()
{
  int k, n=nfit_cool-1;
  Real term;

  // populate Yk following equation A6 in Townsend (2009) 
  Yk[n] = 0.0;
  for (k=n-1; k>=0; k--){
    term = (lambda_cooling_curve[n]/lambda_cooling_curve[k]) * 
              (T_cooling_curve[k]/T_cooling_curve[n]);

    if (exponent_cooling_curve[k] == 1.0)
      term *= log(T_cooling_curve[k]/T_cooling_curve[k+1]);
    else
      term *= ((1.0 - pow(T_cooling_curve[k]/T_cooling_curve[k+1], 
                            exponent_cooling_curve[k]-1.0)) / 
                (1.0-exponent_cooling_curve[k]));

    Yk[k] = Yk[k+1] - term;
  }
  return;
}

// ============================================================================
// piecewise power-law fit to the cooling curve with
// temperature in keV and L in erg cm^3 / s 
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
    return 0.0;
  }
}
// ============================================================================


// ============================================================================
// see Lambda_T() or equation A1 of Townsend (2009) for the definition 
static Real Y(const Real T)
{
  int k, n=nfit_cool-1;
  Real term;

  // first find the temperature bin 
  for(k=n; k>=0; k--){
    if (T >= T_cooling_curve[k])
      break;
  }

  // calculate Y using equation A5 in Townsend (2009) 
  term = (lambda_cooling_curve[n]/lambda_cooling_curve[k]) * 
          (T_cooling_curve[k]/T_cooling_curve[n]);

  if (exponent_cooling_curve[k] == 1.0)
    term *= log(T_cooling_curve[k]/T);
  else
    term *= ((1.0 - pow(T_cooling_curve[k]/T, exponent_cooling_curve[k]-1.0)) / 
              (1.0-exponent_cooling_curve[k]));

  return (Yk[k] + term);
}
// ============================================================================


// ============================================================================
static Real Yinv(const Real Y1)
{
  int k, n=nfit_cool-1;
  Real term;

  // find the bin i in which the final temperature will be 
  for(k=n; k>=0; k--){
    if (Y(T_cooling_curve[k]) >= Y1)
      break;
  }
  
  // calculate Yinv using equation A7 in Townsend (2009) 
  term = (lambda_cooling_curve[k]/lambda_cooling_curve[n]) * 
          (T_cooling_curve[n]/T_cooling_curve[k]);
  term *= (Y1 - Yk[k]);

  if (exponent_cooling_curve[k] == 1.0)
    term = exp(-1.0*term);
  else{
    term = pow(1.0 - (1.0-exponent_cooling_curve[k])*term,
               1.0/(1.0-exponent_cooling_curve[k]));
  }

  return (T_cooling_curve[k] * term);
}
// ============================================================================
