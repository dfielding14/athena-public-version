//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file adiabatic_hydro.cpp
//  \brief implements functions in class EquationOfState for adiabatic hydrodynamics`

// C/C++ headers
#include <cmath>   // sqrt()
#include <cfloat>  // FLT_MIN

// Athena++ headers
#include "eos.hpp"
#include "../hydro/hydro.hpp"
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../field/field.hpp"

// EquationOfState constructor

EquationOfState::EquationOfState(MeshBlock *pmb, ParameterInput *pin) {
  pmy_block_ = pmb;
  gamma_ = pin->GetReal("hydro", "gamma");
  density_floor_  = pin->GetOrAddReal("hydro","dfloor", std::sqrt(1024*(FLT_MIN)));
  pressure_floor_ = pin->GetOrAddReal("hydro","pfloor", std::sqrt(1024*(FLT_MIN)));
  velocity_ceiling_ = pin->GetOrAddReal("hydro","vceil",0.0);

  neighbor_flooring_ = pin->GetOrAddBoolean("problem","neighbor_flooring_", false); // average the neighboring cells to get floor value rather than reset to the floor value
}

// destructor

EquationOfState::~EquationOfState() {
}

//----------------------------------------------------------------------------------------
// \!fn void EquationOfState::ConservedToPrimitive(AthenaArray<Real> &cons,
//           const AthenaArray<Real> &prim_old, const FaceField &b,
//           AthenaArray<Real> &prim, AthenaArray<Real> &bcc, Coordinates *pco,
//           int il, int iu, int jl, int ju, int kl, int ku)
// \brief Converts conserved into primitive variables in adiabatic hydro.

void EquationOfState::ConservedToPrimitive(AthenaArray<Real> &cons,
  const AthenaArray<Real> &prim_old, const FaceField &b, AthenaArray<Real> &prim,
  AthenaArray<Real> &bcc, Coordinates *pco, int il,int iu, int jl,int ju, int kl,int ku) {
  Real gm1 = GetGamma() - 1.0;

  for (int k=kl; k<=ku; ++k) {
  for (int j=jl; j<=ju; ++j) {
#pragma omp simd
    for (int i=il; i<=iu; ++i) {
      Real& u_d  = cons(IDN,k,j,i);
      Real& u_m1 = cons(IM1,k,j,i);
      Real& u_m2 = cons(IM2,k,j,i);
      Real& u_m3 = cons(IM3,k,j,i);
      Real& u_e  = cons(IEN,k,j,i);

      Real& w_d  = prim(IDN,k,j,i);
      Real& w_vx = prim(IVX,k,j,i);
      Real& w_vy = prim(IVY,k,j,i);
      Real& w_vz = prim(IVZ,k,j,i);
      Real& w_p  = prim(IPR,k,j,i);

      // apply density floor, without changing momentum or energy
      if (neighbor_flooring_ == false) {
        u_d = (u_d > density_floor_) ?  u_d : density_floor_;
      } else {
        if (u_d < density_floor_){
          Real n_neighbors = 0.0; 
          Real d_neighbors = 0.0; 
          if ((k+1 <= ku+NGHOST)&&(cons(IDN,k+1,j,i)>density_floor_)){
            n_neighbors += 1.0;
            d_neighbors += cons(IDN,k+1,j,i);
          }
          if ((j+1 <= ju+NGHOST)&&(cons(IDN,k,j+1,i)>density_floor_)){
            n_neighbors += 1.0;
            d_neighbors += cons(IDN,k,j+1,i);
          }
          if ((i+1 <= iu+NGHOST)&&(cons(IDN,k,j,i+1)>density_floor_)){
            n_neighbors += 1.0;
            d_neighbors += cons(IDN,k,j,i+1);
          }
          if ((k-1 >= kl-NGHOST)&&(cons(IDN,k-1,j,i)>density_floor_)){
            n_neighbors += 1.0;
            d_neighbors += cons(IDN,k-1,j,i);
          }
          if ((j-1 >= jl-NGHOST)&&(cons(IDN,k,j-1,i)>density_floor_)){
            n_neighbors += 1.0;
            d_neighbors += cons(IDN,k,j-1,i);
          }
          if ((i-1 >= il-NGHOST)&&(cons(IDN,k,j,i-1)>density_floor_)){
            n_neighbors += 1.0;
            d_neighbors += cons(IDN,k,j,i-1);
          }
          u_d = (d_neighbors/n_neighbors > density_floor_) ? d_neighbors/n_neighbors : density_floor_;
        }
      }

      w_d = u_d;

      Real di = 1.0/u_d;
      w_vx = u_m1*di;
      w_vy = u_m2*di;
      w_vz = u_m3*di;

      Real ke = 0.5*di*(SQR(u_m1) + SQR(u_m2) + SQR(u_m3));
      w_p = gm1*(u_e - ke);

      // apply pressure floor, correct total energy
      if (neighbor_flooring_ == false) {
        u_e = (w_p > pressure_floor_) ?  u_e : ((pressure_floor_/gm1) + ke);
        w_p = (w_p > pressure_floor_) ?  w_p : pressure_floor_;
      } else {
        if (w_p < pressure_floor_) {
          Real n_neighbors = 0.0; 
          Real p_neighbors = 0.0; 
          if ((k+1 <= ku+NGHOST)&&(prim(IPR,k+1,j,i)>pressure_floor_)){
            n_neighbors += 1.0;
            p_neighbors += prim(IPR,k+1,j,i);
          }
          if ((j+1 <= ju+NGHOST)&&(prim(IPR,k,j+1,i)>pressure_floor_)){
            n_neighbors += 1.0;
            p_neighbors += prim(IPR,k,j+1,i);
          }
          if ((i+1 <= iu+NGHOST)&&(prim(IPR,k,j,i+1)>pressure_floor_)){
            n_neighbors += 1.0;
            p_neighbors += prim(IPR,k,j,i+1);
          }
          if ((k-1 >= kl-NGHOST)&&(prim(IPR,k-1,j,i)>pressure_floor_)){
            n_neighbors += 1.0;
            p_neighbors += prim(IPR,k-1,j,i);
          }
          if ((j-1 >= jl-NGHOST)&&(prim(IPR,k,j-1,i)>pressure_floor_)){
            n_neighbors += 1.0;
            p_neighbors += prim(IPR,k,j-1,i);
          }
          if ((i-1 >= il-NGHOST)&&(prim(IPR,k,j,i-1)>pressure_floor_)){
            n_neighbors += 1.0;
            p_neighbors += prim(IPR,k,j,i-1);
          }
          u_e = (p_neighbors/n_neighbors > pressure_floor_) ?  u_e : ((pressure_floor_/gm1) + ke);
          w_p = (p_neighbors/n_neighbors > pressure_floor_) ?  p_neighbors/n_neighbors : pressure_floor_;
        }
      }

      // apply velocity magnitude ceiling, correct total energy
      if (velocity_ceiling_ > 0.0) {
        if (neighbor_flooring_ == false) {
          if (std::abs(w_vx) > velocity_ceiling_){
            u_e -= 0.5 * di * SQR(u_m1);
            w_vx = w_vx/std::abs(w_vx) * velocity_ceiling_ ;
            u_m1 = w_vx/std::abs(w_vx) * velocity_ceiling_ * u_d;
            u_e += 0.5 * di * SQR(u_m1);
          }
          if (std::abs(w_vy) > velocity_ceiling_){
            u_e -= 0.5 * di * SQR(u_m2);
            w_vy = w_vy/std::abs(w_vy) * velocity_ceiling_ ;
            u_m2 = w_vy/std::abs(w_vy) * velocity_ceiling_ * u_d;
            u_e += 0.5 * di * SQR(u_m2);
          }
          if (std::abs(w_vz) > velocity_ceiling_){
            u_e -= 0.5 * di * SQR(u_m3);
            w_vz = w_vz/std::abs(w_vz) * velocity_ceiling_ ;
            u_m3 = w_vz/std::abs(w_vz) * velocity_ceiling_ * u_d;
            u_e += 0.5 * di * SQR(u_m3);
          }
        } else {
          if ((std::abs(w_vx) > velocity_ceiling_)||(std::abs(w_vy) > velocity_ceiling_)||(std::abs(w_vz) > velocity_ceiling_)){
            std::cout << " v > velocity_ceiling_ " << " u_d = " << u_d << " u_e = " << u_e <<  " w_vx = " << w_vx << " w_vy = " << w_vy <<  " w_vz = " << w_vz <<   " i j k = " << i << " " << j << " " << k << " " << " x y z = " << pco->x1v(i) << " " << pco->x2v(j) << " " << pco->x3v(k) << " " << "\n";
            Real n_neighbors = 0.0;
            Real m1_neighbors = 0.0;
            Real m2_neighbors = 0.0;
            Real m3_neighbors = 0.0;
            if ((k+1 <= ku+NGHOST)&&(std::abs(prim(IVX,k+1,j,i))<velocity_ceiling_)&&(std::abs(prim(IVY,k+1,j,i))<velocity_ceiling_)&&(std::abs(prim(IVZ,k+1,j,i))<velocity_ceiling_)){
              n_neighbors += 1.0;
              m1_neighbors += cons(IM1,k+1,j,i);
              m2_neighbors += cons(IM2,k+1,j,i);
              m3_neighbors += cons(IM3,k+1,j,i);
            }
            if ((j+1 <= ju+NGHOST)&&(std::abs(prim(IVX,k,j+1,i))<velocity_ceiling_)&&(std::abs(prim(IVY,k,j+1,i))<velocity_ceiling_)&&(std::abs(prim(IVZ,k,j+1,i))<velocity_ceiling_)){
              n_neighbors += 1.0;
              m1_neighbors += cons(IM1,k,j+1,i);
              m2_neighbors += cons(IM2,k,j+1,i);
              m3_neighbors += cons(IM3,k,j+1,i);
            }
            if ((i+1 <= iu+NGHOST)&&(std::abs(prim(IVX,k,j,i+1))<velocity_ceiling_)&&(std::abs(prim(IVY,k,j,i+1))<velocity_ceiling_)&&(std::abs(prim(IVZ,k,j,i+1))<velocity_ceiling_)){
              n_neighbors += 1.0;
              m1_neighbors += cons(IM1,k,j,i+1);
              m2_neighbors += cons(IM2,k,j,i+1);
              m3_neighbors += cons(IM3,k,j,i+1);
            }
            if ((k-1 >= kl-NGHOST)&&(std::abs(prim(IVX,k-1,j,i))<velocity_ceiling_)&&(std::abs(prim(IVY,k-1,j,i))<velocity_ceiling_)&&(std::abs(prim(IVZ,k-1,j,i))<velocity_ceiling_)){
              n_neighbors += 1.0;
              m1_neighbors += cons(IM1,k-1,j,i);
              m2_neighbors += cons(IM2,k-1,j,i);
              m3_neighbors += cons(IM3,k-1,j,i);
            }
            if ((j-1 >= jl-NGHOST)&&(std::abs(prim(IVX,k,j-1,i))<velocity_ceiling_)&&(std::abs(prim(IVY,k,j-1,i))<velocity_ceiling_)&&(std::abs(prim(IVZ,k,j-1,i))<velocity_ceiling_)){
              n_neighbors += 1.0;
              m1_neighbors += cons(IM1,k,j-1,i);
              m2_neighbors += cons(IM2,k,j-1,i);
              m3_neighbors += cons(IM3,k,j-1,i);
            }
            if ((i-1 >= il-NGHOST)&&(std::abs(prim(IVX,k,j,i-1))<velocity_ceiling_)&&(std::abs(prim(IVY,k,j,i-1))<velocity_ceiling_)&&(std::abs(prim(IVZ,k,j,i-1))<velocity_ceiling_)){
              n_neighbors += 1.0;
              m1_neighbors += cons(IM1,k,j,i-1);
              m2_neighbors += cons(IM2,k,j,i-1);
              m3_neighbors += cons(IM3,k,j,i-1);
            }
            u_e -= ke;
            if (n_neighbors>0.0){
              u_m1 = m1_neighbors/n_neighbors;
              u_m2 = m2_neighbors/n_neighbors;
              u_m3 = m3_neighbors/n_neighbors;
            } else {
              u_m1 = 0.0 ;
              u_m2 = 0.0 ;
              u_m3 = 0.0 ;
            }
            w_vx = u_m1*di;
            w_vy = u_m2*di;
            w_vz = u_m3*di;
            ke = 0.5*di*(SQR(u_m1) + SQR(u_m2) + SQR(u_m3));
            u_e += ke; 
            std::cout << " after vceil u_e " << u_e << " w_vx " << w_vx << " w_vy " << w_vy << " w_vz " << w_vz << " n_neighbors " << n_neighbors << "\n";
          }
        }
      }
    }
  }}
  return;
}


//----------------------------------------------------------------------------------------
// \!fn void EquationOfState::PrimitiveToConserved(const AthenaArray<Real> &prim,
//           const AthenaArray<Real> &bc, AthenaArray<Real> &cons, Coordinates *pco,
//           int il, int iu, int jl, int ju, int kl, int ku);
// \brief Converts primitive variables into conservative variables

void EquationOfState::PrimitiveToConserved(const AthenaArray<Real> &prim,
     const AthenaArray<Real> &bc, AthenaArray<Real> &cons, Coordinates *pco,
     int il, int iu, int jl, int ju, int kl, int ku) {
  Real igm1 = 1.0/(GetGamma() - 1.0);

  // Force outer-loop vectorization
#pragma omp simd
  for (int k=kl; k<=ku; ++k) {
  for (int j=jl; j<=ju; ++j) {
    //#pragma omp simd
#pragma novector
    for (int i=il; i<=iu; ++i) {
      Real& u_d  = cons(IDN,k,j,i);
      Real& u_m1 = cons(IM1,k,j,i);
      Real& u_m2 = cons(IM2,k,j,i);
      Real& u_m3 = cons(IM3,k,j,i);
      Real& u_e  = cons(IEN,k,j,i);

      const Real& w_d  = prim(IDN,k,j,i);
      const Real& w_vx = prim(IVX,k,j,i);
      const Real& w_vy = prim(IVY,k,j,i);
      const Real& w_vz = prim(IVZ,k,j,i);
      const Real& w_p  = prim(IPR,k,j,i);

      u_d = w_d;
      u_m1 = w_vx*w_d;
      u_m2 = w_vy*w_d;
      u_m3 = w_vz*w_d;
      u_e = w_p*igm1 + 0.5*w_d*(SQR(w_vx) + SQR(w_vy) + SQR(w_vz));
    }
  }}

  return;
}

//----------------------------------------------------------------------------------------
// \!fn Real EquationOfState::SoundSpeed(Real prim[NHYDRO])
// \brief returns adiabatic sound speed given vector of primitive variables
Real EquationOfState::SoundSpeed(const Real prim[NHYDRO]) {
  return std::sqrt(gamma_*prim[IPR]/prim[IDN]);
}

//---------------------------------------------------------------------------------------
// \!fn void EquationOfState::ApplyPrimitiveFloors(AthenaArray<Real> &prim,
//           int k, int j, int i)
// \brief Apply density and pressure floors to reconstructed L/R cell interface states
void EquationOfState::ApplyPrimitiveFloors(AthenaArray<Real> &prim, int k, int j, int i) {
  Real& w_d  = prim(IDN,k,j,i);
  Real& w_p  = prim(IPR,k,j,i);

  // apply density floor
  w_d = (w_d > density_floor_) ?  w_d : density_floor_;
  // apply pressure floor
  w_p = (w_p > pressure_floor_) ?  w_p : pressure_floor_;

  return;
}
