//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file turbulence.cpp
//  \brief implementation of functions in class Turbulence

// C/C++ headers
#include <iostream>
#include <sstream>    // sstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()
#include <cmath>
#include <algorithm>

// Athena++ headers
#include "athena_fft.hpp"
#include "turbulence.hpp"
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../mesh/mesh.hpp"
#include "../coordinates/coordinates.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../utils/utils.hpp"

//----------------------------------------------------------------------------------------
//! \fn TurbulenceDriver::TurbulenceDriver(Mesh *pm, ParameterInput *pin)
//  \brief TurbulenceDriver constructor

TurbulenceDriver::TurbulenceDriver(Mesh *pm, ParameterInput *pin)
 : FFTDriver(pm, pin) {

  rseed = pin->GetOrAddInteger("problem","rseed",-1); // seed for random number.

  nlow = pin->GetOrAddInteger("problem","nlow",0); // cut-off wavenumber
  nhigh = pin->GetOrAddInteger("problem","nhigh",pm->mesh_size.nx1/2); // cut-off wavenumber, high:
  expo = pin->GetOrAddReal("problem","expo",2); // power-law exponent
  dedt = pin->GetReal("problem","dedt"); // turbulence amplitude
  dtdrive = pin->GetReal("problem","dtdrive"); // driving interval
  tdrive = pm->time;
  // added by DBF
  tcorr = pin->GetOrAddReal("problem","tcorr", 0.0); // correlation time scale
  z_turb = pin->GetOrAddReal("problem","z_turb", 0.0); // scale height of driving pattern 
  dpert = pin->GetOrAddReal("problem","dpert", 0.0); // density perturbation = delta_rho / rho 
  f_solenoidal = pin->GetOrAddReal("problem","f_solenoidal", -1.0); // density perturbation = delta_rho / rho 


  if (pm->turb_flag == 0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in TurbulenceDriver::TurbulenceDriver" << std::endl
        << "Turbulence flag is set to zero! Shouldn't reach here!" << std::endl;
    throw std::runtime_error(msg.str().c_str());
    return;
  } else {
#ifndef FFT
    std::stringstream msg;
    msg << "### FATAL ERROR in TurbulenceDriver::TurbulenceDriver" << std::endl
        << "non zero Turbulence flag is set without FFT!" << std::endl;
    throw std::runtime_error(msg.str().c_str());
    return;
#endif
  }

  int nx1=pm->pblock->block_size.nx1+2*NGHOST;
  int nx2=pm->pblock->block_size.nx2+2*NGHOST;
  int nx3=pm->pblock->block_size.nx3+2*NGHOST;


  // added by DBF
  // if turb_flag == 4 then its a density perturbation and not velocity
  if (pm->turb_flag != 4) {
    vel = new AthenaArray<Real>[3];
    for(int nv=0; nv<3; nv++) vel[nv].NewAthenaArray(nmb,nx3,nx2,nx1);
  } else {
    drho = new AthenaArray<Real>[1];
    drho[0].NewAthenaArray(nmb,nx3,nx2,nx1);
  }

  InitializeFFTBlock(true);
  QuickCreatePlan();
  dvol = pmy_fb->dx1*pmy_fb->dx2*pmy_fb->dx3;

  if (f_solenoidal>=0.0) {
    fv_solenoidal_ = new AthenaFFTComplex*[3];
    fv_compressive_ = new AthenaFFTComplex*[3];
    for (int nv=0; nv<3; nv++){
      fv_solenoidal_[nv] = new AthenaFFTComplex[pmy_fb->cnt_];
      fv_compressive_[nv] = new AthenaFFTComplex[pmy_fb->cnt_];
    } 
    if (Globals::my_rank==0)
      std::cout << "Using solenoidal compressive partitioning with f_solenoidal = " << f_solenoidal <<"\n";
  }


}

// destructor
TurbulenceDriver::~TurbulenceDriver() {
  if (vel != NULL) {
    for (int nv=0; nv<3; nv++) vel[nv].DeleteAthenaArray();
    delete [] vel;
  } 
  if (drho != NULL) {
    drho[0].DeleteAthenaArray();
    delete [] drho;
  }
}

//----------------------------------------------------------------------------------------
//! \fn void TurbulenceDriver::Driving(void)
//  \brief Generate and Perturb the velocity field

void TurbulenceDriver::Driving(void) {
  Mesh *pm=pmy_mesh_;
  bool new_perturb = false;

// check driving time interval to generate new perturbation
  if ((pm->time >= tdrive) and (pm->turb_flag != 4)) {
    if (Globals::my_rank==0)
      std::cout << "generating turbulence at " << pm->time << std::endl;
    Generate();
    tdrive = pm->time + dtdrive;
    new_perturb = true;
  }

  switch(pm->turb_flag) {
    case 1: // turb_flag == 1 : decaying turbulence
      Perturb(0);
      break;
    case 2: // turb_flag == 2 : impulsively driven turbulence
      if (new_perturb) Perturb(dtdrive);
      break;
    case 3: // turb_flag == 3 : continuously driven turbulence
      Perturb(pm->dt);
      break;
    case 4: // turb_flag == 4 : density perturbation
      if(pm->time==0.0){
        Generate1D();
        PerturbDensity();
        if (Globals::my_rank==0)
          std::cout << "Perturbed Density at " << pm->time << std::endl;
      }
      break;
    default:
      std::stringstream msg;
      msg << "### FATAL ERROR in TurbulenceDriver::Driving" << std::endl
          << "Turbulence flag " << pm->turb_flag << " is not supported!" << std::endl;
      throw std::runtime_error(msg.str().c_str());
  }

  return;

}

//----------------------------------------------------------------------------------------
//! \fn void TurbulenceDriver::Generate()
//  \brief Generate velocity pertubation.

void TurbulenceDriver::Generate(void) {
  Mesh *pm=pmy_mesh_;
  FFTBlock *pfb = pmy_fb;
  AthenaFFTPlan *plan = pfb->bplan_;
  AthenaFFTIndex *idx = pfb->b_in_;

  int nbs=nslist_[Globals::my_rank];
  int nbe=nbs+nblist_[Globals::my_rank]-1;
  int knx1=pfb->knx[0],knx2=pfb->knx[1],knx3=pfb->knx[2];

  if (tcorr == 0.0){ // original
    for (int nv=0; nv<3; nv++) {
      AthenaArray<Real> &dv = vel[nv], dv_mb;
      AthenaFFTComplex *fv = pfb->in_;

      PowerSpectrum(fv);
      if (f_solenoidal>=0.0){
        // copy fv to fv_solenoidal
        // subtract off the compressive part of fv_solenoidal
        // make a new fv
        // copy new fv to fv_compressive
        // subtract off the solenoidal part of fv_compressive
        // add fv_solenoidal to fv_compressive with the relative
        //   strength set by f_solenoidal
        // copy new combined fv to original fv and proceed
        for (int k=0; k<knx3; k++) {
          for (int j=0; j<knx2; j++) {
            for (int i=0; i<knx1; i++) {
              int64_t kidx=pfb->GetIndex(i,j,k,idx);
              // Copy to fv_solenoidal_
              fv_solenoidal_[nv][kidx][0] = fv[kidx][0];
              fv_solenoidal_[nv][kidx][1] = fv[kidx][1];
            }
          }
        }
        Project(fv_solenoidal_, 1); // delete the compressive part 
        if (f_solenoidal < 1.0){
          PowerSpectrum(fv);
          for (int k=0; k<knx3; k++) {
            for (int j=0; j<knx2; j++) {
              for (int i=0; i<knx1; i++) {
                int64_t kidx=pfb->GetIndex(i,j,k,idx);
                // Copy to fv_compressive_
                fv_compressive_[nv][kidx][0] = fv[kidx][0];
                fv_compressive_[nv][kidx][1] = fv[kidx][1];
              }
            }
          }
          Project(fv_compressive_, 0); // delete the solenoidal part
          for (int k=0; k<knx3; k++) {
            for (int j=0; j<knx2; j++) {
              for (int i=0; i<knx1; i++) {
                int64_t kidx=pfb->GetIndex(i,j,k,idx);
                // Copy to back to fv with proper ratio
                fv[kidx][0] = (1-f_solenoidal)*fv_compressive_[nv][kidx][0] + f_solenoidal*fv_solenoidal_[nv][kidx][0];
                fv[kidx][1] = (1-f_solenoidal)*fv_compressive_[nv][kidx][1] + f_solenoidal*fv_solenoidal_[nv][kidx][1];
              }
            }
          }
        } else { // f_solenoidal == 1
          for (int k=0; k<knx3; k++) {
            for (int j=0; j<knx2; j++) {
              for (int i=0; i<knx1; i++) {
                int64_t kidx=pfb->GetIndex(i,j,k,idx);
                // Copy to back to fv with proper ratio
                fv[kidx][0] = fv_solenoidal_[nv][kidx][0];
                fv[kidx][1] = fv_solenoidal_[nv][kidx][1];
              }
            }
          }
        }
      } // f_solenoidal>=0
      pfb->Execute(plan);

      for (int igid=nbs, nb=0;igid<=nbe;igid++, nb++) {
        MeshBlock *pmb=pm->FindMeshBlock(igid);
        if (pmb != NULL) {
          dv_mb.InitWithShallowSlice(dv, 4, nb, 1);
          pfb->RetrieveResult(dv_mb,1,NGHOST,pmb->loc,pmb->block_size);
        }
      }
    }
  } else { // with Ornstein-Uhlenbeck smoothing
    Real factor;
    factor = exp(-dtdrive/tcorr);
    if (Globals::my_rank==0) std::cout << "starting generate with OU, factor = " << factor <<"\n";

    int is=pm->pblock->is, ie=pm->pblock->ie;
    int js=pm->pblock->js, je=pm->pblock->je;
    int ks=pm->pblock->ks, ke=pm->pblock->ke;
    
    for (int nv=0; nv<3; nv++) {
      AthenaArray<Real> &dv = vel[nv], dv_mb, dv_mb_previous;
      AthenaFFTComplex *fv = pfb->in_;

      PowerSpectrum(fv);
      if (f_solenoidal>=0.0){
        for (int k=0; k<knx3; k++) {
          for (int j=0; j<knx2; j++) {
            for (int i=0; i<knx1; i++) {
              int64_t kidx=pfb->GetIndex(i,j,k,idx);
              // Copy to fv_solenoidal_
              fv_solenoidal_[nv][kidx][0] = fv[kidx][0];
              fv_solenoidal_[nv][kidx][1] = fv[kidx][1];
            }
          }
        }
        Project(fv_solenoidal_, 1); // delete the compressive part 
        PowerSpectrum(fv);
        for (int k=0; k<knx3; k++) {
          for (int j=0; j<knx2; j++) {
            for (int i=0; i<knx1; i++) {
              int64_t kidx=pfb->GetIndex(i,j,k,idx);
              // Copy to fv_compressive_
              fv_compressive_[nv][kidx][0] = fv[kidx][0];
              fv_compressive_[nv][kidx][1] = fv[kidx][1];
            }
          }
        }
        Project(fv_compressive_, 0); // delete the solenoidal part
        for (int k=0; k<knx3; k++) {
          for (int j=0; j<knx2; j++) {
            for (int i=0; i<knx1; i++) {
              int64_t kidx=pfb->GetIndex(i,j,k,idx);
              // Copy to back to fv with proper ratio
              fv[kidx][0] = (1-f_solenoidal)*fv_compressive_[nv][kidx][0] + f_solenoidal*fv_solenoidal_[nv][kidx][0];
              fv[kidx][1] = (1-f_solenoidal)*fv_compressive_[nv][kidx][1] + f_solenoidal*fv_solenoidal_[nv][kidx][1];
            }
          }
        }
      } // f_solenoidal>=0
      pfb->Execute(plan);

      for (int igid=nbs, nb=0;igid<=nbe;igid++, nb++) {
        MeshBlock *pmb=pm->FindMeshBlock(igid);
        if (pmb != NULL) {
          dv_mb.InitWithShallowSlice(dv, 4, nb, 1);
          dv_mb_previous.NewAthenaArray(dv_mb.GetDim3(),dv_mb.GetDim2(),dv_mb.GetDim1());
          dv_mb_previous = dv_mb;
          pfb->RetrieveResult(dv_mb,1,NGHOST,pmb->loc,pmb->block_size);
          for (int k=ks; k<=ke; k++) {
            for (int j=js; j<=je; j++) {
              for (int i=is; i<=ie; i++) {
                if (z_turb > 0.0){
                  Real z = pmb->pcoord->x3v(k); // could do this in the k loop but this way it is all grouped under z_turb
                  dv_mb(k,j,i) *= 0.5 * (1.0 + std::tanh( (z_turb-fabs(z))/(fabs(z_turb)*0.5) )); // smoothly make driving pattern go to zero above z_turb
                }
                dv_mb(k,j,i) *= sqrt(1-SQR(factor)); // Lynn, Parrish, Quataert, and Chandran 2012 eq. (26)
                dv_mb(k,j,i) += factor*dv_mb_previous(k,j,i);
              }
            }
          }
        }
      }
      dv_mb_previous.DeleteAthenaArray();
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void TurbulenceDriver::Generate1D(void)
//  \brief Generate velocity pertubation.

void TurbulenceDriver::Generate1D(void)
{
  Mesh *pm=pmy_mesh_;
  FFTBlock *pfb = pmy_fb;
  AthenaFFTPlan *plan = pfb->bplan_; //backward fft

  int nbs=nslist_[Globals::my_rank];
  int nbe=nbs+nblist_[Globals::my_rank]-1;

  AthenaArray<Real> &dd = drho[0], drho_mb;
  AthenaFFTComplex *fd = pfb->in_;

  PowerSpectrum(fd);//place power spectrum fluctuations into fv=pfb->in_

  pfb->Execute(plan);//backward fft the power spectrum fluctuations

  for(int igid=nbs, nb=0;igid<=nbe;igid++, nb++){
    MeshBlock *pmb=pm->FindMeshBlock(igid);
    if(pmb != NULL){
      drho_mb.InitWithShallowSlice(dd, 4, nb, 1);
      pfb->RetrieveResult(drho_mb,1,NGHOST,pmb->loc,pmb->block_size);
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void TurbulenceDriver::PowerSpectrum(AthenaFFTComplex *amp)
//  \brief Generate Power spectrum in Fourier space with power-law

void TurbulenceDriver::PowerSpectrum(AthenaFFTComplex *amp) {
  Real pcoeff;
  FFTBlock *pfb = pmy_fb;
  AthenaFFTIndex *idx = pfb->b_in_;
  int knx1=pfb->knx[0],knx2=pfb->knx[1],knx3=pfb->knx[2];
// set random amplitudes with gaussian deviation
  for (int k=0; k<knx3; k++) {
    for (int j=0; j<knx2; j++) {
      for (int i=0; i<knx1; i++) {
        Real q1=ran2(&rseed);
        Real q2=ran2(&rseed);
        Real q3=std::sqrt(-2.0*std::log(q1+1.e-20))*std::cos(2.0*PI*q2);
        q1=ran2(&rseed);
        int64_t kidx=pfb->GetIndex(i,j,k,idx);
        amp[kidx][0] = q3*std::cos(2.0*PI*q1);
        amp[kidx][1] = q3*std::sin(2.0*PI*q1);
      }
    }
  }

// set power spectrum: only power-law
  for (int k=0; k<knx3; k++) {
    for (int j=0; j<knx2; j++) {
      for (int i=0; i<knx1; i++) {
        int64_t nx=GetKcomp(i,pfb->kdisp[0],pfb->kNx[0]);
        int64_t ny=GetKcomp(j,pfb->kdisp[1],pfb->kNx[1]);
        int64_t nz=GetKcomp(k,pfb->kdisp[2],pfb->kNx[2]);
        Real nmag = std::sqrt(nx*nx+ny*ny+nz*nz);
        Real kx=nx*pfb->dkx[0];
        Real ky=ny*pfb->dkx[1];
        Real kz=nz*pfb->dkx[2];
        Real kmag = std::sqrt(kx*kx+ky*ky+kz*kz);

        int64_t gidx = pfb->GetGlobalIndex(i,j,k);

        if (gidx == 0) {
          pcoeff = 0.0;
        } else {
          if ((nmag > nlow) && (nmag < nhigh)) {
            pcoeff = 1.0/std::pow(kmag,(expo+2.0)/2.0);
          } else {
            pcoeff = 0.0;
          }
        }
        int64_t kidx=pfb->GetIndex(i,j,k,idx);
        amp[kidx][0] *= pcoeff;
        amp[kidx][1] *= pcoeff;
      }
    }
  }
  return;
}



//----------------------------------------------------------------------------------------
//! \fn void TurbulenceDriver::Project(AthenaFFTComplex **fv)
//  \brief calculates divergence free part of the perturbation
//         and then either removes it or keeps only it.

void TurbulenceDriver::Project(AthenaFFTComplex **fv, int solenoidal) {
  // if solenoidal == 1 then subtract off the div part 
  // if solenoidal == 0 then keep div part
  FFTBlock *pfb = pmy_fb;
  AthenaFFTIndex *idx = pfb->b_in_;
  int knx1=pfb->knx[0],knx2=pfb->knx[1],knx3=pfb->knx[2];
  Real divsum = 0.0;
  for (int k=0; k<knx3; k++) {
    for (int j=0; j<knx2; j++) {
      for (int i=0; i<knx1; i++) {
        // Get khat
        int64_t nx=GetKcomp(i,pfb->kdisp[0],pfb->kNx[0]);
        int64_t ny=GetKcomp(j,pfb->kdisp[1],pfb->kNx[1]);
        int64_t nz=GetKcomp(k,pfb->kdisp[2],pfb->kNx[2]);
        Real kx=nx*pfb->dkx[0]; // dkx[i] gets weirdly permuted (based on meshblocks) when using MPI!!!
        Real ky=ny*pfb->dkx[1]; // JS: I *think* I fixed, Lx[i] wasn't in the functions to swap and permute axes
        Real kz=nz*pfb->dkx[2];
        Real kmag = std::sqrt(kx*kx+ky*ky+kz*kz);
        // khat stored in kx,ky,kz
        kx /= kmag;
        ky /= kmag;
        kz /= kmag;
        
        int64_t kidx=pfb->GetIndex(i,j,k,idx);
        if (kmag != 0.0) {
          // Form (khat.f)
          Real kdotf_re = kx*fv[0][kidx][0] + ky*fv[1][kidx][0] + kz*fv[2][kidx][0];
          Real kdotf_im = kx*fv[0][kidx][1] + ky*fv[1][kidx][1] + kz*fv[2][kidx][1];
          
          if (solenoidal == 1){
            fv[0][kidx][0] -= kdotf_re * kx;
            fv[1][kidx][0] -= kdotf_re * ky;
            fv[2][kidx][0] -= kdotf_re * kz;
            
            fv[0][kidx][1] -= kdotf_im * kx;
            fv[1][kidx][1] -= kdotf_im * ky;
            fv[2][kidx][1] -= kdotf_im * kz;
          } 
          if (solenoidal == 0){
            fv[0][kidx][0] = kdotf_re * kx;
            fv[1][kidx][0] = kdotf_re * ky;
            fv[2][kidx][0] = kdotf_re * kz;
            
            fv[0][kidx][1] = kdotf_im * kx;
            fv[1][kidx][1] = kdotf_im * ky;
            fv[2][kidx][1] = kdotf_im * kz;
          } 
        }
      }
    }       
  }
  return;
}



//----------------------------------------------------------------------------------------
//! \fn void TurbulenceDriver::Perturb(Real dt)
//  \brief Add velocity perturbation to the hydro variables

void TurbulenceDriver::Perturb(Real dt) {
  Mesh *pm = pmy_mesh_;
  std::stringstream msg;
  int nbs=nslist_[Globals::my_rank];
  int nbe=nbs+nblist_[Globals::my_rank]-1;

  int is=pm->pblock->is, ie=pm->pblock->ie;
  int js=pm->pblock->js, je=pm->pblock->je;
  int ks=pm->pblock->ks, ke=pm->pblock->ke;

  int mpierr;
  Real aa, b, c, s, de, v1, v2, v3, den, M1, M2, M3;
  Real m[4] = {0}, gm[4];
  AthenaArray<Real> &dv1 = vel[0], &dv2 = vel[1], &dv3 = vel[2];

  for (int igid=nbs, nb=0; igid<=nbe; igid++, nb++) {
    MeshBlock *pmb=pm->FindMeshBlock(igid);
    if (pmb != NULL) {
      for (int k=ks; k<=ke; k++) {
        for (int j=js; j<=je; j++) {
          for (int i=is; i<=ie; i++) {
            den = pmb->phydro->u(IDN,k,j,i);
            m[0] += den;
            m[1] += den*dv1(nb,k,j,i);
            m[2] += den*dv2(nb,k,j,i);
            m[3] += den*dv3(nb,k,j,i);
          }
        }
      }
    }
  }

#ifdef MPI_PARALLEL
// Sum the perturbations over all processors
  mpierr = MPI_Allreduce(m, gm, 4, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  if (mpierr) {
    msg << "[normalize]: MPI_Allreduce error = "
        << mpierr << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }
  // Ask Changgoo about this
  for (int n=0; n<4; n++) m[n]=gm[n];
#endif // MPI_PARALLEL

  for (int nb=0; nb<nmb; nb++) {
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie; i++) {
          dv1(nb,k,j,i) -= m[1]/m[0];
          dv2(nb,k,j,i) -= m[2]/m[0];
          dv3(nb,k,j,i) -= m[3]/m[0];
        }
      }
    }
  }

  // Calculate unscaled energy of perturbations
  m[0] = 0.0;
  m[1] = 0.0;
  for (int igid=nbs, nb=0;igid<=nbe;igid++, nb++) {
    MeshBlock *pmb=pm->FindMeshBlock(igid);
    if (pmb != NULL) {
      for (int k=ks; k<=ke; k++) {
        for (int j=js; j<=je; j++) {
          for (int i=is; i<=ie; i++) {
            v1 = dv1(nb,k,j,i);
            v2 = dv2(nb,k,j,i);
            v3 = dv3(nb,k,j,i);
            den = pmb->phydro->u(IDN,k,j,i);
            M1 = pmb->phydro->u(IM1,k,j,i);
            M2 = pmb->phydro->u(IM2,k,j,i);
            M3 = pmb->phydro->u(IM3,k,j,i);
            m[0] += den*(SQR(v1) + SQR(v2) + SQR(v3));
            m[1] += M1*v1 + M2*v2 + M3*v3;
          }
        }
      }
    }
  }

#ifdef MPI_PARALLEL
  // Sum the perturbations over all processors
  mpierr = MPI_Allreduce(m, gm, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  if (mpierr) {
    msg << "[normalize]: MPI_Allreduce error = "
        << mpierr << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }
  //  if (mpierr) ath_error("[normalize]: MPI_Allreduce error = %d\n", mpierr);
  m[0] = gm[0];  m[1] = gm[1];
#endif // MPI_PARALLEL

  // Rescale to give the correct energy injection rate
  if (pm->turb_flag > 1) {
    // driven turbulence
    de = dedt*dt;
    if (Globals::my_rank==0)
      std::cout << "driven turbulence with " << de << std::endl;
  } else {
    // decaying turbulence (all in one shot)
    de = dedt;
    if (Globals::my_rank==0)
      std::cout << "decaying turbulence with " << de << std::endl;
  }
  aa = 0.5*m[0];
  aa = std::max(aa,static_cast<Real>(1.0e-20));
  b = m[1];
  c = -de/dvol;
  if (b >= 0.0)
    s = (-2.0*c)/(b + std::sqrt(b*b - 4.0*aa*c));
  else
    s = (-b + std::sqrt(b*b - 4.0*aa*c))/(2.0*aa);

  if (std::isnan(s)) std::cout << "[perturb]: s is NaN!" << std::endl;

  // Apply momentum pertubations
  for (int igid=nbs, nb=0; igid<=nbe; igid++, nb++) {
    MeshBlock *pmb=pm->FindMeshBlock(igid);
    if (pmb != NULL) {
      for (int k=ks; k<=ke; k++) {
        for (int j=js; j<=je; j++) {
          for (int i=is; i<=ie; i++) {
            v1 = dv1(nb,k,j,i);
            v2 = dv2(nb,k,j,i);
            v3 = dv3(nb,k,j,i);
            den = pmb->phydro->u(IDN,k,j,i);
            M1 = pmb->phydro->u(IM1,k,j,i);
            M2 = pmb->phydro->u(IM2,k,j,i);
            M3 = pmb->phydro->u(IM3,k,j,i);

            if (NON_BAROTROPIC_EOS) {
              pmb->phydro->u(IEN,k,j,i) += s*(M1*v1+M2*v2+M3*v3)
                                         + 0.5*s*s*den*(SQR(v1)+SQR(v2)+SQR(v3));
            }
            pmb->phydro->u(IM1,k,j,i) += s*den*v1;
            pmb->phydro->u(IM2,k,j,i) += s*den*v2;
            pmb->phydro->u(IM3,k,j,i) += s*den*v3;
          }
        }
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void TurbulenceDriver::PerturbDensity(void)
//  \brief Add density perturbation to the hydro variables

void TurbulenceDriver::PerturbDensity(void){
  Mesh *pm = pmy_mesh_;
  std::stringstream msg;
  int nbs=nslist_[Globals::my_rank];
  int nbe=nbs+nblist_[Globals::my_rank]-1;

  int is=pm->pblock->is, ie=pm->pblock->ie;
  int js=pm->pblock->js, je=pm->pblock->je;
  int ks=pm->pblock->ks, ke=pm->pblock->ke;

  int mpierr;
  Real aa, b, c, s, de, v1, v2, v3, den, M1, M2, M3;
  Real m[2] = {0}, gm[2];
  AthenaArray<Real> &dd = drho[0];

  //calculate the mean
  for(int igid=nbs, nb=0;igid<=nbe;igid++, nb++){
    MeshBlock *pmb=pm->FindMeshBlock(igid);
    if(pmb != NULL){
      for (int k=ks; k<=ke; k++) {
        for (int j=js; j<=je; j++) {
          for (int i=is; i<=ie; i++) {
            m[0] += 1.0;
            m[1] += dd(nb,k,j,i);
          }
        }
      }
    }
  }

#ifdef MPI_PARALLEL
// Sum the perturbations over all processors 
  mpierr = MPI_Allreduce(m, gm, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  if (mpierr) {
    msg << "[normalize]: MPI_Allreduce error = " 
  << mpierr << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }
  for (int n=0;n<2;n++) m[n]=gm[n];
#endif // MPI_PARALLEL 

  // subtrack off the mean
  for(int nb=0;nb<nmb;nb++){
    for(int k=ks;k<=ke;k++){
      for(int j=js;j<=je;j++){
        for(int i=is;i<=ie;i++){
          dd(nb,k,j,i) -= m[1]/m[0];
        }
      }
    }
  }

  // calculated the standard deviation
  m[0] = 0.0;  m[1] = 0.0;
  gm[0] = 0.0;  gm[1] = 0.0;

  for(int igid=nbs, nb=0;igid<=nbe;igid++, nb++){
    MeshBlock *pmb=pm->FindMeshBlock(igid);
    if(pmb != NULL){
      for (int k=ks; k<=ke; k++) {
        for (int j=js; j<=je; j++) {
          for (int i=is; i<=ie; i++) {
            m[0] += 1.0;
            m[1] += SQR(dd(nb,k,j,i));
          }
        }
      }
    }
  }

#ifdef MPI_PARALLEL
// Sum the perturbations over all processors 
  mpierr = MPI_Allreduce(m, gm, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  if (mpierr) {
    msg << "[normalize]: MPI_Allreduce error = " 
  << mpierr << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }
  for (int n=0;n<2;n++) m[n]=gm[n];
#endif // MPI_PARALLEL 

  // normalize by the standard deviation and multiply by desired dpert
  for(int nb=0;nb<nmb;nb++){
    for(int k=ks;k<=ke;k++){
      for(int j=js;j<=je;j++){
        for(int i=is;i<=ie;i++){
          dd(nb,k,j,i) /= sqrt(m[1]/m[0]);
          dd(nb,k,j,i) *= dpert; 
          if (dd(nb,k,j,i)<-0.9) dd(nb,k,j,i) = -0.9;
          if (dd(nb,k,j,i)> 0.9) dd(nb,k,j,i) =  0.9;
        }
      }
    }
  }

  // Apply density pertubations keeping velocity constant and adjusting internal energy
  for(int igid=nbs, nb=0;igid<=nbe;igid++, nb++){
    MeshBlock *pmb=pm->FindMeshBlock(igid);
    if(pmb != NULL){
      for (int k=ks; k<=ke; k++) {
        for (int j=js; j<=je; j++) {
          for (int i=is; i<=ie; i++) {
            if (NON_BAROTROPIC_EOS) {
              pmb->phydro->u(IEN,k,j,i) -= 
                0.5*(SQR(pmb->phydro->u(IM1,k,j,i)) + 
                     SQR(pmb->phydro->u(IM2,k,j,i)) +
                     SQR(pmb->phydro->u(IM3,k,j,i)) )/pmb->phydro->u(IDN,k,j,i);
            }
            pmb->phydro->u(IDN,k,j,i) *= (1.0+dd(nb,k,j,i));
            pmb->phydro->u(IM1,k,j,i) *= (1.0+dd(nb,k,j,i));
            pmb->phydro->u(IM2,k,j,i) *= (1.0+dd(nb,k,j,i));
            pmb->phydro->u(IM3,k,j,i) *= (1.0+dd(nb,k,j,i));
            if (NON_BAROTROPIC_EOS) {
              pmb->phydro->u(IEN,k,j,i) += 
                0.5*(SQR(pmb->phydro->u(IM1,k,j,i)) + 
                     SQR(pmb->phydro->u(IM2,k,j,i)) +
                     SQR(pmb->phydro->u(IM3,k,j,i)) )/pmb->phydro->u(IDN,k,j,i);
            }
          }
        }
      }
    }
  }
  return;
}


//----------------------------------------------------------------------------------------
//! \fn void TurbulenceDriver::GetKcomp(int idx, int disp, int Nx)
//  \brief Get k index, which runs from 0, 1, ... Nx/2-1, -Nx/2, -Nx/2+1, ..., -1.

int64_t TurbulenceDriver::GetKcomp(int idx, int disp, int Nx) {
  return ((idx+disp) - static_cast<int64_t>(2*(idx+disp)/Nx)*Nx);
}
