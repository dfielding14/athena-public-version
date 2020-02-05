// Logo obtained by cooling function on circular AMR grid

// C++ headers
#include <algorithm>  // max
#include <cmath>      // abs, cos, sin
#include <cstdio>     // fclose, FILE, fopen, fread, fseek, SEEK_SET

// Athena++ headers
#include "../mesh/mesh.hpp"
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../parameter_input.hpp"
#include "../bvals/bvals.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"

// Declarations
void Source(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &bcc, AthenaArray<Real> &cons);
int RefinementCondition(MeshBlock *pmb);

// Global variables
static Real gamma_adi;
static Real scale;
static Real temp_hot, temp_cold, tau_hot_1, tau_hot_2, tau_cold_1, tau_cold_2;
static Real blast_start, blast_end, blast_r, blast_h, blast_temp, blast_vel;
static Real tau_damping;
static Real amr_curvature;
static int width, height;

//----------------------------------------------------------------------------------------

void Mesh::InitUserMeshData(ParameterInput *pin)
{
  // Read inputs
  gamma_adi = pin->GetReal("hydro", "gamma");
  scale = pin->GetReal("problem", "scale");
  temp_hot = pin->GetReal("problem", "temp_hot");
  temp_cold = pin->GetReal("problem", "temp_cold");
  tau_hot_1 = pin->GetReal("problem", "tau_hot_1");
  tau_hot_2 = pin->GetReal("problem", "tau_hot_2");
  tau_cold_1 = pin->GetReal("problem", "tau_cold_1");
  tau_cold_2 = pin->GetReal("problem", "tau_cold_2");
  blast_start = pin->GetReal("problem", "blast_start");
  blast_end = pin->GetReal("problem", "blast_end");
  blast_r = pin->GetReal("problem", "blast_r");
  blast_h = pin->GetReal("problem", "blast_h");
  blast_temp = pin->GetReal("problem", "blast_temp");
  blast_vel = pin->GetReal("problem", "blast_vel");
  tau_damping = pin->GetReal("problem", "tau_damping");
  amr_curvature = pin->GetReal("problem", "amr_curvature");

  // Read file header
  unsigned char head[54];
  FILE *bmp = fopen("logobw.bmp", "r");
  fread(head, 1, 54, bmp);
  width = *reinterpret_cast<int *>(&head[18]);
  height = *reinterpret_cast<int *>(&head[22]);
  int offset = *reinterpret_cast<int *>(&head[10]);

  // Read pixel data
  int row_size = width/32 * 4;
  if (width%32 != 0) {
    row_size += 4;
  }
  int file_size = row_size * height;
  unsigned char *data = new unsigned char[file_size];
  fseek(bmp, offset, SEEK_SET);
  fread(data, 1, file_size, bmp);

  // Prepare user data
  AllocateIntUserMeshDataField(1);
  iuser_mesh_data[0].NewAthenaArray(width, height);
  for (int j = 0; j < height; ++j) {
    for (int i = 0; i < width/8; ++i) {
      int fpos = j*row_size + i;
      for (int k = 0; k < 8; ++k) {
        int rev_i = i*8 + 7 - k;
        iuser_mesh_data[0](j,rev_i) = (data[fpos] >> k) & 1;
      }
    }
  }

  // Enroll user functions
  EnrollUserExplicitSourceFunction(Source);
  EnrollUserRefinementCondition(RefinementCondition);

  // Clean up
  delete [] data;
  fclose(bmp);
  return;
}

//----------------------------------------------------------------------------------------

void MeshBlock::ProblemGenerator(ParameterInput *pin)
{
  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is; i <= ie; ++i) {
        phydro->u(IDN,k,j,i) = 1.0;
        phydro->u(IM1,k,j,i) = 0.0;
        phydro->u(IM2,k,j,i) = 0.0;
        phydro->u(IM3,k,j,i) = 0.0;
        phydro->u(IEN,k,j,i) = phydro->u(IDN,k,j,i) * temp_hot / (gamma_adi-1.0);
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------

void Source(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &bcc, AthenaArray<Real> &cons)
{
  // Apply cooling function
  Real tau_hot = tau_hot_1;
  Real tau_cold = tau_cold_1;
  if (time >= blast_start) {
    tau_hot = tau_hot_2;
    tau_cold = tau_cold_2;
  }
  Real ratio_hot = dt / tau_hot;
  Real ratio_cold = dt / tau_cold;
  Real r_max = pmb->pmy_mesh->mesh_size.x1max;
  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      Real phi = pmb->pcoord->x2v(j);
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        Real r = pmb->pcoord->x1v(i);
        Real x = r * std::cos(phi);
        Real y = r * std::sin(phi);
        Real fx = width * x/r_max * scale;
        Real fy = height * y/r_max * scale;
        Real kinetic = 0.5 / cons(IDN,k,j,i)
            * (SQR(cons(IM1,k,j,i)) + SQR(cons(IM2,k,j,i)) + SQR(cons(IM3,k,j,i)));
        Real internal = cons(IEN,k,j,i) - kinetic;
        Real temperature = internal * (gamma_adi-1.0) / cons(IDN,k,j,i);
        Real ratio = ratio_hot;
        Real temp_set = temp_hot;
        if (std::abs(fx) < width/2 and std::abs(fy) < height/2) {
          int j_pos = fy + height/2;
          int i_pos = fx + width/2;
          if (pmb->pmy_mesh->iuser_mesh_data[0](j_pos,i_pos) == 0) {
            ratio = ratio_cold;
            temp_set = temp_cold;
          }
        }
        cons(IEN,k,j,i) = cons(IDN,k,j,i) / (gamma_adi-1.0)
            * (temperature + ratio * temp_set) / (1.0+ratio) + kinetic;
      }
    }
  }

  // Apply kinetic damping
  if (time >= blast_start) {
    Real ratio = dt / tau_damping;
    for (int k = pmb->ks; k <= pmb->ke; ++k) {
      for (int j = pmb->js; j <= pmb->je; ++j) {
        Real phi = pmb->pcoord->x2v(j);
        for (int i = pmb->is; i <= pmb->ie; ++i) {
          Real r = pmb->pcoord->x1v(i);
          Real x = r * std::cos(phi);
          Real y = r * std::sin(phi);
          Real fx = width * x/r_max * scale;
          Real fy = height * y/r_max * scale;
          if (std::abs(fx) < width/2 and std::abs(fy) < height/2) {
            int j_pos = fy + height/2;
            int i_pos = fx + width/2;
            if (pmb->pmy_mesh->iuser_mesh_data[0](j_pos,i_pos) == 0) {
              const Real &rho = cons(IDN,k,j,i);
              Real &m1 = cons(IM1,k,j,i);
              Real &m2 = cons(IM2,k,j,i);
              Real &m3 = cons(IM3,k,j,i);
              Real &e = cons(IEN,k,j,i);
              Real kinetic = 0.5 / rho * (SQR(m1) + SQR(m2) + SQR(m3));
              m1 = m1 / (1.0 + ratio);
              m2 = m2 / (1.0 + ratio);
              m3 = m3 / (1.0 + ratio);
              Real kinetic_set = 0.5 / rho * (SQR(m1) + SQR(m2) + SQR(m3));
              e += kinetic_set - kinetic;
            }
          }
        }
      }
    }
  }

  // Apply blast wave
  if (time >= blast_start and time < blast_end) {
    for (int k = pmb->ks; k <= pmb->ke; ++k) {
      for (int j = pmb->js; j <= pmb->je; ++j) {
        Real phi = pmb->pcoord->x2v(j);
        for (int i = pmb->is; i <= pmb->ie; ++i) {
          Real r = pmb->pcoord->x1v(i);
          Real x = r * std::cos(phi);
          Real y = r * std::sin(phi);
          if ((blast_r > 0.0 and r <= blast_r)
              or (blast_h > 0.0 and std::abs(y) <= blast_h)) {
            Real rho = cons(IDN,k,j,i);
            Real v1, v2;
            if (blast_r > 0.0 and r <= blast_r) {
              v1 = blast_vel;
              v2 = 0.0;
            } else {
              v1 = blast_vel * y / r;
              v2 = blast_vel * x / r;
            }
            cons(IM1,k,j,i) = rho * v1;
            cons(IM2,k,j,i) = rho * v2;
            cons(IM3,k,j,i) = 0.0;
            Real kinetic = 0.5 * cons(IDN,k,j,i) * (SQR(v1) + SQR(v2));
            Real internal = cons(IDN,k,j,i) * blast_temp / (gamma_adi-1.0);
            cons(IEN,k,j,i) = kinetic + internal;
          }
        }
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------

int RefinementCondition(MeshBlock *pmb)
{
  // Calculate temperature curvature
  AthenaArray<Real> &prim = pmb->phydro->w;
  Real maxeps = 0.0;
  int k = pmb->ks;
  for (int j = pmb->js; j <= pmb->je; ++j) {
    for (int i = pmb->is; i <= pmb->ie; ++i) {
      Real temp_m = prim(IPR,k,j,i-1) / prim(IDN,k,j,i-1);
      Real temp_c = prim(IPR,k,j,i) / prim(IDN,k,j,i);
      Real temp_p = prim(IPR,k,j,i+1) / prim(IDN,k,j,i+1);
      Real eps_r = std::abs(temp_m - 2.0*temp_c + temp_p);
      temp_m = prim(IPR,k,j-1,i) / prim(IDN,k,j-1,i);
      temp_p = prim(IPR,k,j+1,i) / prim(IDN,k,j+1,i);
      Real eps_phi = std::abs(temp_m - 2.0*temp_c + temp_p);
      Real eps = (eps_r + eps_phi) / temp_c;
      maxeps = std::max(maxeps, eps);
    }
  }

  // Determine refinement
  if(maxeps > amr_curvature) {
    return 1;
  }
  if(maxeps < 0.5*amr_curvature) {
    return -1;
  }
  return 0;
}
