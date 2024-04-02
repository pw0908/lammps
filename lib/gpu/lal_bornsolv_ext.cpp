/***************************************************************************
                              bornsolve_ext.cpp
                             -------------------
                           Pierre J Walker (Caltech)

  Functions for LAMMPS access to born solvation acceleration routines.

 __________________________________________________________________________
    This file is part of the LAMMPS Accelerator Library (LAMMPS_AL)
 __________________________________________________________________________

    begin                :
    email                : pjwalker@caltech.edu
 ***************************************************************************/

#include <iostream>
#include <cassert>
#include <cmath>

#include "lal_bornsolv.h"

using namespace std;
using namespace LAMMPS_AL;

static Bornsolv<PRECISION,ACC_PRECISION> BORNSOLVMF;

// ---------------------------------------------------------------------------
// Allocate memory on host and device and copy constants to device
// ---------------------------------------------------------------------------
int bornsolv_gpu_init(const int ntypes, double **cutsq, double **host_epsilon,
                  double **host_sigma, double **host_bornsolv1, double **host_bornsolv2,
                  double **offset, double *special_lj, const int inum,
                  const int nall, const int max_nbors,  const int maxspecial,
                  const double cell_size, int &gpu_mode, FILE *screen) {
  BORNSOLVMF.clear();
  gpu_mode=BORNSOLVMF.device->gpu_mode();
  double gpu_split=BORNSOLVMF.device->particle_split();
  int first_gpu=BORNSOLVMF.device->first_device();
  int last_gpu=BORNSOLVMF.device->last_device();
  int world_me=BORNSOLVMF.device->world_me();
  int gpu_rank=BORNSOLVMF.device->gpu_rank();
  int procs_per_gpu=BORNSOLVMF.device->procs_per_gpu();

  BORNSOLVMF.device->init_message(screen,"bornsolv",first_gpu,last_gpu);

  bool message=false;
  if (BORNSOLVMF.device->replica_me()==0 && screen)
    message=true;

  if (message) {
    fprintf(screen,"Initializing Device and compiling on process 0...");
    fflush(screen);
  }

  int init_ok=0;
  if (world_me==0)
    init_ok=BORNSOLVMF.init(ntypes, cutsq, host_epsilon, host_sigma, host_bornsolv1,
                        host_bornsolv2,
                        offset, special_lj, inum, nall, max_nbors,
                        maxspecial, cell_size, gpu_split, screen);

  BORNSOLVMF.device->world_barrier();
  if (message)
    fprintf(screen,"Done.\n");

  for (int i=0; i<procs_per_gpu; i++) {
    if (message) {
      if (last_gpu-first_gpu==0)
        fprintf(screen,"Initializing Device %d on core %d...",first_gpu,i);
      else
        fprintf(screen,"Initializing Devices %d-%d on core %d...",first_gpu,
                last_gpu,i);
      fflush(screen);
    }
    if (gpu_rank==i && world_me!=0)
      init_ok=BORNSOLVMF.init(ntypes, cutsq, host_epsilon, host_sigma, host_bornsolv1,
                          host_bornsolv2,
                          offset, special_lj, inum, nall, max_nbors,
                          maxspecial, cell_size, gpu_split, screen);

    BORNSOLVMF.device->serialize_init();
    if (message)
      fprintf(screen,"Done.\n");
  }
  if (message)
    fprintf(screen,"\n");

  if (init_ok==0)
    BORNSOLVMF.estimate_gpu_overhead();
  return init_ok;
}

// ---------------------------------------------------------------------------
// Copy updated coeffs from host to device
// ---------------------------------------------------------------------------
void bornsolv_gpu_reinit(const int ntypes, double **host_epsilon,
                  double **host_sigma, double **host_bornsolv1, double **host_bornsolv2, 
                  double **offset) {
  int world_me=BORNSOLVMF.device->world_me();
  int gpu_rank=BORNSOLVMF.device->gpu_rank();
  int procs_per_gpu=BORNSOLVMF.device->procs_per_gpu();

  if (world_me==0)
    BORNSOLVMF.reinit(ntypes, host_epsilon, host_sigma, host_bornsolv1,
                  host_bornsolv2, offset);

  BORNSOLVMF.device->world_barrier();

  for (int i=0; i<procs_per_gpu; i++) {
    if (gpu_rank==i && world_me!=0)
      BORNSOLVMF.reinit(ntypes, host_epsilon, host_sigma, host_bornsolv1,
                    host_bornsolv2, offset);

    BORNSOLVMF.device->serialize_init();
  }
}

void bornsolv_gpu_clear() {
  BORNSOLVMF.clear();
}

int ** bornsolv_gpu_compute_n(const int ago, const int inum_full,
                          const int nall, double **host_x, int *host_type,
                          double *sublo, double *subhi, tagint *tag, int **nspecial,
                          tagint **special, const bool eflag, const bool vflag,
                          const bool eatom, const bool vatom, int &host_start,
                          int **ilist, int **jnum, const double cpu_time,
                          bool &success) {
  return BORNSOLVMF.compute(ago, inum_full, nall, host_x, host_type, sublo,
                        subhi, tag, nspecial, special, eflag, vflag, eatom,
                        vatom, host_start, ilist, jnum, cpu_time, success);
}

void bornsolv_gpu_compute(const int ago, const int inum_full, const int nall,
                      double **host_x, int *host_type, int *ilist, int *numj,
                      int **firstneigh, const bool eflag, const bool vflag,
                      const bool eatom, const bool vatom, int &host_start,
                      const double cpu_time, bool &success) {
  BORNSOLVMF.compute(ago,inum_full,nall,host_x,host_type,ilist,numj,
                 firstneigh,eflag,vflag,eatom,vatom,host_start,cpu_time,success);
}

double bornsolv_gpu_bytes() {
  return BORNSOLVMF.host_memory_usage();
}