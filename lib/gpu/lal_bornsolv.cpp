/***************************************************************************
                                 bornsolv.cpp
                             -------------------
                           Pierre J Walker (Caltech)

  Class for acceleration of the born pair style.

 __________________________________________________________________________
    This file is part of the LAMMPS Accelerator Library (LAMMPS_AL)
 __________________________________________________________________________

    begin                :
    email                : pjwalker@caltech.edu
 ***************************************************************************/
#ifdef USE_OPENCL
#include "bornsolv_cl.h"
#elif defined(USE_CUDART)
const char *bornsolv=0;
#else
#include "bornsolv_cubin.h"
#endif

#include "lal_bornsolv.h"
#include <cassert>
namespace LAMMPS_AL {
#define BornsolvT Bornsolv<numtyp, acctyp>

extern Device<PRECISION,ACC_PRECISION> device;

template <class numtyp, class acctyp>
BornsolvT::Bornsolv() : BaseAtomic<numtyp,acctyp>(), _allocated(false) {
}

template <class numtyp, class acctyp>
BornsolvT::~Bornsolv() {
  clear();
}

template <class numtyp, class acctyp>
int BornsolvT::bytes_per_atom(const int max_nbors) const {
  return this->bytes_per_atom_atomic(max_nbors);
}

template <class numtyp, class acctyp>
int BornsolvT::init(const int ntypes, double **host_cutsq, double **host_epsilon,
                double **host_sigma, double **host_bornsolv1, double **host_bornsolv2,
                double **host_offset, double *host_special_lj,
                const int nlocal, const int nall, const int max_nbors,
                const int maxspecial, const double cell_size,
                const double gpu_split, FILE *_screen) {
  int success;
  success=this->init_atomic(nlocal,nall,max_nbors,maxspecial,cell_size,gpu_split,
                            _screen,bornsolv,"k_bornsolv");
  if (success!=0)
    return success;

  // If atom type constants fit in shared memory use fast kernel
  int lj_types=ntypes;
  shared_types=false;
  int max_shared_types=this->device->max_shared_types();
  if (lj_types<=max_shared_types && this->_block_size>=max_shared_types) {
    lj_types=max_shared_types;
    shared_types=true;
  }
  _lj_types=lj_types;

  // Allocate a host write buffer for data initialization
  UCL_H_Vec<numtyp> host_write(lj_types*lj_types*32,*(this->ucl_device),
                               UCL_WRITE_ONLY);

  for (int i=0; i<lj_types*lj_types; i++)
    host_write[i]=0.0;

  coeff1.alloc(lj_types*lj_types,*(this->ucl_device),UCL_READ_ONLY);
  this->atom->type_pack4(ntypes,lj_types,coeff1,host_write,host_epsilon,
                         host_sigma,host_bornsolv1,host_bornsolv2);

  coeff2.alloc(lj_types*lj_types,*(this->ucl_device),UCL_READ_ONLY);
  this->atom->type_pack1(ntypes,lj_types,coeff2,host_write,host_offset);

  cutsq_sigma.alloc(lj_types*lj_types,*(this->ucl_device),UCL_READ_ONLY);
  this->atom->type_pack2(ntypes,lj_types,cutsq_sigma,host_write,host_cutsq,
                         host_sigma);

  UCL_H_Vec<double> dview;
  sp_lj.alloc(4,*(this->ucl_device),UCL_READ_ONLY);
  dview.view(host_special_lj,4,*(this->ucl_device));
  ucl_copy(sp_lj,dview,false);

  _allocated=true;
  this->_max_bytes=coeff1.row_bytes()+coeff2.row_bytes()
   +cutsq_sigma.row_bytes()+sp_lj.row_bytes();
  return 0;
}

template <class numtyp, class acctyp>
void BornsolvT::reinit(const int ntypes, double **host_epsilon, 
           double **host_sigma, double **host_bornsolv1,
           double **host_bornsolv2, double **host_offset) {

  // Allocate a host write buffer for data initialization
  UCL_H_Vec<numtyp> host_write(_lj_types*_lj_types*32,*(this->ucl_device),
                               UCL_WRITE_ONLY);

  for (int i=0; i<_lj_types*_lj_types; i++)
    host_write[i]=0.0;

  this->atom->type_pack4(ntypes,_lj_types,coeff1,host_write,host_epsilon,
                         host_sigma,host_bornsolv1,host_bornsolv2);
  this->atom->type_pack1(ntypes,_lj_types,coeff2,host_write,host_offset);
}

template <class numtyp, class acctyp>
void BornsolvT::clear() {
  if (!_allocated)
    return;
  _allocated=false;

  coeff1.clear();
  coeff2.clear();
  cutsq_sigma.clear();
  sp_lj.clear();
  this->clear_atomic();
}

template <class numtyp, class acctyp>
double BornsolvT::host_memory_usage() const {
  return this->host_memory_usage_atomic()+sizeof(Bornsolv<numtyp,acctyp>);
}

// ---------------------------------------------------------------------------
// Calculate energies, forces, and torques
// ---------------------------------------------------------------------------
template <class numtyp, class acctyp>
int BornsolvT::loop(const int eflag, const int vflag) {
  // Compute the block size and grid size to keep all cores busy
  const int BX=this->block_size();
  int GX=static_cast<int>(ceil(static_cast<double>(this->ans->inum())/
                               (BX/this->_threads_per_atom)));

  int ainum=this->ans->inum();
  int nbor_pitch=this->nbor->nbor_pitch();
  this->time_pair.start();
  if (shared_types) {
    this->k_pair_sel->set_size(GX,BX);
    this->k_pair_sel->run(&this->atom->x, &coeff1,&coeff2,
                          &cutsq_sigma, &sp_lj,
                          &this->nbor->dev_nbor,
                          &this->_nbor_data->begin(),
                          &this->ans->force, &this->ans->engv, &eflag, &vflag,
                          &ainum, &nbor_pitch, &this->_threads_per_atom);
  } else {
    this->k_pair.set_size(GX,BX);
    this->k_pair.run(&this->atom->x, &coeff1, &coeff2,
                     &cutsq_sigma, &_lj_types, &sp_lj,
                     &this->nbor->dev_nbor,
                     &this->_nbor_data->begin(), &this->ans->force,
                     &this->ans->engv, &eflag, &vflag, &ainum,
                     &nbor_pitch, &this->_threads_per_atom);
  }
  this->time_pair.stop();
  return GX;
}

template class Bornsolv<PRECISION,ACC_PRECISION>;
}