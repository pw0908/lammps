/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(bornsolv/kk,PairBornsolvKokkos<LMPDeviceType>);
PairStyle(bornsolv/kk/device,PairBornsolvKokkos<LMPDeviceType>);
PairStyle(bornsolv/kk/host,PairBornsolvKokkos<LMPHostType>);
// clang-format on
#else

// clang-format off
#ifndef LMP_PAIR_BORNSOLV_KOKKOS_H
#define LMP_PAIR_BORNSOLV_KOKKOS_H

#include "pair_kokkos.h"
#include "pair_bornsolv.h"
#include "neigh_list_kokkos.h"

namespace LAMMPS_NS {

template<class DeviceType>
class PairBornsolvKokkos : public PairBornsolv {
 public:
  enum {EnabledNeighFlags=FULL|HALFTHREAD|HALF};
  enum {COUL_FLAG=0};
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  PairBornsolvKokkos(class LAMMPS *);
  ~PairBornsolvKokkos() override;

  void compute(int, int) override;

  void settings(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;

  struct params_bornsolv{
    KOKKOS_INLINE_FUNCTION
    params_bornsolv() {cutsq=0;bornsolv1=0;bornsolv2=0;offset=0;};
    KOKKOS_INLINE_FUNCTION
    params_bornsolv(int /*i*/) {cutsq=0;bornsolv1=0;bornsolv2=0;offset=0;};
    F_FLOAT cutsq,bornsolv1,bornsolv2,offset;
  };

 protected:
  template<bool STACKPARAMS, class Specialisation>
  KOKKOS_INLINE_FUNCTION
  F_FLOAT compute_fpair(const F_FLOAT& rsq, const int& i, const int&j, const int& itype, const int& jtype) const;

  template<bool STACKPARAMS, class Specialisation>
  KOKKOS_INLINE_FUNCTION
  F_FLOAT compute_evdwl(const F_FLOAT& rsq, const int& i, const int&j, const int& itype, const int& jtype) const;

  template<bool STACKPARAMS, class Specialisation>
  KOKKOS_INLINE_FUNCTION
  F_FLOAT compute_ecoul(const F_FLOAT& /*rsq*/, const int& /*i*/, const int& /*j*/,
                        const int& /*itype*/, const int& /*jtype*/) const { return 0; }

  Kokkos::DualView<params_bornsolv**,Kokkos::LayoutRight,DeviceType> k_params;
  typename Kokkos::DualView<params_bornsolv**,Kokkos::LayoutRight,DeviceType>::t_dev_const_um params;
  params_bornsolv m_params[MAX_TYPES_STACKPARAMS+1][MAX_TYPES_STACKPARAMS+1];  // hardwired to space for 12 atom types
  F_FLOAT m_cutsq[MAX_TYPES_STACKPARAMS+1][MAX_TYPES_STACKPARAMS+1];
  typename AT::t_x_array_randomread x;
  typename AT::t_x_array c_x;
  typename AT::t_f_array f;
  typename AT::t_int_1d_randomread type;

  DAT::tdual_efloat_1d k_eatom;
  DAT::tdual_virial_array k_vatom;
  typename AT::t_efloat_1d d_eatom;
  typename AT::t_virial_array d_vatom;

  int newton_pair;
  double special_lj[4];

  typename AT::tdual_ffloat_2d k_cutsq;
  typename AT::t_ffloat_2d d_cutsq;


  int neighflag;
  int nlocal,nall,eflag,vflag;

  void allocate() override;
  friend struct PairComputeFunctor<PairBornsolvKokkos,FULL,true,0>;
  friend struct PairComputeFunctor<PairBornsolvKokkos,FULL,true,1>;
  friend struct PairComputeFunctor<PairBornsolvKokkos,HALF,true>;
  friend struct PairComputeFunctor<PairBornsolvKokkos,HALFTHREAD,true>;
  friend struct PairComputeFunctor<PairBornsolvKokkos,FULL,false,0>;
  friend struct PairComputeFunctor<PairBornsolvKokkos,FULL,false,1>;
  friend struct PairComputeFunctor<PairBornsolvKokkos,HALF,false>;
  friend struct PairComputeFunctor<PairBornsolvKokkos,HALFTHREAD,false>;
  friend EV_FLOAT pair_compute_neighlist<PairBornsolvKokkos,FULL,0>(PairBornsolvKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT pair_compute_neighlist<PairBornsolvKokkos,FULL,1>(PairBornsolvKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT pair_compute_neighlist<PairBornsolvKokkos,HALF>(PairBornsolvKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT pair_compute_neighlist<PairBornsolvKokkos,HALFTHREAD>(PairBornsolvKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT pair_compute<PairBornsolvKokkos>(PairBornsolvKokkos*,NeighListKokkos<DeviceType>*);
  friend void pair_virial_fdotr_compute<PairBornsolvKokkos>(PairBornsolvKokkos*);
};

}

#endif
#endif

