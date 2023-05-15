// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <sfm/types.h>

// define variables used in SVD3x3
#define SVD3X3_INITIALIZE                                                      \
  un &Sa11 = *(un *)(&s11);                                                    \
  un &Sa22 = *(un *)(&s22);                                                    \
  un &Sa33 = *(un *)(&s33);                                                    \
  un Sa21, Sa31, Sa12, Sa32, Sa13, Sa23;                                       \
  un Sc, Ss, Sch, Ssh;                                                         \
  un Stmp1, Stmp2, Stmp3, Stmp4, Stmp5;                                        \
  un Ss11, Ss21, Ss31, Ss22, Ss32, Ss33;                                       \
  un Sqvs, Sqvx, Sqvy, Sqvz;                                                   \
                                                                               \
  Sa11.f = a11;                                                                \
  Sa12.f = a12;                                                                \
  Sa13.f = a13;                                                                \
  Sa21.f = a21;                                                                \
  Sa22.f = a22;                                                                \
  Sa23.f = a23;                                                                \
  Sa31.f = a31;                                                                \
  Sa32.f = a32;                                                                \
  Sa33.f = a33;                                                                \
                                                                               \
  un &Su11 = *(un *)(&u11);                                                    \
  un &Su21 = *(un *)(&u21);                                                    \
  un &Su31 = *(un *)(&u31);                                                    \
  un &Su12 = *(un *)(&u12);                                                    \
  un &Su22 = *(un *)(&u22);                                                    \
  un &Su32 = *(un *)(&u32);                                                    \
  un &Su13 = *(un *)(&u13);                                                    \
  un &Su23 = *(un *)(&u23);                                                    \
  un &Su33 = *(un *)(&u33);                                                    \
                                                                               \
  un &Sv11 = *(un *)(&v11);                                                    \
  un &Sv21 = *(un *)(&v21);                                                    \
  un &Sv31 = *(un *)(&v31);                                                    \
  un &Sv12 = *(un *)(&v12);                                                    \
  un &Sv22 = *(un *)(&v22);                                                    \
  un &Sv32 = *(un *)(&v32);                                                    \
  un &Sv13 = *(un *)(&v13);                                                    \
  un &Sv23 = *(un *)(&v23);                                                    \
  un &Sv33 = *(un *)(&v33);

// compute A^T*A
#define SVD3X3_COMPUTE_ATA                                                     \
  Ss11.f = mul(Sa11.f, Sa11.f);                                                \
  Stmp1.f = mul(Sa21.f, Sa21.f);                                               \
  Ss11.f = add(Stmp1.f, Ss11.f);                                               \
  Stmp1.f = mul(Sa31.f, Sa31.f);                                               \
  Ss11.f = add(Stmp1.f, Ss11.f);                                               \
                                                                               \
  Ss21.f = mul(Sa12.f, Sa11.f);                                                \
  Stmp1.f = mul(Sa22.f, Sa21.f);                                               \
  Ss21.f = add(Stmp1.f, Ss21.f);                                               \
  Stmp1.f = mul(Sa32.f, Sa31.f);                                               \
  Ss21.f = add(Stmp1.f, Ss21.f);                                               \
                                                                               \
  Ss31.f = mul(Sa13.f, Sa11.f);                                                \
  Stmp1.f = mul(Sa23.f, Sa21.f);                                               \
  Ss31.f = add(Stmp1.f, Ss31.f);                                               \
  Stmp1.f = mul(Sa33.f, Sa31.f);                                               \
  Ss31.f = add(Stmp1.f, Ss31.f);                                               \
                                                                               \
  Ss22.f = mul(Sa12.f, Sa12.f);                                                \
  Stmp1.f = mul(Sa22.f, Sa22.f);                                               \
  Ss22.f = add(Stmp1.f, Ss22.f);                                               \
  Stmp1.f = mul(Sa32.f, Sa32.f);                                               \
  Ss22.f = add(Stmp1.f, Ss22.f);                                               \
                                                                               \
  Ss32.f = mul(Sa13.f, Sa12.f);                                                \
  Stmp1.f = mul(Sa23.f, Sa22.f);                                               \
  Ss32.f = add(Stmp1.f, Ss32.f);                                               \
  Stmp1.f = mul(Sa33.f, Sa32.f);                                               \
  Ss32.f = add(Stmp1.f, Ss32.f);                                               \
                                                                               \
  Ss33.f = mul(Sa13.f, Sa13.f);                                                \
  Stmp1.f = mul(Sa23.f, Sa23.f);                                               \
  Ss33.f = add(Stmp1.f, Ss33.f);                                               \
  Stmp1.f = mul(Sa33.f, Sa33.f);                                               \
  Ss33.f = add(Stmp1.f, Ss33.f);

#define SVD3X3_JACOBI_CONJUATION(SS11, SS21, SS31, SS22, SS32, SS33, SQVX,     \
                                 SQVY, SQVZ, STMP1, STMP2, STMP3)              \
  Ssh.f = mul(SS21.f, Sone_half);                                              \
  Stmp5.f = sub(SS11.f, SS22.f);                                               \
                                                                               \
  Stmp2.f = mul(Ssh.f, Ssh.f);                                                 \
  Stmp1.ui = (Stmp2.f >= Stiny_number) ? ff : 0;                               \
                                                                               \
  Ssh.ui = Stmp1.ui & Ssh.ui;                                                  \
  Sch.ui = Stmp1.ui & Stmp5.ui;                                                \
  Stmp2.ui = ~Stmp1.ui & Sone;                                                 \
  Sch.ui = Sch.ui | Stmp2.ui;                                                  \
                                                                               \
  Stmp1.f = mul(Ssh.f, Ssh.f);                                                 \
  Stmp2.f = mul(Sch.f, Sch.f);                                                 \
  Stmp3.f = add(Stmp1.f, Stmp2.f);                                             \
  Stmp4.f = rsqrt(Stmp3.f);                                                    \
                                                                               \
  Ssh.f = mul(Stmp4.f, Ssh.f);                                                 \
  Sch.f = mul(Stmp4.f, Sch.f);                                                 \
  Stmp1.f = Sfour_gamma_squared * Stmp1.f;                                     \
  Stmp1.ui = (Stmp2.f <= Stmp1.f) ? ff : 0;                                    \
                                                                               \
  Stmp2.ui = Ssine_pi_over_eight & Stmp1.ui;                                   \
  Ssh.ui = ~Stmp1.ui & Ssh.ui;                                                 \
  Ssh.ui = Ssh.ui | Stmp2.ui;                                                  \
  Stmp2.ui = Scosine_pi_over_eight & Stmp1.ui;                                 \
  Sch.ui = ~Stmp1.ui & Sch.ui;                                                 \
  Sch.ui = Sch.ui | Stmp2.ui;                                                  \
                                                                               \
  Stmp1.f = mul(Ssh.f, Ssh.f);                                                 \
  Stmp2.f = mul(Sch.f, Sch.f);                                                 \
  Sc.f = sub(Stmp2.f, Stmp1.f);                                                \
  Ss.f = mul(Sch.f, Ssh.f);                                                    \
  Ss.f = add(Ss.f, Ss.f);                                                      \
                                                                               \
  Stmp3.f = add(Stmp1.f, Stmp2.f);                                             \
  SS33.f = mul(SS33.f, Stmp3.f);                                               \
  SS31.f = mul(SS31.f, Stmp3.f);                                               \
  SS32.f = mul(SS32.f, Stmp3.f);                                               \
  SS33.f = mul(SS33.f, Stmp3.f);                                               \
                                                                               \
  Stmp1.f = mul(Ss.f, SS31.f);                                                 \
  Stmp2.f = mul(Ss.f, SS32.f);                                                 \
  SS31.f = mul(Sc.f, SS31.f);                                                  \
  SS32.f = mul(Sc.f, SS32.f);                                                  \
  SS31.f = add(Stmp2.f, SS31.f);                                               \
  SS32.f = sub(SS32.f, Stmp1.f);                                               \
                                                                               \
  Stmp2.f = mul(Ss.f, Ss.f);                                                   \
  Stmp1.f = mul(SS22.f, Stmp2.f);                                              \
  Stmp3.f = mul(SS11.f, Stmp2.f);                                              \
  Stmp4.f = mul(Sc.f, Sc.f);                                                   \
  SS11.f = mul(SS11.f, Stmp4.f);                                               \
  SS22.f = mul(SS22.f, Stmp4.f);                                               \
  SS11.f = add(SS11.f, Stmp1.f);                                               \
  SS22.f = add(SS22.f, Stmp3.f);                                               \
  Stmp4.f = sub(Stmp4.f, Stmp2.f);                                             \
  Stmp2.f = add(SS21.f, SS21.f);                                               \
  SS21.f = mul(SS21.f, Stmp4.f);                                               \
  Stmp4.f = mul(Sc.f, Ss.f);                                                   \
  Stmp2.f = mul(Stmp2.f, Stmp4.f);                                             \
  Stmp5.f = mul(Stmp5.f, Stmp4.f);                                             \
  SS11.f = add(SS11.f, Stmp2.f);                                               \
  SS21.f = sub(SS21.f, Stmp5.f);                                               \
  SS22.f = sub(SS22.f, Stmp2.f);                                               \
                                                                               \
  Stmp1.f = mul(Ssh.f, Sqvx.f);                                                \
  Stmp2.f = mul(Ssh.f, Sqvy.f);                                                \
  Stmp3.f = mul(Ssh.f, Sqvz.f);                                                \
  Ssh.f = mul(Ssh.f, Sqvs.f);                                                  \
                                                                               \
  Sqvs.f = mul(Sch.f, Sqvs.f);                                                 \
  Sqvx.f = mul(Sch.f, Sqvx.f);                                                 \
  Sqvy.f = mul(Sch.f, Sqvy.f);                                                 \
  Sqvz.f = mul(Sch.f, Sqvz.f);                                                 \
                                                                               \
  SQVZ.f = add(SQVZ.f, Ssh.f);                                                 \
  Sqvs.f = sub(Sqvs.f, STMP3.f);                                               \
  SQVX.f = add(SQVX.f, STMP2.f);                                               \
  SQVY.f = sub(SQVY.f, STMP1.f);

#define SVD3X3_COMPUTE_MATRIX_V                                                \
  Stmp2.f = mul(Sqvs.f, Sqvs.f);                                               \
  Stmp1.f = mul(Sqvx.f, Sqvx.f);                                               \
  Stmp2.f = add(Stmp1.f, Stmp2.f);                                             \
  Stmp1.f = mul(Sqvy.f, Sqvy.f);                                               \
  Stmp2.f = add(Stmp1.f, Stmp2.f);                                             \
  Stmp1.f = mul(Sqvz.f, Sqvz.f);                                               \
  Stmp2.f = add(Stmp1.f, Stmp2.f);                                             \
                                                                               \
  Stmp1.f = rsqrt(Stmp2.f);                                                    \
  Stmp4.f = mul(Stmp1.f, Sone_half);                                           \
  Stmp3.f = mul(Stmp1.f, Stmp4.f);                                             \
  Stmp3.f = mul(Stmp1.f, Stmp3.f);                                             \
  Stmp3.f = mul(Stmp2.f, Stmp3.f);                                             \
  Stmp1.f = add(Stmp1.f, Stmp4.f);                                             \
  Stmp1.f = sub(Stmp1.f, Stmp3.f);                                             \
                                                                               \
  Sqvs.f = mul(Sqvs.f, Stmp1.f);                                               \
  Sqvx.f = mul(Sqvx.f, Stmp1.f);                                               \
  Sqvy.f = mul(Sqvy.f, Stmp1.f);                                               \
  Sqvz.f = mul(Sqvz.f, Stmp1.f);                                               \
  Stmp1.f = mul(Sqvx.f, Sqvx.f);                                               \
  Stmp2.f = mul(Sqvy.f, Sqvy.f);                                               \
  Stmp3.f = mul(Sqvz.f, Sqvz.f);                                               \
  Sv11.f = mul(Sqvs.f, Sqvs.f);                                                \
  Sv22.f = sub(Sv11.f, Stmp1.f);                                               \
  Sv33.f = sub(Sv22.f, Stmp2.f);                                               \
  Sv33.f = add(Sv33.f, Stmp3.f);                                               \
  Sv22.f = add(Sv22.f, Stmp2.f);                                               \
  Sv22.f = sub(Sv22.f, Stmp3.f);                                               \
  Sv11.f = add(Sv11.f, Stmp1.f);                                               \
  Sv11.f = sub(Sv11.f, Stmp2.f);                                               \
  Sv11.f = sub(Sv11.f, Stmp3.f);                                               \
  Stmp1.f = add(Sqvx.f, Sqvx.f);                                               \
  Stmp2.f = add(Sqvy.f, Sqvy.f);                                               \
  Stmp3.f = add(Sqvz.f, Sqvz.f);                                               \
  Sv32.f = mul(Sqvs.f, Stmp1.f);                                               \
  Sv13.f = mul(Sqvs.f, Stmp2.f);                                               \
  Sv21.f = mul(Sqvs.f, Stmp3.f);                                               \
  Stmp1.f = mul(Sqvy.f, Stmp1.f);                                              \
  Stmp2.f = mul(Sqvz.f, Stmp2.f);                                              \
  Stmp3.f = mul(Sqvx.f, Stmp3.f);                                              \
  Sv12.f = sub(Stmp1.f, Sv21.f);                                               \
  Sv23.f = sub(Stmp2.f, Sv32.f);                                               \
  Sv31.f = sub(Stmp3.f, Sv13.f);                                               \
  Sv21.f = add(Stmp1.f, Sv21.f);                                               \
  Sv32.f = add(Stmp2.f, Sv32.f);                                               \
  Sv13.f = add(Stmp3.f, Sv13.f);

#define SVD3X3_MULTIPLY_WITH_V                                                 \
  Stmp2.f = Sa12.f;                                                            \
  Stmp3.f = Sa13.f;                                                            \
  Sa12.f = mul(Sv12.f, Sa11.f);                                                \
  Sa13.f = mul(Sv13.f, Sa11.f);                                                \
  Sa11.f = mul(Sv11.f, Sa11.f);                                                \
  Stmp1.f = mul(Sv21.f, Stmp2.f);                                              \
  Sa11.f = add(Sa11.f, Stmp1.f);                                               \
  Stmp1.f = mul(Sv31.f, Stmp3.f);                                              \
  Sa11.f = add(Sa11.f, Stmp1.f);                                               \
  Stmp1.f = mul(Sv22.f, Stmp2.f);                                              \
  Sa12.f = add(Sa12.f, Stmp1.f);                                               \
  Stmp1.f = mul(Sv32.f, Stmp3.f);                                              \
  Sa12.f = add(Sa12.f, Stmp1.f);                                               \
  Stmp1.f = mul(Sv23.f, Stmp2.f);                                              \
  Sa13.f = add(Sa13.f, Stmp1.f);                                               \
  Stmp1.f = mul(Sv33.f, Stmp3.f);                                              \
  Sa13.f = add(Sa13.f, Stmp1.f);                                               \
                                                                               \
  Stmp2.f = Sa22.f;                                                            \
  Stmp3.f = Sa23.f;                                                            \
  Sa22.f = mul(Sv12.f, Sa21.f);                                                \
  Sa23.f = mul(Sv13.f, Sa21.f);                                                \
  Sa21.f = mul(Sv11.f, Sa21.f);                                                \
  Stmp1.f = mul(Sv21.f, Stmp2.f);                                              \
  Sa21.f = add(Sa21.f, Stmp1.f);                                               \
  Stmp1.f = mul(Sv31.f, Stmp3.f);                                              \
  Sa21.f = add(Sa21.f, Stmp1.f);                                               \
  Stmp1.f = mul(Sv22.f, Stmp2.f);                                              \
  Sa22.f = add(Sa22.f, Stmp1.f);                                               \
  Stmp1.f = mul(Sv32.f, Stmp3.f);                                              \
  Sa22.f = add(Sa22.f, Stmp1.f);                                               \
  Stmp1.f = mul(Sv23.f, Stmp2.f);                                              \
  Sa23.f = add(Sa23.f, Stmp1.f);                                               \
  Stmp1.f = mul(Sv33.f, Stmp3.f);                                              \
  Sa23.f = add(Sa23.f, Stmp1.f);                                               \
                                                                               \
  Stmp2.f = Sa32.f;                                                            \
  Stmp3.f = Sa33.f;                                                            \
  Sa32.f = mul(Sv12.f, Sa31.f);                                                \
  Sa33.f = mul(Sv13.f, Sa31.f);                                                \
  Sa31.f = mul(Sv11.f, Sa31.f);                                                \
  Stmp1.f = mul(Sv21.f, Stmp2.f);                                              \
  Sa31.f = add(Sa31.f, Stmp1.f);                                               \
  Stmp1.f = mul(Sv31.f, Stmp3.f);                                              \
  Sa31.f = add(Sa31.f, Stmp1.f);                                               \
  Stmp1.f = mul(Sv22.f, Stmp2.f);                                              \
  Sa32.f = add(Sa32.f, Stmp1.f);                                               \
  Stmp1.f = mul(Sv32.f, Stmp3.f);                                              \
  Sa32.f = add(Sa32.f, Stmp1.f);                                               \
  Stmp1.f = mul(Sv23.f, Stmp2.f);                                              \
  Sa33.f = add(Sa33.f, Stmp1.f);                                               \
  Stmp1.f = mul(Sv33.f, Stmp3.f);                                              \
  Sa33.f = add(Sa33.f, Stmp1.f);

#define SVD3X3_SORT_SINGULAR_VALUES                                            \
  Stmp1.f = mul(Sa11.f, Sa11.f);                                               \
  Stmp4.f = mul(Sa21.f, Sa21.f);                                               \
  Stmp1.f = add(Stmp1.f, Stmp4.f);                                             \
  Stmp4.f = mul(Sa31.f, Sa31.f);                                               \
  Stmp1.f = add(Stmp1.f, Stmp4.f);                                             \
                                                                               \
  Stmp2.f = mul(Sa12.f, Sa12.f);                                               \
  Stmp4.f = mul(Sa22.f, Sa22.f);                                               \
  Stmp2.f = add(Stmp2.f, Stmp4.f);                                             \
  Stmp4.f = mul(Sa32.f, Sa32.f);                                               \
  Stmp2.f = add(Stmp2.f, Stmp4.f);                                             \
                                                                               \
  Stmp3.f = mul(Sa13.f, Sa13.f);                                               \
  Stmp4.f = mul(Sa23.f, Sa23.f);                                               \
  Stmp3.f = add(Stmp3.f, Stmp4.f);                                             \
  Stmp4.f = mul(Sa33.f, Sa33.f);                                               \
  Stmp3.f = add(Stmp3.f, Stmp4.f);                                             \
                                                                               \
  Stmp4.ui = (Stmp1.f < Stmp2.f) ? ff : 0;                                     \
  Stmp5.ui = Sa11.ui ^ Sa12.ui;                                                \
  Stmp5.ui = Stmp5.ui & Stmp4.ui;                                              \
  Sa11.ui = Sa11.ui ^ Stmp5.ui;                                                \
  Sa12.ui = Sa12.ui ^ Stmp5.ui;                                                \
                                                                               \
  Stmp5.ui = Sa21.ui ^ Sa22.ui;                                                \
  Stmp5.ui = Stmp5.ui & Stmp4.ui;                                              \
  Sa21.ui = Sa21.ui ^ Stmp5.ui;                                                \
  Sa22.ui = Sa22.ui ^ Stmp5.ui;                                                \
                                                                               \
  Stmp5.ui = Sa31.ui ^ Sa32.ui;                                                \
  Stmp5.ui = Stmp5.ui & Stmp4.ui;                                              \
  Sa31.ui = Sa31.ui ^ Stmp5.ui;                                                \
  Sa32.ui = Sa32.ui ^ Stmp5.ui;                                                \
                                                                               \
  Stmp5.ui = Sv11.ui ^ Sv12.ui;                                                \
  Stmp5.ui = Stmp5.ui & Stmp4.ui;                                              \
  Sv11.ui = Sv11.ui ^ Stmp5.ui;                                                \
  Sv12.ui = Sv12.ui ^ Stmp5.ui;                                                \
                                                                               \
  Stmp5.ui = Sv21.ui ^ Sv22.ui;                                                \
  Stmp5.ui = Stmp5.ui & Stmp4.ui;                                              \
  Sv21.ui = Sv21.ui ^ Stmp5.ui;                                                \
  Sv22.ui = Sv22.ui ^ Stmp5.ui;                                                \
                                                                               \
  Stmp5.ui = Sv31.ui ^ Sv32.ui;                                                \
  Stmp5.ui = Stmp5.ui & Stmp4.ui;                                              \
  Sv31.ui = Sv31.ui ^ Stmp5.ui;                                                \
  Sv32.ui = Sv32.ui ^ Stmp5.ui;                                                \
                                                                               \
  Stmp5.ui = Stmp1.ui ^ Stmp2.ui;                                              \
  Stmp5.ui = Stmp5.ui & Stmp4.ui;                                              \
  Stmp1.ui = Stmp1.ui ^ Stmp5.ui;                                              \
  Stmp2.ui = Stmp2.ui ^ Stmp5.ui;                                              \
                                                                               \
  Stmp5.f = -2.;                                                               \
  Stmp5.ui = Stmp5.ui & Stmp4.ui;                                              \
  Stmp4.f = 1.;                                                                \
  Stmp4.f = add(Stmp4.f, Stmp5.f);                                             \
                                                                               \
  Sa12.f = mul(Sa12.f, Stmp4.f);                                               \
  Sa22.f = mul(Sa22.f, Stmp4.f);                                               \
  Sa32.f = mul(Sa32.f, Stmp4.f);                                               \
                                                                               \
  Sv12.f = mul(Sv12.f, Stmp4.f);                                               \
  Sv22.f = mul(Sv22.f, Stmp4.f);                                               \
  Sv32.f = mul(Sv32.f, Stmp4.f);                                               \
                                                                               \
  Stmp4.ui = (Stmp1.f < Stmp3.f) ? ff : 0;                                     \
  Stmp5.ui = Sa11.ui ^ Sa13.ui;                                                \
  Stmp5.ui = Stmp5.ui & Stmp4.ui;                                              \
  Sa11.ui = Sa11.ui ^ Stmp5.ui;                                                \
  Sa13.ui = Sa13.ui ^ Stmp5.ui;                                                \
                                                                               \
  Stmp5.ui = Sa21.ui ^ Sa23.ui;                                                \
  Stmp5.ui = Stmp5.ui & Stmp4.ui;                                              \
  Sa21.ui = Sa21.ui ^ Stmp5.ui;                                                \
  Sa23.ui = Sa23.ui ^ Stmp5.ui;                                                \
                                                                               \
  Stmp5.ui = Sa31.ui ^ Sa33.ui;                                                \
  Stmp5.ui = Stmp5.ui & Stmp4.ui;                                              \
  Sa31.ui = Sa31.ui ^ Stmp5.ui;                                                \
  Sa33.ui = Sa33.ui ^ Stmp5.ui;                                                \
                                                                               \
  Stmp5.ui = Sv11.ui ^ Sv13.ui;                                                \
  Stmp5.ui = Stmp5.ui & Stmp4.ui;                                              \
  Sv11.ui = Sv11.ui ^ Stmp5.ui;                                                \
  Sv13.ui = Sv13.ui ^ Stmp5.ui;                                                \
                                                                               \
  Stmp5.ui = Sv21.ui ^ Sv23.ui;                                                \
  Stmp5.ui = Stmp5.ui & Stmp4.ui;                                              \
  Sv21.ui = Sv21.ui ^ Stmp5.ui;                                                \
  Sv23.ui = Sv23.ui ^ Stmp5.ui;                                                \
                                                                               \
  Stmp5.ui = Sv31.ui ^ Sv33.ui;                                                \
  Stmp5.ui = Stmp5.ui & Stmp4.ui;                                              \
  Sv31.ui = Sv31.ui ^ Stmp5.ui;                                                \
  Sv33.ui = Sv33.ui ^ Stmp5.ui;                                                \
                                                                               \
  Stmp5.ui = Stmp1.ui ^ Stmp3.ui;                                              \
  Stmp5.ui = Stmp5.ui & Stmp4.ui;                                              \
  Stmp1.ui = Stmp1.ui ^ Stmp5.ui;                                              \
  Stmp3.ui = Stmp3.ui ^ Stmp5.ui;                                              \
                                                                               \
  Stmp5.f = -2.;                                                               \
  Stmp5.ui = Stmp5.ui & Stmp4.ui;                                              \
  Stmp4.f = 1.;                                                                \
  Stmp4.f = add(Stmp4.f, Stmp5.f);                                             \
                                                                               \
  Sa11.f = mul(Sa11.f, Stmp4.f);                                               \
  Sa21.f = mul(Sa21.f, Stmp4.f);                                               \
  Sa31.f = mul(Sa31.f, Stmp4.f);                                               \
                                                                               \
  Sv11.f = mul(Sv11.f, Stmp4.f);                                               \
  Sv21.f = mul(Sv21.f, Stmp4.f);                                               \
  Sv31.f = mul(Sv31.f, Stmp4.f);                                               \
                                                                               \
  Stmp4.ui = (Stmp2.f < Stmp3.f) ? ff : 0;                                     \
  Stmp5.ui = Sa12.ui ^ Sa13.ui;                                                \
  Stmp5.ui = Stmp5.ui & Stmp4.ui;                                              \
  Sa12.ui = Sa12.ui ^ Stmp5.ui;                                                \
  Sa13.ui = Sa13.ui ^ Stmp5.ui;                                                \
                                                                               \
  Stmp5.ui = Sa22.ui ^ Sa23.ui;                                                \
  Stmp5.ui = Stmp5.ui & Stmp4.ui;                                              \
  Sa22.ui = Sa22.ui ^ Stmp5.ui;                                                \
  Sa23.ui = Sa23.ui ^ Stmp5.ui;                                                \
                                                                               \
  Stmp5.ui = Sa32.ui ^ Sa33.ui;                                                \
  Stmp5.ui = Stmp5.ui & Stmp4.ui;                                              \
  Sa32.ui = Sa32.ui ^ Stmp5.ui;                                                \
  Sa33.ui = Sa33.ui ^ Stmp5.ui;                                                \
                                                                               \
  Stmp5.ui = Sv12.ui ^ Sv13.ui;                                                \
  Stmp5.ui = Stmp5.ui & Stmp4.ui;                                              \
  Sv12.ui = Sv12.ui ^ Stmp5.ui;                                                \
  Sv13.ui = Sv13.ui ^ Stmp5.ui;                                                \
                                                                               \
  Stmp5.ui = Sv22.ui ^ Sv23.ui;                                                \
  Stmp5.ui = Stmp5.ui & Stmp4.ui;                                              \
  Sv22.ui = Sv22.ui ^ Stmp5.ui;                                                \
  Sv23.ui = Sv23.ui ^ Stmp5.ui;                                                \
                                                                               \
  Stmp5.ui = Sv32.ui ^ Sv33.ui;                                                \
  Stmp5.ui = Stmp5.ui & Stmp4.ui;                                              \
  Sv32.ui = Sv32.ui ^ Stmp5.ui;                                                \
  Sv33.ui = Sv33.ui ^ Stmp5.ui;                                                \
                                                                               \
  Stmp5.ui = Stmp2.ui ^ Stmp3.ui;                                              \
  Stmp5.ui = Stmp5.ui & Stmp4.ui;                                              \
  Stmp2.ui = Stmp2.ui ^ Stmp5.ui;                                              \
  Stmp3.ui = Stmp3.ui ^ Stmp5.ui;                                              \
                                                                               \
  Stmp5.f = -2.;                                                               \
  Stmp5.ui = Stmp5.ui & Stmp4.ui;                                              \
  Stmp4.f = 1.;                                                                \
  Stmp4.f = add(Stmp4.f, Stmp5.f);                                             \
                                                                               \
  Sa13.f = mul(Sa13.f, Stmp4.f);                                               \
  Sa23.f = mul(Sa23.f, Stmp4.f);                                               \
  Sa33.f = mul(Sa33.f, Stmp4.f);                                               \
                                                                               \
  Sv13.f = mul(Sv13.f, Stmp4.f);                                               \
  Sv23.f = mul(Sv23.f, Stmp4.f);                                               \
  Sv33.f = mul(Sv33.f, Stmp4.f);

#define SVD3X3_QR(SAPIVOT, SANPIVOT, SA11, SA21, SA12, SA22, SA13, SA23, SU11, \
                  SU12, SU21, SU22, SU31, SU32)                                \
  Ssh.f = mul(SANPIVOT.f, SANPIVOT.f);                                         \
  Ssh.ui = (Ssh.f >= Ssmall_number) ? ff : 0;                                  \
  Ssh.ui = Ssh.ui & SANPIVOT.ui;                                               \
                                                                               \
  Stmp5.f = 0.;                                                                \
  Sch.f = sub(Stmp5.f, SAPIVOT.f);                                             \
  Sch.f = max(Sch.f, SAPIVOT.f);                                               \
  Sch.f = max(Sch.f, Ssmall_number);                                           \
  Stmp5.ui = (SAPIVOT.f >= Stmp5.f) ? ff : 0;                                  \
                                                                               \
  Stmp1.f = mul(Sch.f, Sch.f);                                                 \
  Stmp2.f = mul(Ssh.f, Ssh.f);                                                 \
  Stmp2.f = add(Stmp1.f, Stmp2.f);                                             \
  Stmp1.f = rsqrt(Stmp2.f);                                                    \
                                                                               \
  Stmp4.f = mul(Stmp1.f, Sone_half);                                           \
  Stmp3.f = mul(Stmp1.f, Stmp4.f);                                             \
  Stmp3.f = mul(Stmp1.f, Stmp3.f);                                             \
  Stmp3.f = mul(Stmp2.f, Stmp3.f);                                             \
  Stmp1.f = add(Stmp1.f, Stmp4.f);                                             \
  Stmp1.f = sub(Stmp1.f, Stmp3.f);                                             \
  Stmp1.f = mul(Stmp1.f, Stmp2.f);                                             \
                                                                               \
  Sch.f = add(Sch.f, Stmp1.f);                                                 \
                                                                               \
  Stmp1.ui = ~Stmp5.ui & Ssh.ui;                                               \
  Stmp2.ui = ~Stmp5.ui & Sch.ui;                                               \
  Sch.ui = Stmp5.ui & Sch.ui;                                                  \
  Ssh.ui = Stmp5.ui & Ssh.ui;                                                  \
  Sch.ui = Sch.ui | Stmp1.ui;                                                  \
  Ssh.ui = Ssh.ui | Stmp2.ui;                                                  \
                                                                               \
  Stmp1.f = mul(Sch.f, Sch.f);                                                 \
  Stmp2.f = mul(Ssh.f, Ssh.f);                                                 \
  Stmp2.f = add(Stmp1.f, Stmp2.f);                                             \
  Stmp1.f = rsqrt(Stmp2.f);                                                    \
                                                                               \
  Stmp4.f = mul(Stmp1.f, Sone_half);                                           \
  Stmp3.f = mul(Stmp1.f, Stmp4.f);                                             \
  Stmp3.f = mul(Stmp1.f, Stmp3.f);                                             \
  Stmp3.f = mul(Stmp2.f, Stmp3.f);                                             \
  Stmp1.f = add(Stmp1.f, Stmp4.f);                                             \
  Stmp1.f = sub(Stmp1.f, Stmp3.f);                                             \
                                                                               \
  Sch.f = mul(Sch.f, Stmp1.f);                                                 \
  Ssh.f = mul(Ssh.f, Stmp1.f);                                                 \
                                                                               \
  Sc.f = mul(Sch.f, Sch.f);                                                    \
  Ss.f = mul(Ssh.f, Ssh.f);                                                    \
  Sc.f = sub(Sc.f, Ss.f);                                                      \
  Ss.f = mul(Ssh.f, Sch.f);                                                    \
  Ss.f = add(Ss.f, Ss.f);                                                      \
                                                                               \
  Stmp1.f = mul(Ss.f, SA11.f);                                                 \
  Stmp2.f = mul(Ss.f, SA21.f);                                                 \
  SA11.f = mul(Sc.f, SA11.f);                                                  \
  SA21.f = mul(Sc.f, SA21.f);                                                  \
  SA11.f = add(SA11.f, Stmp2.f);                                               \
  SA21.f = sub(SA21.f, Stmp1.f);                                               \
                                                                               \
  Stmp1.f = mul(Ss.f, SA12.f);                                                 \
  Stmp2.f = mul(Ss.f, SA22.f);                                                 \
  SA12.f = mul(Sc.f, SA12.f);                                                  \
  SA22.f = mul(Sc.f, SA22.f);                                                  \
  SA12.f = add(SA12.f, Stmp2.f);                                               \
  SA22.f = sub(SA22.f, Stmp1.f);                                               \
                                                                               \
  Stmp1.f = mul(Ss.f, SA13.f);                                                 \
  Stmp2.f = mul(Ss.f, SA23.f);                                                 \
  SA13.f = mul(Sc.f, SA13.f);                                                  \
  SA23.f = mul(Sc.f, SA23.f);                                                  \
  SA13.f = add(SA13.f, Stmp2.f);                                               \
  SA23.f = sub(SA23.f, Stmp1.f);                                               \
                                                                               \
  Stmp1.f = mul(Ss.f, SU11.f);                                                 \
  Stmp2.f = mul(Ss.f, SU12.f);                                                 \
  SU11.f = mul(Sc.f, SU11.f);                                                  \
  SU12.f = mul(Sc.f, SU12.f);                                                  \
  SU11.f = add(SU11.f, Stmp2.f);                                               \
  SU12.f = sub(SU12.f, Stmp1.f);                                               \
                                                                               \
  Stmp1.f = mul(Ss.f, SU21.f);                                                 \
  Stmp2.f = mul(Ss.f, SU22.f);                                                 \
  SU21.f = mul(Sc.f, SU21.f);                                                  \
  SU22.f = mul(Sc.f, SU22.f);                                                  \
  SU21.f = add(SU21.f, Stmp2.f);                                               \
  SU22.f = sub(SU22.f, Stmp1.f);                                               \
                                                                               \
  Stmp1.f = mul(Ss.f, SU31.f);                                                 \
  Stmp2.f = mul(Ss.f, SU32.f);                                                 \
  SU31.f = mul(Sc.f, SU31.f);                                                  \
  SU32.f = mul(Sc.f, SU32.f);                                                  \
  SU31.f = add(SU31.f, Stmp2.f);                                               \
  SU32.f = sub(SU32.f, Stmp1.f);

namespace sfm {
namespace utils {
namespace internal {
__device__ __forceinline__ void
svd3x3(const float &a11, const float &a12, const float &a13, const float &a21,
       const float &a22, const float &a23, const float &a31, const float &a32,
       const float &a33, // input A
       float &u11, float &u12, float &u13, float &u21, float &u22, float &u23,
       float &u31, float &u32, float &u33, // output U
       float &s11, float &s22, float &s33, // output S
       float &v11, float &v12, float &v13, float &v21, float &v22, float &v23,
       float &v31, float &v32, float &v33 // output V
) {
#define un unf
#define Sone 1065353216u
#define Ssine_pi_over_eight 1053028117u
#define Scosine_pi_over_eight 1064076127u
#define Sone_half 0.5f
#define Ssmall_number 1.e-12f
#define Stiny_number 1.e-20f
#define Sfour_gamma_squared 5.8284271247461898f
#define ff 0xffffffff
#define add(a, b) __fadd_rn(a, b)
#define sub(a, b) __fsub_rn(a, b)
#define mul(a, b) __fmul_rn(a, b)
#define rsqrt(a) __frsqrt_rn(a)

  un &Sa11 = *(un *)(&s11);
  un &Sa22 = *(un *)(&s22);
  un &Sa33 = *(un *)(&s33);
  un Sa21, Sa31, Sa12, Sa32, Sa13, Sa23;
  un Sc, Ss, Sch, Ssh;
  un Stmp1, Stmp2, Stmp3, Stmp4, Stmp5;
  un Ss11, Ss21, Ss31, Ss22, Ss32, Ss33;
  un Sqvs, Sqvx, Sqvy, Sqvz;

  Sa11.f = a11;
  Sa12.f = a12;
  Sa13.f = a13;
  Sa21.f = a21;
  Sa22.f = a22;
  Sa23.f = a23;
  Sa31.f = a31;
  Sa32.f = a32;
  Sa33.f = a33;

  un &Su11 = *(un *)(&u11);
  un &Su21 = *(un *)(&u21);
  un &Su31 = *(un *)(&u31);
  un &Su12 = *(un *)(&u12);
  un &Su22 = *(un *)(&u22);
  un &Su32 = *(un *)(&u32);
  un &Su13 = *(un *)(&u13);
  un &Su23 = *(un *)(&u23);
  un &Su33 = *(un *)(&u33);

  un &Sv11 = *(un *)(&v11);
  un &Sv21 = *(un *)(&v21);
  un &Sv31 = *(un *)(&v31);
  un &Sv12 = *(un *)(&v12);
  un &Sv22 = *(un *)(&v22);
  un &Sv32 = *(un *)(&v32);
  un &Sv13 = *(un *)(&v13);
  un &Sv23 = *(un *)(&v23);
  un &Sv33 = *(un *)(&v33);

  SVD3X3_COMPUTE_ATA

#if DEBUG
  printf("%f %f %f\n", Ss11.f, Ss21.f, Ss31.f);
  printf("%f %f %f\n", Ss21.f, Ss22.f, Ss32.f);
  printf("%f %f %f\n", Ss31.f, Ss32.f, Ss33.f);
  printf("\n");
#endif

  Sqvs.f = 1.f;
  Sqvx.f = 0.f;
  Sqvy.f = 0.f;
  Sqvz.f = 0.f;

  for (int i = 0; i < 8; i++) {
    // First Jacobi conjugation
    SVD3X3_JACOBI_CONJUATION(Ss11, Ss21, Ss31, Ss22, Ss32, Ss33, Sqvx, Sqvy,
                             Sqvz, Stmp1, Stmp2, Stmp3)
#if DEBUG
    printf("%f %f %f\n", Sqvx.f, Sqvy.f, Sqvz.f);
    printf("\n");
#endif

    // Second Jacobi conjugation
    SVD3X3_JACOBI_CONJUATION(Ss22, Ss32, Ss21, Ss33, Ss31, Ss11, Sqvy, Sqvz,
                             Sqvx, Stmp2, Stmp3, Stmp1)
#if DEBUG
    printf("%f %f %f\n", Sqvx.f, Sqvy.f, Sqvz.f);
    printf("\n");
#endif

    // Third Jacobi conjugation
    SVD3X3_JACOBI_CONJUATION(Ss33, Ss31, Ss32, Ss11, Ss21, Ss22, Sqvz, Sqvx,
                             Sqvy, Stmp3, Stmp1, Stmp2)
#if DEBUG
    printf("%f %f %f\n", Sqvx.f, Sqvy.f, Sqvz.f);
    printf("\n");
#endif
  }

  SVD3X3_COMPUTE_MATRIX_V

#if DEBUG
  printf("%f %f %f\n", Sv11.f, Sv21.f, Sv31.f);
  printf("%f %f %f\n", Sv21.f, Sv22.f, Sv32.f);
  printf("%f %f %f\n", Sv31.f, Sv32.f, Sv33.f);
  printf("\n");
#endif

  SVD3X3_MULTIPLY_WITH_V

  SVD3X3_SORT_SINGULAR_VALUES

  Su11.f = 1.;
  Su12.f = 0.;
  Su13.f = 0.;
  Su21.f = 0.;
  Su22.f = 1.;
  Su23.f = 0.;
  Su31.f = 0.;
  Su32.f = 0.;
  Su33.f = 1.;

  // First Givens rotation
  SVD3X3_QR(Sa11, Sa21, Sa11, Sa21, Sa12, Sa22, Sa13, Sa23, Su11, Su12, Su21,
            Su22, Su31, Su32)

  // Second Givens rotation
  SVD3X3_QR(Sa11, Sa31, Sa11, Sa31, Sa12, Sa32, Sa13, Sa33, Su11, Su13, Su21,
            Su23, Su31, Su33)

  // Third Givens Rotation
  SVD3X3_QR(Sa22, Sa32, Sa21, Sa31, Sa22, Sa32, Sa23, Sa33, Su12, Su13, Su22,
            Su23, Su32, Su33)

#undef un
#undef Sone
#undef Ssine_pi_over_eight
#undef Scosine_pi_over_eight
#undef Sone_half
#undef Ssmall_number
#undef Stiny_number
#undef Sfour_gamma_squared
#undef ff
#undef add
#undef sub
#undef mul
#undef rsqrt
}

__device__ __forceinline__ void
svd3x3(const double &a11, const double &a12, const double &a13,
       const double &a21, const double &a22, const double &a23,
       const double &a31, const double &a32,
       const double &a33, // input A
       double &u11, double &u12, double &u13, double &u21, double &u22,
       double &u23, double &u31, double &u32, double &u33, // output U
       double &s11, double &s22, double &s33,              // output S
       double &v11, double &v12, double &v13, double &v21, double &v22,
       double &v23, double &v31, double &v32, double &v33 // output V
) {
#define un und
#define Sone 4607182418800017408u
#define Ssine_pi_over_eight 4600565431771507043u
#define Scosine_pi_over_eight 4606496786581982534u
#define Sone_half 0.5
#define Ssmall_number 1.e-16
#define Stiny_number 1.e-32
#define Sfour_gamma_squared 5.8284271247461898
#define ff 0xffffffffffffffff
#define add(a, b) __dadd_rn(a, b)
#define sub(a, b) __dsub_rn(a, b)
#define mul(a, b) __dmul_rn(a, b)
#define rsqrt(a) (rsqrt(a))

  un &Sa11 = *(un *)(&s11);
  un &Sa22 = *(un *)(&s22);
  un &Sa33 = *(un *)(&s33);
  un Sa21, Sa31, Sa12, Sa32, Sa13, Sa23;
  un Sc, Ss, Sch, Ssh;
  un Stmp1, Stmp2, Stmp3, Stmp4, Stmp5;
  un Ss11, Ss21, Ss31, Ss22, Ss32, Ss33;
  un Sqvs, Sqvx, Sqvy, Sqvz;

  Sa11.f = a11;
  Sa12.f = a12;
  Sa13.f = a13;
  Sa21.f = a21;
  Sa22.f = a22;
  Sa23.f = a23;
  Sa31.f = a31;
  Sa32.f = a32;
  Sa33.f = a33;

  un &Su11 = *(un *)(&u11);
  un &Su21 = *(un *)(&u21);
  un &Su31 = *(un *)(&u31);
  un &Su12 = *(un *)(&u12);
  un &Su22 = *(un *)(&u22);
  un &Su32 = *(un *)(&u32);
  un &Su13 = *(un *)(&u13);
  un &Su23 = *(un *)(&u23);
  un &Su33 = *(un *)(&u33);

  un &Sv11 = *(un *)(&v11);
  un &Sv21 = *(un *)(&v21);
  un &Sv31 = *(un *)(&v31);
  un &Sv12 = *(un *)(&v12);
  un &Sv22 = *(un *)(&v22);
  un &Sv32 = *(un *)(&v32);
  un &Sv13 = *(un *)(&v13);
  un &Sv23 = *(un *)(&v23);
  un &Sv33 = *(un *)(&v33);

  SVD3X3_COMPUTE_ATA

#if DEBUG
  printf("%f %f %f\n", Ss11.f, Ss21.f, Ss31.f);
  printf("%f %f %f\n", Ss21.f, Ss22.f, Ss32.f);
  printf("%f %f %f\n", Ss31.f, Ss32.f, Ss33.f);
  printf("\n");
#endif

  Sqvs.f = 1.0;
  Sqvx.f = 0.0;
  Sqvy.f = 0.0;
  Sqvz.f = 0.0;

  for (int i = 0; i < 8; i++) {
    // First Jacobi conjugation
    SVD3X3_JACOBI_CONJUATION(Ss11, Ss21, Ss31, Ss22, Ss32, Ss33, Sqvx, Sqvy,
                             Sqvz, Stmp1, Stmp2, Stmp3)
#if DEBUG
    printf("%f %f %f\n", Sqvx.f, Sqvy.f, Sqvz.f);
    printf("\n");
#endif
    // Second Jacobi conjugation
    SVD3X3_JACOBI_CONJUATION(Ss22, Ss32, Ss21, Ss33, Ss31, Ss11, Sqvy, Sqvz,
                             Sqvx, Stmp2, Stmp3, Stmp1)
#if DEBUG
    printf("%f %f %f\n", Sqvx.f, Sqvy.f, Sqvz.f);
    printf("\n");
#endif
    // Third Jacobi conjugation
    SVD3X3_JACOBI_CONJUATION(Ss33, Ss31, Ss32, Ss11, Ss21, Ss22, Sqvz, Sqvx,
                             Sqvy, Stmp3, Stmp1, Stmp2)
#if DEBUG
    printf("%f %f %f\n", Sqvx.f, Sqvy.f, Sqvz.f);
    printf("\n");
#endif
  }

  SVD3X3_COMPUTE_MATRIX_V

#if DEBUG
  printf("%f %f %f\n", Sv11.f, Sv21.f, Sv31.f);
  printf("%f %f %f\n", Sv21.f, Sv22.f, Sv32.f);
  printf("%f %f %f\n", Sv31.f, Sv32.f, Sv33.f);
  printf("\n");
#endif

  SVD3X3_MULTIPLY_WITH_V

  SVD3X3_SORT_SINGULAR_VALUES

  Su11.f = 1.0;
  Su12.f = 0.0;
  Su13.f = 0.0;
  Su21.f = 0.0;
  Su22.f = 1.0;
  Su23.f = 0.0;
  Su31.f = 0.0;
  Su32.f = 0.0;
  Su33.f = 1.0;

  // First Givens rotation
  SVD3X3_QR(Sa11, Sa21, Sa11, Sa21, Sa12, Sa22, Sa13, Sa23, Su11, Su12, Su21,
            Su22, Su31, Su32)

  // Second Givens rotation
  SVD3X3_QR(Sa11, Sa31, Sa11, Sa31, Sa12, Sa32, Sa13, Sa33, Su11, Su13, Su21,
            Su23, Su31, Su33)

  // Third Givens Rotation
  SVD3X3_QR(Sa22, Sa32, Sa21, Sa31, Sa22, Sa32, Sa23, Sa33, Su12, Su13, Su22,
            Su23, Su32, Su33)

#undef un
#undef Sone
#undef Ssine_pi_over_eight
#undef Scosine_pi_over_eight
#undef Sone_half
#undef Ssmall_number
#undef Stiny_number
#undef Sfour_gamma_squared
#undef ff
#undef add
#undef sub
#undef mul
#undef rsqrt
}
} // namespace internal
} // namespace utils
} // namespace sfm