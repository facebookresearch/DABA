// Copyright (c) Meta Platforms, Inc. and affiliates.

/*
 * Copyright (c) 2011-2013 NVIDIA Corporation. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *   Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 *
 *   Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 *   Neither the name of NVIDIA Corporation nor the names of its contributors
 *   may be used to endorse or promote products derived from this software
 *   without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * release 1.1:
 *
 * o added slu_matinv_batch() for batched inversion of float matrices
 * o added special kernels for faster processing of very small matrices
 * o added tuning for sm_35 to the configuration template class
 *
 */

#include "lu.cuh"
#include "operations.h"

#define GRID_DIM_LIMIT (65520)

#define ARCH_SM60 (3)

#define GPU_ARCH (ARCH_SM60)

namespace sfm {
namespace utils {
namespace internal {
/* Poor man's typeid */
template <typename T, int arch> class config { public: };

template <> class config<float, ARCH_SM60> {
public:
  typedef float absValType;

  enum { gje3MinDim = 2 };
  enum { gje3MaxDim = 109 };
  enum { gje3MinBlks = 1 };
  enum { gje3MaxThrds = 2048 }; /* sm_35, 32 registers per thread */

  enum { gje3DimX_00 = -1 };
  enum { gje3DimX_01 = -1 };
  enum { gje3DimX_02 = 2 };
  enum { gje3DimX_03 = 3 };
  enum { gje3DimX_04 = 4 };
  enum { gje3DimX_05 = 5 };
  enum { gje3DimX_06 = 3 };
  enum { gje3DimX_07 = 7 };
  enum { gje3DimX_08 = 8 };
  enum { gje3DimX_09 = 9 };
  enum { gje3DimX_10 = 5 };
  enum { gje3DimX_11 = 6 };
  enum { gje3DimX_12 = 4 };
  enum { gje3DimX_13 = 7 };
  enum { gje3DimX_14 = 7 };
  enum { gje3DimX_15 = 5 };
  enum { gje3DimX_16 = 8 };
  enum { gje3DimX_17 = 5 };
  enum { gje3DimX_18 = 5 };
  enum { gje3DimX_19 = 5 };
  enum { gje3DimX_20 = 4 };
  enum { gje3DimX_21 = 6 };
  enum { gje3DimX_22 = 2 };
  enum { gje3DimX_23 = 4 };
  enum { gje3DimX_24 = 4 };
  enum { gje3DimX_25 = 5 };
  enum { gje3DimX_26 = 2 };
  enum { gje3DimX_27 = 4 };
  enum { gje3DimX_28 = 4 };
  enum { gje3DimX_29 = 5 };
  enum { gje3DimX_30 = 5 };
  enum { gje3DimX_31 = 4 };
  enum { gje3DimX_32 = 4 };
  enum { gje3DimX_33 = 3 };
  enum { gje3DimX_34 = 6 };
  enum { gje3DimX_35 = 3 };
  enum { gje3DimX_36 = 4 };
  enum { gje3DimX_37 = 5 };
  enum { gje3DimX_38 = 6 };
  enum { gje3DimX_39 = 7 };
  enum { gje3DimX_40 = 7 };
  enum { gje3DimX_41 = 6 };
  enum { gje3DimX_42 = 6 };
  enum { gje3DimX_43 = 6 };
  enum { gje3DimX_44 = 4 };
  enum { gje3DimX_45 = 7 };
  enum { gje3DimX_46 = 8 };
  enum { gje3DimX_47 = 8 };
  enum { gje3DimX_48 = 8 };
  enum { gje3DimX_49 = 10 };
  enum { gje3DimX_50 = 10 };
  enum { gje3DimX_51 = 8 };
  enum { gje3DimX_52 = 4 };
  enum { gje3DimX_53 = 9 };
  enum { gje3DimX_54 = 7 };
  enum { gje3DimX_55 = 11 };
  enum { gje3DimX_56 = 8 };
  enum { gje3DimX_57 = 10 };
  enum { gje3DimX_58 = 10 };
  enum { gje3DimX_59 = 9 };
  enum { gje3DimX_60 = 8 };
  enum { gje3DimX_61 = 6 };
  enum { gje3DimX_62 = 6 };
  enum { gje3DimX_63 = 11 };
  enum { gje3DimX_64 = 12 };
  enum { gje3DimX_65 = 12 };
  enum { gje3DimX_66 = 11 };
  enum { gje3DimX_67 = 8 };
  enum { gje3DimX_68 = 8 };
  enum { gje3DimX_69 = 5 };
  enum { gje3DimX_70 = 6 };
  enum { gje3DimX_71 = 7 };
  enum { gje3DimX_72 = 9 };
  enum { gje3DimX_73 = 9 };
  enum { gje3DimX_74 = 10 };
  enum { gje3DimX_75 = 11 };
  enum { gje3DimX_76 = 12 };
  enum { gje3DimX_77 = 9 };
  enum { gje3DimX_78 = 12 };
  enum { gje3DimX_79 = 12 };
  enum { gje3DimX_80 = 10 };
  enum { gje3DimX_81 = 12 };
  enum { gje3DimX_82 = 12 };
  enum { gje3DimX_83 = 12 };
  enum { gje3DimX_84 = 12 };
  enum { gje3DimX_85 = 11 };
  enum { gje3DimX_86 = 11 };
  enum { gje3DimX_87 = 11 };
  enum { gje3DimX_88 = 11 };
  enum { gje3DimX_89 = 10 };
  enum { gje3DimX_90 = 10 };
  enum { gje3DimX_91 = 11 };
  enum { gje3DimX_92 = 11 };
  enum { gje3DimX_93 = 11 };
  enum { gje3DimX_94 = 10 };
  enum { gje3DimX_95 = 9 };
  enum { gje3DimX_96 = 8 };
  enum { gje3DimX_97 = 10 };
  enum { gje3DimX_98 = 10 };
  enum { gje3DimX_99 = 9 };
  enum { gje3DimX_100 = 10 };
  enum { gje3DimX_101 = 10 };
  enum { gje3DimX_102 = 6 };
  enum { gje3DimX_103 = 7 };
  enum { gje3DimX_104 = 9 };
  enum { gje3DimX_105 = 9 };
  enum { gje3DimX_106 = 9 };
  enum { gje3DimX_107 = 9 };
  enum { gje3DimX_108 = 9 };
  enum { gje3DimX_109 = 9 };

  enum { gje3Pad_00 = 0 };
  enum { gje3Pad_01 = 0 };
  enum { gje3Pad_02 = 0 };
  enum { gje3Pad_03 = 0 };
  enum { gje3Pad_04 = 0 };
  enum { gje3Pad_05 = 0 };
  enum { gje3Pad_06 = 0 };
  enum { gje3Pad_07 = 0 };
  enum { gje3Pad_08 = 1 };
  enum { gje3Pad_09 = 5 };
  enum { gje3Pad_10 = 0 };
  enum { gje3Pad_11 = 1 };
  enum { gje3Pad_12 = 0 };
  enum { gje3Pad_13 = 1 };
  enum { gje3Pad_14 = 1 };
  enum { gje3Pad_15 = 5 };
  enum { gje3Pad_16 = 1 };
  enum { gje3Pad_17 = 3 };
  enum { gje3Pad_18 = 5 };
  enum { gje3Pad_19 = 4 };
  enum { gje3Pad_20 = 0 };
  enum { gje3Pad_21 = 5 };
  enum { gje3Pad_22 = 4 };
  enum { gje3Pad_23 = 5 };
  enum { gje3Pad_24 = 4 };
  enum { gje3Pad_25 = 2 };
  enum { gje3Pad_26 = 0 };
  enum { gje3Pad_27 = 1 };
  enum { gje3Pad_28 = 0 };
  enum { gje3Pad_29 = 1 };
  enum { gje3Pad_30 = 4 };
  enum { gje3Pad_31 = 5 };
  enum { gje3Pad_32 = 4 };
  enum { gje3Pad_33 = 2 };
  enum { gje3Pad_34 = 4 };
  enum { gje3Pad_35 = 0 };
  enum { gje3Pad_36 = 0 };
  enum { gje3Pad_37 = 0 };
  enum { gje3Pad_38 = 0 };
  enum { gje3Pad_39 = 0 };
  enum { gje3Pad_40 = 1 };
  enum { gje3Pad_41 = 4 };
  enum { gje3Pad_42 = 3 };
  enum { gje3Pad_43 = 2 };
  enum { gje3Pad_44 = 0 };
  enum { gje3Pad_45 = 1 };
  enum { gje3Pad_46 = 4 };
  enum { gje3Pad_47 = 2 };
  enum { gje3Pad_48 = 4 };
  enum { gje3Pad_49 = 4 };
  enum { gje3Pad_50 = 3 };
  enum { gje3Pad_51 = 5 };
  enum { gje3Pad_52 = 0 };
  enum { gje3Pad_53 = 0 };
  enum { gje3Pad_54 = 0 };
  enum { gje3Pad_55 = 4 };
  enum { gje3Pad_56 = 1 };
  enum { gje3Pad_57 = 1 };
  enum { gje3Pad_58 = 1 };
  enum { gje3Pad_59 = 0 };
  enum { gje3Pad_60 = 0 };
  enum { gje3Pad_61 = 0 };
  enum { gje3Pad_62 = 0 };
  enum { gje3Pad_63 = 5 };
  enum { gje3Pad_64 = 5 };
  enum { gje3Pad_65 = 5 };
  enum { gje3Pad_66 = 4 };
  enum { gje3Pad_67 = 5 };
  enum { gje3Pad_68 = 4 };
  enum { gje3Pad_69 = 0 };
  enum { gje3Pad_70 = 0 };
  enum { gje3Pad_71 = 0 };
  enum { gje3Pad_72 = 1 };
  enum { gje3Pad_73 = 0 };
  enum { gje3Pad_74 = 0 };
  enum { gje3Pad_75 = 0 };
  enum { gje3Pad_76 = 0 };
  enum { gje3Pad_77 = 0 };
  enum { gje3Pad_78 = 5 };
  enum { gje3Pad_79 = 5 };
  enum { gje3Pad_80 = 5 };
  enum { gje3Pad_81 = 5 };
  enum { gje3Pad_82 = 3 };
  enum { gje3Pad_83 = 2 };
  enum { gje3Pad_84 = 2 };
  enum { gje3Pad_85 = 3 };
  enum { gje3Pad_86 = 1 };
  enum { gje3Pad_87 = 2 };
  enum { gje3Pad_88 = 1 };
  enum { gje3Pad_89 = 1 };
  enum { gje3Pad_90 = 1 };
  enum { gje3Pad_91 = 0 };
  enum { gje3Pad_92 = 0 };
  enum { gje3Pad_93 = 0 };
  enum { gje3Pad_94 = 0 };
  enum { gje3Pad_95 = 5 };
  enum { gje3Pad_96 = 4 };
  enum { gje3Pad_97 = 5 };
  enum { gje3Pad_98 = 5 };
  enum { gje3Pad_99 = 2 };
  enum { gje3Pad_100 = 5 };
  enum { gje3Pad_101 = 5 };
  enum { gje3Pad_102 = 0 };
  enum { gje3Pad_103 = 0 };
  enum { gje3Pad_104 = 1 };
  enum { gje3Pad_105 = 0 };
  enum { gje3Pad_106 = 2 };
  enum { gje3Pad_107 = 2 };
  enum { gje3Pad_108 = 2 };
  enum { gje3Pad_109 = 1 };

  enum { gje3SrchThrd_00 = -1 };
  enum { gje3SrchThrd_01 = -1 };
  enum { gje3SrchThrd_02 = 1 };
  enum { gje3SrchThrd_03 = 1 };
  enum { gje3SrchThrd_04 = 1 };
  enum { gje3SrchThrd_05 = 1 };
  enum { gje3SrchThrd_06 = 1 };
  enum { gje3SrchThrd_07 = 1 };
  enum { gje3SrchThrd_08 = 2 };
  enum { gje3SrchThrd_09 = 2 };
  enum { gje3SrchThrd_10 = 2 };
  enum { gje3SrchThrd_11 = 2 };
  enum { gje3SrchThrd_12 = 2 };
  enum { gje3SrchThrd_13 = 2 };
  enum { gje3SrchThrd_14 = 2 };
  enum { gje3SrchThrd_15 = 2 };
  enum { gje3SrchThrd_16 = 2 };
  enum { gje3SrchThrd_17 = 2 };
  enum { gje3SrchThrd_18 = 2 };
  enum { gje3SrchThrd_19 = 2 };
  enum { gje3SrchThrd_20 = 2 };
  enum { gje3SrchThrd_21 = 2 };
  enum { gje3SrchThrd_22 = 2 };
  enum { gje3SrchThrd_23 = 2 };
  enum { gje3SrchThrd_24 = 2 };
  enum { gje3SrchThrd_25 = 2 };
  enum { gje3SrchThrd_26 = 2 };
  enum { gje3SrchThrd_27 = 2 };
  enum { gje3SrchThrd_28 = 2 };
  enum { gje3SrchThrd_29 = 2 };
  enum { gje3SrchThrd_30 = 2 };
  enum { gje3SrchThrd_31 = 2 };
  enum { gje3SrchThrd_32 = 2 };
  enum { gje3SrchThrd_33 = 2 };
  enum { gje3SrchThrd_34 = 2 };
  enum { gje3SrchThrd_35 = 2 };
  enum { gje3SrchThrd_36 = 2 };
  enum { gje3SrchThrd_37 = 2 };
  enum { gje3SrchThrd_38 = 2 };
  enum { gje3SrchThrd_39 = 2 };
  enum { gje3SrchThrd_40 = 2 };
  enum { gje3SrchThrd_41 = 2 };
  enum { gje3SrchThrd_42 = 2 };
  enum { gje3SrchThrd_43 = 2 };
  enum { gje3SrchThrd_44 = 2 };
  enum { gje3SrchThrd_45 = 2 };
  enum { gje3SrchThrd_46 = 2 };
  enum { gje3SrchThrd_47 = 2 };
  enum { gje3SrchThrd_48 = 2 };
  enum { gje3SrchThrd_49 = 4 };
  enum { gje3SrchThrd_50 = 4 };
  enum { gje3SrchThrd_51 = 4 };
  enum { gje3SrchThrd_52 = 4 };
  enum { gje3SrchThrd_53 = 4 };
  enum { gje3SrchThrd_54 = 4 };
  enum { gje3SrchThrd_55 = 4 };
  enum { gje3SrchThrd_56 = 4 };
  enum { gje3SrchThrd_57 = 4 };
  enum { gje3SrchThrd_58 = 4 };
  enum { gje3SrchThrd_59 = 4 };
  enum { gje3SrchThrd_60 = 4 };
  enum { gje3SrchThrd_61 = 4 };
  enum { gje3SrchThrd_62 = 4 };
  enum { gje3SrchThrd_63 = 4 };
  enum { gje3SrchThrd_64 = 4 };
  enum { gje3SrchThrd_65 = 4 };
  enum { gje3SrchThrd_66 = 4 };
  enum { gje3SrchThrd_67 = 4 };
  enum { gje3SrchThrd_68 = 4 };
  enum { gje3SrchThrd_69 = 4 };
  enum { gje3SrchThrd_70 = 4 };
  enum { gje3SrchThrd_71 = 4 };
  enum { gje3SrchThrd_72 = 4 };
  enum { gje3SrchThrd_73 = 4 };
  enum { gje3SrchThrd_74 = 4 };
  enum { gje3SrchThrd_75 = 4 };
  enum { gje3SrchThrd_76 = 4 };
  enum { gje3SrchThrd_77 = 4 };
  enum { gje3SrchThrd_78 = 4 };
  enum { gje3SrchThrd_79 = 4 };
  enum { gje3SrchThrd_80 = 4 };
  enum { gje3SrchThrd_81 = 4 };
  enum { gje3SrchThrd_82 = 4 };
  enum { gje3SrchThrd_83 = 4 };
  enum { gje3SrchThrd_84 = 4 };
  enum { gje3SrchThrd_85 = 4 };
  enum { gje3SrchThrd_86 = 4 };
  enum { gje3SrchThrd_87 = 4 };
  enum { gje3SrchThrd_88 = 4 };
  enum { gje3SrchThrd_89 = 4 };
  enum { gje3SrchThrd_90 = 4 };
  enum { gje3SrchThrd_91 = 4 };
  enum { gje3SrchThrd_92 = 4 };
  enum { gje3SrchThrd_93 = 4 };
  enum { gje3SrchThrd_94 = 4 };
  enum { gje3SrchThrd_95 = 4 };
  enum { gje3SrchThrd_96 = 4 };
  enum { gje3SrchThrd_97 = 4 };
  enum { gje3SrchThrd_98 = 4 };
  enum { gje3SrchThrd_99 = 4 };
  enum { gje3SrchThrd_100 = 4 };
  enum { gje3SrchThrd_101 = 4 };
  enum { gje3SrchThrd_102 = 4 };
  enum { gje3SrchThrd_103 = 4 };
  enum { gje3SrchThrd_104 = 4 };
  enum { gje3SrchThrd_105 = 4 };
  enum { gje3SrchThrd_106 = 4 };
  enum { gje3SrchThrd_107 = 4 };
  enum { gje3SrchThrd_108 = 4 };
  enum { gje3SrchThrd_109 = 4 };

  enum { matInv2x2MinBatch = 1200 };
  enum { matInv3x3MinBatch = 1000 };
  enum { matInv4x4MinBatch = 900 };
  enum { matInv5x5MinBatch = 900 };
  enum { matInv6x6MinBatch = 900 };
  enum { matInv7x7MinBatch = 1000 };
  enum { matInv8x8MinBatch = 1000 };
  enum { matInv9x9MinBatch = 1000 };
  enum { matInv10x10MinBatch = 1000 };
  enum { matInvMinDim = 2 };
  enum { matInvMaxDim = 10 };
};

template <> class config<double, ARCH_SM60> {
public:
  typedef double absValType;

  enum { gje3MinDim = 2 };
  enum { gje3MaxDim = 77 };
  enum { gje3MinBlks = 1 };
  enum { gje3MaxThrds = 2048 }; /* sm_35, 32 registers per thread */

  enum { gje3DimX_00 = -1 };
  enum { gje3DimX_01 = -1 };
  enum { gje3DimX_02 = 2 };
  enum { gje3DimX_03 = 3 };
  enum { gje3DimX_04 = 4 };
  enum { gje3DimX_05 = 5 };
  enum { gje3DimX_06 = 6 };
  enum { gje3DimX_07 = 7 };
  enum { gje3DimX_08 = 8 };
  enum { gje3DimX_09 = 9 };
  enum { gje3DimX_10 = 5 };
  enum { gje3DimX_11 = 11 };
  enum { gje3DimX_12 = 8 };
  enum { gje3DimX_13 = 7 };
  enum { gje3DimX_14 = 7 };
  enum { gje3DimX_15 = 5 };
  enum { gje3DimX_16 = 8 };
  enum { gje3DimX_17 = 5 };
  enum { gje3DimX_18 = 5 };
  enum { gje3DimX_19 = 5 };
  enum { gje3DimX_20 = 8 };
  enum { gje3DimX_21 = 7 };
  enum { gje3DimX_22 = 4 };
  enum { gje3DimX_23 = 8 };
  enum { gje3DimX_24 = 8 };
  enum { gje3DimX_25 = 5 };
  enum { gje3DimX_26 = 6 };
  enum { gje3DimX_27 = 7 };
  enum { gje3DimX_28 = 8 };
  enum { gje3DimX_29 = 8 };
  enum { gje3DimX_30 = 6 };
  enum { gje3DimX_31 = 8 };
  enum { gje3DimX_32 = 8 };
  enum { gje3DimX_33 = 11 };
  enum { gje3DimX_34 = 6 };
  enum { gje3DimX_35 = 9 };
  enum { gje3DimX_36 = 8 };
  enum { gje3DimX_37 = 10 };
  enum { gje3DimX_38 = 10 };
  enum { gje3DimX_39 = 8 };
  enum { gje3DimX_40 = 10 };
  enum { gje3DimX_41 = 9 };
  enum { gje3DimX_42 = 10 };
  enum { gje3DimX_43 = 9 };
  enum { gje3DimX_44 = 8 };
  enum { gje3DimX_45 = 9 };
  enum { gje3DimX_46 = 12 };
  enum { gje3DimX_47 = 12 };
  enum { gje3DimX_48 = 12 };
  enum { gje3DimX_49 = 12 };
  enum { gje3DimX_50 = 12 };
  enum { gje3DimX_51 = 9 };
  enum { gje3DimX_52 = 8 };
  enum { gje3DimX_53 = 11 };
  enum { gje3DimX_54 = 12 };
  enum { gje3DimX_55 = 11 };
  enum { gje3DimX_56 = 12 };
  enum { gje3DimX_57 = 11 };
  enum { gje3DimX_58 = 12 };
  enum { gje3DimX_59 = 12 };
  enum { gje3DimX_60 = 12 };
  enum { gje3DimX_61 = 12 };
  enum { gje3DimX_62 = 12 };
  enum { gje3DimX_63 = 11 };
  enum { gje3DimX_64 = 12 };
  enum { gje3DimX_65 = 12 };
  enum { gje3DimX_66 = 12 };
  enum { gje3DimX_67 = 12 };
  enum { gje3DimX_68 = 12 };
  enum { gje3DimX_69 = 12 };
  enum { gje3DimX_70 = 12 };
  enum { gje3DimX_71 = 9 };
  enum { gje3DimX_72 = 9 };
  enum { gje3DimX_73 = 11 };
  enum { gje3DimX_74 = 10 };
  enum { gje3DimX_75 = 11 };
  enum { gje3DimX_76 = 12 };
  enum { gje3DimX_77 = 11 };
  enum { gje3DimX_78 = -1 };
  enum { gje3DimX_79 = -1 };
  enum { gje3DimX_80 = -1 };
  enum { gje3DimX_81 = -1 };
  enum { gje3DimX_82 = -1 };
  enum { gje3DimX_83 = -1 };
  enum { gje3DimX_84 = -1 };
  enum { gje3DimX_85 = -1 };
  enum { gje3DimX_86 = -1 };
  enum { gje3DimX_87 = -1 };
  enum { gje3DimX_88 = -1 };
  enum { gje3DimX_89 = -1 };
  enum { gje3DimX_90 = -1 };
  enum { gje3DimX_91 = -1 };
  enum { gje3DimX_92 = -1 };
  enum { gje3DimX_93 = -1 };
  enum { gje3DimX_94 = -1 };
  enum { gje3DimX_95 = -1 };
  enum { gje3DimX_96 = -1 };
  enum { gje3DimX_97 = -1 };
  enum { gje3DimX_98 = -1 };
  enum { gje3DimX_99 = -1 };
  enum { gje3DimX_100 = -1 };
  enum { gje3DimX_101 = -1 };
  enum { gje3DimX_102 = -1 };
  enum { gje3DimX_103 = -1 };
  enum { gje3DimX_104 = -1 };
  enum { gje3DimX_105 = -1 };
  enum { gje3DimX_106 = -1 };
  enum { gje3DimX_107 = -1 };
  enum { gje3DimX_108 = -1 };
  enum { gje3DimX_109 = -1 };

  enum { gje3Pad_00 = 0 };
  enum { gje3Pad_01 = 0 };
  enum { gje3Pad_02 = 0 };
  enum { gje3Pad_03 = 0 };
  enum { gje3Pad_04 = 0 };
  enum { gje3Pad_05 = 0 };
  enum { gje3Pad_06 = 0 };
  enum { gje3Pad_07 = 0 };
  enum { gje3Pad_08 = 1 };
  enum { gje3Pad_09 = 0 };
  enum { gje3Pad_10 = 0 };
  enum { gje3Pad_11 = 0 };
  enum { gje3Pad_12 = 0 };
  enum { gje3Pad_13 = 0 };
  enum { gje3Pad_14 = 0 };
  enum { gje3Pad_15 = 3 };
  enum { gje3Pad_16 = 4 };
  enum { gje3Pad_17 = 1 };
  enum { gje3Pad_18 = 0 };
  enum { gje3Pad_19 = 0 };
  enum { gje3Pad_20 = 0 };
  enum { gje3Pad_21 = 2 };
  enum { gje3Pad_22 = 0 };
  enum { gje3Pad_23 = 2 };
  enum { gje3Pad_24 = 1 };
  enum { gje3Pad_25 = 0 };
  enum { gje3Pad_26 = 0 };
  enum { gje3Pad_27 = 0 };
  enum { gje3Pad_28 = 0 };
  enum { gje3Pad_29 = 0 };
  enum { gje3Pad_30 = 0 };
  enum { gje3Pad_31 = 5 };
  enum { gje3Pad_32 = 4 };
  enum { gje3Pad_33 = 2 };
  enum { gje3Pad_34 = 0 };
  enum { gje3Pad_35 = 2 };
  enum { gje3Pad_36 = 0 };
  enum { gje3Pad_37 = 1 };
  enum { gje3Pad_38 = 0 };
  enum { gje3Pad_39 = 5 };
  enum { gje3Pad_40 = 2 };
  enum { gje3Pad_41 = 0 };
  enum { gje3Pad_42 = 0 };
  enum { gje3Pad_43 = 2 };
  enum { gje3Pad_44 = 0 };
  enum { gje3Pad_45 = 1 };
  enum { gje3Pad_46 = 0 };
  enum { gje3Pad_47 = 5 };
  enum { gje3Pad_48 = 5 };
  enum { gje3Pad_49 = 5 };
  enum { gje3Pad_50 = 4 };
  enum { gje3Pad_51 = 2 };
  enum { gje3Pad_52 = 0 };
  enum { gje3Pad_53 = 0 };
  enum { gje3Pad_54 = 0 };
  enum { gje3Pad_55 = 4 };
  enum { gje3Pad_56 = 4 };
  enum { gje3Pad_57 = 2 };
  enum { gje3Pad_58 = 2 };
  enum { gje3Pad_59 = 1 };
  enum { gje3Pad_60 = 0 };
  enum { gje3Pad_61 = 1 };
  enum { gje3Pad_62 = 0 };
  enum { gje3Pad_63 = 5 };
  enum { gje3Pad_64 = 5 };
  enum { gje3Pad_65 = 4 };
  enum { gje3Pad_66 = 4 };
  enum { gje3Pad_67 = 3 };
  enum { gje3Pad_68 = 2 };
  enum { gje3Pad_69 = 1 };
  enum { gje3Pad_70 = 0 };
  enum { gje3Pad_71 = 2 };
  enum { gje3Pad_72 = 1 };
  enum { gje3Pad_73 = 2 };
  enum { gje3Pad_74 = 0 };
  enum { gje3Pad_75 = 0 };
  enum { gje3Pad_76 = 0 };
  enum { gje3Pad_77 = 0 };
  enum { gje3Pad_78 = 0 };
  enum { gje3Pad_79 = 0 };
  enum { gje3Pad_80 = 0 };
  enum { gje3Pad_81 = 0 };
  enum { gje3Pad_82 = 0 };
  enum { gje3Pad_83 = 0 };
  enum { gje3Pad_84 = 0 };
  enum { gje3Pad_85 = 0 };
  enum { gje3Pad_86 = 0 };
  enum { gje3Pad_87 = 0 };
  enum { gje3Pad_88 = 0 };
  enum { gje3Pad_89 = 0 };
  enum { gje3Pad_90 = 0 };
  enum { gje3Pad_91 = 0 };
  enum { gje3Pad_92 = 0 };
  enum { gje3Pad_93 = 0 };
  enum { gje3Pad_94 = 0 };
  enum { gje3Pad_95 = 0 };
  enum { gje3Pad_96 = 0 };
  enum { gje3Pad_97 = 0 };
  enum { gje3Pad_98 = 0 };
  enum { gje3Pad_99 = 0 };
  enum { gje3Pad_100 = 0 };
  enum { gje3Pad_101 = 0 };
  enum { gje3Pad_102 = 0 };
  enum { gje3Pad_103 = 0 };
  enum { gje3Pad_104 = 0 };
  enum { gje3Pad_105 = 0 };
  enum { gje3Pad_106 = 0 };
  enum { gje3Pad_107 = 0 };
  enum { gje3Pad_108 = 0 };
  enum { gje3Pad_109 = 0 };

  enum { gje3SrchThrd_00 = -1 };
  enum { gje3SrchThrd_01 = -1 };
  enum { gje3SrchThrd_02 = 1 };
  enum { gje3SrchThrd_03 = 2 };
  enum { gje3SrchThrd_04 = 2 };
  enum { gje3SrchThrd_05 = 2 };
  enum { gje3SrchThrd_06 = 2 };
  enum { gje3SrchThrd_07 = 2 };
  enum { gje3SrchThrd_08 = 2 };
  enum { gje3SrchThrd_09 = 2 };
  enum { gje3SrchThrd_10 = 2 };
  enum { gje3SrchThrd_11 = 2 };
  enum { gje3SrchThrd_12 = 2 };
  enum { gje3SrchThrd_13 = 2 };
  enum { gje3SrchThrd_14 = 2 };
  enum { gje3SrchThrd_15 = 2 };
  enum { gje3SrchThrd_16 = 2 };
  enum { gje3SrchThrd_17 = 2 };
  enum { gje3SrchThrd_18 = 2 };
  enum { gje3SrchThrd_19 = 2 };
  enum { gje3SrchThrd_20 = 3 };
  enum { gje3SrchThrd_21 = 3 };
  enum { gje3SrchThrd_22 = 3 };
  enum { gje3SrchThrd_23 = 3 };
  enum { gje3SrchThrd_24 = 3 };
  enum { gje3SrchThrd_25 = 3 };
  enum { gje3SrchThrd_26 = 3 };
  enum { gje3SrchThrd_27 = 3 };
  enum { gje3SrchThrd_28 = 3 };
  enum { gje3SrchThrd_29 = 3 };
  enum { gje3SrchThrd_30 = 3 };
  enum { gje3SrchThrd_31 = 3 };
  enum { gje3SrchThrd_32 = 3 };
  enum { gje3SrchThrd_33 = 3 };
  enum { gje3SrchThrd_34 = 3 };
  enum { gje3SrchThrd_35 = 3 };
  enum { gje3SrchThrd_36 = 3 };
  enum { gje3SrchThrd_37 = 3 };
  enum { gje3SrchThrd_38 = 3 };
  enum { gje3SrchThrd_39 = 3 };
  enum { gje3SrchThrd_40 = 3 };
  enum { gje3SrchThrd_41 = 5 };
  enum { gje3SrchThrd_42 = 5 };
  enum { gje3SrchThrd_43 = 5 };
  enum { gje3SrchThrd_44 = 5 };
  enum { gje3SrchThrd_45 = 5 };
  enum { gje3SrchThrd_46 = 5 };
  enum { gje3SrchThrd_47 = 5 };
  enum { gje3SrchThrd_48 = 5 };
  enum { gje3SrchThrd_49 = 5 };
  enum { gje3SrchThrd_50 = 5 };
  enum { gje3SrchThrd_51 = 5 };
  enum { gje3SrchThrd_52 = 5 };
  enum { gje3SrchThrd_53 = 5 };
  enum { gje3SrchThrd_54 = 5 };
  enum { gje3SrchThrd_55 = 5 };
  enum { gje3SrchThrd_56 = 5 };
  enum { gje3SrchThrd_57 = 5 };
  enum { gje3SrchThrd_58 = 5 };
  enum { gje3SrchThrd_59 = 5 };
  enum { gje3SrchThrd_60 = 5 };
  enum { gje3SrchThrd_61 = 5 };
  enum { gje3SrchThrd_62 = 5 };
  enum { gje3SrchThrd_63 = 5 };
  enum { gje3SrchThrd_64 = 5 };
  enum { gje3SrchThrd_65 = 5 };
  enum { gje3SrchThrd_66 = 5 };
  enum { gje3SrchThrd_67 = 5 };
  enum { gje3SrchThrd_68 = 5 };
  enum { gje3SrchThrd_69 = 5 };
  enum { gje3SrchThrd_70 = 5 };
  enum { gje3SrchThrd_71 = 5 };
  enum { gje3SrchThrd_72 = 5 };
  enum { gje3SrchThrd_73 = 5 };
  enum { gje3SrchThrd_74 = 5 };
  enum { gje3SrchThrd_75 = 5 };
  enum { gje3SrchThrd_76 = 5 };
  enum { gje3SrchThrd_77 = 5 };
  enum { gje3SrchThrd_78 = -1 };
  enum { gje3SrchThrd_79 = -1 };
  enum { gje3SrchThrd_80 = -1 };
  enum { gje3SrchThrd_81 = -1 };
  enum { gje3SrchThrd_82 = -1 };
  enum { gje3SrchThrd_83 = -1 };
  enum { gje3SrchThrd_84 = -1 };
  enum { gje3SrchThrd_85 = -1 };
  enum { gje3SrchThrd_86 = -1 };
  enum { gje3SrchThrd_87 = -1 };
  enum { gje3SrchThrd_88 = -1 };
  enum { gje3SrchThrd_89 = -1 };
  enum { gje3SrchThrd_90 = -1 };
  enum { gje3SrchThrd_91 = -1 };
  enum { gje3SrchThrd_92 = -1 };
  enum { gje3SrchThrd_93 = -1 };
  enum { gje3SrchThrd_94 = -1 };
  enum { gje3SrchThrd_95 = -1 };
  enum { gje3SrchThrd_96 = -1 };
  enum { gje3SrchThrd_97 = -1 };
  enum { gje3SrchThrd_98 = -1 };
  enum { gje3SrchThrd_99 = -1 };
  enum { gje3SrchThrd_100 = -1 };
  enum { gje3SrchThrd_101 = -1 };
  enum { gje3SrchThrd_102 = -1 };
  enum { gje3SrchThrd_103 = -1 };
  enum { gje3SrchThrd_104 = -1 };
  enum { gje3SrchThrd_105 = -1 };
  enum { gje3SrchThrd_106 = -1 };
  enum { gje3SrchThrd_107 = -1 };
  enum { gje3SrchThrd_108 = -1 };
  enum { gje3SrchThrd_109 = -1 };

  enum { matInv2x2MinBatch = 1300 };
  enum { matInv3x3MinBatch = 1100 };
  enum { matInv4x4MinBatch = 1100 };
  enum { matInv5x5MinBatch = 1100 };
  enum { matInv6x6MinBatch = 1100 };
  enum { matInv7x7MinBatch = 1100 };
  enum { matInv8x8MinBatch = 1100 };
  enum { matInv9x9MinBatch = 1100 };
  enum { matInv10x10MinBatch = 1200 };
  enum { matInvMinDim = 2 };
  enum { matInvMaxDim = 10 };
};

/* row-major matrix of array */
#define As(row, col) As[(N + ofs) * (col) + (row)]
#define AsInv(row, col) AsInv[(N + ofs) * (col) + (row)]

#define Ainv(row, col) Ainv[(row * N + col) * batch + thrdNum]
#define USE_PIVOTING 1

template <typename T, int arch>
__global__ void lu_matinv_2x2_matrix_per_thread(const T *A, T *Ainv, int batch) {
  const int blkNum = blockIdx.y * gridDim.x + blockIdx.x;
  const int thrdNum = blkNum * blockDim.x + threadIdx.x;
  const int N = 2;
  int perm0, perm1;
  int icol0, icol1;
  T AA00, AA01;
  T AA10, AA11;
  T tmp;
#if USE_PIVOTING
  typename config<T, arch>::absValType t;
  typename config<T, arch>::absValType p;
  int i, pvt;
#endif

  if (thrdNum < batch) {

    AA00 = A[0 * batch + thrdNum];
    AA10 = A[1 * batch + thrdNum];
    AA01 = A[2 * batch + thrdNum];
    AA11 = A[3 * batch + thrdNum];

    perm0 = 0;
    perm1 = 1;

    /****************** iteration 0 ***********/

#if USE_PIVOTING
    /* search pivot row */
    p = absOp(AA00);
    pvt = 0;
    t = absOp(AA10);
    if (t > p) {
      p = t;
      pvt = 1;
    }

    /* swap pivot row with row 0 */
    if (pvt == 1) {
      tmp = AA00;
      AA00 = AA10;
      AA10 = tmp;
      tmp = AA01;
      AA01 = AA11;
      AA11 = tmp;
      /* update permutation vector based on row swap */
      i = perm0;
      perm0 = perm1;
      perm1 = i;
    }
#endif // USE_PIVOTING

    /* scale current row */
    tmp = rcpOp(AA00);
    icol0 = perm0;
    AA00 = tmp;
    AA01 = mulOp(tmp, AA01);

    /* eliminate above and below current row */
    tmp = AA10;
    AA10 = mulOp(negOp(tmp), AA00);
    AA11 = fmnaOp(tmp, AA01, AA11);

    /****************** iteration 1 ***********/

    /* scale current row */
    tmp = rcpOp(AA11);
    icol1 = perm1;
    AA10 = mulOp(tmp, AA10);
    AA11 = tmp;

    /* eliminate above and below current row */
    tmp = AA01;
    AA00 = fmnaOp(tmp, AA10, AA00);
    AA01 = mulOp(negOp(tmp), AA11);

    /* sort columns into the correct order */
    Ainv(0, icol0) = AA00;
    Ainv(1, icol0) = AA10;
    Ainv(0, icol1) = AA01;
    Ainv(1, icol1) = AA11;
  }
}

template <typename T, int arch>
__global__ void lu_matinv_3x3_matrix_per_thread(const T *A, T *Ainv, int batch) {
  const int blkNum = blockIdx.y * gridDim.x + blockIdx.x;
  const int thrdNum = blkNum * blockDim.x + threadIdx.x;
  const int N = 3;
  int perm0, perm1, perm2;
  int icol0, icol1, icol2;
  T AA00, AA01, AA02;
  T AA10, AA11, AA12;
  T AA20, AA21, AA22;
  T tmp;
#if USE_PIVOTING
  typename config<T, arch>::absValType t;
  typename config<T, arch>::absValType p;
  int i, pvt;
#endif

  if (thrdNum < batch) {

    AA00 = A[0 * batch + thrdNum];
    AA10 = A[1 * batch + thrdNum];
    AA20 = A[2 * batch + thrdNum];
    AA01 = A[3 * batch + thrdNum];
    AA11 = A[4 * batch + thrdNum];
    AA21 = A[5 * batch + thrdNum];
    AA02 = A[6 * batch + thrdNum];
    AA12 = A[7 * batch + thrdNum];
    AA22 = A[8 * batch + thrdNum];

    perm0 = 0;
    perm1 = 1;
    perm2 = 2;

    /****************** iteration 0 ***********/

#if USE_PIVOTING
    /* search pivot row */
    p = absOp(AA00);
    pvt = 0;
    t = absOp(AA10);
    if (t > p) {
      p = t;
      pvt = 1;
    }
    t = absOp(AA20);
    if (t > p) {
      p = t;
      pvt = 2;
    }

    /* swap pivot row with row 0 */
    if (pvt == 1) {
      tmp = AA00;
      AA00 = AA10;
      AA10 = tmp;
      tmp = AA01;
      AA01 = AA11;
      AA11 = tmp;
      tmp = AA02;
      AA02 = AA12;
      AA12 = tmp;
      /* update permutation vector based on row swap */
      i = perm0;
      perm0 = perm1;
      perm1 = i;
    }
    if (pvt == 2) {
      tmp = AA00;
      AA00 = AA20;
      AA20 = tmp;
      tmp = AA01;
      AA01 = AA21;
      AA21 = tmp;
      tmp = AA02;
      AA02 = AA22;
      AA22 = tmp;
      /* update permutation vector based on row swap */
      i = perm0;
      perm0 = perm2;
      perm2 = i;
    }
#endif // USE_PIVOTING

    /* scale current row */
    tmp = rcpOp(AA00);
    icol0 = perm0;
    AA00 = tmp;
    AA01 = mulOp(tmp, AA01);
    AA02 = mulOp(tmp, AA02);

    /* eliminate above and below current row */
    tmp = AA10;
    AA10 = mulOp(negOp(tmp), AA00);
    AA11 = fmnaOp(tmp, AA01, AA11);
    AA12 = fmnaOp(tmp, AA02, AA12);

    tmp = AA20;
    AA20 = mulOp(negOp(tmp), AA00);
    AA21 = fmnaOp(tmp, AA01, AA21);
    AA22 = fmnaOp(tmp, AA02, AA22);

    /****************** iteration 1 ***********/

#if USE_PIVOTING
    /* search pivot row */
    p = absOp(AA11);
    pvt = 1;
    t = absOp(AA21);
    if (t > p) {
      p = t;
      pvt = 2;
    }

    /* swap pivot row with row 1 */
    if (pvt == 2) {
      tmp = AA10;
      AA10 = AA20;
      AA20 = tmp;
      tmp = AA11;
      AA11 = AA21;
      AA21 = tmp;
      tmp = AA12;
      AA12 = AA22;
      AA22 = tmp;
      /* update permutation vector based on row swap */
      i = perm1;
      perm1 = perm2;
      perm2 = i;
    }
#endif // USE_PIVOTING

    /* scale current row */
    tmp = rcpOp(AA11);
    icol1 = perm1;
    AA10 = mulOp(tmp, AA10);
    AA11 = tmp;
    AA12 = mulOp(tmp, AA12);

    /* eliminate above and below current row */
    tmp = AA01;
    AA00 = fmnaOp(tmp, AA10, AA00);
    AA01 = mulOp(negOp(tmp), AA11);
    AA02 = fmnaOp(tmp, AA12, AA02);

    tmp = AA21;
    AA20 = fmnaOp(tmp, AA10, AA20);
    AA21 = mulOp(negOp(tmp), AA11);
    AA22 = fmnaOp(tmp, AA12, AA22);

    /****************** iteration 2 ****************/

    /* scale current row */
    tmp = rcpOp(AA22);
    icol2 = perm2;
    AA20 = mulOp(tmp, AA20);
    AA21 = mulOp(tmp, AA21);
    AA22 = tmp;

    /* eliminate above and below current row */
    tmp = AA02;
    AA00 = fmnaOp(tmp, AA20, AA00);
    AA01 = fmnaOp(tmp, AA21, AA01);
    AA02 = mulOp(negOp(tmp), AA22);

    tmp = AA12;
    AA10 = fmnaOp(tmp, AA20, AA10);
    AA11 = fmnaOp(tmp, AA21, AA11);
    AA12 = mulOp(negOp(tmp), AA22);

    /* sort columns into the correct order */
    Ainv(0, icol0) = AA00;
    Ainv(1, icol0) = AA10;
    Ainv(2, icol0) = AA20;
    Ainv(0, icol1) = AA01;
    Ainv(1, icol1) = AA11;
    Ainv(2, icol1) = AA21;
    Ainv(0, icol2) = AA02;
    Ainv(1, icol2) = AA12;
    Ainv(2, icol2) = AA22;
  }
}

template <typename T, int arch>
__global__ void lu_matinv_4x4_matrix_per_thread(const T *A, T *Ainv, int batch) {
  const int blkNum = blockIdx.y * gridDim.x + blockIdx.x;
  const int thrdNum = blkNum * blockDim.x + threadIdx.x;
  const int N = 4;
  int perm0, perm1, perm2, perm3;
  int icol0, icol1, icol2, icol3;
  T AA00, AA01, AA02, AA03;
  T AA10, AA11, AA12, AA13;
  T AA20, AA21, AA22, AA23;
  T AA30, AA31, AA32, AA33;
  T tmp;
#if USE_PIVOTING
  typename config<T, arch>::absValType t;
  typename config<T, arch>::absValType p;
  int i, pvt;
#endif

  if (thrdNum < batch) {

    AA00 = A[0 * batch + thrdNum];
    AA10 = A[1 * batch + thrdNum];
    AA20 = A[2 * batch + thrdNum];
    AA30 = A[3 * batch + thrdNum];
    AA01 = A[4 * batch + thrdNum];
    AA11 = A[5 * batch + thrdNum];
    AA21 = A[6 * batch + thrdNum];
    AA31 = A[7 * batch + thrdNum];
    AA02 = A[8 * batch + thrdNum];
    AA12 = A[9 * batch + thrdNum];
    AA22 = A[10 * batch + thrdNum];
    AA32 = A[11 * batch + thrdNum];
    AA03 = A[12 * batch + thrdNum];
    AA13 = A[13 * batch + thrdNum];
    AA23 = A[14 * batch + thrdNum];
    AA33 = A[15 * batch + thrdNum];

    perm0 = 0;
    perm1 = 1;
    perm2 = 2;
    perm3 = 3;

    /****************** iteration 0 ***********/

#if USE_PIVOTING
    /* search pivot row */
    p = absOp(AA00);
    pvt = 0;
    t = absOp(AA10);
    if (t > p) {
      p = t;
      pvt = 1;
    }
    t = absOp(AA20);
    if (t > p) {
      p = t;
      pvt = 2;
    }
    t = absOp(AA30);
    if (t > p) {
      p = t;
      pvt = 3;
    }

    /* swap pivot row with row 0 */
    if (pvt == 1) {
      tmp = AA00;
      AA00 = AA10;
      AA10 = tmp;
      tmp = AA01;
      AA01 = AA11;
      AA11 = tmp;
      tmp = AA02;
      AA02 = AA12;
      AA12 = tmp;
      tmp = AA03;
      AA03 = AA13;
      AA13 = tmp;
      /* update permutation vector based on row swap */
      i = perm0;
      perm0 = perm1;
      perm1 = i;
    }
    if (pvt == 2) {
      tmp = AA00;
      AA00 = AA20;
      AA20 = tmp;
      tmp = AA01;
      AA01 = AA21;
      AA21 = tmp;
      tmp = AA02;
      AA02 = AA22;
      AA22 = tmp;
      tmp = AA03;
      AA03 = AA23;
      AA23 = tmp;
      /* update permutation vector based on row swap */
      i = perm0;
      perm0 = perm2;
      perm2 = i;
    }
    if (pvt == 3) {
      tmp = AA00;
      AA00 = AA30;
      AA30 = tmp;
      tmp = AA01;
      AA01 = AA31;
      AA31 = tmp;
      tmp = AA02;
      AA02 = AA32;
      AA32 = tmp;
      tmp = AA03;
      AA03 = AA33;
      AA33 = tmp;
      /* update permutation vector based on row swap */
      i = perm0;
      perm0 = perm3;
      perm3 = i;
    }
#endif // USE_PIVOTING

    /* scale current row */
    tmp = rcpOp(AA00);
    icol0 = perm0;
    AA00 = tmp;
    AA01 = mulOp(tmp, AA01);
    AA02 = mulOp(tmp, AA02);
    AA03 = mulOp(tmp, AA03);

    /* eliminate above and below current row */
    tmp = AA10;
    AA10 = mulOp(negOp(tmp), AA00);
    AA11 = fmnaOp(tmp, AA01, AA11);
    AA12 = fmnaOp(tmp, AA02, AA12);
    AA13 = fmnaOp(tmp, AA03, AA13);

    tmp = AA20;
    AA20 = mulOp(negOp(tmp), AA00);
    AA21 = fmnaOp(tmp, AA01, AA21);
    AA22 = fmnaOp(tmp, AA02, AA22);
    AA23 = fmnaOp(tmp, AA03, AA23);

    tmp = AA30;
    AA30 = mulOp(negOp(tmp), AA00);
    AA31 = fmnaOp(tmp, AA01, AA31);
    AA32 = fmnaOp(tmp, AA02, AA32);
    AA33 = fmnaOp(tmp, AA03, AA33);

    /****************** iteration 1 ***********/

#if USE_PIVOTING
    /* search pivot row */
    p = absOp(AA11);
    pvt = 1;
    t = absOp(AA21);
    if (t > p) {
      p = t;
      pvt = 2;
    }
    t = absOp(AA31);
    if (t > p) {
      p = t;
      pvt = 3;
    }

    /* swap pivot row with row 1 */
    if (pvt == 2) {
      tmp = AA10;
      AA10 = AA20;
      AA20 = tmp;
      tmp = AA11;
      AA11 = AA21;
      AA21 = tmp;
      tmp = AA12;
      AA12 = AA22;
      AA22 = tmp;
      tmp = AA13;
      AA13 = AA23;
      AA23 = tmp;
      /* update permutation vector based on row swap */
      i = perm1;
      perm1 = perm2;
      perm2 = i;
    }
    if (pvt == 3) {
      tmp = AA10;
      AA10 = AA30;
      AA30 = tmp;
      tmp = AA11;
      AA11 = AA31;
      AA31 = tmp;
      tmp = AA12;
      AA12 = AA32;
      AA32 = tmp;
      tmp = AA13;
      AA13 = AA33;
      AA33 = tmp;
      /* update permutation vector based on row swap */
      i = perm1;
      perm1 = perm3;
      perm3 = i;
    }
#endif // USE_PIVOTING

    /* scale current row */
    tmp = rcpOp(AA11);
    icol1 = perm1;
    AA10 = mulOp(tmp, AA10);
    AA11 = tmp;
    AA12 = mulOp(tmp, AA12);
    AA13 = mulOp(tmp, AA13);

    /* eliminate above and below current row */
    tmp = AA01;
    AA00 = fmnaOp(tmp, AA10, AA00);
    AA01 = mulOp(negOp(tmp), AA11);
    AA02 = fmnaOp(tmp, AA12, AA02);
    AA03 = fmnaOp(tmp, AA13, AA03);

    tmp = AA21;
    AA20 = fmnaOp(tmp, AA10, AA20);
    AA21 = mulOp(negOp(tmp), AA11);
    AA22 = fmnaOp(tmp, AA12, AA22);
    AA23 = fmnaOp(tmp, AA13, AA23);

    tmp = AA31;
    AA30 = fmnaOp(tmp, AA10, AA30);
    AA31 = mulOp(negOp(tmp), AA11);
    AA32 = fmnaOp(tmp, AA12, AA32);
    AA33 = fmnaOp(tmp, AA13, AA33);

    /****************** iteration 2 ****************/

#if USE_PIVOTING
    /* search pivot row */
    p = absOp(AA22);
    pvt = 2;
    t = absOp(AA32);
    if (t > p) {
      p = t;
      pvt = 3;
    }

    /* swap pivot row with row 2 */
    if (pvt == 3) {
      tmp = AA20;
      AA20 = AA30;
      AA30 = tmp;
      tmp = AA21;
      AA21 = AA31;
      AA31 = tmp;
      tmp = AA22;
      AA22 = AA32;
      AA32 = tmp;
      tmp = AA23;
      AA23 = AA33;
      AA33 = tmp;
      /* update permutation vector based on row swap */
      i = perm2;
      perm2 = perm3;
      perm3 = i;
    }
#endif // USE_PIVOTING

    /* scale current row */
    tmp = rcpOp(AA22);
    icol2 = perm2;
    AA20 = mulOp(tmp, AA20);
    AA21 = mulOp(tmp, AA21);
    AA22 = tmp;
    AA23 = mulOp(tmp, AA23);

    /* eliminate above and below current row */
    tmp = AA02;
    AA00 = fmnaOp(tmp, AA20, AA00);
    AA01 = fmnaOp(tmp, AA21, AA01);
    AA02 = mulOp(negOp(tmp), AA22);
    AA03 = fmnaOp(tmp, AA23, AA03);

    tmp = AA12;
    AA10 = fmnaOp(tmp, AA20, AA10);
    AA11 = fmnaOp(tmp, AA21, AA11);
    AA12 = mulOp(negOp(tmp), AA22);
    AA13 = fmnaOp(tmp, AA23, AA13);

    tmp = AA32;
    AA30 = fmnaOp(tmp, AA20, AA30);
    AA31 = fmnaOp(tmp, AA21, AA31);
    AA32 = mulOp(negOp(tmp), AA22);
    AA33 = fmnaOp(tmp, AA23, AA33);

    /****************** iteration 3 ****************/

    /* scale current row */
    tmp = rcpOp(AA33);
    icol3 = perm3;
    AA30 = mulOp(tmp, AA30);
    AA31 = mulOp(tmp, AA31);
    AA32 = mulOp(tmp, AA32);
    AA33 = tmp;

    /* eliminate above and below current row */
    tmp = AA03;
    AA00 = fmnaOp(tmp, AA30, AA00);
    AA01 = fmnaOp(tmp, AA31, AA01);
    AA02 = fmnaOp(tmp, AA32, AA02);
    AA03 = mulOp(negOp(tmp), AA33);

    tmp = AA13;
    AA10 = fmnaOp(tmp, AA30, AA10);
    AA11 = fmnaOp(tmp, AA31, AA11);
    AA12 = fmnaOp(tmp, AA32, AA12);
    AA13 = mulOp(negOp(tmp), AA33);

    tmp = AA23;
    AA20 = fmnaOp(tmp, AA30, AA20);
    AA21 = fmnaOp(tmp, AA31, AA21);
    AA22 = fmnaOp(tmp, AA32, AA22);
    AA23 = mulOp(negOp(tmp), AA33);

    /* sort columns into the correct order */
    Ainv(0, icol0) = AA00;
    Ainv(1, icol0) = AA10;
    Ainv(2, icol0) = AA20;
    Ainv(3, icol0) = AA30;
    Ainv(0, icol1) = AA01;
    Ainv(1, icol1) = AA11;
    Ainv(2, icol1) = AA21;
    Ainv(3, icol1) = AA31;
    Ainv(0, icol2) = AA02;
    Ainv(1, icol2) = AA12;
    Ainv(2, icol2) = AA22;
    Ainv(3, icol2) = AA32;
    Ainv(0, icol3) = AA03;
    Ainv(1, icol3) = AA13;
    Ainv(2, icol3) = AA23;
    Ainv(3, icol3) = AA33;
  }
}

template <typename T, int arch>
__global__ void lu_matinv_5x5_matrix_per_thread(const T *A, T *Ainv, int batch) {
  const int blkNum = blockIdx.y * gridDim.x + blockIdx.x;
  const int thrdNum = blkNum * blockDim.x + threadIdx.x;
  const int N = 5;
  int perm0, perm1, perm2, perm3, perm4;
  int icol0, icol1, icol2, icol3, icol4;
  T AA00, AA01, AA02, AA03, AA04;
  T AA10, AA11, AA12, AA13, AA14;
  T AA20, AA21, AA22, AA23, AA24;
  T AA30, AA31, AA32, AA33, AA34;
  T AA40, AA41, AA42, AA43, AA44;
  T tmp;
#if USE_PIVOTING
  typename config<T, arch>::absValType t;
  typename config<T, arch>::absValType p;
  int i, pvt;
#endif

  if (thrdNum < batch) {

    AA00 = A[0 * batch + thrdNum];
    AA10 = A[1 * batch + thrdNum];
    AA20 = A[2 * batch + thrdNum];
    AA30 = A[3 * batch + thrdNum];
    AA40 = A[4 * batch + thrdNum];
    AA01 = A[5 * batch + thrdNum];
    AA11 = A[6 * batch + thrdNum];
    AA21 = A[7 * batch + thrdNum];
    AA31 = A[8 * batch + thrdNum];
    AA41 = A[9 * batch + thrdNum];
    AA02 = A[10 * batch + thrdNum];
    AA12 = A[11 * batch + thrdNum];
    AA22 = A[12 * batch + thrdNum];
    AA32 = A[13 * batch + thrdNum];
    AA42 = A[14 * batch + thrdNum];
    AA03 = A[15 * batch + thrdNum];
    AA13 = A[16 * batch + thrdNum];
    AA23 = A[17 * batch + thrdNum];
    AA33 = A[18 * batch + thrdNum];
    AA43 = A[19 * batch + thrdNum];
    AA04 = A[20 * batch + thrdNum];
    AA14 = A[21 * batch + thrdNum];
    AA24 = A[22 * batch + thrdNum];
    AA34 = A[23 * batch + thrdNum];
    AA44 = A[24 * batch + thrdNum];

    perm0 = 0;
    perm1 = 1;
    perm2 = 2;
    perm3 = 3;
    perm4 = 4;

    /****************** iteration 0 ***********/

#if USE_PIVOTING
    /* search pivot row */
    p = absOp(AA00);
    pvt = 0;
    t = absOp(AA10);
    if (t > p) {
      p = t;
      pvt = 1;
    }
    t = absOp(AA20);
    if (t > p) {
      p = t;
      pvt = 2;
    }
    t = absOp(AA30);
    if (t > p) {
      p = t;
      pvt = 3;
    }
    t = absOp(AA40);
    if (t > p) {
      p = t;
      pvt = 4;
    }

    /* swap pivot row with row 0 */
    if (pvt == 1) {
      tmp = AA00;
      AA00 = AA10;
      AA10 = tmp;
      tmp = AA01;
      AA01 = AA11;
      AA11 = tmp;
      tmp = AA02;
      AA02 = AA12;
      AA12 = tmp;
      tmp = AA03;
      AA03 = AA13;
      AA13 = tmp;
      tmp = AA04;
      AA04 = AA14;
      AA14 = tmp;
      /* update permutation vector based on row swap */
      i = perm0;
      perm0 = perm1;
      perm1 = i;
    }
    if (pvt == 2) {
      tmp = AA00;
      AA00 = AA20;
      AA20 = tmp;
      tmp = AA01;
      AA01 = AA21;
      AA21 = tmp;
      tmp = AA02;
      AA02 = AA22;
      AA22 = tmp;
      tmp = AA03;
      AA03 = AA23;
      AA23 = tmp;
      tmp = AA04;
      AA04 = AA24;
      AA24 = tmp;
      /* update permutation vector based on row swap */
      i = perm0;
      perm0 = perm2;
      perm2 = i;
    }
    if (pvt == 3) {
      tmp = AA00;
      AA00 = AA30;
      AA30 = tmp;
      tmp = AA01;
      AA01 = AA31;
      AA31 = tmp;
      tmp = AA02;
      AA02 = AA32;
      AA32 = tmp;
      tmp = AA03;
      AA03 = AA33;
      AA33 = tmp;
      tmp = AA04;
      AA04 = AA34;
      AA34 = tmp;
      /* update permutation vector based on row swap */
      i = perm0;
      perm0 = perm3;
      perm3 = i;
    }
    if (pvt == 4) {
      tmp = AA00;
      AA00 = AA40;
      AA40 = tmp;
      tmp = AA01;
      AA01 = AA41;
      AA41 = tmp;
      tmp = AA02;
      AA02 = AA42;
      AA42 = tmp;
      tmp = AA03;
      AA03 = AA43;
      AA43 = tmp;
      tmp = AA04;
      AA04 = AA44;
      AA44 = tmp;
      /* update permutation vector based on row swap */
      i = perm0;
      perm0 = perm4;
      perm4 = i;
    }
#endif // USE_PIVOTING

    /* scale current row */
    tmp = rcpOp(AA00);
    icol0 = perm0;
    AA00 = tmp;
    AA01 = mulOp(tmp, AA01);
    AA02 = mulOp(tmp, AA02);
    AA03 = mulOp(tmp, AA03);
    AA04 = mulOp(tmp, AA04);

    /* eliminate above and below current row */
    tmp = AA10;
    AA10 = mulOp(negOp(tmp), AA00);
    AA11 = fmnaOp(tmp, AA01, AA11);
    AA12 = fmnaOp(tmp, AA02, AA12);
    AA13 = fmnaOp(tmp, AA03, AA13);
    AA14 = fmnaOp(tmp, AA04, AA14);

    tmp = AA20;
    AA20 = mulOp(negOp(tmp), AA00);
    AA21 = fmnaOp(tmp, AA01, AA21);
    AA22 = fmnaOp(tmp, AA02, AA22);
    AA23 = fmnaOp(tmp, AA03, AA23);
    AA24 = fmnaOp(tmp, AA04, AA24);

    tmp = AA30;
    AA30 = mulOp(negOp(tmp), AA00);
    AA31 = fmnaOp(tmp, AA01, AA31);
    AA32 = fmnaOp(tmp, AA02, AA32);
    AA33 = fmnaOp(tmp, AA03, AA33);
    AA34 = fmnaOp(tmp, AA04, AA34);

    tmp = AA40;
    AA40 = mulOp(negOp(tmp), AA00);
    AA41 = fmnaOp(tmp, AA01, AA41);
    AA42 = fmnaOp(tmp, AA02, AA42);
    AA43 = fmnaOp(tmp, AA03, AA43);
    AA44 = fmnaOp(tmp, AA04, AA44);

    /****************** iteration 1 ***********/

#if USE_PIVOTING
    /* search pivot row */
    p = absOp(AA11);
    pvt = 1;
    t = absOp(AA21);
    if (t > p) {
      p = t;
      pvt = 2;
    }
    t = absOp(AA31);
    if (t > p) {
      p = t;
      pvt = 3;
    }
    t = absOp(AA41);
    if (t > p) {
      p = t;
      pvt = 4;
    }

    /* swap pivot row with row 1 */
    if (pvt == 2) {
      tmp = AA10;
      AA10 = AA20;
      AA20 = tmp;
      tmp = AA11;
      AA11 = AA21;
      AA21 = tmp;
      tmp = AA12;
      AA12 = AA22;
      AA22 = tmp;
      tmp = AA13;
      AA13 = AA23;
      AA23 = tmp;
      tmp = AA14;
      AA14 = AA24;
      AA24 = tmp;
      /* update permutation vector based on row swap */
      i = perm1;
      perm1 = perm2;
      perm2 = i;
    }
    if (pvt == 3) {
      tmp = AA10;
      AA10 = AA30;
      AA30 = tmp;
      tmp = AA11;
      AA11 = AA31;
      AA31 = tmp;
      tmp = AA12;
      AA12 = AA32;
      AA32 = tmp;
      tmp = AA13;
      AA13 = AA33;
      AA33 = tmp;
      tmp = AA14;
      AA14 = AA34;
      AA34 = tmp;
      /* update permutation vector based on row swap */
      i = perm1;
      perm1 = perm3;
      perm3 = i;
    }
    if (pvt == 4) {
      tmp = AA10;
      AA10 = AA40;
      AA40 = tmp;
      tmp = AA11;
      AA11 = AA41;
      AA41 = tmp;
      tmp = AA12;
      AA12 = AA42;
      AA42 = tmp;
      tmp = AA13;
      AA13 = AA43;
      AA43 = tmp;
      tmp = AA14;
      AA14 = AA44;
      AA44 = tmp;
      /* update permutation vector based on row swap */
      i = perm1;
      perm1 = perm4;
      perm4 = i;
    }
#endif // USE_PIVOTING

    /* scale current row */
    tmp = rcpOp(AA11);
    icol1 = perm1;
    AA10 = mulOp(tmp, AA10);
    AA11 = tmp;
    AA12 = mulOp(tmp, AA12);
    AA13 = mulOp(tmp, AA13);
    AA14 = mulOp(tmp, AA14);

    /* eliminate above and below current row */
    tmp = AA01;
    AA00 = fmnaOp(tmp, AA10, AA00);
    AA01 = mulOp(negOp(tmp), AA11);
    AA02 = fmnaOp(tmp, AA12, AA02);
    AA03 = fmnaOp(tmp, AA13, AA03);
    AA04 = fmnaOp(tmp, AA14, AA04);

    tmp = AA21;
    AA20 = fmnaOp(tmp, AA10, AA20);
    AA21 = mulOp(negOp(tmp), AA11);
    AA22 = fmnaOp(tmp, AA12, AA22);
    AA23 = fmnaOp(tmp, AA13, AA23);
    AA24 = fmnaOp(tmp, AA14, AA24);

    tmp = AA31;
    AA30 = fmnaOp(tmp, AA10, AA30);
    AA31 = mulOp(negOp(tmp), AA11);
    AA32 = fmnaOp(tmp, AA12, AA32);
    AA33 = fmnaOp(tmp, AA13, AA33);
    AA34 = fmnaOp(tmp, AA14, AA34);

    tmp = AA41;
    AA40 = fmnaOp(tmp, AA10, AA40);
    AA41 = mulOp(negOp(tmp), AA11);
    AA42 = fmnaOp(tmp, AA12, AA42);
    AA43 = fmnaOp(tmp, AA13, AA43);
    AA44 = fmnaOp(tmp, AA14, AA44);

    /****************** iteration 2 ****************/

#if USE_PIVOTING
    /* search pivot row */
    p = absOp(AA22);
    pvt = 2;
    t = absOp(AA32);
    if (t > p) {
      p = t;
      pvt = 3;
    }
    t = absOp(AA42);
    if (t > p) {
      p = t;
      pvt = 4;
    }

    /* swap pivot row with row 2 */
    if (pvt == 3) {
      tmp = AA20;
      AA20 = AA30;
      AA30 = tmp;
      tmp = AA21;
      AA21 = AA31;
      AA31 = tmp;
      tmp = AA22;
      AA22 = AA32;
      AA32 = tmp;
      tmp = AA23;
      AA23 = AA33;
      AA33 = tmp;
      tmp = AA24;
      AA24 = AA34;
      AA34 = tmp;
      /* update permutation vector based on row swap */
      i = perm2;
      perm2 = perm3;
      perm3 = i;
    }
    if (pvt == 4) {
      tmp = AA20;
      AA20 = AA40;
      AA40 = tmp;
      tmp = AA21;
      AA21 = AA41;
      AA41 = tmp;
      tmp = AA22;
      AA22 = AA42;
      AA42 = tmp;
      tmp = AA23;
      AA23 = AA43;
      AA43 = tmp;
      tmp = AA24;
      AA24 = AA44;
      AA44 = tmp;
      /* update permutation vector based on row swap */
      i = perm2;
      perm2 = perm4;
      perm4 = i;
    }
#endif // USE_PIVOTING

    /* scale current row */
    tmp = rcpOp(AA22);
    icol2 = perm2;
    AA20 = mulOp(tmp, AA20);
    AA21 = mulOp(tmp, AA21);
    AA22 = tmp;
    AA23 = mulOp(tmp, AA23);
    AA24 = mulOp(tmp, AA24);

    /* eliminate above and below current row */
    tmp = AA02;
    AA00 = fmnaOp(tmp, AA20, AA00);
    AA01 = fmnaOp(tmp, AA21, AA01);
    AA02 = mulOp(negOp(tmp), AA22);
    AA03 = fmnaOp(tmp, AA23, AA03);
    AA04 = fmnaOp(tmp, AA24, AA04);

    tmp = AA12;
    AA10 = fmnaOp(tmp, AA20, AA10);
    AA11 = fmnaOp(tmp, AA21, AA11);
    AA12 = mulOp(negOp(tmp), AA22);
    AA13 = fmnaOp(tmp, AA23, AA13);
    AA14 = fmnaOp(tmp, AA24, AA14);

    tmp = AA32;
    AA30 = fmnaOp(tmp, AA20, AA30);
    AA31 = fmnaOp(tmp, AA21, AA31);
    AA32 = mulOp(negOp(tmp), AA22);
    AA33 = fmnaOp(tmp, AA23, AA33);
    AA34 = fmnaOp(tmp, AA24, AA34);

    tmp = AA42;
    AA40 = fmnaOp(tmp, AA20, AA40);
    AA41 = fmnaOp(tmp, AA21, AA41);
    AA42 = mulOp(negOp(tmp), AA22);
    AA43 = fmnaOp(tmp, AA23, AA43);
    AA44 = fmnaOp(tmp, AA24, AA44);

    /****************** iteration 3 ****************/

#if USE_PIVOTING
    /* search pivot row */
    p = absOp(AA33);
    pvt = 3;
    t = absOp(AA43);
    if (t > p) {
      p = t;
      pvt = 4;
    }

    /* swap pivot row with row 3 */
    if (pvt == 4) {
      tmp = AA30;
      AA30 = AA40;
      AA40 = tmp;
      tmp = AA31;
      AA31 = AA41;
      AA41 = tmp;
      tmp = AA32;
      AA32 = AA42;
      AA42 = tmp;
      tmp = AA33;
      AA33 = AA43;
      AA43 = tmp;
      tmp = AA34;
      AA34 = AA44;
      AA44 = tmp;
      /* update permutation vector based on row swap */
      i = perm3;
      perm3 = perm4;
      perm4 = i;
    }
#endif // USE_PIVOTING

    /* scale current row */
    tmp = rcpOp(AA33);
    icol3 = perm3;
    AA30 = mulOp(tmp, AA30);
    AA31 = mulOp(tmp, AA31);
    AA32 = mulOp(tmp, AA32);
    AA33 = tmp;
    AA34 = mulOp(tmp, AA34);

    /* eliminate above and below current row */
    tmp = AA03;
    AA00 = fmnaOp(tmp, AA30, AA00);
    AA01 = fmnaOp(tmp, AA31, AA01);
    AA02 = fmnaOp(tmp, AA32, AA02);
    AA03 = mulOp(negOp(tmp), AA33);
    AA04 = fmnaOp(tmp, AA34, AA04);

    tmp = AA13;
    AA10 = fmnaOp(tmp, AA30, AA10);
    AA11 = fmnaOp(tmp, AA31, AA11);
    AA12 = fmnaOp(tmp, AA32, AA12);
    AA13 = mulOp(negOp(tmp), AA33);
    AA14 = fmnaOp(tmp, AA34, AA14);

    tmp = AA23;
    AA20 = fmnaOp(tmp, AA30, AA20);
    AA21 = fmnaOp(tmp, AA31, AA21);
    AA22 = fmnaOp(tmp, AA32, AA22);
    AA23 = mulOp(negOp(tmp), AA33);
    AA24 = fmnaOp(tmp, AA34, AA24);

    tmp = AA43;
    AA40 = fmnaOp(tmp, AA30, AA40);
    AA41 = fmnaOp(tmp, AA31, AA41);
    AA42 = fmnaOp(tmp, AA32, AA42);
    AA43 = mulOp(negOp(tmp), AA33);
    AA44 = fmnaOp(tmp, AA34, AA44);

    /****************** iteration 4 ****************/

    /* scale current row */
    tmp = rcpOp(AA44);
    icol4 = perm4;
    AA40 = mulOp(tmp, AA40);
    AA41 = mulOp(tmp, AA41);
    AA42 = mulOp(tmp, AA42);
    AA43 = mulOp(tmp, AA43);
    AA44 = tmp;

    /* eliminate above and below current row */
    tmp = AA04;
    AA00 = fmnaOp(tmp, AA40, AA00);
    AA01 = fmnaOp(tmp, AA41, AA01);
    AA02 = fmnaOp(tmp, AA42, AA02);
    AA03 = fmnaOp(tmp, AA43, AA03);
    AA04 = mulOp(negOp(tmp), AA44);

    tmp = AA14;
    AA10 = fmnaOp(tmp, AA40, AA10);
    AA11 = fmnaOp(tmp, AA41, AA11);
    AA12 = fmnaOp(tmp, AA42, AA12);
    AA13 = fmnaOp(tmp, AA43, AA13);
    AA14 = mulOp(negOp(tmp), AA44);

    tmp = AA24;
    AA20 = fmnaOp(tmp, AA40, AA20);
    AA21 = fmnaOp(tmp, AA41, AA21);
    AA22 = fmnaOp(tmp, AA42, AA22);
    AA23 = fmnaOp(tmp, AA43, AA23);
    AA24 = mulOp(negOp(tmp), AA44);

    tmp = AA34;
    AA30 = fmnaOp(tmp, AA40, AA30);
    AA31 = fmnaOp(tmp, AA41, AA31);
    AA32 = fmnaOp(tmp, AA42, AA32);
    AA33 = fmnaOp(tmp, AA43, AA33);
    AA34 = mulOp(negOp(tmp), AA44);

    /* sort columns into the correct order */
    Ainv(0, icol0) = AA00;
    Ainv(1, icol0) = AA10;
    Ainv(2, icol0) = AA20;
    Ainv(3, icol0) = AA30;
    Ainv(4, icol0) = AA40;
    Ainv(0, icol1) = AA01;
    Ainv(1, icol1) = AA11;
    Ainv(2, icol1) = AA21;
    Ainv(3, icol1) = AA31;
    Ainv(4, icol1) = AA41;
    Ainv(0, icol2) = AA02;
    Ainv(1, icol2) = AA12;
    Ainv(2, icol2) = AA22;
    Ainv(3, icol2) = AA32;
    Ainv(4, icol2) = AA42;
    Ainv(0, icol3) = AA03;
    Ainv(1, icol3) = AA13;
    Ainv(2, icol3) = AA23;
    Ainv(3, icol3) = AA33;
    Ainv(4, icol3) = AA43;
    Ainv(0, icol4) = AA04;
    Ainv(1, icol4) = AA14;
    Ainv(2, icol4) = AA24;
    Ainv(3, icol4) = AA34;
    Ainv(4, icol4) = AA44;
  }
}

template <typename T, int arch>
__global__ void lu_matinv_6x6_matrix_per_thread(const T *A, T *Ainv, int batch) {
  const int blkNum = blockIdx.y * gridDim.x + blockIdx.x;
  const int thrdNum = blkNum * blockDim.x + threadIdx.x;
  const int N = 6;
  int perm0, perm1, perm2, perm3, perm4, perm5;
  int icol0, icol1, icol2, icol3, icol4, icol5;
  T AA00, AA01, AA02, AA03, AA04, AA05;
  T AA10, AA11, AA12, AA13, AA14, AA15;
  T AA20, AA21, AA22, AA23, AA24, AA25;
  T AA30, AA31, AA32, AA33, AA34, AA35;
  T AA40, AA41, AA42, AA43, AA44, AA45;
  T AA50, AA51, AA52, AA53, AA54, AA55;
  T tmp;
#if USE_PIVOTING
  typename config<T, arch>::absValType t;
  typename config<T, arch>::absValType p;
  int i, pvt;
#endif

  if (thrdNum < batch) {

    AA00 = A[0 * batch + thrdNum];
    AA10 = A[1 * batch + thrdNum];
    AA20 = A[2 * batch + thrdNum];
    AA30 = A[3 * batch + thrdNum];
    AA40 = A[4 * batch + thrdNum];
    AA50 = A[5 * batch + thrdNum];
    AA01 = A[6 * batch + thrdNum];
    AA11 = A[7 * batch + thrdNum];
    AA21 = A[8 * batch + thrdNum];
    AA31 = A[9 * batch + thrdNum];
    AA41 = A[10 * batch + thrdNum];
    AA51 = A[11 * batch + thrdNum];
    AA02 = A[12 * batch + thrdNum];
    AA12 = A[13 * batch + thrdNum];
    AA22 = A[14 * batch + thrdNum];
    AA32 = A[15 * batch + thrdNum];
    AA42 = A[16 * batch + thrdNum];
    AA52 = A[17 * batch + thrdNum];
    AA03 = A[18 * batch + thrdNum];
    AA13 = A[19 * batch + thrdNum];
    AA23 = A[20 * batch + thrdNum];
    AA33 = A[21 * batch + thrdNum];
    AA43 = A[22 * batch + thrdNum];
    AA53 = A[23 * batch + thrdNum];
    AA04 = A[24 * batch + thrdNum];
    AA14 = A[25 * batch + thrdNum];
    AA24 = A[26 * batch + thrdNum];
    AA34 = A[27 * batch + thrdNum];
    AA44 = A[28 * batch + thrdNum];
    AA54 = A[29 * batch + thrdNum];
    AA05 = A[30 * batch + thrdNum];
    AA15 = A[31 * batch + thrdNum];
    AA25 = A[32 * batch + thrdNum];
    AA35 = A[33 * batch + thrdNum];
    AA45 = A[34 * batch + thrdNum];
    AA55 = A[35 * batch + thrdNum];

    perm0 = 0;
    perm1 = 1;
    perm2 = 2;
    perm3 = 3;
    perm4 = 4;
    perm5 = 5;

    /****************** iteration 0 ***********/

#if USE_PIVOTING
    /* search pivot row */
    p = absOp(AA00);
    pvt = 0;
    t = absOp(AA10);
    if (t > p) {
      p = t;
      pvt = 1;
    }
    t = absOp(AA20);
    if (t > p) {
      p = t;
      pvt = 2;
    }
    t = absOp(AA30);
    if (t > p) {
      p = t;
      pvt = 3;
    }
    t = absOp(AA40);
    if (t > p) {
      p = t;
      pvt = 4;
    }
    t = absOp(AA50);
    if (t > p) {
      p = t;
      pvt = 5;
    }

    /* swap pivot row with row 0 */
    if (pvt == 1) {
      tmp = AA00;
      AA00 = AA10;
      AA10 = tmp;
      tmp = AA01;
      AA01 = AA11;
      AA11 = tmp;
      tmp = AA02;
      AA02 = AA12;
      AA12 = tmp;
      tmp = AA03;
      AA03 = AA13;
      AA13 = tmp;
      tmp = AA04;
      AA04 = AA14;
      AA14 = tmp;
      tmp = AA05;
      AA05 = AA15;
      AA15 = tmp;
      /* update permutation vector based on row swap */
      i = perm0;
      perm0 = perm1;
      perm1 = i;
    }
    if (pvt == 2) {
      tmp = AA00;
      AA00 = AA20;
      AA20 = tmp;
      tmp = AA01;
      AA01 = AA21;
      AA21 = tmp;
      tmp = AA02;
      AA02 = AA22;
      AA22 = tmp;
      tmp = AA03;
      AA03 = AA23;
      AA23 = tmp;
      tmp = AA04;
      AA04 = AA24;
      AA24 = tmp;
      tmp = AA05;
      AA05 = AA25;
      AA25 = tmp;
      /* update permutation vector based on row swap */
      i = perm0;
      perm0 = perm2;
      perm2 = i;
    }
    if (pvt == 3) {
      tmp = AA00;
      AA00 = AA30;
      AA30 = tmp;
      tmp = AA01;
      AA01 = AA31;
      AA31 = tmp;
      tmp = AA02;
      AA02 = AA32;
      AA32 = tmp;
      tmp = AA03;
      AA03 = AA33;
      AA33 = tmp;
      tmp = AA04;
      AA04 = AA34;
      AA34 = tmp;
      tmp = AA05;
      AA05 = AA35;
      AA35 = tmp;
      /* update permutation vector based on row swap */
      i = perm0;
      perm0 = perm3;
      perm3 = i;
    }
    if (pvt == 4) {
      tmp = AA00;
      AA00 = AA40;
      AA40 = tmp;
      tmp = AA01;
      AA01 = AA41;
      AA41 = tmp;
      tmp = AA02;
      AA02 = AA42;
      AA42 = tmp;
      tmp = AA03;
      AA03 = AA43;
      AA43 = tmp;
      tmp = AA04;
      AA04 = AA44;
      AA44 = tmp;
      tmp = AA05;
      AA05 = AA45;
      AA45 = tmp;
      /* update permutation vector based on row swap */
      i = perm0;
      perm0 = perm4;
      perm4 = i;
    }
    if (pvt == 5) {
      tmp = AA00;
      AA00 = AA50;
      AA50 = tmp;
      tmp = AA01;
      AA01 = AA51;
      AA51 = tmp;
      tmp = AA02;
      AA02 = AA52;
      AA52 = tmp;
      tmp = AA03;
      AA03 = AA53;
      AA53 = tmp;
      tmp = AA04;
      AA04 = AA54;
      AA54 = tmp;
      tmp = AA05;
      AA05 = AA55;
      AA55 = tmp;
      /* update permutation vector based on row swap */
      i = perm0;
      perm0 = perm5;
      perm5 = i;
    }
#endif // USE_PIVOTING

    /* scale current row */
    tmp = rcpOp(AA00);
    icol0 = perm0;
    AA00 = tmp;
    AA01 = mulOp(tmp, AA01);
    AA02 = mulOp(tmp, AA02);
    AA03 = mulOp(tmp, AA03);
    AA04 = mulOp(tmp, AA04);
    AA05 = mulOp(tmp, AA05);

    /* eliminate above and below current row */
    tmp = AA10;
    AA10 = mulOp(negOp(tmp), AA00);
    AA11 = fmnaOp(tmp, AA01, AA11);
    AA12 = fmnaOp(tmp, AA02, AA12);
    AA13 = fmnaOp(tmp, AA03, AA13);
    AA14 = fmnaOp(tmp, AA04, AA14);
    AA15 = fmnaOp(tmp, AA05, AA15);

    tmp = AA20;
    AA20 = mulOp(negOp(tmp), AA00);
    AA21 = fmnaOp(tmp, AA01, AA21);
    AA22 = fmnaOp(tmp, AA02, AA22);
    AA23 = fmnaOp(tmp, AA03, AA23);
    AA24 = fmnaOp(tmp, AA04, AA24);
    AA25 = fmnaOp(tmp, AA05, AA25);

    tmp = AA30;
    AA30 = mulOp(negOp(tmp), AA00);
    AA31 = fmnaOp(tmp, AA01, AA31);
    AA32 = fmnaOp(tmp, AA02, AA32);
    AA33 = fmnaOp(tmp, AA03, AA33);
    AA34 = fmnaOp(tmp, AA04, AA34);
    AA35 = fmnaOp(tmp, AA05, AA35);

    tmp = AA40;
    AA40 = mulOp(negOp(tmp), AA00);
    AA41 = fmnaOp(tmp, AA01, AA41);
    AA42 = fmnaOp(tmp, AA02, AA42);
    AA43 = fmnaOp(tmp, AA03, AA43);
    AA44 = fmnaOp(tmp, AA04, AA44);
    AA45 = fmnaOp(tmp, AA05, AA45);

    tmp = AA50;
    AA50 = mulOp(negOp(tmp), AA00);
    AA51 = fmnaOp(tmp, AA01, AA51);
    AA52 = fmnaOp(tmp, AA02, AA52);
    AA53 = fmnaOp(tmp, AA03, AA53);
    AA54 = fmnaOp(tmp, AA04, AA54);
    AA55 = fmnaOp(tmp, AA05, AA55);

    /****************** iteration 1 ***********/

#if USE_PIVOTING
    /* search pivot row */
    p = absOp(AA11);
    pvt = 1;
    t = absOp(AA21);
    if (t > p) {
      p = t;
      pvt = 2;
    }
    t = absOp(AA31);
    if (t > p) {
      p = t;
      pvt = 3;
    }
    t = absOp(AA41);
    if (t > p) {
      p = t;
      pvt = 4;
    }
    t = absOp(AA51);
    if (t > p) {
      p = t;
      pvt = 5;
    }

    /* swap pivot row with row 1 */
    if (pvt == 2) {
      tmp = AA10;
      AA10 = AA20;
      AA20 = tmp;
      tmp = AA11;
      AA11 = AA21;
      AA21 = tmp;
      tmp = AA12;
      AA12 = AA22;
      AA22 = tmp;
      tmp = AA13;
      AA13 = AA23;
      AA23 = tmp;
      tmp = AA14;
      AA14 = AA24;
      AA24 = tmp;
      tmp = AA15;
      AA15 = AA25;
      AA25 = tmp;
      /* update permutation vector based on row swap */
      i = perm1;
      perm1 = perm2;
      perm2 = i;
    }
    if (pvt == 3) {
      tmp = AA10;
      AA10 = AA30;
      AA30 = tmp;
      tmp = AA11;
      AA11 = AA31;
      AA31 = tmp;
      tmp = AA12;
      AA12 = AA32;
      AA32 = tmp;
      tmp = AA13;
      AA13 = AA33;
      AA33 = tmp;
      tmp = AA14;
      AA14 = AA34;
      AA34 = tmp;
      tmp = AA15;
      AA15 = AA35;
      AA35 = tmp;
      /* update permutation vector based on row swap */
      i = perm1;
      perm1 = perm3;
      perm3 = i;
    }
    if (pvt == 4) {
      tmp = AA10;
      AA10 = AA40;
      AA40 = tmp;
      tmp = AA11;
      AA11 = AA41;
      AA41 = tmp;
      tmp = AA12;
      AA12 = AA42;
      AA42 = tmp;
      tmp = AA13;
      AA13 = AA43;
      AA43 = tmp;
      tmp = AA14;
      AA14 = AA44;
      AA44 = tmp;
      tmp = AA15;
      AA15 = AA45;
      AA45 = tmp;
      /* update permutation vector based on row swap */
      i = perm1;
      perm1 = perm4;
      perm4 = i;
    }
    if (pvt == 5) {
      tmp = AA10;
      AA10 = AA50;
      AA50 = tmp;
      tmp = AA11;
      AA11 = AA51;
      AA51 = tmp;
      tmp = AA12;
      AA12 = AA52;
      AA52 = tmp;
      tmp = AA13;
      AA13 = AA53;
      AA53 = tmp;
      tmp = AA14;
      AA14 = AA54;
      AA54 = tmp;
      tmp = AA15;
      AA15 = AA55;
      AA55 = tmp;
      /* update permutation vector based on row swap */
      i = perm1;
      perm1 = perm5;
      perm5 = i;
    }
#endif // USE_PIVOTING

    /* scale current row */
    tmp = rcpOp(AA11);
    icol1 = perm1;
    AA10 = mulOp(tmp, AA10);
    AA11 = tmp;
    AA12 = mulOp(tmp, AA12);
    AA13 = mulOp(tmp, AA13);
    AA14 = mulOp(tmp, AA14);
    AA15 = mulOp(tmp, AA15);

    /* eliminate above and below current row */
    tmp = AA01;
    AA00 = fmnaOp(tmp, AA10, AA00);
    AA01 = mulOp(negOp(tmp), AA11);
    AA02 = fmnaOp(tmp, AA12, AA02);
    AA03 = fmnaOp(tmp, AA13, AA03);
    AA04 = fmnaOp(tmp, AA14, AA04);
    AA05 = fmnaOp(tmp, AA15, AA05);

    tmp = AA21;
    AA20 = fmnaOp(tmp, AA10, AA20);
    AA21 = mulOp(negOp(tmp), AA11);
    AA22 = fmnaOp(tmp, AA12, AA22);
    AA23 = fmnaOp(tmp, AA13, AA23);
    AA24 = fmnaOp(tmp, AA14, AA24);
    AA25 = fmnaOp(tmp, AA15, AA25);

    tmp = AA31;
    AA30 = fmnaOp(tmp, AA10, AA30);
    AA31 = mulOp(negOp(tmp), AA11);
    AA32 = fmnaOp(tmp, AA12, AA32);
    AA33 = fmnaOp(tmp, AA13, AA33);
    AA34 = fmnaOp(tmp, AA14, AA34);
    AA35 = fmnaOp(tmp, AA15, AA35);

    tmp = AA41;
    AA40 = fmnaOp(tmp, AA10, AA40);
    AA41 = mulOp(negOp(tmp), AA11);
    AA42 = fmnaOp(tmp, AA12, AA42);
    AA43 = fmnaOp(tmp, AA13, AA43);
    AA44 = fmnaOp(tmp, AA14, AA44);
    AA45 = fmnaOp(tmp, AA15, AA45);

    tmp = AA51;
    AA50 = fmnaOp(tmp, AA10, AA50);
    AA51 = mulOp(negOp(tmp), AA11);
    AA52 = fmnaOp(tmp, AA12, AA52);
    AA53 = fmnaOp(tmp, AA13, AA53);
    AA54 = fmnaOp(tmp, AA14, AA54);
    AA55 = fmnaOp(tmp, AA15, AA55);

    /****************** iteration 2 ****************/

#if USE_PIVOTING
    /* search pivot row */
    p = absOp(AA22);
    pvt = 2;
    t = absOp(AA32);
    if (t > p) {
      p = t;
      pvt = 3;
    }
    t = absOp(AA42);
    if (t > p) {
      p = t;
      pvt = 4;
    }
    t = absOp(AA52);
    if (t > p) {
      p = t;
      pvt = 5;
    }

    /* swap pivot row with row 2 */
    if (pvt == 3) {
      tmp = AA20;
      AA20 = AA30;
      AA30 = tmp;
      tmp = AA21;
      AA21 = AA31;
      AA31 = tmp;
      tmp = AA22;
      AA22 = AA32;
      AA32 = tmp;
      tmp = AA23;
      AA23 = AA33;
      AA33 = tmp;
      tmp = AA24;
      AA24 = AA34;
      AA34 = tmp;
      tmp = AA25;
      AA25 = AA35;
      AA35 = tmp;
      /* update permutation vector based on row swap */
      i = perm2;
      perm2 = perm3;
      perm3 = i;
    }
    if (pvt == 4) {
      tmp = AA20;
      AA20 = AA40;
      AA40 = tmp;
      tmp = AA21;
      AA21 = AA41;
      AA41 = tmp;
      tmp = AA22;
      AA22 = AA42;
      AA42 = tmp;
      tmp = AA23;
      AA23 = AA43;
      AA43 = tmp;
      tmp = AA24;
      AA24 = AA44;
      AA44 = tmp;
      tmp = AA25;
      AA25 = AA45;
      AA45 = tmp;
      /* update permutation vector based on row swap */
      i = perm2;
      perm2 = perm4;
      perm4 = i;
    }
    if (pvt == 5) {
      tmp = AA20;
      AA20 = AA50;
      AA50 = tmp;
      tmp = AA21;
      AA21 = AA51;
      AA51 = tmp;
      tmp = AA22;
      AA22 = AA52;
      AA52 = tmp;
      tmp = AA23;
      AA23 = AA53;
      AA53 = tmp;
      tmp = AA24;
      AA24 = AA54;
      AA54 = tmp;
      tmp = AA25;
      AA25 = AA55;
      AA55 = tmp;
      /* update permutation vector based on row swap */
      i = perm2;
      perm2 = perm5;
      perm5 = i;
    }
#endif // USE_PIVOTING

    /* scale current row */
    tmp = rcpOp(AA22);
    icol2 = perm2;
    AA20 = mulOp(tmp, AA20);
    AA21 = mulOp(tmp, AA21);
    AA22 = tmp;
    AA23 = mulOp(tmp, AA23);
    AA24 = mulOp(tmp, AA24);
    AA25 = mulOp(tmp, AA25);

    /* eliminate above and below current row */
    tmp = AA02;
    AA00 = fmnaOp(tmp, AA20, AA00);
    AA01 = fmnaOp(tmp, AA21, AA01);
    AA02 = mulOp(negOp(tmp), AA22);
    AA03 = fmnaOp(tmp, AA23, AA03);
    AA04 = fmnaOp(tmp, AA24, AA04);
    AA05 = fmnaOp(tmp, AA25, AA05);

    tmp = AA12;
    AA10 = fmnaOp(tmp, AA20, AA10);
    AA11 = fmnaOp(tmp, AA21, AA11);
    AA12 = mulOp(negOp(tmp), AA22);
    AA13 = fmnaOp(tmp, AA23, AA13);
    AA14 = fmnaOp(tmp, AA24, AA14);
    AA15 = fmnaOp(tmp, AA25, AA15);

    tmp = AA32;
    AA30 = fmnaOp(tmp, AA20, AA30);
    AA31 = fmnaOp(tmp, AA21, AA31);
    AA32 = mulOp(negOp(tmp), AA22);
    AA33 = fmnaOp(tmp, AA23, AA33);
    AA34 = fmnaOp(tmp, AA24, AA34);
    AA35 = fmnaOp(tmp, AA25, AA35);

    tmp = AA42;
    AA40 = fmnaOp(tmp, AA20, AA40);
    AA41 = fmnaOp(tmp, AA21, AA41);
    AA42 = mulOp(negOp(tmp), AA22);
    AA43 = fmnaOp(tmp, AA23, AA43);
    AA44 = fmnaOp(tmp, AA24, AA44);
    AA45 = fmnaOp(tmp, AA25, AA45);

    tmp = AA52;
    AA50 = fmnaOp(tmp, AA20, AA50);
    AA51 = fmnaOp(tmp, AA21, AA51);
    AA52 = mulOp(negOp(tmp), AA22);
    AA53 = fmnaOp(tmp, AA23, AA53);
    AA54 = fmnaOp(tmp, AA24, AA54);
    AA55 = fmnaOp(tmp, AA25, AA55);

    /****************** iteration 3 ****************/

#if USE_PIVOTING
    /* search pivot row */
    p = absOp(AA33);
    pvt = 3;
    t = absOp(AA43);
    if (t > p) {
      p = t;
      pvt = 4;
    }
    t = absOp(AA53);
    if (t > p) {
      p = t;
      pvt = 5;
    }

    /* swap pivot row with row 3 */
    if (pvt == 4) {
      tmp = AA30;
      AA30 = AA40;
      AA40 = tmp;
      tmp = AA31;
      AA31 = AA41;
      AA41 = tmp;
      tmp = AA32;
      AA32 = AA42;
      AA42 = tmp;
      tmp = AA33;
      AA33 = AA43;
      AA43 = tmp;
      tmp = AA34;
      AA34 = AA44;
      AA44 = tmp;
      tmp = AA35;
      AA35 = AA45;
      AA45 = tmp;
      /* update permutation vector based on row swap */
      i = perm3;
      perm3 = perm4;
      perm4 = i;
    }
    if (pvt == 5) {
      tmp = AA30;
      AA30 = AA50;
      AA50 = tmp;
      tmp = AA31;
      AA31 = AA51;
      AA51 = tmp;
      tmp = AA32;
      AA32 = AA52;
      AA52 = tmp;
      tmp = AA33;
      AA33 = AA53;
      AA53 = tmp;
      tmp = AA34;
      AA34 = AA54;
      AA54 = tmp;
      tmp = AA35;
      AA35 = AA55;
      AA55 = tmp;
      /* update permutation vector based on row swap */
      i = perm3;
      perm3 = perm5;
      perm5 = i;
    }
#endif // USE_PIVOTING

    /* scale current row */
    tmp = rcpOp(AA33);
    icol3 = perm3;
    AA30 = mulOp(tmp, AA30);
    AA31 = mulOp(tmp, AA31);
    AA32 = mulOp(tmp, AA32);
    AA33 = tmp;
    AA34 = mulOp(tmp, AA34);
    AA35 = mulOp(tmp, AA35);

    /* eliminate above and below current row */
    tmp = AA03;
    AA00 = fmnaOp(tmp, AA30, AA00);
    AA01 = fmnaOp(tmp, AA31, AA01);
    AA02 = fmnaOp(tmp, AA32, AA02);
    AA03 = mulOp(negOp(tmp), AA33);
    AA04 = fmnaOp(tmp, AA34, AA04);
    AA05 = fmnaOp(tmp, AA35, AA05);

    tmp = AA13;
    AA10 = fmnaOp(tmp, AA30, AA10);
    AA11 = fmnaOp(tmp, AA31, AA11);
    AA12 = fmnaOp(tmp, AA32, AA12);
    AA13 = mulOp(negOp(tmp), AA33);
    AA14 = fmnaOp(tmp, AA34, AA14);
    AA15 = fmnaOp(tmp, AA35, AA15);

    tmp = AA23;
    AA20 = fmnaOp(tmp, AA30, AA20);
    AA21 = fmnaOp(tmp, AA31, AA21);
    AA22 = fmnaOp(tmp, AA32, AA22);
    AA23 = mulOp(negOp(tmp), AA33);
    AA24 = fmnaOp(tmp, AA34, AA24);
    AA25 = fmnaOp(tmp, AA35, AA25);

    tmp = AA43;
    AA40 = fmnaOp(tmp, AA30, AA40);
    AA41 = fmnaOp(tmp, AA31, AA41);
    AA42 = fmnaOp(tmp, AA32, AA42);
    AA43 = mulOp(negOp(tmp), AA33);
    AA44 = fmnaOp(tmp, AA34, AA44);
    AA45 = fmnaOp(tmp, AA35, AA45);

    tmp = AA53;
    AA50 = fmnaOp(tmp, AA30, AA50);
    AA51 = fmnaOp(tmp, AA31, AA51);
    AA52 = fmnaOp(tmp, AA32, AA52);
    AA53 = mulOp(negOp(tmp), AA33);
    AA54 = fmnaOp(tmp, AA34, AA54);
    AA55 = fmnaOp(tmp, AA35, AA55);

    /****************** iteration 4 ****************/

#if USE_PIVOTING
    /* search pivot row */
    p = absOp(AA44);
    pvt = 4;
    t = absOp(AA54);
    if (t > p) {
      p = t;
      pvt = 5;
    }

    /* swap pivot row with row 4 */
    if (pvt == 5) {
      tmp = AA40;
      AA40 = AA50;
      AA50 = tmp;
      tmp = AA41;
      AA41 = AA51;
      AA51 = tmp;
      tmp = AA42;
      AA42 = AA52;
      AA52 = tmp;
      tmp = AA43;
      AA43 = AA53;
      AA53 = tmp;
      tmp = AA44;
      AA44 = AA54;
      AA54 = tmp;
      tmp = AA45;
      AA45 = AA55;
      AA55 = tmp;
      /* update permutation vector based on row swap */
      i = perm4;
      perm4 = perm5;
      perm5 = i;
    }
#endif // USE_PIVOTING

    /* scale current row */
    tmp = rcpOp(AA44);
    icol4 = perm4;
    AA40 = mulOp(tmp, AA40);
    AA41 = mulOp(tmp, AA41);
    AA42 = mulOp(tmp, AA42);
    AA43 = mulOp(tmp, AA43);
    AA44 = tmp;
    AA45 = mulOp(tmp, AA45);

    /* eliminate above and below current row */
    tmp = AA04;
    AA00 = fmnaOp(tmp, AA40, AA00);
    AA01 = fmnaOp(tmp, AA41, AA01);
    AA02 = fmnaOp(tmp, AA42, AA02);
    AA03 = fmnaOp(tmp, AA43, AA03);
    AA04 = mulOp(negOp(tmp), AA44);
    AA05 = fmnaOp(tmp, AA45, AA05);

    tmp = AA14;
    AA10 = fmnaOp(tmp, AA40, AA10);
    AA11 = fmnaOp(tmp, AA41, AA11);
    AA12 = fmnaOp(tmp, AA42, AA12);
    AA13 = fmnaOp(tmp, AA43, AA13);
    AA14 = mulOp(negOp(tmp), AA44);
    AA15 = fmnaOp(tmp, AA45, AA15);

    tmp = AA24;
    AA20 = fmnaOp(tmp, AA40, AA20);
    AA21 = fmnaOp(tmp, AA41, AA21);
    AA22 = fmnaOp(tmp, AA42, AA22);
    AA23 = fmnaOp(tmp, AA43, AA23);
    AA24 = mulOp(negOp(tmp), AA44);
    AA25 = fmnaOp(tmp, AA45, AA25);

    tmp = AA34;
    AA30 = fmnaOp(tmp, AA40, AA30);
    AA31 = fmnaOp(tmp, AA41, AA31);
    AA32 = fmnaOp(tmp, AA42, AA32);
    AA33 = fmnaOp(tmp, AA43, AA33);
    AA34 = mulOp(negOp(tmp), AA44);
    AA35 = fmnaOp(tmp, AA45, AA35);

    tmp = AA54;
    AA50 = fmnaOp(tmp, AA40, AA50);
    AA51 = fmnaOp(tmp, AA41, AA51);
    AA52 = fmnaOp(tmp, AA42, AA52);
    AA53 = fmnaOp(tmp, AA43, AA53);
    AA54 = mulOp(negOp(tmp), AA44);
    AA55 = fmnaOp(tmp, AA45, AA55);

    /****************** iteration 5 ****************/

    /* scale current row */
    tmp = rcpOp(AA55);
    icol5 = perm5;
    AA50 = mulOp(tmp, AA50);
    AA51 = mulOp(tmp, AA51);
    AA52 = mulOp(tmp, AA52);
    AA53 = mulOp(tmp, AA53);
    AA54 = mulOp(tmp, AA54);
    AA55 = tmp;

    /* eliminate above and below current row */
    tmp = AA05;
    AA00 = fmnaOp(tmp, AA50, AA00);
    AA01 = fmnaOp(tmp, AA51, AA01);
    AA02 = fmnaOp(tmp, AA52, AA02);
    AA03 = fmnaOp(tmp, AA53, AA03);
    AA04 = fmnaOp(tmp, AA54, AA04);
    AA05 = mulOp(negOp(tmp), AA55);

    tmp = AA15;
    AA10 = fmnaOp(tmp, AA50, AA10);
    AA11 = fmnaOp(tmp, AA51, AA11);
    AA12 = fmnaOp(tmp, AA52, AA12);
    AA13 = fmnaOp(tmp, AA53, AA13);
    AA14 = fmnaOp(tmp, AA54, AA14);
    AA15 = mulOp(negOp(tmp), AA55);

    tmp = AA25;
    AA20 = fmnaOp(tmp, AA50, AA20);
    AA21 = fmnaOp(tmp, AA51, AA21);
    AA22 = fmnaOp(tmp, AA52, AA22);
    AA23 = fmnaOp(tmp, AA53, AA23);
    AA24 = fmnaOp(tmp, AA54, AA24);
    AA25 = mulOp(negOp(tmp), AA55);

    tmp = AA35;
    AA30 = fmnaOp(tmp, AA50, AA30);
    AA31 = fmnaOp(tmp, AA51, AA31);
    AA32 = fmnaOp(tmp, AA52, AA32);
    AA33 = fmnaOp(tmp, AA53, AA33);
    AA34 = fmnaOp(tmp, AA54, AA34);
    AA35 = mulOp(negOp(tmp), AA55);

    tmp = AA45;
    AA40 = fmnaOp(tmp, AA50, AA40);
    AA41 = fmnaOp(tmp, AA51, AA41);
    AA42 = fmnaOp(tmp, AA52, AA42);
    AA43 = fmnaOp(tmp, AA53, AA43);
    AA44 = fmnaOp(tmp, AA54, AA44);
    AA45 = mulOp(negOp(tmp), AA55);

    /* sort columns into the correct order */
    Ainv(0, icol0) = AA00;
    Ainv(1, icol0) = AA10;
    Ainv(2, icol0) = AA20;
    Ainv(3, icol0) = AA30;
    Ainv(4, icol0) = AA40;
    Ainv(5, icol0) = AA50;
    Ainv(0, icol1) = AA01;
    Ainv(1, icol1) = AA11;
    Ainv(2, icol1) = AA21;
    Ainv(3, icol1) = AA31;
    Ainv(4, icol1) = AA41;
    Ainv(5, icol1) = AA51;
    Ainv(0, icol2) = AA02;
    Ainv(1, icol2) = AA12;
    Ainv(2, icol2) = AA22;
    Ainv(3, icol2) = AA32;
    Ainv(4, icol2) = AA42;
    Ainv(5, icol2) = AA52;
    Ainv(0, icol3) = AA03;
    Ainv(1, icol3) = AA13;
    Ainv(2, icol3) = AA23;
    Ainv(3, icol3) = AA33;
    Ainv(4, icol3) = AA43;
    Ainv(5, icol3) = AA53;
    Ainv(0, icol4) = AA04;
    Ainv(1, icol4) = AA14;
    Ainv(2, icol4) = AA24;
    Ainv(3, icol4) = AA34;
    Ainv(4, icol4) = AA44;
    Ainv(5, icol4) = AA54;
    Ainv(0, icol5) = AA05;
    Ainv(1, icol5) = AA15;
    Ainv(2, icol5) = AA25;
    Ainv(3, icol5) = AA35;
    Ainv(4, icol5) = AA45;
    Ainv(5, icol5) = AA55;
  }
}

template <typename T, int arch>
__global__ void lu_matinv_7x7_matrix_per_thread(const T *A, T *Ainv, int batch) {
  const int blkNum = blockIdx.y * gridDim.x + blockIdx.x;
  const int thrdNum = blkNum * blockDim.x + threadIdx.x;
  const int N = 7;
  int perm0, perm1, perm2, perm3, perm4, perm5, perm6;
  int icol0, icol1, icol2, icol3, icol4, icol5, icol6;
  T AA00, AA01, AA02, AA03, AA04, AA05, AA06;
  T AA10, AA11, AA12, AA13, AA14, AA15, AA16;
  T AA20, AA21, AA22, AA23, AA24, AA25, AA26;
  T AA30, AA31, AA32, AA33, AA34, AA35, AA36;
  T AA40, AA41, AA42, AA43, AA44, AA45, AA46;
  T AA50, AA51, AA52, AA53, AA54, AA55, AA56;
  T AA60, AA61, AA62, AA63, AA64, AA65, AA66;
  T tmp;
#if USE_PIVOTING
  typename config<T, arch>::absValType t;
  typename config<T, arch>::absValType p;
  int i, pvt;
#endif

  if (thrdNum < batch) {

    AA00 = A[0 * batch + thrdNum];
    AA10 = A[1 * batch + thrdNum];
    AA20 = A[2 * batch + thrdNum];
    AA30 = A[3 * batch + thrdNum];
    AA40 = A[4 * batch + thrdNum];
    AA50 = A[5 * batch + thrdNum];
    AA60 = A[6 * batch + thrdNum];
    AA01 = A[7 * batch + thrdNum];
    AA11 = A[8 * batch + thrdNum];
    AA21 = A[9 * batch + thrdNum];
    AA31 = A[10 * batch + thrdNum];
    AA41 = A[11 * batch + thrdNum];
    AA51 = A[12 * batch + thrdNum];
    AA61 = A[13 * batch + thrdNum];
    AA02 = A[14 * batch + thrdNum];
    AA12 = A[15 * batch + thrdNum];
    AA22 = A[16 * batch + thrdNum];
    AA32 = A[17 * batch + thrdNum];
    AA42 = A[18 * batch + thrdNum];
    AA52 = A[19 * batch + thrdNum];
    AA62 = A[20 * batch + thrdNum];
    AA03 = A[21 * batch + thrdNum];
    AA13 = A[22 * batch + thrdNum];
    AA23 = A[23 * batch + thrdNum];
    AA33 = A[24 * batch + thrdNum];
    AA43 = A[25 * batch + thrdNum];
    AA53 = A[26 * batch + thrdNum];
    AA63 = A[27 * batch + thrdNum];
    AA04 = A[28 * batch + thrdNum];
    AA14 = A[29 * batch + thrdNum];
    AA24 = A[30 * batch + thrdNum];
    AA34 = A[31 * batch + thrdNum];
    AA44 = A[32 * batch + thrdNum];
    AA54 = A[33 * batch + thrdNum];
    AA64 = A[34 * batch + thrdNum];
    AA05 = A[35 * batch + thrdNum];
    AA15 = A[36 * batch + thrdNum];
    AA25 = A[37 * batch + thrdNum];
    AA35 = A[38 * batch + thrdNum];
    AA45 = A[39 * batch + thrdNum];
    AA55 = A[40 * batch + thrdNum];
    AA65 = A[41 * batch + thrdNum];
    AA06 = A[42 * batch + thrdNum];
    AA16 = A[43 * batch + thrdNum];
    AA26 = A[44 * batch + thrdNum];
    AA36 = A[45 * batch + thrdNum];
    AA46 = A[46 * batch + thrdNum];
    AA56 = A[47 * batch + thrdNum];
    AA66 = A[48 * batch + thrdNum];

    perm0 = 0;
    perm1 = 1;
    perm2 = 2;
    perm3 = 3;
    perm4 = 4;
    perm5 = 5;
    perm6 = 6;

    /****************** iteration 0 ***********/

#if USE_PIVOTING
    /* search pivot row */
    p = absOp(AA00);
    pvt = 0;
    t = absOp(AA10);
    if (t > p) {
      p = t;
      pvt = 1;
    }
    t = absOp(AA20);
    if (t > p) {
      p = t;
      pvt = 2;
    }
    t = absOp(AA30);
    if (t > p) {
      p = t;
      pvt = 3;
    }
    t = absOp(AA40);
    if (t > p) {
      p = t;
      pvt = 4;
    }
    t = absOp(AA50);
    if (t > p) {
      p = t;
      pvt = 5;
    }
    t = absOp(AA60);
    if (t > p) {
      p = t;
      pvt = 6;
    }

    /* swap pivot row with row 0 */
    if (pvt == 1) {
      tmp = AA00;
      AA00 = AA10;
      AA10 = tmp;
      tmp = AA01;
      AA01 = AA11;
      AA11 = tmp;
      tmp = AA02;
      AA02 = AA12;
      AA12 = tmp;
      tmp = AA03;
      AA03 = AA13;
      AA13 = tmp;
      tmp = AA04;
      AA04 = AA14;
      AA14 = tmp;
      tmp = AA05;
      AA05 = AA15;
      AA15 = tmp;
      tmp = AA06;
      AA06 = AA16;
      AA16 = tmp;
      /* update permutation vector based on row swap */
      i = perm0;
      perm0 = perm1;
      perm1 = i;
    }
    if (pvt == 2) {
      tmp = AA00;
      AA00 = AA20;
      AA20 = tmp;
      tmp = AA01;
      AA01 = AA21;
      AA21 = tmp;
      tmp = AA02;
      AA02 = AA22;
      AA22 = tmp;
      tmp = AA03;
      AA03 = AA23;
      AA23 = tmp;
      tmp = AA04;
      AA04 = AA24;
      AA24 = tmp;
      tmp = AA05;
      AA05 = AA25;
      AA25 = tmp;
      tmp = AA06;
      AA06 = AA26;
      AA26 = tmp;
      /* update permutation vector based on row swap */
      i = perm0;
      perm0 = perm2;
      perm2 = i;
    }
    if (pvt == 3) {
      tmp = AA00;
      AA00 = AA30;
      AA30 = tmp;
      tmp = AA01;
      AA01 = AA31;
      AA31 = tmp;
      tmp = AA02;
      AA02 = AA32;
      AA32 = tmp;
      tmp = AA03;
      AA03 = AA33;
      AA33 = tmp;
      tmp = AA04;
      AA04 = AA34;
      AA34 = tmp;
      tmp = AA05;
      AA05 = AA35;
      AA35 = tmp;
      tmp = AA06;
      AA06 = AA36;
      AA36 = tmp;
      /* update permutation vector based on row swap */
      i = perm0;
      perm0 = perm3;
      perm3 = i;
    }
    if (pvt == 4) {
      tmp = AA00;
      AA00 = AA40;
      AA40 = tmp;
      tmp = AA01;
      AA01 = AA41;
      AA41 = tmp;
      tmp = AA02;
      AA02 = AA42;
      AA42 = tmp;
      tmp = AA03;
      AA03 = AA43;
      AA43 = tmp;
      tmp = AA04;
      AA04 = AA44;
      AA44 = tmp;
      tmp = AA05;
      AA05 = AA45;
      AA45 = tmp;
      tmp = AA06;
      AA06 = AA46;
      AA46 = tmp;
      /* update permutation vector based on row swap */
      i = perm0;
      perm0 = perm4;
      perm4 = i;
    }
    if (pvt == 5) {
      tmp = AA00;
      AA00 = AA50;
      AA50 = tmp;
      tmp = AA01;
      AA01 = AA51;
      AA51 = tmp;
      tmp = AA02;
      AA02 = AA52;
      AA52 = tmp;
      tmp = AA03;
      AA03 = AA53;
      AA53 = tmp;
      tmp = AA04;
      AA04 = AA54;
      AA54 = tmp;
      tmp = AA05;
      AA05 = AA55;
      AA55 = tmp;
      tmp = AA06;
      AA06 = AA56;
      AA56 = tmp;
      /* update permutation vector based on row swap */
      i = perm0;
      perm0 = perm5;
      perm5 = i;
    }
    if (pvt == 6) {
      tmp = AA00;
      AA00 = AA60;
      AA60 = tmp;
      tmp = AA01;
      AA01 = AA61;
      AA61 = tmp;
      tmp = AA02;
      AA02 = AA62;
      AA62 = tmp;
      tmp = AA03;
      AA03 = AA63;
      AA63 = tmp;
      tmp = AA04;
      AA04 = AA64;
      AA64 = tmp;
      tmp = AA05;
      AA05 = AA65;
      AA65 = tmp;
      tmp = AA06;
      AA06 = AA66;
      AA66 = tmp;
      /* update permutation vector based on row swap */
      i = perm0;
      perm0 = perm6;
      perm6 = i;
    }
#endif // USE_PIVOTING

    /* scale current row */
    tmp = rcpOp(AA00);
    icol0 = perm0;
    AA00 = tmp;
    AA01 = mulOp(tmp, AA01);
    AA02 = mulOp(tmp, AA02);
    AA03 = mulOp(tmp, AA03);
    AA04 = mulOp(tmp, AA04);
    AA05 = mulOp(tmp, AA05);
    AA06 = mulOp(tmp, AA06);

    /* eliminate above and below current row */
    tmp = AA10;
    AA10 = mulOp(negOp(tmp), AA00);
    AA11 = fmnaOp(tmp, AA01, AA11);
    AA12 = fmnaOp(tmp, AA02, AA12);
    AA13 = fmnaOp(tmp, AA03, AA13);
    AA14 = fmnaOp(tmp, AA04, AA14);
    AA15 = fmnaOp(tmp, AA05, AA15);
    AA16 = fmnaOp(tmp, AA06, AA16);

    tmp = AA20;
    AA20 = mulOp(negOp(tmp), AA00);
    AA21 = fmnaOp(tmp, AA01, AA21);
    AA22 = fmnaOp(tmp, AA02, AA22);
    AA23 = fmnaOp(tmp, AA03, AA23);
    AA24 = fmnaOp(tmp, AA04, AA24);
    AA25 = fmnaOp(tmp, AA05, AA25);
    AA26 = fmnaOp(tmp, AA06, AA26);

    tmp = AA30;
    AA30 = mulOp(negOp(tmp), AA00);
    AA31 = fmnaOp(tmp, AA01, AA31);
    AA32 = fmnaOp(tmp, AA02, AA32);
    AA33 = fmnaOp(tmp, AA03, AA33);
    AA34 = fmnaOp(tmp, AA04, AA34);
    AA35 = fmnaOp(tmp, AA05, AA35);
    AA36 = fmnaOp(tmp, AA06, AA36);

    tmp = AA40;
    AA40 = mulOp(negOp(tmp), AA00);
    AA41 = fmnaOp(tmp, AA01, AA41);
    AA42 = fmnaOp(tmp, AA02, AA42);
    AA43 = fmnaOp(tmp, AA03, AA43);
    AA44 = fmnaOp(tmp, AA04, AA44);
    AA45 = fmnaOp(tmp, AA05, AA45);
    AA46 = fmnaOp(tmp, AA06, AA46);

    tmp = AA50;
    AA50 = mulOp(negOp(tmp), AA00);
    AA51 = fmnaOp(tmp, AA01, AA51);
    AA52 = fmnaOp(tmp, AA02, AA52);
    AA53 = fmnaOp(tmp, AA03, AA53);
    AA54 = fmnaOp(tmp, AA04, AA54);
    AA55 = fmnaOp(tmp, AA05, AA55);
    AA56 = fmnaOp(tmp, AA06, AA56);

    tmp = AA60;
    AA60 = mulOp(negOp(tmp), AA00);
    AA61 = fmnaOp(tmp, AA01, AA61);
    AA62 = fmnaOp(tmp, AA02, AA62);
    AA63 = fmnaOp(tmp, AA03, AA63);
    AA64 = fmnaOp(tmp, AA04, AA64);
    AA65 = fmnaOp(tmp, AA05, AA65);
    AA66 = fmnaOp(tmp, AA06, AA66);

    /****************** iteration 1 ***********/

#if USE_PIVOTING
    /* search pivot row */
    p = absOp(AA11);
    pvt = 1;
    t = absOp(AA21);
    if (t > p) {
      p = t;
      pvt = 2;
    }
    t = absOp(AA31);
    if (t > p) {
      p = t;
      pvt = 3;
    }
    t = absOp(AA41);
    if (t > p) {
      p = t;
      pvt = 4;
    }
    t = absOp(AA51);
    if (t > p) {
      p = t;
      pvt = 5;
    }
    t = absOp(AA61);
    if (t > p) {
      p = t;
      pvt = 6;
    }

    /* swap pivot row with row 1 */
    if (pvt == 2) {
      tmp = AA10;
      AA10 = AA20;
      AA20 = tmp;
      tmp = AA11;
      AA11 = AA21;
      AA21 = tmp;
      tmp = AA12;
      AA12 = AA22;
      AA22 = tmp;
      tmp = AA13;
      AA13 = AA23;
      AA23 = tmp;
      tmp = AA14;
      AA14 = AA24;
      AA24 = tmp;
      tmp = AA15;
      AA15 = AA25;
      AA25 = tmp;
      tmp = AA16;
      AA16 = AA26;
      AA26 = tmp;
      /* update permutation vector based on row swap */
      i = perm1;
      perm1 = perm2;
      perm2 = i;
    }
    if (pvt == 3) {
      tmp = AA10;
      AA10 = AA30;
      AA30 = tmp;
      tmp = AA11;
      AA11 = AA31;
      AA31 = tmp;
      tmp = AA12;
      AA12 = AA32;
      AA32 = tmp;
      tmp = AA13;
      AA13 = AA33;
      AA33 = tmp;
      tmp = AA14;
      AA14 = AA34;
      AA34 = tmp;
      tmp = AA15;
      AA15 = AA35;
      AA35 = tmp;
      tmp = AA16;
      AA16 = AA36;
      AA36 = tmp;
      /* update permutation vector based on row swap */
      i = perm1;
      perm1 = perm3;
      perm3 = i;
    }
    if (pvt == 4) {
      tmp = AA10;
      AA10 = AA40;
      AA40 = tmp;
      tmp = AA11;
      AA11 = AA41;
      AA41 = tmp;
      tmp = AA12;
      AA12 = AA42;
      AA42 = tmp;
      tmp = AA13;
      AA13 = AA43;
      AA43 = tmp;
      tmp = AA14;
      AA14 = AA44;
      AA44 = tmp;
      tmp = AA15;
      AA15 = AA45;
      AA45 = tmp;
      tmp = AA16;
      AA16 = AA46;
      AA46 = tmp;
      /* update permutation vector based on row swap */
      i = perm1;
      perm1 = perm4;
      perm4 = i;
    }
    if (pvt == 5) {
      tmp = AA10;
      AA10 = AA50;
      AA50 = tmp;
      tmp = AA11;
      AA11 = AA51;
      AA51 = tmp;
      tmp = AA12;
      AA12 = AA52;
      AA52 = tmp;
      tmp = AA13;
      AA13 = AA53;
      AA53 = tmp;
      tmp = AA14;
      AA14 = AA54;
      AA54 = tmp;
      tmp = AA15;
      AA15 = AA55;
      AA55 = tmp;
      tmp = AA16;
      AA16 = AA56;
      AA56 = tmp;
      /* update permutation vector based on row swap */
      i = perm1;
      perm1 = perm5;
      perm5 = i;
    }
    if (pvt == 6) {
      tmp = AA10;
      AA10 = AA60;
      AA60 = tmp;
      tmp = AA11;
      AA11 = AA61;
      AA61 = tmp;
      tmp = AA12;
      AA12 = AA62;
      AA62 = tmp;
      tmp = AA13;
      AA13 = AA63;
      AA63 = tmp;
      tmp = AA14;
      AA14 = AA64;
      AA64 = tmp;
      tmp = AA15;
      AA15 = AA65;
      AA65 = tmp;
      tmp = AA16;
      AA16 = AA66;
      AA66 = tmp;
      /* update permutation vector based on row swap */
      i = perm1;
      perm1 = perm6;
      perm6 = i;
    }
#endif // USE_PIVOTING

    /* scale current row */
    tmp = rcpOp(AA11);
    icol1 = perm1;
    AA10 = mulOp(tmp, AA10);
    AA11 = tmp;
    AA12 = mulOp(tmp, AA12);
    AA13 = mulOp(tmp, AA13);
    AA14 = mulOp(tmp, AA14);
    AA15 = mulOp(tmp, AA15);
    AA16 = mulOp(tmp, AA16);

    /* eliminate above and below current row */
    tmp = AA01;
    AA00 = fmnaOp(tmp, AA10, AA00);
    AA01 = mulOp(negOp(tmp), AA11);
    AA02 = fmnaOp(tmp, AA12, AA02);
    AA03 = fmnaOp(tmp, AA13, AA03);
    AA04 = fmnaOp(tmp, AA14, AA04);
    AA05 = fmnaOp(tmp, AA15, AA05);
    AA06 = fmnaOp(tmp, AA16, AA06);

    tmp = AA21;
    AA20 = fmnaOp(tmp, AA10, AA20);
    AA21 = mulOp(negOp(tmp), AA11);
    AA22 = fmnaOp(tmp, AA12, AA22);
    AA23 = fmnaOp(tmp, AA13, AA23);
    AA24 = fmnaOp(tmp, AA14, AA24);
    AA25 = fmnaOp(tmp, AA15, AA25);
    AA26 = fmnaOp(tmp, AA16, AA26);

    tmp = AA31;
    AA30 = fmnaOp(tmp, AA10, AA30);
    AA31 = mulOp(negOp(tmp), AA11);
    AA32 = fmnaOp(tmp, AA12, AA32);
    AA33 = fmnaOp(tmp, AA13, AA33);
    AA34 = fmnaOp(tmp, AA14, AA34);
    AA35 = fmnaOp(tmp, AA15, AA35);
    AA36 = fmnaOp(tmp, AA16, AA36);

    tmp = AA41;
    AA40 = fmnaOp(tmp, AA10, AA40);
    AA41 = mulOp(negOp(tmp), AA11);
    AA42 = fmnaOp(tmp, AA12, AA42);
    AA43 = fmnaOp(tmp, AA13, AA43);
    AA44 = fmnaOp(tmp, AA14, AA44);
    AA45 = fmnaOp(tmp, AA15, AA45);
    AA46 = fmnaOp(tmp, AA16, AA46);

    tmp = AA51;
    AA50 = fmnaOp(tmp, AA10, AA50);
    AA51 = mulOp(negOp(tmp), AA11);
    AA52 = fmnaOp(tmp, AA12, AA52);
    AA53 = fmnaOp(tmp, AA13, AA53);
    AA54 = fmnaOp(tmp, AA14, AA54);
    AA55 = fmnaOp(tmp, AA15, AA55);
    AA56 = fmnaOp(tmp, AA16, AA56);

    tmp = AA61;
    AA60 = fmnaOp(tmp, AA10, AA60);
    AA61 = mulOp(negOp(tmp), AA11);
    AA62 = fmnaOp(tmp, AA12, AA62);
    AA63 = fmnaOp(tmp, AA13, AA63);
    AA64 = fmnaOp(tmp, AA14, AA64);
    AA65 = fmnaOp(tmp, AA15, AA65);
    AA66 = fmnaOp(tmp, AA16, AA66);

    /****************** iteration 2 ****************/

#if USE_PIVOTING
    /* search pivot row */
    p = absOp(AA22);
    pvt = 2;
    t = absOp(AA32);
    if (t > p) {
      p = t;
      pvt = 3;
    }
    t = absOp(AA42);
    if (t > p) {
      p = t;
      pvt = 4;
    }
    t = absOp(AA52);
    if (t > p) {
      p = t;
      pvt = 5;
    }
    t = absOp(AA62);
    if (t > p) {
      p = t;
      pvt = 6;
    }

    /* swap pivot row with row 2 */
    if (pvt == 3) {
      tmp = AA20;
      AA20 = AA30;
      AA30 = tmp;
      tmp = AA21;
      AA21 = AA31;
      AA31 = tmp;
      tmp = AA22;
      AA22 = AA32;
      AA32 = tmp;
      tmp = AA23;
      AA23 = AA33;
      AA33 = tmp;
      tmp = AA24;
      AA24 = AA34;
      AA34 = tmp;
      tmp = AA25;
      AA25 = AA35;
      AA35 = tmp;
      tmp = AA26;
      AA26 = AA36;
      AA36 = tmp;
      /* update permutation vector based on row swap */
      i = perm2;
      perm2 = perm3;
      perm3 = i;
    }
    if (pvt == 4) {
      tmp = AA20;
      AA20 = AA40;
      AA40 = tmp;
      tmp = AA21;
      AA21 = AA41;
      AA41 = tmp;
      tmp = AA22;
      AA22 = AA42;
      AA42 = tmp;
      tmp = AA23;
      AA23 = AA43;
      AA43 = tmp;
      tmp = AA24;
      AA24 = AA44;
      AA44 = tmp;
      tmp = AA25;
      AA25 = AA45;
      AA45 = tmp;
      tmp = AA26;
      AA26 = AA46;
      AA46 = tmp;
      /* update permutation vector based on row swap */
      i = perm2;
      perm2 = perm4;
      perm4 = i;
    }
    if (pvt == 5) {
      tmp = AA20;
      AA20 = AA50;
      AA50 = tmp;
      tmp = AA21;
      AA21 = AA51;
      AA51 = tmp;
      tmp = AA22;
      AA22 = AA52;
      AA52 = tmp;
      tmp = AA23;
      AA23 = AA53;
      AA53 = tmp;
      tmp = AA24;
      AA24 = AA54;
      AA54 = tmp;
      tmp = AA25;
      AA25 = AA55;
      AA55 = tmp;
      tmp = AA26;
      AA26 = AA56;
      AA56 = tmp;
      /* update permutation vector based on row swap */
      i = perm2;
      perm2 = perm5;
      perm5 = i;
    }
    if (pvt == 6) {
      tmp = AA20;
      AA20 = AA60;
      AA60 = tmp;
      tmp = AA21;
      AA21 = AA61;
      AA61 = tmp;
      tmp = AA22;
      AA22 = AA62;
      AA62 = tmp;
      tmp = AA23;
      AA23 = AA63;
      AA63 = tmp;
      tmp = AA24;
      AA24 = AA64;
      AA64 = tmp;
      tmp = AA25;
      AA25 = AA65;
      AA65 = tmp;
      tmp = AA26;
      AA26 = AA66;
      AA66 = tmp;
      /* update permutation vector based on row swap */
      i = perm2;
      perm2 = perm6;
      perm6 = i;
    }
#endif // USE_PIVOTING

    /* scale current row */
    tmp = rcpOp(AA22);
    icol2 = perm2;
    AA20 = mulOp(tmp, AA20);
    AA21 = mulOp(tmp, AA21);
    AA22 = tmp;
    AA23 = mulOp(tmp, AA23);
    AA24 = mulOp(tmp, AA24);
    AA25 = mulOp(tmp, AA25);
    AA26 = mulOp(tmp, AA26);

    /* eliminate above and below current row */
    tmp = AA02;
    AA00 = fmnaOp(tmp, AA20, AA00);
    AA01 = fmnaOp(tmp, AA21, AA01);
    AA02 = mulOp(negOp(tmp), AA22);
    AA03 = fmnaOp(tmp, AA23, AA03);
    AA04 = fmnaOp(tmp, AA24, AA04);
    AA05 = fmnaOp(tmp, AA25, AA05);
    AA06 = fmnaOp(tmp, AA26, AA06);

    tmp = AA12;
    AA10 = fmnaOp(tmp, AA20, AA10);
    AA11 = fmnaOp(tmp, AA21, AA11);
    AA12 = mulOp(negOp(tmp), AA22);
    AA13 = fmnaOp(tmp, AA23, AA13);
    AA14 = fmnaOp(tmp, AA24, AA14);
    AA15 = fmnaOp(tmp, AA25, AA15);
    AA16 = fmnaOp(tmp, AA26, AA16);

    tmp = AA32;
    AA30 = fmnaOp(tmp, AA20, AA30);
    AA31 = fmnaOp(tmp, AA21, AA31);
    AA32 = mulOp(negOp(tmp), AA22);
    AA33 = fmnaOp(tmp, AA23, AA33);
    AA34 = fmnaOp(tmp, AA24, AA34);
    AA35 = fmnaOp(tmp, AA25, AA35);
    AA36 = fmnaOp(tmp, AA26, AA36);

    tmp = AA42;
    AA40 = fmnaOp(tmp, AA20, AA40);
    AA41 = fmnaOp(tmp, AA21, AA41);
    AA42 = mulOp(negOp(tmp), AA22);
    AA43 = fmnaOp(tmp, AA23, AA43);
    AA44 = fmnaOp(tmp, AA24, AA44);
    AA45 = fmnaOp(tmp, AA25, AA45);
    AA46 = fmnaOp(tmp, AA26, AA46);

    tmp = AA52;
    AA50 = fmnaOp(tmp, AA20, AA50);
    AA51 = fmnaOp(tmp, AA21, AA51);
    AA52 = mulOp(negOp(tmp), AA22);
    AA53 = fmnaOp(tmp, AA23, AA53);
    AA54 = fmnaOp(tmp, AA24, AA54);
    AA55 = fmnaOp(tmp, AA25, AA55);
    AA56 = fmnaOp(tmp, AA26, AA56);

    tmp = AA62;
    AA60 = fmnaOp(tmp, AA20, AA60);
    AA61 = fmnaOp(tmp, AA21, AA61);
    AA62 = mulOp(negOp(tmp), AA22);
    AA63 = fmnaOp(tmp, AA23, AA63);
    AA64 = fmnaOp(tmp, AA24, AA64);
    AA65 = fmnaOp(tmp, AA25, AA65);
    AA66 = fmnaOp(tmp, AA26, AA66);

    /****************** iteration 3 ****************/

#if USE_PIVOTING
    /* search pivot row */
    p = absOp(AA33);
    pvt = 3;
    t = absOp(AA43);
    if (t > p) {
      p = t;
      pvt = 4;
    }
    t = absOp(AA53);
    if (t > p) {
      p = t;
      pvt = 5;
    }
    t = absOp(AA63);
    if (t > p) {
      p = t;
      pvt = 6;
    }

    /* swap pivot row with row 3 */
    if (pvt == 4) {
      tmp = AA30;
      AA30 = AA40;
      AA40 = tmp;
      tmp = AA31;
      AA31 = AA41;
      AA41 = tmp;
      tmp = AA32;
      AA32 = AA42;
      AA42 = tmp;
      tmp = AA33;
      AA33 = AA43;
      AA43 = tmp;
      tmp = AA34;
      AA34 = AA44;
      AA44 = tmp;
      tmp = AA35;
      AA35 = AA45;
      AA45 = tmp;
      tmp = AA36;
      AA36 = AA46;
      AA46 = tmp;
      /* update permutation vector based on row swap */
      i = perm3;
      perm3 = perm4;
      perm4 = i;
    }
    if (pvt == 5) {
      tmp = AA30;
      AA30 = AA50;
      AA50 = tmp;
      tmp = AA31;
      AA31 = AA51;
      AA51 = tmp;
      tmp = AA32;
      AA32 = AA52;
      AA52 = tmp;
      tmp = AA33;
      AA33 = AA53;
      AA53 = tmp;
      tmp = AA34;
      AA34 = AA54;
      AA54 = tmp;
      tmp = AA35;
      AA35 = AA55;
      AA55 = tmp;
      tmp = AA36;
      AA36 = AA56;
      AA56 = tmp;
      /* update permutation vector based on row swap */
      i = perm3;
      perm3 = perm5;
      perm5 = i;
    }
    if (pvt == 6) {
      tmp = AA30;
      AA30 = AA60;
      AA60 = tmp;
      tmp = AA31;
      AA31 = AA61;
      AA61 = tmp;
      tmp = AA32;
      AA32 = AA62;
      AA62 = tmp;
      tmp = AA33;
      AA33 = AA63;
      AA63 = tmp;
      tmp = AA34;
      AA34 = AA64;
      AA64 = tmp;
      tmp = AA35;
      AA35 = AA65;
      AA65 = tmp;
      tmp = AA36;
      AA36 = AA66;
      AA66 = tmp;
      /* update permutation vector based on row swap */
      i = perm3;
      perm3 = perm6;
      perm6 = i;
    }
#endif // USE_PIVOTING

    /* scale current row */
    tmp = rcpOp(AA33);
    icol3 = perm3;
    AA30 = mulOp(tmp, AA30);
    AA31 = mulOp(tmp, AA31);
    AA32 = mulOp(tmp, AA32);
    AA33 = tmp;
    AA34 = mulOp(tmp, AA34);
    AA35 = mulOp(tmp, AA35);
    AA36 = mulOp(tmp, AA36);

    /* eliminate above and below current row */
    tmp = AA03;
    AA00 = fmnaOp(tmp, AA30, AA00);
    AA01 = fmnaOp(tmp, AA31, AA01);
    AA02 = fmnaOp(tmp, AA32, AA02);
    AA03 = mulOp(negOp(tmp), AA33);
    AA04 = fmnaOp(tmp, AA34, AA04);
    AA05 = fmnaOp(tmp, AA35, AA05);
    AA06 = fmnaOp(tmp, AA36, AA06);

    tmp = AA13;
    AA10 = fmnaOp(tmp, AA30, AA10);
    AA11 = fmnaOp(tmp, AA31, AA11);
    AA12 = fmnaOp(tmp, AA32, AA12);
    AA13 = mulOp(negOp(tmp), AA33);
    AA14 = fmnaOp(tmp, AA34, AA14);
    AA15 = fmnaOp(tmp, AA35, AA15);
    AA16 = fmnaOp(tmp, AA36, AA16);

    tmp = AA23;
    AA20 = fmnaOp(tmp, AA30, AA20);
    AA21 = fmnaOp(tmp, AA31, AA21);
    AA22 = fmnaOp(tmp, AA32, AA22);
    AA23 = mulOp(negOp(tmp), AA33);
    AA24 = fmnaOp(tmp, AA34, AA24);
    AA25 = fmnaOp(tmp, AA35, AA25);
    AA26 = fmnaOp(tmp, AA36, AA26);

    tmp = AA43;
    AA40 = fmnaOp(tmp, AA30, AA40);
    AA41 = fmnaOp(tmp, AA31, AA41);
    AA42 = fmnaOp(tmp, AA32, AA42);
    AA43 = mulOp(negOp(tmp), AA33);
    AA44 = fmnaOp(tmp, AA34, AA44);
    AA45 = fmnaOp(tmp, AA35, AA45);
    AA46 = fmnaOp(tmp, AA36, AA46);

    tmp = AA53;
    AA50 = fmnaOp(tmp, AA30, AA50);
    AA51 = fmnaOp(tmp, AA31, AA51);
    AA52 = fmnaOp(tmp, AA32, AA52);
    AA53 = mulOp(negOp(tmp), AA33);
    AA54 = fmnaOp(tmp, AA34, AA54);
    AA55 = fmnaOp(tmp, AA35, AA55);
    AA56 = fmnaOp(tmp, AA36, AA56);

    tmp = AA63;
    AA60 = fmnaOp(tmp, AA30, AA60);
    AA61 = fmnaOp(tmp, AA31, AA61);
    AA62 = fmnaOp(tmp, AA32, AA62);
    AA63 = mulOp(negOp(tmp), AA33);
    AA64 = fmnaOp(tmp, AA34, AA64);
    AA65 = fmnaOp(tmp, AA35, AA65);
    AA66 = fmnaOp(tmp, AA36, AA66);

    /****************** iteration 4 ****************/

#if USE_PIVOTING
    /* search pivot row */
    p = absOp(AA44);
    pvt = 4;
    t = absOp(AA54);
    if (t > p) {
      p = t;
      pvt = 5;
    }
    t = absOp(AA64);
    if (t > p) {
      p = t;
      pvt = 6;
    }

    /* swap pivot row with row 4 */
    if (pvt == 5) {
      tmp = AA40;
      AA40 = AA50;
      AA50 = tmp;
      tmp = AA41;
      AA41 = AA51;
      AA51 = tmp;
      tmp = AA42;
      AA42 = AA52;
      AA52 = tmp;
      tmp = AA43;
      AA43 = AA53;
      AA53 = tmp;
      tmp = AA44;
      AA44 = AA54;
      AA54 = tmp;
      tmp = AA45;
      AA45 = AA55;
      AA55 = tmp;
      tmp = AA46;
      AA46 = AA56;
      AA56 = tmp;
      /* update permutation vector based on row swap */
      i = perm4;
      perm4 = perm5;
      perm5 = i;
    }
    if (pvt == 6) {
      tmp = AA40;
      AA40 = AA60;
      AA60 = tmp;
      tmp = AA41;
      AA41 = AA61;
      AA61 = tmp;
      tmp = AA42;
      AA42 = AA62;
      AA62 = tmp;
      tmp = AA43;
      AA43 = AA63;
      AA63 = tmp;
      tmp = AA44;
      AA44 = AA64;
      AA64 = tmp;
      tmp = AA45;
      AA45 = AA65;
      AA65 = tmp;
      tmp = AA46;
      AA46 = AA66;
      AA66 = tmp;
      /* update permutation vector based on row swap */
      i = perm4;
      perm4 = perm6;
      perm6 = i;
    }
#endif // USE_PIVOTING

    /* scale current row */
    tmp = rcpOp(AA44);
    icol4 = perm4;
    AA40 = mulOp(tmp, AA40);
    AA41 = mulOp(tmp, AA41);
    AA42 = mulOp(tmp, AA42);
    AA43 = mulOp(tmp, AA43);
    AA44 = tmp;
    AA45 = mulOp(tmp, AA45);
    AA46 = mulOp(tmp, AA46);

    /* eliminate above and below current row */
    tmp = AA04;
    AA00 = fmnaOp(tmp, AA40, AA00);
    AA01 = fmnaOp(tmp, AA41, AA01);
    AA02 = fmnaOp(tmp, AA42, AA02);
    AA03 = fmnaOp(tmp, AA43, AA03);
    AA04 = mulOp(negOp(tmp), AA44);
    AA05 = fmnaOp(tmp, AA45, AA05);
    AA06 = fmnaOp(tmp, AA46, AA06);

    tmp = AA14;
    AA10 = fmnaOp(tmp, AA40, AA10);
    AA11 = fmnaOp(tmp, AA41, AA11);
    AA12 = fmnaOp(tmp, AA42, AA12);
    AA13 = fmnaOp(tmp, AA43, AA13);
    AA14 = mulOp(negOp(tmp), AA44);
    AA15 = fmnaOp(tmp, AA45, AA15);
    AA16 = fmnaOp(tmp, AA46, AA16);

    tmp = AA24;
    AA20 = fmnaOp(tmp, AA40, AA20);
    AA21 = fmnaOp(tmp, AA41, AA21);
    AA22 = fmnaOp(tmp, AA42, AA22);
    AA23 = fmnaOp(tmp, AA43, AA23);
    AA24 = mulOp(negOp(tmp), AA44);
    AA25 = fmnaOp(tmp, AA45, AA25);
    AA26 = fmnaOp(tmp, AA46, AA26);

    tmp = AA34;
    AA30 = fmnaOp(tmp, AA40, AA30);
    AA31 = fmnaOp(tmp, AA41, AA31);
    AA32 = fmnaOp(tmp, AA42, AA32);
    AA33 = fmnaOp(tmp, AA43, AA33);
    AA34 = mulOp(negOp(tmp), AA44);
    AA35 = fmnaOp(tmp, AA45, AA35);
    AA36 = fmnaOp(tmp, AA46, AA36);

    tmp = AA54;
    AA50 = fmnaOp(tmp, AA40, AA50);
    AA51 = fmnaOp(tmp, AA41, AA51);
    AA52 = fmnaOp(tmp, AA42, AA52);
    AA53 = fmnaOp(tmp, AA43, AA53);
    AA54 = mulOp(negOp(tmp), AA44);
    AA55 = fmnaOp(tmp, AA45, AA55);
    AA56 = fmnaOp(tmp, AA46, AA56);

    tmp = AA64;
    AA60 = fmnaOp(tmp, AA40, AA60);
    AA61 = fmnaOp(tmp, AA41, AA61);
    AA62 = fmnaOp(tmp, AA42, AA62);
    AA63 = fmnaOp(tmp, AA43, AA63);
    AA64 = mulOp(negOp(tmp), AA44);
    AA65 = fmnaOp(tmp, AA45, AA65);
    AA66 = fmnaOp(tmp, AA46, AA66);

    /****************** iteration 5 ****************/

#if USE_PIVOTING
    /* search pivot row */
    p = absOp(AA55);
    pvt = 5;
    t = absOp(AA65);
    if (t > p) {
      p = t;
      pvt = 6;
    }

    /* swap pivot row with row 5 */
    if (pvt == 6) {
      tmp = AA50;
      AA50 = AA60;
      AA60 = tmp;
      tmp = AA51;
      AA51 = AA61;
      AA61 = tmp;
      tmp = AA52;
      AA52 = AA62;
      AA62 = tmp;
      tmp = AA53;
      AA53 = AA63;
      AA63 = tmp;
      tmp = AA54;
      AA54 = AA64;
      AA64 = tmp;
      tmp = AA55;
      AA55 = AA65;
      AA65 = tmp;
      tmp = AA56;
      AA56 = AA66;
      AA66 = tmp;
      /* update permutation vector based on row swap */
      i = perm5;
      perm5 = perm6;
      perm6 = i;
    }
#endif // USE_PIVOTING

    /* scale current row */
    tmp = rcpOp(AA55);
    icol5 = perm5;
    AA50 = mulOp(tmp, AA50);
    AA51 = mulOp(tmp, AA51);
    AA52 = mulOp(tmp, AA52);
    AA53 = mulOp(tmp, AA53);
    AA54 = mulOp(tmp, AA54);
    AA55 = tmp;
    AA56 = mulOp(tmp, AA56);

    /* eliminate above and below current row */
    tmp = AA05;
    AA00 = fmnaOp(tmp, AA50, AA00);
    AA01 = fmnaOp(tmp, AA51, AA01);
    AA02 = fmnaOp(tmp, AA52, AA02);
    AA03 = fmnaOp(tmp, AA53, AA03);
    AA04 = fmnaOp(tmp, AA54, AA04);
    AA05 = mulOp(negOp(tmp), AA55);
    AA06 = fmnaOp(tmp, AA56, AA06);

    tmp = AA15;
    AA10 = fmnaOp(tmp, AA50, AA10);
    AA11 = fmnaOp(tmp, AA51, AA11);
    AA12 = fmnaOp(tmp, AA52, AA12);
    AA13 = fmnaOp(tmp, AA53, AA13);
    AA14 = fmnaOp(tmp, AA54, AA14);
    AA15 = mulOp(negOp(tmp), AA55);
    AA16 = fmnaOp(tmp, AA56, AA16);

    tmp = AA25;
    AA20 = fmnaOp(tmp, AA50, AA20);
    AA21 = fmnaOp(tmp, AA51, AA21);
    AA22 = fmnaOp(tmp, AA52, AA22);
    AA23 = fmnaOp(tmp, AA53, AA23);
    AA24 = fmnaOp(tmp, AA54, AA24);
    AA25 = mulOp(negOp(tmp), AA55);
    AA26 = fmnaOp(tmp, AA56, AA26);

    tmp = AA35;
    AA30 = fmnaOp(tmp, AA50, AA30);
    AA31 = fmnaOp(tmp, AA51, AA31);
    AA32 = fmnaOp(tmp, AA52, AA32);
    AA33 = fmnaOp(tmp, AA53, AA33);
    AA34 = fmnaOp(tmp, AA54, AA34);
    AA35 = mulOp(negOp(tmp), AA55);
    AA36 = fmnaOp(tmp, AA56, AA36);

    tmp = AA45;
    AA40 = fmnaOp(tmp, AA50, AA40);
    AA41 = fmnaOp(tmp, AA51, AA41);
    AA42 = fmnaOp(tmp, AA52, AA42);
    AA43 = fmnaOp(tmp, AA53, AA43);
    AA44 = fmnaOp(tmp, AA54, AA44);
    AA45 = mulOp(negOp(tmp), AA55);
    AA46 = fmnaOp(tmp, AA56, AA46);

    tmp = AA65;
    AA60 = fmnaOp(tmp, AA50, AA60);
    AA61 = fmnaOp(tmp, AA51, AA61);
    AA62 = fmnaOp(tmp, AA52, AA62);
    AA63 = fmnaOp(tmp, AA53, AA63);
    AA64 = fmnaOp(tmp, AA54, AA64);
    AA65 = mulOp(negOp(tmp), AA55);
    AA66 = fmnaOp(tmp, AA56, AA66);

    /****************** iteration 6 ****************/

    /* scale current row */
    tmp = rcpOp(AA66);
    icol6 = perm6;
    AA60 = mulOp(tmp, AA60);
    AA61 = mulOp(tmp, AA61);
    AA62 = mulOp(tmp, AA62);
    AA63 = mulOp(tmp, AA63);
    AA64 = mulOp(tmp, AA64);
    AA65 = mulOp(tmp, AA65);
    AA66 = tmp;

    /* eliminate above and below current row */
    tmp = AA06;
    AA00 = fmnaOp(tmp, AA60, AA00);
    AA01 = fmnaOp(tmp, AA61, AA01);
    AA02 = fmnaOp(tmp, AA62, AA02);
    AA03 = fmnaOp(tmp, AA63, AA03);
    AA04 = fmnaOp(tmp, AA64, AA04);
    AA05 = fmnaOp(tmp, AA65, AA05);
    AA06 = mulOp(negOp(tmp), AA66);

    tmp = AA16;
    AA10 = fmnaOp(tmp, AA60, AA10);
    AA11 = fmnaOp(tmp, AA61, AA11);
    AA12 = fmnaOp(tmp, AA62, AA12);
    AA13 = fmnaOp(tmp, AA63, AA13);
    AA14 = fmnaOp(tmp, AA64, AA14);
    AA15 = fmnaOp(tmp, AA65, AA15);
    AA16 = mulOp(negOp(tmp), AA66);

    tmp = AA26;
    AA20 = fmnaOp(tmp, AA60, AA20);
    AA21 = fmnaOp(tmp, AA61, AA21);
    AA22 = fmnaOp(tmp, AA62, AA22);
    AA23 = fmnaOp(tmp, AA63, AA23);
    AA24 = fmnaOp(tmp, AA64, AA24);
    AA25 = fmnaOp(tmp, AA65, AA25);
    AA26 = mulOp(negOp(tmp), AA66);

    tmp = AA36;
    AA30 = fmnaOp(tmp, AA60, AA30);
    AA31 = fmnaOp(tmp, AA61, AA31);
    AA32 = fmnaOp(tmp, AA62, AA32);
    AA33 = fmnaOp(tmp, AA63, AA33);
    AA34 = fmnaOp(tmp, AA64, AA34);
    AA35 = fmnaOp(tmp, AA65, AA35);
    AA36 = mulOp(negOp(tmp), AA66);

    tmp = AA46;
    AA40 = fmnaOp(tmp, AA60, AA40);
    AA41 = fmnaOp(tmp, AA61, AA41);
    AA42 = fmnaOp(tmp, AA62, AA42);
    AA43 = fmnaOp(tmp, AA63, AA43);
    AA44 = fmnaOp(tmp, AA64, AA44);
    AA45 = fmnaOp(tmp, AA65, AA45);
    AA46 = mulOp(negOp(tmp), AA66);

    tmp = AA56;
    AA50 = fmnaOp(tmp, AA60, AA50);
    AA51 = fmnaOp(tmp, AA61, AA51);
    AA52 = fmnaOp(tmp, AA62, AA52);
    AA53 = fmnaOp(tmp, AA63, AA53);
    AA54 = fmnaOp(tmp, AA64, AA54);
    AA55 = fmnaOp(tmp, AA65, AA55);
    AA56 = mulOp(negOp(tmp), AA66);

    /* sort columns into the correct order */
    Ainv(0, icol0) = AA00;
    Ainv(1, icol0) = AA10;
    Ainv(2, icol0) = AA20;
    Ainv(3, icol0) = AA30;
    Ainv(4, icol0) = AA40;
    Ainv(5, icol0) = AA50;
    Ainv(6, icol0) = AA60;
    Ainv(0, icol1) = AA01;
    Ainv(1, icol1) = AA11;
    Ainv(2, icol1) = AA21;
    Ainv(3, icol1) = AA31;
    Ainv(4, icol1) = AA41;
    Ainv(5, icol1) = AA51;
    Ainv(6, icol1) = AA61;
    Ainv(0, icol2) = AA02;
    Ainv(1, icol2) = AA12;
    Ainv(2, icol2) = AA22;
    Ainv(3, icol2) = AA32;
    Ainv(4, icol2) = AA42;
    Ainv(5, icol2) = AA52;
    Ainv(6, icol2) = AA62;
    Ainv(0, icol3) = AA03;
    Ainv(1, icol3) = AA13;
    Ainv(2, icol3) = AA23;
    Ainv(3, icol3) = AA33;
    Ainv(4, icol3) = AA43;
    Ainv(5, icol3) = AA53;
    Ainv(6, icol3) = AA63;
    Ainv(0, icol4) = AA04;
    Ainv(1, icol4) = AA14;
    Ainv(2, icol4) = AA24;
    Ainv(3, icol4) = AA34;
    Ainv(4, icol4) = AA44;
    Ainv(5, icol4) = AA54;
    Ainv(6, icol4) = AA64;
    Ainv(0, icol5) = AA05;
    Ainv(1, icol5) = AA15;
    Ainv(2, icol5) = AA25;
    Ainv(3, icol5) = AA35;
    Ainv(4, icol5) = AA45;
    Ainv(5, icol5) = AA55;
    Ainv(6, icol5) = AA65;
    Ainv(0, icol6) = AA06;
    Ainv(1, icol6) = AA16;
    Ainv(2, icol6) = AA26;
    Ainv(3, icol6) = AA36;
    Ainv(4, icol6) = AA46;
    Ainv(5, icol6) = AA56;
    Ainv(6, icol6) = AA66;
  }
}

template <typename T, int arch>
__global__ void lu_matinv_8x8_matrix_per_thread(const T *A, T *Ainv, int batch) {
  const int blkNum = blockIdx.y * gridDim.x + blockIdx.x;
  const int thrdNum = blkNum * blockDim.x + threadIdx.x;
  const int N = 8;
  int perm0, perm1, perm2, perm3, perm4, perm5, perm6, perm7;
  int icol0, icol1, icol2, icol3, icol4, icol5, icol6, icol7;
  T AA00, AA01, AA02, AA03, AA04, AA05, AA06, AA07;
  T AA10, AA11, AA12, AA13, AA14, AA15, AA16, AA17;
  T AA20, AA21, AA22, AA23, AA24, AA25, AA26, AA27;
  T AA30, AA31, AA32, AA33, AA34, AA35, AA36, AA37;
  T AA40, AA41, AA42, AA43, AA44, AA45, AA46, AA47;
  T AA50, AA51, AA52, AA53, AA54, AA55, AA56, AA57;
  T AA60, AA61, AA62, AA63, AA64, AA65, AA66, AA67;
  T AA70, AA71, AA72, AA73, AA74, AA75, AA76, AA77;
  T tmp;
#if USE_PIVOTING
  typename config<T, arch>::absValType t;
  typename config<T, arch>::absValType p;
  int i, pvt;
#endif

  if (thrdNum < batch) {

    AA00 = A[0 * batch + thrdNum];
    AA10 = A[1 * batch + thrdNum];
    AA20 = A[2 * batch + thrdNum];
    AA30 = A[3 * batch + thrdNum];
    AA40 = A[4 * batch + thrdNum];
    AA50 = A[5 * batch + thrdNum];
    AA60 = A[6 * batch + thrdNum];
    AA70 = A[7 * batch + thrdNum];
    AA01 = A[8 * batch + thrdNum];
    AA11 = A[9 * batch + thrdNum];
    AA21 = A[10 * batch + thrdNum];
    AA31 = A[11 * batch + thrdNum];
    AA41 = A[12 * batch + thrdNum];
    AA51 = A[13 * batch + thrdNum];
    AA61 = A[14 * batch + thrdNum];
    AA71 = A[15 * batch + thrdNum];
    AA02 = A[16 * batch + thrdNum];
    AA12 = A[17 * batch + thrdNum];
    AA22 = A[18 * batch + thrdNum];
    AA32 = A[19 * batch + thrdNum];
    AA42 = A[20 * batch + thrdNum];
    AA52 = A[21 * batch + thrdNum];
    AA62 = A[22 * batch + thrdNum];
    AA72 = A[23 * batch + thrdNum];
    AA03 = A[24 * batch + thrdNum];
    AA13 = A[25 * batch + thrdNum];
    AA23 = A[26 * batch + thrdNum];
    AA33 = A[27 * batch + thrdNum];
    AA43 = A[28 * batch + thrdNum];
    AA53 = A[29 * batch + thrdNum];
    AA63 = A[30 * batch + thrdNum];
    AA73 = A[31 * batch + thrdNum];
    AA04 = A[32 * batch + thrdNum];
    AA14 = A[33 * batch + thrdNum];
    AA24 = A[34 * batch + thrdNum];
    AA34 = A[35 * batch + thrdNum];
    AA44 = A[36 * batch + thrdNum];
    AA54 = A[37 * batch + thrdNum];
    AA64 = A[38 * batch + thrdNum];
    AA74 = A[39 * batch + thrdNum];
    AA05 = A[40 * batch + thrdNum];
    AA15 = A[41 * batch + thrdNum];
    AA25 = A[42 * batch + thrdNum];
    AA35 = A[43 * batch + thrdNum];
    AA45 = A[44 * batch + thrdNum];
    AA55 = A[45 * batch + thrdNum];
    AA65 = A[46 * batch + thrdNum];
    AA75 = A[47 * batch + thrdNum];
    AA06 = A[48 * batch + thrdNum];
    AA16 = A[49 * batch + thrdNum];
    AA26 = A[50 * batch + thrdNum];
    AA36 = A[51 * batch + thrdNum];
    AA46 = A[52 * batch + thrdNum];
    AA56 = A[53 * batch + thrdNum];
    AA66 = A[54 * batch + thrdNum];
    AA76 = A[55 * batch + thrdNum];
    AA07 = A[56 * batch + thrdNum];
    AA17 = A[57 * batch + thrdNum];
    AA27 = A[58 * batch + thrdNum];
    AA37 = A[59 * batch + thrdNum];
    AA47 = A[60 * batch + thrdNum];
    AA57 = A[61 * batch + thrdNum];
    AA67 = A[62 * batch + thrdNum];
    AA77 = A[63 * batch + thrdNum];

    perm0 = 0;
    perm1 = 1;
    perm2 = 2;
    perm3 = 3;
    perm4 = 4;
    perm5 = 5;
    perm6 = 6;
    perm7 = 7;

    /****************** iteration 0 ***********/

#if USE_PIVOTING
    /* search pivot row */
    p = absOp(AA00);
    pvt = 0;
    t = absOp(AA10);
    if (t > p) {
      p = t;
      pvt = 1;
    }
    t = absOp(AA20);
    if (t > p) {
      p = t;
      pvt = 2;
    }
    t = absOp(AA30);
    if (t > p) {
      p = t;
      pvt = 3;
    }
    t = absOp(AA40);
    if (t > p) {
      p = t;
      pvt = 4;
    }
    t = absOp(AA50);
    if (t > p) {
      p = t;
      pvt = 5;
    }
    t = absOp(AA60);
    if (t > p) {
      p = t;
      pvt = 6;
    }
    t = absOp(AA70);
    if (t > p) {
      p = t;
      pvt = 7;
    }

    /* swap pivot row with row 0 */
    if (pvt == 1) {
      tmp = AA00;
      AA00 = AA10;
      AA10 = tmp;
      tmp = AA01;
      AA01 = AA11;
      AA11 = tmp;
      tmp = AA02;
      AA02 = AA12;
      AA12 = tmp;
      tmp = AA03;
      AA03 = AA13;
      AA13 = tmp;
      tmp = AA04;
      AA04 = AA14;
      AA14 = tmp;
      tmp = AA05;
      AA05 = AA15;
      AA15 = tmp;
      tmp = AA06;
      AA06 = AA16;
      AA16 = tmp;
      tmp = AA07;
      AA07 = AA17;
      AA17 = tmp;
      /* update permutation vector based on row swap */
      i = perm0;
      perm0 = perm1;
      perm1 = i;
    }
    if (pvt == 2) {
      tmp = AA00;
      AA00 = AA20;
      AA20 = tmp;
      tmp = AA01;
      AA01 = AA21;
      AA21 = tmp;
      tmp = AA02;
      AA02 = AA22;
      AA22 = tmp;
      tmp = AA03;
      AA03 = AA23;
      AA23 = tmp;
      tmp = AA04;
      AA04 = AA24;
      AA24 = tmp;
      tmp = AA05;
      AA05 = AA25;
      AA25 = tmp;
      tmp = AA06;
      AA06 = AA26;
      AA26 = tmp;
      tmp = AA07;
      AA07 = AA27;
      AA27 = tmp;
      /* update permutation vector based on row swap */
      i = perm0;
      perm0 = perm2;
      perm2 = i;
    }
    if (pvt == 3) {
      tmp = AA00;
      AA00 = AA30;
      AA30 = tmp;
      tmp = AA01;
      AA01 = AA31;
      AA31 = tmp;
      tmp = AA02;
      AA02 = AA32;
      AA32 = tmp;
      tmp = AA03;
      AA03 = AA33;
      AA33 = tmp;
      tmp = AA04;
      AA04 = AA34;
      AA34 = tmp;
      tmp = AA05;
      AA05 = AA35;
      AA35 = tmp;
      tmp = AA06;
      AA06 = AA36;
      AA36 = tmp;
      tmp = AA07;
      AA07 = AA37;
      AA37 = tmp;
      /* update permutation vector based on row swap */
      i = perm0;
      perm0 = perm3;
      perm3 = i;
    }
    if (pvt == 4) {
      tmp = AA00;
      AA00 = AA40;
      AA40 = tmp;
      tmp = AA01;
      AA01 = AA41;
      AA41 = tmp;
      tmp = AA02;
      AA02 = AA42;
      AA42 = tmp;
      tmp = AA03;
      AA03 = AA43;
      AA43 = tmp;
      tmp = AA04;
      AA04 = AA44;
      AA44 = tmp;
      tmp = AA05;
      AA05 = AA45;
      AA45 = tmp;
      tmp = AA06;
      AA06 = AA46;
      AA46 = tmp;
      tmp = AA07;
      AA07 = AA47;
      AA47 = tmp;
      /* update permutation vector based on row swap */
      i = perm0;
      perm0 = perm4;
      perm4 = i;
    }
    if (pvt == 5) {
      tmp = AA00;
      AA00 = AA50;
      AA50 = tmp;
      tmp = AA01;
      AA01 = AA51;
      AA51 = tmp;
      tmp = AA02;
      AA02 = AA52;
      AA52 = tmp;
      tmp = AA03;
      AA03 = AA53;
      AA53 = tmp;
      tmp = AA04;
      AA04 = AA54;
      AA54 = tmp;
      tmp = AA05;
      AA05 = AA55;
      AA55 = tmp;
      tmp = AA06;
      AA06 = AA56;
      AA56 = tmp;
      tmp = AA07;
      AA07 = AA57;
      AA57 = tmp;
      /* update permutation vector based on row swap */
      i = perm0;
      perm0 = perm5;
      perm5 = i;
    }
    if (pvt == 6) {
      tmp = AA00;
      AA00 = AA60;
      AA60 = tmp;
      tmp = AA01;
      AA01 = AA61;
      AA61 = tmp;
      tmp = AA02;
      AA02 = AA62;
      AA62 = tmp;
      tmp = AA03;
      AA03 = AA63;
      AA63 = tmp;
      tmp = AA04;
      AA04 = AA64;
      AA64 = tmp;
      tmp = AA05;
      AA05 = AA65;
      AA65 = tmp;
      tmp = AA06;
      AA06 = AA66;
      AA66 = tmp;
      tmp = AA07;
      AA07 = AA67;
      AA67 = tmp;
      /* update permutation vector based on row swap */
      i = perm0;
      perm0 = perm6;
      perm6 = i;
    }
    if (pvt == 7) {
      tmp = AA00;
      AA00 = AA70;
      AA70 = tmp;
      tmp = AA01;
      AA01 = AA71;
      AA71 = tmp;
      tmp = AA02;
      AA02 = AA72;
      AA72 = tmp;
      tmp = AA03;
      AA03 = AA73;
      AA73 = tmp;
      tmp = AA04;
      AA04 = AA74;
      AA74 = tmp;
      tmp = AA05;
      AA05 = AA75;
      AA75 = tmp;
      tmp = AA06;
      AA06 = AA76;
      AA76 = tmp;
      tmp = AA07;
      AA07 = AA77;
      AA77 = tmp;
      /* update permutation vector based on row swap */
      i = perm0;
      perm0 = perm7;
      perm7 = i;
    }
#endif // USE_PIVOTING

    /* scale current row */
    tmp = rcpOp(AA00);
    icol0 = perm0;
    AA00 = tmp;
    AA01 = mulOp(tmp, AA01);
    AA02 = mulOp(tmp, AA02);
    AA03 = mulOp(tmp, AA03);
    AA04 = mulOp(tmp, AA04);
    AA05 = mulOp(tmp, AA05);
    AA06 = mulOp(tmp, AA06);
    AA07 = mulOp(tmp, AA07);

    /* eliminate above and below current row */
    tmp = AA10;
    AA10 = mulOp(negOp(tmp), AA00);
    AA11 = fmnaOp(tmp, AA01, AA11);
    AA12 = fmnaOp(tmp, AA02, AA12);
    AA13 = fmnaOp(tmp, AA03, AA13);
    AA14 = fmnaOp(tmp, AA04, AA14);
    AA15 = fmnaOp(tmp, AA05, AA15);
    AA16 = fmnaOp(tmp, AA06, AA16);
    AA17 = fmnaOp(tmp, AA07, AA17);

    tmp = AA20;
    AA20 = mulOp(negOp(tmp), AA00);
    AA21 = fmnaOp(tmp, AA01, AA21);
    AA22 = fmnaOp(tmp, AA02, AA22);
    AA23 = fmnaOp(tmp, AA03, AA23);
    AA24 = fmnaOp(tmp, AA04, AA24);
    AA25 = fmnaOp(tmp, AA05, AA25);
    AA26 = fmnaOp(tmp, AA06, AA26);
    AA27 = fmnaOp(tmp, AA07, AA27);

    tmp = AA30;
    AA30 = mulOp(negOp(tmp), AA00);
    AA31 = fmnaOp(tmp, AA01, AA31);
    AA32 = fmnaOp(tmp, AA02, AA32);
    AA33 = fmnaOp(tmp, AA03, AA33);
    AA34 = fmnaOp(tmp, AA04, AA34);
    AA35 = fmnaOp(tmp, AA05, AA35);
    AA36 = fmnaOp(tmp, AA06, AA36);
    AA37 = fmnaOp(tmp, AA07, AA37);

    tmp = AA40;
    AA40 = mulOp(negOp(tmp), AA00);
    AA41 = fmnaOp(tmp, AA01, AA41);
    AA42 = fmnaOp(tmp, AA02, AA42);
    AA43 = fmnaOp(tmp, AA03, AA43);
    AA44 = fmnaOp(tmp, AA04, AA44);
    AA45 = fmnaOp(tmp, AA05, AA45);
    AA46 = fmnaOp(tmp, AA06, AA46);
    AA47 = fmnaOp(tmp, AA07, AA47);

    tmp = AA50;
    AA50 = mulOp(negOp(tmp), AA00);
    AA51 = fmnaOp(tmp, AA01, AA51);
    AA52 = fmnaOp(tmp, AA02, AA52);
    AA53 = fmnaOp(tmp, AA03, AA53);
    AA54 = fmnaOp(tmp, AA04, AA54);
    AA55 = fmnaOp(tmp, AA05, AA55);
    AA56 = fmnaOp(tmp, AA06, AA56);
    AA57 = fmnaOp(tmp, AA07, AA57);

    tmp = AA60;
    AA60 = mulOp(negOp(tmp), AA00);
    AA61 = fmnaOp(tmp, AA01, AA61);
    AA62 = fmnaOp(tmp, AA02, AA62);
    AA63 = fmnaOp(tmp, AA03, AA63);
    AA64 = fmnaOp(tmp, AA04, AA64);
    AA65 = fmnaOp(tmp, AA05, AA65);
    AA66 = fmnaOp(tmp, AA06, AA66);
    AA67 = fmnaOp(tmp, AA07, AA67);

    tmp = AA70;
    AA70 = mulOp(negOp(tmp), AA00);
    AA71 = fmnaOp(tmp, AA01, AA71);
    AA72 = fmnaOp(tmp, AA02, AA72);
    AA73 = fmnaOp(tmp, AA03, AA73);
    AA74 = fmnaOp(tmp, AA04, AA74);
    AA75 = fmnaOp(tmp, AA05, AA75);
    AA76 = fmnaOp(tmp, AA06, AA76);
    AA77 = fmnaOp(tmp, AA07, AA77);

    /****************** iteration 1 ***********/

#if USE_PIVOTING
    /* search pivot row */
    p = absOp(AA11);
    pvt = 1;
    t = absOp(AA21);
    if (t > p) {
      p = t;
      pvt = 2;
    }
    t = absOp(AA31);
    if (t > p) {
      p = t;
      pvt = 3;
    }
    t = absOp(AA41);
    if (t > p) {
      p = t;
      pvt = 4;
    }
    t = absOp(AA51);
    if (t > p) {
      p = t;
      pvt = 5;
    }
    t = absOp(AA61);
    if (t > p) {
      p = t;
      pvt = 6;
    }
    t = absOp(AA71);
    if (t > p) {
      p = t;
      pvt = 7;
    }

    /* swap pivot row with row 1 */
    if (pvt == 2) {
      tmp = AA10;
      AA10 = AA20;
      AA20 = tmp;
      tmp = AA11;
      AA11 = AA21;
      AA21 = tmp;
      tmp = AA12;
      AA12 = AA22;
      AA22 = tmp;
      tmp = AA13;
      AA13 = AA23;
      AA23 = tmp;
      tmp = AA14;
      AA14 = AA24;
      AA24 = tmp;
      tmp = AA15;
      AA15 = AA25;
      AA25 = tmp;
      tmp = AA16;
      AA16 = AA26;
      AA26 = tmp;
      tmp = AA17;
      AA17 = AA27;
      AA27 = tmp;
      /* update permutation vector based on row swap */
      i = perm1;
      perm1 = perm2;
      perm2 = i;
    }
    if (pvt == 3) {
      tmp = AA10;
      AA10 = AA30;
      AA30 = tmp;
      tmp = AA11;
      AA11 = AA31;
      AA31 = tmp;
      tmp = AA12;
      AA12 = AA32;
      AA32 = tmp;
      tmp = AA13;
      AA13 = AA33;
      AA33 = tmp;
      tmp = AA14;
      AA14 = AA34;
      AA34 = tmp;
      tmp = AA15;
      AA15 = AA35;
      AA35 = tmp;
      tmp = AA16;
      AA16 = AA36;
      AA36 = tmp;
      tmp = AA17;
      AA17 = AA37;
      AA37 = tmp;
      /* update permutation vector based on row swap */
      i = perm1;
      perm1 = perm3;
      perm3 = i;
    }
    if (pvt == 4) {
      tmp = AA10;
      AA10 = AA40;
      AA40 = tmp;
      tmp = AA11;
      AA11 = AA41;
      AA41 = tmp;
      tmp = AA12;
      AA12 = AA42;
      AA42 = tmp;
      tmp = AA13;
      AA13 = AA43;
      AA43 = tmp;
      tmp = AA14;
      AA14 = AA44;
      AA44 = tmp;
      tmp = AA15;
      AA15 = AA45;
      AA45 = tmp;
      tmp = AA16;
      AA16 = AA46;
      AA46 = tmp;
      tmp = AA17;
      AA17 = AA47;
      AA47 = tmp;
      /* update permutation vector based on row swap */
      i = perm1;
      perm1 = perm4;
      perm4 = i;
    }
    if (pvt == 5) {
      tmp = AA10;
      AA10 = AA50;
      AA50 = tmp;
      tmp = AA11;
      AA11 = AA51;
      AA51 = tmp;
      tmp = AA12;
      AA12 = AA52;
      AA52 = tmp;
      tmp = AA13;
      AA13 = AA53;
      AA53 = tmp;
      tmp = AA14;
      AA14 = AA54;
      AA54 = tmp;
      tmp = AA15;
      AA15 = AA55;
      AA55 = tmp;
      tmp = AA16;
      AA16 = AA56;
      AA56 = tmp;
      tmp = AA17;
      AA17 = AA57;
      AA57 = tmp;
      /* update permutation vector based on row swap */
      i = perm1;
      perm1 = perm5;
      perm5 = i;
    }
    if (pvt == 6) {
      tmp = AA10;
      AA10 = AA60;
      AA60 = tmp;
      tmp = AA11;
      AA11 = AA61;
      AA61 = tmp;
      tmp = AA12;
      AA12 = AA62;
      AA62 = tmp;
      tmp = AA13;
      AA13 = AA63;
      AA63 = tmp;
      tmp = AA14;
      AA14 = AA64;
      AA64 = tmp;
      tmp = AA15;
      AA15 = AA65;
      AA65 = tmp;
      tmp = AA16;
      AA16 = AA66;
      AA66 = tmp;
      tmp = AA17;
      AA17 = AA67;
      AA67 = tmp;
      /* update permutation vector based on row swap */
      i = perm1;
      perm1 = perm6;
      perm6 = i;
    }
    if (pvt == 7) {
      tmp = AA10;
      AA10 = AA70;
      AA70 = tmp;
      tmp = AA11;
      AA11 = AA71;
      AA71 = tmp;
      tmp = AA12;
      AA12 = AA72;
      AA72 = tmp;
      tmp = AA13;
      AA13 = AA73;
      AA73 = tmp;
      tmp = AA14;
      AA14 = AA74;
      AA74 = tmp;
      tmp = AA15;
      AA15 = AA75;
      AA75 = tmp;
      tmp = AA16;
      AA16 = AA76;
      AA76 = tmp;
      tmp = AA17;
      AA17 = AA77;
      AA77 = tmp;
      /* update permutation vector based on row swap */
      i = perm1;
      perm1 = perm7;
      perm7 = i;
    }
#endif // USE_PIVOTING

    /* scale current row */
    tmp = rcpOp(AA11);
    icol1 = perm1;
    AA10 = mulOp(tmp, AA10);
    AA11 = tmp;
    AA12 = mulOp(tmp, AA12);
    AA13 = mulOp(tmp, AA13);
    AA14 = mulOp(tmp, AA14);
    AA15 = mulOp(tmp, AA15);
    AA16 = mulOp(tmp, AA16);
    AA17 = mulOp(tmp, AA17);

    /* eliminate above and below current row */
    tmp = AA01;
    AA00 = fmnaOp(tmp, AA10, AA00);
    AA01 = mulOp(negOp(tmp), AA11);
    AA02 = fmnaOp(tmp, AA12, AA02);
    AA03 = fmnaOp(tmp, AA13, AA03);
    AA04 = fmnaOp(tmp, AA14, AA04);
    AA05 = fmnaOp(tmp, AA15, AA05);
    AA06 = fmnaOp(tmp, AA16, AA06);
    AA07 = fmnaOp(tmp, AA17, AA07);

    tmp = AA21;
    AA20 = fmnaOp(tmp, AA10, AA20);
    AA21 = mulOp(negOp(tmp), AA11);
    AA22 = fmnaOp(tmp, AA12, AA22);
    AA23 = fmnaOp(tmp, AA13, AA23);
    AA24 = fmnaOp(tmp, AA14, AA24);
    AA25 = fmnaOp(tmp, AA15, AA25);
    AA26 = fmnaOp(tmp, AA16, AA26);
    AA27 = fmnaOp(tmp, AA17, AA27);

    tmp = AA31;
    AA30 = fmnaOp(tmp, AA10, AA30);
    AA31 = mulOp(negOp(tmp), AA11);
    AA32 = fmnaOp(tmp, AA12, AA32);
    AA33 = fmnaOp(tmp, AA13, AA33);
    AA34 = fmnaOp(tmp, AA14, AA34);
    AA35 = fmnaOp(tmp, AA15, AA35);
    AA36 = fmnaOp(tmp, AA16, AA36);
    AA37 = fmnaOp(tmp, AA17, AA37);

    tmp = AA41;
    AA40 = fmnaOp(tmp, AA10, AA40);
    AA41 = mulOp(negOp(tmp), AA11);
    AA42 = fmnaOp(tmp, AA12, AA42);
    AA43 = fmnaOp(tmp, AA13, AA43);
    AA44 = fmnaOp(tmp, AA14, AA44);
    AA45 = fmnaOp(tmp, AA15, AA45);
    AA46 = fmnaOp(tmp, AA16, AA46);
    AA47 = fmnaOp(tmp, AA17, AA47);

    tmp = AA51;
    AA50 = fmnaOp(tmp, AA10, AA50);
    AA51 = mulOp(negOp(tmp), AA11);
    AA52 = fmnaOp(tmp, AA12, AA52);
    AA53 = fmnaOp(tmp, AA13, AA53);
    AA54 = fmnaOp(tmp, AA14, AA54);
    AA55 = fmnaOp(tmp, AA15, AA55);
    AA56 = fmnaOp(tmp, AA16, AA56);
    AA57 = fmnaOp(tmp, AA17, AA57);

    tmp = AA61;
    AA60 = fmnaOp(tmp, AA10, AA60);
    AA61 = mulOp(negOp(tmp), AA11);
    AA62 = fmnaOp(tmp, AA12, AA62);
    AA63 = fmnaOp(tmp, AA13, AA63);
    AA64 = fmnaOp(tmp, AA14, AA64);
    AA65 = fmnaOp(tmp, AA15, AA65);
    AA66 = fmnaOp(tmp, AA16, AA66);
    AA67 = fmnaOp(tmp, AA17, AA67);

    tmp = AA71;
    AA70 = fmnaOp(tmp, AA10, AA70);
    AA71 = mulOp(negOp(tmp), AA11);
    AA72 = fmnaOp(tmp, AA12, AA72);
    AA73 = fmnaOp(tmp, AA13, AA73);
    AA74 = fmnaOp(tmp, AA14, AA74);
    AA75 = fmnaOp(tmp, AA15, AA75);
    AA76 = fmnaOp(tmp, AA16, AA76);
    AA77 = fmnaOp(tmp, AA17, AA77);

    /****************** iteration 2 ****************/

#if USE_PIVOTING
    /* search pivot row */
    p = absOp(AA22);
    pvt = 2;
    t = absOp(AA32);
    if (t > p) {
      p = t;
      pvt = 3;
    }
    t = absOp(AA42);
    if (t > p) {
      p = t;
      pvt = 4;
    }
    t = absOp(AA52);
    if (t > p) {
      p = t;
      pvt = 5;
    }
    t = absOp(AA62);
    if (t > p) {
      p = t;
      pvt = 6;
    }
    t = absOp(AA72);
    if (t > p) {
      p = t;
      pvt = 7;
    }

    /* swap pivot row with row 2 */
    if (pvt == 3) {
      tmp = AA20;
      AA20 = AA30;
      AA30 = tmp;
      tmp = AA21;
      AA21 = AA31;
      AA31 = tmp;
      tmp = AA22;
      AA22 = AA32;
      AA32 = tmp;
      tmp = AA23;
      AA23 = AA33;
      AA33 = tmp;
      tmp = AA24;
      AA24 = AA34;
      AA34 = tmp;
      tmp = AA25;
      AA25 = AA35;
      AA35 = tmp;
      tmp = AA26;
      AA26 = AA36;
      AA36 = tmp;
      tmp = AA27;
      AA27 = AA37;
      AA37 = tmp;
      /* update permutation vector based on row swap */
      i = perm2;
      perm2 = perm3;
      perm3 = i;
    }
    if (pvt == 4) {
      tmp = AA20;
      AA20 = AA40;
      AA40 = tmp;
      tmp = AA21;
      AA21 = AA41;
      AA41 = tmp;
      tmp = AA22;
      AA22 = AA42;
      AA42 = tmp;
      tmp = AA23;
      AA23 = AA43;
      AA43 = tmp;
      tmp = AA24;
      AA24 = AA44;
      AA44 = tmp;
      tmp = AA25;
      AA25 = AA45;
      AA45 = tmp;
      tmp = AA26;
      AA26 = AA46;
      AA46 = tmp;
      tmp = AA27;
      AA27 = AA47;
      AA47 = tmp;
      /* update permutation vector based on row swap */
      i = perm2;
      perm2 = perm4;
      perm4 = i;
    }
    if (pvt == 5) {
      tmp = AA20;
      AA20 = AA50;
      AA50 = tmp;
      tmp = AA21;
      AA21 = AA51;
      AA51 = tmp;
      tmp = AA22;
      AA22 = AA52;
      AA52 = tmp;
      tmp = AA23;
      AA23 = AA53;
      AA53 = tmp;
      tmp = AA24;
      AA24 = AA54;
      AA54 = tmp;
      tmp = AA25;
      AA25 = AA55;
      AA55 = tmp;
      tmp = AA26;
      AA26 = AA56;
      AA56 = tmp;
      tmp = AA27;
      AA27 = AA57;
      AA57 = tmp;
      /* update permutation vector based on row swap */
      i = perm2;
      perm2 = perm5;
      perm5 = i;
    }
    if (pvt == 6) {
      tmp = AA20;
      AA20 = AA60;
      AA60 = tmp;
      tmp = AA21;
      AA21 = AA61;
      AA61 = tmp;
      tmp = AA22;
      AA22 = AA62;
      AA62 = tmp;
      tmp = AA23;
      AA23 = AA63;
      AA63 = tmp;
      tmp = AA24;
      AA24 = AA64;
      AA64 = tmp;
      tmp = AA25;
      AA25 = AA65;
      AA65 = tmp;
      tmp = AA26;
      AA26 = AA66;
      AA66 = tmp;
      tmp = AA27;
      AA27 = AA67;
      AA67 = tmp;
      /* update permutation vector based on row swap */
      i = perm2;
      perm2 = perm6;
      perm6 = i;
    }
    if (pvt == 7) {
      tmp = AA20;
      AA20 = AA70;
      AA70 = tmp;
      tmp = AA21;
      AA21 = AA71;
      AA71 = tmp;
      tmp = AA22;
      AA22 = AA72;
      AA72 = tmp;
      tmp = AA23;
      AA23 = AA73;
      AA73 = tmp;
      tmp = AA24;
      AA24 = AA74;
      AA74 = tmp;
      tmp = AA25;
      AA25 = AA75;
      AA75 = tmp;
      tmp = AA26;
      AA26 = AA76;
      AA76 = tmp;
      tmp = AA27;
      AA27 = AA77;
      AA77 = tmp;
      /* update permutation vector based on row swap */
      i = perm2;
      perm2 = perm7;
      perm7 = i;
    }
#endif // USE_PIVOTING

    /* scale current row */
    tmp = rcpOp(AA22);
    icol2 = perm2;
    AA20 = mulOp(tmp, AA20);
    AA21 = mulOp(tmp, AA21);
    AA22 = tmp;
    AA23 = mulOp(tmp, AA23);
    AA24 = mulOp(tmp, AA24);
    AA25 = mulOp(tmp, AA25);
    AA26 = mulOp(tmp, AA26);
    AA27 = mulOp(tmp, AA27);

    /* eliminate above and below current row */
    tmp = AA02;
    AA00 = fmnaOp(tmp, AA20, AA00);
    AA01 = fmnaOp(tmp, AA21, AA01);
    AA02 = mulOp(negOp(tmp), AA22);
    AA03 = fmnaOp(tmp, AA23, AA03);
    AA04 = fmnaOp(tmp, AA24, AA04);
    AA05 = fmnaOp(tmp, AA25, AA05);
    AA06 = fmnaOp(tmp, AA26, AA06);
    AA07 = fmnaOp(tmp, AA27, AA07);

    tmp = AA12;
    AA10 = fmnaOp(tmp, AA20, AA10);
    AA11 = fmnaOp(tmp, AA21, AA11);
    AA12 = mulOp(negOp(tmp), AA22);
    AA13 = fmnaOp(tmp, AA23, AA13);
    AA14 = fmnaOp(tmp, AA24, AA14);
    AA15 = fmnaOp(tmp, AA25, AA15);
    AA16 = fmnaOp(tmp, AA26, AA16);
    AA17 = fmnaOp(tmp, AA27, AA17);

    tmp = AA32;
    AA30 = fmnaOp(tmp, AA20, AA30);
    AA31 = fmnaOp(tmp, AA21, AA31);
    AA32 = mulOp(negOp(tmp), AA22);
    AA33 = fmnaOp(tmp, AA23, AA33);
    AA34 = fmnaOp(tmp, AA24, AA34);
    AA35 = fmnaOp(tmp, AA25, AA35);
    AA36 = fmnaOp(tmp, AA26, AA36);
    AA37 = fmnaOp(tmp, AA27, AA37);

    tmp = AA42;
    AA40 = fmnaOp(tmp, AA20, AA40);
    AA41 = fmnaOp(tmp, AA21, AA41);
    AA42 = mulOp(negOp(tmp), AA22);
    AA43 = fmnaOp(tmp, AA23, AA43);
    AA44 = fmnaOp(tmp, AA24, AA44);
    AA45 = fmnaOp(tmp, AA25, AA45);
    AA46 = fmnaOp(tmp, AA26, AA46);
    AA47 = fmnaOp(tmp, AA27, AA47);

    tmp = AA52;
    AA50 = fmnaOp(tmp, AA20, AA50);
    AA51 = fmnaOp(tmp, AA21, AA51);
    AA52 = mulOp(negOp(tmp), AA22);
    AA53 = fmnaOp(tmp, AA23, AA53);
    AA54 = fmnaOp(tmp, AA24, AA54);
    AA55 = fmnaOp(tmp, AA25, AA55);
    AA56 = fmnaOp(tmp, AA26, AA56);
    AA57 = fmnaOp(tmp, AA27, AA57);

    tmp = AA62;
    AA60 = fmnaOp(tmp, AA20, AA60);
    AA61 = fmnaOp(tmp, AA21, AA61);
    AA62 = mulOp(negOp(tmp), AA22);
    AA63 = fmnaOp(tmp, AA23, AA63);
    AA64 = fmnaOp(tmp, AA24, AA64);
    AA65 = fmnaOp(tmp, AA25, AA65);
    AA66 = fmnaOp(tmp, AA26, AA66);
    AA67 = fmnaOp(tmp, AA27, AA67);

    tmp = AA72;
    AA70 = fmnaOp(tmp, AA20, AA70);
    AA71 = fmnaOp(tmp, AA21, AA71);
    AA72 = mulOp(negOp(tmp), AA22);
    AA73 = fmnaOp(tmp, AA23, AA73);
    AA74 = fmnaOp(tmp, AA24, AA74);
    AA75 = fmnaOp(tmp, AA25, AA75);
    AA76 = fmnaOp(tmp, AA26, AA76);
    AA77 = fmnaOp(tmp, AA27, AA77);

    /****************** iteration 3 ****************/

#if USE_PIVOTING
    /* search pivot row */
    p = absOp(AA33);
    pvt = 3;
    t = absOp(AA43);
    if (t > p) {
      p = t;
      pvt = 4;
    }
    t = absOp(AA53);
    if (t > p) {
      p = t;
      pvt = 5;
    }
    t = absOp(AA63);
    if (t > p) {
      p = t;
      pvt = 6;
    }
    t = absOp(AA73);
    if (t > p) {
      p = t;
      pvt = 7;
    }

    /* swap pivot row with row 3 */
    if (pvt == 4) {
      tmp = AA30;
      AA30 = AA40;
      AA40 = tmp;
      tmp = AA31;
      AA31 = AA41;
      AA41 = tmp;
      tmp = AA32;
      AA32 = AA42;
      AA42 = tmp;
      tmp = AA33;
      AA33 = AA43;
      AA43 = tmp;
      tmp = AA34;
      AA34 = AA44;
      AA44 = tmp;
      tmp = AA35;
      AA35 = AA45;
      AA45 = tmp;
      tmp = AA36;
      AA36 = AA46;
      AA46 = tmp;
      tmp = AA37;
      AA37 = AA47;
      AA47 = tmp;
      /* update permutation vector based on row swap */
      i = perm3;
      perm3 = perm4;
      perm4 = i;
    }
    if (pvt == 5) {
      tmp = AA30;
      AA30 = AA50;
      AA50 = tmp;
      tmp = AA31;
      AA31 = AA51;
      AA51 = tmp;
      tmp = AA32;
      AA32 = AA52;
      AA52 = tmp;
      tmp = AA33;
      AA33 = AA53;
      AA53 = tmp;
      tmp = AA34;
      AA34 = AA54;
      AA54 = tmp;
      tmp = AA35;
      AA35 = AA55;
      AA55 = tmp;
      tmp = AA36;
      AA36 = AA56;
      AA56 = tmp;
      tmp = AA37;
      AA37 = AA57;
      AA57 = tmp;
      /* update permutation vector based on row swap */
      i = perm3;
      perm3 = perm5;
      perm5 = i;
    }
    if (pvt == 6) {
      tmp = AA30;
      AA30 = AA60;
      AA60 = tmp;
      tmp = AA31;
      AA31 = AA61;
      AA61 = tmp;
      tmp = AA32;
      AA32 = AA62;
      AA62 = tmp;
      tmp = AA33;
      AA33 = AA63;
      AA63 = tmp;
      tmp = AA34;
      AA34 = AA64;
      AA64 = tmp;
      tmp = AA35;
      AA35 = AA65;
      AA65 = tmp;
      tmp = AA36;
      AA36 = AA66;
      AA66 = tmp;
      tmp = AA37;
      AA37 = AA67;
      AA67 = tmp;
      /* update permutation vector based on row swap */
      i = perm3;
      perm3 = perm6;
      perm6 = i;
    }
    if (pvt == 7) {
      tmp = AA30;
      AA30 = AA70;
      AA70 = tmp;
      tmp = AA31;
      AA31 = AA71;
      AA71 = tmp;
      tmp = AA32;
      AA32 = AA72;
      AA72 = tmp;
      tmp = AA33;
      AA33 = AA73;
      AA73 = tmp;
      tmp = AA34;
      AA34 = AA74;
      AA74 = tmp;
      tmp = AA35;
      AA35 = AA75;
      AA75 = tmp;
      tmp = AA36;
      AA36 = AA76;
      AA76 = tmp;
      tmp = AA37;
      AA37 = AA77;
      AA77 = tmp;
      /* update permutation vector based on row swap */
      i = perm3;
      perm3 = perm7;
      perm7 = i;
    }
#endif // USE_PIVOTING

    /* scale current row */
    tmp = rcpOp(AA33);
    icol3 = perm3;
    AA30 = mulOp(tmp, AA30);
    AA31 = mulOp(tmp, AA31);
    AA32 = mulOp(tmp, AA32);
    AA33 = tmp;
    AA34 = mulOp(tmp, AA34);
    AA35 = mulOp(tmp, AA35);
    AA36 = mulOp(tmp, AA36);
    AA37 = mulOp(tmp, AA37);

    /* eliminate above and below current row */
    tmp = AA03;
    AA00 = fmnaOp(tmp, AA30, AA00);
    AA01 = fmnaOp(tmp, AA31, AA01);
    AA02 = fmnaOp(tmp, AA32, AA02);
    AA03 = mulOp(negOp(tmp), AA33);
    AA04 = fmnaOp(tmp, AA34, AA04);
    AA05 = fmnaOp(tmp, AA35, AA05);
    AA06 = fmnaOp(tmp, AA36, AA06);
    AA07 = fmnaOp(tmp, AA37, AA07);

    tmp = AA13;
    AA10 = fmnaOp(tmp, AA30, AA10);
    AA11 = fmnaOp(tmp, AA31, AA11);
    AA12 = fmnaOp(tmp, AA32, AA12);
    AA13 = mulOp(negOp(tmp), AA33);
    AA14 = fmnaOp(tmp, AA34, AA14);
    AA15 = fmnaOp(tmp, AA35, AA15);
    AA16 = fmnaOp(tmp, AA36, AA16);
    AA17 = fmnaOp(tmp, AA37, AA17);

    tmp = AA23;
    AA20 = fmnaOp(tmp, AA30, AA20);
    AA21 = fmnaOp(tmp, AA31, AA21);
    AA22 = fmnaOp(tmp, AA32, AA22);
    AA23 = mulOp(negOp(tmp), AA33);
    AA24 = fmnaOp(tmp, AA34, AA24);
    AA25 = fmnaOp(tmp, AA35, AA25);
    AA26 = fmnaOp(tmp, AA36, AA26);
    AA27 = fmnaOp(tmp, AA37, AA27);

    tmp = AA43;
    AA40 = fmnaOp(tmp, AA30, AA40);
    AA41 = fmnaOp(tmp, AA31, AA41);
    AA42 = fmnaOp(tmp, AA32, AA42);
    AA43 = mulOp(negOp(tmp), AA33);
    AA44 = fmnaOp(tmp, AA34, AA44);
    AA45 = fmnaOp(tmp, AA35, AA45);
    AA46 = fmnaOp(tmp, AA36, AA46);
    AA47 = fmnaOp(tmp, AA37, AA47);

    tmp = AA53;
    AA50 = fmnaOp(tmp, AA30, AA50);
    AA51 = fmnaOp(tmp, AA31, AA51);
    AA52 = fmnaOp(tmp, AA32, AA52);
    AA53 = mulOp(negOp(tmp), AA33);
    AA54 = fmnaOp(tmp, AA34, AA54);
    AA55 = fmnaOp(tmp, AA35, AA55);
    AA56 = fmnaOp(tmp, AA36, AA56);
    AA57 = fmnaOp(tmp, AA37, AA57);

    tmp = AA63;
    AA60 = fmnaOp(tmp, AA30, AA60);
    AA61 = fmnaOp(tmp, AA31, AA61);
    AA62 = fmnaOp(tmp, AA32, AA62);
    AA63 = mulOp(negOp(tmp), AA33);
    AA64 = fmnaOp(tmp, AA34, AA64);
    AA65 = fmnaOp(tmp, AA35, AA65);
    AA66 = fmnaOp(tmp, AA36, AA66);
    AA67 = fmnaOp(tmp, AA37, AA67);

    tmp = AA73;
    AA70 = fmnaOp(tmp, AA30, AA70);
    AA71 = fmnaOp(tmp, AA31, AA71);
    AA72 = fmnaOp(tmp, AA32, AA72);
    AA73 = mulOp(negOp(tmp), AA33);
    AA74 = fmnaOp(tmp, AA34, AA74);
    AA75 = fmnaOp(tmp, AA35, AA75);
    AA76 = fmnaOp(tmp, AA36, AA76);
    AA77 = fmnaOp(tmp, AA37, AA77);

    /****************** iteration 4 ****************/

#if USE_PIVOTING
    /* search pivot row */
    p = absOp(AA44);
    pvt = 4;
    t = absOp(AA54);
    if (t > p) {
      p = t;
      pvt = 5;
    }
    t = absOp(AA64);
    if (t > p) {
      p = t;
      pvt = 6;
    }
    t = absOp(AA74);
    if (t > p) {
      p = t;
      pvt = 7;
    }

    /* swap pivot row with row 4 */
    if (pvt == 5) {
      tmp = AA40;
      AA40 = AA50;
      AA50 = tmp;
      tmp = AA41;
      AA41 = AA51;
      AA51 = tmp;
      tmp = AA42;
      AA42 = AA52;
      AA52 = tmp;
      tmp = AA43;
      AA43 = AA53;
      AA53 = tmp;
      tmp = AA44;
      AA44 = AA54;
      AA54 = tmp;
      tmp = AA45;
      AA45 = AA55;
      AA55 = tmp;
      tmp = AA46;
      AA46 = AA56;
      AA56 = tmp;
      tmp = AA47;
      AA47 = AA57;
      AA57 = tmp;
      /* update permutation vector based on row swap */
      i = perm4;
      perm4 = perm5;
      perm5 = i;
    }
    if (pvt == 6) {
      tmp = AA40;
      AA40 = AA60;
      AA60 = tmp;
      tmp = AA41;
      AA41 = AA61;
      AA61 = tmp;
      tmp = AA42;
      AA42 = AA62;
      AA62 = tmp;
      tmp = AA43;
      AA43 = AA63;
      AA63 = tmp;
      tmp = AA44;
      AA44 = AA64;
      AA64 = tmp;
      tmp = AA45;
      AA45 = AA65;
      AA65 = tmp;
      tmp = AA46;
      AA46 = AA66;
      AA66 = tmp;
      tmp = AA47;
      AA47 = AA67;
      AA67 = tmp;
      /* update permutation vector based on row swap */
      i = perm4;
      perm4 = perm6;
      perm6 = i;
    }
    if (pvt == 7) {
      tmp = AA40;
      AA40 = AA70;
      AA70 = tmp;
      tmp = AA41;
      AA41 = AA71;
      AA71 = tmp;
      tmp = AA42;
      AA42 = AA72;
      AA72 = tmp;
      tmp = AA43;
      AA43 = AA73;
      AA73 = tmp;
      tmp = AA44;
      AA44 = AA74;
      AA74 = tmp;
      tmp = AA45;
      AA45 = AA75;
      AA75 = tmp;
      tmp = AA46;
      AA46 = AA76;
      AA76 = tmp;
      tmp = AA47;
      AA47 = AA77;
      AA77 = tmp;
      /* update permutation vector based on row swap */
      i = perm4;
      perm4 = perm7;
      perm7 = i;
    }
#endif // USE_PIVOTING

    /* scale current row */
    tmp = rcpOp(AA44);
    icol4 = perm4;
    AA40 = mulOp(tmp, AA40);
    AA41 = mulOp(tmp, AA41);
    AA42 = mulOp(tmp, AA42);
    AA43 = mulOp(tmp, AA43);
    AA44 = tmp;
    AA45 = mulOp(tmp, AA45);
    AA46 = mulOp(tmp, AA46);
    AA47 = mulOp(tmp, AA47);

    /* eliminate above and below current row */
    tmp = AA04;
    AA00 = fmnaOp(tmp, AA40, AA00);
    AA01 = fmnaOp(tmp, AA41, AA01);
    AA02 = fmnaOp(tmp, AA42, AA02);
    AA03 = fmnaOp(tmp, AA43, AA03);
    AA04 = mulOp(negOp(tmp), AA44);
    AA05 = fmnaOp(tmp, AA45, AA05);
    AA06 = fmnaOp(tmp, AA46, AA06);
    AA07 = fmnaOp(tmp, AA47, AA07);

    tmp = AA14;
    AA10 = fmnaOp(tmp, AA40, AA10);
    AA11 = fmnaOp(tmp, AA41, AA11);
    AA12 = fmnaOp(tmp, AA42, AA12);
    AA13 = fmnaOp(tmp, AA43, AA13);
    AA14 = mulOp(negOp(tmp), AA44);
    AA15 = fmnaOp(tmp, AA45, AA15);
    AA16 = fmnaOp(tmp, AA46, AA16);
    AA17 = fmnaOp(tmp, AA47, AA17);

    tmp = AA24;
    AA20 = fmnaOp(tmp, AA40, AA20);
    AA21 = fmnaOp(tmp, AA41, AA21);
    AA22 = fmnaOp(tmp, AA42, AA22);
    AA23 = fmnaOp(tmp, AA43, AA23);
    AA24 = mulOp(negOp(tmp), AA44);
    AA25 = fmnaOp(tmp, AA45, AA25);
    AA26 = fmnaOp(tmp, AA46, AA26);
    AA27 = fmnaOp(tmp, AA47, AA27);

    tmp = AA34;
    AA30 = fmnaOp(tmp, AA40, AA30);
    AA31 = fmnaOp(tmp, AA41, AA31);
    AA32 = fmnaOp(tmp, AA42, AA32);
    AA33 = fmnaOp(tmp, AA43, AA33);
    AA34 = mulOp(negOp(tmp), AA44);
    AA35 = fmnaOp(tmp, AA45, AA35);
    AA36 = fmnaOp(tmp, AA46, AA36);
    AA37 = fmnaOp(tmp, AA47, AA37);

    tmp = AA54;
    AA50 = fmnaOp(tmp, AA40, AA50);
    AA51 = fmnaOp(tmp, AA41, AA51);
    AA52 = fmnaOp(tmp, AA42, AA52);
    AA53 = fmnaOp(tmp, AA43, AA53);
    AA54 = mulOp(negOp(tmp), AA44);
    AA55 = fmnaOp(tmp, AA45, AA55);
    AA56 = fmnaOp(tmp, AA46, AA56);
    AA57 = fmnaOp(tmp, AA47, AA57);

    tmp = AA64;
    AA60 = fmnaOp(tmp, AA40, AA60);
    AA61 = fmnaOp(tmp, AA41, AA61);
    AA62 = fmnaOp(tmp, AA42, AA62);
    AA63 = fmnaOp(tmp, AA43, AA63);
    AA64 = mulOp(negOp(tmp), AA44);
    AA65 = fmnaOp(tmp, AA45, AA65);
    AA66 = fmnaOp(tmp, AA46, AA66);
    AA67 = fmnaOp(tmp, AA47, AA67);

    tmp = AA74;
    AA70 = fmnaOp(tmp, AA40, AA70);
    AA71 = fmnaOp(tmp, AA41, AA71);
    AA72 = fmnaOp(tmp, AA42, AA72);
    AA73 = fmnaOp(tmp, AA43, AA73);
    AA74 = mulOp(negOp(tmp), AA44);
    AA75 = fmnaOp(tmp, AA45, AA75);
    AA76 = fmnaOp(tmp, AA46, AA76);
    AA77 = fmnaOp(tmp, AA47, AA77);

    /****************** iteration 5 ****************/

#if USE_PIVOTING
    /* search pivot row */
    p = absOp(AA55);
    pvt = 5;
    t = absOp(AA65);
    if (t > p) {
      p = t;
      pvt = 6;
    }
    t = absOp(AA75);
    if (t > p) {
      p = t;
      pvt = 7;
    }

    /* swap pivot row with row 5 */
    if (pvt == 6) {
      tmp = AA50;
      AA50 = AA60;
      AA60 = tmp;
      tmp = AA51;
      AA51 = AA61;
      AA61 = tmp;
      tmp = AA52;
      AA52 = AA62;
      AA62 = tmp;
      tmp = AA53;
      AA53 = AA63;
      AA63 = tmp;
      tmp = AA54;
      AA54 = AA64;
      AA64 = tmp;
      tmp = AA55;
      AA55 = AA65;
      AA65 = tmp;
      tmp = AA56;
      AA56 = AA66;
      AA66 = tmp;
      tmp = AA57;
      AA57 = AA67;
      AA67 = tmp;
      /* update permutation vector based on row swap */
      i = perm5;
      perm5 = perm6;
      perm6 = i;
    }
    if (pvt == 7) {
      tmp = AA50;
      AA50 = AA70;
      AA70 = tmp;
      tmp = AA51;
      AA51 = AA71;
      AA71 = tmp;
      tmp = AA52;
      AA52 = AA72;
      AA72 = tmp;
      tmp = AA53;
      AA53 = AA73;
      AA73 = tmp;
      tmp = AA54;
      AA54 = AA74;
      AA74 = tmp;
      tmp = AA55;
      AA55 = AA75;
      AA75 = tmp;
      tmp = AA56;
      AA56 = AA76;
      AA76 = tmp;
      tmp = AA57;
      AA57 = AA77;
      AA77 = tmp;
      /* update permutation vector based on row swap */
      i = perm5;
      perm5 = perm7;
      perm7 = i;
    }
#endif // USE_PIVOTING

    /* scale current row */
    tmp = rcpOp(AA55);
    icol5 = perm5;
    AA50 = mulOp(tmp, AA50);
    AA51 = mulOp(tmp, AA51);
    AA52 = mulOp(tmp, AA52);
    AA53 = mulOp(tmp, AA53);
    AA54 = mulOp(tmp, AA54);
    AA55 = tmp;
    AA56 = mulOp(tmp, AA56);
    AA57 = mulOp(tmp, AA57);

    /* eliminate above and below current row */
    tmp = AA05;
    AA00 = fmnaOp(tmp, AA50, AA00);
    AA01 = fmnaOp(tmp, AA51, AA01);
    AA02 = fmnaOp(tmp, AA52, AA02);
    AA03 = fmnaOp(tmp, AA53, AA03);
    AA04 = fmnaOp(tmp, AA54, AA04);
    AA05 = mulOp(negOp(tmp), AA55);
    AA06 = fmnaOp(tmp, AA56, AA06);
    AA07 = fmnaOp(tmp, AA57, AA07);

    tmp = AA15;
    AA10 = fmnaOp(tmp, AA50, AA10);
    AA11 = fmnaOp(tmp, AA51, AA11);
    AA12 = fmnaOp(tmp, AA52, AA12);
    AA13 = fmnaOp(tmp, AA53, AA13);
    AA14 = fmnaOp(tmp, AA54, AA14);
    AA15 = mulOp(negOp(tmp), AA55);
    AA16 = fmnaOp(tmp, AA56, AA16);
    AA17 = fmnaOp(tmp, AA57, AA17);

    tmp = AA25;
    AA20 = fmnaOp(tmp, AA50, AA20);
    AA21 = fmnaOp(tmp, AA51, AA21);
    AA22 = fmnaOp(tmp, AA52, AA22);
    AA23 = fmnaOp(tmp, AA53, AA23);
    AA24 = fmnaOp(tmp, AA54, AA24);
    AA25 = mulOp(negOp(tmp), AA55);
    AA26 = fmnaOp(tmp, AA56, AA26);
    AA27 = fmnaOp(tmp, AA57, AA27);

    tmp = AA35;
    AA30 = fmnaOp(tmp, AA50, AA30);
    AA31 = fmnaOp(tmp, AA51, AA31);
    AA32 = fmnaOp(tmp, AA52, AA32);
    AA33 = fmnaOp(tmp, AA53, AA33);
    AA34 = fmnaOp(tmp, AA54, AA34);
    AA35 = mulOp(negOp(tmp), AA55);
    AA36 = fmnaOp(tmp, AA56, AA36);
    AA37 = fmnaOp(tmp, AA57, AA37);

    tmp = AA45;
    AA40 = fmnaOp(tmp, AA50, AA40);
    AA41 = fmnaOp(tmp, AA51, AA41);
    AA42 = fmnaOp(tmp, AA52, AA42);
    AA43 = fmnaOp(tmp, AA53, AA43);
    AA44 = fmnaOp(tmp, AA54, AA44);
    AA45 = mulOp(negOp(tmp), AA55);
    AA46 = fmnaOp(tmp, AA56, AA46);
    AA47 = fmnaOp(tmp, AA57, AA47);

    tmp = AA65;
    AA60 = fmnaOp(tmp, AA50, AA60);
    AA61 = fmnaOp(tmp, AA51, AA61);
    AA62 = fmnaOp(tmp, AA52, AA62);
    AA63 = fmnaOp(tmp, AA53, AA63);
    AA64 = fmnaOp(tmp, AA54, AA64);
    AA65 = mulOp(negOp(tmp), AA55);
    AA66 = fmnaOp(tmp, AA56, AA66);
    AA67 = fmnaOp(tmp, AA57, AA67);

    tmp = AA75;
    AA70 = fmnaOp(tmp, AA50, AA70);
    AA71 = fmnaOp(tmp, AA51, AA71);
    AA72 = fmnaOp(tmp, AA52, AA72);
    AA73 = fmnaOp(tmp, AA53, AA73);
    AA74 = fmnaOp(tmp, AA54, AA74);
    AA75 = mulOp(negOp(tmp), AA55);
    AA76 = fmnaOp(tmp, AA56, AA76);
    AA77 = fmnaOp(tmp, AA57, AA77);

    /****************** iteration 6 ****************/

#if USE_PIVOTING
    /* search pivot row */
    p = absOp(AA66);
    pvt = 6;
    t = absOp(AA76);
    if (t > p) {
      p = t;
      pvt = 7;
    }

    /* swap pivot row with row 6 */
    if (pvt == 7) {
      tmp = AA60;
      AA60 = AA70;
      AA70 = tmp;
      tmp = AA61;
      AA61 = AA71;
      AA71 = tmp;
      tmp = AA62;
      AA62 = AA72;
      AA72 = tmp;
      tmp = AA63;
      AA63 = AA73;
      AA73 = tmp;
      tmp = AA64;
      AA64 = AA74;
      AA74 = tmp;
      tmp = AA65;
      AA65 = AA75;
      AA75 = tmp;
      tmp = AA66;
      AA66 = AA76;
      AA76 = tmp;
      tmp = AA67;
      AA67 = AA77;
      AA77 = tmp;
      /* update permutation vector based on row swap */
      i = perm6;
      perm6 = perm7;
      perm7 = i;
    }
#endif // USE_PIVOTING

    /* scale current row */
    tmp = rcpOp(AA66);
    icol6 = perm6;
    AA60 = mulOp(tmp, AA60);
    AA61 = mulOp(tmp, AA61);
    AA62 = mulOp(tmp, AA62);
    AA63 = mulOp(tmp, AA63);
    AA64 = mulOp(tmp, AA64);
    AA65 = mulOp(tmp, AA65);
    AA66 = tmp;
    AA67 = mulOp(tmp, AA67);

    /* eliminate above and below current row */
    tmp = AA06;
    AA00 = fmnaOp(tmp, AA60, AA00);
    AA01 = fmnaOp(tmp, AA61, AA01);
    AA02 = fmnaOp(tmp, AA62, AA02);
    AA03 = fmnaOp(tmp, AA63, AA03);
    AA04 = fmnaOp(tmp, AA64, AA04);
    AA05 = fmnaOp(tmp, AA65, AA05);
    AA06 = mulOp(negOp(tmp), AA66);
    AA07 = fmnaOp(tmp, AA67, AA07);

    tmp = AA16;
    AA10 = fmnaOp(tmp, AA60, AA10);
    AA11 = fmnaOp(tmp, AA61, AA11);
    AA12 = fmnaOp(tmp, AA62, AA12);
    AA13 = fmnaOp(tmp, AA63, AA13);
    AA14 = fmnaOp(tmp, AA64, AA14);
    AA15 = fmnaOp(tmp, AA65, AA15);
    AA16 = mulOp(negOp(tmp), AA66);
    AA17 = fmnaOp(tmp, AA67, AA17);

    tmp = AA26;
    AA20 = fmnaOp(tmp, AA60, AA20);
    AA21 = fmnaOp(tmp, AA61, AA21);
    AA22 = fmnaOp(tmp, AA62, AA22);
    AA23 = fmnaOp(tmp, AA63, AA23);
    AA24 = fmnaOp(tmp, AA64, AA24);
    AA25 = fmnaOp(tmp, AA65, AA25);
    AA26 = mulOp(negOp(tmp), AA66);
    AA27 = fmnaOp(tmp, AA67, AA27);

    tmp = AA36;
    AA30 = fmnaOp(tmp, AA60, AA30);
    AA31 = fmnaOp(tmp, AA61, AA31);
    AA32 = fmnaOp(tmp, AA62, AA32);
    AA33 = fmnaOp(tmp, AA63, AA33);
    AA34 = fmnaOp(tmp, AA64, AA34);
    AA35 = fmnaOp(tmp, AA65, AA35);
    AA36 = mulOp(negOp(tmp), AA66);
    AA37 = fmnaOp(tmp, AA67, AA37);

    tmp = AA46;
    AA40 = fmnaOp(tmp, AA60, AA40);
    AA41 = fmnaOp(tmp, AA61, AA41);
    AA42 = fmnaOp(tmp, AA62, AA42);
    AA43 = fmnaOp(tmp, AA63, AA43);
    AA44 = fmnaOp(tmp, AA64, AA44);
    AA45 = fmnaOp(tmp, AA65, AA45);
    AA46 = mulOp(negOp(tmp), AA66);
    AA47 = fmnaOp(tmp, AA67, AA47);

    tmp = AA56;
    AA50 = fmnaOp(tmp, AA60, AA50);
    AA51 = fmnaOp(tmp, AA61, AA51);
    AA52 = fmnaOp(tmp, AA62, AA52);
    AA53 = fmnaOp(tmp, AA63, AA53);
    AA54 = fmnaOp(tmp, AA64, AA54);
    AA55 = fmnaOp(tmp, AA65, AA55);
    AA56 = mulOp(negOp(tmp), AA66);
    AA57 = fmnaOp(tmp, AA67, AA57);

    tmp = AA76;
    AA70 = fmnaOp(tmp, AA60, AA70);
    AA71 = fmnaOp(tmp, AA61, AA71);
    AA72 = fmnaOp(tmp, AA62, AA72);
    AA73 = fmnaOp(tmp, AA63, AA73);
    AA74 = fmnaOp(tmp, AA64, AA74);
    AA75 = fmnaOp(tmp, AA65, AA75);
    AA76 = mulOp(negOp(tmp), AA66);
    AA77 = fmnaOp(tmp, AA67, AA77);

    /****************** iteration 7 ****************/

    /* scale current row */
    tmp = rcpOp(AA77);
    icol7 = perm7;
    AA70 = mulOp(tmp, AA70);
    AA71 = mulOp(tmp, AA71);
    AA72 = mulOp(tmp, AA72);
    AA73 = mulOp(tmp, AA73);
    AA74 = mulOp(tmp, AA74);
    AA75 = mulOp(tmp, AA75);
    AA76 = mulOp(tmp, AA76);
    AA77 = tmp;

    /* eliminate above and below current row */
    tmp = AA07;
    AA00 = fmnaOp(tmp, AA70, AA00);
    AA01 = fmnaOp(tmp, AA71, AA01);
    AA02 = fmnaOp(tmp, AA72, AA02);
    AA03 = fmnaOp(tmp, AA73, AA03);
    AA04 = fmnaOp(tmp, AA74, AA04);
    AA05 = fmnaOp(tmp, AA75, AA05);
    AA06 = fmnaOp(tmp, AA76, AA06);
    AA07 = mulOp(negOp(tmp), AA77);

    tmp = AA17;
    AA10 = fmnaOp(tmp, AA70, AA10);
    AA11 = fmnaOp(tmp, AA71, AA11);
    AA12 = fmnaOp(tmp, AA72, AA12);
    AA13 = fmnaOp(tmp, AA73, AA13);
    AA14 = fmnaOp(tmp, AA74, AA14);
    AA15 = fmnaOp(tmp, AA75, AA15);
    AA16 = fmnaOp(tmp, AA76, AA16);
    AA17 = mulOp(negOp(tmp), AA77);

    tmp = AA27;
    AA20 = fmnaOp(tmp, AA70, AA20);
    AA21 = fmnaOp(tmp, AA71, AA21);
    AA22 = fmnaOp(tmp, AA72, AA22);
    AA23 = fmnaOp(tmp, AA73, AA23);
    AA24 = fmnaOp(tmp, AA74, AA24);
    AA25 = fmnaOp(tmp, AA75, AA25);
    AA26 = fmnaOp(tmp, AA76, AA26);
    AA27 = mulOp(negOp(tmp), AA77);

    tmp = AA37;
    AA30 = fmnaOp(tmp, AA70, AA30);
    AA31 = fmnaOp(tmp, AA71, AA31);
    AA32 = fmnaOp(tmp, AA72, AA32);
    AA33 = fmnaOp(tmp, AA73, AA33);
    AA34 = fmnaOp(tmp, AA74, AA34);
    AA35 = fmnaOp(tmp, AA75, AA35);
    AA36 = fmnaOp(tmp, AA76, AA36);
    AA37 = mulOp(negOp(tmp), AA77);

    tmp = AA47;
    AA40 = fmnaOp(tmp, AA70, AA40);
    AA41 = fmnaOp(tmp, AA71, AA41);
    AA42 = fmnaOp(tmp, AA72, AA42);
    AA43 = fmnaOp(tmp, AA73, AA43);
    AA44 = fmnaOp(tmp, AA74, AA44);
    AA45 = fmnaOp(tmp, AA75, AA45);
    AA46 = fmnaOp(tmp, AA76, AA46);
    AA47 = mulOp(negOp(tmp), AA77);

    tmp = AA57;
    AA50 = fmnaOp(tmp, AA70, AA50);
    AA51 = fmnaOp(tmp, AA71, AA51);
    AA52 = fmnaOp(tmp, AA72, AA52);
    AA53 = fmnaOp(tmp, AA73, AA53);
    AA54 = fmnaOp(tmp, AA74, AA54);
    AA55 = fmnaOp(tmp, AA75, AA55);
    AA56 = fmnaOp(tmp, AA76, AA56);
    AA57 = mulOp(negOp(tmp), AA77);

    tmp = AA67;
    AA60 = fmnaOp(tmp, AA70, AA60);
    AA61 = fmnaOp(tmp, AA71, AA61);
    AA62 = fmnaOp(tmp, AA72, AA62);
    AA63 = fmnaOp(tmp, AA73, AA63);
    AA64 = fmnaOp(tmp, AA74, AA64);
    AA65 = fmnaOp(tmp, AA75, AA65);
    AA66 = fmnaOp(tmp, AA76, AA66);
    AA67 = mulOp(negOp(tmp), AA77);

    /* sort columns into the correct order */
    Ainv(0, icol0) = AA00;
    Ainv(1, icol0) = AA10;
    Ainv(2, icol0) = AA20;
    Ainv(3, icol0) = AA30;
    Ainv(4, icol0) = AA40;
    Ainv(5, icol0) = AA50;
    Ainv(6, icol0) = AA60;
    Ainv(7, icol0) = AA70;
    Ainv(0, icol1) = AA01;
    Ainv(1, icol1) = AA11;
    Ainv(2, icol1) = AA21;
    Ainv(3, icol1) = AA31;
    Ainv(4, icol1) = AA41;
    Ainv(5, icol1) = AA51;
    Ainv(6, icol1) = AA61;
    Ainv(7, icol1) = AA71;
    Ainv(0, icol2) = AA02;
    Ainv(1, icol2) = AA12;
    Ainv(2, icol2) = AA22;
    Ainv(3, icol2) = AA32;
    Ainv(4, icol2) = AA42;
    Ainv(5, icol2) = AA52;
    Ainv(6, icol2) = AA62;
    Ainv(7, icol2) = AA72;
    Ainv(0, icol3) = AA03;
    Ainv(1, icol3) = AA13;
    Ainv(2, icol3) = AA23;
    Ainv(3, icol3) = AA33;
    Ainv(4, icol3) = AA43;
    Ainv(5, icol3) = AA53;
    Ainv(6, icol3) = AA63;
    Ainv(7, icol3) = AA73;
    Ainv(0, icol4) = AA04;
    Ainv(1, icol4) = AA14;
    Ainv(2, icol4) = AA24;
    Ainv(3, icol4) = AA34;
    Ainv(4, icol4) = AA44;
    Ainv(5, icol4) = AA54;
    Ainv(6, icol4) = AA64;
    Ainv(7, icol4) = AA74;
    Ainv(0, icol5) = AA05;
    Ainv(1, icol5) = AA15;
    Ainv(2, icol5) = AA25;
    Ainv(3, icol5) = AA35;
    Ainv(4, icol5) = AA45;
    Ainv(5, icol5) = AA55;
    Ainv(6, icol5) = AA65;
    Ainv(7, icol5) = AA75;
    Ainv(0, icol6) = AA06;
    Ainv(1, icol6) = AA16;
    Ainv(2, icol6) = AA26;
    Ainv(3, icol6) = AA36;
    Ainv(4, icol6) = AA46;
    Ainv(5, icol6) = AA56;
    Ainv(6, icol6) = AA66;
    Ainv(7, icol6) = AA76;
    Ainv(0, icol7) = AA07;
    Ainv(1, icol7) = AA17;
    Ainv(2, icol7) = AA27;
    Ainv(3, icol7) = AA37;
    Ainv(4, icol7) = AA47;
    Ainv(5, icol7) = AA57;
    Ainv(6, icol7) = AA67;
    Ainv(7, icol7) = AA77;
  }
}

template <typename T, int arch>
__global__ void lu_matinv_9x9_matrix_per_thread(const T *A, T *Ainv, int batch) {
  const int blkNum = blockIdx.y * gridDim.x + blockIdx.x;
  const int thrdNum = blkNum * blockDim.x + threadIdx.x;
  const int N = 9;
  int perm0, perm1, perm2, perm3, perm4, perm5, perm6, perm7, perm8;
  int icol0, icol1, icol2, icol3, icol4, icol5, icol6, icol7, icol8;
  T AA00, AA01, AA02, AA03, AA04, AA05, AA06, AA07, AA08;
  T AA10, AA11, AA12, AA13, AA14, AA15, AA16, AA17, AA18;
  T AA20, AA21, AA22, AA23, AA24, AA25, AA26, AA27, AA28;
  T AA30, AA31, AA32, AA33, AA34, AA35, AA36, AA37, AA38;
  T AA40, AA41, AA42, AA43, AA44, AA45, AA46, AA47, AA48;
  T AA50, AA51, AA52, AA53, AA54, AA55, AA56, AA57, AA58;
  T AA60, AA61, AA62, AA63, AA64, AA65, AA66, AA67, AA68;
  T AA70, AA71, AA72, AA73, AA74, AA75, AA76, AA77, AA78;
  T AA80, AA81, AA82, AA83, AA84, AA85, AA86, AA87, AA88;
  T tmp;
#if USE_PIVOTING
  typename config<T, arch>::absValType t;
  typename config<T, arch>::absValType p;
  int i, pvt;
#endif

  if (thrdNum < batch) {

    AA00 = A[0 * batch + thrdNum];
    AA10 = A[1 * batch + thrdNum];
    AA20 = A[2 * batch + thrdNum];
    AA30 = A[3 * batch + thrdNum];
    AA40 = A[4 * batch + thrdNum];
    AA50 = A[5 * batch + thrdNum];
    AA60 = A[6 * batch + thrdNum];
    AA70 = A[7 * batch + thrdNum];
    AA80 = A[8 * batch + thrdNum];
    AA01 = A[9 * batch + thrdNum];
    AA11 = A[10 * batch + thrdNum];
    AA21 = A[11 * batch + thrdNum];
    AA31 = A[12 * batch + thrdNum];
    AA41 = A[13 * batch + thrdNum];
    AA51 = A[14 * batch + thrdNum];
    AA61 = A[15 * batch + thrdNum];
    AA71 = A[16 * batch + thrdNum];
    AA81 = A[17 * batch + thrdNum];
    AA02 = A[18 * batch + thrdNum];
    AA12 = A[19 * batch + thrdNum];
    AA22 = A[20 * batch + thrdNum];
    AA32 = A[21 * batch + thrdNum];
    AA42 = A[22 * batch + thrdNum];
    AA52 = A[23 * batch + thrdNum];
    AA62 = A[24 * batch + thrdNum];
    AA72 = A[25 * batch + thrdNum];
    AA82 = A[26 * batch + thrdNum];
    AA03 = A[27 * batch + thrdNum];
    AA13 = A[28 * batch + thrdNum];
    AA23 = A[29 * batch + thrdNum];
    AA33 = A[30 * batch + thrdNum];
    AA43 = A[31 * batch + thrdNum];
    AA53 = A[32 * batch + thrdNum];
    AA63 = A[33 * batch + thrdNum];
    AA73 = A[34 * batch + thrdNum];
    AA83 = A[35 * batch + thrdNum];
    AA04 = A[36 * batch + thrdNum];
    AA14 = A[37 * batch + thrdNum];
    AA24 = A[38 * batch + thrdNum];
    AA34 = A[39 * batch + thrdNum];
    AA44 = A[40 * batch + thrdNum];
    AA54 = A[41 * batch + thrdNum];
    AA64 = A[42 * batch + thrdNum];
    AA74 = A[43 * batch + thrdNum];
    AA84 = A[44 * batch + thrdNum];
    AA05 = A[45 * batch + thrdNum];
    AA15 = A[46 * batch + thrdNum];
    AA25 = A[47 * batch + thrdNum];
    AA35 = A[48 * batch + thrdNum];
    AA45 = A[49 * batch + thrdNum];
    AA55 = A[50 * batch + thrdNum];
    AA65 = A[51 * batch + thrdNum];
    AA75 = A[52 * batch + thrdNum];
    AA85 = A[53 * batch + thrdNum];
    AA06 = A[54 * batch + thrdNum];
    AA16 = A[55 * batch + thrdNum];
    AA26 = A[56 * batch + thrdNum];
    AA36 = A[57 * batch + thrdNum];
    AA46 = A[58 * batch + thrdNum];
    AA56 = A[59 * batch + thrdNum];
    AA66 = A[60 * batch + thrdNum];
    AA76 = A[61 * batch + thrdNum];
    AA86 = A[62 * batch + thrdNum];
    AA07 = A[63 * batch + thrdNum];
    AA17 = A[64 * batch + thrdNum];
    AA27 = A[65 * batch + thrdNum];
    AA37 = A[66 * batch + thrdNum];
    AA47 = A[67 * batch + thrdNum];
    AA57 = A[68 * batch + thrdNum];
    AA67 = A[69 * batch + thrdNum];
    AA77 = A[70 * batch + thrdNum];
    AA87 = A[71 * batch + thrdNum];
    AA08 = A[72 * batch + thrdNum];
    AA18 = A[73 * batch + thrdNum];
    AA28 = A[74 * batch + thrdNum];
    AA38 = A[75 * batch + thrdNum];
    AA48 = A[76 * batch + thrdNum];
    AA58 = A[77 * batch + thrdNum];
    AA68 = A[78 * batch + thrdNum];
    AA78 = A[79 * batch + thrdNum];
    AA88 = A[80 * batch + thrdNum];

    perm0 = 0;
    perm1 = 1;
    perm2 = 2;
    perm3 = 3;
    perm4 = 4;
    perm5 = 5;
    perm6 = 6;
    perm7 = 7;
    perm8 = 8;

    /****************** iteration 0 ***********/

#if USE_PIVOTING
    /* search pivot row */
    p = absOp(AA00);
    pvt = 0;
    t = absOp(AA10);
    if (t > p) {
      p = t;
      pvt = 1;
    }
    t = absOp(AA20);
    if (t > p) {
      p = t;
      pvt = 2;
    }
    t = absOp(AA30);
    if (t > p) {
      p = t;
      pvt = 3;
    }
    t = absOp(AA40);
    if (t > p) {
      p = t;
      pvt = 4;
    }
    t = absOp(AA50);
    if (t > p) {
      p = t;
      pvt = 5;
    }
    t = absOp(AA60);
    if (t > p) {
      p = t;
      pvt = 6;
    }
    t = absOp(AA70);
    if (t > p) {
      p = t;
      pvt = 7;
    }
    t = absOp(AA80);
    if (t > p) {
      p = t;
      pvt = 8;
    }

    /* swap pivot row with row 0 */
    if (pvt == 1) {
      tmp = AA00;
      AA00 = AA10;
      AA10 = tmp;
      tmp = AA01;
      AA01 = AA11;
      AA11 = tmp;
      tmp = AA02;
      AA02 = AA12;
      AA12 = tmp;
      tmp = AA03;
      AA03 = AA13;
      AA13 = tmp;
      tmp = AA04;
      AA04 = AA14;
      AA14 = tmp;
      tmp = AA05;
      AA05 = AA15;
      AA15 = tmp;
      tmp = AA06;
      AA06 = AA16;
      AA16 = tmp;
      tmp = AA07;
      AA07 = AA17;
      AA17 = tmp;
      tmp = AA08;
      AA08 = AA18;
      AA18 = tmp;
      /* update permutation vector based on row swap */
      i = perm0;
      perm0 = perm1;
      perm1 = i;
    }
    if (pvt == 2) {
      tmp = AA00;
      AA00 = AA20;
      AA20 = tmp;
      tmp = AA01;
      AA01 = AA21;
      AA21 = tmp;
      tmp = AA02;
      AA02 = AA22;
      AA22 = tmp;
      tmp = AA03;
      AA03 = AA23;
      AA23 = tmp;
      tmp = AA04;
      AA04 = AA24;
      AA24 = tmp;
      tmp = AA05;
      AA05 = AA25;
      AA25 = tmp;
      tmp = AA06;
      AA06 = AA26;
      AA26 = tmp;
      tmp = AA07;
      AA07 = AA27;
      AA27 = tmp;
      tmp = AA08;
      AA08 = AA28;
      AA28 = tmp;
      /* update permutation vector based on row swap */
      i = perm0;
      perm0 = perm2;
      perm2 = i;
    }
    if (pvt == 3) {
      tmp = AA00;
      AA00 = AA30;
      AA30 = tmp;
      tmp = AA01;
      AA01 = AA31;
      AA31 = tmp;
      tmp = AA02;
      AA02 = AA32;
      AA32 = tmp;
      tmp = AA03;
      AA03 = AA33;
      AA33 = tmp;
      tmp = AA04;
      AA04 = AA34;
      AA34 = tmp;
      tmp = AA05;
      AA05 = AA35;
      AA35 = tmp;
      tmp = AA06;
      AA06 = AA36;
      AA36 = tmp;
      tmp = AA07;
      AA07 = AA37;
      AA37 = tmp;
      tmp = AA08;
      AA08 = AA38;
      AA38 = tmp;
      /* update permutation vector based on row swap */
      i = perm0;
      perm0 = perm3;
      perm3 = i;
    }
    if (pvt == 4) {
      tmp = AA00;
      AA00 = AA40;
      AA40 = tmp;
      tmp = AA01;
      AA01 = AA41;
      AA41 = tmp;
      tmp = AA02;
      AA02 = AA42;
      AA42 = tmp;
      tmp = AA03;
      AA03 = AA43;
      AA43 = tmp;
      tmp = AA04;
      AA04 = AA44;
      AA44 = tmp;
      tmp = AA05;
      AA05 = AA45;
      AA45 = tmp;
      tmp = AA06;
      AA06 = AA46;
      AA46 = tmp;
      tmp = AA07;
      AA07 = AA47;
      AA47 = tmp;
      tmp = AA08;
      AA08 = AA48;
      AA48 = tmp;
      /* update permutation vector based on row swap */
      i = perm0;
      perm0 = perm4;
      perm4 = i;
    }
    if (pvt == 5) {
      tmp = AA00;
      AA00 = AA50;
      AA50 = tmp;
      tmp = AA01;
      AA01 = AA51;
      AA51 = tmp;
      tmp = AA02;
      AA02 = AA52;
      AA52 = tmp;
      tmp = AA03;
      AA03 = AA53;
      AA53 = tmp;
      tmp = AA04;
      AA04 = AA54;
      AA54 = tmp;
      tmp = AA05;
      AA05 = AA55;
      AA55 = tmp;
      tmp = AA06;
      AA06 = AA56;
      AA56 = tmp;
      tmp = AA07;
      AA07 = AA57;
      AA57 = tmp;
      tmp = AA08;
      AA08 = AA58;
      AA58 = tmp;
      /* update permutation vector based on row swap */
      i = perm0;
      perm0 = perm5;
      perm5 = i;
    }
    if (pvt == 6) {
      tmp = AA00;
      AA00 = AA60;
      AA60 = tmp;
      tmp = AA01;
      AA01 = AA61;
      AA61 = tmp;
      tmp = AA02;
      AA02 = AA62;
      AA62 = tmp;
      tmp = AA03;
      AA03 = AA63;
      AA63 = tmp;
      tmp = AA04;
      AA04 = AA64;
      AA64 = tmp;
      tmp = AA05;
      AA05 = AA65;
      AA65 = tmp;
      tmp = AA06;
      AA06 = AA66;
      AA66 = tmp;
      tmp = AA07;
      AA07 = AA67;
      AA67 = tmp;
      tmp = AA08;
      AA08 = AA68;
      AA68 = tmp;
      /* update permutation vector based on row swap */
      i = perm0;
      perm0 = perm6;
      perm6 = i;
    }
    if (pvt == 7) {
      tmp = AA00;
      AA00 = AA70;
      AA70 = tmp;
      tmp = AA01;
      AA01 = AA71;
      AA71 = tmp;
      tmp = AA02;
      AA02 = AA72;
      AA72 = tmp;
      tmp = AA03;
      AA03 = AA73;
      AA73 = tmp;
      tmp = AA04;
      AA04 = AA74;
      AA74 = tmp;
      tmp = AA05;
      AA05 = AA75;
      AA75 = tmp;
      tmp = AA06;
      AA06 = AA76;
      AA76 = tmp;
      tmp = AA07;
      AA07 = AA77;
      AA77 = tmp;
      tmp = AA08;
      AA08 = AA78;
      AA78 = tmp;
      /* update permutation vector based on row swap */
      i = perm0;
      perm0 = perm7;
      perm7 = i;
    }
    if (pvt == 8) {
      tmp = AA00;
      AA00 = AA80;
      AA80 = tmp;
      tmp = AA01;
      AA01 = AA81;
      AA81 = tmp;
      tmp = AA02;
      AA02 = AA82;
      AA82 = tmp;
      tmp = AA03;
      AA03 = AA83;
      AA83 = tmp;
      tmp = AA04;
      AA04 = AA84;
      AA84 = tmp;
      tmp = AA05;
      AA05 = AA85;
      AA85 = tmp;
      tmp = AA06;
      AA06 = AA86;
      AA86 = tmp;
      tmp = AA07;
      AA07 = AA87;
      AA87 = tmp;
      tmp = AA08;
      AA08 = AA88;
      AA88 = tmp;
      /* update permutation vector based on row swap */
      i = perm0;
      perm0 = perm8;
      perm8 = i;
    }
#endif // USE_PIVOTING

    /* scale current row */
    tmp = rcpOp(AA00);
    icol0 = perm0;
    AA00 = tmp;
    AA01 = mulOp(tmp, AA01);
    AA02 = mulOp(tmp, AA02);
    AA03 = mulOp(tmp, AA03);
    AA04 = mulOp(tmp, AA04);
    AA05 = mulOp(tmp, AA05);
    AA06 = mulOp(tmp, AA06);
    AA07 = mulOp(tmp, AA07);
    AA08 = mulOp(tmp, AA08);

    /* eliminate above and below current row */
    tmp = AA10;
    AA10 = mulOp(negOp(tmp), AA00);
    AA11 = fmnaOp(tmp, AA01, AA11);
    AA12 = fmnaOp(tmp, AA02, AA12);
    AA13 = fmnaOp(tmp, AA03, AA13);
    AA14 = fmnaOp(tmp, AA04, AA14);
    AA15 = fmnaOp(tmp, AA05, AA15);
    AA16 = fmnaOp(tmp, AA06, AA16);
    AA17 = fmnaOp(tmp, AA07, AA17);
    AA18 = fmnaOp(tmp, AA08, AA18);

    tmp = AA20;
    AA20 = mulOp(negOp(tmp), AA00);
    AA21 = fmnaOp(tmp, AA01, AA21);
    AA22 = fmnaOp(tmp, AA02, AA22);
    AA23 = fmnaOp(tmp, AA03, AA23);
    AA24 = fmnaOp(tmp, AA04, AA24);
    AA25 = fmnaOp(tmp, AA05, AA25);
    AA26 = fmnaOp(tmp, AA06, AA26);
    AA27 = fmnaOp(tmp, AA07, AA27);
    AA28 = fmnaOp(tmp, AA08, AA28);

    tmp = AA30;
    AA30 = mulOp(negOp(tmp), AA00);
    AA31 = fmnaOp(tmp, AA01, AA31);
    AA32 = fmnaOp(tmp, AA02, AA32);
    AA33 = fmnaOp(tmp, AA03, AA33);
    AA34 = fmnaOp(tmp, AA04, AA34);
    AA35 = fmnaOp(tmp, AA05, AA35);
    AA36 = fmnaOp(tmp, AA06, AA36);
    AA37 = fmnaOp(tmp, AA07, AA37);
    AA38 = fmnaOp(tmp, AA08, AA38);

    tmp = AA40;
    AA40 = mulOp(negOp(tmp), AA00);
    AA41 = fmnaOp(tmp, AA01, AA41);
    AA42 = fmnaOp(tmp, AA02, AA42);
    AA43 = fmnaOp(tmp, AA03, AA43);
    AA44 = fmnaOp(tmp, AA04, AA44);
    AA45 = fmnaOp(tmp, AA05, AA45);
    AA46 = fmnaOp(tmp, AA06, AA46);
    AA47 = fmnaOp(tmp, AA07, AA47);
    AA48 = fmnaOp(tmp, AA08, AA48);

    tmp = AA50;
    AA50 = mulOp(negOp(tmp), AA00);
    AA51 = fmnaOp(tmp, AA01, AA51);
    AA52 = fmnaOp(tmp, AA02, AA52);
    AA53 = fmnaOp(tmp, AA03, AA53);
    AA54 = fmnaOp(tmp, AA04, AA54);
    AA55 = fmnaOp(tmp, AA05, AA55);
    AA56 = fmnaOp(tmp, AA06, AA56);
    AA57 = fmnaOp(tmp, AA07, AA57);
    AA58 = fmnaOp(tmp, AA08, AA58);

    tmp = AA60;
    AA60 = mulOp(negOp(tmp), AA00);
    AA61 = fmnaOp(tmp, AA01, AA61);
    AA62 = fmnaOp(tmp, AA02, AA62);
    AA63 = fmnaOp(tmp, AA03, AA63);
    AA64 = fmnaOp(tmp, AA04, AA64);
    AA65 = fmnaOp(tmp, AA05, AA65);
    AA66 = fmnaOp(tmp, AA06, AA66);
    AA67 = fmnaOp(tmp, AA07, AA67);
    AA68 = fmnaOp(tmp, AA08, AA68);

    tmp = AA70;
    AA70 = mulOp(negOp(tmp), AA00);
    AA71 = fmnaOp(tmp, AA01, AA71);
    AA72 = fmnaOp(tmp, AA02, AA72);
    AA73 = fmnaOp(tmp, AA03, AA73);
    AA74 = fmnaOp(tmp, AA04, AA74);
    AA75 = fmnaOp(tmp, AA05, AA75);
    AA76 = fmnaOp(tmp, AA06, AA76);
    AA77 = fmnaOp(tmp, AA07, AA77);
    AA78 = fmnaOp(tmp, AA08, AA78);

    tmp = AA80;
    AA80 = mulOp(negOp(tmp), AA00);
    AA81 = fmnaOp(tmp, AA01, AA81);
    AA82 = fmnaOp(tmp, AA02, AA82);
    AA83 = fmnaOp(tmp, AA03, AA83);
    AA84 = fmnaOp(tmp, AA04, AA84);
    AA85 = fmnaOp(tmp, AA05, AA85);
    AA86 = fmnaOp(tmp, AA06, AA86);
    AA87 = fmnaOp(tmp, AA07, AA87);
    AA88 = fmnaOp(tmp, AA08, AA88);

    /****************** iteration 1 ***********/

#if USE_PIVOTING
    /* search pivot row */
    p = absOp(AA11);
    pvt = 1;
    t = absOp(AA21);
    if (t > p) {
      p = t;
      pvt = 2;
    }
    t = absOp(AA31);
    if (t > p) {
      p = t;
      pvt = 3;
    }
    t = absOp(AA41);
    if (t > p) {
      p = t;
      pvt = 4;
    }
    t = absOp(AA51);
    if (t > p) {
      p = t;
      pvt = 5;
    }
    t = absOp(AA61);
    if (t > p) {
      p = t;
      pvt = 6;
    }
    t = absOp(AA71);
    if (t > p) {
      p = t;
      pvt = 7;
    }
    t = absOp(AA81);
    if (t > p) {
      p = t;
      pvt = 8;
    }

    /* swap pivot row with row 1 */
    if (pvt == 2) {
      tmp = AA10;
      AA10 = AA20;
      AA20 = tmp;
      tmp = AA11;
      AA11 = AA21;
      AA21 = tmp;
      tmp = AA12;
      AA12 = AA22;
      AA22 = tmp;
      tmp = AA13;
      AA13 = AA23;
      AA23 = tmp;
      tmp = AA14;
      AA14 = AA24;
      AA24 = tmp;
      tmp = AA15;
      AA15 = AA25;
      AA25 = tmp;
      tmp = AA16;
      AA16 = AA26;
      AA26 = tmp;
      tmp = AA17;
      AA17 = AA27;
      AA27 = tmp;
      tmp = AA18;
      AA18 = AA28;
      AA28 = tmp;
      /* update permutation vector based on row swap */
      i = perm1;
      perm1 = perm2;
      perm2 = i;
    }
    if (pvt == 3) {
      tmp = AA10;
      AA10 = AA30;
      AA30 = tmp;
      tmp = AA11;
      AA11 = AA31;
      AA31 = tmp;
      tmp = AA12;
      AA12 = AA32;
      AA32 = tmp;
      tmp = AA13;
      AA13 = AA33;
      AA33 = tmp;
      tmp = AA14;
      AA14 = AA34;
      AA34 = tmp;
      tmp = AA15;
      AA15 = AA35;
      AA35 = tmp;
      tmp = AA16;
      AA16 = AA36;
      AA36 = tmp;
      tmp = AA17;
      AA17 = AA37;
      AA37 = tmp;
      tmp = AA18;
      AA18 = AA38;
      AA38 = tmp;
      /* update permutation vector based on row swap */
      i = perm1;
      perm1 = perm3;
      perm3 = i;
    }
    if (pvt == 4) {
      tmp = AA10;
      AA10 = AA40;
      AA40 = tmp;
      tmp = AA11;
      AA11 = AA41;
      AA41 = tmp;
      tmp = AA12;
      AA12 = AA42;
      AA42 = tmp;
      tmp = AA13;
      AA13 = AA43;
      AA43 = tmp;
      tmp = AA14;
      AA14 = AA44;
      AA44 = tmp;
      tmp = AA15;
      AA15 = AA45;
      AA45 = tmp;
      tmp = AA16;
      AA16 = AA46;
      AA46 = tmp;
      tmp = AA17;
      AA17 = AA47;
      AA47 = tmp;
      tmp = AA18;
      AA18 = AA48;
      AA48 = tmp;
      /* update permutation vector based on row swap */
      i = perm1;
      perm1 = perm4;
      perm4 = i;
    }
    if (pvt == 5) {
      tmp = AA10;
      AA10 = AA50;
      AA50 = tmp;
      tmp = AA11;
      AA11 = AA51;
      AA51 = tmp;
      tmp = AA12;
      AA12 = AA52;
      AA52 = tmp;
      tmp = AA13;
      AA13 = AA53;
      AA53 = tmp;
      tmp = AA14;
      AA14 = AA54;
      AA54 = tmp;
      tmp = AA15;
      AA15 = AA55;
      AA55 = tmp;
      tmp = AA16;
      AA16 = AA56;
      AA56 = tmp;
      tmp = AA17;
      AA17 = AA57;
      AA57 = tmp;
      tmp = AA18;
      AA18 = AA58;
      AA58 = tmp;
      /* update permutation vector based on row swap */
      i = perm1;
      perm1 = perm5;
      perm5 = i;
    }
    if (pvt == 6) {
      tmp = AA10;
      AA10 = AA60;
      AA60 = tmp;
      tmp = AA11;
      AA11 = AA61;
      AA61 = tmp;
      tmp = AA12;
      AA12 = AA62;
      AA62 = tmp;
      tmp = AA13;
      AA13 = AA63;
      AA63 = tmp;
      tmp = AA14;
      AA14 = AA64;
      AA64 = tmp;
      tmp = AA15;
      AA15 = AA65;
      AA65 = tmp;
      tmp = AA16;
      AA16 = AA66;
      AA66 = tmp;
      tmp = AA17;
      AA17 = AA67;
      AA67 = tmp;
      tmp = AA18;
      AA18 = AA68;
      AA68 = tmp;
      /* update permutation vector based on row swap */
      i = perm1;
      perm1 = perm6;
      perm6 = i;
    }
    if (pvt == 7) {
      tmp = AA10;
      AA10 = AA70;
      AA70 = tmp;
      tmp = AA11;
      AA11 = AA71;
      AA71 = tmp;
      tmp = AA12;
      AA12 = AA72;
      AA72 = tmp;
      tmp = AA13;
      AA13 = AA73;
      AA73 = tmp;
      tmp = AA14;
      AA14 = AA74;
      AA74 = tmp;
      tmp = AA15;
      AA15 = AA75;
      AA75 = tmp;
      tmp = AA16;
      AA16 = AA76;
      AA76 = tmp;
      tmp = AA17;
      AA17 = AA77;
      AA77 = tmp;
      tmp = AA18;
      AA18 = AA78;
      AA78 = tmp;
      /* update permutation vector based on row swap */
      i = perm1;
      perm1 = perm7;
      perm7 = i;
    }
    if (pvt == 8) {
      tmp = AA10;
      AA10 = AA80;
      AA80 = tmp;
      tmp = AA11;
      AA11 = AA81;
      AA81 = tmp;
      tmp = AA12;
      AA12 = AA82;
      AA82 = tmp;
      tmp = AA13;
      AA13 = AA83;
      AA83 = tmp;
      tmp = AA14;
      AA14 = AA84;
      AA84 = tmp;
      tmp = AA15;
      AA15 = AA85;
      AA85 = tmp;
      tmp = AA16;
      AA16 = AA86;
      AA86 = tmp;
      tmp = AA17;
      AA17 = AA87;
      AA87 = tmp;
      tmp = AA18;
      AA18 = AA88;
      AA88 = tmp;
      /* update permutation vector based on row swap */
      i = perm1;
      perm1 = perm8;
      perm8 = i;
    }
#endif // USE_PIVOTING

    /* scale current row */
    tmp = rcpOp(AA11);
    icol1 = perm1;
    AA10 = mulOp(tmp, AA10);
    AA11 = tmp;
    AA12 = mulOp(tmp, AA12);
    AA13 = mulOp(tmp, AA13);
    AA14 = mulOp(tmp, AA14);
    AA15 = mulOp(tmp, AA15);
    AA16 = mulOp(tmp, AA16);
    AA17 = mulOp(tmp, AA17);
    AA18 = mulOp(tmp, AA18);

    /* eliminate above and below current row */
    tmp = AA01;
    AA00 = fmnaOp(tmp, AA10, AA00);
    AA01 = mulOp(negOp(tmp), AA11);
    AA02 = fmnaOp(tmp, AA12, AA02);
    AA03 = fmnaOp(tmp, AA13, AA03);
    AA04 = fmnaOp(tmp, AA14, AA04);
    AA05 = fmnaOp(tmp, AA15, AA05);
    AA06 = fmnaOp(tmp, AA16, AA06);
    AA07 = fmnaOp(tmp, AA17, AA07);
    AA08 = fmnaOp(tmp, AA18, AA08);

    tmp = AA21;
    AA20 = fmnaOp(tmp, AA10, AA20);
    AA21 = mulOp(negOp(tmp), AA11);
    AA22 = fmnaOp(tmp, AA12, AA22);
    AA23 = fmnaOp(tmp, AA13, AA23);
    AA24 = fmnaOp(tmp, AA14, AA24);
    AA25 = fmnaOp(tmp, AA15, AA25);
    AA26 = fmnaOp(tmp, AA16, AA26);
    AA27 = fmnaOp(tmp, AA17, AA27);
    AA28 = fmnaOp(tmp, AA18, AA28);

    tmp = AA31;
    AA30 = fmnaOp(tmp, AA10, AA30);
    AA31 = mulOp(negOp(tmp), AA11);
    AA32 = fmnaOp(tmp, AA12, AA32);
    AA33 = fmnaOp(tmp, AA13, AA33);
    AA34 = fmnaOp(tmp, AA14, AA34);
    AA35 = fmnaOp(tmp, AA15, AA35);
    AA36 = fmnaOp(tmp, AA16, AA36);
    AA37 = fmnaOp(tmp, AA17, AA37);
    AA38 = fmnaOp(tmp, AA18, AA38);

    tmp = AA41;
    AA40 = fmnaOp(tmp, AA10, AA40);
    AA41 = mulOp(negOp(tmp), AA11);
    AA42 = fmnaOp(tmp, AA12, AA42);
    AA43 = fmnaOp(tmp, AA13, AA43);
    AA44 = fmnaOp(tmp, AA14, AA44);
    AA45 = fmnaOp(tmp, AA15, AA45);
    AA46 = fmnaOp(tmp, AA16, AA46);
    AA47 = fmnaOp(tmp, AA17, AA47);
    AA48 = fmnaOp(tmp, AA18, AA48);

    tmp = AA51;
    AA50 = fmnaOp(tmp, AA10, AA50);
    AA51 = mulOp(negOp(tmp), AA11);
    AA52 = fmnaOp(tmp, AA12, AA52);
    AA53 = fmnaOp(tmp, AA13, AA53);
    AA54 = fmnaOp(tmp, AA14, AA54);
    AA55 = fmnaOp(tmp, AA15, AA55);
    AA56 = fmnaOp(tmp, AA16, AA56);
    AA57 = fmnaOp(tmp, AA17, AA57);
    AA58 = fmnaOp(tmp, AA18, AA58);

    tmp = AA61;
    AA60 = fmnaOp(tmp, AA10, AA60);
    AA61 = mulOp(negOp(tmp), AA11);
    AA62 = fmnaOp(tmp, AA12, AA62);
    AA63 = fmnaOp(tmp, AA13, AA63);
    AA64 = fmnaOp(tmp, AA14, AA64);
    AA65 = fmnaOp(tmp, AA15, AA65);
    AA66 = fmnaOp(tmp, AA16, AA66);
    AA67 = fmnaOp(tmp, AA17, AA67);
    AA68 = fmnaOp(tmp, AA18, AA68);

    tmp = AA71;
    AA70 = fmnaOp(tmp, AA10, AA70);
    AA71 = mulOp(negOp(tmp), AA11);
    AA72 = fmnaOp(tmp, AA12, AA72);
    AA73 = fmnaOp(tmp, AA13, AA73);
    AA74 = fmnaOp(tmp, AA14, AA74);
    AA75 = fmnaOp(tmp, AA15, AA75);
    AA76 = fmnaOp(tmp, AA16, AA76);
    AA77 = fmnaOp(tmp, AA17, AA77);
    AA78 = fmnaOp(tmp, AA18, AA78);

    tmp = AA81;
    AA80 = fmnaOp(tmp, AA10, AA80);
    AA81 = mulOp(negOp(tmp), AA11);
    AA82 = fmnaOp(tmp, AA12, AA82);
    AA83 = fmnaOp(tmp, AA13, AA83);
    AA84 = fmnaOp(tmp, AA14, AA84);
    AA85 = fmnaOp(tmp, AA15, AA85);
    AA86 = fmnaOp(tmp, AA16, AA86);
    AA87 = fmnaOp(tmp, AA17, AA87);
    AA88 = fmnaOp(tmp, AA18, AA88);

    /****************** iteration 2 ****************/

#if USE_PIVOTING
    /* search pivot row */
    p = absOp(AA22);
    pvt = 2;
    t = absOp(AA32);
    if (t > p) {
      p = t;
      pvt = 3;
    }
    t = absOp(AA42);
    if (t > p) {
      p = t;
      pvt = 4;
    }
    t = absOp(AA52);
    if (t > p) {
      p = t;
      pvt = 5;
    }
    t = absOp(AA62);
    if (t > p) {
      p = t;
      pvt = 6;
    }
    t = absOp(AA72);
    if (t > p) {
      p = t;
      pvt = 7;
    }
    t = absOp(AA82);
    if (t > p) {
      p = t;
      pvt = 8;
    }

    /* swap pivot row with row 2 */
    if (pvt == 3) {
      tmp = AA20;
      AA20 = AA30;
      AA30 = tmp;
      tmp = AA21;
      AA21 = AA31;
      AA31 = tmp;
      tmp = AA22;
      AA22 = AA32;
      AA32 = tmp;
      tmp = AA23;
      AA23 = AA33;
      AA33 = tmp;
      tmp = AA24;
      AA24 = AA34;
      AA34 = tmp;
      tmp = AA25;
      AA25 = AA35;
      AA35 = tmp;
      tmp = AA26;
      AA26 = AA36;
      AA36 = tmp;
      tmp = AA27;
      AA27 = AA37;
      AA37 = tmp;
      tmp = AA28;
      AA28 = AA38;
      AA38 = tmp;
      /* update permutation vector based on row swap */
      i = perm2;
      perm2 = perm3;
      perm3 = i;
    }
    if (pvt == 4) {
      tmp = AA20;
      AA20 = AA40;
      AA40 = tmp;
      tmp = AA21;
      AA21 = AA41;
      AA41 = tmp;
      tmp = AA22;
      AA22 = AA42;
      AA42 = tmp;
      tmp = AA23;
      AA23 = AA43;
      AA43 = tmp;
      tmp = AA24;
      AA24 = AA44;
      AA44 = tmp;
      tmp = AA25;
      AA25 = AA45;
      AA45 = tmp;
      tmp = AA26;
      AA26 = AA46;
      AA46 = tmp;
      tmp = AA27;
      AA27 = AA47;
      AA47 = tmp;
      tmp = AA28;
      AA28 = AA48;
      AA48 = tmp;
      /* update permutation vector based on row swap */
      i = perm2;
      perm2 = perm4;
      perm4 = i;
    }
    if (pvt == 5) {
      tmp = AA20;
      AA20 = AA50;
      AA50 = tmp;
      tmp = AA21;
      AA21 = AA51;
      AA51 = tmp;
      tmp = AA22;
      AA22 = AA52;
      AA52 = tmp;
      tmp = AA23;
      AA23 = AA53;
      AA53 = tmp;
      tmp = AA24;
      AA24 = AA54;
      AA54 = tmp;
      tmp = AA25;
      AA25 = AA55;
      AA55 = tmp;
      tmp = AA26;
      AA26 = AA56;
      AA56 = tmp;
      tmp = AA27;
      AA27 = AA57;
      AA57 = tmp;
      tmp = AA28;
      AA28 = AA58;
      AA58 = tmp;
      /* update permutation vector based on row swap */
      i = perm2;
      perm2 = perm5;
      perm5 = i;
    }
    if (pvt == 6) {
      tmp = AA20;
      AA20 = AA60;
      AA60 = tmp;
      tmp = AA21;
      AA21 = AA61;
      AA61 = tmp;
      tmp = AA22;
      AA22 = AA62;
      AA62 = tmp;
      tmp = AA23;
      AA23 = AA63;
      AA63 = tmp;
      tmp = AA24;
      AA24 = AA64;
      AA64 = tmp;
      tmp = AA25;
      AA25 = AA65;
      AA65 = tmp;
      tmp = AA26;
      AA26 = AA66;
      AA66 = tmp;
      tmp = AA27;
      AA27 = AA67;
      AA67 = tmp;
      tmp = AA28;
      AA28 = AA68;
      AA68 = tmp;
      /* update permutation vector based on row swap */
      i = perm2;
      perm2 = perm6;
      perm6 = i;
    }
    if (pvt == 7) {
      tmp = AA20;
      AA20 = AA70;
      AA70 = tmp;
      tmp = AA21;
      AA21 = AA71;
      AA71 = tmp;
      tmp = AA22;
      AA22 = AA72;
      AA72 = tmp;
      tmp = AA23;
      AA23 = AA73;
      AA73 = tmp;
      tmp = AA24;
      AA24 = AA74;
      AA74 = tmp;
      tmp = AA25;
      AA25 = AA75;
      AA75 = tmp;
      tmp = AA26;
      AA26 = AA76;
      AA76 = tmp;
      tmp = AA27;
      AA27 = AA77;
      AA77 = tmp;
      tmp = AA28;
      AA28 = AA78;
      AA78 = tmp;
      /* update permutation vector based on row swap */
      i = perm2;
      perm2 = perm7;
      perm7 = i;
    }
    if (pvt == 8) {
      tmp = AA20;
      AA20 = AA80;
      AA80 = tmp;
      tmp = AA21;
      AA21 = AA81;
      AA81 = tmp;
      tmp = AA22;
      AA22 = AA82;
      AA82 = tmp;
      tmp = AA23;
      AA23 = AA83;
      AA83 = tmp;
      tmp = AA24;
      AA24 = AA84;
      AA84 = tmp;
      tmp = AA25;
      AA25 = AA85;
      AA85 = tmp;
      tmp = AA26;
      AA26 = AA86;
      AA86 = tmp;
      tmp = AA27;
      AA27 = AA87;
      AA87 = tmp;
      tmp = AA28;
      AA28 = AA88;
      AA88 = tmp;
      /* update permutation vector based on row swap */
      i = perm2;
      perm2 = perm8;
      perm8 = i;
    }
#endif // USE_PIVOTING

    /* scale current row */
    tmp = rcpOp(AA22);
    icol2 = perm2;
    AA20 = mulOp(tmp, AA20);
    AA21 = mulOp(tmp, AA21);
    AA22 = tmp;
    AA23 = mulOp(tmp, AA23);
    AA24 = mulOp(tmp, AA24);
    AA25 = mulOp(tmp, AA25);
    AA26 = mulOp(tmp, AA26);
    AA27 = mulOp(tmp, AA27);
    AA28 = mulOp(tmp, AA28);

    /* eliminate above and below current row */
    tmp = AA02;
    AA00 = fmnaOp(tmp, AA20, AA00);
    AA01 = fmnaOp(tmp, AA21, AA01);
    AA02 = mulOp(negOp(tmp), AA22);
    AA03 = fmnaOp(tmp, AA23, AA03);
    AA04 = fmnaOp(tmp, AA24, AA04);
    AA05 = fmnaOp(tmp, AA25, AA05);
    AA06 = fmnaOp(tmp, AA26, AA06);
    AA07 = fmnaOp(tmp, AA27, AA07);
    AA08 = fmnaOp(tmp, AA28, AA08);

    tmp = AA12;
    AA10 = fmnaOp(tmp, AA20, AA10);
    AA11 = fmnaOp(tmp, AA21, AA11);
    AA12 = mulOp(negOp(tmp), AA22);
    AA13 = fmnaOp(tmp, AA23, AA13);
    AA14 = fmnaOp(tmp, AA24, AA14);
    AA15 = fmnaOp(tmp, AA25, AA15);
    AA16 = fmnaOp(tmp, AA26, AA16);
    AA17 = fmnaOp(tmp, AA27, AA17);
    AA18 = fmnaOp(tmp, AA28, AA18);

    tmp = AA32;
    AA30 = fmnaOp(tmp, AA20, AA30);
    AA31 = fmnaOp(tmp, AA21, AA31);
    AA32 = mulOp(negOp(tmp), AA22);
    AA33 = fmnaOp(tmp, AA23, AA33);
    AA34 = fmnaOp(tmp, AA24, AA34);
    AA35 = fmnaOp(tmp, AA25, AA35);
    AA36 = fmnaOp(tmp, AA26, AA36);
    AA37 = fmnaOp(tmp, AA27, AA37);
    AA38 = fmnaOp(tmp, AA28, AA38);

    tmp = AA42;
    AA40 = fmnaOp(tmp, AA20, AA40);
    AA41 = fmnaOp(tmp, AA21, AA41);
    AA42 = mulOp(negOp(tmp), AA22);
    AA43 = fmnaOp(tmp, AA23, AA43);
    AA44 = fmnaOp(tmp, AA24, AA44);
    AA45 = fmnaOp(tmp, AA25, AA45);
    AA46 = fmnaOp(tmp, AA26, AA46);
    AA47 = fmnaOp(tmp, AA27, AA47);
    AA48 = fmnaOp(tmp, AA28, AA48);

    tmp = AA52;
    AA50 = fmnaOp(tmp, AA20, AA50);
    AA51 = fmnaOp(tmp, AA21, AA51);
    AA52 = mulOp(negOp(tmp), AA22);
    AA53 = fmnaOp(tmp, AA23, AA53);
    AA54 = fmnaOp(tmp, AA24, AA54);
    AA55 = fmnaOp(tmp, AA25, AA55);
    AA56 = fmnaOp(tmp, AA26, AA56);
    AA57 = fmnaOp(tmp, AA27, AA57);
    AA58 = fmnaOp(tmp, AA28, AA58);

    tmp = AA62;
    AA60 = fmnaOp(tmp, AA20, AA60);
    AA61 = fmnaOp(tmp, AA21, AA61);
    AA62 = mulOp(negOp(tmp), AA22);
    AA63 = fmnaOp(tmp, AA23, AA63);
    AA64 = fmnaOp(tmp, AA24, AA64);
    AA65 = fmnaOp(tmp, AA25, AA65);
    AA66 = fmnaOp(tmp, AA26, AA66);
    AA67 = fmnaOp(tmp, AA27, AA67);
    AA68 = fmnaOp(tmp, AA28, AA68);

    tmp = AA72;
    AA70 = fmnaOp(tmp, AA20, AA70);
    AA71 = fmnaOp(tmp, AA21, AA71);
    AA72 = mulOp(negOp(tmp), AA22);
    AA73 = fmnaOp(tmp, AA23, AA73);
    AA74 = fmnaOp(tmp, AA24, AA74);
    AA75 = fmnaOp(tmp, AA25, AA75);
    AA76 = fmnaOp(tmp, AA26, AA76);
    AA77 = fmnaOp(tmp, AA27, AA77);
    AA78 = fmnaOp(tmp, AA28, AA78);

    tmp = AA82;
    AA80 = fmnaOp(tmp, AA20, AA80);
    AA81 = fmnaOp(tmp, AA21, AA81);
    AA82 = mulOp(negOp(tmp), AA22);
    AA83 = fmnaOp(tmp, AA23, AA83);
    AA84 = fmnaOp(tmp, AA24, AA84);
    AA85 = fmnaOp(tmp, AA25, AA85);
    AA86 = fmnaOp(tmp, AA26, AA86);
    AA87 = fmnaOp(tmp, AA27, AA87);
    AA88 = fmnaOp(tmp, AA28, AA88);

    /****************** iteration 3 ****************/

#if USE_PIVOTING
    /* search pivot row */
    p = absOp(AA33);
    pvt = 3;
    t = absOp(AA43);
    if (t > p) {
      p = t;
      pvt = 4;
    }
    t = absOp(AA53);
    if (t > p) {
      p = t;
      pvt = 5;
    }
    t = absOp(AA63);
    if (t > p) {
      p = t;
      pvt = 6;
    }
    t = absOp(AA73);
    if (t > p) {
      p = t;
      pvt = 7;
    }
    t = absOp(AA83);
    if (t > p) {
      p = t;
      pvt = 8;
    }

    /* swap pivot row with row 3 */
    if (pvt == 4) {
      tmp = AA30;
      AA30 = AA40;
      AA40 = tmp;
      tmp = AA31;
      AA31 = AA41;
      AA41 = tmp;
      tmp = AA32;
      AA32 = AA42;
      AA42 = tmp;
      tmp = AA33;
      AA33 = AA43;
      AA43 = tmp;
      tmp = AA34;
      AA34 = AA44;
      AA44 = tmp;
      tmp = AA35;
      AA35 = AA45;
      AA45 = tmp;
      tmp = AA36;
      AA36 = AA46;
      AA46 = tmp;
      tmp = AA37;
      AA37 = AA47;
      AA47 = tmp;
      tmp = AA38;
      AA38 = AA48;
      AA48 = tmp;
      /* update permutation vector based on row swap */
      i = perm3;
      perm3 = perm4;
      perm4 = i;
    }
    if (pvt == 5) {
      tmp = AA30;
      AA30 = AA50;
      AA50 = tmp;
      tmp = AA31;
      AA31 = AA51;
      AA51 = tmp;
      tmp = AA32;
      AA32 = AA52;
      AA52 = tmp;
      tmp = AA33;
      AA33 = AA53;
      AA53 = tmp;
      tmp = AA34;
      AA34 = AA54;
      AA54 = tmp;
      tmp = AA35;
      AA35 = AA55;
      AA55 = tmp;
      tmp = AA36;
      AA36 = AA56;
      AA56 = tmp;
      tmp = AA37;
      AA37 = AA57;
      AA57 = tmp;
      tmp = AA38;
      AA38 = AA58;
      AA58 = tmp;
      /* update permutation vector based on row swap */
      i = perm3;
      perm3 = perm5;
      perm5 = i;
    }
    if (pvt == 6) {
      tmp = AA30;
      AA30 = AA60;
      AA60 = tmp;
      tmp = AA31;
      AA31 = AA61;
      AA61 = tmp;
      tmp = AA32;
      AA32 = AA62;
      AA62 = tmp;
      tmp = AA33;
      AA33 = AA63;
      AA63 = tmp;
      tmp = AA34;
      AA34 = AA64;
      AA64 = tmp;
      tmp = AA35;
      AA35 = AA65;
      AA65 = tmp;
      tmp = AA36;
      AA36 = AA66;
      AA66 = tmp;
      tmp = AA37;
      AA37 = AA67;
      AA67 = tmp;
      tmp = AA38;
      AA38 = AA68;
      AA68 = tmp;
      /* update permutation vector based on row swap */
      i = perm3;
      perm3 = perm6;
      perm6 = i;
    }
    if (pvt == 7) {
      tmp = AA30;
      AA30 = AA70;
      AA70 = tmp;
      tmp = AA31;
      AA31 = AA71;
      AA71 = tmp;
      tmp = AA32;
      AA32 = AA72;
      AA72 = tmp;
      tmp = AA33;
      AA33 = AA73;
      AA73 = tmp;
      tmp = AA34;
      AA34 = AA74;
      AA74 = tmp;
      tmp = AA35;
      AA35 = AA75;
      AA75 = tmp;
      tmp = AA36;
      AA36 = AA76;
      AA76 = tmp;
      tmp = AA37;
      AA37 = AA77;
      AA77 = tmp;
      tmp = AA38;
      AA38 = AA78;
      AA78 = tmp;
      /* update permutation vector based on row swap */
      i = perm3;
      perm3 = perm7;
      perm7 = i;
    }
    if (pvt == 8) {
      tmp = AA30;
      AA30 = AA80;
      AA80 = tmp;
      tmp = AA31;
      AA31 = AA81;
      AA81 = tmp;
      tmp = AA32;
      AA32 = AA82;
      AA82 = tmp;
      tmp = AA33;
      AA33 = AA83;
      AA83 = tmp;
      tmp = AA34;
      AA34 = AA84;
      AA84 = tmp;
      tmp = AA35;
      AA35 = AA85;
      AA85 = tmp;
      tmp = AA36;
      AA36 = AA86;
      AA86 = tmp;
      tmp = AA37;
      AA37 = AA87;
      AA87 = tmp;
      tmp = AA38;
      AA38 = AA88;
      AA88 = tmp;
      /* update permutation vector based on row swap */
      i = perm3;
      perm3 = perm8;
      perm8 = i;
    }
#endif // USE_PIVOTING

    /* scale current row */
    tmp = rcpOp(AA33);
    icol3 = perm3;
    AA30 = mulOp(tmp, AA30);
    AA31 = mulOp(tmp, AA31);
    AA32 = mulOp(tmp, AA32);
    AA33 = tmp;
    AA34 = mulOp(tmp, AA34);
    AA35 = mulOp(tmp, AA35);
    AA36 = mulOp(tmp, AA36);
    AA37 = mulOp(tmp, AA37);
    AA38 = mulOp(tmp, AA38);

    /* eliminate above and below current row */
    tmp = AA03;
    AA00 = fmnaOp(tmp, AA30, AA00);
    AA01 = fmnaOp(tmp, AA31, AA01);
    AA02 = fmnaOp(tmp, AA32, AA02);
    AA03 = mulOp(negOp(tmp), AA33);
    AA04 = fmnaOp(tmp, AA34, AA04);
    AA05 = fmnaOp(tmp, AA35, AA05);
    AA06 = fmnaOp(tmp, AA36, AA06);
    AA07 = fmnaOp(tmp, AA37, AA07);
    AA08 = fmnaOp(tmp, AA38, AA08);

    tmp = AA13;
    AA10 = fmnaOp(tmp, AA30, AA10);
    AA11 = fmnaOp(tmp, AA31, AA11);
    AA12 = fmnaOp(tmp, AA32, AA12);
    AA13 = mulOp(negOp(tmp), AA33);
    AA14 = fmnaOp(tmp, AA34, AA14);
    AA15 = fmnaOp(tmp, AA35, AA15);
    AA16 = fmnaOp(tmp, AA36, AA16);
    AA17 = fmnaOp(tmp, AA37, AA17);
    AA18 = fmnaOp(tmp, AA38, AA18);

    tmp = AA23;
    AA20 = fmnaOp(tmp, AA30, AA20);
    AA21 = fmnaOp(tmp, AA31, AA21);
    AA22 = fmnaOp(tmp, AA32, AA22);
    AA23 = mulOp(negOp(tmp), AA33);
    AA24 = fmnaOp(tmp, AA34, AA24);
    AA25 = fmnaOp(tmp, AA35, AA25);
    AA26 = fmnaOp(tmp, AA36, AA26);
    AA27 = fmnaOp(tmp, AA37, AA27);
    AA28 = fmnaOp(tmp, AA38, AA28);

    tmp = AA43;
    AA40 = fmnaOp(tmp, AA30, AA40);
    AA41 = fmnaOp(tmp, AA31, AA41);
    AA42 = fmnaOp(tmp, AA32, AA42);
    AA43 = mulOp(negOp(tmp), AA33);
    AA44 = fmnaOp(tmp, AA34, AA44);
    AA45 = fmnaOp(tmp, AA35, AA45);
    AA46 = fmnaOp(tmp, AA36, AA46);
    AA47 = fmnaOp(tmp, AA37, AA47);
    AA48 = fmnaOp(tmp, AA38, AA48);

    tmp = AA53;
    AA50 = fmnaOp(tmp, AA30, AA50);
    AA51 = fmnaOp(tmp, AA31, AA51);
    AA52 = fmnaOp(tmp, AA32, AA52);
    AA53 = mulOp(negOp(tmp), AA33);
    AA54 = fmnaOp(tmp, AA34, AA54);
    AA55 = fmnaOp(tmp, AA35, AA55);
    AA56 = fmnaOp(tmp, AA36, AA56);
    AA57 = fmnaOp(tmp, AA37, AA57);
    AA58 = fmnaOp(tmp, AA38, AA58);

    tmp = AA63;
    AA60 = fmnaOp(tmp, AA30, AA60);
    AA61 = fmnaOp(tmp, AA31, AA61);
    AA62 = fmnaOp(tmp, AA32, AA62);
    AA63 = mulOp(negOp(tmp), AA33);
    AA64 = fmnaOp(tmp, AA34, AA64);
    AA65 = fmnaOp(tmp, AA35, AA65);
    AA66 = fmnaOp(tmp, AA36, AA66);
    AA67 = fmnaOp(tmp, AA37, AA67);
    AA68 = fmnaOp(tmp, AA38, AA68);

    tmp = AA73;
    AA70 = fmnaOp(tmp, AA30, AA70);
    AA71 = fmnaOp(tmp, AA31, AA71);
    AA72 = fmnaOp(tmp, AA32, AA72);
    AA73 = mulOp(negOp(tmp), AA33);
    AA74 = fmnaOp(tmp, AA34, AA74);
    AA75 = fmnaOp(tmp, AA35, AA75);
    AA76 = fmnaOp(tmp, AA36, AA76);
    AA77 = fmnaOp(tmp, AA37, AA77);
    AA78 = fmnaOp(tmp, AA38, AA78);

    tmp = AA83;
    AA80 = fmnaOp(tmp, AA30, AA80);
    AA81 = fmnaOp(tmp, AA31, AA81);
    AA82 = fmnaOp(tmp, AA32, AA82);
    AA83 = mulOp(negOp(tmp), AA33);
    AA84 = fmnaOp(tmp, AA34, AA84);
    AA85 = fmnaOp(tmp, AA35, AA85);
    AA86 = fmnaOp(tmp, AA36, AA86);
    AA87 = fmnaOp(tmp, AA37, AA87);
    AA88 = fmnaOp(tmp, AA38, AA88);

    /****************** iteration 4 ****************/

#if USE_PIVOTING
    /* search pivot row */
    p = absOp(AA44);
    pvt = 4;
    t = absOp(AA54);
    if (t > p) {
      p = t;
      pvt = 5;
    }
    t = absOp(AA64);
    if (t > p) {
      p = t;
      pvt = 6;
    }
    t = absOp(AA74);
    if (t > p) {
      p = t;
      pvt = 7;
    }
    t = absOp(AA84);
    if (t > p) {
      p = t;
      pvt = 8;
    }

    /* swap pivot row with row 4 */
    if (pvt == 5) {
      tmp = AA40;
      AA40 = AA50;
      AA50 = tmp;
      tmp = AA41;
      AA41 = AA51;
      AA51 = tmp;
      tmp = AA42;
      AA42 = AA52;
      AA52 = tmp;
      tmp = AA43;
      AA43 = AA53;
      AA53 = tmp;
      tmp = AA44;
      AA44 = AA54;
      AA54 = tmp;
      tmp = AA45;
      AA45 = AA55;
      AA55 = tmp;
      tmp = AA46;
      AA46 = AA56;
      AA56 = tmp;
      tmp = AA47;
      AA47 = AA57;
      AA57 = tmp;
      tmp = AA48;
      AA48 = AA58;
      AA58 = tmp;
      /* update permutation vector based on row swap */
      i = perm4;
      perm4 = perm5;
      perm5 = i;
    }
    if (pvt == 6) {
      tmp = AA40;
      AA40 = AA60;
      AA60 = tmp;
      tmp = AA41;
      AA41 = AA61;
      AA61 = tmp;
      tmp = AA42;
      AA42 = AA62;
      AA62 = tmp;
      tmp = AA43;
      AA43 = AA63;
      AA63 = tmp;
      tmp = AA44;
      AA44 = AA64;
      AA64 = tmp;
      tmp = AA45;
      AA45 = AA65;
      AA65 = tmp;
      tmp = AA46;
      AA46 = AA66;
      AA66 = tmp;
      tmp = AA47;
      AA47 = AA67;
      AA67 = tmp;
      tmp = AA48;
      AA48 = AA68;
      AA68 = tmp;
      /* update permutation vector based on row swap */
      i = perm4;
      perm4 = perm6;
      perm6 = i;
    }
    if (pvt == 7) {
      tmp = AA40;
      AA40 = AA70;
      AA70 = tmp;
      tmp = AA41;
      AA41 = AA71;
      AA71 = tmp;
      tmp = AA42;
      AA42 = AA72;
      AA72 = tmp;
      tmp = AA43;
      AA43 = AA73;
      AA73 = tmp;
      tmp = AA44;
      AA44 = AA74;
      AA74 = tmp;
      tmp = AA45;
      AA45 = AA75;
      AA75 = tmp;
      tmp = AA46;
      AA46 = AA76;
      AA76 = tmp;
      tmp = AA47;
      AA47 = AA77;
      AA77 = tmp;
      tmp = AA48;
      AA48 = AA78;
      AA78 = tmp;
      /* update permutation vector based on row swap */
      i = perm4;
      perm4 = perm7;
      perm7 = i;
    }
    if (pvt == 8) {
      tmp = AA40;
      AA40 = AA80;
      AA80 = tmp;
      tmp = AA41;
      AA41 = AA81;
      AA81 = tmp;
      tmp = AA42;
      AA42 = AA82;
      AA82 = tmp;
      tmp = AA43;
      AA43 = AA83;
      AA83 = tmp;
      tmp = AA44;
      AA44 = AA84;
      AA84 = tmp;
      tmp = AA45;
      AA45 = AA85;
      AA85 = tmp;
      tmp = AA46;
      AA46 = AA86;
      AA86 = tmp;
      tmp = AA47;
      AA47 = AA87;
      AA87 = tmp;
      tmp = AA48;
      AA48 = AA88;
      AA88 = tmp;
      /* update permutation vector based on row swap */
      i = perm4;
      perm4 = perm8;
      perm8 = i;
    }
#endif // USE_PIVOTING

    /* scale current row */
    tmp = rcpOp(AA44);
    icol4 = perm4;
    AA40 = mulOp(tmp, AA40);
    AA41 = mulOp(tmp, AA41);
    AA42 = mulOp(tmp, AA42);
    AA43 = mulOp(tmp, AA43);
    AA44 = tmp;
    AA45 = mulOp(tmp, AA45);
    AA46 = mulOp(tmp, AA46);
    AA47 = mulOp(tmp, AA47);
    AA48 = mulOp(tmp, AA48);

    /* eliminate above and below current row */
    tmp = AA04;
    AA00 = fmnaOp(tmp, AA40, AA00);
    AA01 = fmnaOp(tmp, AA41, AA01);
    AA02 = fmnaOp(tmp, AA42, AA02);
    AA03 = fmnaOp(tmp, AA43, AA03);
    AA04 = mulOp(negOp(tmp), AA44);
    AA05 = fmnaOp(tmp, AA45, AA05);
    AA06 = fmnaOp(tmp, AA46, AA06);
    AA07 = fmnaOp(tmp, AA47, AA07);
    AA08 = fmnaOp(tmp, AA48, AA08);

    tmp = AA14;
    AA10 = fmnaOp(tmp, AA40, AA10);
    AA11 = fmnaOp(tmp, AA41, AA11);
    AA12 = fmnaOp(tmp, AA42, AA12);
    AA13 = fmnaOp(tmp, AA43, AA13);
    AA14 = mulOp(negOp(tmp), AA44);
    AA15 = fmnaOp(tmp, AA45, AA15);
    AA16 = fmnaOp(tmp, AA46, AA16);
    AA17 = fmnaOp(tmp, AA47, AA17);
    AA18 = fmnaOp(tmp, AA48, AA18);

    tmp = AA24;
    AA20 = fmnaOp(tmp, AA40, AA20);
    AA21 = fmnaOp(tmp, AA41, AA21);
    AA22 = fmnaOp(tmp, AA42, AA22);
    AA23 = fmnaOp(tmp, AA43, AA23);
    AA24 = mulOp(negOp(tmp), AA44);
    AA25 = fmnaOp(tmp, AA45, AA25);
    AA26 = fmnaOp(tmp, AA46, AA26);
    AA27 = fmnaOp(tmp, AA47, AA27);
    AA28 = fmnaOp(tmp, AA48, AA28);

    tmp = AA34;
    AA30 = fmnaOp(tmp, AA40, AA30);
    AA31 = fmnaOp(tmp, AA41, AA31);
    AA32 = fmnaOp(tmp, AA42, AA32);
    AA33 = fmnaOp(tmp, AA43, AA33);
    AA34 = mulOp(negOp(tmp), AA44);
    AA35 = fmnaOp(tmp, AA45, AA35);
    AA36 = fmnaOp(tmp, AA46, AA36);
    AA37 = fmnaOp(tmp, AA47, AA37);
    AA38 = fmnaOp(tmp, AA48, AA38);

    tmp = AA54;
    AA50 = fmnaOp(tmp, AA40, AA50);
    AA51 = fmnaOp(tmp, AA41, AA51);
    AA52 = fmnaOp(tmp, AA42, AA52);
    AA53 = fmnaOp(tmp, AA43, AA53);
    AA54 = mulOp(negOp(tmp), AA44);
    AA55 = fmnaOp(tmp, AA45, AA55);
    AA56 = fmnaOp(tmp, AA46, AA56);
    AA57 = fmnaOp(tmp, AA47, AA57);
    AA58 = fmnaOp(tmp, AA48, AA58);

    tmp = AA64;
    AA60 = fmnaOp(tmp, AA40, AA60);
    AA61 = fmnaOp(tmp, AA41, AA61);
    AA62 = fmnaOp(tmp, AA42, AA62);
    AA63 = fmnaOp(tmp, AA43, AA63);
    AA64 = mulOp(negOp(tmp), AA44);
    AA65 = fmnaOp(tmp, AA45, AA65);
    AA66 = fmnaOp(tmp, AA46, AA66);
    AA67 = fmnaOp(tmp, AA47, AA67);
    AA68 = fmnaOp(tmp, AA48, AA68);

    tmp = AA74;
    AA70 = fmnaOp(tmp, AA40, AA70);
    AA71 = fmnaOp(tmp, AA41, AA71);
    AA72 = fmnaOp(tmp, AA42, AA72);
    AA73 = fmnaOp(tmp, AA43, AA73);
    AA74 = mulOp(negOp(tmp), AA44);
    AA75 = fmnaOp(tmp, AA45, AA75);
    AA76 = fmnaOp(tmp, AA46, AA76);
    AA77 = fmnaOp(tmp, AA47, AA77);
    AA78 = fmnaOp(tmp, AA48, AA78);

    tmp = AA84;
    AA80 = fmnaOp(tmp, AA40, AA80);
    AA81 = fmnaOp(tmp, AA41, AA81);
    AA82 = fmnaOp(tmp, AA42, AA82);
    AA83 = fmnaOp(tmp, AA43, AA83);
    AA84 = mulOp(negOp(tmp), AA44);
    AA85 = fmnaOp(tmp, AA45, AA85);
    AA86 = fmnaOp(tmp, AA46, AA86);
    AA87 = fmnaOp(tmp, AA47, AA87);
    AA88 = fmnaOp(tmp, AA48, AA88);

    /****************** iteration 5 ****************/

#if USE_PIVOTING
    /* search pivot row */
    p = absOp(AA55);
    pvt = 5;
    t = absOp(AA65);
    if (t > p) {
      p = t;
      pvt = 6;
    }
    t = absOp(AA75);
    if (t > p) {
      p = t;
      pvt = 7;
    }
    t = absOp(AA85);
    if (t > p) {
      p = t;
      pvt = 8;
    }

    /* swap pivot row with row 5 */
    if (pvt == 6) {
      tmp = AA50;
      AA50 = AA60;
      AA60 = tmp;
      tmp = AA51;
      AA51 = AA61;
      AA61 = tmp;
      tmp = AA52;
      AA52 = AA62;
      AA62 = tmp;
      tmp = AA53;
      AA53 = AA63;
      AA63 = tmp;
      tmp = AA54;
      AA54 = AA64;
      AA64 = tmp;
      tmp = AA55;
      AA55 = AA65;
      AA65 = tmp;
      tmp = AA56;
      AA56 = AA66;
      AA66 = tmp;
      tmp = AA57;
      AA57 = AA67;
      AA67 = tmp;
      tmp = AA58;
      AA58 = AA68;
      AA68 = tmp;
      /* update permutation vector based on row swap */
      i = perm5;
      perm5 = perm6;
      perm6 = i;
    }
    if (pvt == 7) {
      tmp = AA50;
      AA50 = AA70;
      AA70 = tmp;
      tmp = AA51;
      AA51 = AA71;
      AA71 = tmp;
      tmp = AA52;
      AA52 = AA72;
      AA72 = tmp;
      tmp = AA53;
      AA53 = AA73;
      AA73 = tmp;
      tmp = AA54;
      AA54 = AA74;
      AA74 = tmp;
      tmp = AA55;
      AA55 = AA75;
      AA75 = tmp;
      tmp = AA56;
      AA56 = AA76;
      AA76 = tmp;
      tmp = AA57;
      AA57 = AA77;
      AA77 = tmp;
      tmp = AA58;
      AA58 = AA78;
      AA78 = tmp;
      /* update permutation vector based on row swap */
      i = perm5;
      perm5 = perm7;
      perm7 = i;
    }
    if (pvt == 8) {
      tmp = AA50;
      AA50 = AA80;
      AA80 = tmp;
      tmp = AA51;
      AA51 = AA81;
      AA81 = tmp;
      tmp = AA52;
      AA52 = AA82;
      AA82 = tmp;
      tmp = AA53;
      AA53 = AA83;
      AA83 = tmp;
      tmp = AA54;
      AA54 = AA84;
      AA84 = tmp;
      tmp = AA55;
      AA55 = AA85;
      AA85 = tmp;
      tmp = AA56;
      AA56 = AA86;
      AA86 = tmp;
      tmp = AA57;
      AA57 = AA87;
      AA87 = tmp;
      tmp = AA58;
      AA58 = AA88;
      AA88 = tmp;
      /* update permutation vector based on row swap */
      i = perm5;
      perm5 = perm8;
      perm8 = i;
    }
#endif // USE_PIVOTING

    /* scale current row */
    tmp = rcpOp(AA55);
    icol5 = perm5;
    AA50 = mulOp(tmp, AA50);
    AA51 = mulOp(tmp, AA51);
    AA52 = mulOp(tmp, AA52);
    AA53 = mulOp(tmp, AA53);
    AA54 = mulOp(tmp, AA54);
    AA55 = tmp;
    AA56 = mulOp(tmp, AA56);
    AA57 = mulOp(tmp, AA57);
    AA58 = mulOp(tmp, AA58);

    /* eliminate above and below current row */
    tmp = AA05;
    AA00 = fmnaOp(tmp, AA50, AA00);
    AA01 = fmnaOp(tmp, AA51, AA01);
    AA02 = fmnaOp(tmp, AA52, AA02);
    AA03 = fmnaOp(tmp, AA53, AA03);
    AA04 = fmnaOp(tmp, AA54, AA04);
    AA05 = mulOp(negOp(tmp), AA55);
    AA06 = fmnaOp(tmp, AA56, AA06);
    AA07 = fmnaOp(tmp, AA57, AA07);
    AA08 = fmnaOp(tmp, AA58, AA08);

    tmp = AA15;
    AA10 = fmnaOp(tmp, AA50, AA10);
    AA11 = fmnaOp(tmp, AA51, AA11);
    AA12 = fmnaOp(tmp, AA52, AA12);
    AA13 = fmnaOp(tmp, AA53, AA13);
    AA14 = fmnaOp(tmp, AA54, AA14);
    AA15 = mulOp(negOp(tmp), AA55);
    AA16 = fmnaOp(tmp, AA56, AA16);
    AA17 = fmnaOp(tmp, AA57, AA17);
    AA18 = fmnaOp(tmp, AA58, AA18);

    tmp = AA25;
    AA20 = fmnaOp(tmp, AA50, AA20);
    AA21 = fmnaOp(tmp, AA51, AA21);
    AA22 = fmnaOp(tmp, AA52, AA22);
    AA23 = fmnaOp(tmp, AA53, AA23);
    AA24 = fmnaOp(tmp, AA54, AA24);
    AA25 = mulOp(negOp(tmp), AA55);
    AA26 = fmnaOp(tmp, AA56, AA26);
    AA27 = fmnaOp(tmp, AA57, AA27);
    AA28 = fmnaOp(tmp, AA58, AA28);

    tmp = AA35;
    AA30 = fmnaOp(tmp, AA50, AA30);
    AA31 = fmnaOp(tmp, AA51, AA31);
    AA32 = fmnaOp(tmp, AA52, AA32);
    AA33 = fmnaOp(tmp, AA53, AA33);
    AA34 = fmnaOp(tmp, AA54, AA34);
    AA35 = mulOp(negOp(tmp), AA55);
    AA36 = fmnaOp(tmp, AA56, AA36);
    AA37 = fmnaOp(tmp, AA57, AA37);
    AA38 = fmnaOp(tmp, AA58, AA38);

    tmp = AA45;
    AA40 = fmnaOp(tmp, AA50, AA40);
    AA41 = fmnaOp(tmp, AA51, AA41);
    AA42 = fmnaOp(tmp, AA52, AA42);
    AA43 = fmnaOp(tmp, AA53, AA43);
    AA44 = fmnaOp(tmp, AA54, AA44);
    AA45 = mulOp(negOp(tmp), AA55);
    AA46 = fmnaOp(tmp, AA56, AA46);
    AA47 = fmnaOp(tmp, AA57, AA47);
    AA48 = fmnaOp(tmp, AA58, AA48);

    tmp = AA65;
    AA60 = fmnaOp(tmp, AA50, AA60);
    AA61 = fmnaOp(tmp, AA51, AA61);
    AA62 = fmnaOp(tmp, AA52, AA62);
    AA63 = fmnaOp(tmp, AA53, AA63);
    AA64 = fmnaOp(tmp, AA54, AA64);
    AA65 = mulOp(negOp(tmp), AA55);
    AA66 = fmnaOp(tmp, AA56, AA66);
    AA67 = fmnaOp(tmp, AA57, AA67);
    AA68 = fmnaOp(tmp, AA58, AA68);

    tmp = AA75;
    AA70 = fmnaOp(tmp, AA50, AA70);
    AA71 = fmnaOp(tmp, AA51, AA71);
    AA72 = fmnaOp(tmp, AA52, AA72);
    AA73 = fmnaOp(tmp, AA53, AA73);
    AA74 = fmnaOp(tmp, AA54, AA74);
    AA75 = mulOp(negOp(tmp), AA55);
    AA76 = fmnaOp(tmp, AA56, AA76);
    AA77 = fmnaOp(tmp, AA57, AA77);
    AA78 = fmnaOp(tmp, AA58, AA78);

    tmp = AA85;
    AA80 = fmnaOp(tmp, AA50, AA80);
    AA81 = fmnaOp(tmp, AA51, AA81);
    AA82 = fmnaOp(tmp, AA52, AA82);
    AA83 = fmnaOp(tmp, AA53, AA83);
    AA84 = fmnaOp(tmp, AA54, AA84);
    AA85 = mulOp(negOp(tmp), AA55);
    AA86 = fmnaOp(tmp, AA56, AA86);
    AA87 = fmnaOp(tmp, AA57, AA87);
    AA88 = fmnaOp(tmp, AA58, AA88);

    /****************** iteration 6 ****************/

#if USE_PIVOTING
    /* search pivot row */
    p = absOp(AA66);
    pvt = 6;
    t = absOp(AA76);
    if (t > p) {
      p = t;
      pvt = 7;
    }
    t = absOp(AA86);
    if (t > p) {
      p = t;
      pvt = 8;
    }

    /* swap pivot row with row 6 */
    if (pvt == 7) {
      tmp = AA60;
      AA60 = AA70;
      AA70 = tmp;
      tmp = AA61;
      AA61 = AA71;
      AA71 = tmp;
      tmp = AA62;
      AA62 = AA72;
      AA72 = tmp;
      tmp = AA63;
      AA63 = AA73;
      AA73 = tmp;
      tmp = AA64;
      AA64 = AA74;
      AA74 = tmp;
      tmp = AA65;
      AA65 = AA75;
      AA75 = tmp;
      tmp = AA66;
      AA66 = AA76;
      AA76 = tmp;
      tmp = AA67;
      AA67 = AA77;
      AA77 = tmp;
      tmp = AA68;
      AA68 = AA78;
      AA78 = tmp;
      /* update permutation vector based on row swap */
      i = perm6;
      perm6 = perm7;
      perm7 = i;
    }
    if (pvt == 8) {
      tmp = AA60;
      AA60 = AA80;
      AA80 = tmp;
      tmp = AA61;
      AA61 = AA81;
      AA81 = tmp;
      tmp = AA62;
      AA62 = AA82;
      AA82 = tmp;
      tmp = AA63;
      AA63 = AA83;
      AA83 = tmp;
      tmp = AA64;
      AA64 = AA84;
      AA84 = tmp;
      tmp = AA65;
      AA65 = AA85;
      AA85 = tmp;
      tmp = AA66;
      AA66 = AA86;
      AA86 = tmp;
      tmp = AA67;
      AA67 = AA87;
      AA87 = tmp;
      tmp = AA68;
      AA68 = AA88;
      AA88 = tmp;
      /* update permutation vector based on row swap */
      i = perm6;
      perm6 = perm8;
      perm8 = i;
    }
#endif // USE_PIVOTING

    /* scale current row */
    tmp = rcpOp(AA66);
    icol6 = perm6;
    AA60 = mulOp(tmp, AA60);
    AA61 = mulOp(tmp, AA61);
    AA62 = mulOp(tmp, AA62);
    AA63 = mulOp(tmp, AA63);
    AA64 = mulOp(tmp, AA64);
    AA65 = mulOp(tmp, AA65);
    AA66 = tmp;
    AA67 = mulOp(tmp, AA67);
    AA68 = mulOp(tmp, AA68);

    /* eliminate above and below current row */
    tmp = AA06;
    AA00 = fmnaOp(tmp, AA60, AA00);
    AA01 = fmnaOp(tmp, AA61, AA01);
    AA02 = fmnaOp(tmp, AA62, AA02);
    AA03 = fmnaOp(tmp, AA63, AA03);
    AA04 = fmnaOp(tmp, AA64, AA04);
    AA05 = fmnaOp(tmp, AA65, AA05);
    AA06 = mulOp(negOp(tmp), AA66);
    AA07 = fmnaOp(tmp, AA67, AA07);
    AA08 = fmnaOp(tmp, AA68, AA08);

    tmp = AA16;
    AA10 = fmnaOp(tmp, AA60, AA10);
    AA11 = fmnaOp(tmp, AA61, AA11);
    AA12 = fmnaOp(tmp, AA62, AA12);
    AA13 = fmnaOp(tmp, AA63, AA13);
    AA14 = fmnaOp(tmp, AA64, AA14);
    AA15 = fmnaOp(tmp, AA65, AA15);
    AA16 = mulOp(negOp(tmp), AA66);
    AA17 = fmnaOp(tmp, AA67, AA17);
    AA18 = fmnaOp(tmp, AA68, AA18);

    tmp = AA26;
    AA20 = fmnaOp(tmp, AA60, AA20);
    AA21 = fmnaOp(tmp, AA61, AA21);
    AA22 = fmnaOp(tmp, AA62, AA22);
    AA23 = fmnaOp(tmp, AA63, AA23);
    AA24 = fmnaOp(tmp, AA64, AA24);
    AA25 = fmnaOp(tmp, AA65, AA25);
    AA26 = mulOp(negOp(tmp), AA66);
    AA27 = fmnaOp(tmp, AA67, AA27);
    AA28 = fmnaOp(tmp, AA68, AA28);

    tmp = AA36;
    AA30 = fmnaOp(tmp, AA60, AA30);
    AA31 = fmnaOp(tmp, AA61, AA31);
    AA32 = fmnaOp(tmp, AA62, AA32);
    AA33 = fmnaOp(tmp, AA63, AA33);
    AA34 = fmnaOp(tmp, AA64, AA34);
    AA35 = fmnaOp(tmp, AA65, AA35);
    AA36 = mulOp(negOp(tmp), AA66);
    AA37 = fmnaOp(tmp, AA67, AA37);
    AA38 = fmnaOp(tmp, AA68, AA38);

    tmp = AA46;
    AA40 = fmnaOp(tmp, AA60, AA40);
    AA41 = fmnaOp(tmp, AA61, AA41);
    AA42 = fmnaOp(tmp, AA62, AA42);
    AA43 = fmnaOp(tmp, AA63, AA43);
    AA44 = fmnaOp(tmp, AA64, AA44);
    AA45 = fmnaOp(tmp, AA65, AA45);
    AA46 = mulOp(negOp(tmp), AA66);
    AA47 = fmnaOp(tmp, AA67, AA47);
    AA48 = fmnaOp(tmp, AA68, AA48);

    tmp = AA56;
    AA50 = fmnaOp(tmp, AA60, AA50);
    AA51 = fmnaOp(tmp, AA61, AA51);
    AA52 = fmnaOp(tmp, AA62, AA52);
    AA53 = fmnaOp(tmp, AA63, AA53);
    AA54 = fmnaOp(tmp, AA64, AA54);
    AA55 = fmnaOp(tmp, AA65, AA55);
    AA56 = mulOp(negOp(tmp), AA66);
    AA57 = fmnaOp(tmp, AA67, AA57);
    AA58 = fmnaOp(tmp, AA68, AA58);

    tmp = AA76;
    AA70 = fmnaOp(tmp, AA60, AA70);
    AA71 = fmnaOp(tmp, AA61, AA71);
    AA72 = fmnaOp(tmp, AA62, AA72);
    AA73 = fmnaOp(tmp, AA63, AA73);
    AA74 = fmnaOp(tmp, AA64, AA74);
    AA75 = fmnaOp(tmp, AA65, AA75);
    AA76 = mulOp(negOp(tmp), AA66);
    AA77 = fmnaOp(tmp, AA67, AA77);
    AA78 = fmnaOp(tmp, AA68, AA78);

    tmp = AA86;
    AA80 = fmnaOp(tmp, AA60, AA80);
    AA81 = fmnaOp(tmp, AA61, AA81);
    AA82 = fmnaOp(tmp, AA62, AA82);
    AA83 = fmnaOp(tmp, AA63, AA83);
    AA84 = fmnaOp(tmp, AA64, AA84);
    AA85 = fmnaOp(tmp, AA65, AA85);
    AA86 = mulOp(negOp(tmp), AA66);
    AA87 = fmnaOp(tmp, AA67, AA87);
    AA88 = fmnaOp(tmp, AA68, AA88);

    /****************** iteration 7 ****************/

#if USE_PIVOTING
    /* search pivot row */
    p = absOp(AA77);
    pvt = 7;
    t = absOp(AA87);
    if (t > p) {
      p = t;
      pvt = 8;
    }

    /* swap pivot row with row 7 */
    if (pvt == 8) {
      tmp = AA70;
      AA70 = AA80;
      AA80 = tmp;
      tmp = AA71;
      AA71 = AA81;
      AA81 = tmp;
      tmp = AA72;
      AA72 = AA82;
      AA82 = tmp;
      tmp = AA73;
      AA73 = AA83;
      AA83 = tmp;
      tmp = AA74;
      AA74 = AA84;
      AA84 = tmp;
      tmp = AA75;
      AA75 = AA85;
      AA85 = tmp;
      tmp = AA76;
      AA76 = AA86;
      AA86 = tmp;
      tmp = AA77;
      AA77 = AA87;
      AA87 = tmp;
      tmp = AA78;
      AA78 = AA88;
      AA88 = tmp;
      /* update permutation vector based on row swap */
      i = perm7;
      perm7 = perm8;
      perm8 = i;
    }
#endif // USE_PIVOTING

    /* scale current row */
    tmp = rcpOp(AA77);
    icol7 = perm7;
    AA70 = mulOp(tmp, AA70);
    AA71 = mulOp(tmp, AA71);
    AA72 = mulOp(tmp, AA72);
    AA73 = mulOp(tmp, AA73);
    AA74 = mulOp(tmp, AA74);
    AA75 = mulOp(tmp, AA75);
    AA76 = mulOp(tmp, AA76);
    AA77 = tmp;
    AA78 = mulOp(tmp, AA78);

    /* eliminate above and below current row */
    tmp = AA07;
    AA00 = fmnaOp(tmp, AA70, AA00);
    AA01 = fmnaOp(tmp, AA71, AA01);
    AA02 = fmnaOp(tmp, AA72, AA02);
    AA03 = fmnaOp(tmp, AA73, AA03);
    AA04 = fmnaOp(tmp, AA74, AA04);
    AA05 = fmnaOp(tmp, AA75, AA05);
    AA06 = fmnaOp(tmp, AA76, AA06);
    AA07 = mulOp(negOp(tmp), AA77);
    AA08 = fmnaOp(tmp, AA78, AA08);

    tmp = AA17;
    AA10 = fmnaOp(tmp, AA70, AA10);
    AA11 = fmnaOp(tmp, AA71, AA11);
    AA12 = fmnaOp(tmp, AA72, AA12);
    AA13 = fmnaOp(tmp, AA73, AA13);
    AA14 = fmnaOp(tmp, AA74, AA14);
    AA15 = fmnaOp(tmp, AA75, AA15);
    AA16 = fmnaOp(tmp, AA76, AA16);
    AA17 = mulOp(negOp(tmp), AA77);
    AA18 = fmnaOp(tmp, AA78, AA18);

    tmp = AA27;
    AA20 = fmnaOp(tmp, AA70, AA20);
    AA21 = fmnaOp(tmp, AA71, AA21);
    AA22 = fmnaOp(tmp, AA72, AA22);
    AA23 = fmnaOp(tmp, AA73, AA23);
    AA24 = fmnaOp(tmp, AA74, AA24);
    AA25 = fmnaOp(tmp, AA75, AA25);
    AA26 = fmnaOp(tmp, AA76, AA26);
    AA27 = mulOp(negOp(tmp), AA77);
    AA28 = fmnaOp(tmp, AA78, AA28);

    tmp = AA37;
    AA30 = fmnaOp(tmp, AA70, AA30);
    AA31 = fmnaOp(tmp, AA71, AA31);
    AA32 = fmnaOp(tmp, AA72, AA32);
    AA33 = fmnaOp(tmp, AA73, AA33);
    AA34 = fmnaOp(tmp, AA74, AA34);
    AA35 = fmnaOp(tmp, AA75, AA35);
    AA36 = fmnaOp(tmp, AA76, AA36);
    AA37 = mulOp(negOp(tmp), AA77);
    AA38 = fmnaOp(tmp, AA78, AA38);

    tmp = AA47;
    AA40 = fmnaOp(tmp, AA70, AA40);
    AA41 = fmnaOp(tmp, AA71, AA41);
    AA42 = fmnaOp(tmp, AA72, AA42);
    AA43 = fmnaOp(tmp, AA73, AA43);
    AA44 = fmnaOp(tmp, AA74, AA44);
    AA45 = fmnaOp(tmp, AA75, AA45);
    AA46 = fmnaOp(tmp, AA76, AA46);
    AA47 = mulOp(negOp(tmp), AA77);
    AA48 = fmnaOp(tmp, AA78, AA48);

    tmp = AA57;
    AA50 = fmnaOp(tmp, AA70, AA50);
    AA51 = fmnaOp(tmp, AA71, AA51);
    AA52 = fmnaOp(tmp, AA72, AA52);
    AA53 = fmnaOp(tmp, AA73, AA53);
    AA54 = fmnaOp(tmp, AA74, AA54);
    AA55 = fmnaOp(tmp, AA75, AA55);
    AA56 = fmnaOp(tmp, AA76, AA56);
    AA57 = mulOp(negOp(tmp), AA77);
    AA58 = fmnaOp(tmp, AA78, AA58);

    tmp = AA67;
    AA60 = fmnaOp(tmp, AA70, AA60);
    AA61 = fmnaOp(tmp, AA71, AA61);
    AA62 = fmnaOp(tmp, AA72, AA62);
    AA63 = fmnaOp(tmp, AA73, AA63);
    AA64 = fmnaOp(tmp, AA74, AA64);
    AA65 = fmnaOp(tmp, AA75, AA65);
    AA66 = fmnaOp(tmp, AA76, AA66);
    AA67 = mulOp(negOp(tmp), AA77);
    AA68 = fmnaOp(tmp, AA78, AA68);

    tmp = AA87;
    AA80 = fmnaOp(tmp, AA70, AA80);
    AA81 = fmnaOp(tmp, AA71, AA81);
    AA82 = fmnaOp(tmp, AA72, AA82);
    AA83 = fmnaOp(tmp, AA73, AA83);
    AA84 = fmnaOp(tmp, AA74, AA84);
    AA85 = fmnaOp(tmp, AA75, AA85);
    AA86 = fmnaOp(tmp, AA76, AA86);
    AA87 = mulOp(negOp(tmp), AA77);
    AA88 = fmnaOp(tmp, AA78, AA88);

    /****************** iteration 8 ****************/

    /* scale current row */
    tmp = rcpOp(AA88);
    icol8 = perm8;
    AA80 = mulOp(tmp, AA80);
    AA81 = mulOp(tmp, AA81);
    AA82 = mulOp(tmp, AA82);
    AA83 = mulOp(tmp, AA83);
    AA84 = mulOp(tmp, AA84);
    AA85 = mulOp(tmp, AA85);
    AA86 = mulOp(tmp, AA86);
    AA87 = mulOp(tmp, AA87);
    AA88 = tmp;

    /* eliminate above and below current row */
    tmp = AA08;
    AA00 = fmnaOp(tmp, AA80, AA00);
    AA01 = fmnaOp(tmp, AA81, AA01);
    AA02 = fmnaOp(tmp, AA82, AA02);
    AA03 = fmnaOp(tmp, AA83, AA03);
    AA04 = fmnaOp(tmp, AA84, AA04);
    AA05 = fmnaOp(tmp, AA85, AA05);
    AA06 = fmnaOp(tmp, AA86, AA06);
    AA07 = fmnaOp(tmp, AA87, AA07);
    AA08 = mulOp(negOp(tmp), AA88);

    tmp = AA18;
    AA10 = fmnaOp(tmp, AA80, AA10);
    AA11 = fmnaOp(tmp, AA81, AA11);
    AA12 = fmnaOp(tmp, AA82, AA12);
    AA13 = fmnaOp(tmp, AA83, AA13);
    AA14 = fmnaOp(tmp, AA84, AA14);
    AA15 = fmnaOp(tmp, AA85, AA15);
    AA16 = fmnaOp(tmp, AA86, AA16);
    AA17 = fmnaOp(tmp, AA87, AA17);
    AA18 = mulOp(negOp(tmp), AA88);

    tmp = AA28;
    AA20 = fmnaOp(tmp, AA80, AA20);
    AA21 = fmnaOp(tmp, AA81, AA21);
    AA22 = fmnaOp(tmp, AA82, AA22);
    AA23 = fmnaOp(tmp, AA83, AA23);
    AA24 = fmnaOp(tmp, AA84, AA24);
    AA25 = fmnaOp(tmp, AA85, AA25);
    AA26 = fmnaOp(tmp, AA86, AA26);
    AA27 = fmnaOp(tmp, AA87, AA27);
    AA28 = mulOp(negOp(tmp), AA88);

    tmp = AA38;
    AA30 = fmnaOp(tmp, AA80, AA30);
    AA31 = fmnaOp(tmp, AA81, AA31);
    AA32 = fmnaOp(tmp, AA82, AA32);
    AA33 = fmnaOp(tmp, AA83, AA33);
    AA34 = fmnaOp(tmp, AA84, AA34);
    AA35 = fmnaOp(tmp, AA85, AA35);
    AA36 = fmnaOp(tmp, AA86, AA36);
    AA37 = fmnaOp(tmp, AA87, AA37);
    AA38 = mulOp(negOp(tmp), AA88);

    tmp = AA48;
    AA40 = fmnaOp(tmp, AA80, AA40);
    AA41 = fmnaOp(tmp, AA81, AA41);
    AA42 = fmnaOp(tmp, AA82, AA42);
    AA43 = fmnaOp(tmp, AA83, AA43);
    AA44 = fmnaOp(tmp, AA84, AA44);
    AA45 = fmnaOp(tmp, AA85, AA45);
    AA46 = fmnaOp(tmp, AA86, AA46);
    AA47 = fmnaOp(tmp, AA87, AA47);
    AA48 = mulOp(negOp(tmp), AA88);

    tmp = AA58;
    AA50 = fmnaOp(tmp, AA80, AA50);
    AA51 = fmnaOp(tmp, AA81, AA51);
    AA52 = fmnaOp(tmp, AA82, AA52);
    AA53 = fmnaOp(tmp, AA83, AA53);
    AA54 = fmnaOp(tmp, AA84, AA54);
    AA55 = fmnaOp(tmp, AA85, AA55);
    AA56 = fmnaOp(tmp, AA86, AA56);
    AA57 = fmnaOp(tmp, AA87, AA57);
    AA58 = mulOp(negOp(tmp), AA88);

    tmp = AA68;
    AA60 = fmnaOp(tmp, AA80, AA60);
    AA61 = fmnaOp(tmp, AA81, AA61);
    AA62 = fmnaOp(tmp, AA82, AA62);
    AA63 = fmnaOp(tmp, AA83, AA63);
    AA64 = fmnaOp(tmp, AA84, AA64);
    AA65 = fmnaOp(tmp, AA85, AA65);
    AA66 = fmnaOp(tmp, AA86, AA66);
    AA67 = fmnaOp(tmp, AA87, AA67);
    AA68 = mulOp(negOp(tmp), AA88);

    tmp = AA78;
    AA70 = fmnaOp(tmp, AA80, AA70);
    AA71 = fmnaOp(tmp, AA81, AA71);
    AA72 = fmnaOp(tmp, AA82, AA72);
    AA73 = fmnaOp(tmp, AA83, AA73);
    AA74 = fmnaOp(tmp, AA84, AA74);
    AA75 = fmnaOp(tmp, AA85, AA75);
    AA76 = fmnaOp(tmp, AA86, AA76);
    AA77 = fmnaOp(tmp, AA87, AA77);
    AA78 = mulOp(negOp(tmp), AA88);

    /* sort columns into the correct order */
    Ainv(0, icol0) = AA00;
    Ainv(1, icol0) = AA10;
    Ainv(2, icol0) = AA20;
    Ainv(3, icol0) = AA30;
    Ainv(4, icol0) = AA40;
    Ainv(5, icol0) = AA50;
    Ainv(6, icol0) = AA60;
    Ainv(7, icol0) = AA70;
    Ainv(8, icol0) = AA80;
    Ainv(0, icol1) = AA01;
    Ainv(1, icol1) = AA11;
    Ainv(2, icol1) = AA21;
    Ainv(3, icol1) = AA31;
    Ainv(4, icol1) = AA41;
    Ainv(5, icol1) = AA51;
    Ainv(6, icol1) = AA61;
    Ainv(7, icol1) = AA71;
    Ainv(8, icol1) = AA81;
    Ainv(0, icol2) = AA02;
    Ainv(1, icol2) = AA12;
    Ainv(2, icol2) = AA22;
    Ainv(3, icol2) = AA32;
    Ainv(4, icol2) = AA42;
    Ainv(5, icol2) = AA52;
    Ainv(6, icol2) = AA62;
    Ainv(7, icol2) = AA72;
    Ainv(8, icol2) = AA82;
    Ainv(0, icol3) = AA03;
    Ainv(1, icol3) = AA13;
    Ainv(2, icol3) = AA23;
    Ainv(3, icol3) = AA33;
    Ainv(4, icol3) = AA43;
    Ainv(5, icol3) = AA53;
    Ainv(6, icol3) = AA63;
    Ainv(7, icol3) = AA73;
    Ainv(8, icol3) = AA83;
    Ainv(0, icol4) = AA04;
    Ainv(1, icol4) = AA14;
    Ainv(2, icol4) = AA24;
    Ainv(3, icol4) = AA34;
    Ainv(4, icol4) = AA44;
    Ainv(5, icol4) = AA54;
    Ainv(6, icol4) = AA64;
    Ainv(7, icol4) = AA74;
    Ainv(8, icol4) = AA84;
    Ainv(0, icol5) = AA05;
    Ainv(1, icol5) = AA15;
    Ainv(2, icol5) = AA25;
    Ainv(3, icol5) = AA35;
    Ainv(4, icol5) = AA45;
    Ainv(5, icol5) = AA55;
    Ainv(6, icol5) = AA65;
    Ainv(7, icol5) = AA75;
    Ainv(8, icol5) = AA85;
    Ainv(0, icol6) = AA06;
    Ainv(1, icol6) = AA16;
    Ainv(2, icol6) = AA26;
    Ainv(3, icol6) = AA36;
    Ainv(4, icol6) = AA46;
    Ainv(5, icol6) = AA56;
    Ainv(6, icol6) = AA66;
    Ainv(7, icol6) = AA76;
    Ainv(8, icol6) = AA86;
    Ainv(0, icol7) = AA07;
    Ainv(1, icol7) = AA17;
    Ainv(2, icol7) = AA27;
    Ainv(3, icol7) = AA37;
    Ainv(4, icol7) = AA47;
    Ainv(5, icol7) = AA57;
    Ainv(6, icol7) = AA67;
    Ainv(7, icol7) = AA77;
    Ainv(8, icol7) = AA87;
    Ainv(0, icol8) = AA08;
    Ainv(1, icol8) = AA18;
    Ainv(2, icol8) = AA28;
    Ainv(3, icol8) = AA38;
    Ainv(4, icol8) = AA48;
    Ainv(5, icol8) = AA58;
    Ainv(6, icol8) = AA68;
    Ainv(7, icol8) = AA78;
    Ainv(8, icol8) = AA88;
  }
}

template <typename T, int arch>
__global__ void lu_matinv_10x10_matrix_per_thread(const T *A, T *Ainv, int batch) {
  /* This is a hack. The instatiation of this template functions fails when
     arch = ARCH_SM13 and T = cuDoubleComplex, since the generated code needs
     more than 16KB of local memory. Since we don't need an instance of this
     template function on either sm_13 nor sm_20, simply compile out all code
     in the function when T = cuDoubleComplex.
  */
  const int blkNum = blockIdx.y * gridDim.x + blockIdx.x;
  const int thrdNum = blkNum * blockDim.x + threadIdx.x;
  const int N = 10;
  int perm0, perm1, perm2, perm3, perm4, perm5, perm6, perm7, perm8, perm9;
  int icol0, icol1, icol2, icol3, icol4, icol5, icol6, icol7, icol8, icol9;
  T AA00, AA01, AA02, AA03, AA04, AA05, AA06, AA07, AA08, AA09;
  T AA10, AA11, AA12, AA13, AA14, AA15, AA16, AA17, AA18, AA19;
  T AA20, AA21, AA22, AA23, AA24, AA25, AA26, AA27, AA28, AA29;
  T AA30, AA31, AA32, AA33, AA34, AA35, AA36, AA37, AA38, AA39;
  T AA40, AA41, AA42, AA43, AA44, AA45, AA46, AA47, AA48, AA49;
  T AA50, AA51, AA52, AA53, AA54, AA55, AA56, AA57, AA58, AA59;
  T AA60, AA61, AA62, AA63, AA64, AA65, AA66, AA67, AA68, AA69;
  T AA70, AA71, AA72, AA73, AA74, AA75, AA76, AA77, AA78, AA79;
  T AA80, AA81, AA82, AA83, AA84, AA85, AA86, AA87, AA88, AA89;
  T AA90, AA91, AA92, AA93, AA94, AA95, AA96, AA97, AA98, AA99;
  T tmp;
#if USE_PIVOTING
  typename config<T, arch>::absValType t;
  typename config<T, arch>::absValType p;
  int i, pvt;
#endif

  if (thrdNum < batch) {

    AA00 = A[0 * batch + thrdNum];
    AA10 = A[1 * batch + thrdNum];
    AA20 = A[2 * batch + thrdNum];
    AA30 = A[3 * batch + thrdNum];
    AA40 = A[4 * batch + thrdNum];
    AA50 = A[5 * batch + thrdNum];
    AA60 = A[6 * batch + thrdNum];
    AA70 = A[7 * batch + thrdNum];
    AA80 = A[8 * batch + thrdNum];
    AA90 = A[9 * batch + thrdNum];
    AA01 = A[10 * batch + thrdNum];
    AA11 = A[11 * batch + thrdNum];
    AA21 = A[12 * batch + thrdNum];
    AA31 = A[13 * batch + thrdNum];
    AA41 = A[14 * batch + thrdNum];
    AA51 = A[15 * batch + thrdNum];
    AA61 = A[16 * batch + thrdNum];
    AA71 = A[17 * batch + thrdNum];
    AA81 = A[18 * batch + thrdNum];
    AA91 = A[19 * batch + thrdNum];
    AA02 = A[20 * batch + thrdNum];
    AA12 = A[21 * batch + thrdNum];
    AA22 = A[22 * batch + thrdNum];
    AA32 = A[23 * batch + thrdNum];
    AA42 = A[24 * batch + thrdNum];
    AA52 = A[25 * batch + thrdNum];
    AA62 = A[26 * batch + thrdNum];
    AA72 = A[27 * batch + thrdNum];
    AA82 = A[28 * batch + thrdNum];
    AA92 = A[29 * batch + thrdNum];
    AA03 = A[30 * batch + thrdNum];
    AA13 = A[31 * batch + thrdNum];
    AA23 = A[32 * batch + thrdNum];
    AA33 = A[33 * batch + thrdNum];
    AA43 = A[34 * batch + thrdNum];
    AA53 = A[35 * batch + thrdNum];
    AA63 = A[36 * batch + thrdNum];
    AA73 = A[37 * batch + thrdNum];
    AA83 = A[38 * batch + thrdNum];
    AA93 = A[39 * batch + thrdNum];
    AA04 = A[40 * batch + thrdNum];
    AA14 = A[41 * batch + thrdNum];
    AA24 = A[42 * batch + thrdNum];
    AA34 = A[43 * batch + thrdNum];
    AA44 = A[44 * batch + thrdNum];
    AA54 = A[45 * batch + thrdNum];
    AA64 = A[46 * batch + thrdNum];
    AA74 = A[47 * batch + thrdNum];
    AA84 = A[48 * batch + thrdNum];
    AA94 = A[49 * batch + thrdNum];
    AA05 = A[50 * batch + thrdNum];
    AA15 = A[51 * batch + thrdNum];
    AA25 = A[52 * batch + thrdNum];
    AA35 = A[53 * batch + thrdNum];
    AA45 = A[54 * batch + thrdNum];
    AA55 = A[55 * batch + thrdNum];
    AA65 = A[56 * batch + thrdNum];
    AA75 = A[57 * batch + thrdNum];
    AA85 = A[58 * batch + thrdNum];
    AA95 = A[59 * batch + thrdNum];
    AA06 = A[60 * batch + thrdNum];
    AA16 = A[61 * batch + thrdNum];
    AA26 = A[62 * batch + thrdNum];
    AA36 = A[63 * batch + thrdNum];
    AA46 = A[64 * batch + thrdNum];
    AA56 = A[65 * batch + thrdNum];
    AA66 = A[66 * batch + thrdNum];
    AA76 = A[67 * batch + thrdNum];
    AA86 = A[68 * batch + thrdNum];
    AA96 = A[69 * batch + thrdNum];
    AA07 = A[70 * batch + thrdNum];
    AA17 = A[71 * batch + thrdNum];
    AA27 = A[72 * batch + thrdNum];
    AA37 = A[73 * batch + thrdNum];
    AA47 = A[74 * batch + thrdNum];
    AA57 = A[75 * batch + thrdNum];
    AA67 = A[76 * batch + thrdNum];
    AA77 = A[77 * batch + thrdNum];
    AA87 = A[78 * batch + thrdNum];
    AA97 = A[79 * batch + thrdNum];
    AA08 = A[80 * batch + thrdNum];
    AA18 = A[81 * batch + thrdNum];
    AA28 = A[82 * batch + thrdNum];
    AA38 = A[83 * batch + thrdNum];
    AA48 = A[84 * batch + thrdNum];
    AA58 = A[85 * batch + thrdNum];
    AA68 = A[86 * batch + thrdNum];
    AA78 = A[87 * batch + thrdNum];
    AA88 = A[88 * batch + thrdNum];
    AA98 = A[89 * batch + thrdNum];
    AA09 = A[90 * batch + thrdNum];
    AA19 = A[91 * batch + thrdNum];
    AA29 = A[92 * batch + thrdNum];
    AA39 = A[93 * batch + thrdNum];
    AA49 = A[94 * batch + thrdNum];
    AA59 = A[95 * batch + thrdNum];
    AA69 = A[96 * batch + thrdNum];
    AA79 = A[97 * batch + thrdNum];
    AA89 = A[98 * batch + thrdNum];
    AA99 = A[99 * batch + thrdNum];

    perm0 = 0;
    perm1 = 1;
    perm2 = 2;
    perm3 = 3;
    perm4 = 4;
    perm5 = 5;
    perm6 = 6;
    perm7 = 7;
    perm8 = 8;
    perm9 = 9;

    /****************** iteration 0 ***********/

#if USE_PIVOTING
    /* search pivot row */
    p = absOp(AA00);
    pvt = 0;
    t = absOp(AA10);
    if (t > p) {
      p = t;
      pvt = 1;
    }
    t = absOp(AA20);
    if (t > p) {
      p = t;
      pvt = 2;
    }
    t = absOp(AA30);
    if (t > p) {
      p = t;
      pvt = 3;
    }
    t = absOp(AA40);
    if (t > p) {
      p = t;
      pvt = 4;
    }
    t = absOp(AA50);
    if (t > p) {
      p = t;
      pvt = 5;
    }
    t = absOp(AA60);
    if (t > p) {
      p = t;
      pvt = 6;
    }
    t = absOp(AA70);
    if (t > p) {
      p = t;
      pvt = 7;
    }
    t = absOp(AA80);
    if (t > p) {
      p = t;
      pvt = 8;
    }
    t = absOp(AA90);
    if (t > p) {
      p = t;
      pvt = 9;
    }

    /* swap pivot row with row 0 */
    if (pvt == 1) {
      tmp = AA00;
      AA00 = AA10;
      AA10 = tmp;
      tmp = AA01;
      AA01 = AA11;
      AA11 = tmp;
      tmp = AA02;
      AA02 = AA12;
      AA12 = tmp;
      tmp = AA03;
      AA03 = AA13;
      AA13 = tmp;
      tmp = AA04;
      AA04 = AA14;
      AA14 = tmp;
      tmp = AA05;
      AA05 = AA15;
      AA15 = tmp;
      tmp = AA06;
      AA06 = AA16;
      AA16 = tmp;
      tmp = AA07;
      AA07 = AA17;
      AA17 = tmp;
      tmp = AA08;
      AA08 = AA18;
      AA18 = tmp;
      tmp = AA09;
      AA09 = AA19;
      AA19 = tmp;
      /* update permutation vector based on row swap */
      i = perm0;
      perm0 = perm1;
      perm1 = i;
    }
    if (pvt == 2) {
      tmp = AA00;
      AA00 = AA20;
      AA20 = tmp;
      tmp = AA01;
      AA01 = AA21;
      AA21 = tmp;
      tmp = AA02;
      AA02 = AA22;
      AA22 = tmp;
      tmp = AA03;
      AA03 = AA23;
      AA23 = tmp;
      tmp = AA04;
      AA04 = AA24;
      AA24 = tmp;
      tmp = AA05;
      AA05 = AA25;
      AA25 = tmp;
      tmp = AA06;
      AA06 = AA26;
      AA26 = tmp;
      tmp = AA07;
      AA07 = AA27;
      AA27 = tmp;
      tmp = AA08;
      AA08 = AA28;
      AA28 = tmp;
      tmp = AA09;
      AA09 = AA29;
      AA29 = tmp;
      /* update permutation vector based on row swap */
      i = perm0;
      perm0 = perm2;
      perm2 = i;
    }
    if (pvt == 3) {
      tmp = AA00;
      AA00 = AA30;
      AA30 = tmp;
      tmp = AA01;
      AA01 = AA31;
      AA31 = tmp;
      tmp = AA02;
      AA02 = AA32;
      AA32 = tmp;
      tmp = AA03;
      AA03 = AA33;
      AA33 = tmp;
      tmp = AA04;
      AA04 = AA34;
      AA34 = tmp;
      tmp = AA05;
      AA05 = AA35;
      AA35 = tmp;
      tmp = AA06;
      AA06 = AA36;
      AA36 = tmp;
      tmp = AA07;
      AA07 = AA37;
      AA37 = tmp;
      tmp = AA08;
      AA08 = AA38;
      AA38 = tmp;
      tmp = AA09;
      AA09 = AA39;
      AA39 = tmp;
      /* update permutation vector based on row swap */
      i = perm0;
      perm0 = perm3;
      perm3 = i;
    }
    if (pvt == 4) {
      tmp = AA00;
      AA00 = AA40;
      AA40 = tmp;
      tmp = AA01;
      AA01 = AA41;
      AA41 = tmp;
      tmp = AA02;
      AA02 = AA42;
      AA42 = tmp;
      tmp = AA03;
      AA03 = AA43;
      AA43 = tmp;
      tmp = AA04;
      AA04 = AA44;
      AA44 = tmp;
      tmp = AA05;
      AA05 = AA45;
      AA45 = tmp;
      tmp = AA06;
      AA06 = AA46;
      AA46 = tmp;
      tmp = AA07;
      AA07 = AA47;
      AA47 = tmp;
      tmp = AA08;
      AA08 = AA48;
      AA48 = tmp;
      tmp = AA09;
      AA09 = AA49;
      AA49 = tmp;
      /* update permutation vector based on row swap */
      i = perm0;
      perm0 = perm4;
      perm4 = i;
    }
    if (pvt == 5) {
      tmp = AA00;
      AA00 = AA50;
      AA50 = tmp;
      tmp = AA01;
      AA01 = AA51;
      AA51 = tmp;
      tmp = AA02;
      AA02 = AA52;
      AA52 = tmp;
      tmp = AA03;
      AA03 = AA53;
      AA53 = tmp;
      tmp = AA04;
      AA04 = AA54;
      AA54 = tmp;
      tmp = AA05;
      AA05 = AA55;
      AA55 = tmp;
      tmp = AA06;
      AA06 = AA56;
      AA56 = tmp;
      tmp = AA07;
      AA07 = AA57;
      AA57 = tmp;
      tmp = AA08;
      AA08 = AA58;
      AA58 = tmp;
      tmp = AA09;
      AA09 = AA59;
      AA59 = tmp;
      /* update permutation vector based on row swap */
      i = perm0;
      perm0 = perm5;
      perm5 = i;
    }
    if (pvt == 6) {
      tmp = AA00;
      AA00 = AA60;
      AA60 = tmp;
      tmp = AA01;
      AA01 = AA61;
      AA61 = tmp;
      tmp = AA02;
      AA02 = AA62;
      AA62 = tmp;
      tmp = AA03;
      AA03 = AA63;
      AA63 = tmp;
      tmp = AA04;
      AA04 = AA64;
      AA64 = tmp;
      tmp = AA05;
      AA05 = AA65;
      AA65 = tmp;
      tmp = AA06;
      AA06 = AA66;
      AA66 = tmp;
      tmp = AA07;
      AA07 = AA67;
      AA67 = tmp;
      tmp = AA08;
      AA08 = AA68;
      AA68 = tmp;
      tmp = AA09;
      AA09 = AA69;
      AA69 = tmp;
      /* update permutation vector based on row swap */
      i = perm0;
      perm0 = perm6;
      perm6 = i;
    }
    if (pvt == 7) {
      tmp = AA00;
      AA00 = AA70;
      AA70 = tmp;
      tmp = AA01;
      AA01 = AA71;
      AA71 = tmp;
      tmp = AA02;
      AA02 = AA72;
      AA72 = tmp;
      tmp = AA03;
      AA03 = AA73;
      AA73 = tmp;
      tmp = AA04;
      AA04 = AA74;
      AA74 = tmp;
      tmp = AA05;
      AA05 = AA75;
      AA75 = tmp;
      tmp = AA06;
      AA06 = AA76;
      AA76 = tmp;
      tmp = AA07;
      AA07 = AA77;
      AA77 = tmp;
      tmp = AA08;
      AA08 = AA78;
      AA78 = tmp;
      tmp = AA09;
      AA09 = AA79;
      AA79 = tmp;
      /* update permutation vector based on row swap */
      i = perm0;
      perm0 = perm7;
      perm7 = i;
    }
    if (pvt == 8) {
      tmp = AA00;
      AA00 = AA80;
      AA80 = tmp;
      tmp = AA01;
      AA01 = AA81;
      AA81 = tmp;
      tmp = AA02;
      AA02 = AA82;
      AA82 = tmp;
      tmp = AA03;
      AA03 = AA83;
      AA83 = tmp;
      tmp = AA04;
      AA04 = AA84;
      AA84 = tmp;
      tmp = AA05;
      AA05 = AA85;
      AA85 = tmp;
      tmp = AA06;
      AA06 = AA86;
      AA86 = tmp;
      tmp = AA07;
      AA07 = AA87;
      AA87 = tmp;
      tmp = AA08;
      AA08 = AA88;
      AA88 = tmp;
      tmp = AA09;
      AA09 = AA89;
      AA89 = tmp;
      /* update permutation vector based on row swap */
      i = perm0;
      perm0 = perm8;
      perm8 = i;
    }
    if (pvt == 9) {
      tmp = AA00;
      AA00 = AA90;
      AA90 = tmp;
      tmp = AA01;
      AA01 = AA91;
      AA91 = tmp;
      tmp = AA02;
      AA02 = AA92;
      AA92 = tmp;
      tmp = AA03;
      AA03 = AA93;
      AA93 = tmp;
      tmp = AA04;
      AA04 = AA94;
      AA94 = tmp;
      tmp = AA05;
      AA05 = AA95;
      AA95 = tmp;
      tmp = AA06;
      AA06 = AA96;
      AA96 = tmp;
      tmp = AA07;
      AA07 = AA97;
      AA97 = tmp;
      tmp = AA08;
      AA08 = AA98;
      AA98 = tmp;
      tmp = AA09;
      AA09 = AA99;
      AA99 = tmp;
      /* update permutation vector based on row swap */
      i = perm0;
      perm0 = perm9;
      perm9 = i;
    }
#endif // USE_PIVOTING

    /* scale current row */
    tmp = rcpOp(AA00);
    icol0 = perm0;
    AA00 = tmp;
    AA01 = mulOp(tmp, AA01);
    AA02 = mulOp(tmp, AA02);
    AA03 = mulOp(tmp, AA03);
    AA04 = mulOp(tmp, AA04);
    AA05 = mulOp(tmp, AA05);
    AA06 = mulOp(tmp, AA06);
    AA07 = mulOp(tmp, AA07);
    AA08 = mulOp(tmp, AA08);
    AA09 = mulOp(tmp, AA09);

    /* eliminate above and below current row */
    tmp = AA10;
    AA10 = mulOp(negOp(tmp), AA00);
    AA11 = fmnaOp(tmp, AA01, AA11);
    AA12 = fmnaOp(tmp, AA02, AA12);
    AA13 = fmnaOp(tmp, AA03, AA13);
    AA14 = fmnaOp(tmp, AA04, AA14);
    AA15 = fmnaOp(tmp, AA05, AA15);
    AA16 = fmnaOp(tmp, AA06, AA16);
    AA17 = fmnaOp(tmp, AA07, AA17);
    AA18 = fmnaOp(tmp, AA08, AA18);
    AA19 = fmnaOp(tmp, AA09, AA19);

    tmp = AA20;
    AA20 = mulOp(negOp(tmp), AA00);
    AA21 = fmnaOp(tmp, AA01, AA21);
    AA22 = fmnaOp(tmp, AA02, AA22);
    AA23 = fmnaOp(tmp, AA03, AA23);
    AA24 = fmnaOp(tmp, AA04, AA24);
    AA25 = fmnaOp(tmp, AA05, AA25);
    AA26 = fmnaOp(tmp, AA06, AA26);
    AA27 = fmnaOp(tmp, AA07, AA27);
    AA28 = fmnaOp(tmp, AA08, AA28);
    AA29 = fmnaOp(tmp, AA09, AA29);

    tmp = AA30;
    AA30 = mulOp(negOp(tmp), AA00);
    AA31 = fmnaOp(tmp, AA01, AA31);
    AA32 = fmnaOp(tmp, AA02, AA32);
    AA33 = fmnaOp(tmp, AA03, AA33);
    AA34 = fmnaOp(tmp, AA04, AA34);
    AA35 = fmnaOp(tmp, AA05, AA35);
    AA36 = fmnaOp(tmp, AA06, AA36);
    AA37 = fmnaOp(tmp, AA07, AA37);
    AA38 = fmnaOp(tmp, AA08, AA38);
    AA39 = fmnaOp(tmp, AA09, AA39);

    tmp = AA40;
    AA40 = mulOp(negOp(tmp), AA00);
    AA41 = fmnaOp(tmp, AA01, AA41);
    AA42 = fmnaOp(tmp, AA02, AA42);
    AA43 = fmnaOp(tmp, AA03, AA43);
    AA44 = fmnaOp(tmp, AA04, AA44);
    AA45 = fmnaOp(tmp, AA05, AA45);
    AA46 = fmnaOp(tmp, AA06, AA46);
    AA47 = fmnaOp(tmp, AA07, AA47);
    AA48 = fmnaOp(tmp, AA08, AA48);
    AA49 = fmnaOp(tmp, AA09, AA49);

    tmp = AA50;
    AA50 = mulOp(negOp(tmp), AA00);
    AA51 = fmnaOp(tmp, AA01, AA51);
    AA52 = fmnaOp(tmp, AA02, AA52);
    AA53 = fmnaOp(tmp, AA03, AA53);
    AA54 = fmnaOp(tmp, AA04, AA54);
    AA55 = fmnaOp(tmp, AA05, AA55);
    AA56 = fmnaOp(tmp, AA06, AA56);
    AA57 = fmnaOp(tmp, AA07, AA57);
    AA58 = fmnaOp(tmp, AA08, AA58);
    AA59 = fmnaOp(tmp, AA09, AA59);

    tmp = AA60;
    AA60 = mulOp(negOp(tmp), AA00);
    AA61 = fmnaOp(tmp, AA01, AA61);
    AA62 = fmnaOp(tmp, AA02, AA62);
    AA63 = fmnaOp(tmp, AA03, AA63);
    AA64 = fmnaOp(tmp, AA04, AA64);
    AA65 = fmnaOp(tmp, AA05, AA65);
    AA66 = fmnaOp(tmp, AA06, AA66);
    AA67 = fmnaOp(tmp, AA07, AA67);
    AA68 = fmnaOp(tmp, AA08, AA68);
    AA69 = fmnaOp(tmp, AA09, AA69);

    tmp = AA70;
    AA70 = mulOp(negOp(tmp), AA00);
    AA71 = fmnaOp(tmp, AA01, AA71);
    AA72 = fmnaOp(tmp, AA02, AA72);
    AA73 = fmnaOp(tmp, AA03, AA73);
    AA74 = fmnaOp(tmp, AA04, AA74);
    AA75 = fmnaOp(tmp, AA05, AA75);
    AA76 = fmnaOp(tmp, AA06, AA76);
    AA77 = fmnaOp(tmp, AA07, AA77);
    AA78 = fmnaOp(tmp, AA08, AA78);
    AA79 = fmnaOp(tmp, AA09, AA79);

    tmp = AA80;
    AA80 = mulOp(negOp(tmp), AA00);
    AA81 = fmnaOp(tmp, AA01, AA81);
    AA82 = fmnaOp(tmp, AA02, AA82);
    AA83 = fmnaOp(tmp, AA03, AA83);
    AA84 = fmnaOp(tmp, AA04, AA84);
    AA85 = fmnaOp(tmp, AA05, AA85);
    AA86 = fmnaOp(tmp, AA06, AA86);
    AA87 = fmnaOp(tmp, AA07, AA87);
    AA88 = fmnaOp(tmp, AA08, AA88);
    AA89 = fmnaOp(tmp, AA09, AA89);

    tmp = AA90;
    AA90 = mulOp(negOp(tmp), AA00);
    AA91 = fmnaOp(tmp, AA01, AA91);
    AA92 = fmnaOp(tmp, AA02, AA92);
    AA93 = fmnaOp(tmp, AA03, AA93);
    AA94 = fmnaOp(tmp, AA04, AA94);
    AA95 = fmnaOp(tmp, AA05, AA95);
    AA96 = fmnaOp(tmp, AA06, AA96);
    AA97 = fmnaOp(tmp, AA07, AA97);
    AA98 = fmnaOp(tmp, AA08, AA98);
    AA99 = fmnaOp(tmp, AA09, AA99);

    /****************** iteration 1 ***********/

#if USE_PIVOTING
    /* search pivot row */
    p = absOp(AA11);
    pvt = 1;
    t = absOp(AA21);
    if (t > p) {
      p = t;
      pvt = 2;
    }
    t = absOp(AA31);
    if (t > p) {
      p = t;
      pvt = 3;
    }
    t = absOp(AA41);
    if (t > p) {
      p = t;
      pvt = 4;
    }
    t = absOp(AA51);
    if (t > p) {
      p = t;
      pvt = 5;
    }
    t = absOp(AA61);
    if (t > p) {
      p = t;
      pvt = 6;
    }
    t = absOp(AA71);
    if (t > p) {
      p = t;
      pvt = 7;
    }
    t = absOp(AA81);
    if (t > p) {
      p = t;
      pvt = 8;
    }
    t = absOp(AA91);
    if (t > p) {
      p = t;
      pvt = 9;
    }

    /* swap pivot row with row 1 */
    if (pvt == 2) {
      tmp = AA10;
      AA10 = AA20;
      AA20 = tmp;
      tmp = AA11;
      AA11 = AA21;
      AA21 = tmp;
      tmp = AA12;
      AA12 = AA22;
      AA22 = tmp;
      tmp = AA13;
      AA13 = AA23;
      AA23 = tmp;
      tmp = AA14;
      AA14 = AA24;
      AA24 = tmp;
      tmp = AA15;
      AA15 = AA25;
      AA25 = tmp;
      tmp = AA16;
      AA16 = AA26;
      AA26 = tmp;
      tmp = AA17;
      AA17 = AA27;
      AA27 = tmp;
      tmp = AA18;
      AA18 = AA28;
      AA28 = tmp;
      tmp = AA19;
      AA19 = AA29;
      AA29 = tmp;
      /* update permutation vector based on row swap */
      i = perm1;
      perm1 = perm2;
      perm2 = i;
    }
    if (pvt == 3) {
      tmp = AA10;
      AA10 = AA30;
      AA30 = tmp;
      tmp = AA11;
      AA11 = AA31;
      AA31 = tmp;
      tmp = AA12;
      AA12 = AA32;
      AA32 = tmp;
      tmp = AA13;
      AA13 = AA33;
      AA33 = tmp;
      tmp = AA14;
      AA14 = AA34;
      AA34 = tmp;
      tmp = AA15;
      AA15 = AA35;
      AA35 = tmp;
      tmp = AA16;
      AA16 = AA36;
      AA36 = tmp;
      tmp = AA17;
      AA17 = AA37;
      AA37 = tmp;
      tmp = AA18;
      AA18 = AA38;
      AA38 = tmp;
      tmp = AA19;
      AA19 = AA39;
      AA39 = tmp;
      /* update permutation vector based on row swap */
      i = perm1;
      perm1 = perm3;
      perm3 = i;
    }
    if (pvt == 4) {
      tmp = AA10;
      AA10 = AA40;
      AA40 = tmp;
      tmp = AA11;
      AA11 = AA41;
      AA41 = tmp;
      tmp = AA12;
      AA12 = AA42;
      AA42 = tmp;
      tmp = AA13;
      AA13 = AA43;
      AA43 = tmp;
      tmp = AA14;
      AA14 = AA44;
      AA44 = tmp;
      tmp = AA15;
      AA15 = AA45;
      AA45 = tmp;
      tmp = AA16;
      AA16 = AA46;
      AA46 = tmp;
      tmp = AA17;
      AA17 = AA47;
      AA47 = tmp;
      tmp = AA18;
      AA18 = AA48;
      AA48 = tmp;
      tmp = AA19;
      AA19 = AA49;
      AA49 = tmp;
      /* update permutation vector based on row swap */
      i = perm1;
      perm1 = perm4;
      perm4 = i;
    }
    if (pvt == 5) {
      tmp = AA10;
      AA10 = AA50;
      AA50 = tmp;
      tmp = AA11;
      AA11 = AA51;
      AA51 = tmp;
      tmp = AA12;
      AA12 = AA52;
      AA52 = tmp;
      tmp = AA13;
      AA13 = AA53;
      AA53 = tmp;
      tmp = AA14;
      AA14 = AA54;
      AA54 = tmp;
      tmp = AA15;
      AA15 = AA55;
      AA55 = tmp;
      tmp = AA16;
      AA16 = AA56;
      AA56 = tmp;
      tmp = AA17;
      AA17 = AA57;
      AA57 = tmp;
      tmp = AA18;
      AA18 = AA58;
      AA58 = tmp;
      tmp = AA19;
      AA19 = AA59;
      AA59 = tmp;
      /* update permutation vector based on row swap */
      i = perm1;
      perm1 = perm5;
      perm5 = i;
    }
    if (pvt == 6) {
      tmp = AA10;
      AA10 = AA60;
      AA60 = tmp;
      tmp = AA11;
      AA11 = AA61;
      AA61 = tmp;
      tmp = AA12;
      AA12 = AA62;
      AA62 = tmp;
      tmp = AA13;
      AA13 = AA63;
      AA63 = tmp;
      tmp = AA14;
      AA14 = AA64;
      AA64 = tmp;
      tmp = AA15;
      AA15 = AA65;
      AA65 = tmp;
      tmp = AA16;
      AA16 = AA66;
      AA66 = tmp;
      tmp = AA17;
      AA17 = AA67;
      AA67 = tmp;
      tmp = AA18;
      AA18 = AA68;
      AA68 = tmp;
      tmp = AA19;
      AA19 = AA69;
      AA69 = tmp;
      /* update permutation vector based on row swap */
      i = perm1;
      perm1 = perm6;
      perm6 = i;
    }
    if (pvt == 7) {
      tmp = AA10;
      AA10 = AA70;
      AA70 = tmp;
      tmp = AA11;
      AA11 = AA71;
      AA71 = tmp;
      tmp = AA12;
      AA12 = AA72;
      AA72 = tmp;
      tmp = AA13;
      AA13 = AA73;
      AA73 = tmp;
      tmp = AA14;
      AA14 = AA74;
      AA74 = tmp;
      tmp = AA15;
      AA15 = AA75;
      AA75 = tmp;
      tmp = AA16;
      AA16 = AA76;
      AA76 = tmp;
      tmp = AA17;
      AA17 = AA77;
      AA77 = tmp;
      tmp = AA18;
      AA18 = AA78;
      AA78 = tmp;
      tmp = AA19;
      AA19 = AA79;
      AA79 = tmp;
      /* update permutation vector based on row swap */
      i = perm1;
      perm1 = perm7;
      perm7 = i;
    }
    if (pvt == 8) {
      tmp = AA10;
      AA10 = AA80;
      AA80 = tmp;
      tmp = AA11;
      AA11 = AA81;
      AA81 = tmp;
      tmp = AA12;
      AA12 = AA82;
      AA82 = tmp;
      tmp = AA13;
      AA13 = AA83;
      AA83 = tmp;
      tmp = AA14;
      AA14 = AA84;
      AA84 = tmp;
      tmp = AA15;
      AA15 = AA85;
      AA85 = tmp;
      tmp = AA16;
      AA16 = AA86;
      AA86 = tmp;
      tmp = AA17;
      AA17 = AA87;
      AA87 = tmp;
      tmp = AA18;
      AA18 = AA88;
      AA88 = tmp;
      tmp = AA19;
      AA19 = AA89;
      AA89 = tmp;
      /* update permutation vector based on row swap */
      i = perm1;
      perm1 = perm8;
      perm8 = i;
    }
    if (pvt == 9) {
      tmp = AA10;
      AA10 = AA90;
      AA90 = tmp;
      tmp = AA11;
      AA11 = AA91;
      AA91 = tmp;
      tmp = AA12;
      AA12 = AA92;
      AA92 = tmp;
      tmp = AA13;
      AA13 = AA93;
      AA93 = tmp;
      tmp = AA14;
      AA14 = AA94;
      AA94 = tmp;
      tmp = AA15;
      AA15 = AA95;
      AA95 = tmp;
      tmp = AA16;
      AA16 = AA96;
      AA96 = tmp;
      tmp = AA17;
      AA17 = AA97;
      AA97 = tmp;
      tmp = AA18;
      AA18 = AA98;
      AA98 = tmp;
      tmp = AA19;
      AA19 = AA99;
      AA99 = tmp;
      /* update permutation vector based on row swap */
      i = perm1;
      perm1 = perm9;
      perm9 = i;
    }
#endif // USE_PIVOTING

    /* scale current row */
    tmp = rcpOp(AA11);
    icol1 = perm1;
    AA10 = mulOp(tmp, AA10);
    AA11 = tmp;
    AA12 = mulOp(tmp, AA12);
    AA13 = mulOp(tmp, AA13);
    AA14 = mulOp(tmp, AA14);
    AA15 = mulOp(tmp, AA15);
    AA16 = mulOp(tmp, AA16);
    AA17 = mulOp(tmp, AA17);
    AA18 = mulOp(tmp, AA18);
    AA19 = mulOp(tmp, AA19);

    /* eliminate above and below current row */
    tmp = AA01;
    AA00 = fmnaOp(tmp, AA10, AA00);
    AA01 = mulOp(negOp(tmp), AA11);
    AA02 = fmnaOp(tmp, AA12, AA02);
    AA03 = fmnaOp(tmp, AA13, AA03);
    AA04 = fmnaOp(tmp, AA14, AA04);
    AA05 = fmnaOp(tmp, AA15, AA05);
    AA06 = fmnaOp(tmp, AA16, AA06);
    AA07 = fmnaOp(tmp, AA17, AA07);
    AA08 = fmnaOp(tmp, AA18, AA08);
    AA09 = fmnaOp(tmp, AA19, AA09);

    tmp = AA21;
    AA20 = fmnaOp(tmp, AA10, AA20);
    AA21 = mulOp(negOp(tmp), AA11);
    AA22 = fmnaOp(tmp, AA12, AA22);
    AA23 = fmnaOp(tmp, AA13, AA23);
    AA24 = fmnaOp(tmp, AA14, AA24);
    AA25 = fmnaOp(tmp, AA15, AA25);
    AA26 = fmnaOp(tmp, AA16, AA26);
    AA27 = fmnaOp(tmp, AA17, AA27);
    AA28 = fmnaOp(tmp, AA18, AA28);
    AA29 = fmnaOp(tmp, AA19, AA29);

    tmp = AA31;
    AA30 = fmnaOp(tmp, AA10, AA30);
    AA31 = mulOp(negOp(tmp), AA11);
    AA32 = fmnaOp(tmp, AA12, AA32);
    AA33 = fmnaOp(tmp, AA13, AA33);
    AA34 = fmnaOp(tmp, AA14, AA34);
    AA35 = fmnaOp(tmp, AA15, AA35);
    AA36 = fmnaOp(tmp, AA16, AA36);
    AA37 = fmnaOp(tmp, AA17, AA37);
    AA38 = fmnaOp(tmp, AA18, AA38);
    AA39 = fmnaOp(tmp, AA19, AA39);

    tmp = AA41;
    AA40 = fmnaOp(tmp, AA10, AA40);
    AA41 = mulOp(negOp(tmp), AA11);
    AA42 = fmnaOp(tmp, AA12, AA42);
    AA43 = fmnaOp(tmp, AA13, AA43);
    AA44 = fmnaOp(tmp, AA14, AA44);
    AA45 = fmnaOp(tmp, AA15, AA45);
    AA46 = fmnaOp(tmp, AA16, AA46);
    AA47 = fmnaOp(tmp, AA17, AA47);
    AA48 = fmnaOp(tmp, AA18, AA48);
    AA49 = fmnaOp(tmp, AA19, AA49);

    tmp = AA51;
    AA50 = fmnaOp(tmp, AA10, AA50);
    AA51 = mulOp(negOp(tmp), AA11);
    AA52 = fmnaOp(tmp, AA12, AA52);
    AA53 = fmnaOp(tmp, AA13, AA53);
    AA54 = fmnaOp(tmp, AA14, AA54);
    AA55 = fmnaOp(tmp, AA15, AA55);
    AA56 = fmnaOp(tmp, AA16, AA56);
    AA57 = fmnaOp(tmp, AA17, AA57);
    AA58 = fmnaOp(tmp, AA18, AA58);
    AA59 = fmnaOp(tmp, AA19, AA59);

    tmp = AA61;
    AA60 = fmnaOp(tmp, AA10, AA60);
    AA61 = mulOp(negOp(tmp), AA11);
    AA62 = fmnaOp(tmp, AA12, AA62);
    AA63 = fmnaOp(tmp, AA13, AA63);
    AA64 = fmnaOp(tmp, AA14, AA64);
    AA65 = fmnaOp(tmp, AA15, AA65);
    AA66 = fmnaOp(tmp, AA16, AA66);
    AA67 = fmnaOp(tmp, AA17, AA67);
    AA68 = fmnaOp(tmp, AA18, AA68);
    AA69 = fmnaOp(tmp, AA19, AA69);

    tmp = AA71;
    AA70 = fmnaOp(tmp, AA10, AA70);
    AA71 = mulOp(negOp(tmp), AA11);
    AA72 = fmnaOp(tmp, AA12, AA72);
    AA73 = fmnaOp(tmp, AA13, AA73);
    AA74 = fmnaOp(tmp, AA14, AA74);
    AA75 = fmnaOp(tmp, AA15, AA75);
    AA76 = fmnaOp(tmp, AA16, AA76);
    AA77 = fmnaOp(tmp, AA17, AA77);
    AA78 = fmnaOp(tmp, AA18, AA78);
    AA79 = fmnaOp(tmp, AA19, AA79);

    tmp = AA81;
    AA80 = fmnaOp(tmp, AA10, AA80);
    AA81 = mulOp(negOp(tmp), AA11);
    AA82 = fmnaOp(tmp, AA12, AA82);
    AA83 = fmnaOp(tmp, AA13, AA83);
    AA84 = fmnaOp(tmp, AA14, AA84);
    AA85 = fmnaOp(tmp, AA15, AA85);
    AA86 = fmnaOp(tmp, AA16, AA86);
    AA87 = fmnaOp(tmp, AA17, AA87);
    AA88 = fmnaOp(tmp, AA18, AA88);
    AA89 = fmnaOp(tmp, AA19, AA89);

    tmp = AA91;
    AA90 = fmnaOp(tmp, AA10, AA90);
    AA91 = mulOp(negOp(tmp), AA11);
    AA92 = fmnaOp(tmp, AA12, AA92);
    AA93 = fmnaOp(tmp, AA13, AA93);
    AA94 = fmnaOp(tmp, AA14, AA94);
    AA95 = fmnaOp(tmp, AA15, AA95);
    AA96 = fmnaOp(tmp, AA16, AA96);
    AA97 = fmnaOp(tmp, AA17, AA97);
    AA98 = fmnaOp(tmp, AA18, AA98);
    AA99 = fmnaOp(tmp, AA19, AA99);

    /****************** iteration 2 ****************/

#if USE_PIVOTING
    /* search pivot row */
    p = absOp(AA22);
    pvt = 2;
    t = absOp(AA32);
    if (t > p) {
      p = t;
      pvt = 3;
    }
    t = absOp(AA42);
    if (t > p) {
      p = t;
      pvt = 4;
    }
    t = absOp(AA52);
    if (t > p) {
      p = t;
      pvt = 5;
    }
    t = absOp(AA62);
    if (t > p) {
      p = t;
      pvt = 6;
    }
    t = absOp(AA72);
    if (t > p) {
      p = t;
      pvt = 7;
    }
    t = absOp(AA82);
    if (t > p) {
      p = t;
      pvt = 8;
    }
    t = absOp(AA92);
    if (t > p) {
      p = t;
      pvt = 9;
    }

    /* swap pivot row with row 2 */
    if (pvt == 3) {
      tmp = AA20;
      AA20 = AA30;
      AA30 = tmp;
      tmp = AA21;
      AA21 = AA31;
      AA31 = tmp;
      tmp = AA22;
      AA22 = AA32;
      AA32 = tmp;
      tmp = AA23;
      AA23 = AA33;
      AA33 = tmp;
      tmp = AA24;
      AA24 = AA34;
      AA34 = tmp;
      tmp = AA25;
      AA25 = AA35;
      AA35 = tmp;
      tmp = AA26;
      AA26 = AA36;
      AA36 = tmp;
      tmp = AA27;
      AA27 = AA37;
      AA37 = tmp;
      tmp = AA28;
      AA28 = AA38;
      AA38 = tmp;
      tmp = AA29;
      AA29 = AA39;
      AA39 = tmp;
      /* update permutation vector based on row swap */
      i = perm2;
      perm2 = perm3;
      perm3 = i;
    }
    if (pvt == 4) {
      tmp = AA20;
      AA20 = AA40;
      AA40 = tmp;
      tmp = AA21;
      AA21 = AA41;
      AA41 = tmp;
      tmp = AA22;
      AA22 = AA42;
      AA42 = tmp;
      tmp = AA23;
      AA23 = AA43;
      AA43 = tmp;
      tmp = AA24;
      AA24 = AA44;
      AA44 = tmp;
      tmp = AA25;
      AA25 = AA45;
      AA45 = tmp;
      tmp = AA26;
      AA26 = AA46;
      AA46 = tmp;
      tmp = AA27;
      AA27 = AA47;
      AA47 = tmp;
      tmp = AA28;
      AA28 = AA48;
      AA48 = tmp;
      tmp = AA29;
      AA29 = AA49;
      AA49 = tmp;
      /* update permutation vector based on row swap */
      i = perm2;
      perm2 = perm4;
      perm4 = i;
    }
    if (pvt == 5) {
      tmp = AA20;
      AA20 = AA50;
      AA50 = tmp;
      tmp = AA21;
      AA21 = AA51;
      AA51 = tmp;
      tmp = AA22;
      AA22 = AA52;
      AA52 = tmp;
      tmp = AA23;
      AA23 = AA53;
      AA53 = tmp;
      tmp = AA24;
      AA24 = AA54;
      AA54 = tmp;
      tmp = AA25;
      AA25 = AA55;
      AA55 = tmp;
      tmp = AA26;
      AA26 = AA56;
      AA56 = tmp;
      tmp = AA27;
      AA27 = AA57;
      AA57 = tmp;
      tmp = AA28;
      AA28 = AA58;
      AA58 = tmp;
      tmp = AA29;
      AA29 = AA59;
      AA59 = tmp;
      /* update permutation vector based on row swap */
      i = perm2;
      perm2 = perm5;
      perm5 = i;
    }
    if (pvt == 6) {
      tmp = AA20;
      AA20 = AA60;
      AA60 = tmp;
      tmp = AA21;
      AA21 = AA61;
      AA61 = tmp;
      tmp = AA22;
      AA22 = AA62;
      AA62 = tmp;
      tmp = AA23;
      AA23 = AA63;
      AA63 = tmp;
      tmp = AA24;
      AA24 = AA64;
      AA64 = tmp;
      tmp = AA25;
      AA25 = AA65;
      AA65 = tmp;
      tmp = AA26;
      AA26 = AA66;
      AA66 = tmp;
      tmp = AA27;
      AA27 = AA67;
      AA67 = tmp;
      tmp = AA28;
      AA28 = AA68;
      AA68 = tmp;
      tmp = AA29;
      AA29 = AA69;
      AA69 = tmp;
      /* update permutation vector based on row swap */
      i = perm2;
      perm2 = perm6;
      perm6 = i;
    }
    if (pvt == 7) {
      tmp = AA20;
      AA20 = AA70;
      AA70 = tmp;
      tmp = AA21;
      AA21 = AA71;
      AA71 = tmp;
      tmp = AA22;
      AA22 = AA72;
      AA72 = tmp;
      tmp = AA23;
      AA23 = AA73;
      AA73 = tmp;
      tmp = AA24;
      AA24 = AA74;
      AA74 = tmp;
      tmp = AA25;
      AA25 = AA75;
      AA75 = tmp;
      tmp = AA26;
      AA26 = AA76;
      AA76 = tmp;
      tmp = AA27;
      AA27 = AA77;
      AA77 = tmp;
      tmp = AA28;
      AA28 = AA78;
      AA78 = tmp;
      tmp = AA29;
      AA29 = AA79;
      AA79 = tmp;
      /* update permutation vector based on row swap */
      i = perm2;
      perm2 = perm7;
      perm7 = i;
    }
    if (pvt == 8) {
      tmp = AA20;
      AA20 = AA80;
      AA80 = tmp;
      tmp = AA21;
      AA21 = AA81;
      AA81 = tmp;
      tmp = AA22;
      AA22 = AA82;
      AA82 = tmp;
      tmp = AA23;
      AA23 = AA83;
      AA83 = tmp;
      tmp = AA24;
      AA24 = AA84;
      AA84 = tmp;
      tmp = AA25;
      AA25 = AA85;
      AA85 = tmp;
      tmp = AA26;
      AA26 = AA86;
      AA86 = tmp;
      tmp = AA27;
      AA27 = AA87;
      AA87 = tmp;
      tmp = AA28;
      AA28 = AA88;
      AA88 = tmp;
      tmp = AA29;
      AA29 = AA89;
      AA89 = tmp;
      /* update permutation vector based on row swap */
      i = perm2;
      perm2 = perm8;
      perm8 = i;
    }
    if (pvt == 9) {
      tmp = AA20;
      AA20 = AA90;
      AA90 = tmp;
      tmp = AA21;
      AA21 = AA91;
      AA91 = tmp;
      tmp = AA22;
      AA22 = AA92;
      AA92 = tmp;
      tmp = AA23;
      AA23 = AA93;
      AA93 = tmp;
      tmp = AA24;
      AA24 = AA94;
      AA94 = tmp;
      tmp = AA25;
      AA25 = AA95;
      AA95 = tmp;
      tmp = AA26;
      AA26 = AA96;
      AA96 = tmp;
      tmp = AA27;
      AA27 = AA97;
      AA97 = tmp;
      tmp = AA28;
      AA28 = AA98;
      AA98 = tmp;
      tmp = AA29;
      AA29 = AA99;
      AA99 = tmp;
      /* update permutation vector based on row swap */
      i = perm2;
      perm2 = perm9;
      perm9 = i;
    }
#endif // USE_PIVOTING

    /* scale current row */
    tmp = rcpOp(AA22);
    icol2 = perm2;
    AA20 = mulOp(tmp, AA20);
    AA21 = mulOp(tmp, AA21);
    AA22 = tmp;
    AA23 = mulOp(tmp, AA23);
    AA24 = mulOp(tmp, AA24);
    AA25 = mulOp(tmp, AA25);
    AA26 = mulOp(tmp, AA26);
    AA27 = mulOp(tmp, AA27);
    AA28 = mulOp(tmp, AA28);
    AA29 = mulOp(tmp, AA29);

    /* eliminate above and below current row */
    tmp = AA02;
    AA00 = fmnaOp(tmp, AA20, AA00);
    AA01 = fmnaOp(tmp, AA21, AA01);
    AA02 = mulOp(negOp(tmp), AA22);
    AA03 = fmnaOp(tmp, AA23, AA03);
    AA04 = fmnaOp(tmp, AA24, AA04);
    AA05 = fmnaOp(tmp, AA25, AA05);
    AA06 = fmnaOp(tmp, AA26, AA06);
    AA07 = fmnaOp(tmp, AA27, AA07);
    AA08 = fmnaOp(tmp, AA28, AA08);
    AA09 = fmnaOp(tmp, AA29, AA09);

    tmp = AA12;
    AA10 = fmnaOp(tmp, AA20, AA10);
    AA11 = fmnaOp(tmp, AA21, AA11);
    AA12 = mulOp(negOp(tmp), AA22);
    AA13 = fmnaOp(tmp, AA23, AA13);
    AA14 = fmnaOp(tmp, AA24, AA14);
    AA15 = fmnaOp(tmp, AA25, AA15);
    AA16 = fmnaOp(tmp, AA26, AA16);
    AA17 = fmnaOp(tmp, AA27, AA17);
    AA18 = fmnaOp(tmp, AA28, AA18);
    AA19 = fmnaOp(tmp, AA29, AA19);

    tmp = AA32;
    AA30 = fmnaOp(tmp, AA20, AA30);
    AA31 = fmnaOp(tmp, AA21, AA31);
    AA32 = mulOp(negOp(tmp), AA22);
    AA33 = fmnaOp(tmp, AA23, AA33);
    AA34 = fmnaOp(tmp, AA24, AA34);
    AA35 = fmnaOp(tmp, AA25, AA35);
    AA36 = fmnaOp(tmp, AA26, AA36);
    AA37 = fmnaOp(tmp, AA27, AA37);
    AA38 = fmnaOp(tmp, AA28, AA38);
    AA39 = fmnaOp(tmp, AA29, AA39);

    tmp = AA42;
    AA40 = fmnaOp(tmp, AA20, AA40);
    AA41 = fmnaOp(tmp, AA21, AA41);
    AA42 = mulOp(negOp(tmp), AA22);
    AA43 = fmnaOp(tmp, AA23, AA43);
    AA44 = fmnaOp(tmp, AA24, AA44);
    AA45 = fmnaOp(tmp, AA25, AA45);
    AA46 = fmnaOp(tmp, AA26, AA46);
    AA47 = fmnaOp(tmp, AA27, AA47);
    AA48 = fmnaOp(tmp, AA28, AA48);
    AA49 = fmnaOp(tmp, AA29, AA49);

    tmp = AA52;
    AA50 = fmnaOp(tmp, AA20, AA50);
    AA51 = fmnaOp(tmp, AA21, AA51);
    AA52 = mulOp(negOp(tmp), AA22);
    AA53 = fmnaOp(tmp, AA23, AA53);
    AA54 = fmnaOp(tmp, AA24, AA54);
    AA55 = fmnaOp(tmp, AA25, AA55);
    AA56 = fmnaOp(tmp, AA26, AA56);
    AA57 = fmnaOp(tmp, AA27, AA57);
    AA58 = fmnaOp(tmp, AA28, AA58);
    AA59 = fmnaOp(tmp, AA29, AA59);

    tmp = AA62;
    AA60 = fmnaOp(tmp, AA20, AA60);
    AA61 = fmnaOp(tmp, AA21, AA61);
    AA62 = mulOp(negOp(tmp), AA22);
    AA63 = fmnaOp(tmp, AA23, AA63);
    AA64 = fmnaOp(tmp, AA24, AA64);
    AA65 = fmnaOp(tmp, AA25, AA65);
    AA66 = fmnaOp(tmp, AA26, AA66);
    AA67 = fmnaOp(tmp, AA27, AA67);
    AA68 = fmnaOp(tmp, AA28, AA68);
    AA69 = fmnaOp(tmp, AA29, AA69);

    tmp = AA72;
    AA70 = fmnaOp(tmp, AA20, AA70);
    AA71 = fmnaOp(tmp, AA21, AA71);
    AA72 = mulOp(negOp(tmp), AA22);
    AA73 = fmnaOp(tmp, AA23, AA73);
    AA74 = fmnaOp(tmp, AA24, AA74);
    AA75 = fmnaOp(tmp, AA25, AA75);
    AA76 = fmnaOp(tmp, AA26, AA76);
    AA77 = fmnaOp(tmp, AA27, AA77);
    AA78 = fmnaOp(tmp, AA28, AA78);
    AA79 = fmnaOp(tmp, AA29, AA79);

    tmp = AA82;
    AA80 = fmnaOp(tmp, AA20, AA80);
    AA81 = fmnaOp(tmp, AA21, AA81);
    AA82 = mulOp(negOp(tmp), AA22);
    AA83 = fmnaOp(tmp, AA23, AA83);
    AA84 = fmnaOp(tmp, AA24, AA84);
    AA85 = fmnaOp(tmp, AA25, AA85);
    AA86 = fmnaOp(tmp, AA26, AA86);
    AA87 = fmnaOp(tmp, AA27, AA87);
    AA88 = fmnaOp(tmp, AA28, AA88);
    AA89 = fmnaOp(tmp, AA29, AA89);

    tmp = AA92;
    AA90 = fmnaOp(tmp, AA20, AA90);
    AA91 = fmnaOp(tmp, AA21, AA91);
    AA92 = mulOp(negOp(tmp), AA22);
    AA93 = fmnaOp(tmp, AA23, AA93);
    AA94 = fmnaOp(tmp, AA24, AA94);
    AA95 = fmnaOp(tmp, AA25, AA95);
    AA96 = fmnaOp(tmp, AA26, AA96);
    AA97 = fmnaOp(tmp, AA27, AA97);
    AA98 = fmnaOp(tmp, AA28, AA98);
    AA99 = fmnaOp(tmp, AA29, AA99);

    /****************** iteration 3 ****************/

#if USE_PIVOTING
    /* search pivot row */
    p = absOp(AA33);
    pvt = 3;
    t = absOp(AA43);
    if (t > p) {
      p = t;
      pvt = 4;
    }
    t = absOp(AA53);
    if (t > p) {
      p = t;
      pvt = 5;
    }
    t = absOp(AA63);
    if (t > p) {
      p = t;
      pvt = 6;
    }
    t = absOp(AA73);
    if (t > p) {
      p = t;
      pvt = 7;
    }
    t = absOp(AA83);
    if (t > p) {
      p = t;
      pvt = 8;
    }
    t = absOp(AA93);
    if (t > p) {
      p = t;
      pvt = 9;
    }

    /* swap pivot row with row 3 */
    if (pvt == 4) {
      tmp = AA30;
      AA30 = AA40;
      AA40 = tmp;
      tmp = AA31;
      AA31 = AA41;
      AA41 = tmp;
      tmp = AA32;
      AA32 = AA42;
      AA42 = tmp;
      tmp = AA33;
      AA33 = AA43;
      AA43 = tmp;
      tmp = AA34;
      AA34 = AA44;
      AA44 = tmp;
      tmp = AA35;
      AA35 = AA45;
      AA45 = tmp;
      tmp = AA36;
      AA36 = AA46;
      AA46 = tmp;
      tmp = AA37;
      AA37 = AA47;
      AA47 = tmp;
      tmp = AA38;
      AA38 = AA48;
      AA48 = tmp;
      tmp = AA39;
      AA39 = AA49;
      AA49 = tmp;
      /* update permutation vector based on row swap */
      i = perm3;
      perm3 = perm4;
      perm4 = i;
    }
    if (pvt == 5) {
      tmp = AA30;
      AA30 = AA50;
      AA50 = tmp;
      tmp = AA31;
      AA31 = AA51;
      AA51 = tmp;
      tmp = AA32;
      AA32 = AA52;
      AA52 = tmp;
      tmp = AA33;
      AA33 = AA53;
      AA53 = tmp;
      tmp = AA34;
      AA34 = AA54;
      AA54 = tmp;
      tmp = AA35;
      AA35 = AA55;
      AA55 = tmp;
      tmp = AA36;
      AA36 = AA56;
      AA56 = tmp;
      tmp = AA37;
      AA37 = AA57;
      AA57 = tmp;
      tmp = AA38;
      AA38 = AA58;
      AA58 = tmp;
      tmp = AA39;
      AA39 = AA59;
      AA59 = tmp;
      /* update permutation vector based on row swap */
      i = perm3;
      perm3 = perm5;
      perm5 = i;
    }
    if (pvt == 6) {
      tmp = AA30;
      AA30 = AA60;
      AA60 = tmp;
      tmp = AA31;
      AA31 = AA61;
      AA61 = tmp;
      tmp = AA32;
      AA32 = AA62;
      AA62 = tmp;
      tmp = AA33;
      AA33 = AA63;
      AA63 = tmp;
      tmp = AA34;
      AA34 = AA64;
      AA64 = tmp;
      tmp = AA35;
      AA35 = AA65;
      AA65 = tmp;
      tmp = AA36;
      AA36 = AA66;
      AA66 = tmp;
      tmp = AA37;
      AA37 = AA67;
      AA67 = tmp;
      tmp = AA38;
      AA38 = AA68;
      AA68 = tmp;
      tmp = AA39;
      AA39 = AA69;
      AA69 = tmp;
      /* update permutation vector based on row swap */
      i = perm3;
      perm3 = perm6;
      perm6 = i;
    }
    if (pvt == 7) {
      tmp = AA30;
      AA30 = AA70;
      AA70 = tmp;
      tmp = AA31;
      AA31 = AA71;
      AA71 = tmp;
      tmp = AA32;
      AA32 = AA72;
      AA72 = tmp;
      tmp = AA33;
      AA33 = AA73;
      AA73 = tmp;
      tmp = AA34;
      AA34 = AA74;
      AA74 = tmp;
      tmp = AA35;
      AA35 = AA75;
      AA75 = tmp;
      tmp = AA36;
      AA36 = AA76;
      AA76 = tmp;
      tmp = AA37;
      AA37 = AA77;
      AA77 = tmp;
      tmp = AA38;
      AA38 = AA78;
      AA78 = tmp;
      tmp = AA39;
      AA39 = AA79;
      AA79 = tmp;
      /* update permutation vector based on row swap */
      i = perm3;
      perm3 = perm7;
      perm7 = i;
    }
    if (pvt == 8) {
      tmp = AA30;
      AA30 = AA80;
      AA80 = tmp;
      tmp = AA31;
      AA31 = AA81;
      AA81 = tmp;
      tmp = AA32;
      AA32 = AA82;
      AA82 = tmp;
      tmp = AA33;
      AA33 = AA83;
      AA83 = tmp;
      tmp = AA34;
      AA34 = AA84;
      AA84 = tmp;
      tmp = AA35;
      AA35 = AA85;
      AA85 = tmp;
      tmp = AA36;
      AA36 = AA86;
      AA86 = tmp;
      tmp = AA37;
      AA37 = AA87;
      AA87 = tmp;
      tmp = AA38;
      AA38 = AA88;
      AA88 = tmp;
      tmp = AA39;
      AA39 = AA89;
      AA89 = tmp;
      /* update permutation vector based on row swap */
      i = perm3;
      perm3 = perm8;
      perm8 = i;
    }
    if (pvt == 9) {
      tmp = AA30;
      AA30 = AA90;
      AA90 = tmp;
      tmp = AA31;
      AA31 = AA91;
      AA91 = tmp;
      tmp = AA32;
      AA32 = AA92;
      AA92 = tmp;
      tmp = AA33;
      AA33 = AA93;
      AA93 = tmp;
      tmp = AA34;
      AA34 = AA94;
      AA94 = tmp;
      tmp = AA35;
      AA35 = AA95;
      AA95 = tmp;
      tmp = AA36;
      AA36 = AA96;
      AA96 = tmp;
      tmp = AA37;
      AA37 = AA97;
      AA97 = tmp;
      tmp = AA38;
      AA38 = AA98;
      AA98 = tmp;
      tmp = AA39;
      AA39 = AA99;
      AA99 = tmp;
      /* update permutation vector based on row swap */
      i = perm3;
      perm3 = perm9;
      perm9 = i;
    }
#endif // USE_PIVOTING

    /* scale current row */
    tmp = rcpOp(AA33);
    icol3 = perm3;
    AA30 = mulOp(tmp, AA30);
    AA31 = mulOp(tmp, AA31);
    AA32 = mulOp(tmp, AA32);
    AA33 = tmp;
    AA34 = mulOp(tmp, AA34);
    AA35 = mulOp(tmp, AA35);
    AA36 = mulOp(tmp, AA36);
    AA37 = mulOp(tmp, AA37);
    AA38 = mulOp(tmp, AA38);
    AA39 = mulOp(tmp, AA39);

    /* eliminate above and below current row */
    tmp = AA03;
    AA00 = fmnaOp(tmp, AA30, AA00);
    AA01 = fmnaOp(tmp, AA31, AA01);
    AA02 = fmnaOp(tmp, AA32, AA02);
    AA03 = mulOp(negOp(tmp), AA33);
    AA04 = fmnaOp(tmp, AA34, AA04);
    AA05 = fmnaOp(tmp, AA35, AA05);
    AA06 = fmnaOp(tmp, AA36, AA06);
    AA07 = fmnaOp(tmp, AA37, AA07);
    AA08 = fmnaOp(tmp, AA38, AA08);
    AA09 = fmnaOp(tmp, AA39, AA09);

    tmp = AA13;
    AA10 = fmnaOp(tmp, AA30, AA10);
    AA11 = fmnaOp(tmp, AA31, AA11);
    AA12 = fmnaOp(tmp, AA32, AA12);
    AA13 = mulOp(negOp(tmp), AA33);
    AA14 = fmnaOp(tmp, AA34, AA14);
    AA15 = fmnaOp(tmp, AA35, AA15);
    AA16 = fmnaOp(tmp, AA36, AA16);
    AA17 = fmnaOp(tmp, AA37, AA17);
    AA18 = fmnaOp(tmp, AA38, AA18);
    AA19 = fmnaOp(tmp, AA39, AA19);

    tmp = AA23;
    AA20 = fmnaOp(tmp, AA30, AA20);
    AA21 = fmnaOp(tmp, AA31, AA21);
    AA22 = fmnaOp(tmp, AA32, AA22);
    AA23 = mulOp(negOp(tmp), AA33);
    AA24 = fmnaOp(tmp, AA34, AA24);
    AA25 = fmnaOp(tmp, AA35, AA25);
    AA26 = fmnaOp(tmp, AA36, AA26);
    AA27 = fmnaOp(tmp, AA37, AA27);
    AA28 = fmnaOp(tmp, AA38, AA28);
    AA29 = fmnaOp(tmp, AA39, AA29);

    tmp = AA43;
    AA40 = fmnaOp(tmp, AA30, AA40);
    AA41 = fmnaOp(tmp, AA31, AA41);
    AA42 = fmnaOp(tmp, AA32, AA42);
    AA43 = mulOp(negOp(tmp), AA33);
    AA44 = fmnaOp(tmp, AA34, AA44);
    AA45 = fmnaOp(tmp, AA35, AA45);
    AA46 = fmnaOp(tmp, AA36, AA46);
    AA47 = fmnaOp(tmp, AA37, AA47);
    AA48 = fmnaOp(tmp, AA38, AA48);
    AA49 = fmnaOp(tmp, AA39, AA49);

    tmp = AA53;
    AA50 = fmnaOp(tmp, AA30, AA50);
    AA51 = fmnaOp(tmp, AA31, AA51);
    AA52 = fmnaOp(tmp, AA32, AA52);
    AA53 = mulOp(negOp(tmp), AA33);
    AA54 = fmnaOp(tmp, AA34, AA54);
    AA55 = fmnaOp(tmp, AA35, AA55);
    AA56 = fmnaOp(tmp, AA36, AA56);
    AA57 = fmnaOp(tmp, AA37, AA57);
    AA58 = fmnaOp(tmp, AA38, AA58);
    AA59 = fmnaOp(tmp, AA39, AA59);

    tmp = AA63;
    AA60 = fmnaOp(tmp, AA30, AA60);
    AA61 = fmnaOp(tmp, AA31, AA61);
    AA62 = fmnaOp(tmp, AA32, AA62);
    AA63 = mulOp(negOp(tmp), AA33);
    AA64 = fmnaOp(tmp, AA34, AA64);
    AA65 = fmnaOp(tmp, AA35, AA65);
    AA66 = fmnaOp(tmp, AA36, AA66);
    AA67 = fmnaOp(tmp, AA37, AA67);
    AA68 = fmnaOp(tmp, AA38, AA68);
    AA69 = fmnaOp(tmp, AA39, AA69);

    tmp = AA73;
    AA70 = fmnaOp(tmp, AA30, AA70);
    AA71 = fmnaOp(tmp, AA31, AA71);
    AA72 = fmnaOp(tmp, AA32, AA72);
    AA73 = mulOp(negOp(tmp), AA33);
    AA74 = fmnaOp(tmp, AA34, AA74);
    AA75 = fmnaOp(tmp, AA35, AA75);
    AA76 = fmnaOp(tmp, AA36, AA76);
    AA77 = fmnaOp(tmp, AA37, AA77);
    AA78 = fmnaOp(tmp, AA38, AA78);
    AA79 = fmnaOp(tmp, AA39, AA79);

    tmp = AA83;
    AA80 = fmnaOp(tmp, AA30, AA80);
    AA81 = fmnaOp(tmp, AA31, AA81);
    AA82 = fmnaOp(tmp, AA32, AA82);
    AA83 = mulOp(negOp(tmp), AA33);
    AA84 = fmnaOp(tmp, AA34, AA84);
    AA85 = fmnaOp(tmp, AA35, AA85);
    AA86 = fmnaOp(tmp, AA36, AA86);
    AA87 = fmnaOp(tmp, AA37, AA87);
    AA88 = fmnaOp(tmp, AA38, AA88);
    AA89 = fmnaOp(tmp, AA39, AA89);

    tmp = AA93;
    AA90 = fmnaOp(tmp, AA30, AA90);
    AA91 = fmnaOp(tmp, AA31, AA91);
    AA92 = fmnaOp(tmp, AA32, AA92);
    AA93 = mulOp(negOp(tmp), AA33);
    AA94 = fmnaOp(tmp, AA34, AA94);
    AA95 = fmnaOp(tmp, AA35, AA95);
    AA96 = fmnaOp(tmp, AA36, AA96);
    AA97 = fmnaOp(tmp, AA37, AA97);
    AA98 = fmnaOp(tmp, AA38, AA98);
    AA99 = fmnaOp(tmp, AA39, AA99);

    /****************** iteration 4 ****************/

#if USE_PIVOTING
    /* search pivot row */
    p = absOp(AA44);
    pvt = 4;
    t = absOp(AA54);
    if (t > p) {
      p = t;
      pvt = 5;
    }
    t = absOp(AA64);
    if (t > p) {
      p = t;
      pvt = 6;
    }
    t = absOp(AA74);
    if (t > p) {
      p = t;
      pvt = 7;
    }
    t = absOp(AA84);
    if (t > p) {
      p = t;
      pvt = 8;
    }
    t = absOp(AA94);
    if (t > p) {
      p = t;
      pvt = 9;
    }

    /* swap pivot row with row 4 */
    if (pvt == 5) {
      tmp = AA40;
      AA40 = AA50;
      AA50 = tmp;
      tmp = AA41;
      AA41 = AA51;
      AA51 = tmp;
      tmp = AA42;
      AA42 = AA52;
      AA52 = tmp;
      tmp = AA43;
      AA43 = AA53;
      AA53 = tmp;
      tmp = AA44;
      AA44 = AA54;
      AA54 = tmp;
      tmp = AA45;
      AA45 = AA55;
      AA55 = tmp;
      tmp = AA46;
      AA46 = AA56;
      AA56 = tmp;
      tmp = AA47;
      AA47 = AA57;
      AA57 = tmp;
      tmp = AA48;
      AA48 = AA58;
      AA58 = tmp;
      tmp = AA49;
      AA49 = AA59;
      AA59 = tmp;
      /* update permutation vector based on row swap */
      i = perm4;
      perm4 = perm5;
      perm5 = i;
    }
    if (pvt == 6) {
      tmp = AA40;
      AA40 = AA60;
      AA60 = tmp;
      tmp = AA41;
      AA41 = AA61;
      AA61 = tmp;
      tmp = AA42;
      AA42 = AA62;
      AA62 = tmp;
      tmp = AA43;
      AA43 = AA63;
      AA63 = tmp;
      tmp = AA44;
      AA44 = AA64;
      AA64 = tmp;
      tmp = AA45;
      AA45 = AA65;
      AA65 = tmp;
      tmp = AA46;
      AA46 = AA66;
      AA66 = tmp;
      tmp = AA47;
      AA47 = AA67;
      AA67 = tmp;
      tmp = AA48;
      AA48 = AA68;
      AA68 = tmp;
      tmp = AA49;
      AA49 = AA69;
      AA69 = tmp;
      /* update permutation vector based on row swap */
      i = perm4;
      perm4 = perm6;
      perm6 = i;
    }
    if (pvt == 7) {
      tmp = AA40;
      AA40 = AA70;
      AA70 = tmp;
      tmp = AA41;
      AA41 = AA71;
      AA71 = tmp;
      tmp = AA42;
      AA42 = AA72;
      AA72 = tmp;
      tmp = AA43;
      AA43 = AA73;
      AA73 = tmp;
      tmp = AA44;
      AA44 = AA74;
      AA74 = tmp;
      tmp = AA45;
      AA45 = AA75;
      AA75 = tmp;
      tmp = AA46;
      AA46 = AA76;
      AA76 = tmp;
      tmp = AA47;
      AA47 = AA77;
      AA77 = tmp;
      tmp = AA48;
      AA48 = AA78;
      AA78 = tmp;
      tmp = AA49;
      AA49 = AA79;
      AA79 = tmp;
      /* update permutation vector based on row swap */
      i = perm4;
      perm4 = perm7;
      perm7 = i;
    }
    if (pvt == 8) {
      tmp = AA40;
      AA40 = AA80;
      AA80 = tmp;
      tmp = AA41;
      AA41 = AA81;
      AA81 = tmp;
      tmp = AA42;
      AA42 = AA82;
      AA82 = tmp;
      tmp = AA43;
      AA43 = AA83;
      AA83 = tmp;
      tmp = AA44;
      AA44 = AA84;
      AA84 = tmp;
      tmp = AA45;
      AA45 = AA85;
      AA85 = tmp;
      tmp = AA46;
      AA46 = AA86;
      AA86 = tmp;
      tmp = AA47;
      AA47 = AA87;
      AA87 = tmp;
      tmp = AA48;
      AA48 = AA88;
      AA88 = tmp;
      tmp = AA49;
      AA49 = AA89;
      AA89 = tmp;
      /* update permutation vector based on row swap */
      i = perm4;
      perm4 = perm8;
      perm8 = i;
    }
    if (pvt == 9) {
      tmp = AA40;
      AA40 = AA90;
      AA90 = tmp;
      tmp = AA41;
      AA41 = AA91;
      AA91 = tmp;
      tmp = AA42;
      AA42 = AA92;
      AA92 = tmp;
      tmp = AA43;
      AA43 = AA93;
      AA93 = tmp;
      tmp = AA44;
      AA44 = AA94;
      AA94 = tmp;
      tmp = AA45;
      AA45 = AA95;
      AA95 = tmp;
      tmp = AA46;
      AA46 = AA96;
      AA96 = tmp;
      tmp = AA47;
      AA47 = AA97;
      AA97 = tmp;
      tmp = AA48;
      AA48 = AA98;
      AA98 = tmp;
      tmp = AA49;
      AA49 = AA99;
      AA99 = tmp;
      /* update permutation vector based on row swap */
      i = perm4;
      perm4 = perm9;
      perm9 = i;
    }
#endif // USE_PIVOTING

    /* scale current row */
    tmp = rcpOp(AA44);
    icol4 = perm4;
    AA40 = mulOp(tmp, AA40);
    AA41 = mulOp(tmp, AA41);
    AA42 = mulOp(tmp, AA42);
    AA43 = mulOp(tmp, AA43);
    AA44 = tmp;
    AA45 = mulOp(tmp, AA45);
    AA46 = mulOp(tmp, AA46);
    AA47 = mulOp(tmp, AA47);
    AA48 = mulOp(tmp, AA48);
    AA49 = mulOp(tmp, AA49);

    /* eliminate above and below current row */
    tmp = AA04;
    AA00 = fmnaOp(tmp, AA40, AA00);
    AA01 = fmnaOp(tmp, AA41, AA01);
    AA02 = fmnaOp(tmp, AA42, AA02);
    AA03 = fmnaOp(tmp, AA43, AA03);
    AA04 = mulOp(negOp(tmp), AA44);
    AA05 = fmnaOp(tmp, AA45, AA05);
    AA06 = fmnaOp(tmp, AA46, AA06);
    AA07 = fmnaOp(tmp, AA47, AA07);
    AA08 = fmnaOp(tmp, AA48, AA08);
    AA09 = fmnaOp(tmp, AA49, AA09);

    tmp = AA14;
    AA10 = fmnaOp(tmp, AA40, AA10);
    AA11 = fmnaOp(tmp, AA41, AA11);
    AA12 = fmnaOp(tmp, AA42, AA12);
    AA13 = fmnaOp(tmp, AA43, AA13);
    AA14 = mulOp(negOp(tmp), AA44);
    AA15 = fmnaOp(tmp, AA45, AA15);
    AA16 = fmnaOp(tmp, AA46, AA16);
    AA17 = fmnaOp(tmp, AA47, AA17);
    AA18 = fmnaOp(tmp, AA48, AA18);
    AA19 = fmnaOp(tmp, AA49, AA19);

    tmp = AA24;
    AA20 = fmnaOp(tmp, AA40, AA20);
    AA21 = fmnaOp(tmp, AA41, AA21);
    AA22 = fmnaOp(tmp, AA42, AA22);
    AA23 = fmnaOp(tmp, AA43, AA23);
    AA24 = mulOp(negOp(tmp), AA44);
    AA25 = fmnaOp(tmp, AA45, AA25);
    AA26 = fmnaOp(tmp, AA46, AA26);
    AA27 = fmnaOp(tmp, AA47, AA27);
    AA28 = fmnaOp(tmp, AA48, AA28);
    AA29 = fmnaOp(tmp, AA49, AA29);

    tmp = AA34;
    AA30 = fmnaOp(tmp, AA40, AA30);
    AA31 = fmnaOp(tmp, AA41, AA31);
    AA32 = fmnaOp(tmp, AA42, AA32);
    AA33 = fmnaOp(tmp, AA43, AA33);
    AA34 = mulOp(negOp(tmp), AA44);
    AA35 = fmnaOp(tmp, AA45, AA35);
    AA36 = fmnaOp(tmp, AA46, AA36);
    AA37 = fmnaOp(tmp, AA47, AA37);
    AA38 = fmnaOp(tmp, AA48, AA38);
    AA39 = fmnaOp(tmp, AA49, AA39);

    tmp = AA54;
    AA50 = fmnaOp(tmp, AA40, AA50);
    AA51 = fmnaOp(tmp, AA41, AA51);
    AA52 = fmnaOp(tmp, AA42, AA52);
    AA53 = fmnaOp(tmp, AA43, AA53);
    AA54 = mulOp(negOp(tmp), AA44);
    AA55 = fmnaOp(tmp, AA45, AA55);
    AA56 = fmnaOp(tmp, AA46, AA56);
    AA57 = fmnaOp(tmp, AA47, AA57);
    AA58 = fmnaOp(tmp, AA48, AA58);
    AA59 = fmnaOp(tmp, AA49, AA59);

    tmp = AA64;
    AA60 = fmnaOp(tmp, AA40, AA60);
    AA61 = fmnaOp(tmp, AA41, AA61);
    AA62 = fmnaOp(tmp, AA42, AA62);
    AA63 = fmnaOp(tmp, AA43, AA63);
    AA64 = mulOp(negOp(tmp), AA44);
    AA65 = fmnaOp(tmp, AA45, AA65);
    AA66 = fmnaOp(tmp, AA46, AA66);
    AA67 = fmnaOp(tmp, AA47, AA67);
    AA68 = fmnaOp(tmp, AA48, AA68);
    AA69 = fmnaOp(tmp, AA49, AA69);

    tmp = AA74;
    AA70 = fmnaOp(tmp, AA40, AA70);
    AA71 = fmnaOp(tmp, AA41, AA71);
    AA72 = fmnaOp(tmp, AA42, AA72);
    AA73 = fmnaOp(tmp, AA43, AA73);
    AA74 = mulOp(negOp(tmp), AA44);
    AA75 = fmnaOp(tmp, AA45, AA75);
    AA76 = fmnaOp(tmp, AA46, AA76);
    AA77 = fmnaOp(tmp, AA47, AA77);
    AA78 = fmnaOp(tmp, AA48, AA78);
    AA79 = fmnaOp(tmp, AA49, AA79);

    tmp = AA84;
    AA80 = fmnaOp(tmp, AA40, AA80);
    AA81 = fmnaOp(tmp, AA41, AA81);
    AA82 = fmnaOp(tmp, AA42, AA82);
    AA83 = fmnaOp(tmp, AA43, AA83);
    AA84 = mulOp(negOp(tmp), AA44);
    AA85 = fmnaOp(tmp, AA45, AA85);
    AA86 = fmnaOp(tmp, AA46, AA86);
    AA87 = fmnaOp(tmp, AA47, AA87);
    AA88 = fmnaOp(tmp, AA48, AA88);
    AA89 = fmnaOp(tmp, AA49, AA89);

    tmp = AA94;
    AA90 = fmnaOp(tmp, AA40, AA90);
    AA91 = fmnaOp(tmp, AA41, AA91);
    AA92 = fmnaOp(tmp, AA42, AA92);
    AA93 = fmnaOp(tmp, AA43, AA93);
    AA94 = mulOp(negOp(tmp), AA44);
    AA95 = fmnaOp(tmp, AA45, AA95);
    AA96 = fmnaOp(tmp, AA46, AA96);
    AA97 = fmnaOp(tmp, AA47, AA97);
    AA98 = fmnaOp(tmp, AA48, AA98);
    AA99 = fmnaOp(tmp, AA49, AA99);

    /****************** iteration 5 ****************/

#if USE_PIVOTING
    /* search pivot row */
    p = absOp(AA55);
    pvt = 5;
    t = absOp(AA65);
    if (t > p) {
      p = t;
      pvt = 6;
    }
    t = absOp(AA75);
    if (t > p) {
      p = t;
      pvt = 7;
    }
    t = absOp(AA85);
    if (t > p) {
      p = t;
      pvt = 8;
    }
    t = absOp(AA95);
    if (t > p) {
      p = t;
      pvt = 9;
    }

    /* swap pivot row with row 5 */
    if (pvt == 6) {
      tmp = AA50;
      AA50 = AA60;
      AA60 = tmp;
      tmp = AA51;
      AA51 = AA61;
      AA61 = tmp;
      tmp = AA52;
      AA52 = AA62;
      AA62 = tmp;
      tmp = AA53;
      AA53 = AA63;
      AA63 = tmp;
      tmp = AA54;
      AA54 = AA64;
      AA64 = tmp;
      tmp = AA55;
      AA55 = AA65;
      AA65 = tmp;
      tmp = AA56;
      AA56 = AA66;
      AA66 = tmp;
      tmp = AA57;
      AA57 = AA67;
      AA67 = tmp;
      tmp = AA58;
      AA58 = AA68;
      AA68 = tmp;
      tmp = AA59;
      AA59 = AA69;
      AA69 = tmp;
      /* update permutation vector based on row swap */
      i = perm5;
      perm5 = perm6;
      perm6 = i;
    }
    if (pvt == 7) {
      tmp = AA50;
      AA50 = AA70;
      AA70 = tmp;
      tmp = AA51;
      AA51 = AA71;
      AA71 = tmp;
      tmp = AA52;
      AA52 = AA72;
      AA72 = tmp;
      tmp = AA53;
      AA53 = AA73;
      AA73 = tmp;
      tmp = AA54;
      AA54 = AA74;
      AA74 = tmp;
      tmp = AA55;
      AA55 = AA75;
      AA75 = tmp;
      tmp = AA56;
      AA56 = AA76;
      AA76 = tmp;
      tmp = AA57;
      AA57 = AA77;
      AA77 = tmp;
      tmp = AA58;
      AA58 = AA78;
      AA78 = tmp;
      tmp = AA59;
      AA59 = AA79;
      AA79 = tmp;
      /* update permutation vector based on row swap */
      i = perm5;
      perm5 = perm7;
      perm7 = i;
    }
    if (pvt == 8) {
      tmp = AA50;
      AA50 = AA80;
      AA80 = tmp;
      tmp = AA51;
      AA51 = AA81;
      AA81 = tmp;
      tmp = AA52;
      AA52 = AA82;
      AA82 = tmp;
      tmp = AA53;
      AA53 = AA83;
      AA83 = tmp;
      tmp = AA54;
      AA54 = AA84;
      AA84 = tmp;
      tmp = AA55;
      AA55 = AA85;
      AA85 = tmp;
      tmp = AA56;
      AA56 = AA86;
      AA86 = tmp;
      tmp = AA57;
      AA57 = AA87;
      AA87 = tmp;
      tmp = AA58;
      AA58 = AA88;
      AA88 = tmp;
      tmp = AA59;
      AA59 = AA89;
      AA89 = tmp;
      /* update permutation vector based on row swap */
      i = perm5;
      perm5 = perm8;
      perm8 = i;
    }
    if (pvt == 9) {
      tmp = AA50;
      AA50 = AA90;
      AA90 = tmp;
      tmp = AA51;
      AA51 = AA91;
      AA91 = tmp;
      tmp = AA52;
      AA52 = AA92;
      AA92 = tmp;
      tmp = AA53;
      AA53 = AA93;
      AA93 = tmp;
      tmp = AA54;
      AA54 = AA94;
      AA94 = tmp;
      tmp = AA55;
      AA55 = AA95;
      AA95 = tmp;
      tmp = AA56;
      AA56 = AA96;
      AA96 = tmp;
      tmp = AA57;
      AA57 = AA97;
      AA97 = tmp;
      tmp = AA58;
      AA58 = AA98;
      AA98 = tmp;
      tmp = AA59;
      AA59 = AA99;
      AA99 = tmp;
      /* update permutation vector based on row swap */
      i = perm5;
      perm5 = perm9;
      perm9 = i;
    }
#endif // USE_PIVOTING

    /* scale current row */
    tmp = rcpOp(AA55);
    icol5 = perm5;
    AA50 = mulOp(tmp, AA50);
    AA51 = mulOp(tmp, AA51);
    AA52 = mulOp(tmp, AA52);
    AA53 = mulOp(tmp, AA53);
    AA54 = mulOp(tmp, AA54);
    AA55 = tmp;
    AA56 = mulOp(tmp, AA56);
    AA57 = mulOp(tmp, AA57);
    AA58 = mulOp(tmp, AA58);
    AA59 = mulOp(tmp, AA59);

    /* eliminate above and below current row */
    tmp = AA05;
    AA00 = fmnaOp(tmp, AA50, AA00);
    AA01 = fmnaOp(tmp, AA51, AA01);
    AA02 = fmnaOp(tmp, AA52, AA02);
    AA03 = fmnaOp(tmp, AA53, AA03);
    AA04 = fmnaOp(tmp, AA54, AA04);
    AA05 = mulOp(negOp(tmp), AA55);
    AA06 = fmnaOp(tmp, AA56, AA06);
    AA07 = fmnaOp(tmp, AA57, AA07);
    AA08 = fmnaOp(tmp, AA58, AA08);
    AA09 = fmnaOp(tmp, AA59, AA09);

    tmp = AA15;
    AA10 = fmnaOp(tmp, AA50, AA10);
    AA11 = fmnaOp(tmp, AA51, AA11);
    AA12 = fmnaOp(tmp, AA52, AA12);
    AA13 = fmnaOp(tmp, AA53, AA13);
    AA14 = fmnaOp(tmp, AA54, AA14);
    AA15 = mulOp(negOp(tmp), AA55);
    AA16 = fmnaOp(tmp, AA56, AA16);
    AA17 = fmnaOp(tmp, AA57, AA17);
    AA18 = fmnaOp(tmp, AA58, AA18);
    AA19 = fmnaOp(tmp, AA59, AA19);

    tmp = AA25;
    AA20 = fmnaOp(tmp, AA50, AA20);
    AA21 = fmnaOp(tmp, AA51, AA21);
    AA22 = fmnaOp(tmp, AA52, AA22);
    AA23 = fmnaOp(tmp, AA53, AA23);
    AA24 = fmnaOp(tmp, AA54, AA24);
    AA25 = mulOp(negOp(tmp), AA55);
    AA26 = fmnaOp(tmp, AA56, AA26);
    AA27 = fmnaOp(tmp, AA57, AA27);
    AA28 = fmnaOp(tmp, AA58, AA28);
    AA29 = fmnaOp(tmp, AA59, AA29);

    tmp = AA35;
    AA30 = fmnaOp(tmp, AA50, AA30);
    AA31 = fmnaOp(tmp, AA51, AA31);
    AA32 = fmnaOp(tmp, AA52, AA32);
    AA33 = fmnaOp(tmp, AA53, AA33);
    AA34 = fmnaOp(tmp, AA54, AA34);
    AA35 = mulOp(negOp(tmp), AA55);
    AA36 = fmnaOp(tmp, AA56, AA36);
    AA37 = fmnaOp(tmp, AA57, AA37);
    AA38 = fmnaOp(tmp, AA58, AA38);
    AA39 = fmnaOp(tmp, AA59, AA39);

    tmp = AA45;
    AA40 = fmnaOp(tmp, AA50, AA40);
    AA41 = fmnaOp(tmp, AA51, AA41);
    AA42 = fmnaOp(tmp, AA52, AA42);
    AA43 = fmnaOp(tmp, AA53, AA43);
    AA44 = fmnaOp(tmp, AA54, AA44);
    AA45 = mulOp(negOp(tmp), AA55);
    AA46 = fmnaOp(tmp, AA56, AA46);
    AA47 = fmnaOp(tmp, AA57, AA47);
    AA48 = fmnaOp(tmp, AA58, AA48);
    AA49 = fmnaOp(tmp, AA59, AA49);

    tmp = AA65;
    AA60 = fmnaOp(tmp, AA50, AA60);
    AA61 = fmnaOp(tmp, AA51, AA61);
    AA62 = fmnaOp(tmp, AA52, AA62);
    AA63 = fmnaOp(tmp, AA53, AA63);
    AA64 = fmnaOp(tmp, AA54, AA64);
    AA65 = mulOp(negOp(tmp), AA55);
    AA66 = fmnaOp(tmp, AA56, AA66);
    AA67 = fmnaOp(tmp, AA57, AA67);
    AA68 = fmnaOp(tmp, AA58, AA68);
    AA69 = fmnaOp(tmp, AA59, AA69);

    tmp = AA75;
    AA70 = fmnaOp(tmp, AA50, AA70);
    AA71 = fmnaOp(tmp, AA51, AA71);
    AA72 = fmnaOp(tmp, AA52, AA72);
    AA73 = fmnaOp(tmp, AA53, AA73);
    AA74 = fmnaOp(tmp, AA54, AA74);
    AA75 = mulOp(negOp(tmp), AA55);
    AA76 = fmnaOp(tmp, AA56, AA76);
    AA77 = fmnaOp(tmp, AA57, AA77);
    AA78 = fmnaOp(tmp, AA58, AA78);
    AA79 = fmnaOp(tmp, AA59, AA79);

    tmp = AA85;
    AA80 = fmnaOp(tmp, AA50, AA80);
    AA81 = fmnaOp(tmp, AA51, AA81);
    AA82 = fmnaOp(tmp, AA52, AA82);
    AA83 = fmnaOp(tmp, AA53, AA83);
    AA84 = fmnaOp(tmp, AA54, AA84);
    AA85 = mulOp(negOp(tmp), AA55);
    AA86 = fmnaOp(tmp, AA56, AA86);
    AA87 = fmnaOp(tmp, AA57, AA87);
    AA88 = fmnaOp(tmp, AA58, AA88);
    AA89 = fmnaOp(tmp, AA59, AA89);

    tmp = AA95;
    AA90 = fmnaOp(tmp, AA50, AA90);
    AA91 = fmnaOp(tmp, AA51, AA91);
    AA92 = fmnaOp(tmp, AA52, AA92);
    AA93 = fmnaOp(tmp, AA53, AA93);
    AA94 = fmnaOp(tmp, AA54, AA94);
    AA95 = mulOp(negOp(tmp), AA55);
    AA96 = fmnaOp(tmp, AA56, AA96);
    AA97 = fmnaOp(tmp, AA57, AA97);
    AA98 = fmnaOp(tmp, AA58, AA98);
    AA99 = fmnaOp(tmp, AA59, AA99);

    /****************** iteration 6 ****************/

#if USE_PIVOTING
    /* search pivot row */
    p = absOp(AA66);
    pvt = 6;
    t = absOp(AA76);
    if (t > p) {
      p = t;
      pvt = 7;
    }
    t = absOp(AA86);
    if (t > p) {
      p = t;
      pvt = 8;
    }
    t = absOp(AA96);
    if (t > p) {
      p = t;
      pvt = 9;
    }

    /* swap pivot row with row 6 */
    if (pvt == 7) {
      tmp = AA60;
      AA60 = AA70;
      AA70 = tmp;
      tmp = AA61;
      AA61 = AA71;
      AA71 = tmp;
      tmp = AA62;
      AA62 = AA72;
      AA72 = tmp;
      tmp = AA63;
      AA63 = AA73;
      AA73 = tmp;
      tmp = AA64;
      AA64 = AA74;
      AA74 = tmp;
      tmp = AA65;
      AA65 = AA75;
      AA75 = tmp;
      tmp = AA66;
      AA66 = AA76;
      AA76 = tmp;
      tmp = AA67;
      AA67 = AA77;
      AA77 = tmp;
      tmp = AA68;
      AA68 = AA78;
      AA78 = tmp;
      tmp = AA69;
      AA69 = AA79;
      AA79 = tmp;
      /* update permutation vector based on row swap */
      i = perm6;
      perm6 = perm7;
      perm7 = i;
    }
    if (pvt == 8) {
      tmp = AA60;
      AA60 = AA80;
      AA80 = tmp;
      tmp = AA61;
      AA61 = AA81;
      AA81 = tmp;
      tmp = AA62;
      AA62 = AA82;
      AA82 = tmp;
      tmp = AA63;
      AA63 = AA83;
      AA83 = tmp;
      tmp = AA64;
      AA64 = AA84;
      AA84 = tmp;
      tmp = AA65;
      AA65 = AA85;
      AA85 = tmp;
      tmp = AA66;
      AA66 = AA86;
      AA86 = tmp;
      tmp = AA67;
      AA67 = AA87;
      AA87 = tmp;
      tmp = AA68;
      AA68 = AA88;
      AA88 = tmp;
      tmp = AA69;
      AA69 = AA89;
      AA89 = tmp;
      /* update permutation vector based on row swap */
      i = perm6;
      perm6 = perm8;
      perm8 = i;
    }
    if (pvt == 9) {
      tmp = AA60;
      AA60 = AA90;
      AA90 = tmp;
      tmp = AA61;
      AA61 = AA91;
      AA91 = tmp;
      tmp = AA62;
      AA62 = AA92;
      AA92 = tmp;
      tmp = AA63;
      AA63 = AA93;
      AA93 = tmp;
      tmp = AA64;
      AA64 = AA94;
      AA94 = tmp;
      tmp = AA65;
      AA65 = AA95;
      AA95 = tmp;
      tmp = AA66;
      AA66 = AA96;
      AA96 = tmp;
      tmp = AA67;
      AA67 = AA97;
      AA97 = tmp;
      tmp = AA68;
      AA68 = AA98;
      AA98 = tmp;
      tmp = AA69;
      AA69 = AA99;
      AA99 = tmp;
      /* update permutation vector based on row swap */
      i = perm6;
      perm6 = perm9;
      perm9 = i;
    }
#endif // USE_PIVOTING

    /* scale current row */
    tmp = rcpOp(AA66);
    icol6 = perm6;
    AA60 = mulOp(tmp, AA60);
    AA61 = mulOp(tmp, AA61);
    AA62 = mulOp(tmp, AA62);
    AA63 = mulOp(tmp, AA63);
    AA64 = mulOp(tmp, AA64);
    AA65 = mulOp(tmp, AA65);
    AA66 = tmp;
    AA67 = mulOp(tmp, AA67);
    AA68 = mulOp(tmp, AA68);
    AA69 = mulOp(tmp, AA69);

    /* eliminate above and below current row */
    tmp = AA06;
    AA00 = fmnaOp(tmp, AA60, AA00);
    AA01 = fmnaOp(tmp, AA61, AA01);
    AA02 = fmnaOp(tmp, AA62, AA02);
    AA03 = fmnaOp(tmp, AA63, AA03);
    AA04 = fmnaOp(tmp, AA64, AA04);
    AA05 = fmnaOp(tmp, AA65, AA05);
    AA06 = mulOp(negOp(tmp), AA66);
    AA07 = fmnaOp(tmp, AA67, AA07);
    AA08 = fmnaOp(tmp, AA68, AA08);
    AA09 = fmnaOp(tmp, AA69, AA09);

    tmp = AA16;
    AA10 = fmnaOp(tmp, AA60, AA10);
    AA11 = fmnaOp(tmp, AA61, AA11);
    AA12 = fmnaOp(tmp, AA62, AA12);
    AA13 = fmnaOp(tmp, AA63, AA13);
    AA14 = fmnaOp(tmp, AA64, AA14);
    AA15 = fmnaOp(tmp, AA65, AA15);
    AA16 = mulOp(negOp(tmp), AA66);
    AA17 = fmnaOp(tmp, AA67, AA17);
    AA18 = fmnaOp(tmp, AA68, AA18);
    AA19 = fmnaOp(tmp, AA69, AA19);

    tmp = AA26;
    AA20 = fmnaOp(tmp, AA60, AA20);
    AA21 = fmnaOp(tmp, AA61, AA21);
    AA22 = fmnaOp(tmp, AA62, AA22);
    AA23 = fmnaOp(tmp, AA63, AA23);
    AA24 = fmnaOp(tmp, AA64, AA24);
    AA25 = fmnaOp(tmp, AA65, AA25);
    AA26 = mulOp(negOp(tmp), AA66);
    AA27 = fmnaOp(tmp, AA67, AA27);
    AA28 = fmnaOp(tmp, AA68, AA28);
    AA29 = fmnaOp(tmp, AA69, AA29);

    tmp = AA36;
    AA30 = fmnaOp(tmp, AA60, AA30);
    AA31 = fmnaOp(tmp, AA61, AA31);
    AA32 = fmnaOp(tmp, AA62, AA32);
    AA33 = fmnaOp(tmp, AA63, AA33);
    AA34 = fmnaOp(tmp, AA64, AA34);
    AA35 = fmnaOp(tmp, AA65, AA35);
    AA36 = mulOp(negOp(tmp), AA66);
    AA37 = fmnaOp(tmp, AA67, AA37);
    AA38 = fmnaOp(tmp, AA68, AA38);
    AA39 = fmnaOp(tmp, AA69, AA39);

    tmp = AA46;
    AA40 = fmnaOp(tmp, AA60, AA40);
    AA41 = fmnaOp(tmp, AA61, AA41);
    AA42 = fmnaOp(tmp, AA62, AA42);
    AA43 = fmnaOp(tmp, AA63, AA43);
    AA44 = fmnaOp(tmp, AA64, AA44);
    AA45 = fmnaOp(tmp, AA65, AA45);
    AA46 = mulOp(negOp(tmp), AA66);
    AA47 = fmnaOp(tmp, AA67, AA47);
    AA48 = fmnaOp(tmp, AA68, AA48);
    AA49 = fmnaOp(tmp, AA69, AA49);

    tmp = AA56;
    AA50 = fmnaOp(tmp, AA60, AA50);
    AA51 = fmnaOp(tmp, AA61, AA51);
    AA52 = fmnaOp(tmp, AA62, AA52);
    AA53 = fmnaOp(tmp, AA63, AA53);
    AA54 = fmnaOp(tmp, AA64, AA54);
    AA55 = fmnaOp(tmp, AA65, AA55);
    AA56 = mulOp(negOp(tmp), AA66);
    AA57 = fmnaOp(tmp, AA67, AA57);
    AA58 = fmnaOp(tmp, AA68, AA58);
    AA59 = fmnaOp(tmp, AA69, AA59);

    tmp = AA76;
    AA70 = fmnaOp(tmp, AA60, AA70);
    AA71 = fmnaOp(tmp, AA61, AA71);
    AA72 = fmnaOp(tmp, AA62, AA72);
    AA73 = fmnaOp(tmp, AA63, AA73);
    AA74 = fmnaOp(tmp, AA64, AA74);
    AA75 = fmnaOp(tmp, AA65, AA75);
    AA76 = mulOp(negOp(tmp), AA66);
    AA77 = fmnaOp(tmp, AA67, AA77);
    AA78 = fmnaOp(tmp, AA68, AA78);
    AA79 = fmnaOp(tmp, AA69, AA79);

    tmp = AA86;
    AA80 = fmnaOp(tmp, AA60, AA80);
    AA81 = fmnaOp(tmp, AA61, AA81);
    AA82 = fmnaOp(tmp, AA62, AA82);
    AA83 = fmnaOp(tmp, AA63, AA83);
    AA84 = fmnaOp(tmp, AA64, AA84);
    AA85 = fmnaOp(tmp, AA65, AA85);
    AA86 = mulOp(negOp(tmp), AA66);
    AA87 = fmnaOp(tmp, AA67, AA87);
    AA88 = fmnaOp(tmp, AA68, AA88);
    AA89 = fmnaOp(tmp, AA69, AA89);

    tmp = AA96;
    AA90 = fmnaOp(tmp, AA60, AA90);
    AA91 = fmnaOp(tmp, AA61, AA91);
    AA92 = fmnaOp(tmp, AA62, AA92);
    AA93 = fmnaOp(tmp, AA63, AA93);
    AA94 = fmnaOp(tmp, AA64, AA94);
    AA95 = fmnaOp(tmp, AA65, AA95);
    AA96 = mulOp(negOp(tmp), AA66);
    AA97 = fmnaOp(tmp, AA67, AA97);
    AA98 = fmnaOp(tmp, AA68, AA98);
    AA99 = fmnaOp(tmp, AA69, AA99);

    /****************** iteration 7 ****************/

#if USE_PIVOTING
    /* search pivot row */
    p = absOp(AA77);
    pvt = 7;
    t = absOp(AA87);
    if (t > p) {
      p = t;
      pvt = 8;
    }
    t = absOp(AA97);
    if (t > p) {
      p = t;
      pvt = 9;
    }

    /* swap pivot row with row 7 */
    if (pvt == 8) {
      tmp = AA70;
      AA70 = AA80;
      AA80 = tmp;
      tmp = AA71;
      AA71 = AA81;
      AA81 = tmp;
      tmp = AA72;
      AA72 = AA82;
      AA82 = tmp;
      tmp = AA73;
      AA73 = AA83;
      AA83 = tmp;
      tmp = AA74;
      AA74 = AA84;
      AA84 = tmp;
      tmp = AA75;
      AA75 = AA85;
      AA85 = tmp;
      tmp = AA76;
      AA76 = AA86;
      AA86 = tmp;
      tmp = AA77;
      AA77 = AA87;
      AA87 = tmp;
      tmp = AA78;
      AA78 = AA88;
      AA88 = tmp;
      tmp = AA79;
      AA79 = AA89;
      AA89 = tmp;
      /* update permutation vector based on row swap */
      i = perm7;
      perm7 = perm8;
      perm8 = i;
    }
    if (pvt == 9) {
      tmp = AA70;
      AA70 = AA90;
      AA90 = tmp;
      tmp = AA71;
      AA71 = AA91;
      AA91 = tmp;
      tmp = AA72;
      AA72 = AA92;
      AA92 = tmp;
      tmp = AA73;
      AA73 = AA93;
      AA93 = tmp;
      tmp = AA74;
      AA74 = AA94;
      AA94 = tmp;
      tmp = AA75;
      AA75 = AA95;
      AA95 = tmp;
      tmp = AA76;
      AA76 = AA96;
      AA96 = tmp;
      tmp = AA77;
      AA77 = AA97;
      AA97 = tmp;
      tmp = AA78;
      AA78 = AA98;
      AA98 = tmp;
      tmp = AA79;
      AA79 = AA99;
      AA99 = tmp;
      /* update permutation vector based on row swap */
      i = perm7;
      perm7 = perm9;
      perm9 = i;
    }
#endif // USE_PIVOTING

    /* scale current row */
    tmp = rcpOp(AA77);
    icol7 = perm7;
    AA70 = mulOp(tmp, AA70);
    AA71 = mulOp(tmp, AA71);
    AA72 = mulOp(tmp, AA72);
    AA73 = mulOp(tmp, AA73);
    AA74 = mulOp(tmp, AA74);
    AA75 = mulOp(tmp, AA75);
    AA76 = mulOp(tmp, AA76);
    AA77 = tmp;
    AA78 = mulOp(tmp, AA78);
    AA79 = mulOp(tmp, AA79);

    /* eliminate above and below current row */
    tmp = AA07;
    AA00 = fmnaOp(tmp, AA70, AA00);
    AA01 = fmnaOp(tmp, AA71, AA01);
    AA02 = fmnaOp(tmp, AA72, AA02);
    AA03 = fmnaOp(tmp, AA73, AA03);
    AA04 = fmnaOp(tmp, AA74, AA04);
    AA05 = fmnaOp(tmp, AA75, AA05);
    AA06 = fmnaOp(tmp, AA76, AA06);
    AA07 = mulOp(negOp(tmp), AA77);
    AA08 = fmnaOp(tmp, AA78, AA08);
    AA09 = fmnaOp(tmp, AA79, AA09);

    tmp = AA17;
    AA10 = fmnaOp(tmp, AA70, AA10);
    AA11 = fmnaOp(tmp, AA71, AA11);
    AA12 = fmnaOp(tmp, AA72, AA12);
    AA13 = fmnaOp(tmp, AA73, AA13);
    AA14 = fmnaOp(tmp, AA74, AA14);
    AA15 = fmnaOp(tmp, AA75, AA15);
    AA16 = fmnaOp(tmp, AA76, AA16);
    AA17 = mulOp(negOp(tmp), AA77);
    AA18 = fmnaOp(tmp, AA78, AA18);
    AA19 = fmnaOp(tmp, AA79, AA19);

    tmp = AA27;
    AA20 = fmnaOp(tmp, AA70, AA20);
    AA21 = fmnaOp(tmp, AA71, AA21);
    AA22 = fmnaOp(tmp, AA72, AA22);
    AA23 = fmnaOp(tmp, AA73, AA23);
    AA24 = fmnaOp(tmp, AA74, AA24);
    AA25 = fmnaOp(tmp, AA75, AA25);
    AA26 = fmnaOp(tmp, AA76, AA26);
    AA27 = mulOp(negOp(tmp), AA77);
    AA28 = fmnaOp(tmp, AA78, AA28);
    AA29 = fmnaOp(tmp, AA79, AA29);

    tmp = AA37;
    AA30 = fmnaOp(tmp, AA70, AA30);
    AA31 = fmnaOp(tmp, AA71, AA31);
    AA32 = fmnaOp(tmp, AA72, AA32);
    AA33 = fmnaOp(tmp, AA73, AA33);
    AA34 = fmnaOp(tmp, AA74, AA34);
    AA35 = fmnaOp(tmp, AA75, AA35);
    AA36 = fmnaOp(tmp, AA76, AA36);
    AA37 = mulOp(negOp(tmp), AA77);
    AA38 = fmnaOp(tmp, AA78, AA38);
    AA39 = fmnaOp(tmp, AA79, AA39);

    tmp = AA47;
    AA40 = fmnaOp(tmp, AA70, AA40);
    AA41 = fmnaOp(tmp, AA71, AA41);
    AA42 = fmnaOp(tmp, AA72, AA42);
    AA43 = fmnaOp(tmp, AA73, AA43);
    AA44 = fmnaOp(tmp, AA74, AA44);
    AA45 = fmnaOp(tmp, AA75, AA45);
    AA46 = fmnaOp(tmp, AA76, AA46);
    AA47 = mulOp(negOp(tmp), AA77);
    AA48 = fmnaOp(tmp, AA78, AA48);
    AA49 = fmnaOp(tmp, AA79, AA49);

    tmp = AA57;
    AA50 = fmnaOp(tmp, AA70, AA50);
    AA51 = fmnaOp(tmp, AA71, AA51);
    AA52 = fmnaOp(tmp, AA72, AA52);
    AA53 = fmnaOp(tmp, AA73, AA53);
    AA54 = fmnaOp(tmp, AA74, AA54);
    AA55 = fmnaOp(tmp, AA75, AA55);
    AA56 = fmnaOp(tmp, AA76, AA56);
    AA57 = mulOp(negOp(tmp), AA77);
    AA58 = fmnaOp(tmp, AA78, AA58);
    AA59 = fmnaOp(tmp, AA79, AA59);

    tmp = AA67;
    AA60 = fmnaOp(tmp, AA70, AA60);
    AA61 = fmnaOp(tmp, AA71, AA61);
    AA62 = fmnaOp(tmp, AA72, AA62);
    AA63 = fmnaOp(tmp, AA73, AA63);
    AA64 = fmnaOp(tmp, AA74, AA64);
    AA65 = fmnaOp(tmp, AA75, AA65);
    AA66 = fmnaOp(tmp, AA76, AA66);
    AA67 = mulOp(negOp(tmp), AA77);
    AA68 = fmnaOp(tmp, AA78, AA68);
    AA69 = fmnaOp(tmp, AA79, AA69);

    tmp = AA87;
    AA80 = fmnaOp(tmp, AA70, AA80);
    AA81 = fmnaOp(tmp, AA71, AA81);
    AA82 = fmnaOp(tmp, AA72, AA82);
    AA83 = fmnaOp(tmp, AA73, AA83);
    AA84 = fmnaOp(tmp, AA74, AA84);
    AA85 = fmnaOp(tmp, AA75, AA85);
    AA86 = fmnaOp(tmp, AA76, AA86);
    AA87 = mulOp(negOp(tmp), AA77);
    AA88 = fmnaOp(tmp, AA78, AA88);
    AA89 = fmnaOp(tmp, AA79, AA89);

    tmp = AA97;
    AA90 = fmnaOp(tmp, AA70, AA90);
    AA91 = fmnaOp(tmp, AA71, AA91);
    AA92 = fmnaOp(tmp, AA72, AA92);
    AA93 = fmnaOp(tmp, AA73, AA93);
    AA94 = fmnaOp(tmp, AA74, AA94);
    AA95 = fmnaOp(tmp, AA75, AA95);
    AA96 = fmnaOp(tmp, AA76, AA96);
    AA97 = mulOp(negOp(tmp), AA77);
    AA98 = fmnaOp(tmp, AA78, AA98);
    AA99 = fmnaOp(tmp, AA79, AA99);

    /****************** iteration 8 ****************/

#if USE_PIVOTING
    /* search pivot row */
    p = absOp(AA88);
    pvt = 8;
    t = absOp(AA98);
    if (t > p) {
      p = t;
      pvt = 9;
    }

    /* swap pivot row with row 8 */
    if (pvt == 9) {
      tmp = AA80;
      AA80 = AA90;
      AA90 = tmp;
      tmp = AA81;
      AA81 = AA91;
      AA91 = tmp;
      tmp = AA82;
      AA82 = AA92;
      AA92 = tmp;
      tmp = AA83;
      AA83 = AA93;
      AA93 = tmp;
      tmp = AA84;
      AA84 = AA94;
      AA94 = tmp;
      tmp = AA85;
      AA85 = AA95;
      AA95 = tmp;
      tmp = AA86;
      AA86 = AA96;
      AA96 = tmp;
      tmp = AA87;
      AA87 = AA97;
      AA97 = tmp;
      tmp = AA88;
      AA88 = AA98;
      AA98 = tmp;
      tmp = AA89;
      AA89 = AA99;
      AA99 = tmp;
      /* update permutation vector based on row swap */
      i = perm8;
      perm8 = perm9;
      perm9 = i;
    }
#endif // USE_PIVOTING

    /* scale current row */
    tmp = rcpOp(AA88);
    icol8 = perm8;
    AA80 = mulOp(tmp, AA80);
    AA81 = mulOp(tmp, AA81);
    AA82 = mulOp(tmp, AA82);
    AA83 = mulOp(tmp, AA83);
    AA84 = mulOp(tmp, AA84);
    AA85 = mulOp(tmp, AA85);
    AA86 = mulOp(tmp, AA86);
    AA87 = mulOp(tmp, AA87);
    AA88 = tmp;
    AA89 = mulOp(tmp, AA89);

    /* eliminate above and below current row */
    tmp = AA08;
    AA00 = fmnaOp(tmp, AA80, AA00);
    AA01 = fmnaOp(tmp, AA81, AA01);
    AA02 = fmnaOp(tmp, AA82, AA02);
    AA03 = fmnaOp(tmp, AA83, AA03);
    AA04 = fmnaOp(tmp, AA84, AA04);
    AA05 = fmnaOp(tmp, AA85, AA05);
    AA06 = fmnaOp(tmp, AA86, AA06);
    AA07 = fmnaOp(tmp, AA87, AA07);
    AA08 = mulOp(negOp(tmp), AA88);
    AA09 = fmnaOp(tmp, AA89, AA09);

    tmp = AA18;
    AA10 = fmnaOp(tmp, AA80, AA10);
    AA11 = fmnaOp(tmp, AA81, AA11);
    AA12 = fmnaOp(tmp, AA82, AA12);
    AA13 = fmnaOp(tmp, AA83, AA13);
    AA14 = fmnaOp(tmp, AA84, AA14);
    AA15 = fmnaOp(tmp, AA85, AA15);
    AA16 = fmnaOp(tmp, AA86, AA16);
    AA17 = fmnaOp(tmp, AA87, AA17);
    AA18 = mulOp(negOp(tmp), AA88);
    AA19 = fmnaOp(tmp, AA89, AA19);

    tmp = AA28;
    AA20 = fmnaOp(tmp, AA80, AA20);
    AA21 = fmnaOp(tmp, AA81, AA21);
    AA22 = fmnaOp(tmp, AA82, AA22);
    AA23 = fmnaOp(tmp, AA83, AA23);
    AA24 = fmnaOp(tmp, AA84, AA24);
    AA25 = fmnaOp(tmp, AA85, AA25);
    AA26 = fmnaOp(tmp, AA86, AA26);
    AA27 = fmnaOp(tmp, AA87, AA27);
    AA28 = mulOp(negOp(tmp), AA88);
    AA29 = fmnaOp(tmp, AA89, AA29);

    tmp = AA38;
    AA30 = fmnaOp(tmp, AA80, AA30);
    AA31 = fmnaOp(tmp, AA81, AA31);
    AA32 = fmnaOp(tmp, AA82, AA32);
    AA33 = fmnaOp(tmp, AA83, AA33);
    AA34 = fmnaOp(tmp, AA84, AA34);
    AA35 = fmnaOp(tmp, AA85, AA35);
    AA36 = fmnaOp(tmp, AA86, AA36);
    AA37 = fmnaOp(tmp, AA87, AA37);
    AA38 = mulOp(negOp(tmp), AA88);
    AA39 = fmnaOp(tmp, AA89, AA39);

    tmp = AA48;
    AA40 = fmnaOp(tmp, AA80, AA40);
    AA41 = fmnaOp(tmp, AA81, AA41);
    AA42 = fmnaOp(tmp, AA82, AA42);
    AA43 = fmnaOp(tmp, AA83, AA43);
    AA44 = fmnaOp(tmp, AA84, AA44);
    AA45 = fmnaOp(tmp, AA85, AA45);
    AA46 = fmnaOp(tmp, AA86, AA46);
    AA47 = fmnaOp(tmp, AA87, AA47);
    AA48 = mulOp(negOp(tmp), AA88);
    AA49 = fmnaOp(tmp, AA89, AA49);

    tmp = AA58;
    AA50 = fmnaOp(tmp, AA80, AA50);
    AA51 = fmnaOp(tmp, AA81, AA51);
    AA52 = fmnaOp(tmp, AA82, AA52);
    AA53 = fmnaOp(tmp, AA83, AA53);
    AA54 = fmnaOp(tmp, AA84, AA54);
    AA55 = fmnaOp(tmp, AA85, AA55);
    AA56 = fmnaOp(tmp, AA86, AA56);
    AA57 = fmnaOp(tmp, AA87, AA57);
    AA58 = mulOp(negOp(tmp), AA88);
    AA59 = fmnaOp(tmp, AA89, AA59);

    tmp = AA68;
    AA60 = fmnaOp(tmp, AA80, AA60);
    AA61 = fmnaOp(tmp, AA81, AA61);
    AA62 = fmnaOp(tmp, AA82, AA62);
    AA63 = fmnaOp(tmp, AA83, AA63);
    AA64 = fmnaOp(tmp, AA84, AA64);
    AA65 = fmnaOp(tmp, AA85, AA65);
    AA66 = fmnaOp(tmp, AA86, AA66);
    AA67 = fmnaOp(tmp, AA87, AA67);
    AA68 = mulOp(negOp(tmp), AA88);
    AA69 = fmnaOp(tmp, AA89, AA69);

    tmp = AA78;
    AA70 = fmnaOp(tmp, AA80, AA70);
    AA71 = fmnaOp(tmp, AA81, AA71);
    AA72 = fmnaOp(tmp, AA82, AA72);
    AA73 = fmnaOp(tmp, AA83, AA73);
    AA74 = fmnaOp(tmp, AA84, AA74);
    AA75 = fmnaOp(tmp, AA85, AA75);
    AA76 = fmnaOp(tmp, AA86, AA76);
    AA77 = fmnaOp(tmp, AA87, AA77);
    AA78 = mulOp(negOp(tmp), AA88);
    AA79 = fmnaOp(tmp, AA89, AA79);

    tmp = AA98;
    AA90 = fmnaOp(tmp, AA80, AA90);
    AA91 = fmnaOp(tmp, AA81, AA91);
    AA92 = fmnaOp(tmp, AA82, AA92);
    AA93 = fmnaOp(tmp, AA83, AA93);
    AA94 = fmnaOp(tmp, AA84, AA94);
    AA95 = fmnaOp(tmp, AA85, AA95);
    AA96 = fmnaOp(tmp, AA86, AA96);
    AA97 = fmnaOp(tmp, AA87, AA97);
    AA98 = mulOp(negOp(tmp), AA88);
    AA99 = fmnaOp(tmp, AA89, AA99);

    /****************** iteration 9 ****************/

    /* scale current row */
    tmp = rcpOp(AA99);
    icol9 = perm9;
    AA90 = mulOp(tmp, AA90);
    AA91 = mulOp(tmp, AA91);
    AA92 = mulOp(tmp, AA92);
    AA93 = mulOp(tmp, AA93);
    AA94 = mulOp(tmp, AA94);
    AA95 = mulOp(tmp, AA95);
    AA96 = mulOp(tmp, AA96);
    AA97 = mulOp(tmp, AA97);
    AA98 = mulOp(tmp, AA98);
    AA99 = tmp;

    /* eliminate above and below current row */
    tmp = AA09;
    AA00 = fmnaOp(tmp, AA90, AA00);
    AA01 = fmnaOp(tmp, AA91, AA01);
    AA02 = fmnaOp(tmp, AA92, AA02);
    AA03 = fmnaOp(tmp, AA93, AA03);
    AA04 = fmnaOp(tmp, AA94, AA04);
    AA05 = fmnaOp(tmp, AA95, AA05);
    AA06 = fmnaOp(tmp, AA96, AA06);
    AA07 = fmnaOp(tmp, AA97, AA07);
    AA08 = fmnaOp(tmp, AA98, AA08);
    AA09 = mulOp(negOp(tmp), AA99);

    tmp = AA19;
    AA10 = fmnaOp(tmp, AA90, AA10);
    AA11 = fmnaOp(tmp, AA91, AA11);
    AA12 = fmnaOp(tmp, AA92, AA12);
    AA13 = fmnaOp(tmp, AA93, AA13);
    AA14 = fmnaOp(tmp, AA94, AA14);
    AA15 = fmnaOp(tmp, AA95, AA15);
    AA16 = fmnaOp(tmp, AA96, AA16);
    AA17 = fmnaOp(tmp, AA97, AA17);
    AA18 = fmnaOp(tmp, AA98, AA18);
    AA19 = mulOp(negOp(tmp), AA99);

    tmp = AA29;
    AA20 = fmnaOp(tmp, AA90, AA20);
    AA21 = fmnaOp(tmp, AA91, AA21);
    AA22 = fmnaOp(tmp, AA92, AA22);
    AA23 = fmnaOp(tmp, AA93, AA23);
    AA24 = fmnaOp(tmp, AA94, AA24);
    AA25 = fmnaOp(tmp, AA95, AA25);
    AA26 = fmnaOp(tmp, AA96, AA26);
    AA27 = fmnaOp(tmp, AA97, AA27);
    AA28 = fmnaOp(tmp, AA98, AA28);
    AA29 = mulOp(negOp(tmp), AA99);

    tmp = AA39;
    AA30 = fmnaOp(tmp, AA90, AA30);
    AA31 = fmnaOp(tmp, AA91, AA31);
    AA32 = fmnaOp(tmp, AA92, AA32);
    AA33 = fmnaOp(tmp, AA93, AA33);
    AA34 = fmnaOp(tmp, AA94, AA34);
    AA35 = fmnaOp(tmp, AA95, AA35);
    AA36 = fmnaOp(tmp, AA96, AA36);
    AA37 = fmnaOp(tmp, AA97, AA37);
    AA38 = fmnaOp(tmp, AA98, AA38);
    AA39 = mulOp(negOp(tmp), AA99);

    tmp = AA49;
    AA40 = fmnaOp(tmp, AA90, AA40);
    AA41 = fmnaOp(tmp, AA91, AA41);
    AA42 = fmnaOp(tmp, AA92, AA42);
    AA43 = fmnaOp(tmp, AA93, AA43);
    AA44 = fmnaOp(tmp, AA94, AA44);
    AA45 = fmnaOp(tmp, AA95, AA45);
    AA46 = fmnaOp(tmp, AA96, AA46);
    AA47 = fmnaOp(tmp, AA97, AA47);
    AA48 = fmnaOp(tmp, AA98, AA48);
    AA49 = mulOp(negOp(tmp), AA99);

    tmp = AA59;
    AA50 = fmnaOp(tmp, AA90, AA50);
    AA51 = fmnaOp(tmp, AA91, AA51);
    AA52 = fmnaOp(tmp, AA92, AA52);
    AA53 = fmnaOp(tmp, AA93, AA53);
    AA54 = fmnaOp(tmp, AA94, AA54);
    AA55 = fmnaOp(tmp, AA95, AA55);
    AA56 = fmnaOp(tmp, AA96, AA56);
    AA57 = fmnaOp(tmp, AA97, AA57);
    AA58 = fmnaOp(tmp, AA98, AA58);
    AA59 = mulOp(negOp(tmp), AA99);

    tmp = AA69;
    AA60 = fmnaOp(tmp, AA90, AA60);
    AA61 = fmnaOp(tmp, AA91, AA61);
    AA62 = fmnaOp(tmp, AA92, AA62);
    AA63 = fmnaOp(tmp, AA93, AA63);
    AA64 = fmnaOp(tmp, AA94, AA64);
    AA65 = fmnaOp(tmp, AA95, AA65);
    AA66 = fmnaOp(tmp, AA96, AA66);
    AA67 = fmnaOp(tmp, AA97, AA67);
    AA68 = fmnaOp(tmp, AA98, AA68);
    AA69 = mulOp(negOp(tmp), AA99);

    tmp = AA79;
    AA70 = fmnaOp(tmp, AA90, AA70);
    AA71 = fmnaOp(tmp, AA91, AA71);
    AA72 = fmnaOp(tmp, AA92, AA72);
    AA73 = fmnaOp(tmp, AA93, AA73);
    AA74 = fmnaOp(tmp, AA94, AA74);
    AA75 = fmnaOp(tmp, AA95, AA75);
    AA76 = fmnaOp(tmp, AA96, AA76);
    AA77 = fmnaOp(tmp, AA97, AA77);
    AA78 = fmnaOp(tmp, AA98, AA78);
    AA79 = mulOp(negOp(tmp), AA99);

    tmp = AA89;
    AA80 = fmnaOp(tmp, AA90, AA80);
    AA81 = fmnaOp(tmp, AA91, AA81);
    AA82 = fmnaOp(tmp, AA92, AA82);
    AA83 = fmnaOp(tmp, AA93, AA83);
    AA84 = fmnaOp(tmp, AA94, AA84);
    AA85 = fmnaOp(tmp, AA95, AA85);
    AA86 = fmnaOp(tmp, AA96, AA86);
    AA87 = fmnaOp(tmp, AA97, AA87);
    AA88 = fmnaOp(tmp, AA98, AA88);
    AA89 = mulOp(negOp(tmp), AA99);

    /* sort columns into the correct order */
    Ainv(0, icol0) = AA00;
    Ainv(1, icol0) = AA10;
    Ainv(2, icol0) = AA20;
    Ainv(3, icol0) = AA30;
    Ainv(4, icol0) = AA40;
    Ainv(5, icol0) = AA50;
    Ainv(6, icol0) = AA60;
    Ainv(7, icol0) = AA70;
    Ainv(8, icol0) = AA80;
    Ainv(9, icol0) = AA90;
    Ainv(0, icol1) = AA01;
    Ainv(1, icol1) = AA11;
    Ainv(2, icol1) = AA21;
    Ainv(3, icol1) = AA31;
    Ainv(4, icol1) = AA41;
    Ainv(5, icol1) = AA51;
    Ainv(6, icol1) = AA61;
    Ainv(7, icol1) = AA71;
    Ainv(8, icol1) = AA81;
    Ainv(9, icol1) = AA91;
    Ainv(0, icol2) = AA02;
    Ainv(1, icol2) = AA12;
    Ainv(2, icol2) = AA22;
    Ainv(3, icol2) = AA32;
    Ainv(4, icol2) = AA42;
    Ainv(5, icol2) = AA52;
    Ainv(6, icol2) = AA62;
    Ainv(7, icol2) = AA72;
    Ainv(8, icol2) = AA82;
    Ainv(9, icol2) = AA92;
    Ainv(0, icol3) = AA03;
    Ainv(1, icol3) = AA13;
    Ainv(2, icol3) = AA23;
    Ainv(3, icol3) = AA33;
    Ainv(4, icol3) = AA43;
    Ainv(5, icol3) = AA53;
    Ainv(6, icol3) = AA63;
    Ainv(7, icol3) = AA73;
    Ainv(8, icol3) = AA83;
    Ainv(9, icol3) = AA93;
    Ainv(0, icol4) = AA04;
    Ainv(1, icol4) = AA14;
    Ainv(2, icol4) = AA24;
    Ainv(3, icol4) = AA34;
    Ainv(4, icol4) = AA44;
    Ainv(5, icol4) = AA54;
    Ainv(6, icol4) = AA64;
    Ainv(7, icol4) = AA74;
    Ainv(8, icol4) = AA84;
    Ainv(9, icol4) = AA94;
    Ainv(0, icol5) = AA05;
    Ainv(1, icol5) = AA15;
    Ainv(2, icol5) = AA25;
    Ainv(3, icol5) = AA35;
    Ainv(4, icol5) = AA45;
    Ainv(5, icol5) = AA55;
    Ainv(6, icol5) = AA65;
    Ainv(7, icol5) = AA75;
    Ainv(8, icol5) = AA85;
    Ainv(9, icol5) = AA95;
    Ainv(0, icol6) = AA06;
    Ainv(1, icol6) = AA16;
    Ainv(2, icol6) = AA26;
    Ainv(3, icol6) = AA36;
    Ainv(4, icol6) = AA46;
    Ainv(5, icol6) = AA56;
    Ainv(6, icol6) = AA66;
    Ainv(7, icol6) = AA76;
    Ainv(8, icol6) = AA86;
    Ainv(9, icol6) = AA96;
    Ainv(0, icol7) = AA07;
    Ainv(1, icol7) = AA17;
    Ainv(2, icol7) = AA27;
    Ainv(3, icol7) = AA37;
    Ainv(4, icol7) = AA47;
    Ainv(5, icol7) = AA57;
    Ainv(6, icol7) = AA67;
    Ainv(7, icol7) = AA77;
    Ainv(8, icol7) = AA87;
    Ainv(9, icol7) = AA97;
    Ainv(0, icol8) = AA08;
    Ainv(1, icol8) = AA18;
    Ainv(2, icol8) = AA28;
    Ainv(3, icol8) = AA38;
    Ainv(4, icol8) = AA48;
    Ainv(5, icol8) = AA58;
    Ainv(6, icol8) = AA68;
    Ainv(7, icol8) = AA78;
    Ainv(8, icol8) = AA88;
    Ainv(9, icol8) = AA98;
    Ainv(0, icol9) = AA09;
    Ainv(1, icol9) = AA19;
    Ainv(2, icol9) = AA29;
    Ainv(3, icol9) = AA39;
    Ainv(4, icol9) = AA49;
    Ainv(5, icol9) = AA59;
    Ainv(6, icol9) = AA69;
    Ainv(7, icol9) = AA79;
    Ainv(8, icol9) = AA89;
    Ainv(9, icol9) = AA99;
  }
}

extern __shared__ double2 shmem[];

template <typename T, int pad, int pivot_thrds, int arch>
__global__ void __launch_bounds__(config<T, arch>::gje3MaxThrds,
                                  config<T, arch>::gje3MinBlks)
    lu_matinv_gje3(const T *A, T *Ainv, int N, int batch) {
  T *As = (T *)shmem;
  typename config<T, arch>::absValType *Val =
      (typename config<T, arch>::absValType *)(As + (N + pad) * N);
  int *Loc = (int *)(Val + pivot_thrds);
  int *icol = (int *)(Loc + pivot_thrds);
  int *perm = (int *)(icol + N);
  T diagRcp;
  const int ofs = pad;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int blkNum = blockIdx.y * gridDim.x + blockIdx.x;

  if (blkNum >= batch)
    return;

  /* Load matrix into shared memory */
  for (int i = tx; i < N; i += blockDim.x) {
    As(i, ty) = A[(i * N + ty) * batch + blkNum];
  }
  /* initialize row permutation vector */
  if (tx == 0)
    perm[ty] = ty;

  int j = 0;
  do {
    /* Look for pivot */
    __syncthreads();
    if ((tx == 0) && (ty < pivot_thrds)) {
      typename config<T, arch>::absValType val0 = absOp(As(j, j));
      int loc0 = j;
      int i = j + 1 + ty;
      T *dp = &As(i, j);
      const int incr = &As(pivot_thrds, 0) - &As(0, 0);
      while (i < N) {
        typename config<T, arch>::absValType vali = absOp(*dp);
        if (val0 < vali) {
          val0 = vali;
          loc0 = i;
        }
        dp += incr;
        i += pivot_thrds;
      }
      Loc[ty] = loc0;
      if (pivot_thrds > 1)
        Val[ty] = val0;
    }

    /* Swap current row with pivot */
    __syncthreads();
    if (tx == 0) {
      T tmp;
      int it;
      int Pl = Loc[0];
      if (pivot_thrds > 1) {
        typename config<T, arch>::absValType val = Val[0];
        int i = 1;
        for (; i < (pivot_thrds - 1); i++) {
          if (Val[i] > val) {
            Pl = Loc[i];
            val = Val[i];
          }
        }
        if (Val[i] > val) {
          Pl = Loc[i];
        }
      }
      tmp = As(Pl, ty);
      As(Pl, ty) = As(j, ty);
      As(j, ty) = tmp;
      /* update permutation vector based on row swap */
      if (ty == j) {
        it = perm[Pl];
        perm[Pl] = perm[j];
        perm[j] = it;
      }
    }

    /* scale current row, except current column */
    __syncthreads();
    diagRcp = rcpOp(As(j, j));
    if ((tx == 0) && !(ty == j)) {
      As(j, ty) = mulOp(As(j, ty), diagRcp);
    }

    /* update above and below current row, except current column */
    __syncthreads();
    for (int i = tx; i < N; i += blockDim.x) {
      if ((i != j) && !(ty == j)) {
        As(i, ty) = fmnaOp(As(i, j), As(j, ty), As(i, ty));
      }
    }

    /* update current column, and column permutation vector */
    __syncthreads();
    if (tx == 0) {
      As(ty, j) = (ty == j) ? diagRcp : negOp(mulOp(As(ty, j), diagRcp));
      if (ty == j) {
        icol[j] = perm[j];
      }
    }

    j++;
  } while (j < N);

  __syncthreads();
  for (int i = tx; i < N; i += blockDim.x) {
    Ainv[(i * N + icol[ty]) * batch + blkNum] = As(i, ty);
  }
}

template <typename T, int arch>
int lu_matinv_gje3(const T *A_d, T *Ainv_d, int n, int batch,
                cudaStream_t stream) {
  typedef void (*func)(const T *A_d, T *Ainv_d, int n, int batch);

  static int padding[110] = {
      config<T, arch>::gje3Pad_00,  config<T, arch>::gje3Pad_01,
      config<T, arch>::gje3Pad_02,  config<T, arch>::gje3Pad_03,
      config<T, arch>::gje3Pad_04,  config<T, arch>::gje3Pad_05,
      config<T, arch>::gje3Pad_06,  config<T, arch>::gje3Pad_07,
      config<T, arch>::gje3Pad_08,  config<T, arch>::gje3Pad_09,
      config<T, arch>::gje3Pad_10,  config<T, arch>::gje3Pad_11,
      config<T, arch>::gje3Pad_12,  config<T, arch>::gje3Pad_13,
      config<T, arch>::gje3Pad_14,  config<T, arch>::gje3Pad_15,
      config<T, arch>::gje3Pad_16,  config<T, arch>::gje3Pad_17,
      config<T, arch>::gje3Pad_18,  config<T, arch>::gje3Pad_19,
      config<T, arch>::gje3Pad_20,  config<T, arch>::gje3Pad_21,
      config<T, arch>::gje3Pad_22,  config<T, arch>::gje3Pad_23,
      config<T, arch>::gje3Pad_24,  config<T, arch>::gje3Pad_25,
      config<T, arch>::gje3Pad_26,  config<T, arch>::gje3Pad_27,
      config<T, arch>::gje3Pad_28,  config<T, arch>::gje3Pad_29,
      config<T, arch>::gje3Pad_30,  config<T, arch>::gje3Pad_31,
      config<T, arch>::gje3Pad_32,  config<T, arch>::gje3Pad_33,
      config<T, arch>::gje3Pad_34,  config<T, arch>::gje3Pad_35,
      config<T, arch>::gje3Pad_36,  config<T, arch>::gje3Pad_37,
      config<T, arch>::gje3Pad_38,  config<T, arch>::gje3Pad_39,
      config<T, arch>::gje3Pad_40,  config<T, arch>::gje3Pad_41,
      config<T, arch>::gje3Pad_42,  config<T, arch>::gje3Pad_43,
      config<T, arch>::gje3Pad_44,  config<T, arch>::gje3Pad_45,
      config<T, arch>::gje3Pad_46,  config<T, arch>::gje3Pad_47,
      config<T, arch>::gje3Pad_48,  config<T, arch>::gje3Pad_49,
      config<T, arch>::gje3Pad_50,  config<T, arch>::gje3Pad_51,
      config<T, arch>::gje3Pad_52,  config<T, arch>::gje3Pad_53,
      config<T, arch>::gje3Pad_54,  config<T, arch>::gje3Pad_55,
      config<T, arch>::gje3Pad_56,  config<T, arch>::gje3Pad_57,
      config<T, arch>::gje3Pad_58,  config<T, arch>::gje3Pad_59,
      config<T, arch>::gje3Pad_60,  config<T, arch>::gje3Pad_61,
      config<T, arch>::gje3Pad_62,  config<T, arch>::gje3Pad_63,
      config<T, arch>::gje3Pad_64,  config<T, arch>::gje3Pad_65,
      config<T, arch>::gje3Pad_66,  config<T, arch>::gje3Pad_67,
      config<T, arch>::gje3Pad_68,  config<T, arch>::gje3Pad_69,
      config<T, arch>::gje3Pad_70,  config<T, arch>::gje3Pad_71,
      config<T, arch>::gje3Pad_72,  config<T, arch>::gje3Pad_73,
      config<T, arch>::gje3Pad_74,  config<T, arch>::gje3Pad_75,
      config<T, arch>::gje3Pad_76,  config<T, arch>::gje3Pad_77,
      config<T, arch>::gje3Pad_78,  config<T, arch>::gje3Pad_79,
      config<T, arch>::gje3Pad_80,  config<T, arch>::gje3Pad_81,
      config<T, arch>::gje3Pad_82,  config<T, arch>::gje3Pad_83,
      config<T, arch>::gje3Pad_84,  config<T, arch>::gje3Pad_85,
      config<T, arch>::gje3Pad_86,  config<T, arch>::gje3Pad_87,
      config<T, arch>::gje3Pad_88,  config<T, arch>::gje3Pad_89,
      config<T, arch>::gje3Pad_90,  config<T, arch>::gje3Pad_91,
      config<T, arch>::gje3Pad_92,  config<T, arch>::gje3Pad_93,
      config<T, arch>::gje3Pad_94,  config<T, arch>::gje3Pad_95,
      config<T, arch>::gje3Pad_96,  config<T, arch>::gje3Pad_97,
      config<T, arch>::gje3Pad_98,  config<T, arch>::gje3Pad_99,
      config<T, arch>::gje3Pad_100, config<T, arch>::gje3Pad_101,
      config<T, arch>::gje3Pad_102, config<T, arch>::gje3Pad_103,
      config<T, arch>::gje3Pad_104, config<T, arch>::gje3Pad_105,
      config<T, arch>::gje3Pad_106, config<T, arch>::gje3Pad_107,
      config<T, arch>::gje3Pad_108, config<T, arch>::gje3Pad_109};
  static int dimX[110] = {
      config<T, arch>::gje3DimX_00,  config<T, arch>::gje3DimX_01,
      config<T, arch>::gje3DimX_02,  config<T, arch>::gje3DimX_03,
      config<T, arch>::gje3DimX_04,  config<T, arch>::gje3DimX_05,
      config<T, arch>::gje3DimX_06,  config<T, arch>::gje3DimX_07,
      config<T, arch>::gje3DimX_08,  config<T, arch>::gje3DimX_09,
      config<T, arch>::gje3DimX_10,  config<T, arch>::gje3DimX_11,
      config<T, arch>::gje3DimX_12,  config<T, arch>::gje3DimX_13,
      config<T, arch>::gje3DimX_14,  config<T, arch>::gje3DimX_15,
      config<T, arch>::gje3DimX_16,  config<T, arch>::gje3DimX_17,
      config<T, arch>::gje3DimX_18,  config<T, arch>::gje3DimX_19,
      config<T, arch>::gje3DimX_20,  config<T, arch>::gje3DimX_21,
      config<T, arch>::gje3DimX_22,  config<T, arch>::gje3DimX_23,
      config<T, arch>::gje3DimX_24,  config<T, arch>::gje3DimX_25,
      config<T, arch>::gje3DimX_26,  config<T, arch>::gje3DimX_27,
      config<T, arch>::gje3DimX_28,  config<T, arch>::gje3DimX_29,
      config<T, arch>::gje3DimX_30,  config<T, arch>::gje3DimX_31,
      config<T, arch>::gje3DimX_32,  config<T, arch>::gje3DimX_33,
      config<T, arch>::gje3DimX_34,  config<T, arch>::gje3DimX_35,
      config<T, arch>::gje3DimX_36,  config<T, arch>::gje3DimX_37,
      config<T, arch>::gje3DimX_38,  config<T, arch>::gje3DimX_39,
      config<T, arch>::gje3DimX_40,  config<T, arch>::gje3DimX_41,
      config<T, arch>::gje3DimX_42,  config<T, arch>::gje3DimX_43,
      config<T, arch>::gje3DimX_44,  config<T, arch>::gje3DimX_45,
      config<T, arch>::gje3DimX_46,  config<T, arch>::gje3DimX_47,
      config<T, arch>::gje3DimX_48,  config<T, arch>::gje3DimX_49,
      config<T, arch>::gje3DimX_50,  config<T, arch>::gje3DimX_51,
      config<T, arch>::gje3DimX_52,  config<T, arch>::gje3DimX_53,
      config<T, arch>::gje3DimX_54,  config<T, arch>::gje3DimX_55,
      config<T, arch>::gje3DimX_56,  config<T, arch>::gje3DimX_57,
      config<T, arch>::gje3DimX_58,  config<T, arch>::gje3DimX_59,
      config<T, arch>::gje3DimX_60,  config<T, arch>::gje3DimX_61,
      config<T, arch>::gje3DimX_62,  config<T, arch>::gje3DimX_63,
      config<T, arch>::gje3DimX_64,  config<T, arch>::gje3DimX_65,
      config<T, arch>::gje3DimX_66,  config<T, arch>::gje3DimX_67,
      config<T, arch>::gje3DimX_68,  config<T, arch>::gje3DimX_69,
      config<T, arch>::gje3DimX_70,  config<T, arch>::gje3DimX_71,
      config<T, arch>::gje3DimX_72,  config<T, arch>::gje3DimX_73,
      config<T, arch>::gje3DimX_74,  config<T, arch>::gje3DimX_75,
      config<T, arch>::gje3DimX_76,  config<T, arch>::gje3DimX_77,
      config<T, arch>::gje3DimX_78,  config<T, arch>::gje3DimX_79,
      config<T, arch>::gje3DimX_80,  config<T, arch>::gje3DimX_81,
      config<T, arch>::gje3DimX_82,  config<T, arch>::gje3DimX_83,
      config<T, arch>::gje3DimX_84,  config<T, arch>::gje3DimX_85,
      config<T, arch>::gje3DimX_86,  config<T, arch>::gje3DimX_87,
      config<T, arch>::gje3DimX_88,  config<T, arch>::gje3DimX_89,
      config<T, arch>::gje3DimX_90,  config<T, arch>::gje3DimX_91,
      config<T, arch>::gje3DimX_92,  config<T, arch>::gje3DimX_93,
      config<T, arch>::gje3DimX_94,  config<T, arch>::gje3DimX_95,
      config<T, arch>::gje3DimX_96,  config<T, arch>::gje3DimX_97,
      config<T, arch>::gje3DimX_98,  config<T, arch>::gje3DimX_99,
      config<T, arch>::gje3DimX_100, config<T, arch>::gje3DimX_101,
      config<T, arch>::gje3DimX_102, config<T, arch>::gje3DimX_103,
      config<T, arch>::gje3DimX_104, config<T, arch>::gje3DimX_105,
      config<T, arch>::gje3DimX_106, config<T, arch>::gje3DimX_107,
      config<T, arch>::gje3DimX_108, config<T, arch>::gje3DimX_109};
  static int srchThrd[110] = {
      config<T, arch>::gje3SrchThrd_00,  config<T, arch>::gje3SrchThrd_01,
      config<T, arch>::gje3SrchThrd_02,  config<T, arch>::gje3SrchThrd_03,
      config<T, arch>::gje3SrchThrd_04,  config<T, arch>::gje3SrchThrd_05,
      config<T, arch>::gje3SrchThrd_06,  config<T, arch>::gje3SrchThrd_07,
      config<T, arch>::gje3SrchThrd_08,  config<T, arch>::gje3SrchThrd_09,
      config<T, arch>::gje3SrchThrd_10,  config<T, arch>::gje3SrchThrd_11,
      config<T, arch>::gje3SrchThrd_12,  config<T, arch>::gje3SrchThrd_13,
      config<T, arch>::gje3SrchThrd_14,  config<T, arch>::gje3SrchThrd_15,
      config<T, arch>::gje3SrchThrd_16,  config<T, arch>::gje3SrchThrd_17,
      config<T, arch>::gje3SrchThrd_18,  config<T, arch>::gje3SrchThrd_19,
      config<T, arch>::gje3SrchThrd_20,  config<T, arch>::gje3SrchThrd_21,
      config<T, arch>::gje3SrchThrd_22,  config<T, arch>::gje3SrchThrd_23,
      config<T, arch>::gje3SrchThrd_24,  config<T, arch>::gje3SrchThrd_25,
      config<T, arch>::gje3SrchThrd_26,  config<T, arch>::gje3SrchThrd_27,
      config<T, arch>::gje3SrchThrd_28,  config<T, arch>::gje3SrchThrd_29,
      config<T, arch>::gje3SrchThrd_30,  config<T, arch>::gje3SrchThrd_31,
      config<T, arch>::gje3SrchThrd_32,  config<T, arch>::gje3SrchThrd_33,
      config<T, arch>::gje3SrchThrd_34,  config<T, arch>::gje3SrchThrd_35,
      config<T, arch>::gje3SrchThrd_36,  config<T, arch>::gje3SrchThrd_37,
      config<T, arch>::gje3SrchThrd_38,  config<T, arch>::gje3SrchThrd_39,
      config<T, arch>::gje3SrchThrd_40,  config<T, arch>::gje3SrchThrd_41,
      config<T, arch>::gje3SrchThrd_42,  config<T, arch>::gje3SrchThrd_43,
      config<T, arch>::gje3SrchThrd_44,  config<T, arch>::gje3SrchThrd_45,
      config<T, arch>::gje3SrchThrd_46,  config<T, arch>::gje3SrchThrd_47,
      config<T, arch>::gje3SrchThrd_48,  config<T, arch>::gje3SrchThrd_49,
      config<T, arch>::gje3SrchThrd_50,  config<T, arch>::gje3SrchThrd_51,
      config<T, arch>::gje3SrchThrd_52,  config<T, arch>::gje3SrchThrd_53,
      config<T, arch>::gje3SrchThrd_54,  config<T, arch>::gje3SrchThrd_55,
      config<T, arch>::gje3SrchThrd_56,  config<T, arch>::gje3SrchThrd_57,
      config<T, arch>::gje3SrchThrd_58,  config<T, arch>::gje3SrchThrd_59,
      config<T, arch>::gje3SrchThrd_60,  config<T, arch>::gje3SrchThrd_61,
      config<T, arch>::gje3SrchThrd_62,  config<T, arch>::gje3SrchThrd_63,
      config<T, arch>::gje3SrchThrd_64,  config<T, arch>::gje3SrchThrd_65,
      config<T, arch>::gje3SrchThrd_66,  config<T, arch>::gje3SrchThrd_67,
      config<T, arch>::gje3SrchThrd_68,  config<T, arch>::gje3SrchThrd_69,
      config<T, arch>::gje3SrchThrd_70,  config<T, arch>::gje3SrchThrd_71,
      config<T, arch>::gje3SrchThrd_72,  config<T, arch>::gje3SrchThrd_73,
      config<T, arch>::gje3SrchThrd_74,  config<T, arch>::gje3SrchThrd_75,
      config<T, arch>::gje3SrchThrd_76,  config<T, arch>::gje3SrchThrd_77,
      config<T, arch>::gje3SrchThrd_78,  config<T, arch>::gje3SrchThrd_79,
      config<T, arch>::gje3SrchThrd_80,  config<T, arch>::gje3SrchThrd_81,
      config<T, arch>::gje3SrchThrd_82,  config<T, arch>::gje3SrchThrd_83,
      config<T, arch>::gje3SrchThrd_84,  config<T, arch>::gje3SrchThrd_85,
      config<T, arch>::gje3SrchThrd_86,  config<T, arch>::gje3SrchThrd_87,
      config<T, arch>::gje3SrchThrd_88,  config<T, arch>::gje3SrchThrd_89,
      config<T, arch>::gje3SrchThrd_90,  config<T, arch>::gje3SrchThrd_91,
      config<T, arch>::gje3SrchThrd_92,  config<T, arch>::gje3SrchThrd_93,
      config<T, arch>::gje3SrchThrd_94,  config<T, arch>::gje3SrchThrd_95,
      config<T, arch>::gje3SrchThrd_96,  config<T, arch>::gje3SrchThrd_97,
      config<T, arch>::gje3SrchThrd_98,  config<T, arch>::gje3SrchThrd_99,
      config<T, arch>::gje3SrchThrd_100, config<T, arch>::gje3SrchThrd_101,
      config<T, arch>::gje3SrchThrd_102, config<T, arch>::gje3SrchThrd_103,
      config<T, arch>::gje3SrchThrd_104, config<T, arch>::gje3SrchThrd_105,
      config<T, arch>::gje3SrchThrd_106, config<T, arch>::gje3SrchThrd_107,
      config<T, arch>::gje3SrchThrd_108, config<T, arch>::gje3SrchThrd_109};

  func pf[110] = {
      0,
      0,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_02,
                  config<T, arch>::gje3SrchThrd_02, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_03,
                  config<T, arch>::gje3SrchThrd_03, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_04,
                  config<T, arch>::gje3SrchThrd_04, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_05,
                  config<T, arch>::gje3SrchThrd_05, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_06,
                  config<T, arch>::gje3SrchThrd_06, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_07,
                  config<T, arch>::gje3SrchThrd_07, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_08,
                  config<T, arch>::gje3SrchThrd_08, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_09,
                  config<T, arch>::gje3SrchThrd_09, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_10,
                  config<T, arch>::gje3SrchThrd_10, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_11,
                  config<T, arch>::gje3SrchThrd_11, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_12,
                  config<T, arch>::gje3SrchThrd_12, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_13,
                  config<T, arch>::gje3SrchThrd_13, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_14,
                  config<T, arch>::gje3SrchThrd_14, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_15,
                  config<T, arch>::gje3SrchThrd_15, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_16,
                  config<T, arch>::gje3SrchThrd_16, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_17,
                  config<T, arch>::gje3SrchThrd_17, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_18,
                  config<T, arch>::gje3SrchThrd_18, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_19,
                  config<T, arch>::gje3SrchThrd_19, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_20,
                  config<T, arch>::gje3SrchThrd_20, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_21,
                  config<T, arch>::gje3SrchThrd_21, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_22,
                  config<T, arch>::gje3SrchThrd_22, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_23,
                  config<T, arch>::gje3SrchThrd_23, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_24,
                  config<T, arch>::gje3SrchThrd_24, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_25,
                  config<T, arch>::gje3SrchThrd_25, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_26,
                  config<T, arch>::gje3SrchThrd_26, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_27,
                  config<T, arch>::gje3SrchThrd_27, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_28,
                  config<T, arch>::gje3SrchThrd_28, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_29,
                  config<T, arch>::gje3SrchThrd_29, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_30,
                  config<T, arch>::gje3SrchThrd_30, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_31,
                  config<T, arch>::gje3SrchThrd_31, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_32,
                  config<T, arch>::gje3SrchThrd_32, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_33,
                  config<T, arch>::gje3SrchThrd_33, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_34,
                  config<T, arch>::gje3SrchThrd_34, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_35,
                  config<T, arch>::gje3SrchThrd_35, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_36,
                  config<T, arch>::gje3SrchThrd_36, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_37,
                  config<T, arch>::gje3SrchThrd_37, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_38,
                  config<T, arch>::gje3SrchThrd_38, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_39,
                  config<T, arch>::gje3SrchThrd_39, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_40,
                  config<T, arch>::gje3SrchThrd_40, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_41,
                  config<T, arch>::gje3SrchThrd_41, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_42,
                  config<T, arch>::gje3SrchThrd_42, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_43,
                  config<T, arch>::gje3SrchThrd_43, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_44,
                  config<T, arch>::gje3SrchThrd_44, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_45,
                  config<T, arch>::gje3SrchThrd_45, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_46,
                  config<T, arch>::gje3SrchThrd_46, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_47,
                  config<T, arch>::gje3SrchThrd_47, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_48,
                  config<T, arch>::gje3SrchThrd_48, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_49,
                  config<T, arch>::gje3SrchThrd_49, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_50,
                  config<T, arch>::gje3SrchThrd_50, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_51,
                  config<T, arch>::gje3SrchThrd_51, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_52,
                  config<T, arch>::gje3SrchThrd_52, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_53,
                  config<T, arch>::gje3SrchThrd_53, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_54,
                  config<T, arch>::gje3SrchThrd_54, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_55,
                  config<T, arch>::gje3SrchThrd_55, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_56,
                  config<T, arch>::gje3SrchThrd_56, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_57,
                  config<T, arch>::gje3SrchThrd_57, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_58,
                  config<T, arch>::gje3SrchThrd_58, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_59,
                  config<T, arch>::gje3SrchThrd_59, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_60,
                  config<T, arch>::gje3SrchThrd_60, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_61,
                  config<T, arch>::gje3SrchThrd_61, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_62,
                  config<T, arch>::gje3SrchThrd_62, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_63,
                  config<T, arch>::gje3SrchThrd_63, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_64,
                  config<T, arch>::gje3SrchThrd_64, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_65,
                  config<T, arch>::gje3SrchThrd_65, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_66,
                  config<T, arch>::gje3SrchThrd_66, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_67,
                  config<T, arch>::gje3SrchThrd_67, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_68,
                  config<T, arch>::gje3SrchThrd_68, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_69,
                  config<T, arch>::gje3SrchThrd_69, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_70,
                  config<T, arch>::gje3SrchThrd_70, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_71,
                  config<T, arch>::gje3SrchThrd_71, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_72,
                  config<T, arch>::gje3SrchThrd_72, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_73,
                  config<T, arch>::gje3SrchThrd_73, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_74,
                  config<T, arch>::gje3SrchThrd_74, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_75,
                  config<T, arch>::gje3SrchThrd_75, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_76,
                  config<T, arch>::gje3SrchThrd_76, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_77,
                  config<T, arch>::gje3SrchThrd_77, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_78,
                  config<T, arch>::gje3SrchThrd_78, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_79,
                  config<T, arch>::gje3SrchThrd_79, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_80,
                  config<T, arch>::gje3SrchThrd_80, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_81,
                  config<T, arch>::gje3SrchThrd_81, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_82,
                  config<T, arch>::gje3SrchThrd_82, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_83,
                  config<T, arch>::gje3SrchThrd_83, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_84,
                  config<T, arch>::gje3SrchThrd_84, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_85,
                  config<T, arch>::gje3SrchThrd_85, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_86,
                  config<T, arch>::gje3SrchThrd_86, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_87,
                  config<T, arch>::gje3SrchThrd_87, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_88,
                  config<T, arch>::gje3SrchThrd_88, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_89,
                  config<T, arch>::gje3SrchThrd_89, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_90,
                  config<T, arch>::gje3SrchThrd_90, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_91,
                  config<T, arch>::gje3SrchThrd_91, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_92,
                  config<T, arch>::gje3SrchThrd_92, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_93,
                  config<T, arch>::gje3SrchThrd_93, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_94,
                  config<T, arch>::gje3SrchThrd_94, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_95,
                  config<T, arch>::gje3SrchThrd_95, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_96,
                  config<T, arch>::gje3SrchThrd_96, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_97,
                  config<T, arch>::gje3SrchThrd_97, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_98,
                  config<T, arch>::gje3SrchThrd_98, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_99,
                  config<T, arch>::gje3SrchThrd_99, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_100,
                  config<T, arch>::gje3SrchThrd_100, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_101,
                  config<T, arch>::gje3SrchThrd_101, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_102,
                  config<T, arch>::gje3SrchThrd_102, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_103,
                  config<T, arch>::gje3SrchThrd_103, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_104,
                  config<T, arch>::gje3SrchThrd_104, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_105,
                  config<T, arch>::gje3SrchThrd_105, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_106,
                  config<T, arch>::gje3SrchThrd_106, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_107,
                  config<T, arch>::gje3SrchThrd_107, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_108,
                  config<T, arch>::gje3SrchThrd_108, arch>,
      lu_matinv_gje3<T, config<T, arch>::gje3Pad_109,
                  config<T, arch>::gje3SrchThrd_109, arch>,
  };

  if (n < config<T, arch>::gje3MinDim || n > config<T, arch>::gje3MaxDim ||
      batch < 1) {
    return -1;
  }

  dim3 dimBlock(dimX[n], n);
  dim3 dimGrid;
  if (batch <= GRID_DIM_LIMIT) {
    dimGrid.x = batch;
    dimGrid.y = 1;
    dimGrid.z = 1;
  } else {
    dimGrid.x = GRID_DIM_LIMIT;
    dimGrid.y = (batch + GRID_DIM_LIMIT - 1) / GRID_DIM_LIMIT;
    dimGrid.z = 1;
  }
  int smem_size =
      (sizeof(A_d[0]) * (n + padding[n]) * (n) +                    // As
       sizeof(typename config<T, arch>::absValType) * srchThrd[n] + // Val
       sizeof(int) * srchThrd[n] +                                  // Loc
       sizeof(int) * n +                                            // icol
       sizeof(int) * n);                                            // perm
  pf[n]<<<dimGrid, dimBlock, smem_size, stream>>>(A_d, Ainv_d, n, batch);
  /* Check synchronous errors, i.e. pre-launch */
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    return -2;
  }

  return 0;
}

template <typename T, int arch>
int lu_matinv_MatPerThread(const T *A_d, T *Ainv_d, int n, int batch,
                        cudaStream_t stream) {
  typedef void (*func)(const T *A_d, T *Ainv_d, int batch);
  int minBatchSize[11] = {0x7fffffff,
                          0x7fffffff,
                          config<T, arch>::matInv2x2MinBatch,
                          config<T, arch>::matInv3x3MinBatch,
                          config<T, arch>::matInv4x4MinBatch,
                          config<T, arch>::matInv5x5MinBatch,
                          config<T, arch>::matInv6x6MinBatch,
                          config<T, arch>::matInv7x7MinBatch,
                          config<T, arch>::matInv8x8MinBatch,
                          config<T, arch>::matInv9x9MinBatch,
                          config<T, arch>::matInv10x10MinBatch};
  func pf[11] = {0,
                 0,
                 lu_matinv_2x2_matrix_per_thread<T, arch>,
                 lu_matinv_3x3_matrix_per_thread<T, arch>,
                 lu_matinv_4x4_matrix_per_thread<T, arch>,
                 lu_matinv_5x5_matrix_per_thread<T, arch>,
                 lu_matinv_6x6_matrix_per_thread<T, arch>,
                 lu_matinv_7x7_matrix_per_thread<T, arch>,
                 lu_matinv_8x8_matrix_per_thread<T, arch>,
                 lu_matinv_9x9_matrix_per_thread<T, arch>,
                 lu_matinv_10x10_matrix_per_thread<T, arch>};
  cudaError_t err;
  dim3 dimBlock(128);
  dim3 dimGrid;
  int numBlocks;

  if (n < config<T, arch>::matInvMinDim || batch < 1) {
    return -1;
  }

  if (n > config<T, arch>::matInvMaxDim || batch < minBatchSize[n]) {
    return 1;
  }

  switch (n) {
  case 4:
    err = cudaFuncSetCacheConfig(lu_matinv_4x4_matrix_per_thread<T, arch>,
                                 cudaFuncCachePreferL1);
    break;
  case 5:
    err = cudaFuncSetCacheConfig(lu_matinv_5x5_matrix_per_thread<T, arch>,
                                 cudaFuncCachePreferL1);
    break;
  case 6:
    err = cudaFuncSetCacheConfig(lu_matinv_6x6_matrix_per_thread<T, arch>,
                                 cudaFuncCachePreferL1);
    break;
  case 7:
    err = cudaFuncSetCacheConfig(lu_matinv_7x7_matrix_per_thread<T, arch>,
                                 cudaFuncCachePreferL1);
    break;
  case 8:
    err = cudaFuncSetCacheConfig(lu_matinv_8x8_matrix_per_thread<T, arch>,
                                 cudaFuncCachePreferL1);
    break;
  case 9:
    err = cudaFuncSetCacheConfig(lu_matinv_9x9_matrix_per_thread<T, arch>,
                                 cudaFuncCachePreferL1);
    break;
  case 10:
    err = cudaFuncSetCacheConfig(lu_matinv_10x10_matrix_per_thread<T, arch>,
                                 cudaFuncCachePreferL1);
    break;
  default:
    err = cudaSuccess;
    break;
  }
  if (err != cudaSuccess) {
    return -2;
  }
  numBlocks = (batch + dimBlock.x - 1) / dimBlock.x;
  if (numBlocks <= GRID_DIM_LIMIT) {
    dimGrid.x = numBlocks;
    dimGrid.y = 1;
    dimGrid.z = 1;
  } else {
    dimGrid.x = GRID_DIM_LIMIT;
    dimGrid.y = (numBlocks + GRID_DIM_LIMIT - 1) / GRID_DIM_LIMIT;
    dimGrid.z = 1;
  }
  pf[n]<<<dimGrid, dimBlock, 0, stream>>>(A_d, Ainv_d, batch);
  /* Check synchronous errors, i.e. pre-launch */
  err = cudaGetLastError();
  if (cudaSuccess != err) {
    return -2;
  }
  return 0;
}

/* C callable wrapper functions */

template <typename T>
int lu_matinv_batch_async(const T *A, T *Ainv, int n, int batch,
                          cudaStream_t stream) {
  int stat;
  stat = lu_matinv_MatPerThread<T, GPU_ARCH>(A, Ainv, n, batch, stream);
  if (stat <= 0)
    return stat;
  return lu_matinv_gje3<T, GPU_ARCH>(A, Ainv, n, batch, stream);
}

template int lu_matinv_batch_async(const float *A, float *Ainv, int n,
                                   int batch, cudaStream_t stream);

template int lu_matinv_batch_async(const double *A, double *Ainv, int n,
                                   int batch, cudaStream_t stream);
} // namespace internal
} // namespace utils
} // namespace sfm