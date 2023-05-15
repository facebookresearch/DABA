// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <nccl.h>
#include <sfm/types.h>

namespace sfm{
const MPI_Datatype traits<float>::MPI_INT_TYPE = MPI_INT;
const MPI_Datatype traits<float>::MPI_FLOAT_TYPE = MPI_FLOAT;
const ncclDataType_t traits<float>::NCCL_INT_TYPE = ncclInt;
const ncclDataType_t traits<float>::NCCL_FLOAT_TYPE = ncclFloat;

const MPI_Datatype traits<double>::MPI_INT_TYPE = MPI_INT;
const MPI_Datatype traits<double>::MPI_FLOAT_TYPE = MPI_DOUBLE;
const ncclDataType_t traits<double>::NCCL_INT_TYPE = ncclInt;
const ncclDataType_t traits<double>::NCCL_FLOAT_TYPE = ncclDouble;
}