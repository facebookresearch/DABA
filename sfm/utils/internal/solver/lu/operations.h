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

#if !defined(OPERATIONS_H_)
#define OPERATIONS_SOLVE_H_
#include "cuComplex.h"

#define FAST_COMPLEX_ARITH  /* use same approach as LAPACK reference */

template <typename T> __device__ __forceinline__ T mkConst(int i);

__device__ __forceinline__ float fmnaOp (float a, float b, float c)
{
    return -(a * b) + c;
}

__device__ __forceinline__ float mulOp (float a, float b)
{
    return a * b;
}

__device__ __forceinline__ float rcpOp (float a)
{
    return 1.0f / a;
}

__device__ __forceinline__ float absOp (float a)
{
    return fabsf(a);
}

__device__ __forceinline__ float negOp (float a)
{
    return -(a);
}

template <>
__device__ __forceinline__ float mkConst(int i)
{
    return (float)i;
}

__device__ __forceinline__ double fmnaOp (double a, double b, double c)
{
    return -(a * b) + c;
}

__device__ __forceinline__ double mulOp (double a, double b)
{
    return a * b;
}

__device__ __forceinline__ double rcpOp (double a)
{
    return 1.0 / a;
}

__device__ __forceinline__ double absOp (double a)
{
    return fabs(a);
}

__device__ __forceinline__ double negOp (double a)
{
    return -(a);
}

template <>
__device__ __forceinline__ double mkConst(int i)
{
    return (double)i;
}

__device__ __forceinline__ cuComplex fmnaOp (cuComplex a, cuComplex b, 
                                             cuComplex c)
{
    b.x = -b.x;
    b.y = -b.y;
    return cuCfmaf (a, b, c);
}

__device__ __forceinline__ cuComplex mulOp (cuComplex a, cuComplex b)
{
    return cuCmulf (a, b);
}

__device__ __forceinline__ cuComplex rcpOp (cuComplex a)
{
#ifdef FAST_COMPLEX_ARITH
    float t = 1.0f / (a.x * a.x + a.y * a.y);
    return make_cuComplex (a.x * t, -a.y * t);
#else
    cuComplex t = make_cuComplex (1.0f, 0.0f);
    return cuCdivf (t, a);
#endif
}

__device__ __forceinline__ cuComplex negOp (cuComplex a)
{
    cuComplex t = make_cuComplex(-a.x, -a.y); 
    return t;
}

__device__ __forceinline__ double absOp (cuComplex a)
{
#ifdef FAST_COMPLEX_ARITH
    return fabsf(a.x) + fabsf(a.y);
#else
    return cuCabsf (a);
#endif
}

template <>
__device__ __forceinline__ cuComplex mkConst (int i)
{
    return make_cuComplex ((float)i, 0.0);
}

__device__ __forceinline__ cuDoubleComplex fmnaOp (cuDoubleComplex a, 
                                                   cuDoubleComplex b, 
                                                   cuDoubleComplex c)
{
    b.x = -b.x;
    b.y = -b.y;
    return cuCfma (a, b, c);    
}

__device__ __forceinline__ cuDoubleComplex mulOp (cuDoubleComplex a, 
                                                  cuDoubleComplex b)
{
    return cuCmul (a, b);
}

__device__ __forceinline__ cuDoubleComplex rcpOp (cuDoubleComplex a)
{
#ifdef FAST_COMPLEX_ARITH
    double t = 1.0 / (a.x * a.x + a.y * a.y);
    return make_cuDoubleComplex (a.x * t, -a.y * t);
#else
    cuDoubleComplex t = make_cuDoubleComplex (1.0, 0.0);
    return cuCdiv (t, a);
#endif
}

__device__ __forceinline__ cuDoubleComplex negOp (cuDoubleComplex a)
{
    cuDoubleComplex t = make_cuDoubleComplex(-a.x, -a.y); 
    return t;
}

__device__ __forceinline__ double absOp (cuDoubleComplex a)
{
#ifdef FAST_COMPLEX_ARITH
    return fabs(a.x) + fabs(a.y);
#else
    return cuCabs (a);
#endif
}

template <>
__device__ __forceinline__ cuDoubleComplex mkConst (int i)
{
    return make_cuDoubleComplex ((double)i, 0.0);
}

#endif /* OPERATIONS_H_ */
