The high-level interface of the batched solver exposes the functions
dsolve_batch() and zsolve_batch() for solving systems with double-
precision and double complex data, respectively. Both functions are 
designed to solve a batch of N different systems, each with a single
right hand side. All systems must be of the same dimension. Besides 
the batch size and the matrix dimension, the functions expect pointers
to an array of matrices, an array of right hand sides, and an array 
of solution vectors. All arrays are assumed to be stored contiguously,
using column-major layout.

Both high-level API functions call into a single templated function
fast_solve() that is parameterized by data type and architecture. 
This function selects one of three CUDA kernels for performing the 
solve and sets up the launch configuration. The exact configuration
chosen is controlled by a template class config that is parameterized
by data type and architecture.

For performance reasons, each system is loaded into shared memory in
its entirety for the solve. This means that the maximum dimension of
the matrix that can be handled is limited by the available shared 
memory. For GPU architectures >= sm_20 dsolve_batch can handle 
systems up to dimensions 76, and zsolve_batch can handle systems up 
to dimensions 53. Batch size is limited by the amount of GPU memory. 
When loaded into shared memory the matrix is augmented on the right 
with the right hand side vector, allowing both to be manipulated in
parallel. The two-dimensional shared memory layout of the matrix uses
padding to minimize bank conflicts. The amount of padding is optimized
for each matrix size via the configuration class.

The number of thread blocks in the launch configuration is identical
to the batch size, i.e. the number of systems to be solved. Therefore,
each thread block solves a single system. Two-dimensional thread blocks
are used, where the x-dimension is configured for optimal performance
by the template class, and the y-dimension is identical to the number
of columns of the augmented matrix. This allows each thread row to 
handle one row of the augmented matrix in parallel during the solve.

The three kernels used are gauss_jordan1 (used for dimensions 2 
through 9), gauss_jordan2 (used for dimension 10), and gauss2 (for 
dimensions > 10). The switch-over points were determined empirically.
The first two kernels implement the standard Gauss-Jordan algorithm
with partial pivoting, and the third implements straight Gauss 
elimination with partial pivoting. It was established experimentally
that the absence of a separate backsubstitution step in Gauss-Jordan
outweighs the smaller count of floating-point operation of Gauss 
elimimination for small matrices.

The two Gauss-Jordan kernels differ in that for gauss_jordan1 the 
number of thread rows is identical to the number of rows in the matrix,
i.e. each thread handles exactly one element of the augmented matrix,
whereas in gauss_jordan2 the number of threads is less than the number
of matrix rows, so each thread handles more than one matrix element.
The former approach eliminates some overhead for iteration over the
rows of the matrix.

The maximum search in partial pivoting is implemented as a two-stage
process. In the first stage, a small number of threads search for a 
maximum in their respective subset of column elements. In a second 
stage, these partial results are reduced to an overall maximum by a
single thread. This approach was found to be more efficient than the 
traditional binary reduction process. The number of search threads is
fixed at two for the two Gauss-Jordan kernels, but is configurable for
optimal performance via the template class for the Gauss elimination
kernel. The number of search threads is generally a small, single 
digit number.


The high-level interface for the batched matrix inverse exposes the
four functions smatinv_batch(), dmatinv_batch(), cmatinv_batch(), and
zmatinv_batch(). These invert matrices with elements of type float,
float complex, double, and double complex, respectively. Each function
inverts N different square matrices, all of which must have the same size.
The functions expect pointers to an array of input matrices and an array
of output matrices. Both arrays are assumed to be stored contiguously,
with the matrices themselves using column-major layout.

All four functions map to a single templatized function matinv_gje3()
that is parameterized by data type and GPU architecture. This function
selects individual specialized kernels for very small matrices, while
most sizes are handled by a generic templatized kernel.

For performance reasons, each matrix is loaded into shared memory in
its entirety, then in-place Gauss-Jordan elimination is applied, which
si followed by write back of the inverted matrix to the array of result 
matrices. This approach limits the size of matrices that can be handled.
For GPU architectures >= sm_20 smatinv_batch can handle matrices up to a
dimension of 109, cmatinv_batch() and dmatinv_batch() can handle matrices
up to a dimension of 77, and zmatinv_batch() can handle matrices up to a
size dimension of 55. The batch size is limited by the amount of available
GPU memory.

As with the batched solver code, the generic kernel for the matrix inverse
is templatized via a configuration class that provides architecture-specific
values for the x-dimension of the thread block, the number of threads used
during the pivot search, and the padding applied to the shared memory layout
of the matrix. The pivot search uses a two-step approach as described above
for the batched solver.
