<h1 align="center">
    DABA: Decentralized and Accelerated Large-Scale Bundle Adjustment
</h1>

<p align="justify">
Official implementation of <a href="https://arxiv.org/abs/2305.07026">Decentralization and Acceleration Enables Large-Scale Bundle Adjustment</a>. Taosha Fan, Joseph Ortiz, Ming Hsiao, Maurizio Monge, Jing Dong, Todd Murphey, Mustafa Mukadam. In Robotics Science and Systems (RSS), 2023.
</p>

<p align="center">
    <img src="./doc/figure.png" alt="drawing" width="700"/>
</p>

<p align="justify">
In this work, we present <b>D</b>ecentralized and <b>A</b>ccelerated <b>B</b>undle <b>A</b>djustment (<b>DABA</b>), a method that addresses the compute and communication bottleneck for bundle adjustment problems of arbitrary scale. Despite limited peer-to-peer communication, DABA achieves provable convergence to first-order critical points under mild conditions. Through extensive benchmarking with public datasets, we have shown that DABA converges much faster than comparable decentralized baselines, with similar memory usage and communication load. Compared to centralized baselines using a single device, DABA, while being decentralized, yields <b>more accurate solutions</b> with significant speedups of up to <b>953.7x over Ceres and 174.6x over DeepLM</b>. 
</p>


-----

## Dependencies


1. [Eigen >= 3.4.0](https://eigen.tuxfamily.org/index.php?title=Main_Page)
2. [CUDA >= 11.4](https://developer.nvidia.com/cuda-toolkit)
3. [CMake >= 3.18](https://cmake.org)
4. [OpenMPI](https://www.open-mpi.org)
5. [NCCL](https://developer.nvidia.com/nccl)
6. [CUB >= 11.4](https://nvlabs.github.io/cub/)
7. [Thrust >= 2.0](https://thrust.github.io)
8. [Glog](https://github.com/google/glog)
9. [Boost >= 1.60](https://www.boost.org)
10. [Ceres >= 2.0 (optional)](http://ceres-solver.org)


## Quickstart

#### Download [BAL Dataset](https://grail.cs.washington.edu/projects/bal/)

```bash
wget https://grail.cs.washington.edu/projects/bal/data/ladybug/problem-1723-156502-pre.txt.bz2
bzip2 -dk problem-1723-156502-pre.txt.bz2
```

#### Compile

```bash
git clone https://github.com/facebookresearch/DABA.git
cd DABA
mkdir release
cd release
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j16
```

#### Run

```bash
mpiexec -n NUM_DEVICES ./bin/mpi_daba_bal_dataset --dataset /path/to/your/dataset --iters 1000 --loss "trivial" --accelerated true --save true
```

## Citation

If you find this work useful for your research, please cite our paper:

```
@article{fan2023daba,
    title={Decentralization and Acceleration Enables Large-Scale Bundle Adjustment}, 
    author={Fan, Taosha and Ortiz, Joseph and Hsiao, Ming and Monge, Maurizio and Dong, Jing and Murphey, Todd and  Mukadam, Mustafa},
    journal={arXiv:2305.07026},
    year={2023},
}
```

## License

The majority of this project is licensed under [MIT License](./LICENSE). However, a [portion of the code](sfm/graph/) is available under the [Apache 2.0](https://github.com/apache/.github/blob/main/LICENSE) license.
