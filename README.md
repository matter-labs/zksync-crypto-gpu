# ZKsync GPU Acceleration for Cryptography

This repository contains the libraries aimed to provide
GPU acceleration for cryptography used in ZKsync project.

## Setup

In order to work with the workspace, you'll need.

- [Cmake 3.24 or higher](https://apt.kitware.com/)
- [CUDA toolkit](https://developer.nvidia.com/cuda-downloads)
- [Bellman CUDA](github.com/matter-labs/matter-labs/era-bellman-cuda)

For Bellman CUDA the easiest way to install is:

```
git clone git@github.com:matter-labs/era-bellman-cuda.git
cmake -Bera-bellman-cuda/build -Sera-bellman-cuda/ -DCMAKE_BUILD_TYPE=Release
cmake --build era-bellman-cuda/build/
```

Then add the following variable to your config (`.bashrc`/`.zshrc`):

```
export BELLMAN_CUDA_DIR=<PATH_TO>/era-bellman-cuda
```

## Crates

- [bindings-generator](./crates/bindings-generator/)
- [cudart-sys](./crates/cudart-sys/)
- [cudart](./crates/cudart/)
- [criterion-cuda](./crates/criterion-cuda/)
- [boojum-cuda](./crates/boojum-cuda/)
- [shivini](./crates/shivini/)
- [gpu-ffi](./crates/gpu-ffi/)
- [gpu-prover](./crates/gpu-prover/)
- [wrapper-prover](./crates/wrapper-prover/)

## License

Licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache-2.0
license, shall be dual licensed as above, without any additional terms or
conditions.
