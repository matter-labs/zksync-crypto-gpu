#![allow(incomplete_features)]
#![allow(unexpected_cfgs)]
#![feature(generic_const_exprs)]

mod gates;
mod poseidon2_bn;
mod poseidon2_gl;
mod template;

fn main() {
    gates::generate();
    poseidon2_bn::generate();
    poseidon2_gl::generate();
    println!("cargo::rustc-check-cfg=cfg(no_cuda)");
    if era_cudart_sys::is_no_cuda() {
        println!("cargo::warning={}", era_cudart_sys::no_cuda_message!());
        println!("cargo::rustc-cfg=no_cuda");
    } else {
        use era_cudart_sys::{get_cuda_lib_path, get_cuda_version};
        let cuda_version =
            get_cuda_version().expect("Failed to determine the CUDA Toolkit version.");
        if !(cuda_version.starts_with("12.") || cuda_version.starts_with("13.")) {
            println!("cargo::warning=CUDA Toolkit version {cuda_version} detected. This crate is only tested with CUDA Toolkit versions 12.* and 13.*.");
        }
        let cudaarchs = std::env::var("CUDAARCHS").unwrap_or("native".to_string());
        let dst = cmake::Config::new("native")
            .profile("Release")
            .define("CMAKE_CUDA_ARCHITECTURES", cudaarchs)
            .build();
        let boojum_lib_path = dst.to_str().unwrap();
        println!("cargo:rustc-link-search=native={boojum_lib_path}");
        println!("cargo:rustc-link-lib=static=boojum_cuda_native");
        let cuda_lib_path = get_cuda_lib_path().unwrap();
        let cuda_lib_path_str = cuda_lib_path.to_str().unwrap();
        println!("cargo:rustc-link-search=native={cuda_lib_path_str}");
        println!("cargo:rustc-link-lib=cudart");
        #[cfg(target_os = "linux")]
        println!("cargo:rustc-link-lib=stdc++");
    }
}
