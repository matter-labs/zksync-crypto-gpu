use era_cudart_sys::{get_cuda_lib_path, get_cuda_version, is_no_cuda, no_cuda_message};
use std::env;
use std::env::var;
use std::path::Path;

fn main() {
    println!("cargo::rustc-check-cfg=cfg(no_cuda)");
    if is_no_cuda() {
        println!("cargo::warning={}", no_cuda_message!());
        println!("cargo::rustc-cfg=no_cuda");
    } else {
        let cuda_version =
            get_cuda_version().expect("Failed to determine the CUDA Toolkit version.");
        if !cuda_version.starts_with("12.") {
            println!("cargo::warning=CUDA Toolkit version {cuda_version} detected. This crate is only tested with CUDA Toolkit version 12.*.");
        }
        let bellman_cuda_dir = var("BELLMAN_CUDA_DIR").unwrap();
        let bellman_cuda_path = Path::new(&bellman_cuda_dir);
        let bellman_cuda_lib_path = if Path::new(bellman_cuda_path).join("build").exists() {
            Path::new(bellman_cuda_path)
                .join("build")
                .join("src")
                .to_str()
                .unwrap()
                .to_string()
        } else {
            let cudaarchs = var("CUDAARCHS").unwrap_or("native".to_string());
            let dst = cmake::Config::new(bellman_cuda_path)
                .profile("Release")
                .define("CMAKE_CUDA_ARCHITECTURES", cudaarchs)
                .build();
            dst.to_str().unwrap().to_string()
        };
        println!("cargo:rustc-link-search=native={bellman_cuda_lib_path}");
        println!("cargo:rustc-link-lib=static=bellman-cuda");
        let cuda_lib_path = get_cuda_lib_path().unwrap();
        let cuda_lib_path_str = cuda_lib_path.to_str().unwrap();
        println!("cargo:rustc-link-search=native={cuda_lib_path_str}");
        println!("cargo:rustc-link-lib=cudart");
        #[cfg(target_os = "linux")]
        println!("cargo:rustc-link-lib=stdc++");
    }
}
