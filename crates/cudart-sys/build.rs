#![allow(unexpected_cfgs)]

include!("src/utils.rs");

fn main() {
    println!("cargo::rustc-check-cfg=cfg(no_cuda)");
    if is_no_cuda() {
        println!("cargo::warning={}", no_cuda_message!());
        println!("cargo::rustc-cfg=no_cuda");
    } else {
        let cuda_version =
            get_cuda_version().expect("Failed to determine the CUDA Toolkit version.");
        if !(cuda_version.starts_with("12.") || cuda_version.starts_with("13.")) {
            println!("cargo::warning=CUDA Toolkit version {cuda_version} detected. This crate is only tested with CUDA Toolkit versions 12.* and 13.*.");
        }
        let cuda_lib_path = get_cuda_lib_path().unwrap();
        let cuda_lib_path_str = cuda_lib_path.to_str().unwrap();
        println!("cargo:rustc-link-search=native={cuda_lib_path_str}");
        println!("cargo:rustc-link-lib=cudart");
    }
}
