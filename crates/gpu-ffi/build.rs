extern crate bindgen;
use era_cudart_sys::get_cuda_lib_path;
use std::env::var;
use std::path::Path;
use std::{env, path::PathBuf};
// build.rs

fn main() {
    let bellman_cuda_path = if let Ok(path) = std::env::var("BELLMAN_CUDA_DIR") {
        path
    } else {
        // we need to instruct rustc so that it will find libbellman-cuda.a
        //   - if dep is resolved via git(cargo checks ~/.cargo/git/checkouts/)
        //   - if dep is resolved via local path
        //   - if you want to build on a macos or only for rust analyzer
        //      just `export BELLMAN_CUDA_DIR=$PWD/bellman-cuda`
        // so we will benefit from env variable for now
        todo!("set BELLMAN_CUDA_DIR=$PWD")
    };

    generate_bindings(&bellman_cuda_path);

    #[cfg(not(target_os = "macos"))]
    link_multiexp_library(&bellman_cuda_path); // FIXME enable
}

fn generate_bindings(bellman_cuda_path: &str) {
    println!("generating bindings");
    let header_file = &format!("{}/src/bellman-cuda.h", bellman_cuda_path);
    const OUT_FILE: &str = "bindings.rs";
    println!("cargo:rerun-if-changed={}", header_file);

    let bindings = bindgen::Builder::default()
        // The input header we would like to generate
        // bindings for.
        .header(header_file)
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(format!(
        "{}/{}",
        env::current_dir().unwrap().to_str().unwrap(),
        "src"
    ));
    println!("out path {:?}", out_path.to_str());
    bindings
        .write_to_file(out_path.join(OUT_FILE))
        .expect("Couldn't write bindings!");
}

fn link_multiexp_library(bellman_cuda_path: &str) {
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
