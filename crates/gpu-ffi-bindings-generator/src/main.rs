fn main() {
    let bellman_cuda_dir = std::env::var("BELLMAN_CUDA_DIR").unwrap();
    let bellman_cuda_path = std::path::Path::new(&bellman_cuda_dir);
    let header = bellman_cuda_path.join("src").join("bellman-cuda.h");
    let bindings = bindgen::Builder::default()
        .header(header.to_str().unwrap())
        .generate_comments(false)
        .layout_tests(false)
        .size_t_is_usize(false)
        .generate()
        .expect("Unable to generate bindings")
        .to_string();
    println!("{bindings}");
}
