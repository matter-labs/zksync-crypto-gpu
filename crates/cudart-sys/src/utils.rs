use std::fs::File;
use std::path::{Path, PathBuf};

pub fn get_cuda_path() -> Option<&'static Path> {
    #[cfg(target_os = "linux")]
    {
        for path_name in [option_env!("CUDA_PATH"), Some("/usr/local/cuda")].iter().flatten() {
            println!("trying {path_name}...");
            let path = Path::new(path_name);
            if path.exists() {
                println!("CUDA installation found at `{}`", path.display());
                return Some(path)
            }
        }
        None
    }
    #[cfg(target_os = "windows")]
    {
        option_env!("CUDA_PATH").map(Path::new)
    }
    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
    {
        None
    }
}

pub fn get_cuda_include_path() -> Option<PathBuf> {
    get_cuda_path().map(|path| path.join("include"))
}

pub fn get_cuda_lib_path() -> Option<PathBuf> {
    #[cfg(target_os = "linux")]
    {
        get_cuda_path().map(|path| path.join("lib64"))
    }
    #[cfg(target_os = "windows")]
    {
        get_cuda_path().map(|path| path.join("lib/x64"))
    }
    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
    {
        None
    }
}

pub fn get_cuda_version() -> Option<String> {
    if let Some(version) = option_env!("CUDA_VERSION") {
        println!("CUDA version defined in CUDA_VERSION as `{}`", version);
        Some(version.to_string())
    } else if let Some(path) = get_cuda_path() {
        println!("inferring CUDA version from nvcc output...");
        let re = regex_lite::Regex::new(r"V(?<version>\d{2}\.\d+\.\d+)").unwrap();
        let nvcc_out = std::process::Command::new("nvcc")
            .arg("--version")
            .output()
            .expect("failed to start `nvcc`");
        let nvcc_str = std::str::from_utf8(&nvcc_out.stdout).expect("`nvcc` output is not UTF8");
        let captures = re.captures(&nvcc_str).unwrap();
        let version = captures
            .get(0)
            .expect("unable to find nvcc version in the form VMM.mm.pp in the output of `nvcc --version`:\n{nvcc_str}")
            .as_str()
            .to_string();
        println!("CUDA version inferred to be `{version}`.");
        Some(version)
    } else {
        None
    }
}

pub fn is_no_cuda() -> bool {
    if cfg!(no_cuda) {
        true
    } else {
        let no_cuda = option_env!("ZKSYNC_USE_CUDA_STUBS").unwrap_or("");
        no_cuda == "1" || no_cuda.to_lowercase() == "true" || no_cuda.to_lowercase() == "yes"
    }
}

#[macro_export]
macro_rules! no_cuda_message {
    () => {
        concat!(
            env!("CARGO_PKG_NAME"),
            " was compiled without CUDA Toolkit, CUDA functions were replaced by stubs."
        )
    };
}
