use std::fs::File;
use std::path::{Path, PathBuf};

pub fn get_cuda_path() -> Option<&'static Path> {
    #[cfg(target_os = "linux")]
    {
        let path = Path::new("/usr/local/cuda");
        if path.exists() {
            Some(path)
        } else {
            None
        }
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
        Some(version.to_string())
    } else if let Some(path) = get_cuda_path() {
        let file = File::open(path.join("version.json")).expect("CUDA Toolkit should be installed");
        let reader = std::io::BufReader::new(file);
        let value: serde_json::Value = serde_json::from_reader(reader).unwrap();
        Some(value["cuda"]["version"].as_str().unwrap().to_string())
    } else {
        None
    }
}

#[cfg(no_cuda)]
#[macro_export]
macro_rules! no_cuda_message {
    () => {
        concat!(
            env!("CARGO_PKG_NAME"),
            " was compiled without CUDA Toolkit, CUDA functions were replaced by stubs."
        )
    };
}
