use midnight_curves::Bls12;
use midnight_proofs::poly::kzg::params::ParamsKZG;
use midnight_proofs::utils::SerdeFormat;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::env;
use std::fs::{self, File};
use std::io::{BufReader, Write};
use std::path::{Path, PathBuf};
use thiserror::Error;

const MAX_K: u32 = 20;
const DEFAULT_SRS_DIR: &str = "./assets";

#[derive(Error, Debug)]
pub enum SrsError {
    #[error("No SRS available for circuits of size k={0} (max: {MAX_K})")]
    CircuitTooLarge(u32),

    #[error("Failed to create SRS directory '{0}': {1}")]
    CreateDirectoryFailed(String, #[source] std::io::Error),

    #[error("Failed to open SRS file at '{0}': {1}. (Did you set SRS_DIR?)")]
    OpenFileFailed(String, #[source] std::io::Error),

    #[error("Failed to create SRS file '{0}': {1}")]
    CreateFileFailed(String, #[source] std::io::Error),

    #[error("Failed to read SRS params from '{0}': {1}")]
    ReadParamsFailed(String, #[source] std::io::Error),

    #[error("Failed to write SRS file '{0}': {1}")]
    WriteFileFailed(String, #[source] std::io::Error),

    #[error("Failed to serialize params: {0}")]
    SerializeFailed(#[source] std::io::Error),
}

type Result<T> = std::result::Result<T, SrsError>;

fn get_srs_dir() -> String {
    env::var("SRS_DIR").unwrap_or_else(|_| DEFAULT_SRS_DIR.into())
}

fn get_srs_paths(srs_dir: &str, filename_prefix: &str, k: u32) -> (PathBuf, PathBuf) {
    let specific_path = PathBuf::from(format!("{srs_dir}/{filename_prefix}_2p{k}"));
    let max_path = PathBuf::from(format!("{srs_dir}/{filename_prefix}_2p{MAX_K}"));

    let fetching_path = if specific_path.exists() {
        specific_path.clone()
    } else {
        max_path
    };

    (specific_path, fetching_path)
}

fn read_params(path: &Path) -> Result<ParamsKZG<Bls12>> {
    let file =
        File::open(path).map_err(|e| SrsError::OpenFileFailed(path.display().to_string(), e))?;

    ParamsKZG::read_custom(&mut BufReader::new(file), SerdeFormat::RawBytesUnchecked)
        .map_err(|e| SrsError::ReadParamsFailed(path.display().to_string(), e))
}

fn write_params(params: &ParamsKZG<Bls12>, path: &Path) -> Result<()> {
    let mut buf = Vec::new();
    params
        .write_custom(&mut buf, SerdeFormat::RawBytesUnchecked)
        .map_err(SrsError::SerializeFailed)?;

    let mut file = File::create(path)
        .map_err(|e| SrsError::CreateFileFailed(path.display().to_string(), e))?;

    file.write_all(&buf)
        .map_err(|e| SrsError::WriteFileFailed(path.display().to_string(), e))
}

fn load_and_cache_params(
    k: u32,
    specific_path: &Path,
    fetching_path: &Path,
) -> Result<ParamsKZG<Bls12>> {
    let mut params = read_params(fetching_path)?;

    // If we loaded the MAX_K file, downsize and cache the per-k file
    if fetching_path != specific_path {
        params.downsize(k);
        write_params(&params, specific_path)?;
    }

    Ok(params)
}

/// Loads a deterministic mock SRS for testing.
///
/// # Safety
///
/// This MUST NOT be used in production. This is for testing only.
#[allow(unused)]
pub fn mock_srs_agg(k: u32) -> Result<ParamsKZG<Bls12>> {
    if k > MAX_K {
        return Err(SrsError::CircuitTooLarge(k));
    }

    let srs_dir = get_srs_dir();
    let (specific_path, fetching_path) = get_srs_paths(&srs_dir, "bls_mock", k);

    // If the mock params file doesn't exist, create it via unsafe_setup
    if !fetching_path.exists() {
        fs::create_dir_all(&srs_dir)
            .map_err(|e| SrsError::CreateDirectoryFailed(srs_dir.clone(), e))?;

        let rng = StdRng::seed_from_u64(0xDEAD_BEEF);
        let params = ParamsKZG::<Bls12>::unsafe_setup(MAX_K, rng);

        write_params(&params, &fetching_path)?;
    }

    load_and_cache_params(k, &specific_path, &fetching_path)
}

/// Loads SRS parameters using an unsafe mock setup (deterministic).
pub fn filecoin_srs_agg(k: u32) -> Result<ParamsKZG<Bls12>> {
    if k > MAX_K {
        return Err(SrsError::CircuitTooLarge(k));
    }

    let rng = StdRng::seed_from_u64(0xDEAD_BEEF);
    let mut params = ParamsKZG::<Bls12>::unsafe_setup(MAX_K, rng);
    params.downsize(k);
    Ok(params)
}
