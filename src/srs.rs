use rand::SeedableRng;

use midnight_curves::Bls12;
use midnight_proofs::poly::kzg::params::ParamsKZG;
use midnight_proofs::utils::SerdeFormat;
use rand::rngs::StdRng;
use std::env;
use std::fs::File;
use std::io::{BufReader, Write};
use std::path::Path;

macro_rules! ensure {
    ($cond:expr, $($arg:tt)*) => {
        if !$cond {
            return Err(io_other(format!($($arg)*)));
        }
    };
}
fn io_other(msg: impl Into<String>) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::Other, msg.into())
}

// Deterministic mock SRS for testing only MUST NOT be used in production
pub fn mock_srs_agg(k: u32) -> Result<ParamsKZG<Bls12>, std::io::Error> {
    ensure!(
        k <= 20,
        "No Filecoin SRS available for circuits of size k={}",
        k
    );

    let srs_dir = env::var("SRS_DIR").unwrap_or_else(|_| "./examples/assets".into());
    let srs_path = format!("{srs_dir}/bls_mock_2p{k}");
    let fetching_path = if Path::new(&srs_path).exists() {
        srs_path.clone()
    } else {
        format!("{srs_dir}/bls_mock_2p20")
    };

    // If the (mock) params file we're about to read doesn't exist, create it via unsafe_setup.
    if !Path::new(&fetching_path).exists() {
        std::fs::create_dir_all(&srs_dir)
            .map_err(|e| io_other(format!("Failed to create SRS_DIR '{}': {e}", srs_dir)))?;

        let rng = StdRng::seed_from_u64(0xDEAD_BEEF_u64);

        let params = ParamsKZG::<Bls12>::unsafe_setup(20, rng);

        let mut buf = Vec::new();
        params
            .write_custom(&mut buf, SerdeFormat::RawBytesUnchecked)
            .map_err(|e| io_other(format!("Failed to serialize mock params: {e}")))?;

        let mut file = File::create(&fetching_path).map_err(|e| {
            io_other(format!(
                "Failed to create mock SRS file '{}': {e}",
                fetching_path
            ))
        })?;
        file.write_all(&buf).map_err(|e| {
            io_other(format!(
                "Failed to write mock SRS file '{}': {e}",
                fetching_path
            ))
        })?;
    }

    let params_fs = File::open(Path::new(&fetching_path)).map_err(|e| {
        io_other(format!(
            "Failed to open SRS file at '{}': {e}. (Did you set SRS_DIR?)",
            fetching_path
        ))
    })?;

    let mut params: ParamsKZG<Bls12> = ParamsKZG::read_custom::<_>(
        &mut BufReader::new(params_fs),
        SerdeFormat::RawBytesUnchecked,
    )
    .map_err(|e| {
        io_other(format!(
            "Failed to read SRS params from '{}': {e}",
            fetching_path
        ))
    })?;

    // If we loaded the MAX_K file, downsize and cache the per-k file
    if fetching_path != srs_path {
        params.downsize(k);

        let mut buf = Vec::new();
        params
            .write_custom(&mut buf, SerdeFormat::RawBytesUnchecked)
            .map_err(|e| io_other(format!("Failed to serialize downsized params: {e}")))?;

        let mut file = File::create(&srs_path).map_err(|e| {
            io_other(format!(
                "Failed to create mock SRS cache file '{}': {e}",
                srs_path
            ))
        })?;
        file.write_all(&buf).map_err(|e| {
            io_other(format!(
                "Failed to write mock SRS cache '{}': {e}",
                srs_path
            ))
        })?;
    }

    Ok(params)
}

pub fn filecoin_srs_agg(k: u32) -> Result<ParamsKZG<Bls12>, std::io::Error> {
    ensure!(
        k <= 20,
        "No Filecoin SRS available for circuits of size k={}",
        k
    );

    let srs_dir = env::var("SRS_DIR").unwrap_or_else(|_| "./examples/assets".into());
    let srs_path = format!("{srs_dir}/bls_filecoin_2p{k}");
    let fetching_path = if Path::new(&srs_path).exists() {
        srs_path.clone()
    } else {
        format!("{srs_dir}/bls_filecoin_2p20")
    };

    let params_fs = File::open(Path::new(&fetching_path)).map_err(|e| {
        io_other(format!(
            "Failed to open SRS file at '{}': {e}. (Did you set SRS_DIR?)",
            fetching_path
        ))
    })?;

    let mut params: ParamsKZG<Bls12> = ParamsKZG::read_custom::<_>(
        &mut BufReader::new(params_fs),
        SerdeFormat::RawBytesUnchecked,
    )
    .map_err(|e| {
        io_other(format!(
            "Failed to read SRS params from '{}': {e}",
            fetching_path
        ))
    })?;

    if fetching_path != srs_path {
        params.downsize(k);

        let mut buf = Vec::new();
        params
            .write_custom(&mut buf, SerdeFormat::RawBytesUnchecked)
            .map_err(|e| io_other(format!("Failed to serialize downsized params: {e}")))?;

        let mut file = File::create(&srs_path).map_err(|e| {
            io_other(format!(
                "Failed to create SRS cache file '{}': {e}",
                srs_path
            ))
        })?;
        file.write_all(&buf[..])
            .map_err(|e| io_other(format!("Failed to write SRS cache '{}': {e}", srs_path)))?;
    }

    Ok(params)
}
