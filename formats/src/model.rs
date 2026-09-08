use sixseven_core::ModelArtifact;
use std::{
    io::{Read, Write},
    path::Path,
};

const MAGIC: &[u8; 8] = b"6SEVEN01";
const MAX_HEADER: u64 = 1_048_576;

pub fn load(path: impl AsRef<Path>) -> Result<ModelArtifact, crate::Error> {
    let mut file = std::fs::File::open(path)?;
    let mut magic = [0; 8];
    file.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(crate::Error::Invalid("invalid model format".into()));
    }
    let mut length = [0; 8];
    file.read_exact(&mut length)?;
    let length = u64::from_le_bytes(length);
    if length > MAX_HEADER {
        return Err(crate::Error::Invalid("model header exceeds limit".into()));
    }
    let mut bytes = vec![0; length as usize];
    file.read_exact(&mut bytes)?;
    let mut artifact: ModelArtifact = serde_json::from_slice(&bytes)?;
    if !artifact.payload.is_empty() {
        return Err(crate::Error::Invalid(
            "payload must follow the model header".into(),
        ));
    }
    file.read_to_end(&mut artifact.payload)?;
    Ok(artifact)
}

pub fn save(path: impl AsRef<Path>, artifact: &ModelArtifact) -> Result<(), crate::Error> {
    let path = path.as_ref();
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let header = ModelArtifact {
        algorithm: artifact.algorithm.clone(),
        revision: artifact.revision,
        model_version: artifact.model_version,
        config: artifact.config.clone(),
        metadata: artifact.metadata.clone(),
        payload: Vec::new(),
    };
    let bytes = serde_json::to_vec(&header)?;
    if bytes.len() as u64 > MAX_HEADER {
        return Err(crate::Error::Invalid("model header exceeds limit".into()));
    }
    let mut file = tempfile::NamedTempFile::new_in(parent)?;
    file.write_all(MAGIC)?;
    file.write_all(&(bytes.len() as u64).to_le_bytes())?;
    file.write_all(&bytes)?;
    file.write_all(&artifact.payload)?;
    file.as_file().sync_all()?;
    file.persist(path).map_err(|e| e.error)?;
    Ok(())
}
