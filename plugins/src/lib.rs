mod native;
mod process;
mod sdk;
pub use sdk::Server;
pub use serde_json;

use base64::{Engine, engine::general_purpose::STANDARD};
pub use native::{Buffer, PluginApi};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sixseven_core::{
    Address, AlgorithmDescriptor, Feedback, Generated, GenerationState, Observation, Provider,
    Registry, StoredModel, TargetModel, TgaError,
};
use std::{
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::Duration,
};

pub const PROTOCOL_VERSION: u32 = 1;
pub const MAX_MESSAGE_BYTES: usize = 256 * 1024 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PluginConfig {
    Native {
        path: PathBuf,
    },
    Python {
        interpreter: PathBuf,
        script: PathBuf,
        #[serde(default = "default_timeout")]
        timeout_seconds: u64,
    },
}

fn default_timeout() -> u64 {
    300
}

trait Transport: Send {
    fn exchange(&mut self, request: &Value) -> Result<Value, TgaError>;
}

impl PluginConfig {
    fn connect(&self) -> Result<Client, TgaError> {
        let transport: Box<dyn Transport> = match self {
            Self::Native { path } => Box::new(native::Native::open(path)?),
            Self::Python {
                interpreter,
                script,
                timeout_seconds,
            } => {
                if *timeout_seconds == 0 {
                    return Err(TgaError::Config("plugin timeout must be positive".into()));
                }
                Box::new(process::Process::open(
                    interpreter,
                    script,
                    Duration::from_secs(*timeout_seconds),
                )?)
            }
        };
        Ok(Client {
            transport,
            sequence: 0,
            failed: false,
        })
    }
}

pub fn load_config(registry: &mut Registry, path: &Path) -> Result<(), TgaError> {
    let contents = std::fs::read(path).map_err(plugin_error)?;
    let mut configs: Vec<PluginConfig> = serde_json::from_slice(&contents).map_err(plugin_error)?;
    let parent = path.parent().unwrap_or(Path::new("."));
    for config in &mut configs {
        match config {
            PluginConfig::Native { path } => {
                if path.is_relative() {
                    *path = parent.join(&path);
                }
            }
            PluginConfig::Python {
                script,
                interpreter,
                ..
            } => {
                if script.is_relative() {
                    *script = parent.join(&script);
                }
                if interpreter.is_relative() && interpreter.components().count() > 1 {
                    *interpreter = parent.join(&interpreter);
                }
            }
        }
        register(registry, config.clone())?;
    }
    Ok(())
}

pub fn register(registry: &mut Registry, config: PluginConfig) -> Result<(), TgaError> {
    let mut client = config.connect()?;
    let descriptor: AlgorithmDescriptor =
        serde_json::from_value(client.call("describe", json!({}))?).map_err(plugin_error)?;
    registry.register(Arc::new(PluginProvider { config, descriptor }))
}

struct Client {
    transport: Box<dyn Transport>,
    sequence: u64,
    failed: bool,
}

impl Client {
    fn call(&mut self, method: &str, params: Value) -> Result<Value, TgaError> {
        if self.failed {
            return Err(plugin_error("plugin instance has failed"));
        }
        self.sequence += 1;
        let result = self.transport.exchange(&json!({ "version": PROTOCOL_VERSION, "id": self.sequence, "method": method, "params": params }));
        let response = match result {
            Ok(value) => value,
            Err(error) => {
                self.failed = true;
                return Err(error);
            }
        };
        if response.get("id").and_then(Value::as_u64) != Some(self.sequence) {
            self.failed = true;
            return Err(plugin_error("plugin response id mismatch"));
        }
        if let Some(error) = response.get("error") {
            self.failed = true;
            return Err(plugin_error(error));
        }
        response.get("result").cloned().ok_or_else(|| {
            self.failed = true;
            plugin_error("missing plugin result")
        })
    }
}

struct PluginProvider {
    config: PluginConfig,
    descriptor: AlgorithmDescriptor,
}

impl PluginProvider {
    fn model(&self, method: &str, params: Value) -> Result<Box<dyn StoredModel>, TgaError> {
        let mut client = self.config.connect()?;
        let descriptor: AlgorithmDescriptor =
            serde_json::from_value(client.call("describe", json!({}))?).map_err(plugin_error)?;
        if descriptor.id != self.descriptor.id
            || descriptor.model_version != self.descriptor.model_version
        {
            return Err(plugin_error("plugin descriptor changed after registration"));
        }
        if !client.call(method, params)?.is_null() {
            return Err(plugin_error("train and load must return null"));
        }
        Ok(Box::new(PluginModel {
            algorithm_id: self.descriptor.id.clone(),
            client: Mutex::new(client),
        }))
    }
}

impl Provider for PluginProvider {
    fn descriptor(&self) -> AlgorithmDescriptor {
        self.descriptor.clone()
    }
    fn train(
        &self,
        config: Value,
        observations: &[Observation],
    ) -> Result<Box<dyn StoredModel>, TgaError> {
        self.model(
            "train",
            json!({ "config": config, "observations": observations }),
        )
    }
    fn load(&self, version: u32, payload: &[u8]) -> Result<Box<dyn StoredModel>, TgaError> {
        self.model(
            "load",
            json!({ "model_version": version, "payload": STANDARD.encode(payload) }),
        )
    }
}

struct PluginModel {
    algorithm_id: sixseven_core::AlgorithmId,
    client: Mutex<Client>,
}

impl PluginModel {
    fn request<T>(
        &self,
        method: &str,
        params: Value,
        decode: impl FnOnce(Value) -> Result<T, TgaError>,
    ) -> Result<T, TgaError> {
        let mut client = self
            .client
            .lock()
            .map_err(|_| plugin_error("plugin lock poisoned"))?;
        let result = client.call(method, params).and_then(decode);
        client.failed |= result.is_err();
        result
    }
}

pub(crate) const MAX_GENERATION_SIZE: usize = 65536;

#[derive(Deserialize)]
struct GeneratedResponse {
    addresses: Vec<Address>,
    state: GenerationState,
}

impl TargetModel for PluginModel {
    fn generate(&mut self, output: &mut [Address]) -> Result<Generated, TgaError> {
        sixseven_core::generation::validate_output(output)?;
        let limit = output.len().min(MAX_GENERATION_SIZE);
        self.request("generate", json!({ "limit": limit }), |value| {
            let response: GeneratedResponse =
                serde_json::from_value(value).map_err(plugin_error)?;
            let generated = Generated {
                written: response.addresses.len(),
                state: response.state,
            };
            sixseven_core::generation::validate_generated(generated, limit)?;
            output[..generated.written].copy_from_slice(&response.addresses);
            Ok(generated)
        })
    }
    fn apply_feedback(&mut self, feedback: &[Feedback]) -> Result<(), TgaError> {
        self.request("feedback", json!({ "items": feedback }), |value| {
            if value.is_null() {
                Ok(())
            } else {
                Err(plugin_error("feedback must return null"))
            }
        })
    }
}

impl StoredModel for PluginModel {
    fn algorithm_id(&self) -> sixseven_core::AlgorithmId {
        self.algorithm_id.clone()
    }
    fn encode(&self) -> Result<Vec<u8>, TgaError> {
        self.request("save", json!({}), |value| {
            STANDARD
                .decode(
                    value
                        .as_str()
                        .ok_or_else(|| plugin_error("save must return base64"))?,
                )
                .map_err(plugin_error)
        })
    }
}

fn plugin_error(error: impl std::fmt::Display) -> TgaError {
    TgaError::Model(format!("plugin: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct InvalidBatch;
    impl Transport for InvalidBatch {
        fn exchange(&mut self, request: &Value) -> Result<Value, TgaError> {
            Ok(json!({"id": request["id"], "result": {"addresses": [], "state": "ready"}}))
        }
    }

    #[test]
    fn invalid_batch_fails_the_instance() {
        let mut model = PluginModel {
            algorithm_id: sixseven_core::AlgorithmId::new("test/bad").unwrap(),
            client: Mutex::new(Client {
                transport: Box::new(InvalidBatch),
                sequence: 0,
                failed: false,
            }),
        };
        assert!(model.generate(&mut [[0; 16]; 1]).is_err());
        assert!(model.client.lock().unwrap().failed);
        assert!(
            model
                .generate(&mut [[0; 16]; 1])
                .unwrap_err()
                .to_string()
                .contains("instance has failed")
        );
    }
}
