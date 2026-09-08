use crate::{
    Address, Algorithm, AlgorithmId, AlgorithmSpec, Feedback, Generated, ModelArtifact,
    Observation, TargetModel, TgaError,
};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, marker::PhantomData, sync::Arc};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmDescriptor {
    pub id: AlgorithmId,
    pub description: String,
    pub model_version: u32,
}

pub trait StoredModel: TargetModel {
    fn algorithm_id(&self) -> AlgorithmId;
    fn encode(&self) -> Result<Vec<u8>, TgaError>;
}

pub trait Provider: Send + Sync {
    fn descriptor(&self) -> AlgorithmDescriptor;
    fn train(
        &self,
        config: serde_json::Value,
        observations: &[Observation],
    ) -> Result<Box<dyn StoredModel>, TgaError>;
    fn load(&self, version: u32, payload: &[u8]) -> Result<Box<dyn StoredModel>, TgaError>;
}

#[derive(Default, Clone)]
pub struct Registry {
    providers: BTreeMap<String, Arc<dyn Provider>>,
}

impl Registry {
    pub fn register(&mut self, provider: Arc<dyn Provider>) -> Result<(), TgaError> {
        let id = provider.descriptor().id.to_string();
        if self.providers.contains_key(&id) {
            return Err(TgaError::Config(format!(
                "algorithm already registered: {id}"
            )));
        }
        self.providers.insert(id, provider);
        Ok(())
    }

    pub fn register_algorithm<A: Algorithm>(&mut self) -> Result<(), TgaError> {
        self.register(Arc::new(Builtin::<A>(PhantomData)))
    }

    pub fn list(&self) -> Vec<AlgorithmDescriptor> {
        self.providers.values().map(|p| p.descriptor()).collect()
    }

    pub fn provider(&self, id: &AlgorithmId) -> Result<&Arc<dyn Provider>, TgaError> {
        self.providers
            .get(id.as_str())
            .ok_or_else(|| TgaError::UnknownAlgorithm(id.to_string()))
    }

    pub fn train(
        &self,
        spec: &AlgorithmSpec,
        observations: &[Observation],
    ) -> Result<ModelArtifact, TgaError> {
        let provider = self.provider(&spec.algorithm)?;
        let model = provider.train(spec.config.clone(), observations)?;
        Ok(ModelArtifact {
            algorithm: spec.algorithm.clone(),
            revision: 1,
            model_version: provider.descriptor().model_version,
            payload: model.encode()?,
            config: spec.config.clone(),
            metadata: BTreeMap::new(),
        })
    }

    pub fn open(&self, artifact: &ModelArtifact) -> Result<Box<dyn StoredModel>, TgaError> {
        self.provider(&artifact.algorithm)?
            .load(artifact.model_version, &artifact.payload)
    }

    pub fn update(
        &self,
        artifact: &mut ModelArtifact,
        feedback: &[Feedback],
    ) -> Result<(), TgaError> {
        let mut model = self.open(artifact)?;
        model.apply_feedback(feedback)?;
        self.save_model(artifact, model.as_ref())
    }

    pub fn save_model(
        &self,
        artifact: &mut ModelArtifact,
        model: &dyn StoredModel,
    ) -> Result<(), TgaError> {
        if model.algorithm_id() != artifact.algorithm {
            return Err(TgaError::Model(
                "model algorithm does not match artifact".into(),
            ));
        }
        let version = self
            .provider(&artifact.algorithm)?
            .descriptor()
            .model_version;
        let payload = model.encode()?;
        artifact.revision = artifact
            .revision
            .checked_add(1)
            .ok_or_else(|| TgaError::Model("revision overflow".into()))?;
        artifact.model_version = version;
        artifact.payload = payload;
        Ok(())
    }
}

struct Builtin<A>(PhantomData<A>);

impl<A: Algorithm> Provider for Builtin<A> {
    fn descriptor(&self) -> AlgorithmDescriptor {
        AlgorithmDescriptor {
            id: AlgorithmId::new(A::ID).expect("valid algorithm id"),
            description: A::DESCRIPTION.into(),
            model_version: A::MODEL_VERSION,
        }
    }

    fn train(
        &self,
        config: serde_json::Value,
        observations: &[Observation],
    ) -> Result<Box<dyn StoredModel>, TgaError> {
        let algorithm: A =
            serde_json::from_value(config).map_err(|e| TgaError::Config(e.to_string()))?;
        Ok(Box::new(BuiltinModel::<A>(algorithm.train(observations)?)))
    }

    fn load(&self, version: u32, payload: &[u8]) -> Result<Box<dyn StoredModel>, TgaError> {
        Ok(Box::new(BuiltinModel::<A>(A::decode_model(
            version, payload,
        )?)))
    }
}

struct BuiltinModel<A: Algorithm>(A::Model);

impl<A: Algorithm> TargetModel for BuiltinModel<A> {
    fn allows_repeated_probes(&self) -> bool {
        self.0.allows_repeated_probes()
    }

    fn detected_aliases(&self) -> &[crate::Ipv6Prefix] {
        self.0.detected_aliases()
    }

    fn set_budget(&mut self, budget: usize) {
        self.0.set_budget(budget);
    }
    fn generate(&mut self, output: &mut [Address]) -> Result<Generated, TgaError> {
        self.0.generate(output)
    }
    fn apply_feedback(&mut self, feedback: &[Feedback]) -> Result<(), TgaError> {
        self.0.apply_feedback(feedback)
    }
}

impl<A: Algorithm> StoredModel for BuiltinModel<A> {
    fn algorithm_id(&self) -> AlgorithmId {
        AlgorithmId::new(A::ID).expect("valid algorithm id")
    }
    fn encode(&self) -> Result<Vec<u8>, TgaError> {
        A::encode_model(&self.0)
    }
}
