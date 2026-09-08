use crate::{PROTOCOL_VERSION, plugin_error};
use base64::{Engine, engine::general_purpose::STANDARD};
use serde_json::{Value, json};
use sixseven_core::{Algorithm, Provider, Registry, StoredModel, TgaError};
use std::sync::Arc;

pub struct Server {
    provider: Arc<dyn Provider>,
    model: Option<Box<dyn StoredModel>>,
    output: Vec<sixseven_core::Address>,
}

impl Server {
    pub fn new<A: Algorithm>() -> Self {
        let mut registry = Registry::default();
        registry.register_algorithm::<A>().expect("valid provider");
        let id = sixseven_core::AlgorithmId::new(A::ID).expect("valid algorithm id");
        Self {
            provider: registry.provider(&id).unwrap().clone(),
            model: None,
            output: Vec::new(),
        }
    }

    pub fn request(&mut self, request: Value) -> Value {
        let result = self.dispatch(&request);
        match result {
            Ok(value) => json!({ "id": request["id"], "result": value }),
            Err(error) => json!({ "id": request["id"], "error": { "message": error.to_string() } }),
        }
    }

    fn dispatch(&mut self, request: &Value) -> Result<Value, TgaError> {
        if request["version"].as_u64() != Some(u64::from(PROTOCOL_VERSION)) {
            return Err(plugin_error("unsupported protocol version"));
        }
        let params = &request["params"];
        match request["method"].as_str() {
            Some("describe") => {
                serde_json::to_value(self.provider.descriptor()).map_err(plugin_error)
            }
            Some("train") => {
                self.model = Some(
                    self.provider.train(
                        params["config"].clone(),
                        &serde_json::from_value::<Vec<_>>(params["observations"].clone())
                            .map_err(plugin_error)?,
                    )?,
                );
                Ok(Value::Null)
            }
            Some("load") => {
                let version: u32 = serde_json::from_value(params["model_version"].clone())
                    .map_err(plugin_error)?;
                let payload = STANDARD
                    .decode(
                        params["payload"]
                            .as_str()
                            .ok_or_else(|| plugin_error("missing payload"))?,
                    )
                    .map_err(plugin_error)?;
                self.model = Some(self.provider.load(version, &payload)?);
                Ok(Value::Null)
            }
            Some("generate") => {
                let limit: usize =
                    serde_json::from_value(params["limit"].clone()).map_err(plugin_error)?;
                if limit == 0 || limit > crate::MAX_GENERATION_SIZE {
                    return Err(plugin_error("invalid generation limit"));
                }
                self.output.resize(limit, [0; 16]);
                let model = self
                    .model
                    .as_mut()
                    .ok_or_else(|| plugin_error("model is not loaded"))?;
                let generated = model.generate(&mut self.output)?;
                sixseven_core::generation::validate_generated(generated, self.output.len())?;
                Ok(
                    json!({ "addresses": &self.output[..generated.written], "state": generated.state }),
                )
            }
            Some("feedback") => {
                self.model()?.apply_feedback(
                    &serde_json::from_value::<Vec<_>>(params["items"].clone())
                        .map_err(plugin_error)?,
                )?;
                Ok(Value::Null)
            }
            Some("save") => Ok(Value::String(STANDARD.encode(self.model()?.encode()?))),
            Some("close") => {
                self.model = None;
                Ok(Value::Null)
            }
            _ => Err(plugin_error("unknown plugin method")),
        }
    }

    fn model(&mut self) -> Result<&mut Box<dyn StoredModel>, TgaError> {
        self.model
            .as_mut()
            .ok_or_else(|| plugin_error("model is not loaded"))
    }
}

#[macro_export]
macro_rules! export_plugin {
    ($algorithm:ty) => {
        unsafe extern "C" fn create() -> *mut std::ffi::c_void {
            std::panic::catch_unwind(|| {
                Box::into_raw(Box::new($crate::Server::new::<$algorithm>())).cast()
            })
            .unwrap_or(std::ptr::null_mut())
        }
        unsafe extern "C" fn call(
            context: *mut std::ffi::c_void,
            input: *const u8,
            len: usize,
            output: *mut $crate::Buffer,
        ) -> i32 {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                if context.is_null()
                    || input.is_null()
                    || output.is_null()
                    || len > $crate::MAX_MESSAGE_BYTES
                {
                    return Err(());
                }
                let request = $crate::serde_json::from_slice(unsafe {
                    std::slice::from_raw_parts(input, len)
                })
                .map_err(|_| ())?;
                let response = unsafe { &mut *context.cast::<$crate::Server>() }.request(request);
                let bytes = $crate::serde_json::to_vec(&response)
                    .map_err(|_| ())?
                    .into_boxed_slice();
                let len = bytes.len();
                let data = Box::into_raw(bytes).cast::<u8>();
                unsafe {
                    *output = $crate::Buffer { data, len };
                }
                Ok::<_, ()>(())
            }));
            if matches!(result, Ok(Ok(()))) { 0 } else { 1 }
        }
        unsafe extern "C" fn release(_: *mut std::ffi::c_void, buffer: $crate::Buffer) {
            if !buffer.data.is_null() {
                unsafe {
                    drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(
                        buffer.data as *mut u8,
                        buffer.len,
                    )));
                }
            }
        }
        unsafe extern "C" fn destroy(context: *mut std::ffi::c_void) {
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                if !context.is_null() {
                    unsafe {
                        drop(Box::from_raw(context.cast::<$crate::Server>()));
                    }
                }
            }));
        }
        #[unsafe(no_mangle)]
        pub extern "C" fn sixseven_plugin_v1() -> *const $crate::PluginApi {
            static API: $crate::PluginApi = $crate::PluginApi {
                abi_version: 1,
                struct_size: std::mem::size_of::<$crate::PluginApi>(),
                create,
                call,
                release,
                destroy,
            };
            &API
        }
    };
}
