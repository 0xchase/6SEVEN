use super::{MAX_MESSAGE_BYTES, Transport, plugin_error};
use libloading::Library;
use serde_json::Value;
use sixseven_core::TgaError;
use std::{ffi::c_void, path::Path};

#[repr(C)]
pub struct Buffer {
    pub data: *const u8,
    pub len: usize,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PluginApi {
    pub abi_version: u32,
    pub struct_size: usize,
    pub create: unsafe extern "C" fn() -> *mut c_void,
    pub call: unsafe extern "C" fn(*mut c_void, *const u8, usize, *mut Buffer) -> i32,
    pub release: unsafe extern "C" fn(*mut c_void, Buffer),
    pub destroy: unsafe extern "C" fn(*mut c_void),
}

pub(super) struct Native {
    context: *mut c_void,
    api: PluginApi,
    _library: Library,
}

// The ABI permits transferring instances between threads with serialized calls.
unsafe impl Send for Native {}

impl Native {
    pub(super) fn open(path: &Path) -> Result<Self, TgaError> {
        unsafe {
            let library = Library::new(path).map_err(plugin_error)?;
            let entry = library
                .get::<unsafe extern "C" fn() -> *const PluginApi>(b"sixseven_plugin_v1\0")
                .map_err(plugin_error)?;
            let pointer = entry();
            if pointer.is_null() {
                return Err(plugin_error("null plugin API"));
            }
            if (*pointer).abi_version != 1
                || (*pointer).struct_size != std::mem::size_of::<PluginApi>()
            {
                return Err(plugin_error("incompatible native plugin ABI"));
            }
            let api = *pointer;
            let context = (api.create)();
            if context.is_null() {
                return Err(plugin_error("plugin creation failed"));
            }
            Ok(Self {
                context,
                api,
                _library: library,
            })
        }
    }
}

impl Transport for Native {
    fn exchange(&mut self, request: &Value) -> Result<Value, TgaError> {
        let bytes = serde_json::to_vec(request).map_err(plugin_error)?;
        if bytes.len() > MAX_MESSAGE_BYTES {
            return Err(plugin_error("request exceeds limit"));
        }
        let mut output = Buffer {
            data: std::ptr::null(),
            len: 0,
        };
        unsafe {
            let code = (self.api.call)(self.context, bytes.as_ptr(), bytes.len(), &mut output);
            let result = if code != 0 {
                Err(plugin_error(format!("native call failed with code {code}")))
            } else if output.data.is_null() || output.len > MAX_MESSAGE_BYTES {
                Err(plugin_error("invalid native response buffer"))
            } else {
                serde_json::from_slice(std::slice::from_raw_parts(output.data, output.len))
                    .map_err(plugin_error)
            };
            if !output.data.is_null() {
                (self.api.release)(self.context, output);
            }
            result
        }
    }
}

impl Drop for Native {
    fn drop(&mut self) {
        unsafe {
            (self.api.destroy)(self.context);
        }
    }
}
