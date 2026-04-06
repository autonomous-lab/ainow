//! AINow application.
//!
//! With "desktop" feature: Tauri desktop app with embedded Axum server.
//! Without: standalone Axum server only.

pub mod agent;
pub mod conversation;
pub mod llm;
pub mod model_manager;
pub mod server;
pub mod settings;
pub mod state;
pub mod tools;
pub mod types;

#[cfg(feature = "desktop")]
use settings::{get_settings, update_settings, Settings, SettingsState};
#[cfg(feature = "desktop")]
use std::sync::Mutex;

#[cfg(feature = "desktop")]
#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(
            tauri_plugin_log::Builder::default()
                .level(log::LevelFilter::Info)
                .build(),
        )
        .manage(SettingsState(Mutex::new(Settings::load())))
        .setup(|_app| {
            let settings = Settings::load();

            // Start Axum server in background
            tauri::async_runtime::spawn(async move {
                server::start(settings).await;
            });

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![get_settings, update_settings])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
