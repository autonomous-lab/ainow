//! AINow Tauri application.
//!
//! Integrates an Axum HTTP/WebSocket server with Tauri for
//! a desktop AI voice assistant.

mod agent;
mod conversation;
mod llm;
mod model_manager;
mod server;
mod settings;
mod state;
mod tools;
mod types;

use settings::{get_settings, update_settings, Settings, SettingsState};
use std::sync::Mutex;

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
