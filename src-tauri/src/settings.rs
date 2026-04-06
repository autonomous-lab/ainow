//! App settings persisted to a JSON file in the app data directory.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Mutex;

/// Application settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
    /// Base URL for the LLM API (e.g. "http://localhost:8080/v1")
    pub llm_base_url: String,
    /// API key for the LLM provider
    pub llm_api_key: String,
    /// Model name to use
    pub llm_model: String,
    /// Base URL for online/cloud LLM
    pub online_base_url: String,
    /// API key for online LLM
    pub online_api_key: String,
    /// Online model name
    pub online_model: String,
    /// Path to llama-server executable
    pub llama_server_exe: String,
    /// Directory for GGUF model files
    pub models_dir: String,
    /// Port for the Axum HTTP server
    pub port: u16,
    /// System prompt
    pub system_prompt: String,
}

impl Default for Settings {
    fn default() -> Self {
        let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
        Self {
            llm_base_url: "http://localhost:8080/v1".to_string(),
            llm_api_key: String::new(),
            llm_model: "local".to_string(),
            online_base_url: "https://openrouter.ai/api/v1".to_string(),
            online_api_key: String::new(),
            online_model: "google/gemini-3.1-flash-lite-preview".to_string(),
            llama_server_exe: "llama-server".to_string(),
            models_dir: home.join("models").to_string_lossy().to_string(),
            port: 3040,
            system_prompt: String::new(),
        }
    }
}

/// Where to store the settings file.
fn settings_path() -> PathBuf {
    let dir = dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("ainow");
    std::fs::create_dir_all(&dir).ok();
    dir.join("settings.json")
}

impl Settings {
    /// Load settings from disk, falling back to defaults.
    pub fn load() -> Self {
        let path = settings_path();
        if path.exists() {
            match std::fs::read_to_string(&path) {
                Ok(contents) => match serde_json::from_str(&contents) {
                    Ok(s) => return s,
                    Err(e) => log::warn!("Failed to parse settings: {e}"),
                },
                Err(e) => log::warn!("Failed to read settings: {e}"),
            }
        }
        Self::default()
    }

    /// Save settings to disk.
    pub fn save(&self) -> Result<()> {
        let path = settings_path();
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }
}

#[cfg(feature = "desktop")]
pub struct SettingsState(pub Mutex<Settings>);

#[cfg(feature = "desktop")]
#[tauri::command]
pub fn get_settings(state: tauri::State<'_, SettingsState>) -> Result<Settings, String> {
    let settings = state.0.lock().map_err(|e| e.to_string())?;
    Ok(settings.clone())
}

#[cfg(feature = "desktop")]
#[tauri::command]
pub fn update_settings(
    new_settings: Settings,
    state: tauri::State<'_, SettingsState>,
) -> Result<(), String> {
    let mut settings = state.0.lock().map_err(|e| e.to_string())?;
    *settings = new_settings;
    settings.save().map_err(|e: anyhow::Error| e.to_string())?;
    Ok(())
}
