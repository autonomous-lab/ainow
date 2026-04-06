//! Standalone server — runs AINow without Tauri (no desktop window).
//!
//! Usage:
//!   cargo run --bin ainow-server --no-default-features
//!   # Open http://localhost:3040 in your browser

use app_lib::settings::Settings;

#[tokio::main]
async fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_secs()
        .init();

    let settings = Settings::load();
    log::info!("AINow server starting on port {}", settings.port);
    log::info!("Open http://localhost:{}", settings.port);

    app_lib::server::start(settings).await;
}
