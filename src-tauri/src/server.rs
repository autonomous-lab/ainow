//! Axum HTTP server + WebSocket handler.
//!
//! Endpoints:
//! - GET /         -> Serve the browser UI (index.html)
//! - GET /health   -> Health check
//! - GET /api/config -> Return initial config
//! - WS  /ws/browser -> WebSocket for browser voice/chat

use axum::{
    extract::{
        ws::{WebSocket, WebSocketUpgrade},
        State,
    },
    response::{Html, IntoResponse, Json},
    routing::get,
    Router,
};
use std::path::PathBuf;
use std::sync::Arc;
use tower_http::cors::CorsLayer;

use crate::conversation::run_conversation;
use crate::settings::Settings;

/// Shared server state passed to handlers.
#[derive(Clone)]
struct ServerState {
    settings: Settings,
    static_dir: PathBuf,
}

/// Start the Axum HTTP server on the configured port.
pub async fn start(settings: Settings) {
    let port = settings.port;
    let static_dir = find_static_dir();

    let state = ServerState {
        settings,
        static_dir,
    };

    let app = Router::new()
        .route("/", get(serve_index))
        .route("/health", get(health))
        .route("/api/config", get(api_config))
        .route("/ws/browser", get(ws_handler))
        .layer(CorsLayer::permissive())
        .with_state(Arc::new(state));

    let addr = format!("0.0.0.0:{port}");
    log::info!("Axum server starting on http://localhost:{port}");

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .expect("Failed to bind to address");

    axum::serve(listener, app)
        .await
        .expect("Axum server error");
}

/// Find the static directory containing index.html.
fn find_static_dir() -> PathBuf {
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()));

    let candidates = [
        PathBuf::from("src/static"),
        PathBuf::from("../src/static"),
        exe_dir
            .as_ref()
            .map(|d| d.join("../src/static"))
            .unwrap_or_default(),
        exe_dir
            .as_ref()
            .map(|d| d.join("../../src/static"))
            .unwrap_or_default(),
        PathBuf::from("D:/dev/fastcall/ainow/src/static"),
    ];

    for candidate in &candidates {
        if candidate.join("index.html").exists() {
            return candidate.clone();
        }
    }

    PathBuf::from("src/static")
}

async fn serve_index(State(state): State<Arc<ServerState>>) -> impl IntoResponse {
    let path = state.static_dir.join("index.html");
    match tokio::fs::read_to_string(&path).await {
        Ok(content) => Html(content).into_response(),
        Err(e) => {
            log::error!("Failed to read index.html from {}: {e}", path.display());
            Html(format!(
                "<h1>Error</h1><p>Could not load index.html: {e}</p>"
            ))
            .into_response()
        }
    }
}

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({"status": "ok"}))
}

async fn api_config(State(state): State<Arc<ServerState>>) -> Json<serde_json::Value> {
    Json(serde_json::json!({"system_prompt": state.settings.system_prompt}))
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<ServerState>>,
) -> impl IntoResponse {
    let settings = state.settings.clone();
    ws.on_upgrade(move |socket| handle_ws(socket, settings))
}

async fn handle_ws(socket: WebSocket, settings: Settings) {
    log::info!("Browser WebSocket connected");
    run_conversation(socket, settings).await;
    log::info!("Browser WebSocket disconnected");
}
