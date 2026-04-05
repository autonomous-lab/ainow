//! Model manager -- starts/stops llama-server with different GGUF models.
//!
//! Manages the llama-server process lifecycle:
//! - Start/stop llama-server as a child process
//! - Health check polling
//! - Auto-download from GitHub releases (llama-server binary)
//! - Auto-download models from HuggingFace
//! - Kill existing process on port

use anyhow::{bail, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use tokio::process::{Child, Command};

const LLAMA_SERVER_PORT: u16 = 8080;

/// Configuration for a model.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub name: String,
    pub model_path: Option<String>,
    pub mmproj_path: Option<String>,
    pub hf_repo: Option<String>,
    pub hf_files: Vec<String>,
    pub ctx: Option<String>,
    /// If true, this is an online model (no llama-server needed).
    pub online: bool,
    pub base_url: Option<String>,
    pub api_key_env: Option<String>,
    pub model_id: Option<String>,
}

/// Common llama-server arguments.
const COMMON_ARGS: &[&str] = &[
    "-c", "262144",
    "-np", "1",
    "--fit", "on",
    "--fit-target", "1024",
    "-fa", "on",
    "-t", "20",
    "--no-mmap",
    "--jinja",
    "-ctk", "q4_0",
    "-ctv", "q4_0",
    "--reasoning", "off",
    "--reasoning-budget", "0",
];

/// Build the default model registry.
pub fn build_models(models_dir: &str) -> HashMap<String, ModelConfig> {
    let md = |subdir: &str, file: &str| -> String {
        Path::new(models_dir).join(subdir).join(file).to_string_lossy().to_string()
    };

    let mut models = HashMap::new();

    models.insert("qwen3.5-0.8b".to_string(), ModelConfig {
        name: "Qwen 0.8B".to_string(),
        model_path: Some(md("Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q8_0.gguf")),
        mmproj_path: Some(md("Qwen3.5-0.8B-GGUF", "mmproj-Qwen3.5-0.8B-BF16.gguf")),
        hf_repo: Some("lmstudio-community/Qwen3.5-0.8B-GGUF".to_string()),
        hf_files: vec!["Qwen3.5-0.8B-Q8_0.gguf".to_string(), "mmproj-Qwen3.5-0.8B-BF16.gguf".to_string()],
        ctx: None, online: false, base_url: None, api_key_env: None, model_id: None,
    });

    models.insert("qwen3.5-4b".to_string(), ModelConfig {
        name: "Qwen 4B".to_string(),
        model_path: Some(md("Qwen3.5-4B-GGUF", "Qwen3.5-4B-Q4_K_M.gguf")),
        mmproj_path: Some(md("Qwen3.5-4B-GGUF", "mmproj-Qwen3.5-4B-BF16.gguf")),
        hf_repo: Some("lmstudio-community/Qwen3.5-4B-GGUF".to_string()),
        hf_files: vec!["Qwen3.5-4B-Q4_K_M.gguf".to_string(), "mmproj-Qwen3.5-4B-BF16.gguf".to_string()],
        ctx: None, online: false, base_url: None, api_key_env: None, model_id: None,
    });

    models.insert("qwen3.5-9b".to_string(), ModelConfig {
        name: "Qwen 9B".to_string(),
        model_path: Some(md("Qwen3.5-9B-GGUF", "Qwen3.5-9B-UD-Q4_K_XL.gguf")),
        mmproj_path: Some(md("Qwen3.5-9B-GGUF", "mmproj-Qwen3.5-9B-BF16.gguf")),
        hf_repo: Some("lmstudio-community/Qwen3.5-9B-GGUF".to_string()),
        hf_files: vec!["Qwen3.5-9B-UD-Q4_K_XL.gguf".to_string(), "mmproj-Qwen3.5-9B-BF16.gguf".to_string()],
        ctx: None, online: false, base_url: None, api_key_env: None, model_id: None,
    });

    models.insert("qwen3.5-27b".to_string(), ModelConfig {
        name: "Qwen 27B".to_string(),
        model_path: Some(md("Qwen3.5-27B-GGUF", "Qwen3.5-27B-UD-IQ3_XXS.gguf")),
        mmproj_path: Some(md("Qwen3.5-27B-GGUF", "mmproj-BF16.gguf")),
        hf_repo: Some("lmstudio-community/Qwen3.5-27B-GGUF".to_string()),
        hf_files: vec!["Qwen3.5-27B-UD-IQ3_XXS.gguf".to_string(), "mmproj-BF16.gguf".to_string()],
        ctx: Some("32768".to_string()),
        online: false, base_url: None, api_key_env: None, model_id: None,
    });

    models.insert("online".to_string(), ModelConfig {
        name: "Online".to_string(),
        model_path: None, mmproj_path: None,
        hf_repo: None, hf_files: vec![],
        ctx: None, online: true,
        base_url: Some("https://openrouter.ai/api/v1".to_string()),
        api_key_env: Some("ONLINE_API_KEY".to_string()),
        model_id: Some("google/gemini-3.1-flash-lite-preview".to_string()),
    });

    models
}

/// Short aliases for model IDs.
pub fn model_aliases() -> HashMap<String, String> {
    let mut m = HashMap::new();
    m.insert("0.8b".to_string(), "qwen3.5-0.8b".to_string());
    m.insert("4b".to_string(), "qwen3.5-4b".to_string());
    m.insert("9b".to_string(), "qwen3.5-9b".to_string());
    m.insert("27b".to_string(), "qwen3.5-27b".to_string());
    m.insert("online".to_string(), "online".to_string());
    m
}

/// Manages the llama-server process.
pub struct ModelManager {
    process: Option<Child>,
    current_model: Option<String>,
    llama_server_exe: String,
    models_dir: String,
    port: u16,
}

impl ModelManager {
    pub fn new(llama_server_exe: String, models_dir: String) -> Self {
        Self {
            process: None,
            current_model: None,
            llama_server_exe,
            models_dir,
            port: LLAMA_SERVER_PORT,
        }
    }

    pub fn current_model(&self) -> Option<&str> {
        self.current_model.as_deref()
    }

    /// Check if llama-server is healthy.
    async fn check_health(&self) -> bool {
        let url = format!("http://localhost:{}/health", self.port);
        match reqwest::Client::new()
            .get(&url)
            .timeout(std::time::Duration::from_secs(2))
            .send()
            .await
        {
            Ok(resp) => resp.status().is_success(),
            Err(_) => false,
        }
    }

    /// Stop the current llama-server process.
    pub async fn stop(&mut self) {
        if let Some(mut proc) = self.process.take() {
            let _ = proc.kill().await;
            let _ = proc.wait().await;
            log::info!("llama-server stopped");
        } else {
            // Try to kill whatever is on the port (Windows)
            self.kill_port_process().await;
        }
        self.current_model = None;
    }

    /// Kill any process listening on our port (Windows).
    async fn kill_port_process(&self) {
        if cfg!(windows) {
            let output = Command::new("cmd")
                .args(["/C", &format!("netstat -ano | findstr :{} | findstr LISTENING", self.port)])
                .stdout(Stdio::piped())
                .stderr(Stdio::null())
                .output()
                .await;

            if let Ok(output) = output {
                let text = String::from_utf8_lossy(&output.stdout);
                for line in text.lines() {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if let Some(pid_str) = parts.last() {
                        if let Ok(pid) = pid_str.parse::<u32>() {
                            let _ = Command::new("taskkill")
                                .args(["/PID", &pid.to_string(), "/F"])
                                .stdout(Stdio::null())
                                .stderr(Stdio::null())
                                .output()
                                .await;
                            log::info!("Killed existing process on port {} (PID {})", self.port, pid);
                            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                        }
                    }
                }
            }
        }
    }

    /// Ensure the llama-server binary exists, downloading if necessary.
    async fn ensure_llama_server(&mut self) -> Result<()> {
        let exe_path = Path::new(&self.llama_server_exe);

        if exe_path.is_file() {
            return Ok(());
        }

        // Only auto-download if using default name (not a custom path)
        if exe_path.parent().is_some() && exe_path.parent() != Some(Path::new("")) {
            bail!("llama-server not found at: {}", self.llama_server_exe);
        }

        log::info!("llama-server not found, downloading latest release...");

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()?;

        let resp = client
            .get("https://api.github.com/repos/ggml-org/llama.cpp/releases/latest")
            .header("User-Agent", "AINow")
            .send()
            .await?;

        let release: serde_json::Value = resp.json().await?;
        let tag = release["tag_name"].as_str().unwrap_or("unknown");

        let system = if cfg!(windows) {
            "win"
        } else if cfg!(target_os = "linux") {
            "linux"
        } else {
            "darwin"
        };

        let assets = release["assets"].as_array().ok_or_else(|| anyhow::anyhow!("No assets in release"))?;

        let mut target = None;
        let mut cudart_target = None;
        for asset in assets {
            let name = asset["name"].as_str().unwrap_or("");
            if name.contains(system) && name.contains("cuda") && name.ends_with(".zip") {
                if !name.contains("cudart") {
                    target = Some(asset);
                } else {
                    cudart_target = Some(asset);
                }
            }
        }

        let target = target.ok_or_else(|| anyhow::anyhow!("No suitable llama-server release found for {system}"))?;

        let install_dir = Path::new(&self.models_dir)
            .parent()
            .unwrap_or(Path::new("."))
            .join("llama-server");
        tokio::fs::create_dir_all(&install_dir).await?;

        for asset in [Some(target), cudart_target].into_iter().flatten() {
            let url = asset["browser_download_url"].as_str().unwrap_or("");
            let name = asset["name"].as_str().unwrap_or("download.zip");
            let zip_path = install_dir.join(name);

            log::info!("Downloading {name}...");
            let resp = reqwest::Client::new()
                .get(url)
                .send()
                .await?;
            let bytes = resp.bytes().await?;
            tokio::fs::write(&zip_path, &bytes).await?;

            // Extract zip using system unzip or PowerShell
            if cfg!(windows) {
                let _ = Command::new("powershell")
                    .args([
                        "-Command",
                        &format!(
                            "Expand-Archive -Path '{}' -DestinationPath '{}' -Force",
                            zip_path.display(),
                            install_dir.display()
                        ),
                    ])
                    .output()
                    .await?;
            } else {
                let _ = Command::new("unzip")
                    .args(["-o", &zip_path.to_string_lossy(), "-d", &install_dir.to_string_lossy()])
                    .output()
                    .await?;
            }
            tokio::fs::remove_file(&zip_path).await?;
        }

        let exe_name = if cfg!(windows) {
            "llama-server.exe"
        } else {
            "llama-server"
        };
        let installed_path = install_dir.join(exe_name);
        if !installed_path.is_file() {
            bail!("Download succeeded but {exe_name} not found in {}", install_dir.display());
        }

        self.llama_server_exe = installed_path.to_string_lossy().to_string();
        log::info!("llama-server {tag} installed to {}", install_dir.display());
        Ok(())
    }

    /// Ensure model files exist, downloading from HuggingFace if necessary.
    async fn ensure_model_files(&self, config: &ModelConfig) -> Result<()> {
        let hf_repo = match &config.hf_repo {
            Some(r) => r,
            None => return Ok(()),
        };

        let subfolder = hf_repo.split('/').last().unwrap_or(hf_repo);
        let local_dir = Path::new(&self.models_dir).join(subfolder);

        for filename in &config.hf_files {
            let local_path = local_dir.join(filename);
            if local_path.exists() {
                continue;
            }

            log::info!("Downloading {filename} from {hf_repo}...");
            tokio::fs::create_dir_all(&local_dir).await?;

            let url = format!(
                "https://huggingface.co/{}/resolve/main/{}",
                hf_repo, filename
            );

            let client = reqwest::Client::builder()
                .redirect(reqwest::redirect::Policy::limited(10))
                .build()?;
            let resp = client.get(&url).send().await?;

            if !resp.status().is_success() {
                bail!("Failed to download {filename}: {}", resp.status());
            }

            let bytes = resp.bytes().await?;
            tokio::fs::write(&local_path, &bytes).await?;
            log::info!("Downloaded {filename}");
        }

        Ok(())
    }

    /// Start llama-server with the given model.
    pub async fn start(&mut self, model_id: &str) -> Result<()> {
        let models = build_models(&self.models_dir);
        let config = models
            .get(model_id)
            .ok_or_else(|| anyhow::anyhow!("Unknown model: {model_id}"))?
            .clone();

        if config.online {
            log::info!("Online model selected, no llama-server needed");
            self.current_model = Some(model_id.to_string());
            return Ok(());
        }

        // Auto-download llama-server and model files if missing
        self.ensure_llama_server().await?;
        self.ensure_model_files(&config).await?;

        log::info!("Starting llama-server with {}...", config.name);

        // Stop any existing server
        if self.check_health().await {
            log::info!("Stopping existing llama-server...");
            self.stop().await;
        }

        // Build command
        let model_path = config.model_path.as_deref().unwrap_or("");
        let mut args: Vec<String> = vec!["-m".to_string(), model_path.to_string()];
        if let Some(mmproj) = &config.mmproj_path {
            args.push("--mmproj".to_string());
            args.push(mmproj.clone());
        }
        for arg in COMMON_ARGS {
            args.push(arg.to_string());
        }
        args.push("--port".to_string());
        args.push(self.port.to_string());

        // Per-model context override
        if let Some(ctx) = &config.ctx {
            if let Some(pos) = args.iter().position(|a| a == "-c") {
                if pos + 1 < args.len() {
                    args[pos + 1] = ctx.clone();
                }
            }
        }

        let child = Command::new(&self.llama_server_exe)
            .args(&args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;

        self.process = Some(child);

        // Wait for health check
        let max_wait = 120;
        let poll_interval = std::time::Duration::from_secs(1);
        let mut elapsed = 0u64;

        while elapsed < max_wait {
            tokio::time::sleep(poll_interval).await;
            elapsed += 1;

            // Check if process exited
            if let Some(ref mut proc) = self.process {
                if let Ok(Some(status)) = proc.try_wait() {
                    bail!("llama-server exited unexpectedly (code {:?})", status.code());
                }
            }

            if self.check_health().await {
                self.current_model = Some(model_id.to_string());
                log::info!("llama-server ready with {} ({elapsed}s)", config.name);
                return Ok(());
            }

            if elapsed % 10 == 0 {
                log::info!("  Waiting for llama-server... ({elapsed}s)");
            }
        }

        bail!("llama-server did not become healthy within {max_wait}s");
    }
}
