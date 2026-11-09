// A browser page cannot bind a socket.
#[cfg(target_arch = "wasm32")]
fn main() {}

// binds a socket and reads the local filesystem, which wasm32 lacks
#[cfg(not(target_arch = "wasm32"))]
fn main() -> std::process::ExitCode {
    let outcome: Result<(), cross_llama_chat::AppError> = server::run();

    match outcome {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            std::process::ExitCode::FAILURE
        }
    }
}

// binds a socket and reads the local filesystem, which wasm32 lacks
#[cfg(not(target_arch = "wasm32"))]
mod server {
    use std::convert::Infallible;
    use std::net::Ipv4Addr;
    use std::path::{Path, PathBuf};

    use axum::Router;
    use axum::body::Body;
    use axum::extract::Request;
    use axum::http::header::{HeaderMap, HeaderName, HeaderValue};
    use axum::http::{StatusCode, Uri};
    use axum::response::{IntoResponse, Response};
    use tokio::net::TcpListener;
    use tokio::runtime::Runtime;
    use tower::ServiceExt;
    use tower_http::services::ServeDir;
    use tower_http::services::fs::ServeFileSystemResponseBody;

    use cross_llama_chat::{AppError, model_file};

    const DEFAULT_PORT: u16 = 4417;
    const WLLAMA_URL_PREFIX: &str = "/wllama/";
    const MODELS_URL_PREFIX: &str = "/models/";
    const CROSS_ORIGIN_OPENER_POLICY: HeaderName = HeaderName::from_static("cross-origin-opener-policy");
    const CROSS_ORIGIN_EMBEDDER_POLICY: HeaderName = HeaderName::from_static("cross-origin-embedder-policy");

    struct Mount {
        url_prefix: &'static str,
        directory: PathBuf,
    }

    pub fn run() -> Result<(), AppError> {
        let port: u16 = read_port()?;
        check_served_files()?;

        let runtime: Runtime = tokio::runtime::Builder::new_current_thread().enable_all().build()?;
        runtime.block_on(serve(port))
    }

    fn read_port() -> Result<u16, AppError> {
        let port_argument: Option<String> = std::env::args().nth(1);

        let Some(port_argument) = port_argument
        else {
            return Ok(DEFAULT_PORT);
        };

        Ok(port_argument.parse::<u16>()?)
    }

    fn check_served_files() -> Result<(), AppError> {
        model_file::check_model_file(&model_file::get_default_model_path())?;

        let site_index_path: PathBuf = get_site_directory().join("index.html");
        let wllama_module_path: PathBuf = get_wllama_directory().join("index.js");

        for required_path in [site_index_path, wllama_module_path] {
            if !required_path.is_file() {
                return Err(AppError::new(&format!(
                    "{} is missing; ./scripts/serve-web.sh builds the site and fetches wllama before serving",
                    required_path.display(),
                )));
            }
        }

        Ok(())
    }

    async fn serve(port: u16) -> Result<(), AppError> {
        let listener: TcpListener = TcpListener::bind((Ipv4Addr::LOCALHOST, port)).await?;
        println!("Serving the chat demo at http://127.0.0.1:{port}/");

        let router: Router = Router::new().fallback(serve_file);
        axum::serve(listener, router).await?;

        Ok(())
    }

    fn get_package_directory() -> &'static Path {
        Path::new(env!("CARGO_MANIFEST_DIR"))
    }

    fn get_site_directory() -> PathBuf {
        get_package_directory().join("dist")
    }

    fn get_wllama_directory() -> PathBuf {
        get_package_directory().join("build").join("wllama")
    }

    async fn serve_file(request: Request) -> Response {
        let file_request: Option<(PathBuf, Request)> = route_to_directory(request);

        let Some((directory, file_request)) = file_request
        else {
            return StatusCode::BAD_REQUEST.into_response();
        };

        // ServeDir answers Range requests and sends Content-Length, which wllama needs to check its cached model.
        let served: Result<axum::http::Response<ServeFileSystemResponseBody>, Infallible> = ServeDir::new(directory).oneshot(file_request).await;
        let Ok(file_response) = served;

        let mut response: Response = file_response.map(Body::new);
        add_cross_origin_isolation_headers(response.headers_mut());

        response
    }

    /// Returns the directory that holds the requested file, and the request with its URL path made relative to that directory.
    fn route_to_directory(mut request: Request) -> Option<(PathBuf, Request)> {
        let mounts: [Mount; 2] = [
            Mount {
                url_prefix: WLLAMA_URL_PREFIX,
                directory: get_wllama_directory(),
            },
            Mount {
                url_prefix: MODELS_URL_PREFIX,
                directory: model_file::get_models_directory(),
            },
        ];
        let request_path: String = request.uri().path().to_string();

        for mount in mounts {
            let Some(path_in_directory) = request_path.strip_prefix(mount.url_prefix)
            else {
                continue;
            };

            let relative_uri: Uri = format!("/{path_in_directory}").parse::<Uri>().ok()?;
            *request.uri_mut() = relative_uri;

            return Some((mount.directory, request));
        }

        Some((get_site_directory(), request))
    }

    /// wllama's multithreaded build needs a cross-origin isolated page.
    fn add_cross_origin_isolation_headers(headers: &mut HeaderMap) {
        headers.insert(CROSS_ORIGIN_OPENER_POLICY, HeaderValue::from_static("same-origin"));
        headers.insert(CROSS_ORIGIN_EMBEDDER_POLICY, HeaderValue::from_static("require-corp"));
    }
}
