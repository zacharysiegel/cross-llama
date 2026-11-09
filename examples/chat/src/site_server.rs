use std::convert::Infallible;
use std::future::Future;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::pin::Pin;

use any_spawner::{Executor, ExecutorError};
use axum::Router;
use axum::body::Body;
use axum::extract::Request;
use axum::http::header::{HeaderMap, HeaderName, HeaderValue};
use axum::http::{StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use leptos::config::{ConfFile, LeptosOptions};
use tokio::net::TcpListener;
use tower::ServiceExt;
use tower_http::services::ServeDir;
use tower_http::services::fs::ServeFileSystemResponseBody;

use crate::{AppError, chat_page, model_file, site_paths};

const PAGE_PATH: &str = "/";
const CROSS_ORIGIN_OPENER_POLICY: HeaderName = HeaderName::from_static("cross-origin-opener-policy");
const CROSS_ORIGIN_EMBEDDER_POLICY: HeaderName = HeaderName::from_static("cross-origin-embedder-policy");

type PageResponseFuture = Pin<Box<dyn Future<Output = Response> + Send>>;

struct Mount {
    url_prefix: &'static str,
    directory: PathBuf,
}

/// Serves at the address cargo-leptos passes in its `LEPTOS_*` variables.
pub async fn serve() -> Result<(), AppError> {
    check_served_files()?;

    let configuration: ConfFile = leptos::config::get_configuration(None)
        .map_err(|error| AppError::from_error("cannot read cargo-leptos's settings", Box::new(error)))?;
    let leptos_options: LeptosOptions = configuration.leptos_options;
    let address: SocketAddr = leptos_options.site_addr;
    let router: Router = create_router(leptos_options);

    let listener: TcpListener = TcpListener::bind(address).await?;
    println!("Serving the chat demo at http://{address}/");

    axum::serve(listener, router).await?;

    Ok(())
}

pub fn create_router(leptos_options: LeptosOptions) -> Router {
    let site_root: PathBuf = PathBuf::from(leptos_options.site_root.as_ref());
    // leptos renders the page in tasks on this executor; it can be set only once in a process.
    let _executor_initialization: Result<(), ExecutorError> = Executor::init_tokio();
    let render_page = leptos_axum::render_app_to_stream(move || chat_page::shell(leptos_options.clone()));

    Router::new()
        .route(PAGE_PATH, get(move |request: Request| serve_page(render_page(request))))
        .fallback(move |request: Request| serve_file(site_root.clone(), request))
}

fn check_served_files() -> Result<(), AppError> {
    model_file::check_model_file(&model_file::get_default_model_path())?;

    let wllama_module_path: PathBuf = get_wllama_directory().join("index.js");

    if !wllama_module_path.is_file() {
        return Err(AppError::new(&format!(
            "{} is missing; ./scripts/serve-web.sh fetches wllama before serving",
            wllama_module_path.display(),
        )));
    }

    Ok(())
}

fn get_wllama_directory() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("build").join("wllama")
}

async fn serve_page(page_response: PageResponseFuture) -> Response {
    let mut response: Response = page_response.await;
    add_cross_origin_isolation_headers(response.headers_mut());

    response
}

async fn serve_file(site_root: PathBuf, request: Request) -> Response {
    let file_request: Option<(PathBuf, Request)> = route_to_directory(site_root, request);

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
fn route_to_directory(site_root: PathBuf, mut request: Request) -> Option<(PathBuf, Request)> {
    let mounts: [Mount; 2] = [
        Mount {
            url_prefix: site_paths::WLLAMA_URL_PREFIX,
            directory: get_wllama_directory(),
        },
        Mount {
            url_prefix: site_paths::MODELS_URL_PREFIX,
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

    Some((site_root, request))
}

/// wllama's multithreaded build needs a cross-origin isolated page.
fn add_cross_origin_isolation_headers(headers: &mut HeaderMap) {
    headers.insert(CROSS_ORIGIN_OPENER_POLICY, HeaderValue::from_static("same-origin"));
    headers.insert(CROSS_ORIGIN_EMBEDDER_POLICY, HeaderValue::from_static("require-corp"));
}

#[cfg(test)]
mod tests {
    use axum::Router;
    use axum::body::Body;
    use axum::extract::Request;
    use axum::http::{StatusCode, header};
    use axum::response::Response;
    use leptos::config::LeptosOptions;
    use tower::ServiceExt;

    use crate::{model_file, site_paths};

    #[tokio::test]
    async fn create_router_renders_the_page_cross_origin_isolated() {
        let response: Response = send(Request::get(super::PAGE_PATH).body(Body::empty()).unwrap()).await;

        assert_eq!(response.status(), StatusCode::OK);
        assert_cross_origin_isolated(&response);

        let body: axum::body::Bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let page_text: &str = std::str::from_utf8(&body).unwrap();
        assert!(page_text.contains("<h1>cross-llama chat</h1>"));
    }

    #[tokio::test]
    async fn create_router_answers_a_model_range_request_with_its_length() {
        let model_url: String = format!("{}{}", site_paths::MODELS_URL_PREFIX, model_file::MODEL_FILE_NAME);
        let request: Request = Request::get(model_url).header(header::RANGE, "bytes=0-3").body(Body::empty()).unwrap();

        let response: Response = send(request).await;

        assert_eq!(response.status(), StatusCode::PARTIAL_CONTENT);
        assert_eq!(response.headers()[header::CONTENT_LENGTH], "4");
        assert_cross_origin_isolated(&response);

        let body: axum::body::Bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
        assert_eq!(&body[..], b"GGUF");
    }

    async fn send(request: Request) -> Response {
        let router: Router = super::create_router(LeptosOptions::builder().output_name("cross-llama-chat").build());

        router.oneshot(request).await.unwrap()
    }

    fn assert_cross_origin_isolated(response: &Response) {
        assert_eq!(response.headers()[super::CROSS_ORIGIN_OPENER_POLICY], "same-origin");
        assert_eq!(response.headers()[super::CROSS_ORIGIN_EMBEDDER_POLICY], "require-corp");
    }
}
