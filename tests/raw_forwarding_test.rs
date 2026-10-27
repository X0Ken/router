mod common;

use axum::{
    body::{to_bytes, Body, Bytes},
    extract::{Request, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use common::{create_test_context, test_app::create_test_app};
use futures_util::{stream, StreamExt};
use std::sync::{Arc, Mutex};
use tower::ServiceExt;
use vllm_router_rs::{
    config::{PolicyConfig, RetryConfig, RouterConfig},
    core::{BasicWorker, Worker, WorkerType},
    routers::{RouterFactory, RouterTrait},
};

#[derive(Clone, Debug)]
struct Captured {
    uri: String,
    headers: HeaderMap,
    body: Bytes,
}

type Captures = Arc<Mutex<Vec<Captured>>>;

async fn backend(State(captures): State<Captures>, req: Request) -> Response {
    let (parts, body) = req.into_parts();
    let body = to_bytes(body, 1024 * 1024).await.unwrap();
    let scenario = parts
        .headers
        .get("x-scenario")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_owned();
    let count = {
        let mut captures = captures.lock().unwrap();
        captures.push(Captured {
            uri: parts.uri.to_string(),
            headers: parts.headers,
            body: body.clone(),
        });
        captures.len()
    };
    if scenario == "retry" && count == 1 {
        return (StatusCode::SERVICE_UNAVAILABLE, "try another attempt").into_response();
    }
    if scenario == "error" {
        return (
            StatusCode::UNPROCESSABLE_ENTITY,
            [("x-backend-error", "preserved")],
            "backend validation detail",
        )
            .into_response();
    }
    if scenario == "redirect" {
        return (
            StatusCode::TEMPORARY_REDIRECT,
            [("location", "/redirect-target")],
            "original redirect",
        )
            .into_response();
    }
    if scenario == "encoded" {
        return (
            [
                ("content-encoding", "gzip"),
                ("content-type", "application/octet-stream"),
            ],
            body,
        )
            .into_response();
    }
    if scenario == "stream" {
        let chunks = stream::iter([
            Ok::<_, std::io::Error>(Bytes::from_static(b"data: example\n\n")),
            Ok(Bytes::from_static(b"data: [DONE]\n\n")),
        ]);
        return (
            [("content-type", "text/event-stream")],
            Body::from_stream(chunks),
        )
            .into_response();
    }
    if scenario == "hold" {
        let chunks =
            stream::once(async { Ok::<_, std::io::Error>(Bytes::from_static(b"data: first\n\n")) })
                .chain(stream::pending());
        return (
            [("content-type", "text/event-stream")],
            Body::from_stream(chunks),
        )
            .into_response();
    }
    let mut response = ([("content-type", "application/octet-stream")], body).into_response();
    response
        .headers_mut()
        .append("set-cookie", "first=1".parse().unwrap());
    response
        .headers_mut()
        .append("set-cookie", "second=2".parse().unwrap());
    response
}

struct Fixture {
    app: axum::Router,
    captures: Captures,
    worker: Arc<BasicWorker>,
    ctx: Arc<vllm_router_rs::server::AppContext>,
    task: tokio::task::JoinHandle<()>,
}

impl Drop for Fixture {
    fn drop(&mut self) {
        self.task.abort();
    }
}

impl Fixture {
    async fn new(mut config: RouterConfig) -> Self {
        config.policy = PolicyConfig::CacheAware {
            cache_threshold: 0.5,
            balance_abs_threshold: 3,
            balance_rel_threshold: 1.5,
            eviction_interval_secs: 60,
            max_tree_size: 1024,
        };
        config.retry = RetryConfig {
            max_retries: 2,
            initial_backoff_ms: 1,
            max_backoff_ms: 1,
            jitter_factor: 0.0,
            ..Default::default()
        };
        let captures: Captures = Arc::default();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let backend_app = axum::Router::new()
            .fallback(backend)
            .with_state(captures.clone());
        let task = tokio::spawn(async move { axum::serve(listener, backend_app).await.unwrap() });
        let ctx = create_test_context(config.clone());
        let router: Arc<dyn RouterTrait> = Arc::from(
            RouterFactory::create_regular_router(&[], &ctx)
                .await
                .unwrap(),
        );
        let worker = Arc::new(BasicWorker::new(url, WorkerType::Regular));
        ctx.worker_registry.register(worker.clone());
        let app = create_test_app(router, reqwest::Client::new(), &config);
        Self {
            app,
            captures,
            worker,
            ctx,
            task,
        }
    }

    async fn send(&self, path: &str, body: impl Into<Body>, scenario: &str) -> Response {
        self.app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(path)
                    .header("content-type", "application/json; charset=utf-8")
                    .header("x-scenario", scenario)
                    .header("connection", "x-hop-secret")
                    .header("x-hop-secret", "remove-me")
                    .header("x-client-extension", "keep-me")
                    .body(body.into())
                    .unwrap(),
            )
            .await
            .unwrap()
    }
}

#[tokio::test]
async fn preserves_redirects_and_encoded_response_bodies() {
    let f = Fixture::new(RouterConfig::default()).await;
    let response = f.send("/v1/chat/completions", "{}", "redirect").await;
    assert_eq!(response.status(), StatusCode::TEMPORARY_REDIRECT);
    assert_eq!(response.headers()["location"], "/redirect-target");
    assert_eq!(
        to_bytes(response.into_body(), 4096).await.unwrap(),
        "original redirect"
    );
    assert_eq!(f.captures.lock().unwrap().len(), 1);
    let opaque = Bytes::from_static(b"\x1f\x8bopaque-compressed-content");
    let response = f.send("/backend/native", opaque.clone(), "encoded").await;
    assert_eq!(response.headers()["content-encoding"], "gzip");
    assert_eq!(to_bytes(response.into_body(), 4096).await.unwrap(), opaque);
}

#[tokio::test]
async fn keeps_session_routing_while_forwarding_changing_messages() {
    let f = Fixture::new(RouterConfig::default()).await;
    let other = Fixture::new(RouterConfig::default()).await;
    f.ctx.worker_registry.register(other.worker.clone());
    // These workers are registered after fixture construction; initialize the
    // cache trees as server startup does after discovering its worker pool.
    f.ctx
        .policy_registry
        .get_default_policy()
        .init_workers(&f.ctx.worker_registry.get_all());
    for i in 0..10 {
        let body = format!(
            r#"{{"session_params":{{"session_id":"same-session"}},"messages":[{{"role":"developer","content":"changing message {i}"}}]}}"#
        );
        let response = f.send("/v1/chat/completions", body.clone(), "").await;
        assert_eq!(to_bytes(response.into_body(), 4096).await.unwrap(), body);
    }
    let counts = [
        f.captures.lock().unwrap().len(),
        other.captures.lock().unwrap().len(),
    ];
    assert!(
        counts == [10, 0] || counts == [0, 10],
        "session moved between workers: {counts:?}"
    );
}

#[tokio::test]
async fn forwards_native_requests_byte_for_byte_without_schema_validation() {
    let f = Fixture::new(RouterConfig::default()).await;
    let bodies: Vec<&[u8]> = vec![
        br#"{ "model":"dsv4", "messages":[{"role":"developer","content":[{"type":"future","unknown":true}]},{"role":"assistant","content":null,"reasoning_content":"keep","future":{"n":1}}], "tools":[{"type":"new-tool","config":{"x":1}}], "reasoning_effort":73, "temperature":0.1234567890123456789, "n":18446744073709551616, "future":null }"#,
        br#"{"messages":[{"role":"latest_reminder","content":"keep"}],"stream":"backend-specific","unknown":{"array":[1,null,false]}}"#,
        br#"{"messages":[{"role":"custom-future-role"}],"ext":1e400}"#,
        b"{ deliberately invalid JSON",
        b"\x00\xffopaque-body\n",
    ];
    for path in [
        "/v1/chat/completions",
        "/v1/completions",
        "/v1/responses",
        "/v1/messages",
        "/generate",
        "/v1/embeddings",
        "/backend/native",
    ] {
        for body in &bodies {
            let uri = format!("{path}?extension=a%2Fb&extension=c");
            let response = f.send(&uri, Bytes::copy_from_slice(body), "").await;
            assert_eq!(response.status(), StatusCode::OK, "{path}");
            assert_eq!(response.headers().get_all("set-cookie").iter().count(), 2);
            assert_eq!(
                to_bytes(response.into_body(), 1024 * 1024).await.unwrap(),
                *body
            );
            let captures = f.captures.lock().unwrap();
            let captured = captures.last().unwrap();
            assert_eq!(captured.uri, uri);
            assert_eq!(captured.body, *body);
            assert_eq!(
                captured.headers["content-type"],
                "application/json; charset=utf-8"
            );
            assert_eq!(captured.headers["x-client-extension"], "keep-me");
            assert!(!captured.headers.contains_key("x-hop-secret"));
            assert_eq!(f.worker.load(), 0);
        }
    }
}

#[tokio::test]
async fn preserves_backend_errors_even_when_client_requested_streaming() {
    let f = Fixture::new(RouterConfig::default()).await;
    let response = f
        .send(
            "/v1/chat/completions",
            r#"{"stream":true,"messages":[]}"#,
            "error",
        )
        .await;
    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(
        response.headers()["content-type"],
        "text/plain; charset=utf-8"
    );
    assert_eq!(response.headers()["x-backend-error"], "preserved");
    assert_eq!(
        to_bytes(response.into_body(), 4096).await.unwrap(),
        "backend validation detail"
    );
    assert_eq!(f.captures.lock().unwrap().len(), 1);
    assert_eq!(f.worker.load(), 0);
}

#[tokio::test]
async fn retries_replay_original_body_and_release_each_load_increment() {
    let f = Fixture::new(RouterConfig::default()).await;
    let body = "{ \"messages\": [{\"role\":\"developer\",\"content\":\"hi\"}] }";
    let response = f.send("/v1/chat/completions", body, "retry").await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(f.worker.load(), 1);
    assert_eq!(to_bytes(response.into_body(), 4096).await.unwrap(), body);
    assert_eq!(f.worker.load(), 0);
    let captured = f.captures.lock().unwrap();
    assert_eq!(captured.len(), 2);
    assert!(captured.iter().all(|request| request.body == body));
}

#[tokio::test]
async fn streams_without_requiring_a_stream_flag_and_releases_on_disconnect() {
    let f = Fixture::new(RouterConfig::default()).await;
    let response = f.send("/v1/messages", "{}", "stream").await;
    assert_eq!(response.headers()["content-type"], "text/event-stream");
    assert_eq!(f.worker.load(), 1);
    assert_eq!(
        to_bytes(response.into_body(), 4096).await.unwrap(),
        "data: example\n\ndata: [DONE]\n\n"
    );
    assert_eq!(f.worker.load(), 0);
    let response = f.send("/v1/messages", "{}", "hold").await;
    assert_eq!(f.worker.load(), 1);
    drop(response);
    assert_eq!(f.worker.load(), 0);
}

#[tokio::test]
async fn retains_authorization_and_payload_limits() {
    let f = Fixture::new(RouterConfig {
        api_key_validation_urls: vec!["http://127.0.0.1:1/validate".into()],
        ..Default::default()
    })
    .await;
    assert_eq!(
        f.send("/v1/chat/completions", "{}", "").await.status(),
        StatusCode::UNAUTHORIZED
    );
    assert!(f.captures.lock().unwrap().is_empty());
    let f = Fixture::new(RouterConfig {
        max_payload_size: 4,
        ..Default::default()
    })
    .await;
    assert_eq!(
        f.send("/v1/chat/completions", "12345678", "")
            .await
            .status(),
        StatusCode::PAYLOAD_TOO_LARGE
    );
    assert!(f.captures.lock().unwrap().is_empty());
}
