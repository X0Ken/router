mod common;

use axum::{
    body::{to_bytes, Body},
    http::{Request, StatusCode},
    routing::get,
    Json,
};
use common::{create_test_context, test_app::create_test_app};
use reqwest::Client;
use serde_json::{json, Value};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use tower::ServiceExt;
use vllm_router_rs::{
    config::{RouterConfig, RoutingMode},
    core::{BasicWorker, Worker, WorkerType},
    routers::{RouterFactory, RouterTrait},
};

fn model_request() -> Request<Body> {
    Request::builder()
        .uri("/v1/models")
        .body(Body::empty())
        .unwrap()
}

async fn empty_router(config: &RouterConfig) -> Arc<dyn RouterTrait> {
    Arc::from(
        RouterFactory::create_regular_router(&[], &create_test_context(config.clone()))
            .await
            .unwrap(),
    )
}

#[tokio::test]
async fn fixed_models_are_available_without_workers() {
    let config = RouterConfig {
        served_model_name: Some("dsv4".into()),
        ..Default::default()
    };
    let app = create_test_app(empty_router(&config).await, Client::new(), &config);
    let response = app.oneshot(model_request()).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()["content-type"], "application/json");
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), 4096).await.unwrap()).unwrap();
    assert_eq!(
        body,
        json!({"object":"list","data":[{
            "id":"dsv4", "object":"model", "created":0, "owned_by":"vllm"
        }]})
    );
}

#[tokio::test]
async fn fixed_models_skip_unhealthy_workers_and_default_still_proxies() {
    let calls = Arc::new(AtomicUsize::new(0));
    let count = calls.clone();
    let worker_app = axum::Router::new().route(
        "/v1/models",
        get(move || {
            count.fetch_add(1, Ordering::SeqCst);
            async { Json(json!({"object":"list","data":[{"id":"backend-model"}]})) }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move { axum::serve(listener, worker_app).await.unwrap() });
    let mut config = RouterConfig {
        served_model_name: Some("dsv4".into()),
        ..Default::default()
    };
    let ctx = create_test_context(config.clone());
    let router: Arc<dyn RouterTrait> = Arc::from(
        RouterFactory::create_regular_router(&[], &ctx)
            .await
            .unwrap(),
    );
    let worker = Arc::new(BasicWorker::new(url, WorkerType::Regular));
    worker.set_healthy(false);
    ctx.worker_registry.register(worker);

    let app = create_test_app(router.clone(), Client::new(), &config);
    let response = app.oneshot(model_request()).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(calls.load(Ordering::SeqCst), 0);

    config.served_model_name = None;
    let app = create_test_app(router, Client::new(), &config);
    let response = app.oneshot(model_request()).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), 4096).await.unwrap()).unwrap();
    assert_eq!(body["data"][0]["id"], "backend-model");
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    task.abort();
}

#[tokio::test]
async fn fixed_models_do_not_bypass_authorization() {
    let config = RouterConfig {
        served_model_name: Some("dsv4".into()),
        api_key_validation_urls: vec!["http://127.0.0.1:1/validate".into()],
        ..Default::default()
    };
    let app = create_test_app(empty_router(&config).await, Client::new(), &config);
    let response = app.oneshot(model_request()).await.unwrap();
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
}

#[test]
fn validate_fixed_model_config_and_legacy_serialization() {
    let mut config = RouterConfig {
        mode: RoutingMode::Regular {
            worker_urls: vec!["http://127.0.0.1:8900".into()],
        },
        served_model_name: Some("org/model-v1".into()),
        ..Default::default()
    };
    config.validate().unwrap();
    for name in ["", " ", " model", "model ", "model\nname"] {
        config.served_model_name = Some(name.into());
        assert!(
            config.validate().is_err(),
            "accepted invalid model name: {name:?}"
        );
    }
    config.served_model_name = Some("dsv4".into());
    config.enable_igw = true;
    assert!(config.validate().is_err());
    let mut serialized = serde_json::to_value(config).unwrap();
    serialized
        .as_object_mut()
        .unwrap()
        .remove("served_model_name");
    let legacy: RouterConfig = serde_json::from_value(serialized).unwrap();
    assert_eq!(legacy.served_model_name, None);
}
