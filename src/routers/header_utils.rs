use axum::body::Body;
use axum::extract::Request;
use axum::http::HeaderMap;

pub use crate::otel_http::TRACE_HEADER_NAMES;

/// Preserve end-to-end header values, including repeated and encoded headers.
/// Framing and headers nominated by Connection belong to this HTTP hop.
pub fn proxy_headers(source: &HeaderMap) -> HeaderMap {
    let connection_headers: Vec<String> = source
        .get_all("connection")
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| {
            value
                .split(',')
                .map(|name| name.trim().to_ascii_lowercase())
        })
        .collect();
    let mut headers = HeaderMap::new();
    for (name, value) in source {
        if !matches!(
            name.as_str(),
            "connection"
                | "keep-alive"
                | "proxy-authenticate"
                | "proxy-authorization"
                | "te"
                | "trailer"
                | "trailers"
                | "transfer-encoding"
                | "upgrade"
                | "content-length"
                | "host"
        ) && !connection_headers.iter().any(|hop| hop == name.as_str())
        {
            headers.append(name.clone(), value.clone());
        }
    }
    headers
}

/// Copy request headers to a Vec of name-value string pairs
/// Used for forwarding headers to backend workers
pub fn copy_request_headers(req: &Request<Body>) -> Vec<(String, String)> {
    req.headers()
        .iter()
        .filter_map(|(name, value)| {
            // Convert header value to string, skipping non-UTF8 headers
            value
                .to_str()
                .ok()
                .map(|v| (name.to_string(), v.to_string()))
        })
        .collect()
}

/// Convert headers from reqwest Response to axum HeaderMap
/// Filters out hop-by-hop headers that shouldn't be forwarded
pub fn preserve_response_headers(reqwest_headers: &HeaderMap) -> HeaderMap {
    let mut headers = HeaderMap::new();

    for (name, value) in reqwest_headers.iter() {
        // Skip hop-by-hop headers that shouldn't be forwarded
        let name_str = name.as_str().to_lowercase();
        if should_forward_header(&name_str) {
            // The original name and value are already valid, so we can just clone them
            headers.insert(name.clone(), value.clone());
        }
    }

    headers
}

/// Determine if a header should be forwarded from backend to client
fn should_forward_header(name: &str) -> bool {
    // List of headers that should NOT be forwarded (hop-by-hop headers)
    !matches!(
        name,
        "connection" |
        "keep-alive" |
        "proxy-authenticate" |
        "proxy-authorization" |
        "te" |
        "trailers" |
        "transfer-encoding" |
        "upgrade" |
        "content-encoding" | // Let axum/hyper handle encoding
        "host" // Should not forward the backend's host header
    )
}

/// Propagate OpenTelemetry trace headers to a reqwest RequestBuilder
///
/// When OTel is enabled: actively injects the current span's trace context,
/// making the router's span the parent of the backend request's span.
/// When OTel is disabled: passively forwards existing trace headers from
/// the incoming request.
pub fn propagate_trace_headers(
    request: reqwest::RequestBuilder,
    headers: Option<&HeaderMap>,
) -> reqwest::RequestBuilder {
    crate::otel_http::propagate_trace_headers(request, headers)
}

/// Propagate specific headers from incoming request to outgoing reqwest RequestBuilder
///
/// This is a general-purpose helper for selectively forwarding headers by name.
/// Only headers whose names match the provided list (case-insensitive) are propagated.
///
/// # Arguments
/// * `request` - The reqwest RequestBuilder to add headers to
/// * `headers` - Optional incoming headers to check
/// * `header_names` - List of header names to propagate (matched case-insensitively)
///
/// # Returns
/// The RequestBuilder with matching headers added
pub fn propagate_headers(
    mut request: reqwest::RequestBuilder,
    headers: Option<&HeaderMap>,
    header_names: &[&str],
) -> reqwest::RequestBuilder {
    if let Some(h) = headers {
        for &name in header_names {
            if let Some(value) = h.get(name) {
                request = request.header(name, value);
            }
        }
    }
    request
}
