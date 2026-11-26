use axum::{
    body::Body,
    http::{header::CONTENT_TYPE, StatusCode},
    response::{IntoResponse, Response},
};
use bytes::Bytes;
use futures_util::StreamExt;
use parking_lot::Mutex;
use serde_json::Value;
use std::{collections::HashMap, future::Future, sync::Arc};
use tokio::sync::{mpsc, oneshot};
use tokio_stream::wrappers::ReceiverStream;

pub(super) struct PrefillPermit {
    busy: Arc<Mutex<HashMap<String, usize>>>,
    worker: String,
}

impl PrefillPermit {
    pub(super) fn acquire(
        busy: Arc<Mutex<HashMap<String, usize>>>,
        locked: &mut HashMap<String, usize>,
        worker: &str,
    ) -> Self {
        *locked.entry(worker.to_owned()).or_default() += 1;
        Self {
            busy,
            worker: worker.to_owned(),
        }
    }
}

impl Drop for PrefillPermit {
    fn drop(&mut self) {
        let mut busy = self.busy.lock();
        if let std::collections::hash_map::Entry::Occupied(mut entry) =
            busy.entry(self.worker.clone())
        {
            if *entry.get() == 1 {
                entry.remove();
            } else {
                *entry.get_mut() -= 1;
            }
        }
    }
}

/// While prefill is unconfirmed, keep observing the admitted upstream request
/// even if the downstream disappears. After prefill, propagate disconnect
/// by dropping the upstream body. The channel only buffers response bytes.
pub(super) async fn forward_until_prefill(
    upstream: impl Future<Output = Response> + Send + 'static,
    permit: PrefillPermit,
) -> Response {
    let (head_tx, head_rx) = oneshot::channel();
    tokio::spawn(async move {
        if head_tx.is_closed() {
            return;
        }

        let mut permit = Some(permit);
        let response = upstream.await;
        if !response.status().is_success() {
            drop(permit.take());
            let _ = head_tx.send(response);
            return;
        }
        let (parts, body) = response.into_parts();
        let mut observer = parts
            .headers
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .filter(|value| {
                value
                    .split(';')
                    .next()
                    .is_some_and(|mime| mime.trim().eq_ignore_ascii_case("text/event-stream"))
            })
            .map(|_| FirstToken::default());
        let (body_tx, body_rx) = mpsc::channel(1);
        let _ = head_tx.send(Response::from_parts(
            parts,
            Body::from_stream(ReceiverStream::new(body_rx)),
        ));
        let mut stream = body.into_data_stream();
        loop {
            let chunk = if permit.is_some() {
                stream.next().await
            } else {
                tokio::select! {
                    _ = body_tx.closed() => break,
                    chunk = stream.next() => chunk,
                }
            };
            let Some(chunk) = chunk else { break };
            if let (Some(parser), Ok(bytes)) = (&mut observer, &chunk) {
                if parser.observe(bytes) {
                    drop(permit.take());
                    observer = None;
                }
            }
            if chunk.is_err() {
                drop(stream);
                drop(permit);
                let _ = body_tx.send(chunk).await;
                return;
            }
            if body_tx.send(chunk).await.is_err() && permit.is_none() {
                break;
            }
        }
        drop(stream);
        drop(permit);
    });
    head_rx
        .await
        .unwrap_or_else(|_| StatusCode::BAD_GATEWAY.into_response())
}

#[derive(Default)]
struct FirstToken {
    line: Vec<u8>,
    data: Vec<u8>,
    skip_event: bool,
}

impl FirstToken {
    fn observe(&mut self, bytes: &Bytes) -> bool {
        // Bound memory for unsupported or unusually large frames, then try
        // again on the next event. Forwarding always preserves original bytes.
        const MAX_EVENT_BYTES: usize = 64 * 1024;
        for part in bytes.split_inclusive(|byte| *byte == b'\n') {
            if self.line.len() + part.len() <= MAX_EVENT_BYTES {
                self.line.extend_from_slice(part);
            } else {
                self.skip_event = true;
                self.line.clear();
                self.line.push(b'!');
            }
            if !part.ends_with(b"\n") {
                continue;
            }
            let mut line = self.line.as_slice();
            if line.ends_with(b"\n") {
                line = &line[..line.len() - 1];
            }
            let line = line.strip_suffix(b"\r").unwrap_or(line);
            if line.is_empty() {
                let first = !self.skip_event
                    && serde_json::from_slice::<Value>(&self.data)
                        .is_ok_and(|data| contains_output(&data));
                self.data.clear();
                self.skip_event = false;
                if first {
                    return true;
                }
            } else if let Some(data) = line.strip_prefix(b"data:") {
                if !self.skip_event && self.data.len() + data.len() < MAX_EVENT_BYTES {
                    self.data.extend_from_slice(data);
                    self.data.push(b'\n');
                } else {
                    self.skip_event = true;
                }
            }
            self.line.clear();
        }
        false
    }
}

fn nonempty_text(value: &Value) -> bool {
    value.as_str().is_some_and(|text| !text.is_empty())
}

fn contains_output(event: &Value) -> bool {
    if let Some(choices) = event["choices"].as_array() {
        return choices.iter().any(|choice| {
            let delta = &choice["delta"];
            nonempty_text(&choice["text"])
                || ["content", "reasoning", "reasoning_content", "refusal"]
                    .iter()
                    .any(|field| nonempty_text(&delta[*field]))
                || delta["tool_calls"]
                    .as_array()
                    .is_some_and(|calls| !calls.is_empty())
                || delta["function_call"].is_object()
        });
    }

    match event["type"].as_str() {
        Some(
            "response.output_text.delta"
            | "response.reasoning_text.delta"
            | "response.reasoning_summary_text.delta"
            | "response.refusal.delta"
            | "response.function_call_arguments.delta",
        ) => nonempty_text(&event["delta"]),
        Some("content_block_delta") => ["text", "thinking", "partial_json"]
            .iter()
            .any(|field| nonempty_text(&event["delta"][*field])),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn disconnect_before_headers_holds_permit_until_non_stream_response_finishes() {
        let busy = Arc::new(Mutex::new(HashMap::new()));
        let permit = PrefillPermit::acquire(busy.clone(), &mut busy.lock(), "worker");
        let (started_tx, started_rx) = oneshot::channel();
        let (finish_tx, finish_rx) = oneshot::channel();
        let downstream = tokio::spawn(forward_until_prefill(
            async move {
                started_tx.send(()).unwrap();
                finish_rx.await.unwrap();
                "complete".into_response()
            },
            permit,
        ));

        started_rx.await.unwrap();
        downstream.abort();
        let _ = downstream.await;
        assert_eq!(busy.lock().get("worker"), Some(&1));
        finish_tx.send(()).unwrap();

        tokio::time::timeout(std::time::Duration::from_secs(2), async {
            while !busy.lock().is_empty() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
    }

    #[test]
    fn first_token_requires_output_and_supports_multiple_protocols() {
        for frame in [
            br#"{"choices":[{"text":"x"}]}"#.as_slice(),
            br#"{"choices":[{"delta":{"tool_calls":[{"index":0}]}}]}"#,
            br#"{"type":"response.reasoning_text.delta","delta":"x"}"#,
            br#"{"type":"content_block_delta","delta":{"partial_json":"{"}}"#,
        ] {
            let mut parser = FirstToken::default();
            assert!(!parser.observe(&Bytes::from_static(
                b": heartbeat\n\ndata: {\"type\":\"response.created\"}\n\ndata: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n"
            )));
            assert!(!parser.observe(&Bytes::from_static(b"data: ")));
            assert!(!parser.observe(&Bytes::copy_from_slice(frame)));
            assert!(parser.observe(&Bytes::from_static(b"\r\n\r\n")));
        }
    }
}
