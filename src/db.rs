//! Turso (libSQL) HTTP client over the Hrana v2 `/v2/pipeline` protocol.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── turso HTTP types ──────────────────────────────────────────────────────────
#[derive(Serialize)]
struct TursoRequest {
    requests: Vec<TursoStatement>,
}

#[derive(Serialize)]
struct TursoStatement {
    #[serde(rename = "type")]
    stmt_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    stmt: Option<TursoStmtInner>, // Now points to an inner struct
}

#[derive(Serialize)]
struct TursoStmtInner {
    sql: String,
    args: Vec<TursoArg>,
}

// Hrana requires args to look like {"type": "text", "value": "foo"}
// This exact serde macro configuration achieves that automatically.
#[derive(Serialize, Clone)]
#[serde(tag = "type", content = "value")]
pub enum TursoArg {
    #[serde(rename = "text")]
    Text(String),
    #[serde(rename = "integer")]
    Integer(String),
    #[serde(rename = "null")]
    Null,
}

#[derive(Deserialize, Debug)]
struct TursoResponse {
    results: Vec<TursoResult>,
}

#[derive(Deserialize, Debug)]
struct TursoResult {
    #[serde(rename = "type")]
    result_type: String,
    response: Option<TursoResultResponse>,
    error: Option<TursoError>,
}

#[derive(Deserialize, Debug)]
struct TursoResultResponse {
    #[serde(rename = "type")]
    response_type: String,
    result: Option<TursoRows>,
}

#[derive(Deserialize, Debug)]
struct TursoRows {
    cols: Vec<TursoCol>,
    rows: Vec<Vec<serde_json::Value>>,
}

#[derive(Deserialize, Debug)]
struct TursoCol {
    name: String,
}

#[derive(Deserialize, Debug)]
struct TursoError {
    message: String,
}

// ── turso HTTP client ─────────────────────────────────────────────────────────
pub struct Turso {
    http: reqwest::Client,
    url: String,
    token: String,
}

impl Turso {
    pub fn new(
        http: reqwest::Client,
        url: String,
        token: String,
    ) -> Self {
        Self { http, url, token }
    }

    async fn pipeline(
        &self,
        sql: &str,
        args: Vec<TursoArg>,
    ) -> Result<TursoResponse, String> {
        let stmts = vec![
            TursoStatement {
                stmt_type: "execute".to_string(),
                stmt: Some(TursoStmtInner {
                    sql: sql.to_string(),
                    args,
                }),
            },
            TursoStatement {
                stmt_type: "close".to_string(),
                stmt: None,
            },
        ];

        let res = self
            .http
            .post(format!(
                "{}/v2/pipeline",
                self.url
            ))
            .bearer_auth(&self.token)
            .json(&TursoRequest { requests: stmts })
            .send()
            .await
            .map_err(|e| format!("Turso request failed: {e}"))?;

        if !res
            .status()
            .is_success()
        {
            return Err(format!(
                "Turso returned status {}",
                res.status()
            ));
        }

        res.json()
            .await
            .map_err(|e| format!("Turso parse error: {e}"))
    }

    pub async fn execute(
        &self,
        sql: &str,
        args: Vec<TursoArg>,
    ) -> Result<(), String> {
        let body = self
            .pipeline(sql, args)
            .await?;

        // check for errors in results
        for result in &body.results {
            if let Some(err) = &result.error {
                return Err(format!(
                    "Turso SQL error: {}",
                    err.message
                ));
            }
        }

        Ok(())
    }

    pub async fn query(
        &self,
        sql: &str,
        args: Vec<TursoArg>,
    ) -> Result<Vec<HashMap<String, serde_json::Value>>, String> {
        let body = self
            .pipeline(sql, args)
            .await?;

        // extract rows from first execute result
        for result in &body.results {
            if let Some(err) = &result.error {
                return Err(format!(
                    "Turso SQL error: {}",
                    err.message
                ));
            }
            if result.result_type == "ok"
                && let Some(resp) = &result.response
                && let Some(rows_data) = &resp.result
            {
                let col_names: Vec<&str> = rows_data
                    .cols
                    .iter()
                    .map(|c| {
                        c.name
                            .as_str()
                    })
                    .collect();

                let rows = rows_data
                    .rows
                    .iter()
                    .map(|row| {
                        col_names
                            .iter()
                            .zip(row.iter())
                            .map(|(col, val)| {
                                (
                                    col.to_string(),
                                    decode_cell(val),
                                )
                            })
                            .collect::<HashMap<String, serde_json::Value>>()
                    })
                    .collect();

                return Ok(rows);
            }
        }

        Ok(vec![])
    }
}

/// Hrana returns each result cell as `{"type": "text"|"integer"|"float"|"null"|
/// "blob", "value": ...}`. Flatten it to a plain JSON value so callers can use
/// `.as_str()` / `.as_i64()`. Cells already in plain form are returned unchanged.
fn decode_cell(cell: &serde_json::Value) -> serde_json::Value {
    let Some(ty) = cell
        .get("type")
        .and_then(|t| t.as_str())
    else {
        return cell.clone();
    };
    let value = cell.get("value");
    match ty {
        "null" => serde_json::Value::Null,
        // Hrana encodes integers as strings to preserve 64-bit precision.
        "integer" => value
            .and_then(|v| v.as_str())
            .and_then(|s| {
                s.parse::<i64>()
                    .ok()
            })
            .map(serde_json::Value::from)
            .or_else(|| value.cloned())
            .unwrap_or(serde_json::Value::Null),
        // text / float / blob → the underlying value.
        _ => value
            .cloned()
            .unwrap_or(serde_json::Value::Null),
    }
}
