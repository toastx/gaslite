//! Generic text helpers shared across modules.

/// Extracts code from a DeepSeek response, keeping only the contents of
/// ```` ```solidity ````/```` ```yul ````/```` ``` ```` fenced blocks.
/// Handles multi-block responses where each function is wrapped separately.
/// If no fences are present, returns the trimmed input unchanged.
pub fn strip_code_fences(s: &str) -> String {
    let s = s.trim();
    if !s.contains("```") {
        return s.to_string();
    }
    let mut result: Vec<&str> = Vec::new();
    let mut in_fence = false;
    let mut found_fence = false;
    for line in s.lines() {
        let t = line.trim();
        if t == "```" || t.starts_with("```solidity") || t.starts_with("```yul") {
            found_fence = true;
            in_fence = !in_fence;
            continue;
        }
        if in_fence {
            result.push(line);
        }
    }
    if found_fence && !result.is_empty() {
        result.join("\n").trim().to_string()
    } else {
        s.to_string()
    }
}
