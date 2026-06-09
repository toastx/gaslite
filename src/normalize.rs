//! Code normalization — a GasAgent-style "Seeker" preprocessing layer.
//!
//! Two canonical forms, both produced by tokenizing with the solang lexer (which
//! drops comments and is not fooled by strings/whitespace):
//!
//! - [`lexical_key`] — comments stripped, whitespace collapsed, **identifiers and
//!   literals kept**. Used as the result-cache key so two submissions that differ
//!   only in formatting/comments collapse to one entry. Safe to return the cached
//!   *output* for, because the optimized code never echoes the input's formatting.
//!
//! - [`structural`] — additionally maps every user identifier to `ID`, numbers to
//!   `NUM`, and string/hex/address literals to placeholders, while keeping EVM/
//!   Solidity builtins (`msg`, `sender`, `sload`, `keccak256`, …) as anchors. This
//!   is a *structural fingerprint*: α-equivalent code (same shape, different names)
//!   normalizes identically. Used only for **detection/matching** ([`PatternMatcher`]),
//!   never to reuse output — matching is safe, output reuse across renames is not.

use solang_parser::lexer::{Lexer, Token};
use solang_parser::pt::Comment;

/// Templates shorter than this (in tokens) are too generic to match on — skipped.
const MIN_TEMPLATE_TOKENS: usize = 8;
/// Cap structural matches per function so retrieval stays bounded.
pub const MAX_STRUCT_MATCHES: usize = 3;

/// EVM/Solidity globals + Yul opcodes that lex as identifiers — kept as anchors in
/// the structural form so matching doesn't collapse `msg.sender`/`sstore` to `ID`.
const BUILTINS: &[&str] = &[
    // Solidity globals / members
    "msg", "sender", "value", "data", "sig", "tx", "origin", "gasprice", "block",
    "timestamp", "number", "coinbase", "prevrandao", "difficulty", "gaslimit",
    "chainid", "basefee", "this", "super", "gasleft", "now",
    // Solidity builtin functions
    "require", "revert", "assert", "keccak256", "sha256", "ripemd160", "ecrecover",
    "addmod", "mulmod", "selfdestruct", "abi", "encode", "encodePacked", "decode",
    "encodeWithSelector", "encodeWithSignature", "slot", "offset", "length",
    // Yul / inline-assembly opcodes
    "add", "sub", "mul", "div", "sdiv", "mod", "smod", "exp", "not", "lt", "gt",
    "slt", "sgt", "eq", "iszero", "and", "or", "xor", "byte", "shl", "shr", "sar",
    "addmod", "mulmod", "signextend", "sload", "sstore", "mload", "mstore",
    "mstore8", "msize", "gas", "caller", "callvalue", "calldataload", "calldatasize",
    "calldatacopy", "codesize", "codecopy", "extcodesize", "extcodecopy", "returndatasize",
    "returndatacopy", "create", "create2", "call", "callcode", "delegatecall",
    "staticcall", "return", "selfbalance", "balance", "origin", "gasprice",
    "blockhash", "coinbase", "pop", "log0", "log1", "log2", "log3", "log4",
];

fn is_builtin(id: &str) -> bool {
    BUILTINS.contains(&id)
}

/// Tokenize `src`, mapping each token to a canonical string. Comments are dropped
/// by the lexer. When `structural`, identifiers/literals are abstracted.
fn canon_tokens(src: &str, structural: bool) -> Vec<String> {
    let mut comments: Vec<Comment> = Vec::new();
    let mut errors = Vec::new();
    let lexer = Lexer::new(src, 0, &mut comments, &mut errors);

    lexer
        .filter_map(|spanned| spanned.ok())
        .map(|(_, tok, _)| {
            if !structural {
                return tok.to_string();
            }
            match tok {
                Token::Identifier(id) if !is_builtin(id) => "ID".to_string(),
                Token::Number(..) | Token::RationalNumber(..) | Token::HexNumber(..) => {
                    "NUM".to_string()
                }
                Token::StringLiteral(..) => "STR".to_string(),
                Token::HexLiteral(..) => "HEX".to_string(),
                Token::AddressLiteral(..) => "ADDR".to_string(),
                other => other.to_string(),
            }
        })
        .collect()
}

/// Cache key: comments stripped + whitespace collapsed, identifiers/literals kept.
pub fn lexical_key(src: &str) -> String {
    canon_tokens(src, false).join(" ")
}

/// Structural fingerprint: identifiers → `ID`, numbers → `NUM`, literals abstracted,
/// builtins kept. α-equivalent code yields the same string.
pub fn structural(src: &str) -> String {
    canon_tokens(src, true).join(" ")
}

/// A deterministic "Seeker" matcher: holds the structural form of each known
/// gas-pattern's `solidity_before`, and finds which patterns a function's
/// structure contains — regardless of variable names or formatting.
#[derive(Default)]
pub struct PatternMatcher {
    /// `(pattern_id, structural_template)` — token-joined, space-padded for
    /// boundary-safe substring matching.
    templates: Vec<(String, String)>,
}

impl PatternMatcher {
    /// Build from `(pattern_id, solidity_before)` pairs. Templates with too few
    /// tokens are dropped as too generic.
    pub fn build(patterns: impl IntoIterator<Item = (String, String)>) -> Self {
        let mut templates = Vec::new();
        for (id, before) in patterns {
            let toks = canon_tokens(&before, true);
            if toks.len() >= MIN_TEMPLATE_TOKENS {
                // Pad so matching respects token boundaries.
                templates.push((id, format!(" {} ", toks.join(" "))));
            }
        }
        Self { templates }
    }

    pub fn len(&self) -> usize {
        self.templates.len()
    }

    pub fn is_empty(&self) -> bool {
        self.templates.is_empty()
    }

    /// Pattern ids whose structural template appears in `fn_src` (capped at
    /// [`MAX_STRUCT_MATCHES`]).
    pub fn match_function(&self, fn_src: &str) -> Vec<String> {
        let hay = format!(" {} ", structural(fn_src));
        self.templates
            .iter()
            .filter(|(_, tpl)| hay.contains(tpl.as_str()))
            .map(|(id, _)| id.clone())
            .take(MAX_STRUCT_MATCHES)
            .collect()
    }
}
