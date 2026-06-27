//! Contract analysis: category detection + per-function extraction + storage
//! layout. Each named function (with a body) is extracted with its exact source
//! text and byte range, so functions can be optimized concurrently and the
//! results spliced back at their original offsets.

use solang_parser::pt::{ContractPart, FunctionTy, Loc, SourceUnitPart};

pub(crate) struct FunctionInfo {
    pub(crate) name: String,
    pub(crate) source: String,
    pub(crate) start: usize,
    pub(crate) end: usize,
}

/// A function header + body size, for the router's skeleton view.
pub(crate) struct FnSig {
    pub(crate) name: String,
    pub(crate) signature: String,
    pub(crate) size: usize,
}

/// The structural view of a parsed contract. `functions` + `storage_layout` drive
/// the per-function/oneshot agents; `file_decls` + `signatures` are the lightweight
/// "skeleton" the orchestrator routes on (no function bodies).
pub(crate) struct ContractSkeleton {
    pub(crate) category: Option<&'static str>,
    pub(crate) functions: Vec<FunctionInfo>,
    /// State-variable declarations, newline-joined (agent slot-derivation context).
    pub(crate) storage_layout: String,
    /// File-level declarations a function depends on to compile: structs, enums,
    /// custom errors, events, modifiers, user types. Injected into scoped agents so
    /// they don't reference or invent the wrong definitions.
    pub(crate) file_decls: String,
    /// Deterministic, per-contract storage-slot derivations (using `.slot` accessors),
    /// so the model uses THIS contract's actual layout instead of copying a retrieved
    /// pattern's (possibly packed/incompatible) slot scheme.
    pub(crate) slot_guide: String,
    pub(crate) signatures: Vec<FnSig>,
}

impl ContractSkeleton {
    /// Render the skeleton (signatures + sizes + decl summary) for the router — no
    /// function bodies, to keep the routing prompt small and its TTFT low.
    pub(crate) fn render(&self) -> String {
        let mut out = String::new();
        out.push_str("FUNCTIONS (name — body bytes):\n");
        for s in &self.signatures {
            out.push_str(&format!(
                "- {} [{} bytes]: {}\n",
                s.name, s.size, s.signature
            ));
        }
        if !self.storage_layout.is_empty() {
            out.push_str("\nSTATE VARIABLES:\n");
            out.push_str(&self.storage_layout);
            out.push('\n');
        }
        if !self.file_decls.is_empty() {
            out.push_str("\nFILE-LEVEL DECLARATIONS:\n");
            out.push_str(&self.file_decls);
            out.push('\n');
        }
        out
    }
}

pub(crate) fn analyze_contract(source: &str) -> ContractSkeleton {
    let empty = || ContractSkeleton {
        category: detect_category_fallback(source),
        functions: vec![],
        storage_layout: String::new(),
        file_decls: String::new(),
        slot_guide: String::new(),
        signatures: vec![],
    };
    let Ok((su, _)) = solang_parser::parse(source, 0) else {
        return empty();
    };

    let mut category: Option<&'static str> = None;
    let mut functions: Vec<FunctionInfo> = Vec::new();
    let mut signatures: Vec<FnSig> = Vec::new();
    let mut storage_vars: Vec<String> = Vec::new();
    // (name, declaration_text) for state vars — drives the slot-derivation guide.
    let mut state_var_defs: Vec<(String, String)> = Vec::new();
    let mut decls: Vec<String> = Vec::new();

    // Append the source text spanned by `loc` to `decls` (trimmed).
    let push_decl = |loc: Loc, decls: &mut Vec<String>| {
        if let Loc::File(_, s, e) = loc
            && let Some(t) = source.get(s..e)
        {
            decls.push(t.trim().to_string());
        }
    };

    for part in su.0 {
        let SourceUnitPart::ContractDefinition(def) = part else {
            continue;
        };

        // Inheritance → category
        for base in &def.base {
            let base_name = base
                .name
                .identifiers
                .iter()
                .map(|id| id.name.to_lowercase())
                .collect::<Vec<_>>()
                .join(".");
            if category.is_none() {
                category = match base_name.as_str() {
                    s if s.contains("erc721") => Some("erc721"),
                    s if s.contains("erc1155") => Some("erc1155"),
                    s if s.contains("erc20") => Some("erc20"),
                    s if s.contains("erc2981") => Some("erc2981"),
                    _ => None,
                };
            }
        }

        for cp in &def.parts {
            match cp {
                ContractPart::VariableDefinition(var) => {
                    if let Loc::File(_, start, end) = var.loc
                        && let Some(text) = source.get(start..end)
                    {
                        let decl = text.trim().to_string();
                        if let Some(name) = &var.name {
                            state_var_defs.push((name.name.clone(), decl.clone()));
                        }
                        storage_vars.push(decl);
                    }
                },
                // File-level dependencies a function may reference — context only.
                ContractPart::StructDefinition(d) => push_decl(d.loc, &mut decls),
                ContractPart::EnumDefinition(d) => push_decl(d.loc, &mut decls),
                ContractPart::EventDefinition(d) => push_decl(d.loc, &mut decls),
                ContractPart::ErrorDefinition(d) => push_decl(d.loc, &mut decls),
                ContractPart::TypeDefinition(d) => push_decl(d.loc, &mut decls),
                ContractPart::FunctionDefinition(func) => {
                    // Modifiers are dependencies, not optimization targets.
                    if matches!(func.ty, FunctionTy::Modifier) {
                        push_decl(func.loc, &mut decls);
                        continue;
                    }
                    // Optimize only named, bodied `function`s (skip constructor/
                    // fallback/receive and abstract declarations).
                    if !matches!(func.ty, FunctionTy::Function) {
                        continue;
                    }
                    let Some(name_ident) = &func.name else {
                        continue;
                    };
                    let Some(_) = &func.body else { continue };
                    let Loc::File(_, start, end) = func.loc else {
                        continue;
                    };
                    let Some(func_text) = source.get(start..end) else {
                        continue;
                    };
                    // Signature = header up to the body's opening brace.
                    let signature = func_text
                        .split_once('{')
                        .map(|(h, _)| h.trim().to_string())
                        .unwrap_or_else(|| func_text.trim().to_string());
                    signatures.push(FnSig {
                        name: name_ident.name.clone(),
                        signature,
                        size: end.saturating_sub(start),
                    });
                    functions.push(FunctionInfo {
                        name: name_ident.name.clone(),
                        source: func_text.to_string(),
                        start,
                        end,
                    });
                },
                _ => {},
            }
        }
    }

    ContractSkeleton {
        category,
        functions,
        storage_layout: storage_vars.join("\n"),
        file_decls: decls.join("\n\n"),
        slot_guide: build_slot_guide(&state_var_defs),
        signatures,
    }
}

/// Deterministic per-contract storage-slot derivations. Each state variable is
/// emitted with the EXACT inline-assembly recipe for its slot, using `.slot`
/// accessors (so solc resolves the slot index — no hardcoded numbers) and the
/// canonical `keccak256(0x00, 0x40)` mapping derivation. Mapping depth is read from
/// the declaration text. This is what stops a non-reasoning model from copying a
/// retrieved pattern's incompatible (e.g. packed ERC721A) slot scheme.
fn build_slot_guide(vars: &[(String, String)]) -> String {
    if vars.is_empty() {
        return String::new();
    }
    let mut out = String::from(
        "EXACT STORAGE SLOTS FOR THIS CONTRACT — derive every slot with these recipes \
         VERBATIM (use the `.slot` accessor; ignore any slot scheme from the retrieved patterns):\n",
    );
    for (name, decl) in vars {
        let depth = decl.matches("mapping(").count();
        let line = match depth {
            0 => format!(
                "- {name} (value type): read sload({name}.slot), write sstore({name}.slot, v)"
            ),
            1 => format!(
                "- {name}[k]: mstore(0x00, k); mstore(0x20, {name}.slot); let s := keccak256(0x00, 0x40)"
            ),
            2 => format!(
                "- {name}[k1][k2]: mstore(0x00, k1); mstore(0x20, {name}.slot); let inner := keccak256(0x00, 0x40); mstore(0x00, k2); mstore(0x20, inner); let s := keccak256(0x00, 0x40)"
            ),
            _ => format!(
                "- {name}: deep mapping — derive each level as keccak256(key ++ parentSlot)"
            ),
        };
        out.push_str(&line);
        out.push('\n');
    }
    out
}

fn detect_category_fallback(source: &str) -> Option<&'static str> {
    let s = source.to_lowercase();
    if s.contains("is erc721") || s.contains(": erc721") {
        return Some("erc721");
    }
    if s.contains("is erc1155") || s.contains(": erc1155") {
        return Some("erc1155");
    }
    if s.contains("is erc2981") || s.contains(": erc2981") {
        return Some("erc2981");
    }
    if s.contains("is erc20") || s.contains(": erc20") {
        return Some("erc20");
    }
    None
}
