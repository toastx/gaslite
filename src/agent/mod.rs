//! LLM layer: the routing orchestrator, the optimization/verification agents, and
//! the rig `Tool` they call to compile candidate rewrites.

pub(crate) mod orchestrator;
pub(crate) mod rig_agent;
pub(crate) mod tools;
pub(crate) mod verify_agent;
