# Gaslite — Web UI

The **Gaslite IDE**: a side-by-side Solidity diff (original ⟷ Gaslite-optimized)
backed by the Monaco editor, with an animated savings rail — headline donut,
deploy / per-call meters, estimated cost saved, and a live log-log gas
simulation. Hover any changed line in the optimized pane to see _why_ it's
cheaper and how much gas it saves.

Built as a **Bun + React + TypeScript** app using Bun's native bundler and dev
server (no Vite / webpack). Monaco is lazy-loaded from a CDN at runtime so its
web workers stay out of the bundle.

## Develop

```bash
bun install
bun run dev        # hot-reloading dev server (defaults to http://localhost:3000)
```

## Build

```bash
bun run build      # -> dist/  (minified, source-mapped)
bun run preview    # build, then serve dist/ statically
bun run typecheck  # tsc --noEmit
```

## Structure

```
index.html          entry HTML — loads src/styles.css and src/main.tsx
src/
  main.tsx          React root
  App.tsx           top bar, diff + idle/analyzing overlays, rail wiring
  Rail.tsx          savings summary: donut, meters, cost, simulation
  MonacoDiff.tsx    Monaco diff editor: sol grammar, theme, hover-to-explain
  monaco.ts         lazy CDN (AMD) loader for monaco-editor
  lib.tsx           formatting, count-up hook, Donut, BarMeter, SavingsGraph
  data.ts           demo contract sources, optimization reasons, gas model
  types.ts          shared Phase type
  styles.css        full app stylesheet
```

## How the demo flows

`App` holds a `phase` state: `idle → analyzing → done`. Pressing **Optimize
contract** kicks off a short "analyzing" animation (scanline + technique chips),
then reveals the optimized source in the right Monaco pane, anchors an accent bar
+ hover popover on each changed line (`data.ts → OPTIMIZATIONS`), and animates the
savings rail in. The editor is frozen (read-only, no scroll) until you hit
Optimize, so the view stays still; once optimized it becomes scrollable.
**Reset** returns to the frozen idle state.

All gas/cost numbers come from the `MODEL` in `src/data.ts` and are illustrative
but internally consistent — swap them for real `forge` output to make it live.

> Note: Monaco loads from `cdn.jsdelivr.net` at runtime, so the editor needs
> network access on first render. Everything else is bundled and works offline.
