declare module "*.png" {
  const src: string;
  export default src;
}
declare module "*.svg" {
  const src: string;
  export default src;
}

/* Bun build inlines `import.meta.env`. Declare the specific keys we read and merge
   into the lib `ImportMeta` (interface declaration merging) rather than overriding
   it, so the rest of `import.meta` keeps its real types. */
interface ImportMetaEnv {
  readonly VITE_GASLITE_API?: string;
  readonly VITE_MANTLE_RPC?: string;
}
interface ImportMeta {
  readonly env?: ImportMetaEnv;
}
