/* Lazy AMD loader for monaco-editor (CDN). Loading Monaco from the CDN keeps
   its web workers out of the bundle — no worker-URL plumbing required. The
   loaded `monaco` global is typed loosely as `any`; the diff component only
   touches a small, stable slice of the API. */

const MONACO_BASE = "https://cdn.jsdelivr.net/npm/monaco-editor@0.52.2/min/";

declare global {
  interface Window {
    __glMonaco?: Promise<any>;
    MonacoEnvironment?: { getWorkerUrl?: () => string; baseUrl?: string };
    require?: any;
  }
}

export function loadMonaco(): Promise<any> {
  if (window.__glMonaco) return window.__glMonaco;
  window.__glMonaco = new Promise((resolve, reject) => {
    window.MonacoEnvironment = {
      getWorkerUrl() {
        return (
          "data:text/javascript;charset=utf-8," +
          encodeURIComponent(
            "self.MonacoEnvironment={baseUrl:'" +
              MONACO_BASE +
              "'};" +
              "importScripts('" +
              MONACO_BASE +
              "vs/base/worker/workerMain.js');",
          )
        );
      },
    };
    const s = document.createElement("script");
    s.src = MONACO_BASE + "vs/loader.js";
    s.onload = () => {
      window.require.config({ paths: { vs: MONACO_BASE + "vs" } });
      window.require(["vs/editor/editor.main"], () => resolve((window as any).monaco), reject);
    };
    s.onerror = reject;
    document.head.appendChild(s);
  });
  return window.__glMonaco;
}
