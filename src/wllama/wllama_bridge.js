// Passes JSON strings and plain values between the Rust side and wllama, which it imports from the app's directory.

// wllama's exit frees nothing while loadModel has yet to create its worker, which loadModel then creates regardless.
const pendingLoads = new WeakMap();

export async function createWllama(wllamaDirectoryUrl) {
  const directoryUrl = new URL(wllamaDirectoryUrl.endsWith('/') ? wllamaDirectoryUrl : `${wllamaDirectoryUrl}/`, globalThis.location.href);
  const { Wllama } = await import(new URL('index.js', directoryUrl).href);
  const wllama = new Wllama({ default: new URL('wllama.wasm', directoryUrl).href }, { suppressNativeLog: true, logger: { ...console, debug: () => {} } });
  wllama.setCompat({
    wasm: new URL('compat/wllama.wasm', directoryUrl).href,
    worker: new URL('compat/wllama.js', directoryUrl).href,
  });
  return wllama;
}

export async function downloadModel(wllama, modelUrl, abortSignal, onProgress) {
  const source = { url: new URL(modelUrl, globalThis.location.href).href };
  return await wllama.modelManager.getModelOrDownload(source, {
    signal: abortSignal,
    // Rust aborts before it frees the callback.
    progressCallback: ({ loaded, total }) => {
      if (!abortSignal.aborted) {
        onProgress(loaded, total);
      }
    },
  });
}

// Resolves to fewer bytes than asked for when the file is shorter.
export async function readModelPrefix(model, byteCount) {
  const [firstFile] = await model.open();
  return new Uint8Array(await firstFile.slice(0, byteCount).arrayBuffer());
}

export async function loadModel(wllama, model, loadParametersJson) {
  const loading = wllama.loadModel(model, JSON.parse(loadParametersJson));
  pendingLoads.set(wllama, loading);
  await loading;
  const { n_ctx, list_tokens_eog } = wllama.getLoadedContextInfo();
  return JSON.stringify({ n_ctx, list_tokens_eog });
}

export async function generate(wllama, requestJson, abortSignal, onChunk) {
  await wllama.createChatCompletion({
    ...JSON.parse(requestJson),
    stream: true,
    return_progress: true,
    timings_per_token: true,
    abortSignal,
    // Rust aborts before it frees the callback.
    onData: (chunk) => {
      if (!abortSignal.aborted) {
        onChunk(JSON.stringify(chunk));
      }
    },
  });
}

export async function exit(wllama) {
  await pendingLoads.get(wllama)?.catch(() => {});
  await wllama.exit();
}
