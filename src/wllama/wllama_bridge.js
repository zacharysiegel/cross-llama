// Passes JSON strings and plain values between the Rust side and wllama, which it imports from the app's directory.

// wllama's exit frees nothing while loadModel has yet to create its worker, which loadModel then creates regardless.
const pendingLoads = new WeakMap();
const LOGGER = { ...console, debug: () => {} };
const MODEL_LOCK_NAME_PREFIX = 'cross-llama model ';

function resolveDirectoryUrl(wllamaDirectoryUrl) {
  return new URL(wllamaDirectoryUrl.endsWith('/') ? wllamaDirectoryUrl : `${wllamaDirectoryUrl}/`, globalThis.location.href);
}

function resolveModelUrl(modelUrl) {
  return new URL(modelUrl, globalThis.location.href).href;
}

async function importWllama(directoryUrl) {
  return await import(new URL('index.js', directoryUrl).href);
}

export async function createWllama(wllamaDirectoryUrl) {
  const directoryUrl = resolveDirectoryUrl(wllamaDirectoryUrl);
  const { Wllama } = await importWllama(directoryUrl);
  const wllama = new Wllama({ default: new URL('wllama.wasm', directoryUrl).href }, { suppressNativeLog: true, logger: LOGGER });
  wllama.setCompat({
    wasm: new URL('compat/wllama.wasm', directoryUrl).href,
    worker: new URL('compat/wllama.js', directoryUrl).href,
  });
  return wllama;
}

// Manages the same cache in the browser's storage as every wllama instance's own model manager.
export async function createModelManager(wllamaDirectoryUrl) {
  const { ModelManager } = await importWllama(resolveDirectoryUrl(wllamaDirectoryUrl));
  return new ModelManager({ logger: LOGGER });
}

export async function downloadModel(modelManager, modelUrl, abortSignal, onProgress) {
  return await modelManager.getModelOrDownload({ url: resolveModelUrl(modelUrl) }, {
    signal: abortSignal,
    // Rust aborts before it frees the callback.
    progressCallback: ({ loaded, total }) => {
      if (!abortSignal.aborted) {
        onProgress(loaded, total);
      }
    },
  });
}

// Lists the cache without a network request.
export async function isModelDownloaded(modelManager, modelUrl) {
  const resolvedModelUrl = resolveModelUrl(modelUrl);
  const cachedModels = await modelManager.getModels();
  return cachedModels.some((cachedModel) => cachedModel.url === resolvedModelUrl);
}

// Resolves once this page holds the model's lock, across every page of the origin; aborting the signal releases it.
export function acquireModelLock(modelUrl, abortSignal) {
  return new Promise((resolve, reject) => {
    const holdUntilAborted = () => new Promise((release) => {
      if (abortSignal.aborted) {
        release();
        return;
      }

      abortSignal.addEventListener('abort', () => release(), { once: true });
      resolve();
    });
    navigator.locks.request(`${MODEL_LOCK_NAME_PREFIX}${resolveModelUrl(modelUrl)}`, { signal: abortSignal }, holdUntilAborted).catch(reject);
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
  const { n_ctx, list_tokens_eog, token_bos, token_eos, add_bos_token, add_eos_token } = wllama.getLoadedContextInfo();
  return JSON.stringify({ n_ctx, list_tokens_eog, token_bos, token_eos, add_bos_token, add_eos_token, chat_template: wllama.getChatTemplate() });
}

export async function generate(wllama, requestJson, abortSignal, onChunk) {
  await wllama.createCompletion({
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
