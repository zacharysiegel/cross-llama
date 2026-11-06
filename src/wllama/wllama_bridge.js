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

export async function probeBrowserCapability() {
  const webGpuAdapter = await requestWebGpuAdapter();
  const storageEstimate = await estimateStorage();

  return JSON.stringify({
    webgpu_adapter: webGpuAdapter && { maximum_buffer_byte_count: webGpuAdapter.limits.maxBufferSize },
    cross_origin_isolated: globalThis.crossOriginIsolated === true,
    wllama_build: needsWllamaCompatibilityBuild() ? 'compatibility' : 'default',
    storage_quota_byte_count: storageEstimate?.quota ?? null,
    storage_usage_byte_count: storageEstimate?.usage ?? null,
    storage_persisted: await isStoragePersisted(),
    device_memory_gigabytes: navigator.deviceMemory ?? null,
  });
}

// A fallback adapter runs on the CPU.
async function requestWebGpuAdapter() {
  let adapter = null;

  try {
    adapter = await navigator.gpu.requestAdapter();
  } catch {
    return null;
  }

  const isFallbackAdapter = adapter?.info?.isFallbackAdapter ?? adapter?.isFallbackAdapter ?? false;
  return adapter && !isFallbackAdapter ? adapter : null;
}

// wllama 3.8.1's rule for switching to its compatibility build.
function needsWllamaCompatibilityBuild() {
  return !('Suspending' in WebAssembly) || !supportsWebAssemblyMemory64();
}

function supportsWebAssemblyMemory64() {
  try {
    new WebAssembly.Memory({ address: 'i64', initial: 1n });
    return true;
  } catch {
    return false;
  }
}

async function estimateStorage() {
  try {
    const { quota, usage } = await navigator.storage.estimate();
    return { quota: Math.floor(quota), usage: Math.floor(usage) };
  } catch {
    return null;
  }
}

async function isStoragePersisted() {
  try {
    return await navigator.storage.persisted();
  } catch {
    return false;
  }
}
