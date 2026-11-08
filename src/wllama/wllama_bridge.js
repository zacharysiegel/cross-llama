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

export async function downloadModel(wllamaDirectoryUrl, modelManager, modelUrl, abortSignal, onProgress) {
  const wllamaModule = await importWllama(resolveDirectoryUrl(wllamaDirectoryUrl));
  const resolvedModelUrl = resolveModelUrl(modelUrl);
  const cachedModel = await readCachedModel(wllamaModule, modelManager, resolvedModelUrl);
  const validModel = findValidModel(wllamaModule, modelManager, resolvedModelUrl, cachedModel);
  // Rust aborts before it frees the callback.
  const reportProgress = (loaded, total) => {
    if (!abortSignal.aborted) {
      onProgress(loaded, total);
    }
  };

  if (validModel) {
    reportProgress(validModel.size, validModel.size);
    return validModel;
  }

  // Keeps each part whose cached size matches its metadata, or, without metadata, the server's size; fetches the others again.
  const completePartByteCount = countCompletePartBytes(cachedModel.partFiles);
  const model = await modelManager.downloadModel({ url: resolvedModelUrl }, {
    signal: abortSignal,
    progressCallback: ({ loaded, total }) => reportProgress(completePartByteCount + loaded, total),
  });

  // No progress so far counts a part kept without metadata. An invalid model's size is not its byte count.
  if (model.validate() === wllamaModule.ModelValidationStatus.VALID) {
    reportProgress(model.size, model.size);
  }

  return model;
}

// Lists the cache without a network request.
export async function isModelDownloaded(wllamaDirectoryUrl, modelManager, modelUrl) {
  const wllamaModule = await importWllama(resolveDirectoryUrl(wllamaDirectoryUrl));
  const resolvedModelUrl = resolveModelUrl(modelUrl);
  const cachedModel = await readCachedModel(wllamaModule, modelManager, resolvedModelUrl);
  return findValidModel(wllamaModule, modelManager, resolvedModelUrl, cachedModel) !== null;
}

/* wllama 3.8.1's getModels, and so getModelOrDownload, throws while any split model in the cache lacks a part's metadata,
   which a download stopped partway leaves, so the bridge reads the one model's parts itself. A part's file without
   metadata is absent from partFiles. */
async function readCachedModel(wllamaModule, modelManager, resolvedModelUrl) {
  const cachedFiles = await modelManager.cacheManager.list();
  const partFiles = wllamaModule.ModelManager.parseModelUrl(resolvedModelUrl)
    .map((partUrl) => cachedFiles.find((cachedFile) => cachedFile.metadata.originalURL === partUrl));
  return { cachedFiles, partFiles };
}

function findValidModel(wllamaModule, modelManager, resolvedModelUrl, { cachedFiles, partFiles }) {
  // Model's constructor throws unless every part has its metadata.
  if (partFiles.includes(undefined)) {
    return null;
  }

  const model = new wllamaModule.Model(modelManager, resolvedModelUrl, undefined, cachedFiles);
  return model.validate() === wllamaModule.ModelValidationStatus.VALID ? model : null;
}

// wllama's download keeps these parts without reporting progress for them.
function countCompletePartBytes(partFiles) {
  return partFiles
    .filter((partFile) => partFile !== undefined && partFile.size === partFile.metadata.originalSize)
    .reduce((byteCount, partFile) => byteCount + partFile.size, 0);
}

// Resolves once this page holds the model's lock, across every page of the origin. Aborting the signal releases it once
// every promise in heldOperations, to which the holder adds, has settled.
export function acquireModelLock(modelUrl, abortSignal, heldOperations) {
  return new Promise((resolve, reject) => {
    const holdUntilReleased = async () => {
      if (!abortSignal.aborted) {
        resolve();
        await waitForAbort(abortSignal);
      }

      await Promise.allSettled(heldOperations);
    };
    navigator.locks.request(`${MODEL_LOCK_NAME_PREFIX}${resolveModelUrl(modelUrl)}`, { signal: abortSignal }, holdUntilReleased).catch(reject);
  });
}

function waitForAbort(abortSignal) {
  return new Promise((resolve) => abortSignal.addEventListener('abort', () => resolve(), { once: true }));
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

// wllama 3.8.1's rule for switching to its compatibility build; setCompat's default mode keeps Firefox on the default build.
function needsWllamaCompatibilityBuild() {
  return (!('Suspending' in WebAssembly) || !supportsWebAssemblyMemory64()) && !isFirefox();
}

// wllama's own user-agent test.
function isFirefox() {
  return /Firefox\/([0-9\.]+)(?:\s|$)/.test(navigator.userAgent);
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
