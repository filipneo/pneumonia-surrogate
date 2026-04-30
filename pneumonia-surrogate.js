/**
 * pneumonia-surrogate.js
 *
 * Custom Web Component wrapping the exported ONNX surrogate model.
 * Runs autoregressive inference at ~30 Hz and emits a "surrogate-data"
 * CustomEvent on each tick with the latest predicted physiological values.
 *
 * Usage:
 *   <pneumonia-surrogate
 *       model-url="./surrogate_model.onnx"
 *       scalers-url="./scalers.json"
 *       seed-url="./golden_seed.json"
 *       total-compliance="60"
 *       dv="150"
 *       cshunt-frac="5">
 *   </pneumonia-surrogate>
 *
 * Public API:
 *   element.start()  - begin inference loop
 *   element.stop()   - pause inference loop (session stays alive)
 *   element.reset()  - stop, restore defaults, reload seed into buffers
 *
 * Emitted event detail shape:
 *   {
 *     plot_vars: { "lungs.q_in[1].p": number, ... },   // 4 waveform variables
 *     monitor_vars: { "filter.y": number, ... },        // 10 clinical variables
 *     step: number,                                     // monotonic step counter
 *   }
 */

import * as ort from 'onnxruntime-web';

// Injected by build.mjs via esbuild --define; falls back to '.' for local dev.
// eslint-disable-next-line no-undef
const _CDN_BASE = typeof ASSET_CDN_BASE !== 'undefined' ? ASSET_CDN_BASE : '.';

const DEFAULT_MODEL_URL   = `${_CDN_BASE}/surrogate_model.onnx`;
const DEFAULT_SCALERS_URL = `${_CDN_BASE}/scalers.json`;
const DEFAULT_SEED_URL    = `${_CDN_BASE}/golden_seed.json`;

// Point WASM binaries to CDN so the bundle works when served from any origin.
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/';
// Disable multi-threading: SharedArrayBuffer requires cross-origin isolation
// headers that are rarely configured on typical web servers.
ort.env.wasm.numThreads = 1;

// Variable name lists must match the order used during training.
const PLOT_VARS = [
  "lungs.q_in[1].p",
  "lungs.q_in[1].m_flow",
  "Ecg.ecg",
  "EithaPressure.pressure",
];

const MONITOR_VARS = [
  "filter.y",
  "currentHeartReat.y",
  "arterialPressure.systolic",
  "arterialPressure.diastolic",
  "arterial.sO2",
  "arterial.pO2",
  "arterial.pCO2",
  "arterial.pH",
  "tissueUnit[1].pH",
  "venous.pH",
];

const CONTROL_VARS = ["TotalCompliance", "DV", "cShuntFrac"];

const NUM_PHYS_VARS = PLOT_VARS.length + MONITOR_VARS.length; // 14
const NUM_CONTROLS = CONTROL_VARS.length; // 3
const WINDOW_SIZE = 90;
const TARGET_HZ = 30;
const FRAME_INTERVAL_MS = 1000 / TARGET_HZ;

// Default UI-space (raw, unscaled) control values matching simulation baseline.
const CONTROL_DEFAULTS = {
  TotalCompliance: 60,
  DV: 150,
  cShuntFrac: 5,
};

// ---- Scaler helpers (pure, no external dependencies) ------------------------

/**
 * StandardScaler forward transform: z = (x - mean) / scale
 * @param {Float32Array} x    Input values of length n.
 * @param {number[]}     mean Array of per-feature means, length n.
 * @param {number[]}     scale Array of per-feature std devs, length n.
 * @returns {Float32Array}
 */
function standardScale(x, mean, scale) {
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) {
    out[i] = (x[i] - mean[i]) / scale[i];
  }
  return out;
}

/**
 * StandardScaler inverse transform: x = z * scale + mean
 * @param {Float32Array} z     Scaled values of length n.
 * @param {number[]}     mean  Per-feature means.
 * @param {number[]}     scale Per-feature std devs.
 * @returns {Float32Array}
 */
function standardInverse(z, mean, scale) {
  const out = new Float32Array(z.length);
  for (let i = 0; i < z.length; i++) {
    out[i] = z[i] * scale[i] + mean[i];
  }
  return out;
}

/**
 * MinMaxScaler forward transform: z = x * scale_ + min_
 * (sklearn stores min_ = feature_range[0] - data_min * scale_ and scale_ = range / (max-min))
 * @param {Float32Array} x      Raw values, length n.
 * @param {number[]}     min_   Scaler min_ attribute, length n.
 * @param {number[]}     scale_ Scaler scale_ attribute, length n.
 * @returns {Float32Array}
 */
function minMaxScale(x, min_, scale_) {
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) {
    out[i] = x[i] * scale_[i] + min_[i];
  }
  return out;
}

// ---- Web Component ----------------------------------------------------------

class PneumoniaSurrogate extends HTMLElement {
  // Reflected HTML attributes.
  static get observedAttributes() {
    return [
      "model-url",
      "scalers-url",
      "seed-url",
      "total-compliance",
      "dv",
      "cshunt-frac",
    ];
  }

  constructor() {
    super();

    // Configuration URLs (default to the published CDN location of each asset).
    this._modelUrl   = DEFAULT_MODEL_URL;
    this._scalersUrl = DEFAULT_SCALERS_URL;
    this._seedUrl    = DEFAULT_SEED_URL;

    // Current raw (UI-space) control values.
    this._controls = { ...CONTROL_DEFAULTS };

    // ONNX session and scaler state (populated during init).
    this._session = null;
    this._physScalerMean = null;
    this._physScalerScale = null;
    this._ctrlScalerMin = null;
    this._ctrlScalerScale = null;

    // Rolling history buffers stored as flat Float32Arrays in row-major order.
    // Row i spans indices [i*cols, (i+1)*cols).
    this._physHistory = new Float32Array(WINDOW_SIZE * NUM_PHYS_VARS);
    this._ctrlHistory = new Float32Array(WINDOW_SIZE * NUM_CONTROLS);

    // Execution control.
    this._running = false;
    this._rafId = null;
    this._lastFrameTime = 0;
    this._stepCounter = 0;

    // Cached init promise to prevent double-initialisation.
    this._initPromise = null;

    this._shadowRoot = this.attachShadow({ mode: "open" });
    this._render();
  }

  // ---- Lifecycle callbacks --------------------------------------------------

  connectedCallback() {
    // Attributes override the built-in CDN defaults when present.
    const modelUrl   = this.getAttribute('model-url');
    const scalersUrl = this.getAttribute('scalers-url');
    const seedUrl    = this.getAttribute('seed-url');
    if (modelUrl)   this._modelUrl   = modelUrl;
    if (scalersUrl) this._scalersUrl = scalersUrl;
    if (seedUrl)    this._seedUrl    = seedUrl;
    this._syncControlsFromAttributes();
  }

  attributeChangedCallback(name, _oldValue, newValue) {
    switch (name) {
      case "model-url":
        this._modelUrl = newValue || DEFAULT_MODEL_URL;
        // Invalidate cached session if the model URL changes.
        this._initPromise = null;
        this._session = null;
        break;
      case 'scalers-url':
        this._scalersUrl = newValue || DEFAULT_SCALERS_URL;
        this._initPromise = null;
        this._session = null;
        break;
      case 'seed-url':
        this._seedUrl = newValue || DEFAULT_SEED_URL;
        this._initPromise = null;
        this._session = null;
        break;
      case "total-compliance":
        if (newValue !== null) this._controls.TotalCompliance = parseFloat(newValue);
        break;
      case "dv":
        if (newValue !== null) this._controls.DV = parseFloat(newValue);
        break;
      case "cshunt-frac":
        if (newValue !== null) this._controls.cShuntFrac = parseFloat(newValue);
        break;
    }
  }

  // ---- JS property accessors (mirror HTML attributes) ----------------------

  get totalCompliance() { return this._controls.TotalCompliance; }
  set totalCompliance(v) {
    this._controls.TotalCompliance = parseFloat(v);
    this.setAttribute("total-compliance", v);
  }

  get dv() { return this._controls.DV; }
  set dv(v) {
    this._controls.DV = parseFloat(v);
    this.setAttribute("dv", v);
  }

  get cShuntFrac() { return this._controls.cShuntFrac; }
  set cShuntFrac(v) {
    this._controls.cShuntFrac = parseFloat(v);
    this.setAttribute("cshunt-frac", v);
  }

  // ---- Public API -----------------------------------------------------------

  /**
   * Load the ONNX session (if needed) and start the inference loop.
   */
  async start() {
    if (this._running) return;
    try {
      await this._ensureInit();
    } catch (err) {
      console.error("[pneumonia-surrogate] Initialisation failed, cannot start:", err);
      this._setStatus("error");
      return;
    }
    this._running = true;
    this._lastFrameTime = performance.now();
    this._setStatus("running");
    this._rafId = requestAnimationFrame((t) => this._loop(t));
  }

  /**
   * Pause the inference loop. The ONNX session remains loaded.
   */
  stop() {
    this._running = false;
    if (this._rafId !== null) {
      cancelAnimationFrame(this._rafId);
      this._rafId = null;
    }
    this._setStatus("idle");
  }

  /**
   * Stop execution, restore control defaults, and reset history buffers from seed.
   */
  async reset() {
    this.stop();
    this._controls = { ...CONTROL_DEFAULTS };
    this.setAttribute("total-compliance", CONTROL_DEFAULTS.TotalCompliance);
    this.setAttribute("dv", CONTROL_DEFAULTS.DV);
    this.setAttribute("cshunt-frac", CONTROL_DEFAULTS.cShuntFrac);
    this._stepCounter = 0;

    // Reload seed into buffers if the session has already been initialised.
    if (this._initPromise !== null) {
      try {
        await this._initPromise;
        await this._loadSeed();
      } catch (_err) {
        // If init failed previously, seed load is a no-op.
      }
    }
    this._setStatus("idle");
  }

  // ---- Initialisation -------------------------------------------------------

  /**
   * Return a cached Promise that resolves once the ONNX session, scalers,
   * and seed buffers are fully loaded. Safe to await multiple times.
   */
  _ensureInit() {
    if (this._initPromise) return this._initPromise;
    this._initPromise = this._init();
    return this._initPromise;
  }

  async _init() {
    this._setStatus("loading");

    if (!this._modelUrl || !this._scalersUrl || !this._seedUrl) {
      // This should only happen if someone explicitly set an attribute to empty string.
      throw new Error(
        'Asset URLs could not be resolved. Ensure model-url, scalers-url, and seed-url are valid.'
      );
    }

    // Load ONNX session.
    console.log("[pneumonia-surrogate] Loading ONNX session from:", this._modelUrl);
    const modelBuffer = await this._fetchArrayBuffer(this._modelUrl);
    this._session = await ort.InferenceSession.create(modelBuffer, {
      executionProviders: ["wasm"],
    });
    console.log("[pneumonia-surrogate] ONNX session ready.");

    // Load scaler parameters.
    console.log("[pneumonia-surrogate] Loading scalers from:", this._scalersUrl);
    const scalersData = await this._fetchJson(this._scalersUrl);
    this._physScalerMean = scalersData.phys_scaler.mean;
    this._physScalerScale = scalersData.phys_scaler.scale;
    this._ctrlScalerMin = scalersData.control_scaler.min_;
    this._ctrlScalerScale = scalersData.control_scaler.scale_;
    console.log("[pneumonia-surrogate] Scalers loaded.");

    // Load golden seed into rolling buffers.
    await this._loadSeed();
  }

  async _loadSeed() {
    console.log("[pneumonia-surrogate] Loading golden seed from:", this._seedUrl);
    let seedData;
    try {
      seedData = await this._fetchJson(this._seedUrl);
    } catch (err) {
      console.warn(
        "[pneumonia-surrogate] Could not load golden seed, initialising from zeros:", err
      );
      this._physHistory = new Float32Array(WINDOW_SIZE * NUM_PHYS_VARS);
      this._ctrlHistory = new Float32Array(WINDOW_SIZE * NUM_CONTROLS);
      return;
    }

    // Flatten 2D arrays (rows x cols) into Float32Arrays in row-major order.
    const phys = seedData.phys; // (90, 14)
    const ctrl = seedData.ctrl; // (90,  3)

    if (phys.length !== WINDOW_SIZE || ctrl.length !== WINDOW_SIZE) {
      throw new Error(
        `Seed shape mismatch: expected (${WINDOW_SIZE}, *), ` +
        `got phys=(${phys.length},*) ctrl=(${ctrl.length},*).`
      );
    }

    this._physHistory = new Float32Array(WINDOW_SIZE * NUM_PHYS_VARS);
    for (let row = 0; row < WINDOW_SIZE; row++) {
      if (phys[row].length !== NUM_PHYS_VARS) {
        throw new Error(
          `Seed phys row ${row} has ${phys[row].length} features, expected ${NUM_PHYS_VARS}.`
        );
      }
      this._physHistory.set(phys[row], row * NUM_PHYS_VARS);
    }

    this._ctrlHistory = new Float32Array(WINDOW_SIZE * NUM_CONTROLS);
    for (let row = 0; row < WINDOW_SIZE; row++) {
      if (ctrl[row].length !== NUM_CONTROLS) {
        throw new Error(
          `Seed ctrl row ${row} has ${ctrl[row].length} features, expected ${NUM_CONTROLS}.`
        );
      }
      this._ctrlHistory.set(ctrl[row], row * NUM_CONTROLS);
    }

    console.log("[pneumonia-surrogate] Golden seed loaded into history buffers.");
  }

  // ---- Inference loop -------------------------------------------------------

  _loop(timestamp) {
    if (!this._running) return;

    if (timestamp - this._lastFrameTime >= FRAME_INTERVAL_MS) {
      this._lastFrameTime = timestamp;
      this._step().catch((err) => {
        console.error("[pneumonia-surrogate] Inference error:", err);
        this.stop();
        this._setStatus("error");
      });
    }

    this._rafId = requestAnimationFrame((t) => this._loop(t));
  }

  async _step() {
    // 1. Read current raw control values and apply MinMaxScaler.
    const rawControls = new Float32Array([
      this._controls.TotalCompliance,
      this._controls.DV,
      this._controls.cShuntFrac,
    ]);
    const scaledControls = minMaxScale(
      rawControls,
      this._ctrlScalerMin,
      this._ctrlScalerScale
    );

    // 2. Roll _ctrlHistory left by one row and write the new scaled row at the end.
    this._ctrlHistory.copyWithin(0, NUM_CONTROLS);
    this._ctrlHistory.set(scaledControls, (WINDOW_SIZE - 1) * NUM_CONTROLS);

    // 3. Build ort.Tensor inputs.
    //    Both histories are already contiguous flat arrays in row-major order.
    const xPhysTensor = new ort.Tensor(
      "float32",
      new Float32Array(this._physHistory), // copy to avoid aliasing issues
      [1, WINDOW_SIZE, NUM_PHYS_VARS]
    );
    const ctrlTensor = new ort.Tensor(
      "float32",
      new Float32Array(this._ctrlHistory),
      [1, WINDOW_SIZE, NUM_CONTROLS]
    );

    // 4. Run inference.
    let results;
    try {
      results = await this._session.run({ x_phys: xPhysTensor, controls: ctrlTensor });
    } catch (err) {
      throw new Error(`ONNX session.run() failed: ${err.message}`);
    }

    const outputData = results["output"].data; // Float32Array of length NUM_PHYS_VARS

    if (outputData.length !== NUM_PHYS_VARS) {
      throw new Error(
        `Unexpected output length ${outputData.length}, expected ${NUM_PHYS_VARS}.`
      );
    }

    // 5. Roll _physHistory left by one row and write the new prediction at the end.
    this._physHistory.copyWithin(0, NUM_PHYS_VARS);
    this._physHistory.set(outputData, (WINDOW_SIZE - 1) * NUM_PHYS_VARS);

    // 6. Inverse-transform to real-world units for event emission.
    const predReal = standardInverse(
      outputData,
      this._physScalerMean,
      this._physScalerScale
    );

    // 7. Package results and dispatch event.
    const plotVarsObj = {};
    for (let i = 0; i < PLOT_VARS.length; i++) {
      plotVarsObj[PLOT_VARS[i]] = predReal[i];
    }

    const monitorVarsObj = {};
    for (let i = 0; i < MONITOR_VARS.length; i++) {
      monitorVarsObj[MONITOR_VARS[i]] = predReal[PLOT_VARS.length + i];
    }

    this._stepCounter++;
    this.dispatchEvent(
      new CustomEvent("surrogate-data", {
        bubbles: true,
        composed: true,
        detail: {
          plot_vars: plotVarsObj,
          monitor_vars: monitorVarsObj,
          step: this._stepCounter,
        },
      })
    );
  }

  // ---- Shadow DOM -----------------------------------------------------------

  _render() {
    this._shadowRoot.innerHTML = `
      <style>
        :host {
          display: inline-block;
          font-family: monospace;
          font-size: 12px;
        }
        #status {
          padding: 2px 6px;
          border-radius: 3px;
          background: #222;
          color: #aaa;
          display: inline-block;
        }
        #status[data-state="running"] { background: #1a3a1a; color: #4caf50; }
        #status[data-state="loading"] { background: #1a2a3a; color: #64b5f6; }
        #status[data-state="error"]   { background: #3a1a1a; color: #ef5350; }
        #status[data-state="idle"]    { background: #222;    color: #aaa; }
      </style>
      <span id="status" data-state="idle">pneumonia-surrogate: idle</span>
      <slot></slot>
    `;
    this._statusEl = this._shadowRoot.getElementById("status");
  }

  _setStatus(state) {
    if (!this._statusEl) return;
    this._statusEl.dataset.state = state;
    this._statusEl.textContent = `pneumonia-surrogate: ${state}`;
  }

  // ---- Fetch helpers --------------------------------------------------------

  async _fetchArrayBuffer(url) {
    let response;
    try {
      response = await fetch(url);
    } catch (err) {
      throw new Error(`Network error fetching ${url}: ${err.message}`);
    }
    if (!response.ok) {
      throw new Error(`HTTP ${response.status} fetching ${url}`);
    }
    return response.arrayBuffer();
  }

  async _fetchJson(url) {
    let response;
    try {
      response = await fetch(url);
    } catch (err) {
      throw new Error(`Network error fetching ${url}: ${err.message}`);
    }
    if (!response.ok) {
      throw new Error(`HTTP ${response.status} fetching ${url}`);
    }
    return response.json();
  }

  // ---- Private helpers ------------------------------------------------------

  _syncControlsFromAttributes() {
    const tc = this.getAttribute("total-compliance");
    const dv = this.getAttribute("dv");
    const cs = this.getAttribute("cshunt-frac");
    if (tc !== null) this._controls.TotalCompliance = parseFloat(tc);
    if (dv !== null) this._controls.DV = parseFloat(dv);
    if (cs !== null) this._controls.cShuntFrac = parseFloat(cs);
  }
}

customElements.define("pneumonia-surrogate", PneumoniaSurrogate);
