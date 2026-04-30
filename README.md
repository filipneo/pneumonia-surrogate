# pneumonia-surrogate

A self-contained Web Component that runs an ONNX surrogate model of a respiratory physiology simulator at ~30 Hz entirely in the browser.

## Installation

```bash
npm install pneumonia-surrogate
```

Or load directly from a CDN with no build step:

```html
<script src="https://cdn.jsdelivr.net/npm/pneumonia-surrogate/dist/pneumonia-surrogate.js"></script>

<pneumonia-surrogate id="sim"></pneumonia-surrogate>

<script>
  const sim = document.getElementById('sim');
  sim.start();
  sim.addEventListener('surrogate-data', (e) => console.log(e.detail));
</script>
```

## Model assets

The three model files are included in the npm package and are also available on jsDelivr CDN. By default the component fetches them from CDN automatically, so no local setup is required. Override the URLs via attributes only if you want to self-host the files.

| File | Description |
|------|-------------|
| `surrogate_model.onnx` | Exported ONNX graph (all weights inlined) |
| `scalers.json` | StandardScaler + MinMaxScaler parameters |
| `golden_seed.json` | 90-step warm-start physiological history |

## Usage

The URL attributes (`model-url`, `scalers-url`, `seed-url`) default to the published CDN location and can be omitted.

```html
<!-- If installed via npm, import from dist/ -->
<script src="node_modules/pneumonia-surrogate/dist/pneumonia-surrogate.js"></script>

<pneumonia-surrogate id="sim"></pneumonia-surrogate>

<script>
  const sim = document.getElementById('sim');

  // Start the inference loop
  sim.start();

  // Listen for predictions (~30 events per second)
  sim.addEventListener('surrogate-data', (event) => {
    const { plot_vars, monitor_vars, step } = event.detail;
    console.log('Step', step, plot_vars, monitor_vars);
  });

  // Change control variables at any time while the simulation is running
  sim.totalCompliance = 80;
  sim.dv = 200;
  sim.cShuntFrac = 10;
</script>
```

## Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `model-url` | string | CDN | URL to `surrogate_model.onnx` |
| `scalers-url` | string | CDN | URL to `scalers.json` |
| `seed-url` | string | CDN | URL to `golden_seed.json` |
| `total-compliance` | number | `60` | Lung compliance [10 – 200] |
| `dv` | number | `150` | Dead volume [150 – 400] |
| `cshunt-frac` | number | `5` | Shunt fraction [2 – 70] |

All three control attributes are also exposed as JS properties (`totalCompliance`, `dv`, `cShuntFrac`) and can be set at any time while the simulation is running.

## Public methods

| Method | Description |
|--------|-------------|
| `start()` | Load the ONNX session (if not already cached) and begin the ~30 Hz inference loop |
| `stop()` | Pause the inference loop; session remains loaded |
| `reset()` | Stop execution, restore control defaults, reload the warm-start seed |

## `surrogate-data` event detail

```js
{
  plot_vars: {
    "lungs.q_in[1].p":    number,  // Airway pressure
    "lungs.q_in[1].m_flow": number, // Airflow
    "Ecg.ecg":            number,  // ECG signal
    "EithaPressure.pressure": number // Arterial pressure waveform
  },
  monitor_vars: {
    "filter.y":                   number,  // Respiratory rate
    "currentHeartReat.y":         number,  // Heart rate
    "arterialPressure.systolic":  number,  // Systolic ABP (mmHg)
    "arterialPressure.diastolic": number,  // Diastolic ABP (mmHg)
    "arterial.sO2":               number,  // SpO2 (%)
    "arterial.pO2":               number,  // PaO2 (mmHg)
    "arterial.pCO2":              number,  // PaCO2 (mmHg)
    "arterial.pH":                number,  // Arterial pH
    "tissueUnit[1].pH":           number,  // Tissue pH
    "venous.pH":                  number   // Venous pH
  },
  step: number  // Monotonic step counter since last start()
}
```

The event bubbles and is `composed: true`, so it crosses Shadow DOM boundaries.
