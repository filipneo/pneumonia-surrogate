# pneumonia-surrogate

A self-contained Web Component that runs an ONNX surrogate model of a respiratory physiology simulator at ~30 Hz entirely in the browser.

## Installation

```bash
npm install pneumonia-surrogate
```

Or load directly from a CDN with no build step:

```html
<script src="https://cdn.jsdelivr.net/npm/pneumonia-surrogate/dist/pneumonia-surrogate.js"></script>
```

## Model assets

The component requires three files that are included in the npm package and served alongside your HTML:

| File | Description |
|------|-------------|
| `surrogate_model.onnx` | Exported ONNX graph (all weights inlined) |
| `scalers.json` | StandardScaler + MinMaxScaler parameters |
| `golden_seed.json` | 90-step warm-start physiological history |

## Usage

```html
<!-- If installed via npm, import from dist/ -->
<script src="node_modules/pneumonia-surrogate/dist/pneumonia-surrogate.js"></script>

<pneumonia-surrogate
    id="sim"
    model-url="./surrogate_model.onnx"
    scalers-url="./scalers.json"
    seed-url="./golden_seed.json"
    total-compliance="60"
    dv="150"
    cshunt-frac="5">
</pneumonia-surrogate>

<script>
  const sim = document.getElementById('sim');

  // Start the inference loop
  sim.start();

  // Listen for predictions (~30 events per second)
  sim.addEventListener('surrogate-data', (event) => {
    const { plot_vars, monitor_vars, step } = event.detail;
    console.log('Step', step, plot_vars, monitor_vars);
  });
</script>
```

## Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `model-url` | string | — | URL to `surrogate_model.onnx` **(required)** |
| `scalers-url` | string | — | URL to `scalers.json` **(required)** |
| `seed-url` | string | — | URL to `golden_seed.json` **(required)** |
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
