# neural-network-demo# Interactive Neural Network Learning Demo

A browser-based playground that trains a small neural network live with **TensorFlow.js**. Pick a dataset, change the architecture, and watch the decision boundary and loss curve update as the model learns. No install or backend needed.

## Run locally
```bash
python3 -m http.server 8000   # then open http://localhost:8000
```
(or just double-click `index.html`)

## Controls
- **Dataset**: XOR, Circle, Spiral
- **Hidden layers / neurons**: model depth and width
- **Activation**: relu, tanh, sigmoid
- **Learning rate**: Adam optimizer step size
- **Train / Pause / Step / Reset**

## Project layout
- `index.html` – the demo (HTML + CSS + JS + TensorFlow.js via CDN)
- `docs/presentation.pptx` – slide deck

## Publish on GitHub Pages
Repo → Settings → Pages → Deploy from branch → `main` / root.

## License
MIT
