# Therfoo — ***now archived***

**Important:** This project is no longer being maintained. Starting **August 3, 2025**, **Therfoo** will be archived in read‑only mode on GitHub. To experiment with and build on a fully maintained version, please use the new project **[Zerfoo](https://github.com/zerfoo/zerfoo)**.

---

## 📌 What Therfoo Was

Therfoo is an embedded **deep learning library for Go**, designed to help hobbyists and learners explore neural network fundamentals (including feed‑forward layers and backpropagation) without needing C‑based dependencies. It was originally written in 2019 and provided a simple yet flexible foundation for building and training small ML models entirely in Go ([GitHub][1]).

Key features included:

* **Zero‑dependency** implementation in pure Go
* Support for basic neural network primitives: fully‑connected layers, ReLU, cross‑entropy
* **Customizable training loops**, so users could clearly see how gradient descent and learning rates work
* Designed for **educational use**, not production inference or large‑scale training
* MIT‑licensed, easy to fork and modify

By design, Therfoo was straightforward, minimal, and intended for hands‑on learning. It inspired dozens of tutorial fork projects but was never meant for large datasets or accelerator support.

---

## 🔧 Why We Archived It

As the Go ML ecosystem matured, it became clear that building and scaling models in Go requires support for:

1. **Low‑precision tensor types** (e.g. float8/float16) to use modern accelerators
2. **ONNX import/export** for interoperability with PyTorch, TensorFlow, etc.
3. **Transformers and attention models** for sequencing tasks
4. Native support for **accelerators (GPU, TPU, etc.)**
5. A **modular architecture**, plugging into optimizers, autodiff engines, model IO pipelines

Therfoo was never designed for any of the above. As demand grew, I decided to focus energy on a fresh rewrite—thus **Zerfoo** was born. It addresses every limitation of Therfoo and is built for modern ML workflows ([LinkedIn][2]).

---

## 🚀 Introducing Zerfoo

The Zerfoo project is a **modular, accelerator‑ready ML framework written entirely in Go**. It combines rigorous type support with production‑level model training pipelines.

### Zerfoo Highlights

| Capability                  | Description                                                                         |
| --------------------------- | ----------------------------------------------------------------------------------- |
| **Float types**             | Supports `float8`, `float16`, `float32`, and `float64` for modern ML use            |
| **ONNX import/export**      | Load models from ONNX format; export fine‑tuned models back                         |
| **Transformer training**    | Train transformer‑class networks from scratch in Go                                 |
| **Autodiff & optimization** | First‑class support for custom optimization loops                                   |
| **Go native**               | No bindings to C++ or Python; fast, safe, Go idiomatic                              |
| **Accelerator support**     | Able to target GPU and TPU backends via Go runtime adapters                         |
| **Educational clarity**     | Full documentation walking through architecture, loss flows, and float quantization |

Zerfoo is an ambitious successor to Therfoo, built to scale from simple demos to production‑ready models in Go.

---

## 🧭 Migration Path

If you used Therfoo and want to migrate:

* **Cores operations** like `Forward()`, `Backward()`, and manual loop logic are replaced by *autodiff-enabled layers and optimizers*.
* Instead of custom `TrainEpoch(...)` calls, Zerfoo uses unified `Trainer` structs and canonical schedule files.
* ONNX models can be imported with `zrn.ImportONNX("model.onnx")`, quantized to float8, and then fine‑tuned on a dataset.
* For a step‑by‑step migration guide for each Therfoo example, see the **Zerfoo README** and **docs** folder.

---

## ⚠️ Legacy Usage (Not Recommended)

Therfoo can still be imported in Go as:

```go
import "github.com/therfoo/therfoo/pkg"
```

Example (keep using old code at your own risk):

```go
net := pkg.NewFFN([]int{784, 256, 10})  
net.Train(...)
```

However, **do not build new ML work on Therfoo**, especially for large datasets, GPU models, or advanced architectures.

---

## 🧪 For Students and Learners

If you’re learning ML fundamentals and loved Therfoo’s transparent training loops:

* Treat Therfoo as a historical snapshot.
* Once comfortable, try rewriting one of your Therfoo examples in Zerfoo to see how real autodiff pipelines evolve.

---

## 📫 Contributions & Questions

This repository is archived, so opening new issues or pull requests is disabled. If you’d like to discuss Therfoo, see sample code, or contribute to its evolution:

1. **Check out Zerfoo’s Issues**: discussions around design, float8 support, ONNX examples
2. **Ping me on GitHub or LinkedIn**—especially if you are working on Go‑based AI infrastructure collaborations

---

## 📜 License

This project and associated assets remain licensed under the [MIT License](./LICENSE).

---

Thank you for building with Therfoo. I hope it inspired your learning journey. I’m excited to see what you create with **Zerfoo**.

[1]: https://github.com/therfoo/therfoo?utm_source=chatgpt.com "GitHub - therfoo/therfoo: An embedded deep learning library for Go."
[2]: https://www.linkedin.com/pulse/tiny-floats-huge-impact-david-ndungu-klmzc?utm_source=chatgpt.com "Tiny Floats, Huge Impact - LinkedIn"
