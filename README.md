# Framelume models

Core ML models that the Framelume macOS app downloads on demand (Settings ▸ Object Removal). They are not bundled with the app because of their size.

## LaMa512 (release `lama-512-v1`)

`LaMa512.mlpackage.zip` is LaMa (big-lama) converted to a Core ML ML program, used for AI object removal on still images.

| | |
|---|---|
| Size | 189 974 724 bytes |
| SHA-256 | `307ae8e3b994ded0b69774fd7d86e0d874726b190434a6dced7f659e7a9c073d` |
| Inputs | `image` 1×3×512×512 RGB 0…1, `mask` 1×1×512×512 (1 = remove) |
| Output | `output` 1×3×512×512 RGB 0…1, already composited |
| Precision | fp32, macOS 13+ |

### Source

- Model: [LaMa](https://github.com/advimman/lama) by Samsung AI Center (Suvorov et al., *Resolution-robust Large Mask Inpainting with Fourier Convolutions*, WACV 2022).
- Weights: `big-lama.pt`, the TorchScript export from <https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt> (md5 `e3aa4aaa15225a33ec84f9f4bc47e500`).

### Rebuild

```bash
pip install coremltools==9.0 torch==2.7.0
python convert_lama_coreml.py big-lama.pt 512 fp32 LaMa512.mlpackage
ditto -c -k --keepParent LaMa512.mlpackage LaMa512.mlpackage.zip
```

## Licence

LaMa's code and weights are released by Samsung AI Center under the Apache License 2.0. The converted model and the conversion script in this repository are distributed under the same licence; see [LICENSE](LICENSE).
