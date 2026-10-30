# Converts the big-lama TorchScript model to the Core ML package Framelume downloads (LaMa512.mlpackage.zip).
#
# Source: big-lama.pt from https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt
#         (md5 e3aa4aaa15225a33ec84f9f4bc47e500), the TorchScript export of LaMa's big-lama weights.
# venv:   pip install coremltools==9.0 torch==2.7.0
# Usage:  python convert_lama_coreml.py big-lama.pt 512 fp32 LaMa512.mlpackage
#         ditto -c -k --keepParent LaMa512.mlpackage LaMa512.mlpackage.zip
#
# Inputs `image` (1x3xSxS RGB 0..1) and `mask` (1x1xSxS, 0/1); output `output` (RGB 0..1, already composited).
# Licence: LaMa (https://github.com/advimman/lama) code and weights by Samsung AI Center, Apache-2.0.
import sys, time, torch, coremltools as ct, numpy as np
src, size, prec, out = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
m = torch.jit.load(src, map_location="cpu").eval()
img = torch.rand(1, 3, size, size); mask = torch.zeros(1, 1, size, size); mask[..., size//3:size//2, size//3:size//2] = 1
t = time.time()
ml = ct.convert(m, inputs=[ct.TensorType(name="image", shape=img.shape), ct.TensorType(name="mask", shape=mask.shape)],
                outputs=[ct.TensorType(name="output")], convert_to="mlprogram",
                compute_precision=ct.precision.FLOAT16 if prec == "fp16" else ct.precision.FLOAT32,
                minimum_deployment_target=ct.target.macOS13)
ml.save(out); print("converted", time.time() - t)
