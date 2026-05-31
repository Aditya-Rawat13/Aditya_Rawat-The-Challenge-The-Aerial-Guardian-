#!/usr/bin/env python3
import argparse
from pathlib import Path
from ultralytics import YOLO


def exp_onnx(mod_pth: str, oup_dir: str, img_siz: int = 640):
    mod = YOLO(mod_pth)
    oup = mod.export(
        format="onnx",
        imgsz=img_siz,
        opset=12,
        simplify=True,
        dynamic=False,
        half=False,
    )
    print(f"✓ ONNX export: {oup}")
    return oup


def exp_trt(mod_pth: str, int8: bool = False, img_siz: int = 640):
    mod = YOLO(mod_pth)
    oup = mod.export(
        format="engine",
        imgsz=img_siz,
        half=True,
        int8=int8,
        device=0,
        workspace=4,
    )
    print(f"✓ TensorRT export: {oup}")
    return oup


def exp_ncc(mod_pth: str, img_siz: int = 640):
    mod = YOLO(mod_pth)
    oup = mod.export(format="ncnn", imgsz=img_siz)
    print(f"✓ NCNN export: {oup}")
    return oup


def ben_chk(mod_pth: str, img_siz: int = 640, n: int = 100):
    import time
    import numpy as np

    if mod_pth.endswith(".onnx"):
        import onnxruntime as ort
        ses = ort.InferenceSession(mod_pth, providers=["CPUExecutionProvider"])
        inp_nam = ses.get_inputs()[0].name
        dum = np.random.randn(1, 3, img_siz, img_siz).astype(np.float32)

        for _ in range(5):
            ses.run(None, {inp_nam: dum})

        t0 = time.perf_counter()
        for _ in range(n):
            ses.run(None, {inp_nam: dum})
        ela = time.perf_counter() - t0
        fps = n / ela
        print(f"ONNX FPS: {fps:.1f} (avg {ela/n*1000:.1f}ms/frame)")
    else:
        print("Benchmark only implemented for ONNX format")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",     default="weights/yolov8n_visdrone.pt")
    p.add_argument("--format",    choices=["onnx", "trt", "ncnn", "all"], default="onnx")
    p.add_argument("--imgsz",     type=int, default=640)
    p.add_argument("--int8",      action="store_true")
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--output-dir", default="weights")
    arg = p.parse_args()

    mod_pth = arg.model
    oup_pth = None

    if arg.format in ("onnx", "all"):
        oup_pth = exp_onnx(mod_pth, arg.output_dir, arg.imgsz)

    if arg.format in ("trt", "all"):
        oup_pth = exp_trt(mod_pth, arg.int8, arg.imgsz)

    if arg.format in ("ncnn", "all"):
        oup_pth = exp_ncc(mod_pth, arg.imgsz)

    if arg.benchmark and oup_pth:
        ben_chk(str(oup_pth), arg.imgsz)

    print("""
Edge targets:
  Jetson Orin Nano (INT8 TRT): ~25-30 FPS with SAHI
  Jetson Nano (FP16 TRT):      ~12-15 FPS with SAHI
  Raspberry Pi 4 (NCNN):       ~4-6 FPS without SAHI
""")


if __name__ == "__main__":
    main()
