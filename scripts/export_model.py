"""
Export YOLOv8n to ONNX and TensorRT for edge deployment.

Supports:
  - ONNX (cross-platform, works on Jetson, Raspberry Pi, etc.)
  - TensorRT (Jetson Orin/Nano — requires NVIDIA TRT)
  - INT8 quantization (for maximum Jetson performance)

Usage:
    python scripts/export_model.py --format onnx
    python scripts/export_model.py --format trt --int8
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def export_onnx(model_path: str, output_dir: str, imgsz: int = 640):
    """Export to ONNX with opset 12 (widely supported)."""
    model = YOLO(model_path)

    out = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=12,
        simplify=True,    # Run ONNX simplifier (reduces graph complexity)
        dynamic=False,    # Fixed batch size = 1 for embedded inference
        half=False,       # FP32 (set True for FP16 if GPU supports it)
    )
    print(f"✓ ONNX export: {out}")
    return out


def export_tensorrt(model_path: str, int8: bool = False, imgsz: int = 640):
    """
    Export to TensorRT engine.
    Run this on the Jetson device itself for best compatibility.
    INT8 calibration requires a calibration dataset for best accuracy.
    """
    model = YOLO(model_path)

    out = model.export(
        format="engine",
        imgsz=imgsz,
        half=True,       # FP16 (required for Jetson)
        int8=int8,       # INT8 quantization (faster, slight accuracy drop)
        device=0,        # Must run on CUDA device
        workspace=4,     # GB of workspace for TRT optimizer
    )
    print(f"✓ TensorRT export: {out}")
    return out


def export_ncnn(model_path: str, imgsz: int = 640):
    """
    Export to NCNN format.
    Ideal for ARM CPUs (Raspberry Pi, mobile) — no GPU required.
    """
    model = YOLO(model_path)
    out = model.export(format="ncnn", imgsz=imgsz)
    print(f"✓ NCNN export: {out}")
    return out


def benchmark(model_path: str, imgsz: int = 640, n: int = 100):
    """Quick FPS benchmark after export."""
    import time
    import numpy as np
    import cv2

    if model_path.endswith(".onnx"):
        import onnxruntime as ort
        sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        inp_name = sess.get_inputs()[0].name
        dummy = np.random.randn(1, 3, imgsz, imgsz).astype(np.float32)

        # Warmup
        for _ in range(5):
            sess.run(None, {inp_name: dummy})

        t0 = time.perf_counter()
        for _ in range(n):
            sess.run(None, {inp_name: dummy})
        elapsed = time.perf_counter() - t0
        fps = n / elapsed
        print(f"ONNX inference FPS: {fps:.1f} (avg {elapsed/n*1000:.1f}ms/frame)")
    else:
        print("Benchmark only implemented for ONNX format")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",   default="weights/yolov8n_visdrone.pt")
    p.add_argument("--format",  choices=["onnx", "trt", "ncnn", "all"], default="onnx")
    p.add_argument("--imgsz",   type=int, default=640)
    p.add_argument("--int8",    action="store_true", help="INT8 quantization (TRT only)")
    p.add_argument("--benchmark", action="store_true", help="Run FPS benchmark after export")
    p.add_argument("--output-dir", default="weights")
    args = p.parse_args()

    model_path = args.model
    out_path = None

    if args.format in ("onnx", "all"):
        out_path = export_onnx(model_path, args.output_dir, args.imgsz)

    if args.format in ("trt", "all"):
        out_path = export_tensorrt(model_path, args.int8, args.imgsz)

    if args.format in ("ncnn", "all"):
        out_path = export_ncnn(model_path, args.imgsz)

    if args.benchmark and out_path:
        benchmark(str(out_path), args.imgsz)

    print("""
Edge Deployment Notes:
  Jetson Orin Nano (INT8 TensorRT):  ~25-30 FPS with SAHI
  Jetson Nano (FP16 TensorRT):       ~12-15 FPS with SAHI
  Raspberry Pi 4 (NCNN):             ~4-6 FPS without SAHI
  
  For Jetson: Always run export on the device itself.
  INT8 calibration needs ~100 representative images for best accuracy.
""")


if __name__ == "__main__":
    main()
