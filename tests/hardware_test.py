#!/usr/bin/env python3
"""
hardware_test.py

Quick checks to determine whether an Intel NPU is reachable/usable
by the typical ipex-llm / Intel stack.

Notes:
 - This script tries safe, non-invasive checks only (no heavy model loads).
 - It uses multiple signals because different stacks (OpenVINO, PyTorch/iPEX, intel-npu-accel)
   expose the NPU differently.
"""

import importlib
import sys
import traceback

def try_import(name):
    try:
        m = importlib.import_module(name)
        ver = getattr(m, "__version__", None)
        return (True, m, ver)
    except Exception as e:
        return (False, None, None)

def check_ipex_llm():
    ok, mod, ver = try_import("ipex_llm")
    if not ok:
        return {"installed": False, "version": None, "detail": "ipex_llm not importable"}
    return {"installed": True, "version": ver, "detail": "ipex_llm import OK"}

def check_torch_xpu():
    ok, torch, _ = try_import("torch")
    if not ok:
        return {"installed": False, "xpu_available": None, "detail": "torch not installed"}
    # Check common Intel XPU indicator
    xpu_avail = None
    try:
        if hasattr(torch, "xpu") and callable(getattr(torch.xpu, "is_available", None)):
            xpu_avail = bool(torch.xpu.is_available())
            return {"installed": True, "xpu_available": xpu_avail, "detail": "torch.xpu.is_available() checked"}
        # Older/newer builds might expose different entrypoints - try generic device query:
        # Many users also rely on torch.cuda (NVIDIA) or other backends; include them for completeness:
        cuda_avail = getattr(torch.cuda, "is_available", lambda: False)()
        return {"installed": True, "xpu_available": False, "cuda_available": bool(cuda_avail),
                "detail": "torch present but torch.xpu not available in this build"}
    except Exception as e:
        return {"installed": True, "xpu_available": None, "detail": f"error checking torch.xpu: {e}"}

def check_openvino_npu():
    ok, ov, ver = try_import("openvino")
    if not ok:
        # older package name may be 'openvino.runtime'
        ok2, ov2, ver2 = try_import("openvino.runtime")
        if not ok2:
            return {"installed": False, "npu_seen": None, "detail": "openvino not installed"}
        ov = ov2
    try:
        # OpenVINO runtimes expose Core
        Core = getattr(ov, "Core", None)
        if Core is None:
            # try runtime submodule
            from openvino.runtime import Core as Core2  # type: ignore
            Core = Core2
        core = Core()
        devices = list(core.available_devices)
        npu_seen = any("NPU" in d.upper() for d in devices)
        return {"installed": True, "devices": devices, "npu_seen": npu_seen, "detail": "OpenVINO available_devices checked"}
    except Exception as e:
        return {"installed": True, "npu_seen": None, "detail": f"error querying OpenVINO: {e}"}

def check_intel_npu_accel_lib():
    ok, mod, ver = try_import("intel_npu_acceleration_library")
    if not ok:
        return {"installed": False, "version": None, "detail": "intel_npu_acceleration_library not installed"}
    # library typically falls back to AUTO mode and may warn if NPU is missing.
    return {"installed": True, "version": ver, "detail": "intel_npu_acceleration_library import OK (may still fallback if driver missing)"}

def check_npu_devices():
    print("\n=== NPU reachability checks for ipex_llm ===\n")

    # 1) ipex_llm import
    print("1) ipex_llm package:")
    ipex = check_ipex_llm()
    if ipex["installed"]:
        print(f"   - ipex_llm import OK (version: {ipex['version']})")
    else:
        print("   - ipex_llm NOT importable. (pip install may be required)")

    # 2) PyTorch / xpu
    print("\n2) PyTorch / XPU check:")
    tx = check_torch_xpu()
    if not tx["installed"]:
        print("   - PyTorch not installed.")
    else:
        xpu_av = tx.get("xpu_available", None)
        if xpu_av is True:
            print("   - torch.xpu.is_available() == True (XPU backend available).")
        elif xpu_av is False:
            # also show CUDA if present
            if tx.get("cuda_available"):
                print("   - torch.xpu.is_available() == False, but CUDA is available (NVIDIA GPU).")
            else:
                print("   - torch.xpu.is_available() == False (XPU not available in this torch build / drivers).")
        else:
            print(f"   - Unable to determine torch.xpu availability: {tx.get('detail')}")

    # 3) OpenVINO device listing (good signal for Intel NPU driver presence)
    print("\n3) OpenVINO device listing (if OpenVINO installed):")
    ov = check_openvino_npu()
    if not ov["installed"]:
        print("   - OpenVINO not installed (skipping).")
    else:
        devices = ov.get("devices", [])
        print(f"   - OpenVINO devices: {devices}")
        if ov.get("npu_seen"):
            print("   - NPU FOUND in OpenVINO available_devices -> NPU appears reachable.")
        elif ov.get("npu_seen") is False:
            print("   - NPU not listed by OpenVINO (driver may be missing or not functional).")
        else:
            print(f"   - Could not determine NPU via OpenVINO: {ov.get('detail')}")

    # 4) Intel NPU Acceleration Library
    print("\n4) intel_npu_acceleration_library (optional support library):")
    npu_lib = check_intel_npu_accel_lib()
    if not npu_lib["installed"]:
        print("   - intel_npu_acceleration_library not installed.")
    else:
        print(f"   - intel_npu_acceleration_library installed (version: {npu_lib.get('version')}). "
              "Import may still warn or fallback to AUTO if driver not present.")

    # 5) Heuristics / summary
    print("\n=== Summary / heuristic ===")
    npu_reachable = False
    reasons = []

    # If OpenVINO explicitly sees NPU -> strong signal
    if ov.get("npu_seen") is True:
        npu_reachable = True
        reasons.append("OpenVINO lists 'NPU' in available_devices (strong signal).")
    # else, if intel_npu_accel library present AND ipex_llm present, that's a positive signal (weaker)
    if npu_lib.get("installed") and ipex.get("installed"):
        reasons.append("Both ipex_llm and intel_npu_acceleration_library are importable (positive sign, but drivers still required).")
    # if torch.xpu true => indicates XPU/GPU, but not necessarily NPU
    if tx.get("xpu_available") is True:
        reasons.append("torch.xpu indicates XPU (Intel GPU) is available - not the NPU but useful if you intended GPU.")
    # decide
    if npu_reachable:
        print("Result: ✅ NPU appears reachable from user-space (OpenVINO detected it).")
    else:
        print("Result: ❌ NPU not detected by the checks above (OpenVINO did not list an NPU).")
    print("\nReasons / notes:")
    for r in reasons:
        print(f" - {r}")

    print("\nSuggested next steps if NPU not detected:")
    print(" - Ensure OS drivers for Intel NPU are installed (Intel recommends Ubuntu 24+ and proper drivers).")
    print(" - Install OpenVINO and verify `core.available_devices` returns 'NPU' (this is a reliable signal).")
    print(" - Install the Intel NPU Acceleration Library (pip install intel-npu-acceleration-library) and check for import warnings.")
    print(" - If using ipex-llm for GPU/XPU, ensure the `ipex-llm[xpu]` variant and Intel oneAPI / compute runtime are installed.")
    print("\n(See printed output above for which packages are missing.)")

if __name__ == "__main__":
    try:
        check_npu_devices()
    except Exception:
        print("Unexpected error while running checks:")
        traceback.print_exc()
        sys.exit(2)