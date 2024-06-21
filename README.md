# flutter_onnx_inference_crash

This is an MRE for a [reported ONNX issue on Android](https://github.com/microsoft/onnxruntime/issues/21082).

To reproduce the crash, simply follow these steps:

1. Connect an Android phone or emulator
2. Run `run_app.sh` to start the app
3. Hit the floating button in bottom right corner to start inference
4. Wait for the app to crash (somewhere in 100 to 1000 inference runs)

If the app does not start, you might have to properly [install Flutter](https://docs.flutter.dev/get-started/install/linux/android?tab=download).
