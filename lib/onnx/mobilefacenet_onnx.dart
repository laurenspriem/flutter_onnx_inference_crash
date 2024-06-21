import 'dart:developer' show log;
import "dart:async";
import 'dart:math' show Random;
import 'dart:typed_data' show ByteData, Float32List;

import 'package:flutter/services.dart' show rootBundle;
import 'package:onnxruntime/onnxruntime.dart';

/// This class is responsible for running the face embedding model (MobileFaceNet) on ONNX runtime, and can be accessed through the singleton instance [MobilefacenetONNX.instance].
class MobilefacenetONNX {
  List<int> sessionAddresses = [0, 0, 0, 0, 0];

  static const name = 'MobilefacenetONNX';

  static const String modelPath = "assets/mobilefacenet_opset15.onnx";

  static const int kInputWidth = 112;
  static const int kInputHeight = 112;
  static const int kEmbeddingSize = 192;
  static const int kNumChannels = 3;

  bool isInitialized = false;

  static final inputImageList = Float32List.fromList(List.generate(
      1 * kInputHeight * kInputWidth * 3, (_) => Random().nextDouble()));

  // Singleton pattern
  MobilefacenetONNX._privateConstructor();
  static final instance = MobilefacenetONNX._privateConstructor();
  factory MobilefacenetONNX() => instance;

  /// Check if the interpreter is initialized, if not initialize it with `loadModel()`
  Future<void> init() async {
    if (!isInitialized) {
      log('init is called');
      OrtEnv.instance.init();
      for (int i = 0; i < sessionAddresses.length; i++) {
        sessionAddresses[i] = await _loadModel();
      }
      log("init is done, model loaded");
      if (sessionAddresses.every((address) => address != -1)) {
        isInitialized = true;
      }
    }
  }

  Future<void> release() async {
    if (isInitialized) {
      for (int i = 0; i < sessionAddresses.length; i++) {
        await _releaseModel(sessionAddresses[i]);
        sessionAddresses[i] = 0;
      }
      isInitialized = false;
    }
  }

  /// Initialize the interpreter by loading the model file.
  Future<int> _loadModel() async {
    final sessionOptions = OrtSessionOptions()
      ..setInterOpNumThreads(1)
      ..setIntraOpNumThreads(1)
      ..setSessionGraphOptimizationLevel(GraphOptimizationLevel.ortEnableAll);
    try {
      final ByteData rawAssetFile = await rootBundle.load(modelPath);
      final session = OrtSession.fromBuffer(
          rawAssetFile.buffer.asUint8List(), sessionOptions);
      return session.address;
    } catch (e, s) {
      log('Face embedding model not loaded:  ${e.toString()},\n ${s.toString()}');
    }
    return -1;
  }

  Future<void> _releaseModel(int sessionAddress) async {
    if (sessionAddress == 0 || sessionAddress == -1) {
      return;
    }
    final session = OrtSession.fromAddress(sessionAddress);
    session.release();
    return;
  }

  static Future<void> predict(Map args) async {
    final sessionAddress = args['sessionAddress'];
    assert(sessionAddress != 0 && sessionAddress != -1);

    final inputShape = [
      1,
      kInputHeight,
      kInputWidth,
      kNumChannels,
    ];
    final inputOrt = OrtValueTensor.createTensorWithDataList(
      inputImageList,
      inputShape,
    );
    final inputs = {'img_inputs': inputOrt};

    // Run inference
    List<OrtValue?>? outputs;
    try {
      final runOptions = OrtRunOptions();
      final session = OrtSession.fromAddress(sessionAddress);
      final time = DateTime.now();
      outputs = session.run(runOptions, inputs);
      // release everything
      inputOrt.release();
      runOptions.release();
      for (final element in outputs) {
        element?.release();
      }
      log(
        '[$name] interpreter.run is finished, in ${DateTime.now().difference(time).inMilliseconds} ms',
      );
    } catch (e, s) {
      log('Error while running inference: $e \n $s');
    }
  }
}
