import "dart:async";
import 'dart:math' show Random;
import 'dart:typed_data' show ByteData, Float32List;
import 'dart:ui' as ui show Image;

import 'package:flutter/services.dart' show rootBundle;
import 'package:logging/logging.dart';
import 'package:onnxruntime/onnxruntime.dart';

/// This class is responsible for running the face detection model (YOLOv5Face) on ONNX runtime, and can be accessed through the singleton instance [YoloFaceONNX.instance].
class YoloFaceONNX {
  static final _logger = Logger('YoloFaceONNX');

  int sessionAddress = 0;

  static const String modelPath = "assets/yolov5s_face_640_640_dynamic.onnx";

  static const int kInputWidth = 640;
  static const int kInputHeight = 640;
  static const int kNumChannels = 3;
  static const double kIouThreshold = 0.4;
  static const double kMinScoreSigmoidThreshold = 0.7;
  static const int kNumKeypoints = 5;

  bool isInitialized = false;

  // Singleton pattern
  YoloFaceONNX._privateConstructor();
  static final instance = YoloFaceONNX._privateConstructor();
  factory YoloFaceONNX() => instance;

  /// Check if the interpreter is initialized, if not initialize it with `loadModel()`
  Future<void> init() async {
    if (!isInitialized) {
      _logger.info('init is called');
      sessionAddress = await _loadModel();
      _logger.info("Face detection model loaded");
      if (sessionAddress != -1) {
        isInitialized = true;
      }
    }
  }

  Future<void> release() async {
    if (isInitialized) {
      await _releaseModel();
      isInitialized = false;
      sessionAddress = 0;
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
      _logger.severe('Face embedding model not loaded', e, s);
    }
    return -1;
  }

  Future<void> _releaseModel() async {
    if (sessionAddress == 0) {
      return;
    }
    final session = OrtSession.fromAddress(sessionAddress);
    session.release();
    return;
  }

  /// Detects faces in the given image data.
  Future<void> predict() async {
    assert(sessionAddress != 0 && sessionAddress != -1);

    final random = Random();
    final randomFloats = List.generate(
        1 * kInputHeight * kInputWidth * 3, (_) => random.nextDouble());
    final inputImageList = Float32List.fromList(randomFloats);
    final inputShape = [
      1,
      kNumChannels,
      kInputHeight,
      kInputWidth,
    ];
    final inputOrt = OrtValueTensor.createTensorWithDataList(
      inputImageList,
      inputShape,
    );
    final inputs = {'input': inputOrt};

    // Run inference
    List<OrtValue?>? outputs;
    try {
      final runOptions = OrtRunOptions();
      final session = OrtSession.fromAddress(sessionAddress);
      outputs = session.run(runOptions, inputs);
      // release everything
      inputOrt.release();
      runOptions.release();
      for (final element in outputs) {
        element?.release();
      }
    } catch (e, s) {
      _logger.severe('Error while running inference: $e \n $s');
    }
    _logger.info(
      'interpreter.run is finished',
    );
  }
}
