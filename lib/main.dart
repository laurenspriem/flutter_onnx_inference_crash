import 'dart:async';
import 'dart:developer' show log;

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter_onnx_inference_crash/onnx/mobilefacenet_onnx.dart';
import 'package:flutter_onnx_inference_crash/onnx/yolo_face_onnx.dart';

void main() async {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter ONNX crash',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: const MyHomePage(title: 'Flutter ONNX crash'),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key, required this.title});

  final String title;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  int _counter = 0;

  bool _isRunning = false;

  Future<void> _runInfiniteIndexing() async {
    log("_runInfiniteIndexing called");
    if (_isRunning) {
      return;
    }
    _isRunning = true;
    try {
      log('start inference loop');
      while (true) {
        await YoloFaceONNX.instance.init();
        await MobilefacenetONNX.instance.init();
        await compute(
            YoloFaceONNX.predict, YoloFaceONNX.instance.sessionAddress);
        await compute(MobilefacenetONNX.predict,
            MobilefacenetONNX.instance.sessionAddress);
        setState(() {
          _counter++;
        });
        // Future.delayed(const Duration(milliseconds: 100));
      }
    } catch (e, s) {
      log(e.toString());
      log(s.toString());
    }
  }

  @override
  Widget build(BuildContext context) {
    // try {
    //   log('Initializing models');
    //   await YoloFaceONNX.instance.init();
    //   await MobilefacenetONNX.instance.init();
    //   log('Initialization done');
    // } catch (e, s) {
    //   log('Error initializing models');
    //   log(e.toString());
    //   log(s.toString());
    // }
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        title: Text(widget.title),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            const Text(
              'Inference runs of both models:',
            ),
            Text(
              '$_counter',
              style: Theme.of(context).textTheme.headlineMedium,
              key: ValueKey(_counter),
            ),
          ],
        ),
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: () {
          unawaited(_runInfiniteIndexing());
        },
        tooltip: 'Start running inference',
        child: const Icon(Icons.add),
      ),
    );
  }
}
