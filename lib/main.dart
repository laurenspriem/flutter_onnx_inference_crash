import 'dart:async';
import 'dart:developer' show log;

import 'package:computer/computer.dart';
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

  Future<void> _runInfiniteIndexing(int addressIndex) async {
    log("_runInfiniteIndexing called");
    try {
      log('start inference loop');
      while (true) {
        final computer = Computer.shared();
        await computer.compute(YoloFaceONNX.predict, param: {
          "sessionAddress": YoloFaceONNX.instance.sessionAddresses[addressIndex]
        });
        await computer.compute(MobilefacenetONNX.predict, param: {
          "sessionAddress":
              MobilefacenetONNX.instance.sessionAddresses[addressIndex]
        });
        setState(() {
          _counter += 1;
        });
      }
    } catch (e, s) {
      log(e.toString());
      log(s.toString());
    }
  }

  Future<void> runAsync() async {
    if (_isRunning) {
      return;
    }
    _isRunning = true;
    await YoloFaceONNX.instance.init();
    await MobilefacenetONNX.instance.init();
    final computer = Computer.shared();
    await computer.turnOn(workersCount: 5);
    for (int i = 0; i < 5; i++) {
      unawaited(_runInfiniteIndexing(i));
    }
  }

  @override
  Widget build(BuildContext context) {
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
          unawaited(runAsync());
        },
        tooltip: 'Start running inference',
        child: const Icon(Icons.add),
      ),
    );
  }
}
