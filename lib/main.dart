import 'package:flutter/material.dart';
import 'package:flutter_onnx_inference_crash/onnx/mobilefacenet_onnx.dart';
import 'package:flutter_onnx_inference_crash/onnx/yolo_face_onnx.dart';

void main() {
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

  void _runInfiniteIndexing() {
    if (_isRunning) {
      return;
    }
    _isRunning = true;
    while (true) {
      YoloFaceONNX.instance.predict();
      MobilefacenetONNX.instance.predict();
      setState(() {
        _counter++;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    YoloFaceONNX.instance.init();
    MobilefacenetONNX.instance.init();
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
            ),
          ],
        ),
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: _runInfiniteIndexing,
        tooltip: 'Start running inference',
        child: const Icon(Icons.add),
      ),
    );
  }
}
