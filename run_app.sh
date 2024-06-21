#!/bin/bash

git submodule update --init --recursive
./flutter/bin/flutter run -t lib/main.dart