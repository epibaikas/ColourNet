## ColourNet: Neural Net visualization with LEDs and fiber optics

This repository contains the design files and code developed to build a neural net visualization, inspired by 
https://github.com/ZackAkil/rgb-neural-net.

The neural net display is powered by a Raspberry Pi Pico W microcontroller, which is responsible for hosting an HTTP
web server and controlling the display's LED strip.

In main.py, a 3x3x2 Fully Connected neural network model is created and trained using the PyTorch framework. At the end
of each training epoch, an HTTP Post request is made to the web server to update the colour of the network weights.