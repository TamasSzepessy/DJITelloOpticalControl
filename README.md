# Optical control for Tello drone

Using Damià Fuentes Escoté's library (https://github.com/damiafuentes/DJITelloPy) as a base.

For my bachelor thesis I created a control system for the Tello with the inbuilt, monocular camera: autonomous navigation and data collection. Demonstration: https://www.youtube.com/watch?v=B8aU0DVYYco

Tested with Python 3.7 and OpenCV 4.1.0 and 7x7 ArUco dictionary.

## Requirements

```
djitellopy
pygame
opencv-python 4.1.0
opencv-contrib 4.1.0
numpy
scipy
pykalman
matplotlib
```

## Instructions

The control system is based on ArUco markers, which must be placed correctly for the drone to navigate through them. Before use, you must generate the ArUco markers from the OpenCV library and list the function of each marker in `src/marker_list/marker_conf.csv` as a configuration file. Each marker type contributes to a setpoint in the markers local coordinate system (decoded by `targeter.py`), to which the drone must navigate to using a simple numeric PID control (`pid.py`), then it can continue its path to the nearest seen marker. All marker path calculations and storing are done by the `Markers` class in `marker_class.py`.

You must also calibrate the drone's camera beforehand with the chessboard calibration algorithm of OpenCV. In `cam_class.py` you can set multiple parameters for  chessboard tile edge length for calibration, marker edge length for measurements and edge filtering for distorted markers.

I modified and complemented the original `djitellopy/tello.py` script with the Tello state read, which runs in a separate thread, not blocking the main execution.

For flight, you must run the `main.py` script, which tries to connect automatically to your Tello. After successful connection, you can take off with "T" key, navigate with the arrow keys, control up-down with W/S and yaw with A/D, after flight, press "L" to land the drone.

In-flight, you can use the following keys for further functions:

- "M" takes a picture and saves it under `images`

- "K" starts calibration mode, which waits for a chessboard, then collects 20 samples of it and calibrates camera (the calibration matrix is saved under `src/calibration_files/camcalib.npz`)

- "C" starts capturing drone coordinates from seen ArUco markers, using the first as global origin (this can be used in manual flight mode)

- "O" starts automatic navigation and video capture through a marker path between a placed "Start" and "End" marker

All control is done in the main loop which calls the separate functions with events and flags. There are four threads running, one for pyGame window and GUI, two for Tello UDP communication and one for parallel video capture. Data can be sent thread safely between the separate threads using queues.

All matrix transformations are done by the functions in `transformations.py`. The basic principle is that when two markers are seen together at the same time, you can calculate the transformation matrix between the two coordinate systems. (Taking multiple samples for averaging the matrices between two markers.) The drone can map its path this way, using a chain of transformations from the global origin, the first seen marker used as a base. All coordinate transformations are done real time, the global points are then saved.

The remaining scripts are for post processing only:

- `plot3d.py` uses Mathplotlib Axes3D to show the flight path in a Cartesian coordinate system and animating it

- `video_writer.py` is the one running in a separate thread for capturing the drone's camera feed while in navigation

## Documentation

The documentation for the project can be found under `doc`. It is a Bachelor thesis written at the Budapest University of Technology and Economics, for the Faculty of Mechatronics. The original Hungarian only has an English abstract, but I used DeepL translator to create a full English version. (Any misinterpretations and broken links are because of this.)

## Author

* **Tamás Szepessy** 

## License

This project is licensed under the MIT License.
