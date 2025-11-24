## Visual Servoing Assignment

Edit the file `cv_control.cpp` with your changes.


#### Building
To compile your code:
```
mkdir build
cd build
cmake ..
make
```

To compile with real camera support (make sure you have a USB camera plugged in), you should instead run `cmake` with the `-DCMAKE_BUILD_TYPE=RealCamera` option.

With a real camera, the simulated robot will not be able to center the image you wave in front of the camera. However, you must test this anyway in order to validate that your image processing pipeline works in real-world camera / lighting conditions. If you do not test this, your code may not work in the real robot presentation and you may lose points.

#### Running
Start `pumasim` in the build/ directory.
