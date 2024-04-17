## OpenCV Application

This C++ file contains various functions for image processing and manipulation using OpenCV (Open Source Computer Vision Library). Below is a brief overview of the functions provided in the file:

### Functions:

1. **testOpenImage()**: Opens a single image selected by the user and displays it.
2. **testOpenImagesFld()**: Opens all BMP images in a selected folder and displays them.
3. **testImageOpenAndSave()**: Opens a predefined image, converts it to grayscale, and saves the result.
4. **testNegativeImage()**: Converts the selected image to its negative using pixel-wise operations.
5. **testNegativeImageFast()**: Similar to `testNegativeImage()` but employs a faster approach using pointers.
6. **testColor2Gray()**: Converts a color image to grayscale using pixel-wise operations.
7. **testBGR2HSV()**: Converts a color image from the BGR color space to the HSV color space and displays the individual components (Hue, Saturation, Value).
8. **testResize()**: Resizes the selected image with and without interpolation and displays the results.
9. **testCanny()**: Applies the Canny edge detection algorithm to the selected image and displays the edges.
10. **testVideoSequence()**: Reads a video file frame by frame, converts each frame to grayscale, and performs edge detection.
11. **testSnap()**: Captures frames from a live video feed and allows the user to save snapshots.
12. **testMouseClick()**: Displays an image and prints color information (RGB) when the user clicks on it.
13. **changeGrayLevels()**: Alters the intensity levels of a grayscale image by a user-defined value.
14. **showColoredSquare()**: Displays a colored square with a black border.
15. **splitChannels()**: Splits the color channels (Red, Green, Blue) of a color image and displays them separately.
16. **BGRtoGrayScale()**: Converts a color image to grayscale.
17. **grayScaleToBlack()**: Converts a grayscale image to black and white based on a threshold.
18. **BGRtoHSV()**: Converts a color image from BGR to HSV color space.
19. **histogramGenerator()**: Generates a histogram of the pixel intensities in a grayscale image and performs thresholding.
20. **floydSteinberg()**: Applies the Floyd-Steinberg dithering algorithm to a grayscale image.

### Usage:
1. Compile the C++ file with an appropriate compiler.
2. Run the compiled executable.
3. Select an option from the menu to perform the desired image processing operation.

### Dependencies:
- OpenCV 2.x or higher.
- Windows operating system (due to the use of certain Windows-specific functions like `openFileDlg`).

