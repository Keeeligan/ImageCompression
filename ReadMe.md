# Image compression
This repository contains the Individual Propedeuse Assessment (IPASS). The assignment was to write an
image compression algorithm. This repository contains two image compression algorithms, being a
simple but rough one, and the other a more complex one. 

For a more detailed description on the algorithms and this assignment check the .docx in the 
"Documentation/" folder.

For the time being the complex algorithm DOES NOT WORK as it should due to a bug in the DCT functions.
Using the GUI is still possible, but the output image for the complex algorithm isn't what it should be.


## GUI
The GUI is built around the input and output image, to start you can select an input image from the
.png files listed in the bottom left. When selecting an image the image will appear in the top left
indicating that you clicked on the image.

Once an image has been selected you can press one of the build buttons to start the building process
for that image, depending on the image size and which algorithm you chose this might take a while.
After the image has been built it will show up as a .pickle file in the bottom right. At the end of
the file name it will say which algorithm was used on that data. The GUI also prevents you from
accidentally building a file that was build with the other algorithm. 

When the .png is done compressing into a .pickle file, it is ready to be built. To build the .pickle
file you first need to select it in the list in the bottom left. Once a .pickle has been selected, you
press the build button for which algorithm the file was made. This will start the rebuilding process
of the image.


## General structure
The project is built upon the GUI ("gui.py"), the GUI sends the commands behind the button presses to 
"image_main.py" which controls the algorithms and has functions to open and save data.

In "algorithm_simple.py" and "algorithm_complex.py" are their respective algorithms located, both for
compressing and rebuilding that compressed data via the specified algorithm.

The images to be compressed are located in the "images/IN/" directory. When an image is compressed, it 
stored in a .pickle file in the "images/STORE/" directory. The rebuilt images are found in the 
"images/OUT/" folder.


## Algorithms used
As said before this app can run two algorithms on images to compress- and build those images.

### Simple algorithm
The simple algorithm is based on two steps. The first one is colour compression. What colour compression
does is that it takes the RGB values and floors them to the closest multiple of 4. So, if a pixel has the
value: (203, 210, 206), it will floor it to: (200, 208, 204). The reason this step is done is to make
more adjacent pixels have the same value.

This helps with the next step, Run Length Encoding (RLE). What RLE does is that it can group adjacent 
pixels. So, if the value (255, 255, 255) is repeated 10 times, it will save ((255, 255, 255), 10) instead 
of saving (255, 255, 255) 10 times individually, saving a lot of space. 

To Rebuild this you only have to paste the colour times the second value in the tuple, creating the 
original image.

### Complex algorithm
The complex algorithm uses a lot more steps, these steps are based off the JPEG algorithm. This algorithm
uses 5 main steps, namely:

1. Colour Space Conversion
2. Chrominance Downsampling
3. Discrete Cosine Transform
4. Quantization
5. Run Length Encoding

Let's start with Colour Space Conversion. When a colour is displayed onto a pixel it will have 3 values:
a red value, a green value, and a blue value (RGB). With these colours you're able to create all the colours
you need to create a good-looking image. But with you can also create an image using different values, YCbCr.
The YCbCr colour space is stored differently to the RGB colour space. It uses 3 main values, namely:
Luminance (Y), Blue Chrominance (Cb), and Red Chrominance (Cr). 

In the next step we can downsample the Chrominance layers, this will not affect the image too much, since
we still have a full sized luminance layer supporting the - now lower quality - chrominance layers. The
chrominance layers will have 1/4th of the original size.

For the third step we use Discrete Cosine Transform (DCT), this step divides the image into blocks of 8x8.
In each block of 8x8 is counted which base images (set of 64) are needed to recreate each block, a more 
detailed description is given within the .docx file in the "documentation/" folder. It is important to note 
that DCT is performed on each channel of YCbCr individually.

The previous step results in an 8x8 block of coefficients of each block that is needed to recreate the image, 
quantization is used to scale these values down, keeping mostly the lower frequency parts of that 8x8 block.

The block of coefficients now doesn't contain many high frequency parts, which means we deleted a lot of
the data whilst the original image hasn't changed much. This can easily be stored using RLE, since pretty
much all the high frequency coefficients in the block are zeros.

To rebuild this data back to an original is explained in the .docx mentioned earlier.


## How to run
To run this file you need to make sure you have installed all the packages. A list from which packages
were used are found below. To run the app all you have to do is run "main.py".

### Packages used
- pygame
- Pillow
- numpy
- pickle
- pandas
- os

## License
Specify what license is used for this project