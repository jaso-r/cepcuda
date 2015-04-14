# cepcuda
A CUDA implementation of Caloric-cost geodesics over digital elevation data.

This is very very very rough code right now.  I'm sorry.  It's a mishmash of CUDA example code and
a bunch of old code from my master's thesis.

Most of my parallel raster scan algorithm is based off of
"Parallel algorithms for approximation of distance maps on parametric surfaces"
http://visl.technion.ac.il/bron/publications/WebDevBroBroKimTOG08.pdf

In order to get this to run you'll need freeglut, glew, and CUDA v6.5.
