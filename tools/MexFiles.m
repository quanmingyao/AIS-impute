% compile all c files

clear; clc;
mex partXY.c;
mex setSval.cpp;
mex updateSparse.c;