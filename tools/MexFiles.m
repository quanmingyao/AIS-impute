% compile all c files

clear; clc;
mex partXY.c;
mex setSval.c;
mex updateSparse.c;