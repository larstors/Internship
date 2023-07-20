all: Task1

Task1: lib.c
	gcc -shared -o lib.so -fPIC lib.c -Wall -Ofast -lm -flto -lgsl -lgslcblas -fopenmp
	