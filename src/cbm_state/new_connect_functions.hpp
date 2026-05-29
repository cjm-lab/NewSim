#pragma once

#include <iostream>
#include <assert.h>
#include <algorithm>
#include <cstring>
#include <random>
#include <cmath>
#include <numeric>
#include <functional>

// util functions here just for now
int* make_shuffled_arr(int max_val);

float get_distance_wrapped(int Ax, int Ay, int Bx, int By, int Nx, int Ny);

void connectGOtoGO(int** GOtoGO_post, int GOConv, float recip,
                   int numGO, int GOx, int GOy, int span);