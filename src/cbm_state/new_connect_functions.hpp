#pragma once

#include <iostream>
#include <assert.h>
#include <algorithm>
#include <cstring>
#include <random>
#include <cmath>
#include <numeric>
#include <functional>
#include <chrono>

// util functions 
std::function<int()> make_generator(int min_val, int max_val);

int* make_shuffled_arr(int max_val);

std::vector<int> make_candidate_vector_from_2D_range(int posX, int posY, int spanX, int spanY, int numX, int numY);

struct Timer {
    std::chrono::high_resolution_clock::time_point startTime;
    std::chrono::high_resolution_clock::time_point endTime;
    std::string message;
    std::string timeFrame;

    Timer(std::string m, std::string tf);
    ~Timer();
    void start();
    void stop();
    void print();
};

void assignMFtoGL(int* MFtoGL_GL, int numGL, int numMF, int MFmaxNumPresyn);

void connectGLtoGR(int** MFtoGR_GR, const int* MFtoGL_GL, int GRConv,
    int numGR, int GRx, int GRy, int GRspanX, int GRspanY,
    int GLx, int GLy);

void connectGLtoGO(int** MFtoGO_GO, const int* MFtoGL_GL, 
    int numGO, int GOx, int GOy, int spanX, int spanY,
    int GLx, int GLy, int numGODend);

void connectGOtoGR(int** GOtoGR_GR, int GRConv,
    int numGR, int spanX, int spanY,
    int GOx, int GOy, int GRx, int GRy);

void connectGRtoGO(int** GRtoGO_GO, int GOConv, int numGO,
    int GOx, int GOy, int GRx, int GRy,
    int spanX, int spanY);

void connectGOtoGO(int** GOtoGO_post, int GOConv, float recip,
                   int numGO, int GOx, int GOy, int span);