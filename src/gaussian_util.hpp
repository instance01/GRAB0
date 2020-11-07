#ifndef GAUSSIAN_UTIL
#define GAUSSIAN_UTIL
#include <random>

float gaussian_pdf(float action, float mu, float sigma);
float sample(float mean, float var, std::mt19937 &generator);

#endif
