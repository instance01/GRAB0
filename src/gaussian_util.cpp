#include <cmath>
#include "gaussian_util.hpp"


float c = std::sqrt(2 * M_PI);


// sigma ~= stddev. NOT variance.


// TODO Unused? Check gaussian_a2c.
float gaussian_pdf(float action, float mu, float sigma) {
  return (1 / (sigma * c)) * std::exp(-.5 * std::pow((action - mu) / sigma, 2.));
}

float sample(float mu, float sigma, std::mt19937 &generator) {
  // TODO Should not recreate distribution here all the time.
  std::normal_distribution<double> noise_distribution(0.0, 1.0);
  // Reparameterized
  auto noise = noise_distribution(generator);
  return mu + sigma * noise;
}
