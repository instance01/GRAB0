#include "easy.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>


template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

EasyEnv::EasyEnv() {
  max_steps = 100;
  expected_mean = {0., 0.};
  expected_stddev = {1., 1.};
  x = 0.;
  y = 0.;
}

EasyEnv::EasyEnv(EasyEnv &other) {
  max_steps = other.max_steps;
  steps = other.steps;
  expected_mean = other.expected_mean;
  expected_stddev = other.expected_stddev;
  x = other.x;
  y = other.y;
}

EasyEnv::~EasyEnv() {
}

std::vector<float>
EasyEnv::reset(std::mt19937 &generator_) {
  steps = 0;
  x = 0.;
  y = 0.;
  return {x, y};
}

std::tuple<std::vector<float>, double, bool>
EasyEnv::step(std::vector<double> &action) {
  steps += 1;
  if (steps > max_steps)
    return {{x, y}, 0., true};

  float action_x = std::clamp(action[0], -1., 1.);
  float action_y = std::clamp(action[1], -1., 1.);

  float goal_x = 1.;
  float goal_y = 1.;

  x += std::pow(action_x, 2) * sgn(action_x);
  y += std::pow(action_y, 2) * sgn(action_y);
  x = std::clamp((float) x, (float) -2., (float) 2.);
  y = std::clamp((float) y, (float) -2., (float) 2.);

  std::vector<float> state = {x, y};

  float reward = std::min(0.5 / std::abs(action_x), 5.) + std::min(0.5 / std::abs(action_y), 5.0);
  reward /= -10;

  bool done = false;
  if (std::abs(x - goal_x) < .2 && std::abs(y - goal_y) < .2) {
    done = true;
    reward = 100;
  }

  auto ret = std::make_tuple(state, reward, done);
  return ret;
}
