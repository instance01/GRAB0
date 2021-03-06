#include "gridworld.hpp"
#include <cmath>
#include <utility>
#include <tuple>
#include <iostream>


GridWorldEnv::GridWorldEnv(GridWorldEnv &other) {
  start = other.start;
  goal = other.goal;
  pos = other.pos;
  dir = other.dir;
  width = other.width;
  height = other.height;
  max_steps = other.max_steps;
  steps = other.steps;
  blocks = other.blocks;
}

GridWorldEnv::GridWorldEnv(
    int width, int height, std::set<std::pair<int, int>> blocks
) : width(width), height(height), blocks(blocks) {
  max_steps = width * height * 4;
  std::mt19937 gen;
  reset(gen);

  expected_mean = {(width - 1) / 2., (height - 1) / 2., .75};
  // std dev of uniform distribution:
  expected_stddev = {width / std::sqrt(12), height / std::sqrt(12), 4. / std::sqrt(12)};
}

void
GridWorldEnv::move(const int &action) {
  // Handle turning left/right.
  if (action == 0)
    dir -= 1;
  if (action == 1)
    dir += 1;
  // That's some weird bullshit man.
  dir = (dir % 4 + 4) % 4;
  //dir %= 4;

  if (action == 2) {
    int xdelta = 0;
    int ydelta = 0;

    if (dir == 0)
      xdelta = 1;
    if (dir == 1)
      ydelta = 1;
    if (dir == 2)
      xdelta = -1;
    if (dir == 3)
      ydelta = -1;

    int newx = pos.first + xdelta;
    int newy = pos.second + ydelta;

    // Check if there is a block at the new location.
    auto key = std::make_pair(newx, newy);
    if (blocks.count(key))
      return;

    // Do the move.
    pos.first = std::min(width, std::max(0, newx));
    pos.second = std::min(height, std::max(0, newy));
  }
}

std::vector<float>
GridWorldEnv::reset(std::mt19937 &generator) {
  start = std::make_pair(1, 1);
  goal = std::make_pair(width - 2, height - 2);
  dir = 0;
  pos = std::make_pair(1, 1);
  steps = 0;

  return {1., 1., 0.};
}

std::tuple<std::vector<float>, double, bool>
GridWorldEnv::step(const int &action) {
  steps += 1;
  if (steps >= max_steps) {
    std::vector<float> obs = {(float) pos.first, (float) pos.second, (float) dir};
    auto ret = std::make_tuple(obs, 0, true);
    return ret;
  }

  move(action);
  double reward = 0.;
  bool done = false;
  if (pos.first == goal.first && pos.second == goal.second) {
    done = true;
    reward = 1. - .9 * (1. * steps / max_steps);
  }
  std::vector<float> obs = {(float) pos.first, (float) pos.second, (float) dir};
  auto ret = std::make_tuple(obs, reward, done);
  return ret;
}
