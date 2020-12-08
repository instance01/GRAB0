#ifndef EASY_HEADER
#define EASY_HEADER
#include <random>
#include <tuple>
#include <utility>
#include <vector>
#include <set>
#include "env.hpp"


class EasyEnv : public Env {
  public:
    std::mt19937 generator;
    int steps = 0;
    float x;
    float y;

    EasyEnv();
    ~EasyEnv();
    EasyEnv(EasyEnv &other);

    std::vector<float> reset(std::mt19937 &generator);
    std::tuple<std::vector<float>, double, bool> step(std::vector<double> &action);
};
#endif
