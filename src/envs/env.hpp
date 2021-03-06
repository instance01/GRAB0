#ifndef ENV_HEADER
#define ENV_HEADER
#include <random>
#include <string>
#include <variant>
#include <vector>
#include <map>
#include <mutex>
#include <memory>


class Env {
  public:
    Env();
    ~Env();

    // For A2C. Just needs to be rough.
    std::vector<float> expected_mean;
    std::vector<float> expected_stddev;
    int max_steps;

    virtual std::tuple<std::vector<float>, double, bool> step(const int &action) {return {};};
    virtual std::tuple<std::vector<float>, double, bool> step(std::vector<double> &action) {return {};};
    virtual std::vector<float> reset(std::mt19937 &generator) {return {};};
};
#endif
