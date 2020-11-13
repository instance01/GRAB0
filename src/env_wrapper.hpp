#ifndef ENV_W_HEADER
#define ENV_W_HEADER
#include <random>
#include <string>
#include <variant>
#include <vector>
#include <map>
#include <mutex>
#include <memory>

#include "envs/env.hpp"
#include "cfg.hpp"


class EnvWrapper {
  public:
    std::shared_ptr<Env> env;
    json params;
    std::string game;

    int reward_exponent;

    EnvWrapper() {};
    ~EnvWrapper() {};

    void init(std::string game, json params);

    std::tuple<std::vector<float>, double, bool> step(std::vector<double> &action);
    std::tuple<std::vector<float>, double, bool> step(const int &action);
    std::vector<float> reset(std::mt19937 &generator);
    std::unique_ptr<EnvWrapper> clone();
};
#endif
