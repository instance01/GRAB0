#ifndef REGISTRY_HEADER
#define REGISTRY_HEADER
#include <vector>
#include <mutex>
#include "replay_buffer.hpp"
#include "game.hpp"

class Registry {
  public:
    // General
    Registry(ReplayBuffer *replay_buffer) : replay_buffer(replay_buffer) {};
    ~Registry() {};
    std::mutex lock;

    // TODO: Bandits
    // std::vector<SingleGradientBandit> bandits;

    // Best game
    ReplayBuffer *replay_buffer;
    double best_reward = -std::numeric_limits<double>::infinity();
    Game best_game;
    void save_if_best(Game game, double reward);
};
#endif
