#ifndef REPLAY_BUFFER_HEADER
#define REPLAY_BUFFER_HEADER
#include <deque>
#include <memory>
#include <random>
#include "cfg.hpp"
#include "game.hpp"

class ReplayBuffer {
  public:
    std::deque<std::shared_ptr<Game>> buffer;
    int window_size;
    bool prioritized_sampling = true;
    std::mt19937 generator;

    float experimental_top_cutoff;

    ReplayBuffer(int window_size, float experimental_top_cutoff, bool prioritized_sampling=false);
    ~ReplayBuffer() {};

    int _uniform();
    int _prioritized(std::vector<double> rewards);
    void add(
        std::vector<std::vector<float>> states,
        std::vector<double> rewards,
        std::vector<std::vector<double>> mcts_actions
    );
    void add(std::shared_ptr<Game> game);
    std::vector<double> get_rewards();
    std::shared_ptr<Game> sample();
    std::shared_ptr<Game> get_best();
    std::vector<std::shared_ptr<Game>> get_top();
    std::vector<std::shared_ptr<Game>> get_top_p(float percentile);
};
#endif
