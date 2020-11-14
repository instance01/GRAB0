#ifndef GAME_HEADER
#define GAME_HEADER
#include <vector>

class Game {
  public:
    std::vector<std::vector<float>> states;
    std::vector<double> rewards;
    std::vector<std::vector<double>> mcts_actions;
    double tot_reward = 0.0;

    bool is_greedy = false;

    void calc_tot_reward_inplace();
    Game() {};
    Game(
      std::vector<std::vector<float>> states,
      std::vector<double> rewards,
      std::vector<std::vector<double>> mcts_actions
    );
    Game(
      std::vector<std::vector<float>> states,
      std::vector<double> rewards,
      std::vector<std::vector<double>> mcts_actions,
      double tot_reward
    ) : states(states), rewards(rewards), mcts_actions(mcts_actions), tot_reward(tot_reward) {};
    ~Game() {};
};
#endif
