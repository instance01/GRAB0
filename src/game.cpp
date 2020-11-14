#include "game.hpp"

Game::calc_tot_reward_inplace() {
  // TODO: Gamma.
  for (auto rew : rewards) {
    tot_reward += rew;
  }
}

Game::Game(
  std::vector<std::vector<float>> states,
  std::vector<double> rewards,
  std::vector<std::vector<double>> mcts_actions
) : states(states), rewards(rewards), mcts_actions(mcts_actions) {
  calc_tot_reward_inplace();
}
