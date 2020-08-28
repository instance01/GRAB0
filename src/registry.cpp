#include <memory>
#include <iostream>
#include "registry.hpp"

void
Registry::save_if_best(Game game, double reward) {
  // It's thread safe !!
  lock.lock();
  // Strictly 'greater than' was a conscious choice.
  if (reward > best_reward) {
    game.is_greedy = true;

    best_reward = reward;
    best_game = game;

    // for (auto actions : game.mcts_actions) {
    //     for (auto a : actions) {
    //         std::cout << a << " ";
    //     }
    //     std::cout << "|";
    // }
    // std::cout << std::endl << reward << std::endl;

    replay_buffer->add(std::make_shared<Game>(std::move(game)));
  }
  lock.unlock();
}
