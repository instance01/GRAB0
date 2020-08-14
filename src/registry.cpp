#include <memory>
#include "registry.hpp"

void
Registry::save_if_best(Game game, double reward) {
  // It's thread safe !!
  lock.lock();
  // Strictly 'greater than' was a concious choice.
  if (reward > best_reward) {
    game.is_greedy = true;

    best_reward = reward;
    best_game = game;

    std::shared_ptr<Game> game_ = std::make_shared<Game>(std::move(game));
    replay_buffer->add(game_);
  }
  lock.unlock();
}
