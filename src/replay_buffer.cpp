#include <numeric>
#include <random>
#include <iostream>

#include "game.hpp"
#include "replay_buffer.hpp"


ReplayBuffer::ReplayBuffer(int window_size, bool prioritized_sampling)
  : window_size(window_size), prioritized_sampling(prioritized_sampling)
{
  std::random_device rd;
  generator = std::mt19937(rd());
}

void
ReplayBuffer::add(
    std::vector<std::vector<float>> states,
    std::vector<double> rewards,
    std::vector<std::vector<double>> mcts_actions
) {
  std::shared_ptr<Game> game = std::make_shared<Game>(states, rewards, mcts_actions);
  buffer.push_back(game);
  if (buffer.size() > window_size)
    buffer.pop_front();
}

void
ReplayBuffer::add(std::shared_ptr<Game> game) {
  buffer.push_back(game);
  if (buffer.size() > window_size)
    buffer.pop_front();
}

std::vector<double>
ReplayBuffer::get_rewards() {
  std::vector<double> tot_reward_per_game;
  for (auto game : buffer) {
    tot_reward_per_game.push_back(std::accumulate(game->rewards.begin(), game->rewards.end(), 0.));
  }
  return tot_reward_per_game;
}

int
ReplayBuffer::_uniform() {
  std::uniform_int_distribution<> distribution(0, buffer.size() - 1);
  return distribution(generator);
}

int
ReplayBuffer::_prioritized(std::vector<double> rewards) {
  std::discrete_distribution<int> distribution(rewards.begin(), rewards.end());
  return distribution(generator);
}

std::shared_ptr<Game>
ReplayBuffer::sample() {
  int idx = 0;
  if (!prioritized_sampling) {
    idx = _uniform();
  } else {
    auto rewards = get_rewards();
    // TODO This accumulate might become a costly operation at some point.
    if (std::accumulate(rewards.begin(), rewards.end(), 0.) == 0.) {
      idx = _uniform();
    } else {
      idx = _prioritized(rewards);
    }
  }
  return buffer[idx];
}

std::shared_ptr<Game>
ReplayBuffer::get_best() {
  // Get best greedy game.
  std::vector<double> rewards = get_rewards();
  std::vector<int> indices;
  std::vector<double> rewards_filtered;

  for (int i = 0; i < (int) buffer.size(); ++i) {
    if (buffer[i]->is_greedy) {
      rewards_filtered.push_back(rewards[i]);
      indices.push_back(i);
    }
  }

  bool no_greedy_found = indices.size() == 0;

  if (no_greedy_found)
    rewards_filtered = rewards;

  int idx = std::distance(
      rewards_filtered.begin(),
      std::max_element(rewards_filtered.begin(), rewards_filtered.end())
  );

  if (no_greedy_found)
    return buffer[idx];

  return buffer[indices[idx]];
}
