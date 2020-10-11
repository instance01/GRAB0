#include <numeric>
#include <random>
#include <iostream>

#include "game.hpp"
#include "replay_buffer.hpp"


ReplayBuffer::ReplayBuffer(int window_size, float experimental_top_cutoff,bool prioritized_sampling)
  : window_size(window_size), experimental_top_cutoff(experimental_top_cutoff), prioritized_sampling(prioritized_sampling)
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

template <typename T, typename Compare>
std::vector<std::size_t> sort_permutation(
    const std::vector<T>& vec,
    Compare compare)
{
    std::vector<std::size_t> p(vec.size());
    std::iota(p.begin(), p.end(), 0);
    std::sort(p.begin(), p.end(),
        [&](std::size_t i, std::size_t j){ return compare(vec[i], vec[j]); });
    return p;
}

template <typename T>
std::vector<T> apply_permutation(
    const std::vector<T>& vec,
    const std::vector<std::size_t>& p)
{
    std::vector<T> sorted_vec(vec.size());
    std::transform(p.begin(), p.end(), sorted_vec.begin(),
        [&](std::size_t i){ return vec[i]; });
    return sorted_vec;
}

std::vector<std::shared_ptr<Game>>
ReplayBuffer::get_top() {
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
    return {};

  // Sort rewards descending and apply permutation to indices.
  auto p = sort_permutation(rewards_filtered, [](double a, double b){ return a > b; });
  indices = apply_permutation(indices, p);

  // Extract top greedy games
  std::vector<std::shared_ptr<Game>> ret;
  for (int i = 0; i < indices.size(); ++i) {
    if (rewards[indices[i]] >= experimental_top_cutoff)
        ret.push_back(buffer[indices[i]]);
  }
  return ret;
}

std::vector<std::shared_ptr<Game>>
ReplayBuffer::get_top_p(float percentile) {
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
    return {};

  // Sort rewards descending and apply permutation to indices.
  auto p = sort_permutation(rewards_filtered, [](double a, double b){ return a > b; });
  indices = apply_permutation(indices, p);

  int idx = (int) (indices.size() * percentile);

  // Extract top greedy games
  std::vector<std::shared_ptr<Game>> ret;
  for (int i = 0; i < idx; ++i) {
    ret.push_back(buffer[indices[i]]);
  }
  return ret;
}
