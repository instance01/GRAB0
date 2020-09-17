#ifndef ALPHAZERO_HEADER
#define ALPHAZERO_HEADER
#include <tuple>
#include "env_wrapper.hpp"
#include "cfg.hpp"
#include "a2c.hpp"
#include "mcts.hpp"
#include "registry.hpp"
#include "replay_buffer.hpp"

std::tuple<int, double, double> evaluate(EnvWrapper env, json &params, A2CLearner a2c_agent, Registry *registry);
std::shared_ptr<Game> run_actor(int idx, std::random_device &rd, EnvWrapper orig_env, json params, A2CLearner a2c_agent, int n_episode, Registry *registry, ReplayBuffer *replay_buffer, bool greedy_bandit=false);
#endif
