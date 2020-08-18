#ifndef ALPHAZERO_HEADER
#define ALPHAZERO_HEADER
#include "env_wrapper.hpp"
#include "cfg.hpp"
#include "a2c.hpp"
#include "mcts.hpp"
#include "registry.hpp"

std::pair<int, double> evaluate(EnvWrapper env, json params, A2CLearner a2c_agent, Registry *registry);
std::shared_ptr<Game> run_actor(EnvWrapper orig_env, json params, A2CLearner a2c_agent, int n_episode, Registry *registry, bool greedy_bandit=false);
#endif
