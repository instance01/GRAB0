#include <iostream>
#include <fstream>
#include "cfg.hpp"

using json = nlohmann::json;


json get_default(std::string base) {
  json params = {
    {"1", {
      {"n_actions", 3},
      {"n_input_features", 3},
      {"n_runs", 10},

      // MCTS
      {"gamma", .99},
      {"c", 1.},  // .001  # Using puct now.
      {"simulations", 50},
      {"horizon", 200},
      {"dirichlet_alpha", .3},
      {"dirichlet_frac", .25},
      // TODO I think this should be based on simulations.
      // base was 19652. But with just 100 simulations (and thus visits only
      // getting to 100 at max) visits don't matter.. At base=50}, hell yes !
      {"pb_c_base", 50},
      {"pb_c_init", 1.25},
      {"tough_ce", false},

      // A2C
      {"alpha", .01},
      {"net_architecture", {64, 64}},
      {"schedule_alpha", false},
      {"scheduler_class", "exp"},  // exp, step, reduce_eval
      {"scheduler_factor", .995},  // For: exp, step, reduce_eval
      {"scheduler_min_lr", .000001},  // For: exp, step, reduce_eval
      {"scheduler_steps", {100, 200}},  // For: step
      {"scheduler_min_good_eval", -120.0},  // For: reduce_eval
      {"scheduler_min_n_good_evals", 15},  // For: reduce_eval
      {"scheduler_consecutive", true},  // For: reduce_eval
      {"use_weight_decay", false},
      {"weight_decay", .00001},
      {"optimizer_class", "adam"},  // adam, sgd
      {"sgd_momentum", .9},

      // AlphaZero
      {"memory_capacity", 1000},
      {"prioritized_sampling", true},
      {"episodes", 100},
      {"n_procs", 4},
      {"n_actors", 20},  // 5000
      {"train_steps", 2000},  // 700000
      {"desired_eval_len", 8},
      {"n_desired_eval_len", 10},
      {"bandit_type", "mcts"},  // mcts, grad
      {"grad_bandit_alpha", 0.01},
      {"use_eps_greedy_learning", true},
      {"eps_greedy_epsilon_decay_factor_train", 0.995},
      {"eps_greedy_epsilon_decay_factor_actor", 0.995},
      {"grad_bandit_init_random", true},
      {"grad_bandit_reward_power", 1},
      {"grad_bandit_tau_schedule_tau", {20, 90, 80}},
      {"grad_bandit_tau_schedule_k", {0, 500, 1000}},
      {"do_warmup", false},
      {"n_warmup", 100},
      {"follow_a2c", false},
      {"eval_break_on_good_avg_reward", false},
      {"eval_min_avg_reward", -110},

      // Other
      {"reward_exponent", 1},

      // TODO unused right now
      {"epsilon", .1},
      {"epsilon_linear_decay", 1. / 10000},  // 10000 is memory_capacity
      {"epsilon_min", 0.01}
    }},
    {"81", {
      {"n_actions", 3},
      {"n_input_features", 2},
      {"n_runs", 10},

      // MCTS
      {"gamma", .99},
      {"simulations", 50},
      {"horizon", 400},
      {"dirichlet_alpha", .3},
      {"dirichlet_frac", .5},
      {"pb_c_base", 19000},
      {"pb_c_init", .3},
      {"tough_ce", true},

      // A2C
      {"alpha", .001},
      {"net_architecture", {64, 64}},
      {"schedule_alpha", true},
      {"scheduler_class", "exp"},  // exp, step, reduce_eval
      {"scheduler_factor", .995},  // For: exp, step, reduce_eval
      {"scheduler_min_lr", .000001},  // For: exp, step, reduce_eval
      {"scheduler_steps", {100, 200}},  // For: step
      {"scheduler_min_good_eval", -120.0},  // For: reduce_eval
      {"scheduler_min_n_good_evals", 20},  // For: reduce_eval
      {"scheduler_consecutive", true},  // For: reduce_eval
      {"use_weight_decay", false},
      {"weight_decay", .00001},
      {"optimizer_class", "adam"},  // adam, sgd
      {"sgd_momentum", .9},

      // AlphaZero
      {"memory_capacity", 1000},
      {"prioritized_sampling", false},
      {"episodes", 500},
      {"n_procs", 8},
      {"n_actors", 40},  // 5000
      {"train_steps", 4000},  // 700000
      {"desired_eval_len", 8},
      {"n_desired_eval_len", 10},
      {"bandit_type", "mcts"},  // mcts, grad
      {"grad_bandit_alpha", 0.01},
      {"use_eps_greedy_learning", true},
      {"eps_greedy_epsilon_decay_factor_train", 0.995},
      {"eps_greedy_epsilon_decay_factor_actor", 0.995},
      {"grad_bandit_init_random", true},
      {"grad_bandit_reward_power", 1},
      {"grad_bandit_tau_schedule_tau", {20, 90, 80}},
      {"grad_bandit_tau_schedule_k", {0, 500, 1000}},
      {"do_warmup", false},
      {"n_warmup", 100},
      {"follow_a2c", false},
      {"eval_break_on_good_avg_reward", false},
      {"eval_min_avg_reward", -110},

      // Other
      {"reward_exponent", 1},
    }},
    {"120", {
      {"n_actions", 3},
      {"n_input_features", 2},
      {"n_runs", 10},

      // MCTS
      {"gamma", .99},
      {"simulations", 50},
      {"horizon", 400},
      {"dirichlet_alpha", .3},
      {"dirichlet_frac", .5},
      {"pb_c_base", 1000},
      {"pb_c_init", .1},
      {"tough_ce", true},

      // A2C
      {"alpha", .001},
      {"net_architecture", {64, 64}},
      {"schedule_alpha", true},
      {"scheduler_class", "reduce_eval"},  // exp, step, reduce_eval
      {"scheduler_factor", .5},  // For: exp, step, reduce_eval
      {"scheduler_min_lr", .000001},  // For: exp, step, reduce_eval
      {"scheduler_steps", {100, 200}},  // For: step
      {"scheduler_min_good_eval", -100.0},  // For: reduce_eval
      {"scheduler_min_n_good_evals", 10},  // For: reduce_eval
      {"scheduler_consecutive", false},  // For: reduce_eval
      {"use_weight_decay", false},
      {"weight_decay", .00001},
      {"optimizer_class", "adam"},  // adam, sgd
      {"sgd_momentum", .9},

      // AlphaZero
      {"memory_capacity", 1000},
      {"prioritized_sampling", false},
      {"episodes", 1000},
      {"n_procs", 8},
      {"n_actors", 40},
      {"train_steps", 4000},
      {"desired_eval_len", 110},
      {"n_desired_eval_len", 100},
      {"bandit_type", "mcts"},  // mcts, grad
      {"grad_bandit_alpha", 0.01},
      {"use_eps_greedy_learning", true},
      {"eps_greedy_epsilon_decay_factor_train", 0.995},
      {"eps_greedy_epsilon_decay_factor_actor", 0.995},
      {"grad_bandit_init_random", true},
      {"grad_bandit_reward_power", 1},
      {"grad_bandit_tau_schedule_tau", {20, 90, 80}},
      {"grad_bandit_tau_schedule_k", {0, 500, 1000}},
      {"do_warmup", false},
      {"n_warmup", 100},
      {"follow_a2c", false},
      {"eval_break_on_good_avg_reward", false},
      {"eval_min_avg_reward", -110},

      // Other
      {"reward_exponent", 1},
    }},
    {"BT20", {
      {"n_actions", 3},
      {"n_input_features", 3},
      {"n_runs", 10},

      // MCTS
      {"gamma", .99},
      {"simulations", 2000},
      {"horizon", 400},
      {"dirichlet_alpha", .3},
      {"dirichlet_frac", .3},
      {"pb_c_base", 19000},
      {"pb_c_init", .3},
      {"tough_ce", true},

      // A2C
      {"alpha", .0005},
      {"net_architecture", {64, 64}},
      {"schedule_alpha", false},
      {"scheduler_class", "exp"},  // exp, step, reduce_eval
      {"scheduler_factor", .995},  // For: exp, step, reduce_eval
      {"scheduler_min_lr", .000001},  // For: exp, step, reduce_eval
      {"scheduler_steps", {100, 200}},  // For: step
      {"scheduler_min_good_eval", -120.0},  // For: reduce_eval
      {"scheduler_min_n_good_evals", 20}, // For: reduce_eval
      {"scheduler_consecutive", true},  // For: reduce_eval
      {"use_weight_decay", false},
      {"weight_decay", .00001},
      {"optimizer_class", "adam"},  // adam, sgd
      {"sgd_momentum", .9},

      // AlphaZero
      {"memory_capacity", 10000},
      {"prioritized_sampling", false},
      {"episodes", 200},
      {"n_procs", 8},
      {"n_actors", 20},  // 5000
      {"train_steps", 4000},  // 700000
      {"desired_eval_len", 8},
      {"n_desired_eval_len", 10},
      {"bandit_type", "grad"},  // mcts, grad
      {"grad_bandit_alpha", 0.1},
      {"use_eps_greedy_learning", true},
      {"eps_greedy_epsilon_decay_factor_train", 0.99},
      {"eps_greedy_epsilon_decay_factor_actor", 0.99},
      {"grad_bandit_init_random", true},
      {"grad_bandit_reward_power", 1},
      {"grad_bandit_tau_schedule_tau", {20, 90, 80}},
      {"grad_bandit_tau_schedule_k", {0, 500, 1000}},
      {"do_warmup", false},
      {"n_warmup", 100},
      {"follow_a2c", false},
      {"eval_break_on_good_avg_reward", false},
      {"eval_min_avg_reward", -110},

      // Other
      {"reward_exponent", 1},
    }}
  };

  return params[base];
}


json load_cfg(std::string param_num) {
  std::ifstream ifs("simulations.json");
  json jsondata = json::parse(ifs);
  std::string base = jsondata[param_num]["base"];
  json ret = get_default(base);
  ret.update(jsondata[param_num]);
  std::cout << ret.dump(2) << std::endl;
  return ret;
}
