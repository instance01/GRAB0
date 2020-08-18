#ifndef GRADIENT_BANDIT_HEADER
#define GRADIENT_BANDIT_HEADER
#include <vector>
#include <tuple>
#include <random>
#include "env_wrapper.hpp"
#include "a2c.hpp"
#include "cfg.hpp"
#include "bandit.hpp"
#include "registry.hpp"


class SingleGradientBandit {
  public:
    int n_actions;
    int n_iter;

    int total_visits = 1;
    std::vector<int> action_visits;
    double mean_reward = 0;
    std::vector<double> action_rewards;
    std::vector<double> H;
    double alpha = 0.1;
    std::mt19937 generator;

    std::vector<double> prob_action;
    std::discrete_distribution<int> distribution;

    SingleGradientBandit(json params);

    std::vector<double> softmax(float tau);
    int sample(std::vector<double> prob_actions);
    std::pair<std::vector<double>, int> policy(float tau);
    void update(std::vector<double> prob_actions, int action, double reward);
};


class GradientBanditSearch : public Bandit {
  public:
    int n_actions;
    int n_iter;
    int horizon;
    std::mt19937 generator;
    int reward_power;
    Registry *registry;

    bool greedy_bandit;
    std::vector<int> tau_schedule_k;
    std::vector<int> tau_schedule_tau;

    std::vector<SingleGradientBandit> bandits;
    EnvWrapper env;

    GradientBanditSearch(EnvWrapper env, A2CLearner a2c_agent, json params, Registry *registry, bool greedy_bandit=false);
    ~GradientBanditSearch() {};

    void reset_policy_cache() {};
    std::vector<double> policy(int i, EnvWrapper env, std::vector<float> obs, bool ret_node=false);
};
#endif
