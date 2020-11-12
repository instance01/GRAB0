#ifndef GAUSSIAN_GRADIENT_BANDIT_HEADER
#define GAUSSIAN_GRADIENT_BANDIT_HEADER
#include <vector>
#include <tuple>
#include <random>
#include "env_wrapper.hpp"
#include "a2c.hpp"
#include "gaussian_a2c.hpp"
#include "cfg.hpp"
#include "bandit.hpp"
#include "registry.hpp"


class SingleGaussianGradientBandit {
  public:
    // TODO: This is just for debugging. Get rid of it.
    float biggest_advantage = -10000;

    bool initialized = false;

    int n_actions;
    int n_iter;

    int total_visits = 1;
    double mean_reward = 0;
    std::vector<double> gaussian_params;
    double alpha = 0.1;
    std::mt19937 generator;

    std::discrete_distribution<int> distribution;

    SingleGaussianGradientBandit(json params, std::mt19937 &generator);

    std::vector<double> sample_from_policy();
    std::pair<std::vector<double>, std::vector<double>> policy();
    void update(
        std::vector<double> prob_actions,
        std::vector<double> action,
        double reward
    );
};


class GaussianGradientBanditSearch : public Bandit {
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

    std::vector<SingleGaussianGradientBandit> bandits;
    EnvWrapper env;

    A2CLearner* a2c_agent;

    json params;
    bool do_print;

    GaussianGradientBanditSearch(
        EnvWrapper env,
        A2CLearner* a2c_agent,
        json params,
        Registry *registry,
        std::mt19937 &generator,
        bool do_print,
        bool greedy_bandit=false
    );
    ~GaussianGradientBanditSearch() {};

    void reset_policy_cache() {};
    std::vector<double> policy(
        int i,
        EnvWrapper &env,
        const std::vector<float> &obs,
        bool ret_node=false
    );
};
#endif
