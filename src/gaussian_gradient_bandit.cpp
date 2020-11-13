#include <random>
#include "gaussian_gradient_bandit.hpp"
#include "gaussian_util.hpp"


SingleGaussianGradientBandit::SingleGaussianGradientBandit(json params, std::mt19937 &generator) : generator(generator) {
  n_actions = params["n_actions"];
  n_iter = params["simulations"];
  alpha = params["grad_bandit_alpha"];

  total_visits = 0;
  mean_reward = 0;

  for (int i = 0; i < n_actions; ++i) {
    gaussian_params.push_back(0.); // mu
    gaussian_params.push_back(1.); // sigma
  }
}

std::vector<double>
SingleGaussianGradientBandit::sample_from_policy() {
  std::vector<double> actions(n_actions);
  for (int i = 0; i < n_actions; ++i) {
    actions[i] = sample(gaussian_params[i*2], gaussian_params[i*2+1], generator);
  }
  return actions;
}

std::pair<std::vector<double>, std::vector<double>>
SingleGaussianGradientBandit::policy() {
  return {gaussian_params, sample_from_policy()};
}

void
SingleGaussianGradientBandit::update(std::vector<double> old_gaussian_params, std::vector<double> action, double reward) {
  total_visits += 1;
  mean_reward += (reward - mean_reward) / total_visits;

  double advantage = reward - mean_reward;

  if (advantage > biggest_advantage)
    biggest_advantage = advantage;

  for (int i = 0; i < n_actions; ++i) {
    double mu = old_gaussian_params[i*2];
    double sigma = old_gaussian_params[i*2+1];
    gaussian_params[i*2] += alpha * (action[i] - mu) / std::pow(sigma, 2.) * advantage;
    gaussian_params[i*2+1] += alpha * (std::pow((action[i] - mu), 2.) / std::pow(sigma, 3.) - 1. / sigma) * advantage;
    // Instead of advantage we could use reward.
    //gaussian_params[i*2] += alpha * (action[i] - mu) / std::pow(sigma, 2.) * reward;
    //gaussian_params[i*2+1] += alpha * (std::pow((action[i] - mu), 2.) / std::pow(sigma, 3.) - 1. / sigma) * reward;
  }
}

GaussianGradientBanditSearch::GaussianGradientBanditSearch(
    EnvWrapper orig_env,
    A2CLearner* a2c_agent,
    json params,
    Registry *registry,
    std::mt19937 &generator,
    bool do_print,
    bool greedy_bandit)
  : a2c_agent(a2c_agent), params(params), registry(registry), generator(generator), do_print(do_print), greedy_bandit(greedy_bandit)
{
  n_actions = params["n_actions"];
  n_iter = params["simulations"];
  int horizon = params["horizon"];
  horizon = std::min(horizon, orig_env.env->max_steps);
  double alpha = params["dirichlet_alpha"];
  double frac = params["dirichlet_frac"];
  bool do_init_random = params["grad_bandit_init_random"];
  reward_power = params["grad_bandit_reward_power"];
  tau_schedule_k = params["grad_bandit_tau_schedule_k"].get<std::vector<int>>();
  tau_schedule_tau = params["grad_bandit_tau_schedule_tau"].get<std::vector<int>>();

  EnvWrapper env_ = *orig_env.clone();

  // TODO: Rethink the following.
  // Clone env because it might be used by other parallel actors.
  env = *orig_env.clone();

  std::vector<float> state = env_.reset(generator);
  int i = 0;
  for (; i < horizon; ++i) {
    // Evaluate current state and predict action probabilities.
    // These are used for initializing the bandit in the next step.
    torch::Tensor gaussian_params_;
    std::tie(gaussian_params_, std::ignore) = a2c_agent->predict_policy({state});
    float* gaussian_params_arr = gaussian_params_.data_ptr<float>();
    auto vec = std::vector<float>(gaussian_params_arr, gaussian_params_arr + n_actions * 2);

    // TODO ?? Dirichlet in gaussian policy?
    // // Add Dirichlet noise.
    // std::gamma_distribution<double> distribution(alpha, 1.);
    // for (int j = 0; j < n_actions; ++j) {
    //   double noise = distribution(generator);
    //   action_probs[0][j] = action_probs[0][j] * (1 - frac) + noise * frac;
    // }

    // Create the bandit.
    auto bandit = SingleGaussianGradientBandit(params, generator);
    bandit.gaussian_params = std::vector<double>(vec.begin(), vec.end());
    std::vector<double> action = bandit.sample_from_policy();
    bandits.push_back(bandit);

    // Continue evaluating.
    double reward;
    bool done;
    std::tie(state, reward, done) = env_.step(action);

    if (done)
      break;
  }

  // In case we evaluated a very good path, add missing bandits (optionally with random
  // initialization).
  std::uniform_real_distribution<double> distribution_(0.0, 1.0);
  for (int j = 0; j < (horizon - i); ++j) {
    std::vector<double> vec;
    for (int i = 0; i < n_actions; ++i) {
      vec.push_back(0.);
      vec.push_back(1.);
    }
    // TODO !! -> no such thing for now in gaussian version.
    // if (do_init_random) {
    //   std::generate(
    //       vec.begin(),
    //       vec.end(),
    //       [distribution_, this] () mutable { return distribution_(this->generator); }
    //   );
    // }

    // Create the bandit.
    auto bandit = SingleGaussianGradientBandit(params, this->generator);
    bandit.gaussian_params = vec;
    bandits.push_back(bandit);
  }
}

std::vector<double>
GaussianGradientBanditSearch::policy(int i, EnvWrapper &orig_env, const std::vector<float> &state, bool ret_node) {
  for (int k = 0; k < n_iter; ++k) {
    std::vector<double> rewards;
    std::vector<std::vector<double>> actions_history;
    std::vector<std::vector<double>> params_history;
    std::vector<std::vector<float>> states;

    EnvWrapper env = *orig_env.clone();

    float tau = 1.;
    if (greedy_bandit) {
      for (int i = 0; i < (int) tau_schedule_k.size(); ++i) {
        if (k >= tau_schedule_k[i])
          tau = 1. / tau_schedule_tau[i];
      }
    }

    double alpha = params["dirichlet_alpha"];
    double frac = params["dirichlet_frac"];

    std::vector<float> obs = state;
    bool done;
    bool won = false;  // TODO REMOVE

    // It could be that horizon is set higher than the maximum horizon of the environment.
    // So let's only loop until the size of bandits.
    int j = i;
    for (; j < (int) bandits.size(); ++j) {
      if (!bandits[j].initialized) {
        torch::Tensor gaussian_params_;
        std::tie(gaussian_params_, std::ignore) = a2c_agent->predict_policy({obs});
        float* gaussian_params_arr = gaussian_params_.data_ptr<float>();
        auto vec = std::vector<float>(gaussian_params_arr, gaussian_params_arr + n_actions * 2);

        // TODO No Dirichlet in gaussian version for now.
        // // Add Dirichlet noise.
        // std::gamma_distribution<double> distribution(alpha, 1.);
        // for (int q = 0; q < n_actions; ++q) {
        //   double noise = distribution(generator);
        //   action_probs[0][q] = action_probs[0][q] * (1 - frac) + noise * frac;
        // }

        // Create the bandit.
        //std::cout << gaussian_params_ << std::endl;
        bandits[j].gaussian_params = std::vector<double>(vec.begin(), vec.end());
        bandits[j].initialized = true;
      }

      std::vector<double> gaussian_params;
      std::vector<double> action;
      std::tie(gaussian_params, action) = bandits[j].policy();

      // TODO. Why keep those histories?
      actions_history.push_back(action);
      params_history.push_back(gaussian_params);

      double reward;
      std::tie(obs, reward, done) = env.step(action);

      states.push_back(obs);

      reward = std::pow(reward, reward_power);
      rewards.push_back(reward);

      if (done) {
        // Since we break, the last ++j of the loop is not executed.
        // To keep things consistent later on, let's do it manually.
        if (reward == 100.0) {
          won = true;
        }
        j += 1;
        break;
      }
    }

    double val = 0;
    if (!done) {
      torch::Tensor value;
      std::tie(std::ignore, value) = a2c_agent->predict_policy({obs});
      val = value.item<double>();
    }

    // TODO Clean code.
    // Update all bandits from this index to the end.
    // For this, we need cumulative rewards.
    std::vector<double> cumulative_rewards;
    double curr_sum = val;
    for (std::vector<double>::reverse_iterator iter = rewards.rbegin(); iter != rewards.rend(); ++iter) {
      curr_sum += *iter;
      cumulative_rewards.push_back(curr_sum);
    }
    std::reverse(cumulative_rewards.begin(), cumulative_rewards.end());

    // This had an off by one mistake. Refer to j += 1 a few lines above.
    int size = std::min((int) bandits.size() - 1, j - i);
    for (int m = 0; m < size; ++m) {
      bandits[m + i].update(params_history[m], actions_history[m], cumulative_rewards[m]);
    }

    if (won) {
      // TODO
      //std::cout << "after update: " << bandits[i].gaussian_params << std::endl;
    }
  }

  std::vector<double> gaussian_params;
  std::tie(gaussian_params, std::ignore) = bandits[i].policy();

  // TODO Biggest advantage
  //std::cout << "BIGGEST A " << bandits[i].biggest_advantage << " |mr:" << bandits[i].mean_reward << std::endl;

  // TODO ?
  // history.mcts_actions.push_back(ret);
  return gaussian_params;
}
