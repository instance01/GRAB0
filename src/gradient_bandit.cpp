#include <random>
#include "gradient_bandit.hpp"


SingleGradientBandit::SingleGradientBandit(json params, std::mt19937 &generator) : generator(generator) {
  n_actions = params["n_actions"];
  n_iter = params["simulations"];

  total_visits = 0;
  action_visits = std::vector<int>(n_actions, 0);
  action_rewards = std::vector<double>(n_actions, 0.0);
  mean_reward = 0;
  prob_action = std::vector<double>(n_actions, 0.0);

  // H is initialized from a2c.
  H = std::vector<double>(n_actions, 0);
  alpha = params["grad_bandit_alpha"];
}

std::vector<double>
SingleGradientBandit::softmax(float tau) {
  double maxH = *std::max_element(H.begin(), H.end());

  double sum = 0;
  for (int i = 0; i < n_actions; ++i) {
    sum += std::exp((H[i] - maxH) / tau);
  }

  for (int i = 0; i < n_actions; ++i) {
    prob_action[i] = std::exp((H[i] - maxH) / tau) / sum;
  }

  return prob_action;
}

int
SingleGradientBandit::sample(std::vector<double> action_probs) {
  distribution.param({action_probs.begin(), action_probs.end()});
  return distribution(generator);
}

std::pair<std::vector<double>, int>
SingleGradientBandit::policy(float tau) {
  auto action_probs = softmax(tau);
  return {action_probs, sample(action_probs)};
}

void
SingleGradientBandit::update(std::vector<double> action_probs, int action, double reward) {
  total_visits += 1;
  action_visits[action] += 1;

  mean_reward += (reward - mean_reward) / total_visits;

  action_rewards[action] += (reward - action_rewards[action]) / action_visits[action];

  H[action] += alpha * (reward - mean_reward) * (1 - action_probs[action]);
  for (int i = 0; i < n_actions; ++i) {
    if (i != action)
      H[i] -= alpha * (reward - mean_reward) * (action_probs[i]);
  }
}

GradientBanditSearch::GradientBanditSearch(
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
    torch::Tensor action_probs;
    std::tie(action_probs, std::ignore) = a2c_agent->predict_policy({state});
    int action = action_probs.argmax().item<int>();

    // TODO Interesting detail: We add noise AFTER taking the argmax.. Not sure which way is better.

    // Add Dirichlet noise.
    std::gamma_distribution<double> distribution(alpha, 1.);
    for (int j = 0; j < n_actions; ++j) {
      double noise = distribution(generator);
      action_probs[0][j] = action_probs[0][j] * (1 - frac) + noise * frac;
    }

    float* action_probs_arr = action_probs.data_ptr<float>();

    // Create the bandit.
    auto bandit = SingleGradientBandit(params, generator);
    auto vec = std::vector<float>(action_probs_arr, action_probs_arr + n_actions);
    bandit.H = std::vector<double>(vec.begin(), vec.end());
    // Initialize with log.
    for (int k = 0; k < (int) vec.size(); ++k) {
      bandit.H[k] = std::log(vec[k]);
    }
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
    std::vector<double> vec{0.33, 0.33, 0.33};
    if (do_init_random) {
      std::generate(
          vec.begin(),
          vec.end(),
          [distribution_, this] () mutable { return distribution_(this->generator); }
      );
    }

    // Create the bandit.
    auto bandit = SingleGradientBandit(params, this->generator);
    bandit.H = vec;
    bandits.push_back(bandit);
  }
}

std::vector<double>
GradientBanditSearch::policy(int i, EnvWrapper orig_env, std::vector<float> state, bool ret_node) {
  // TODO Actions will be continuous at some point. So not int, but double.
  for (int k = 0; k < n_iter; ++k) {
    std::vector<double> rewards;
    std::vector<int> actions;
    std::vector<std::vector<double>> actions_probs_arr;
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

    // It could be that horizon is set higher than the maximum horizon of the environment.
    // So let's only loop until the size of bandits.
    int j = i;
    for (; j < (int) bandits.size(); ++j) {
      if (!bandits[j].initialized) {
        torch::Tensor action_probs;
        std::tie(action_probs, std::ignore) = a2c_agent->predict_policy({obs});

        // Add Dirichlet noise.
        std::gamma_distribution<double> distribution(alpha, 1.);
        for (int q = 0; q < n_actions; ++q) {
          double noise = distribution(generator);
          action_probs[0][q] = action_probs[0][q] * (1 - frac) + noise * frac;
        }

        float* action_probs_arr = action_probs.data_ptr<float>();

        // Create the bandit.
        auto vec = std::vector<float>(action_probs_arr, action_probs_arr + n_actions);
        bandits[j].H = std::vector<double>(vec.begin(), vec.end());
        // Initialize with log.
        for (int r = 0; r < (int) vec.size(); ++r) {
          bandits[j].H[r] = std::log(vec[r]);
          if (do_print)
            std::cout << bandits[j].H[r] << " ";
        }
        if (do_print)
          std::cout << " | ";

        bandits[j].initialized = true;
      }

      std::vector<double> action_probs;
      int action;
      std::tie(action_probs, action) = bandits[j].policy(tau);

      actions.push_back(action);
      actions_probs_arr.push_back(action_probs);

      double reward;
      std::tie(obs, reward, done) = env.step(action);

      states.push_back(obs);

      reward = std::pow(reward, reward_power);
      rewards.push_back(reward);

      if (done) {
        // Since we break, the last ++j of the loop is not executed.
        // To keep things consistent later on, let's do it manually.
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
      bandits[m + i].update(actions_probs_arr[m], actions[m], cumulative_rewards[m]);
    }
  }

  if (do_print) {
    std::cout << std::endl;
    std::cout << std::endl;
    for (auto bandit : bandits) {
      std::cout << bandit.H << " | ";
    }
    std::cout << std::endl;
  }

  auto ret = bandits[i].softmax(1);
  history.mcts_actions.push_back(ret);
  return ret;
}
