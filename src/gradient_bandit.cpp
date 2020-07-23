#include <random>
#include "gradient_bandit.hpp"


SingleGradientBandit::SingleGradientBandit(json params) {
  n_actions = params["n_actions"];
  n_iter = params["simulations"];

  total_visits = 1;
  action_visits = std::vector<int>(n_actions, 0);
  action_rewards = std::vector<double>(n_actions, 0);
  mean_reward = 0;

  // H is initialized from a2c.
  H = std::vector<double>(n_actions, 0);
  alpha = params["grad_bandit_alpha"];
}

std::vector<double>
SingleGradientBandit::softmax() {
  std::vector<double> prob_action(n_actions, 0);

  double maxH = *std::max_element(H.begin(), H.end());

  double sum = 0;
  for (int i = 0; i < n_actions; ++i) {
    sum += std::exp(H[i] - maxH);
  }

  for (int i = 0; i < n_actions; ++i) {
    prob_action[i] = std::exp(H[i] - maxH) / sum;
  }

  return prob_action;
}

int
SingleGradientBandit::sample(std::vector<double> action_probs) {
  std::discrete_distribution<int> distribution(action_probs.begin(), action_probs.end());
  return distribution(generator);
}

std::pair<std::vector<double>, int>
SingleGradientBandit::policy() {
  auto action_probs = softmax();
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

GradientBanditSearch::GradientBanditSearch(EnvWrapper orig_env, A2CLearner a2c_agent, json params) {
  n_actions = params["n_actions"];
  n_iter = params["simulations"];
  horizon = params["horizon"];
  double alpha = params["dirichlet_alpha"];
  double frac = params["dirichlet_frac"];

  EnvWrapper env_ = *orig_env.clone();

  // TODO: Rethink the following.
  // Clone env because it might be used by other parallel actors.
  env = *orig_env.clone();

  std::vector<float> state = env_.reset();
  for (int i = 0; i < horizon; ++i) {
    // Evaluate current state and predict action probabilities.
    // These are used for initializing the bandit in the next step.
    torch::Tensor action_probs;
    std::tie(action_probs, std::ignore) = a2c_agent.predict_policy({state});
    int action = action_probs.argmax().item<int>();

    // Add Dirichlet noise.
    std::gamma_distribution<double> distribution(alpha, 1.);
    for (int j = 0; j < n_actions; ++j) {
      double noise = distribution(generator);
      action_probs[0][j] = action_probs[0][j] * (1 - frac) + noise * frac;
    }

    float* action_probs_arr = action_probs.data_ptr<float>();

    // Create the bandit.
    auto bandit = SingleGradientBandit(params);
    auto vec = std::vector<float>(action_probs_arr, action_probs_arr + n_actions);
    bandit.H = std::vector<double>(vec.begin(), vec.end());
    bandits.push_back(bandit);

    // Continue evaluating.
    double reward;
    bool done;
    std::tie(state, reward, done) = env_.step(action);

    if (done)
      break;
  }
}

std::vector<double>
GradientBanditSearch::policy(int i, EnvWrapper orig_env, std::vector<float> obs, bool ret_node) {
  // TODO Actions will be continuous at some point. So not int, but double.
  for (int k = 0; k < n_iter; ++k) {
    std::vector<double> rewards;
    std::vector<int> actions;
    std::vector<std::vector<double>> actions_probs_arr;

    EnvWrapper env = *orig_env.clone();
    env.reset();

    // TODO Hmm.. It could be that horizon is set HIGHER than the maximum horizon of the
    // environment. So, let's only loop until the size of bandits.
    int j = i;
    for (; j < bandits.size(); ++j) {
      std::vector<double> action_probs;
      int action;
      std::tie(action_probs, action) = bandits[j].policy();

      actions.push_back(action);
      actions_probs_arr.push_back(action_probs);

      double reward;
      bool done;
      std::tie(std::ignore, reward, done) = env.step(action);

      rewards.push_back(reward);

      if (done)
        break;
    }

    std::vector<double> cumulative_rewards(rewards.size(), 0.);
    std::partial_sum(rewards.begin(), rewards.end(), cumulative_rewards.begin(), std::plus<double>());

    for (int m = 0; m < j - i; ++m) {
      bandits[m].update(actions_probs_arr[m], actions[m], rewards[m]);
    }
  }

  return bandits[i].softmax();
}
