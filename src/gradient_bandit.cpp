#include <random>
#include "gradient_bandit.hpp"


SingleGradientBandit::SingleGradientBandit(json params) {
  std::random_device dev;
  generator = std::mt19937(dev());

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

GradientBanditSearch::GradientBanditSearch(EnvWrapper orig_env, A2CLearner a2c_agent, json params, Registry *registry, bool greedy_bandit)
  : registry(registry), greedy_bandit(greedy_bandit)
{
  std::random_device dev;
  generator = std::mt19937(dev());

  n_actions = params["n_actions"];
  n_iter = params["simulations"];
  int horizon = params["horizon"];
  horizon = std::min(horizon, orig_env.env->max_steps);
  double alpha = params["dirichlet_alpha"];
  double frac = params["dirichlet_frac"];
  bool do_init_random = params["grad_bandit_init_random"];
  reward_power = params["grad_bandit_reward_power"];

  EnvWrapper env_ = *orig_env.clone();

  // TODO: Rethink the following.
  // Clone env because it might be used by other parallel actors.
  env = *orig_env.clone();

  std::vector<float> state = env_.reset();
  int i = 0;
  for (; i < horizon; ++i) {
    // Evaluate current state and predict action probabilities.
    // These are used for initializing the bandit in the next step.
    torch::Tensor action_probs;
    std::tie(action_probs, std::ignore) = a2c_agent.predict_policy({state});
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
    auto bandit = SingleGradientBandit(params);
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
    auto bandit = SingleGradientBandit(params);
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

    // TODO Make configurable. (will add soon)
    float tau = 1.0;
    if (greedy_bandit) {
      tau = 1. / 110;
      if (k > 500)
        tau = 1. / 90;
      if (k > 1000)
        tau = 1. / 80;
    }

    // It could be that horizon is set higher than the maximum horizon of the environment.
    // So let's only loop until the size of bandits.
    int j = i;
    for (; j < (int) bandits.size(); ++j) {
      std::vector<double> action_probs;
      int action;
      std::tie(action_probs, action) = bandits[j].policy(tau);

      actions.push_back(action);
      actions_probs_arr.push_back(action_probs);

      std::vector<float> obs;
      double reward;
      bool done;
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

    Game game_ = history;
    game_.states.insert(game_.states.end(), states.begin(), states.end());
    game_.rewards.insert(game_.rewards.end(), rewards.begin(), rewards.end());
    game_.mcts_actions.insert(game_.mcts_actions.end(), actions_probs_arr.begin(), actions_probs_arr.end());
    double total_reward = std::accumulate(game_.rewards.begin(), game_.rewards.end(), 0.0);
    registry->save_if_best(game_, total_reward);

    // TODO Clean code.
    // Update all bandits from this index to the end.
    // For this, we need cumulative rewards.
    std::vector<double> cumulative_rewards;
    double curr_sum = 0;
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

    // Update all bandits from 0 to i.
    total_reward = std::accumulate(rewards.begin(), rewards.end(), 0.0);

    cumulative_rewards = std::vector<double>();
    curr_sum = 0;
    for (std::vector<double>::reverse_iterator iter = game_.rewards.rbegin(); iter != game_.rewards.rend(); ++iter) {
      curr_sum += *iter;
      cumulative_rewards.push_back(curr_sum);
    }
    std::reverse(cumulative_rewards.begin(), cumulative_rewards.end());

    for (int m = 0; m < i; ++m) {
      auto mcts_action = game_.mcts_actions[m];
      auto max_el = std::max_element(mcts_action.begin(), mcts_action.end());
      int argmax_action = std::distance(mcts_action.begin(), max_el);

      bandits[m].update(mcts_action, argmax_action, cumulative_rewards[m]);
    }
  }

  history.states.push_back(state);
  // The following is done in alphazero..
  // history.rewards.push_back(XXXXXXXXXX);

  auto ret = bandits[i].softmax(1);
  history.mcts_actions.push_back(ret);
  return ret;
}
