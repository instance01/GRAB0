#include "gradient_bandit_tree.hpp"
#include <cmath>
#include <algorithm>
#include <random>
#include "util.hpp"


GradientBanditTreeSearch::GradientBanditTreeSearch(EnvWrapper env, A2CLearner a2c_agent, json params, std::mt19937 &generator) : params(params), a2c_agent(a2c_agent), generator(generator) {
  // std::random_device rd;
  // std::mt19937 generator(rd());
  auto obs = env.reset(generator);
  this->env = std::move(env.clone());

  this->root_node = std::make_shared<GradientBanditNode>();
  root_node->env = std::move(env.clone());
  root_node->state = obs;
}

void
GradientBanditTreeSearch::_gen_children_nodes(std::shared_ptr<GradientBanditNode> parent_node) {
  int n_actions = params["n_actions"];
  for (int i = 0; i < n_actions; ++i) {
    auto env = this->env->clone();

    std::vector<float> obs;
    double reward;
    bool done;
    std::tie(obs, reward, done) = env->step(i);

    std::shared_ptr<GradientBanditNode> node = std::make_unique<GradientBanditNode>();
    node->state = obs;
    node->env = std::move(env);
    node->action = i;
    node->reward = reward;
    node->is_terminal = done;
    node->parent = std::weak_ptr(parent_node);
    node->bandit = std::make_shared<SingleGradientBandit>(params, generator);
    init_single_bandit(node->bandit, obs);

    node->torch_state = vec_1d_as_tensor(node->state, torch::kFloat32);

    parent_node->children.push_back(node);
  }
}

std::shared_ptr<GradientBanditNode>
GradientBanditTreeSearch::_expand(std::shared_ptr<GradientBanditNode> parent_node) {
  if (parent_node->children.empty())
    _gen_children_nodes(parent_node);

  for (auto child : parent_node->children) {
    if (child->visits == 0) {
      return child;
    }
  }

  parent_node->is_fully_expanded = true;
  return nullptr;
}

std::shared_ptr<GradientBanditNode>
GradientBanditTreeSearch::_get_best_node(std::shared_ptr<GradientBanditNode> parent_node) {
  // TODO tau!!
  float tau = 1.;
  std::vector<double> action_probs;
  int action;
  std::tie(action_probs, action) = parent_node->bandit->policy(tau);
  return parent_node->children[action];
}

std::shared_ptr<GradientBanditNode>
GradientBanditTreeSearch::select_expand() {
  std::shared_ptr<GradientBanditNode> curr_node = root_node;

  while (true) {
    if (curr_node->is_terminal) {
      break;
    }

    if (curr_node->is_fully_expanded) {
      curr_node = _get_best_node(curr_node);
    } else {
      auto node = _expand(curr_node);
      if (node != nullptr)
        return node;
    }
  }

  return curr_node;
}

void
GradientBanditTreeSearch::backup(std::shared_ptr<GradientBanditNode> curr_node, double Q_val) {
  float cum_reward = Q_val;
  while (curr_node) {
    cum_reward += curr_node->reward;

    curr_node->Q += cum_reward;
    curr_node->visits += 1;
    auto parent = curr_node->parent.lock();
    curr_node = parent;
  }
}

void
GradientBanditTreeSearch::reset_policy_cache() {
  policy_net_cache.clear();
}

void
GradientBanditTreeSearch::init_single_bandit(std::shared_ptr<SingleGradientBandit> bandit, std::vector<float> obs) {
  double alpha = params["dirichlet_alpha"];
  double frac = params["dirichlet_frac"];
  int n_actions = params["n_actions"];

  torch::Tensor action_probs;
  std::tie(action_probs, std::ignore) = a2c_agent.predict_policy({obs});

  // Add Dirichlet noise.
  std::gamma_distribution<double> distribution(alpha, 1.);
  for (int q = 0; q < n_actions; ++q) {
    double noise = distribution(generator);
    action_probs[0][q] = action_probs[0][q] * (1 - frac) + noise * frac;
  }

  float* action_probs_arr = action_probs.data_ptr<float>();

  // Create the bandit.
  auto vec = std::vector<float>(action_probs_arr, action_probs_arr + n_actions);
  bandit->H = std::vector<double>(vec.begin(), vec.end());

  // Initialize with log.
  for (int r = 0; r < (int) vec.size(); ++r) {
    bandit->H[r] = std::log(vec[r]);
  }
  bandit->initialized = true;
}

std::vector<double>
GradientBanditTreeSearch::policy(int i, EnvWrapper env, std::vector<float> obs, bool ret_node) {
  // std::random_device rd;
  // std::mt19937 generator(rd());

  int n_actions = params["n_actions"];
  int n_iter = params["simulations"];

  this->env = std::move(env.clone());
  root_node = std::make_shared<GradientBanditNode>();
  root_node->env = std::move(env.clone());
  root_node->state = obs;
  root_node->bandit = std::make_shared<SingleGradientBandit>(params, generator);
  init_single_bandit(root_node->bandit, obs);

  root_node->torch_state = vec_1d_as_tensor(root_node->state, torch::kFloat32);

  torch::Tensor action_probs;
  std::tie(action_probs, std::ignore) = a2c_agent.predict_policy(root_node->torch_state);
  double alpha = params["dirichlet_alpha"];
  double frac = params["dirichlet_frac"];

  std::gamma_distribution<double> distribution(alpha, 1.);
  for (int j = 0; j < n_actions; ++j) {
    double noise = distribution(generator);
    action_probs[0][j] = action_probs[0][j] * (1 - frac) + noise * frac;
  }

  policy_net_cache[root_node] = action_probs;

  for (int j = 0; j < n_iter; ++j) {
    std::shared_ptr<GradientBanditNode> node = select_expand();
    torch::Tensor value;
    std::tie(std::ignore, value) = a2c_agent.predict_policy(node->torch_state);
    double q_val = value[0][0].item<double>();
    backup(node, q_val);
  }

  std::vector<double> ret;
  for (auto child : root_node->children) {
    ret.push_back(1.0 * child->visits / n_iter);
  }

  return ret;
}
