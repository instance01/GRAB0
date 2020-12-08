#include <string>
#include <map>
#include <any>
#include <variant>
#include <iostream>

#include <torch/torch.h>
#include <torch/types.h>
#include <torch/nn/modules/container/any.h>
#include <torch/nn/modules/container/sequential.h>
#include <torch/nn/functional.h>
#include <torch/optim/optimizer.h>
#include <torch/optim/adam.h>
#include <torch/optim/sgd.h>
#include <c10/core/DeviceType.h>

#include "gaussian_a2c.hpp"
#include "util.hpp"
#include "gaussian_util.hpp"


namespace F = torch::nn::functional;


GaussianA2CNetImpl::GaussianA2CNetImpl(
    int n_input_features, int n_actions, std::vector<int> net_architecture
) : n_input_features(n_input_features), n_actions(n_actions), net_architecture(net_architecture) {
  seq = register_module("seq", torch::nn::Sequential());

  int n_features_before = n_input_features;

  int i = 0;
  for (int layer_features : net_architecture) {
    auto linear = register_module(
        "l" + std::to_string(i),
        torch::nn::Linear(n_features_before, layer_features)
    );
    linear->reset_parameters();
    auto relu = register_module("r" + std::to_string(i + 1), torch::nn::ReLU());
    seq->push_back(linear);
    seq->push_back(relu);
    n_features_before = layer_features;
    i += 2;
  }

  action_head = register_module("a", torch::nn::Linear(n_features_before, n_actions));
  value_head = register_module("v", torch::nn::Linear(n_features_before, 1));

  action_head->reset_parameters();
  value_head->reset_parameters();

  // Initialize the biases such that variance is 1.
  auto p = action_head->named_parameters(false);
  auto b = p.find("bias");
  torch::nn::init::constant_(*b, 1.0);
}

void GaussianA2CNetImpl::reset() {
  action_head->reset_parameters();
  value_head->reset_parameters();
}

std::pair<torch::Tensor, torch::Tensor>
GaussianA2CNetImpl::forward(torch::Tensor input) {
  auto x = input.view({input.size(0), -1});
  x = seq->forward(x);
  auto policy = action_head(x);
  auto value = value_head(x);
  return std::make_pair(policy, value);
}

GaussianA2CLearner::GaussianA2CLearner(json params, EnvWrapper &env) : params(params) {
  int n_actions = params["n_actions"];
  policy_net = GaussianA2CNet(
    params["n_input_features"],
    n_actions * 2,
    params["net_architecture"]
  );

  double lr = params["alpha"];

  std::string optimizer_class = params["optimizer_class"];
  if (optimizer_class == "adam") {
    auto opt = torch::optim::AdamOptions(lr);
    if (params["use_weight_decay"])
      opt.weight_decay(params["weight_decay"]);
    policy_optimizer = std::make_shared<torch::optim::Adam>(policy_net->parameters(), opt);
  } else if (optimizer_class == "sgd") {
    auto opt = torch::optim::SGDOptions(lr);
    opt.momentum(params["sgd_momentum"]);
    policy_optimizer = std::make_shared<torch::optim::SGD>(policy_net->parameters(), opt);
  }

  expected_mean_tensor = torch::from_blob(
      env.env->expected_mean.data(),
      {env.env->expected_mean.size()},
      torch::TensorOptions().dtype(torch::kFloat32)
  );
  expected_stddev_tensor = torch::from_blob(
      env.env->expected_stddev.data(),
      {env.env->expected_stddev.size()},
      torch::TensorOptions().dtype(torch::kFloat32)
  );
}

torch::Tensor
GaussianA2CLearner::normalize(torch::Tensor x) {
  return (x - expected_mean_tensor) / expected_stddev_tensor;
}

torch::Tensor
GaussianA2CLearner::_calc_normalized_rewards(std::vector<double> rewards) {
  // TODO Consider improving this function.
  double gamma = params["gamma"];
  std::vector<double> discounted_rewards;
  double R = 0;
  std::reverse(rewards.begin(), rewards.end());
  for (double reward : rewards) {
    R = reward + gamma * R;
    discounted_rewards.push_back(R);
  }
  std::reverse(discounted_rewards.begin(), discounted_rewards.end());

  std::vector<float> flt(discounted_rewards.begin(), discounted_rewards.end());

  auto discounted_rewards_tensor = torch::from_blob(
      flt.data(),
      flt.size(),
      torch::TensorOptions().dtype(torch::kFloat32)
  ).clone();  // That clone is extremely important. Major fix.

  if (discounted_rewards_tensor.numel() <= 1) {
    return discounted_rewards_tensor;
  }

  discounted_rewards_tensor = (discounted_rewards_tensor - discounted_rewards_tensor.mean());
  auto std = discounted_rewards_tensor.std();
  if (torch::isnan(std).any().item<bool>()) { // TODO: This check is probably not needed anymore.
    return discounted_rewards_tensor;
  }
  if (std.item<float>() != 0) {
    discounted_rewards_tensor /= std;
  }

  return discounted_rewards_tensor;
}

std::pair<torch::Tensor, torch::Tensor>
GaussianA2CLearner::predict_policy(torch::Tensor samples_) {
  torch::Tensor samples = normalize(samples_.to(torch::kFloat32));
  return policy_net->forward(samples);
}

std::pair<torch::Tensor, torch::Tensor>
GaussianA2CLearner::predict_policy(std::vector<std::vector<float>> states) {
  auto flattened_states = flatten_as_float(states);
  auto samples_tensor = vec_2d_as_tensor(
      flattened_states, torch::kFloat32, states.size(), states[0].size()
  );

  torch::Tensor samples = normalize(samples_tensor);
  return policy_net->forward(samples);
}


torch::Tensor
GaussianA2CLearner::update(std::shared_ptr<Game> game, int n_episode, bool debug_print) {
  // TODO Below remove retain_grad things. This is just for debugging.
  policy_net->train();

  std::vector<std::vector<float>> states = game->states;
  std::vector<std::vector<double>> actions = game->mcts_actions;
  std::vector<double> rewards = game->rewards;

  std::vector<float> states_ = flatten_as_float(states);
  auto samples_tensor = vec_2d_as_tensor(
      states_, torch::kFloat32, states.size(), states[0].size()
  );
  torch::Tensor attached_samples = normalize(samples_tensor);
  auto samples = attached_samples.detach_();

  auto normalized_returns = _calc_normalized_rewards(rewards);
  if (normalized_returns.numel() <= 1) {
    std::cout << "d" << normalized_returns << std::endl;
  }

  auto flattened_mcts = flatten_as_float(actions);
  auto attached_mcts_actions = vec_2d_as_tensor(
      flattened_mcts, torch::kFloat32, actions.size(), actions[0].size()
  );
  auto mcts_actions = attached_mcts_actions.detach_();

  // Forward.
  torch::Tensor action_params;
  torch::Tensor values;
  std::tie(action_params, values) = policy_net->forward(samples);

  // Calculate losses.
  torch::Tensor policy_loss;

  auto rows = torch::arange(0, action_params.size(0), torch::kLong);
  // TODO GENERALIZE THIS TO n ACTIONS!!
  int64_t idx_data_mus[2] = {0, 2};
  auto idx_mus = torch::from_blob(idx_data_mus, 2, torch::TensorOptions().dtype(torch::kLong));
  int64_t idx_data_sigmas[2] = {1, 3};
  auto idx_sigmas = torch::from_blob(idx_data_sigmas, 2, torch::TensorOptions().dtype(torch::kLong));

  auto mus = action_params.index({rows, idx_mus.reshape({-1, 1})});
  auto sigmas = action_params.index({rows, idx_sigmas.reshape({-1, 1})});
  mus.retain_grad();
  sigmas.retain_grad();

  auto gaussian_pdf = (1.0 / (torch::abs(sigmas) * std::sqrt(2 * M_PI))) *\
                      torch::exp(-.5 * torch::pow((mcts_actions.reshape({actions[0].size(), -1}) - mus) / torch::abs(sigmas), 2.));
  gaussian_pdf.requires_grad_(true);
  gaussian_pdf.retain_grad();
  policy_loss = (-torch::log(gaussian_pdf + 1e-8) * normalized_returns.detach()).mean();
  policy_loss.retain_grad();

  torch::Tensor value_loss = F::smooth_l1_loss(
      values.reshape(-1),
      normalized_returns.detach(),
      torch::nn::SmoothL1LossOptions(torch::kSum)
  );
  value_loss /= mcts_actions.detach().size(0);

  torch::Tensor loss = policy_loss + value_loss;
  policy_loss.retain_grad();
  loss.retain_grad();

  policy_optimizer->zero_grad();
  loss.backward({}, true, true);
  auto mus_grad = mus.grad();
  int size_ = std::min(10, (int) game->mcts_actions.size());

  // START DEBUGGING
  //if (mus_grad.numel() <= 2) {
  std::vector<float> v_ga(gaussian_pdf.data_ptr<float>(), gaussian_pdf.data_ptr<float>() + gaussian_pdf.numel());
  std::vector<float> v_mus(mus.data_ptr<float>(), mus.data_ptr<float>() + mus.numel());
  std::vector<float> v_sigmas(sigmas.data_ptr<float>(), sigmas.data_ptr<float>() + sigmas.numel());
  std::cout << std::endl;
  std::cout << "\033[1m[M]\033[0m ";
  for (int i = 0; i < size_; ++i) {
    std::cout << v_mus[i] << " " << v_mus[v_mus.size() / 2 + i] << " ";
  }
  std::cout << std::endl;
  std::cout << "\033[1m[S]\033[0m ";
  for (int i = 0; i < size_; ++i) {
    std::cout << v_sigmas[i] << " " << v_sigmas[v_sigmas.size() / 2 + i] << " ";
  }
  std::cout << std::endl;
  std::cout << "\033[1m[G]\033[0m ";
  for (int i = 0; i < size_; ++i) {
    std::cout << v_ga[i] << " ";
  }
  std::cout << std::endl;

  std::vector<float> v_r(normalized_returns.data_ptr<float>(), normalized_returns.data_ptr<float>() + normalized_returns.numel());
  std::cout << "\033[1m[R]\033[0m ";
  //for (int i = 0; i < v_r.size(); ++i) {
  for (int i = 0; i < size_; ++i) {
    std::cout << v_r[i] << " ";
  }
  std::cout << std::endl;
  std::cout << "\033[1m[MC]\033[0m ";
  for (int i = 0; i < size_; ++i) {
    std::cout << game->mcts_actions[i] << " ";
  }
  std::cout << std::endl;

  std::vector<float> v_mus_grad(mus_grad.data_ptr<float>(), mus_grad.data_ptr<float>() + mus_grad.numel());
  //std::cout << " M G " << std::endl;
  //std::cout << v_mus_grad << std::endl;
  //std::cout << std::endl;
  std::cout << "\033[1m[GRAD]\033[0m ";
  for (int i = 0; i < size_; ++i) {
    std::cout << v_mus_grad[i] << " " << v_mus_grad[v_mus_grad.size() / 2 + i] << " ";
  }
  std::cout << std::endl;
  std::cout << "LL " << policy_loss.item<float>() << " " << value_loss.item<float>() << std::endl;
  //}
  // END DEBUGGING

  policy_optimizer->step();

  policy_net->eval();
  torch::NoGradGuard no_grad;

  int size = std::min(10, (int) game->mcts_actions.size());

  // Print vector of mcts_actions, mus, sigmas.
  // if (debug_print) {
  //   std::vector<float> v_mus(mus.data_ptr<float>(), mus.data_ptr<float>() + mus.numel());
  //   std::vector<float> v_sigmas(sigmas.data_ptr<float>(), sigmas.data_ptr<float>() + sigmas.numel());

  //   std::cout << "\033[1m[MC]\033[0m ";
  //   //for (int i = 0; i < game->mcts_actions.size(); ++i) {
  //   for (int i = 0; i < size; ++i) {
  //     //std::cout << game->mcts_actions[i] << " " << game->mcts_actions[game->mcts_actions.size() / 2 + i] << " ";
  //     std::cout << game->mcts_actions[i] << " ";
  //   }
  //   std::cout << std::endl;
  //   std::cout << "\033[1m[M]\033[0m ";
  //   //for (int i = 0; i < v_mus.size(); ++i) {
  //   for (int i = 0; i < size; ++i) {
  //     std::cout << v_mus[i] << " " << v_mus[v_mus.size() / 2 + i] << " ";
  //   }
  //   std::cout << std::endl;
  //   std::cout << "\033[1m[S]\033[0m ";
  //   //for (int i = 0; i < v_sigmas.size(); ++i) {
  //   for (int i = 0; i < size; ++i) {
  //     std::cout << v_sigmas[i] << " " << v_sigmas[v_sigmas.size() / 2 + i] << " ";
  //   }
  //   std::cout << std::endl;
  // }

  return loss;
}
