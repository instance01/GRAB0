#ifndef GAUSSIAN_A2C_HEADER
#define GAUSSIAN_A2C_HEADER
#include <string>
#include <any>
#include <iostream>

#include <torch/torch.h>
#include <torch/types.h>
#include <torch/nn/modules/container/any.h>
#include <torch/nn/modules/container/sequential.h>
#include <torch/nn/functional.h>
#include <torch/optim/optimizer.h>
#include <c10/core/DeviceType.h>

#include "a2c.hpp"
#include "game.hpp"
#include "cfg.hpp"
#include "env_wrapper.hpp"


struct GaussianA2CNetImpl : public torch::nn::Cloneable<GaussianA2CNetImpl> {
  public:
    torch::nn::Sequential seq{nullptr};
    torch::nn::Linear action_head{nullptr};
    torch::nn::Linear value_head{nullptr};

    int n_input_features;
    int n_actions;
    std::vector<int> net_architecture;

    GaussianA2CNetImpl() {};
    ~GaussianA2CNetImpl() {};
    GaussianA2CNetImpl(int n_input_features, int n_actions, std::vector<int> net_architecture);

    void reset() override;
    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor input);
};

TORCH_MODULE(GaussianA2CNet);

class GaussianA2CLearner : public A2CLearner {
  public:
    // torch::Tensor expected_mean_tensor;
    // torch::Tensor expected_stddev_tensor;

    json params;
    GaussianA2CNet policy_net;
    //std::shared_ptr<torch::optim::Optimizer> policy_optimizer;

    GaussianA2CLearner() {};
    GaussianA2CLearner(json params, EnvWrapper &env);
    ~GaussianA2CLearner() {};

    torch::Tensor normalize(torch::Tensor x);
    std::pair<torch::Tensor, torch::Tensor> predict_policy(torch::Tensor samples_);
    std::pair<torch::Tensor, torch::Tensor> predict_policy(std::vector<std::vector<float>> states);
    torch::Tensor _calc_normalized_rewards(std::vector<double> rewards);
    torch::Tensor update(std::shared_ptr<Game> game, int n_episode, bool do_print);
};
#endif
