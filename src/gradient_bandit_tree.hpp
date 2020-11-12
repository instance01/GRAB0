#ifndef GRADIENT_BANDIT_TREE_HEADER
#define GRADIENT_BANDIT_TREE_HEADER
#include <cstdlib>
#include <vector>
#include <map>
#include <random>
#include <torch/torch.h>
#include "bandit.hpp"
#include "env_wrapper.hpp"
#include "a2c.hpp"
#include "cfg.hpp"
#include "gradient_bandit.hpp"


class GradientBanditNode {
  public:
    long _id;
    bool is_fully_expanded = false;
    bool is_terminal = false;
    double reward = 0.;
    std::weak_ptr<GradientBanditNode> parent;
    std::vector<std::shared_ptr<GradientBanditNode>> children;
    int action;
    double Q = 0.;
    int visits = 0;
    std::vector<float> state;
    torch::Tensor torch_state;
    std::shared_ptr<EnvWrapper> env;

    std::shared_ptr<SingleGradientBandit> bandit;

    // TODO Increase that?
    GradientBanditNode() : _id(std::rand() % 2147483648) {};
    ~GradientBanditNode() {};

    bool operator==(GradientBanditNode& other) {
      return _id == other._id;
    }
};

class GradientBanditTreeSearch : public Bandit {
  public:
    json params;
    A2CLearner* a2c_agent;
    // shared_ptr because unique_ptr makes this class uncopyable.
    // And I don't want to define a custom copy function.
    std::shared_ptr<EnvWrapper> env;
    std::shared_ptr<GradientBanditNode> root_node;

    std::mt19937 generator;

    std::unordered_map<std::shared_ptr<GradientBanditNode>, torch::Tensor> policy_net_cache;

    GradientBanditTreeSearch(EnvWrapper env, A2CLearner* a2c_agent, json params, std::mt19937 &generator);
    GradientBanditTreeSearch() {};
    ~GradientBanditTreeSearch() {};

    void _gen_children_nodes(std::shared_ptr<GradientBanditNode> parent_node);
    std::shared_ptr<GradientBanditNode> _expand(std::shared_ptr<GradientBanditNode> parent_node);
    std::shared_ptr<GradientBanditNode> _get_best_node(std::shared_ptr<GradientBanditNode> parent_node);
    std::shared_ptr<GradientBanditNode> select_expand();
    void backup(std::shared_ptr<GradientBanditNode> curr_node, double Q_val);
    void reset_policy_cache();
    std::vector<double> policy(int i, EnvWrapper env, std::vector<float> obs, bool ret_node=false);

    void init_single_bandit(std::shared_ptr<SingleGradientBandit> bandit, std::vector<float> obs);
};
#endif
