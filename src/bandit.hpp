#ifndef BANDIT_HEADER
#define BANDIT_HEADER
#include <vector>
#include "env_wrapper.hpp"
#include "a2c.hpp"
#include "cfg.hpp"
#include "bandit.hpp"


class Bandit {
  public:
    Bandit(EnvWrapper env, A2CLearner a2c_agent, json params);
    Bandit() {};
    virtual ~Bandit() {};

    Game history;

    virtual void reset_policy_cache() {};
    virtual std::vector<double> policy(int i, EnvWrapper &env, const std::vector<float> &obs, bool ret_node=false) {
      return {};
    };
};
#endif
