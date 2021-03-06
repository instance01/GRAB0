#include "fuzz_bandit.hpp"
#include <iostream>
#include <random>
#include <torch/torch.h>
#include "a2c.hpp"
#include "game.hpp"
#include "gradient_bandit.hpp"
#include "gaussian_gradient_bandit.hpp"
#include "gaussian_util.hpp"
#include "envs/lunar_lander.hpp"


void fuzz_bandit(json params, EnvWrapper orig_env) {
  int n_iter = 10;

  for (int i = 0; i < n_iter; ++i) {
    EnvWrapper env_ = *orig_env.clone();

    // A few examples.
    // std::string action_str = "222222221220221220212222222022122";
    // std::string action_str = "12201022012222011222222220122021021202201222";
    // std::string action_str = "102102200012222002020222212220211222222212222222222210222220222222";

    //std::string action_str = "222222222222222222222222222222222212221100000000000000000000000000000000222222222222222222222222222222222222222222222";
    std::string action_str = "22222222222222222222221220011000020001000000000000000000000001000202221222222222222222222222222222222222222220";
    //std::string action_str = "2222222222222222222222120000100000000000000000000000000000002122222222222222222222222222222222222222222211";
    //std::string action_str = "22222222222222222220000000000000000000000000000000000012202221222222222222222222222222222222222222222211";

    // -167
    //std::string action_str = "11111111111111111222221111111112222222222222222222222222222222222222220000011111111111111111111111000000000000000000000000022222222222222222222222222222222222222222222";
    // with 0.0001, 0.9998, 0.0001 we improved the above to the following: so weird..
    //std::string action_str = "110111111111111010210201011111122222221222222222222222222222222221222200000011111021111000100110210200000201001000200021000222222222222122222222222222222222112222";

    // -157
    //std::string action_str = "0111010111111110120220101111011222222222222222222222222222222222222201200001111111211111101100111100001002020000012200012222222222222222222222222222222222222";

    // TODO:
    //std::string action_str = "00000000000000000000000000000000000000000000000011111111111111111111111002222222222222222220000000000000000000000000000000000000000000000000000000000000000111111111001222222222222222222222222222222222";
    // -195:
    //std::string action_str = "000000000000000000000000000000000000000000011111111111111111111111002222222222222222222222222222222222222200000000000000000000000000000000000000000000111111111001222222222222222222222222222222222";

    std::cout << "INPUT  " << action_str << std::endl;

    Game game_;
    for(char& c : action_str) {
      int action = c - '0';

      bool done;
      std::vector<float> state;
      double reward;
      std::tie(state, reward, done) = env_.step(action);

      // std::vector<double> mcts_action{0.0001, 0.0001, 0.0001};
      // mcts_action[action] = 0.9998;
      //std::vector<double> mcts_action{0.01, 0.01, 0.01};
      //mcts_action[action] = 0.98;

      // std::vector<double> mcts_action{0.01, 0.01, 0.01};
      // mcts_action[action] = 0.98;

      // std::vector<double> mcts_action{0.11, 0.11, 0.11};
      // mcts_action[action] = 0.78;

      std::vector<double> mcts_action{0.32, 0.32, 0.32};
      mcts_action[action] = 0.36;

      // std::vector<double> mcts_action{0.22, 0.22, 0.22};
      // mcts_action[action] = 0.56;

      // std::vector<double> mcts_action{0.27, 0.27, 0.27};
      // mcts_action[action] = 0.46;

      game_.states.push_back(state);
      game_.mcts_actions.push_back(mcts_action);
      game_.rewards.push_back(0.0);
    }

    auto a2c_agent = A2CLearner(params, orig_env);

    std::random_device rd;
    std::mt19937 generator(rd());

    EnvWrapper env = *orig_env.clone();
    auto state = env.reset(generator);

    ReplayBuffer *replay_buffer = new ReplayBuffer(3000, false);
    Registry *registry = new Registry(replay_buffer);

    // true: greedy
    GradientBanditSearch *mcts_agent = new GradientBanditSearch(orig_env, &a2c_agent, params, registry, generator, true);
    for (int i = 0; i < (int) game_.mcts_actions.size(); ++i) {
      for (int j = 0; j < mcts_agent->bandits[i].H.size(); ++j) {
        mcts_agent->bandits[i].H[j] = std::log(game_.mcts_actions[i][j]);
      }
    }

    bool greedy = true;
    bool done = false;
    std::shared_ptr<Game> game = std::make_shared<Game>();
    game->is_greedy = greedy;
    std::string mcts_actions = "";
    for (int i = 0; i < 400; ++i) {
      auto mcts_action = mcts_agent->policy(i, env, state);
      
      int sampled_action;
      auto max_el = std::max_element(mcts_action.begin(), mcts_action.end());
      sampled_action = std::distance(mcts_action.begin(), max_el);
      mcts_actions += std::to_string(sampled_action);
      
      torch::Tensor action_probs;
      std::tie(action_probs, std::ignore) = a2c_agent.predict_policy({state});
      
      std::vector<float> next_state;
      double reward;
      std::tie(next_state, reward, done) = env.step(sampled_action);
      
      game->states.push_back(state);
      game->rewards.push_back(reward);
      game->mcts_actions.push_back(mcts_action);

      mcts_agent->history.rewards.push_back(reward);
      
      state = next_state;
      if (done)
        break;
    }

    std::cout << "RESULT " << mcts_actions << std::endl;

    delete registry;
    delete replay_buffer;
  }
}

void test_gaussian_bandit() {
  std::random_device rd;
  std::mt19937 generator(rd());

  auto params = load_cfg("lunar1");
  EnvWrapper env = EnvWrapper();
  env.init("lunar", params);
  bool continuous = params["continuous"];
  A2CLearner* a2c_agent;
  if (continuous) {
    a2c_agent = new GaussianA2CLearner(params, env);
  } else {
    a2c_agent = new A2CLearner(params, env);
  }

  ReplayBuffer *replay_buffer = new ReplayBuffer(
    params["memory_capacity"],
    params["experimental_top_cutoff"],
    params["prioritized_sampling"]
  );
  Registry *registry = new Registry(replay_buffer);

  // TODO REMOVE
  auto lunar_env = std::static_pointer_cast<LunarLanderEnv>(env.env);

  bool done = false;
  auto state = env.reset(generator);

  // TODO REMOVE
  bool allow_heuristic1 = true;
  bool use_heuristic1 = false;
  bool use_heuristic2 = false;

  int i = 0;
  float total_reward = 0.0;
  while (!done) {
    i += 1;
    if (i > 500)
      break;
    auto mcts = new GaussianGradientBanditSearch(env, a2c_agent, params, registry, generator, false, false);
    auto mcts_action = mcts->policy(0, env, state);
    std::vector<float> action_params = std::vector<float>(mcts_action.begin(), mcts_action.end());

    std::vector<double> sampled_action(2);
    for (int i = 0; i < 2; ++i) {
      sampled_action[i] =  sample(action_params[i*2], action_params[i*2+1], generator);
    }

    // TODO REMOVE ; just for testing whether we can even win this game?
    auto pos = lunar_env->lander->GetPosition();
    if (pos.y < 4.09 && allow_heuristic1) {
      use_heuristic1 = true;
    }
    if (use_heuristic1) {
      sampled_action = {1.0, 0.0};
    }
    if (use_heuristic1 && pos.y > 4.09) {
      allow_heuristic1 = false;
      sampled_action = {-1.0, 0.0};
    }
    if (pos.y < 3.92) {
      use_heuristic2 = true;
      allow_heuristic1 = false;
    }
    if (use_heuristic2) {
      sampled_action = {-1.0, 0.0};
    }

    // TODO REMOVE DEBUG PRINTS
    std::cout << "|step: " << i << " |a: " << action_params << " |sa: " << sampled_action << std::endl;
    std::cout << pos.x << " " << pos.y << std::endl;

    float reward;
    std::tie(state, reward, done) = env.step(sampled_action);
    total_reward += reward;
    std::cout << reward << " tot:" << total_reward << " " << done << std::endl;
  }
}
