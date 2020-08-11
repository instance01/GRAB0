#include "fuzz_bandit.hpp"
#include <iostream>
#include <random>
#include <torch/torch.h>
#include "a2c.hpp"
#include "game.hpp"
#include "gradient_bandit.hpp"


void fuzz_bandit(json params, EnvWrapper orig_env) {
  EnvWrapper env_ = *orig_env.clone();

  // A few examples.
  //std::string action_str = "222222221220221220212222222022122";
  //std::string action_str = "12201022012222011222222220122021021202201222";
  std::string action_str = "102102200012222002020222212220211222222212222222222210222220222222";

  std::cout << "INPUT " << action_str << std::endl;

  Game game_;
  for(char& c : action_str) {
    int action = c - '0';

    bool done;
    std::vector<float> state;
    double reward;
    std::tie(state, reward, done) = env_.step(action);

    // Easy mode.
    // std::vector<double> mcts_action{0.01, 0.01, 0.01};
    // mcts_action[action] = 0.98;

    // Tough mode.
    std::vector<double> mcts_action{0.31, 0.31, 0.31};
    mcts_action[action] = 0.38;

    game_.states.push_back(state);
    game_.mcts_actions.push_back(mcts_action);
    game_.rewards.push_back(0.0);
  }

  auto a2c_agent = A2CLearner(params, orig_env);

  EnvWrapper env = *orig_env.clone();
  auto state = env.reset();

  GradientBanditSearch *mcts_agent = new GradientBanditSearch(orig_env, a2c_agent, params);
  for (int i = 0; i < game_.mcts_actions.size(); ++i) {
    mcts_agent->bandits[i].H = game_.mcts_actions[i];
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
    
    state = next_state;
    if (done)
      break;
  }

  std::cout << "RESULT " << mcts_actions << std::endl;
}
