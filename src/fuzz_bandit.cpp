#include "fuzz_bandit.hpp"
#include <iostream>
#include <random>
#include <torch/torch.h>
#include "a2c.hpp"
#include "game.hpp"
#include "gradient_bandit.hpp"


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

    EnvWrapper env = *orig_env.clone();
    auto state = env.reset();

    ReplayBuffer *replay_buffer = new ReplayBuffer(3000, false);
    Registry *registry = new Registry(replay_buffer);

    // true: greedy
    GradientBanditSearch *mcts_agent = new GradientBanditSearch(orig_env, a2c_agent, params, registry, true);
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
