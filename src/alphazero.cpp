#include <random>
#include <iostream>
#include <algorithm>
#include <torch/torch.h>
#include "tensorboard_logger.h"
#include "alphazero.hpp"
#include "replay_buffer.hpp"
#include "tensorboard_util.hpp"
#include "simple_thread_pool.hpp"
#include "cfg.hpp"
#include "lr_scheduler.hpp"
#include "gradient_bandit.hpp"
#include "gaussian_gradient_bandit.hpp"
#include "fuzz_bandit.hpp"
#include "util.hpp"
#include "gradient_bandit_tree.hpp"
#include "gaussian_a2c.hpp"
#include "gaussian_util.hpp"


std::tuple<int, double, double> evaluate(EnvWrapper env, json &params, A2CLearner* a2c_agent, Registry *registry) {
  std::random_device rd;
  std::mt19937 generator(rd());

  bool continuous = params["continuous"];

  env = *env.clone();
  int n_actions = params["n_actions"];

  int min_len = 1e6;
  int max_len = 0;
  int total_length = 0;
  int total_length_sq = 0;
  double min_reward = 1e6;
  double max_reward = -1e6;
  double total_reward = 0.;
  double total_reward_sq = 0.;

  float n_iters = 100.0;
  std::cout << "EVAL ";
  for (int j = 0; j < (int) n_iters; ++j) {
    std::vector<std::vector<float>> states;
    std::vector<std::vector<double>> mcts_actions;
    std::vector<double> rewards;

    std::vector<float> state = env.reset(generator);

    bool done = false;
    std::string actions = "";

    std::cout << std::setprecision(3);
    double curr_reward = 0;
    while (!done) {
      torch::Tensor action_probs;
      torch::Tensor val;
      std::tie(action_probs, val) = a2c_agent->predict_policy({state});

      if (continuous) {
        std::vector<double> sampled_continuous_action(2);
        float* action_probs_arr = action_probs.data_ptr<float>();
        auto vec = std::vector<float>(action_probs_arr, action_probs_arr + n_actions * 2);
        for (int i = 0; i < 2; ++i) {
          sampled_continuous_action[i] = sample(vec[i*2], vec[i*2+1], generator);
        }

        mcts_actions.push_back(sampled_continuous_action);
        states.push_back(state);

        double reward;
        std::tie(state, reward, done) = env.step(sampled_continuous_action);
        curr_reward += reward;
        actions += std::to_string(0); // TODO placeholder

        rewards.push_back(reward);
      } else {
        int action = action_probs.argmax().item<int>();

        float* action_probs_arr = action_probs.data_ptr<float>();
        auto vec = std::vector<float>(action_probs_arr, action_probs_arr + n_actions);
        mcts_actions.push_back(std::vector<double>(vec.begin(), vec.end()));
        states.push_back(state);

        // if (j == 0) {
        //   // TODO Remove. This prints current policy and value in compact way.
        //   float* xx = (float*)action_probs.data_ptr();
        //   for (int i = 0; i < action_probs.sizes()[1]; ++i)
        //     std::cout << std::ceil(xx[i] * 100.0) / 100.0 << " ";
        //   //std::cout << "(" << std::ceil(val.item<float>() * 100.0) / 100.0 << ") # ";
        //   std::cout << "| ";
        // }

        double reward;
        std::tie(state, reward, done) = env.step(action);
        curr_reward += reward;
        actions += std::to_string(action);

        rewards.push_back(reward);
      }
    }
    total_reward += curr_reward;
    total_reward_sq += std::pow(curr_reward, 2);
    min_reward = std::min(min_reward, curr_reward);
    max_reward = std::max(max_reward, curr_reward);

    Game registry_game;
    registry_game.states = states;
    registry_game.rewards = rewards;
    registry_game.mcts_actions = mcts_actions;
    registry_game.tot_reward = total_reward / (j + 1);
    registry->save_if_best(registry_game, total_reward / (j + 1));

    min_len = std::min(min_len, (int) actions.length());
    max_len = std::max(max_len, (int) actions.length());
    total_length += actions.length();
    total_length_sq += std::pow(actions.length(), 2);

    if (j == 0 && !continuous)
      std::cout << actions << " " << actions.length() << std::endl;
    std::cout << curr_reward << " ";
  }
  std::cout << std::endl;
  // TODO This was calculating var/min/max based on length..
  // double var = (total_length_sq - (std::pow(total_length, 2) / n_iters)) / n_iters;
  // double avg = total_reward / n_iters;
  // std::cout << "AVG " << avg <<  " |VAR " << var << " |MIN " << min_len << " |MAX " << max_len << std::endl;
  double var = (total_reward_sq - (std::pow(total_reward, 2) / n_iters)) / n_iters;
  double avg = total_reward / n_iters;
  std::cout << "AVG " << avg <<  " |VAR " << var << " |MIN " << min_reward << " |MAX " << max_reward << std::endl;

  return std::make_tuple(total_length / n_iters, avg, var);
}

float schedule_alpha(
    json params,
    A2CLearner* a2c_agent,
    LRScheduler *lr_scheduler,
    double total_reward,
    int n_episode
) {
  float lr = 0;
  if (params["optimizer_class"] == "adam") {
    // The decision to only use the first param group might be dubious.
    // Keep that in mind. For now it is fine, I checked.
    auto& options = static_cast<torch::optim::AdamOptions&>(
        a2c_agent->policy_optimizer->param_groups()[0].options()
    );
    lr = options.lr();
    if (params["schedule_alpha"]) {
      options.lr(lr_scheduler->step(lr, n_episode, total_reward));
    }
  }
  // TODO Deal with SGD at some point.
  return lr;
}

void write_tensorboard_kpis(
    TensorBoardLogger &writer,
    double mcts_confidence_median,
    int n_run,
    int n_episode,
    int eval_length,
    double total_reward,
    const std::vector<int>& actor_lengths,
    const std::vector<int>& sample_lens,
    const std::vector<double>& losses,
    float lr,
    double avg_loss,
    double var
) {
  writer.add_scalar("Eval/MCTS_Confidence/" + std::to_string(n_run), n_episode, (float) mcts_confidence_median);
  writer.add_scalar("Eval/Eval_var/" + std::to_string(n_run), n_episode, (float) var);

  writer.add_scalar("Eval/Length/" + std::to_string(n_run), n_episode, (float) eval_length);
  writer.add_scalar("Eval/Reward/" + std::to_string(n_run), n_episode, total_reward);
  writer.add_histogram(
      "Actor/Sample_length/" + std::to_string(n_run), n_episode, actor_lengths
  );
  writer.add_histogram(
      "Train/Samples/" + std::to_string(n_run), n_episode, sample_lens
  );
  writer.add_histogram(
      "Train/Loss/" + std::to_string(n_run), n_episode, losses
  );

  std::cout << "LR " << lr << std::endl;

  writer.add_scalar("Train/LearningRate/" + std::to_string(n_run), n_episode, lr);
  writer.add_scalar("Train/AvgLoss/" + std::to_string(n_run), n_episode, avg_loss);
}

std::vector<int> run_actors(
    EnvWrapper &env,
    json &params,
    A2CLearner* a2c_agent,
    int n_episode,
    ReplayBuffer *replay_buffer,
    Registry *registry
) {
  std::random_device rd;
  int n_actors = params["n_actors"];
  int n_procs = params["n_procs"];
  std::vector<int> actor_lengths;

  bool do_warmup = params["do_warmup"];
  int n_warmup = params["n_warmup"];

  // Run self play games in n_procs parallel processes.
  auto pool = SimpleThreadPool(n_procs);
  std::vector<Task*> tasks;
  for (int i = 0; i < n_actors; ++i) {
    auto lambda = [i, &rd, env, params, a2c_agent, n_episode, registry, replay_buffer]() -> std::shared_ptr<Game> {
      return run_actor(i, rd, env, params, a2c_agent, n_episode, registry, replay_buffer);
    };
    Task *task = new Task(lambda);
    pool.add_task(task);
    tasks.push_back(task);
  }
  if (do_warmup) {
    params["do_warmup"] = false;
    for (int i = 0; i < n_warmup; ++i) {
      auto lambda = [i, &rd, env, params, a2c_agent, n_episode, registry, replay_buffer]() -> std::shared_ptr<Game> {
        return run_actor(i + 200, rd, env, params, a2c_agent, n_episode, registry, replay_buffer);
      };
      Task *task = new Task(lambda);
      pool.add_task(task);
      tasks.push_back(task);
    }
  }

  // If using gradient bandits: Add one super greedy bandit.
  std::string bandit_type = params["bandit_type"];
  if (bandit_type == "grad") {
    auto lambda2 = [&rd, env, params, a2c_agent, n_episode, registry, replay_buffer]() -> std::shared_ptr<Game> {
      // 'true' denotes greedy.
      return run_actor(1000, rd, env, params, a2c_agent, n_episode, registry, replay_buffer, true);
    };
    Task *task = new Task(lambda2);
    pool.add_task(task);
    tasks.push_back(task);
  }

  std::vector<std::shared_ptr<Game>> games = pool.join();

  for (Task* task : tasks) {
    delete task;
  }
  tasks.clear();

  for(auto game : games) {
    actor_lengths.push_back(game->states.size());
    replay_buffer->add(std::move(game));
  }

  return actor_lengths;
}

std::pair<int, double> episode(
  TensorBoardLogger &writer,
  int n_run,
  EnvWrapper env,
  A2CLearner* a2c_agent,
  int n_episode,
  ReplayBuffer *replay_buffer,
  json &params,
  LRScheduler *lr_scheduler,
  Registry *registry,
  std::chrono::time_point<std::chrono::high_resolution_clock> start_time
) {
  a2c_agent->policy_net->eval();

  // Run self play games in n_procs parallel processes.
  std::vector<int> actor_lengths = run_actors(
      env, params, a2c_agent, n_episode, replay_buffer, registry
  );

  bool continuous = params["continuous"];

  // Print debug information.
  int n_actors = params["n_actors"];
  std::string bandit_type = params["bandit_type"];
  if (bandit_type == "grad")
    n_actors += 1;
  std::cout << "REWARDS ";
  auto rewards = replay_buffer->get_rewards();
  int size_ = rewards.size() - n_actors - 1;
  for (int i = rewards.size() - 1; i > size_; --i) {
    if (replay_buffer->buffer[i]->is_greedy)
      std::cout << "G";
    std::cout << rewards[i] << " ";
  }
  std::cout << std::endl;

  // Calculate MCTS confidence for debugging purposes
  std::vector<double> max_action_probs;
  size_ = replay_buffer->buffer.size() - n_actors - 1;
  for (int i = replay_buffer->buffer.size() - 1; i > size_; --i) {
    auto game = replay_buffer->buffer[i];
    for (auto mcts_action : game->mcts_actions) {
      auto max_el = std::max_element(mcts_action.begin(), mcts_action.end());
      max_action_probs.push_back(*max_el);
      //std::cout << *max_el << " ";
    }
  }
  //std::cout << std::endl;
  auto mcts_confidence_mean = mean<double>(max_action_probs);
  auto mcts_confidence_median = median<double>(max_action_probs);
  std::cout << "CONFIDENCE " << mcts_confidence_mean << " " << mcts_confidence_median << std::endl;

  // Train network after self play.
  int train_steps = params["train_steps"];
  std::vector<int> sample_lens;
  std::vector<double> losses;

  std::random_device rd;
  std::mt19937 generator(rd());
  std::uniform_real_distribution<> epsgreedy_distribution(0, 1);

  bool experimental_use_top_percentile = params["experimental_use_top_percentile"];
  float experimental_top_percentile = params["experimental_top_percentile"];

  bool debug_do_print = false;
  bool use_eps_greedy_learning = params["use_eps_greedy_learning"];
  float eps_greedy_epsilon_decay_factor = params["eps_greedy_epsilon_decay_factor_train"];

  float epsilon = std::pow(eps_greedy_epsilon_decay_factor, n_episode);
  std::cout << "train epsilon " << epsilon << std::endl;

  bool experimental_use_train_schedule = params["experimental_use_train_schedule"];

  // Sep25 18:30 Added
  // Sep25 19:05 Commented out
  // Sep26 08:10 Enabled again
  // Sep26 22:00 Disabled
  // Sep27 06:57 Enabled
  if (experimental_use_train_schedule) {
    if (n_episode > 300)
      train_steps = 100;
    if (n_episode > 400)
      train_steps = 200;
    if (n_episode > 800)
      train_steps = 500;
  }

  // TRAINING.
  a2c_agent->policy_net->train();
  for (int i = 0; i < train_steps; ++i) {
    std::shared_ptr<Game> game;

    bool greedy = false;
    if (use_eps_greedy_learning)
      greedy = epsgreedy_distribution(generator) > epsilon;

    // a2c debug prints
    bool debug_print = (i == 0) || (i == (train_steps - 1));

    if (greedy) {
      // When greedy, train with top games.
      std::cout << "TOP: ";
      std::vector<std::shared_ptr<Game>> games;
      if (experimental_use_top_percentile) {
        games = replay_buffer->get_top_p(experimental_top_percentile);
      } else {
        games = replay_buffer->get_top();
      }
      for (auto game : games) {
        auto loss = a2c_agent->update(game, n_episode, debug_print);
        debug_print = false;

        std::string actions;

        if (continuous) {
          // double tot_reward = 0;
          // for (auto reward : game->rewards) {
          //   tot_reward += reward;
          //   actions += "0";
          // }
          // std::cout << tot_reward << " ";
          std::cout << game->tot_reward << " ";
        } else {
          for (auto mcts_action : game->mcts_actions) {
            auto max_el = std::max_element(mcts_action.begin(), mcts_action.end());
            actions += std::to_string(std::distance(mcts_action.begin(), max_el));
          }
          std::cout << actions.size() << " ";
        }
        sample_lens.push_back(actions.size());
        losses.push_back(loss.item<double>());
      }
      std::cout << std::endl;

      if (games.size() == 0) {
        game = replay_buffer->sample();
      } else  {
        continue; // !!
      }
    } else {
      game = replay_buffer->sample();
    }

    auto loss = a2c_agent->update(game, n_episode, debug_print);

    std::string actions;
    for (auto mcts_action : game->mcts_actions) {
      auto max_el = std::max_element(mcts_action.begin(), mcts_action.end());
      actions += std::to_string(std::distance(mcts_action.begin(), max_el));
    }

    if (greedy && !debug_do_print) {
      debug_do_print = true;
      std::cout << "first greedy action " << actions << " |eps " << epsilon << std::endl;
    }

    //std::cout << actions << std::endl;

    if (i % 10 == 0) {
      std::cout << "." << std::flush;
    }

    sample_lens.push_back(actions.size());
    losses.push_back(loss.item<double>());
  }

  std::cout << std::endl;

  a2c_agent->policy_net->eval();

  auto curr_time = std::chrono::high_resolution_clock::now();

  double avg_loss = mean<double>(losses);
  std::cout <<
      "AVG LENS " << mean<int>(sample_lens) <<
      " |AVG LOSS " << avg_loss <<
      " |TIME (min) " << std::chrono::duration_cast<std::chrono::minutes>(curr_time - start_time).count() <<
      std::endl;
  int eval_length;
  double total_reward;
  double var;
  std::tie(eval_length, total_reward, var) = evaluate(env, params, a2c_agent, registry);

  float lr = schedule_alpha(params, a2c_agent, lr_scheduler, total_reward, n_episode);

  write_tensorboard_kpis(
    writer,
    mcts_confidence_median,
    n_run,
    n_episode,
    eval_length,
    total_reward,
    actor_lengths,
    sample_lens,
    losses,
    lr,
    avg_loss,
    var);
  std::cout << std::endl;

  return {eval_length, total_reward};
}

std::shared_ptr<Game> run_actor(
    int idx,
    std::random_device &rd,
    EnvWrapper orig_env,
    json params,
    A2CLearner* a2c_agent,
    int n_episode,
    Registry *registry,
    ReplayBuffer *replay_buffer,
    bool greedy_bandit) {
  thread_local std::mt19937 generator(rd());

  EnvWrapper env = *orig_env.clone();
  auto state = env.reset(generator);

  int n_actions = params["n_actions"];

  // TODO Creating a new bandit here every time succs..
  std::string bandit_type = params["bandit_type"];
  Bandit *mcts_agent;
  if (bandit_type == "mcts") {
    mcts_agent = new MCTS(orig_env, a2c_agent, params);
  }
  if (bandit_type == "grad_tree") {
    mcts_agent = new GradientBanditTreeSearch(orig_env, a2c_agent, params, generator);
  }

  std::string mcts_actions = "";

  float eps_greedy_epsilon_decay_factor = params["eps_greedy_epsilon_decay_factor_actor"];
  float epsilon = std::pow(eps_greedy_epsilon_decay_factor, n_episode);
  bool greedy = false;
  std::uniform_real_distribution<> epsgreedy_distribution(0, 1);
  if (params["use_eps_greedy_learning"]) {
    greedy = epsgreedy_distribution(generator) > epsilon;
  }

  // Allow following a2c at the beginning for exploration.
  // Following the bandit is how AlphaZero does it. Following a2c is how this paper does it:
  // `Thinking Fast and Slow with Deep Learning and Tree Search`.
  float eps_greedy_epsilon_decay_factor_a2c = params["eps_greedy_epsilon_decay_factor_follow_a2c"];
  epsilon = std::pow(eps_greedy_epsilon_decay_factor_a2c, n_episode);
  bool follow_bandit_greedily = epsgreedy_distribution(generator) > epsilon;

  int follow_after_ep = params["experimental_follow_after_ep"];
  float follow_eps_power = params["experimental_follow_eps_power"];
  int follow_after_ep2 = params["experimental_follow_after_ep2"];
  float follow_eps_power2 = params["experimental_follow_eps_power2"];
  if (n_episode > follow_after_ep) {
      epsilon = std::pow(epsilon, follow_eps_power);
      follow_bandit_greedily = epsgreedy_distribution(generator) > epsilon;
  }

  if (n_episode > follow_after_ep2) {
      epsilon = std::pow(epsilon, follow_eps_power2);
      follow_bandit_greedily = epsgreedy_distribution(generator) > epsilon;
  }

  if (greedy_bandit)
    greedy = true;

  bool continuous = params["continuous"];

  double total_reward = 0;
  bool done = false;
  std::shared_ptr<Game> game = std::make_shared<Game>();
  game->is_greedy = greedy;
  if (continuous) {
    // TODO: Rethink this. But it makes sense. With gaussian policy there seems to be no greediness so to say..
    // So let's just put all of them as greedy to sample from TOP
    game->is_greedy = true;
  }
  // In the continuous case there is no such thing as game2 tbh.. but yea. Just something to keep in mind.
  std::shared_ptr<Game> game2 = std::make_shared<Game>();
  game2->is_greedy = true;
  for (int i = 0; i < env.env->max_steps; ++i) {
    bool do_print = (i == 0 && idx == 2);
    if (bandit_type == "grad") {
      if (continuous) {
        mcts_agent = new GaussianGradientBanditSearch(orig_env, a2c_agent, params, registry, generator, false, false);
      } else {
        mcts_agent = new GradientBanditSearch(orig_env, a2c_agent, params, registry, generator, do_print, greedy_bandit);
      }
    }
    std::cout << i << " " << std::flush;
    auto mcts_action = mcts_agent->policy(0, env, state);
    if (bandit_type == "grad") {
      delete mcts_agent;
    }

    std::vector<double> sampled_continuous_action(2);
    std::vector<float> next_state;
    double reward;
    if (continuous) {
      // TODO: Not necessary to rename like this. Remove at some point. Just for easier readability right now.
      auto action_params = mcts_action;

      for (int i = 0; i < 2; ++i) {
        sampled_continuous_action[i] = sample(action_params[i*2], action_params[i*2+1], generator);
      }

      torch::Tensor action_probs;
      std::tie(action_probs, std::ignore) = a2c_agent->predict_policy({state});

      // Here we follow a2c.
      float* action_probs_arr = action_probs.data_ptr<float>();
      auto vec = std::vector<float>(action_probs_arr, action_probs_arr + n_actions * 2);
      std::vector<double> a2c_sampled_action(2);
      for (int i = 0; i < 2; ++i) {
        a2c_sampled_action[i] = sample(vec[i*2], vec[i*2+1], generator);
      }
      if (!follow_bandit_greedily) {
        sampled_continuous_action = a2c_sampled_action;
      }

      std::tie(next_state, reward, done) = env.step(sampled_continuous_action);

      if (next_state.empty()) {
          std::cout << reward << " " << done << std::endl; // spoiler: theyre all 0s.
          abort();
      }

      total_reward += reward;
    } else {
      int sampled_action;
      if (greedy) {
        auto max_el = std::max_element(mcts_action.begin(), mcts_action.end());
        sampled_action = std::distance(mcts_action.begin(), max_el);
      } else {
        std::discrete_distribution<int> distribution(mcts_action.begin(), mcts_action.end());
        sampled_action = distribution(generator);
      }
      mcts_actions += std::to_string(sampled_action);

      torch::Tensor action_probs;
      std::tie(action_probs, std::ignore) = a2c_agent->predict_policy({state});

      // Here we follow a2c.
      float* action_probs_arr = action_probs.data_ptr<float>();
      auto vec = std::vector<float>(action_probs_arr, action_probs_arr + n_actions);
      std::discrete_distribution<int> distribution_(vec.begin(), vec.end());
      int a2c_sampled_action = distribution_(generator);
      if (!follow_bandit_greedily) {
        sampled_action = a2c_sampled_action;
      }

      std::tie(next_state, reward, done) = env.step(sampled_action);

      total_reward += reward;

      // If tough_ce=false, we can keep a game2 object here with a fake greedy path (one-hot encoded)
      std::vector<double> soft_greedy_action_probs(n_actions, 0.00001);
      if (!params["tough_ce"] && !greedy) {
        // Adds up to slightly more than 1... TODO Fix at some point.
        soft_greedy_action_probs[sampled_action] = 0.99999;
      }

      game2->states.push_back(state);
      game2->rewards.push_back(reward);
      game2->mcts_actions.push_back(soft_greedy_action_probs);
    }

    game->states.push_back(state);
    game->rewards.push_back(reward);
    if (continuous) {
      game->mcts_actions.push_back(sampled_continuous_action);
    } else {
      game->mcts_actions.push_back(mcts_action);
    }

    state = next_state;

    if (done)
      break;
  }

  if (!continuous)
    std::cout << mcts_actions << std::endl;

  game->tot_reward = total_reward;
  if (continuous) {
    replay_buffer->add(game);
  } else {
    registry->save_if_best(*game, total_reward);
  }

  float experimental_top_cutoff = params["experimental_top_fill"];
  bool experimental_top_fill = params["experimental_top_fill"];
  if (!params["tough_ce"] && !greedy && experimental_top_fill && !continuous) {
    std::cout << "## " << game2->mcts_actions.size() << std::endl;
    game2->tot_reward = total_reward;
    registry->save_if_best(*game2, total_reward);
    if (total_reward >= experimental_top_cutoff)
      replay_buffer->add(game2);
  }

  if (bandit_type != "grad") {
    delete mcts_agent;
  }
  return game;
}

std::tuple<int, int, double> run(EnvWrapper env, json params, int n_run, TensorBoardLogger &writer) {
  std::ostringstream oss;
  oss << params.dump();
  writer.add_text("Info/params/" + std::to_string(n_run), 0, oss.str().c_str());

  auto start_time = std::chrono::high_resolution_clock::now();
  ReplayBuffer *replay_buffer = new ReplayBuffer(
    params["memory_capacity"],
    params["experimental_top_cutoff"],
    params["prioritized_sampling"]
  );
  bool continuous = params["continuous"];
  A2CLearner* a2c_agent;
  if (continuous) {
    a2c_agent = new GaussianA2CLearner(params, env);
  } else {
    a2c_agent = new A2CLearner(params, env);
  }

  LRScheduler *lr_scheduler = nullptr;
  if (params["scheduler_class"] == "exp") {
    lr_scheduler = new ExponentialScheduler(
        params["scheduler_factor"],
        params["scheduler_min_lr"]
    );
  } else if (params["scheduler_class"] == "step") {
    lr_scheduler = new StepScheduler(
        params["scheduler_steps"],
        params["scheduler_factor"],
        params["scheduler_min_lr"]
    );
  } else if (params["scheduler_class"] == "reduce_eval") {
    lr_scheduler = new ReduceOnGoodEval(
        params["scheduler_factor"],
        params["scheduler_min_good_eval"],
        params["scheduler_min_n_good_evals"],
        params["scheduler_min_lr"],
        params["scheduler_consecutive"]
    );
  }

  Registry *registry = new Registry(replay_buffer);

  // Need to have less than or equal desired evaluation length, certain times in a row.
  int is_done_stably = 0;
  int desired_eval_len = params["desired_eval_len"];
  int n_desired_eval_len = params["n_desired_eval_len"];

  int eval_len = 0;
  double total_reward = 0;
  int i = 0;
  //std::vector<double> rewards;
  for (; i < params["episodes"]; ++i) {
    std::cout << "Episode " << i << std::endl;

    std::tie(eval_len, total_reward) = episode(
        writer,
        n_run,
        env,
        a2c_agent,
        i,
        replay_buffer,
        params,
        lr_scheduler,
        registry,
        start_time
    );
    //rewards.push_back(total_reward);

    if (params["eval_break_on_good_avg_reward"]) {
      double min_reward = params["eval_min_avg_reward"];
      if (total_reward >= min_reward) {
        std::cout << "Good evaluation. Breaking.." << std::endl;
        break;
      }
    }

    if (eval_len <= desired_eval_len) {
        is_done_stably += 1;
    } else {
        is_done_stably = 0;
    }

    if (is_done_stably > n_desired_eval_len)
        break;
  }

  // writer.add_histogram(
  //     "Summary/Rewards_All", n_run, rewards
  // );
  writer.add_scalar(
      "Summary/Rewards_All_Single", n_run, total_reward
  );
  delete lr_scheduler;
  delete registry;
  delete replay_buffer;
  return std::make_tuple(i, eval_len, total_reward);
}

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cout << argv[0] << " <game> <param_identifier> [mode]" << std::endl;
    std::cout << "<game> can be 5x5, 8x8, 16x16, mtcar, cart, lander." << std::endl;
    std::cout << "<param_identifier> can be any key from simulations.json, e.g. 110 or FUZZTEST5." << std::endl;
    std::cout << "[mode] can be 'fuzz', used for debugging the gradient bandit." << std::endl;
    return 1;
  }
  std::string game(argv[1]);
  std::string param_num(argv[2]);
  std::cout << "Running with parameters: " << game << " " << param_num << std::endl;

  // TODO This is important. Maybe there's more such improvements?
  at::init_num_threads();

  auto params = load_cfg(param_num);

  EnvWrapper env = EnvWrapper();
  env.init(game, params);

  if (argc > 3) {
    std::string mode(argv[3]);
    if (mode == "fuzz") {
      fuzz_bandit(params, env);
      return 0;
    } else if (mode == "gauss") {
      test_gaussian_bandit();
    }
  }

  TensorBoardLogger writer(gen_log_filename(game, param_num).c_str());

  for (int i = 0; i < params["n_runs"]; ++i)
    run(env, params, i, writer);
}
