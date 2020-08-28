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
#include "fuzz_bandit.hpp"
#include "util.hpp"


std::tuple<int, double, double> evaluate(EnvWrapper env, json &params, A2CLearner a2c_agent, Registry *registry) {
  env = *env.clone();
  int n_actions = params["n_actions"];

  int min_len = 1e6;
  int max_len = 0;
  int total_length = 0;
  int total_length_sq = 0;
  double total_reward = 0.;

  float n_iters = 100.0;
  for (int j = 0; j < (int) n_iters; ++j) {
    std::vector<std::vector<float>> states;
    std::vector<std::vector<double>> mcts_actions;
    std::vector<double> rewards;

    std::vector<float> state = env.reset();

    bool done = false;
    std::string actions = "";

    while (!done) {
      torch::Tensor action_probs;
      torch::Tensor val;
      std::tie(action_probs, val) = a2c_agent.predict_policy({state});

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
      total_reward += reward;
      actions += std::to_string(action);

      rewards.push_back(reward);
    }
    if (j == 0)
      std::cout << std::endl;

    Game registry_game;
    registry_game.states = states;
    registry_game.rewards = rewards;
    registry_game.mcts_actions = mcts_actions;
    registry->save_if_best(registry_game, total_reward);

    min_len = std::min(min_len, (int) actions.length());
    max_len = std::max(max_len, (int) actions.length());
    total_length += actions.length();
    total_length_sq += std::pow(actions.length(), 2);
    // std::cout << "EVAL " << actions << " " << actions.length() << std::endl;
  }
  double var = (total_length_sq - (std::pow(total_length, 2) / n_iters)) / n_iters;
  double avg = total_reward / n_iters;
  std::cout << "AVG " << avg <<  " |VAR " << var << " |MIN " << min_len << " |MAX " << max_len << std::endl;

  return std::make_tuple(total_length / n_iters, avg, var);
}

float schedule_alpha(
    json params,
    A2CLearner a2c_agent,
    LRScheduler *lr_scheduler,
    double total_reward,
    int n_episode
) {
  float lr = 0;
  if (params["optimizer_class"] == "adam") {
    // The decision to only use the first param group might be dubious.
    // Keep that in mind. For now it is fine, I checked.
    auto& options = static_cast<torch::optim::AdamOptions&>(
        a2c_agent.policy_optimizer->param_groups()[0].options()
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
    A2CLearner &a2c_agent,
    int n_episode,
    ReplayBuffer *replay_buffer,
    Registry *registry
) {
  int n_actors = params["n_actors"];
  int n_procs = params["n_procs"];
  std::vector<int> actor_lengths;

  bool do_warmup = params["do_warmup"];
  int n_warmup = params["n_warmup"];

  // Run self play games in n_procs parallel processes.
  auto pool = SimpleThreadPool(n_procs);
  auto lambda = [env, params, a2c_agent, n_episode, registry]() -> std::shared_ptr<Game> {
    return run_actor(env, params, a2c_agent, n_episode, registry);
  };
  std::vector<Task*> tasks;
  for (int i = 0; i < n_actors; ++i) {
    Task *task = new Task(lambda);
    pool.add_task(task);
    tasks.push_back(task);
  }
  if (do_warmup) {
    params["do_warmup"] = false;
    for (int i = 0; i < n_warmup; ++i) {
      Task *task = new Task(lambda);
      pool.add_task(task);
      tasks.push_back(task);
    }
  }

  // If using gradient bandits: Add one super greedy bandit.
  std::string bandit_type = params["bandit_type"];
  if (bandit_type == "grad") {
    auto lambda2 = [env, &params, a2c_agent, n_episode, registry]() -> std::shared_ptr<Game> {
      // 'true' denotes greedy.
      return run_actor(env, params, a2c_agent, n_episode, registry, true);
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
  A2CLearner a2c_agent,
  int n_episode,
  ReplayBuffer *replay_buffer,
  json &params,
  LRScheduler *lr_scheduler,
  Registry *registry,
  std::chrono::time_point<std::chrono::high_resolution_clock> start_time
) {
  a2c_agent.policy_net->eval();

  // Run self play games in n_procs parallel processes.
  std::vector<int> actor_lengths = run_actors(
      env, params, a2c_agent, n_episode, replay_buffer, registry
  );

  // Print debug information.
  int n_actors = params["n_actors"];
  std::string bandit_type = params["bandit_type"];
  if (bandit_type == "grad")
    n_actors += 1;
  std::cout << "REWARDS ";
  auto rewards = replay_buffer->get_rewards();
  int size_ = rewards.size() - n_actors - 1;
  for (int i = rewards.size() - 1; i > size_; --i) {
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

  bool debug_do_print = false;
  bool use_eps_greedy_learning = params["use_eps_greedy_learning"];
  float eps_greedy_epsilon_decay_factor = params["eps_greedy_epsilon_decay_factor_train"];

  float epsilon = std::pow(eps_greedy_epsilon_decay_factor, n_episode);

  a2c_agent.policy_net->train();
  for (int i = 0; i < train_steps; ++i) {
    std::shared_ptr<Game> game;

    bool greedy = false;
    if (use_eps_greedy_learning)
      greedy = epsgreedy_distribution(generator) > epsilon;

    if (greedy) {
      game = replay_buffer->get_best();
    } else {
      game = replay_buffer->sample();
    }

    auto loss = a2c_agent.update(game);

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

  a2c_agent.policy_net->eval();

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

std::shared_ptr<Game> run_actor(EnvWrapper orig_env, json params, A2CLearner a2c_agent, int n_episode, Registry *registry, bool greedy_bandit) {
  EnvWrapper env = *orig_env.clone();
  auto state = env.reset();

  int n_actions = params["n_actions"];

  // TODO Creating a new bandit here every time succs..
  std::string bandit_type = params["bandit_type"];
  Bandit *mcts_agent;
  if (bandit_type == "mcts") {
    mcts_agent = new MCTS(orig_env, a2c_agent, params);
  } else if (bandit_type == "grad") {
    mcts_agent = new GradientBanditSearch(orig_env, a2c_agent, params, registry, greedy_bandit);
  }

  // TODO Make sure this is a good idea to init here.
  std::random_device rd;
  std::mt19937 generator(rd());

  int horizon = params["horizon"];

  std::string mcts_actions = "";

  float eps_greedy_epsilon_decay_factor = params["eps_greedy_epsilon_decay_factor_actor"];
  float epsilon = std::pow(eps_greedy_epsilon_decay_factor, n_episode);
  bool greedy = false;
  std::uniform_real_distribution<> epsgreedy_distribution(0, 1);
  if (params["use_eps_greedy_learning"]) {
    greedy = epsgreedy_distribution(generator) > epsilon;
  }

  if (greedy_bandit)
    greedy = true;

  double total_reward = 0;
  bool done = false;
  std::shared_ptr<Game> game = std::make_shared<Game>();
  game->is_greedy = greedy;
  for (int i = 0; i < horizon; ++i) {
    auto mcts_action = mcts_agent->policy(i, env, state);

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
    std::tie(action_probs, std::ignore) = a2c_agent.predict_policy({state});

    // Allow following a2c at the beginning for exploration.
    // Following the bandit is how AlphaZero does it. Following a2c is how this paper does it:
    // `Thinking Fast and Slow with Deep Learning and Tree Search`.
    bool follow_bandit_greedily = true;
    if (params["follow_a2c"]) {
      epsilon = std::pow(epsilon, 2);
      follow_bandit_greedily = epsgreedy_distribution(generator) > epsilon;
      if (n_episode > 100) {
          epsilon = std::pow(epsilon, 4);
          follow_bandit_greedily = epsgreedy_distribution(generator) > epsilon;
          params["train_steps"] = 1000;
      }
    }

    // Here we follow a2c.
    if (!follow_bandit_greedily) {
      float* action_probs_arr = action_probs.data_ptr<float>();
      auto vec = std::vector<float>(action_probs_arr, action_probs_arr + n_actions);
      std::discrete_distribution<int> distribution_(vec.begin(), vec.end());
      sampled_action = distribution_(generator);
    }

    std::vector<float> next_state;
    double reward;
    std::tie(next_state, reward, done) = env.step(sampled_action);

    total_reward += reward;

    game->states.push_back(state);
    game->rewards.push_back(reward);
    game->mcts_actions.push_back(mcts_action);

    // TODO: Violating encapsulation - not great, not terrible (3.6)
    mcts_agent->history.rewards.push_back(reward);

    state = next_state;

    if (done)
      break;
  }

  registry->save_if_best(*game, total_reward);

  // std::cout << mcts_actions.size() << " " << std::flush;
  // std::cout << mcts_actions << std::endl;
  delete mcts_agent;
  return game;
}

std::tuple<int, int, double> run(EnvWrapper env, json params, int n_run, TensorBoardLogger &writer) {
  std::ostringstream oss;
  oss << params.dump();
  writer.add_text("Info/params/" + std::to_string(n_run), 0, oss.str().c_str());

  auto start_time = std::chrono::high_resolution_clock::now();
  ReplayBuffer *replay_buffer = new ReplayBuffer(
    params["memory_capacity"],
    params["prioritized_sampling"]
  );
  auto a2c_agent = A2CLearner(params, env);
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
    std::cout << "<game> can be 5x5, 8x8, 16x16, mtcar." << std::endl;
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
    }
  }

  TensorBoardLogger writer(gen_log_filename(game, param_num).c_str());

  for (int i = 0; i < params["n_runs"]; ++i)
    run(env, params, i, writer);
}
