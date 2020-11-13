#ifndef FUZZ_BANDIT_HEADER
#define FUZZ_BANDIT_HEADER
#include "env_wrapper.hpp"
#include "cfg.hpp"

void fuzz_bandit(json params, EnvWrapper orig_env);
void test_gaussian_bandit();
#endif
