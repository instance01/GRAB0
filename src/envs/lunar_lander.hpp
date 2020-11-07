#ifndef LUNARLANDER_HEADER
#define LUNARLANDER_HEADER
#include <random>
#include <tuple>
#include <utility>
#include <vector>
#include <set>
#include "env.hpp"
#include "box2d/box2d.h"


class LunarLanderEnv;

class ContactDetector : public b2ContactListener {
  public:
    LunarLanderEnv* env;

    ContactDetector(LunarLanderEnv* env) : env(env) { };
    ContactDetector() {};
    ~ContactDetector() {};

    void BeginContact(b2Contact* contact);
    void EndContact(b2Contact* contact);
};


class LunarLanderEnv : public Env {
  public:
    float FPS = 50.0;
    float SCALE = 30.0;
    float MAIN_ENGINE_POWER = 13.0;
    float SIDE_ENGINE_POWER = 0.6;
    float INITIAL_RANDOM = 1000.0;
    b2Vec2 LANDER_POLY[6] = {
        {-14 / SCALE, +17 / SCALE}, {-17 / SCALE, 0}, {-17 / SCALE, -10 / SCALE},
        {+17 / SCALE, -10 / SCALE}, {+17 / SCALE, 0}, {+14 / SCALE, +17 / SCALE}
    };
    int LEG_AWAY = 20;
    int LEG_DOWN = 18;
    int LEG_W = 2;
    int LEG_H = 8;
    int LEG_SPRING_TORQUE = 40;
    float SIDE_ENGINE_HEIGHT = 14.0;
    float SIDE_ENGINE_AWAY = 12.0;
    int VIEWPORT_W = 600;
    int VIEWPORT_H = 400;
    int CHUNKS = 11;
    float W = VIEWPORT_W / SCALE;
    float H = VIEWPORT_H / SCALE;

    ContactDetector* listener;

    bool initialized = false;
    std::mt19937 generator;
    bool continuous = true;
    int max_steps = 1000;  // This is convenience, for faster training.
    int steps = 0;

    float helipad_y;
    std::vector<float> height;
    b2World* world;
    b2Body* moon;
    b2Body* lander;
    b2Body* leg1;
    b2Body* leg2;
    float prev_shaping = 0.0f;
    bool game_over = false;
    bool leg1_ground_contact = false;
    bool leg2_ground_contact = false;

    LunarLanderEnv();
    LunarLanderEnv(bool continuous);
    ~LunarLanderEnv();
    LunarLanderEnv(LunarLanderEnv &other);

    b2Body* _create_leg(int x);
    void destroy();
    void _create_moon();
    void _reset(std::mt19937 &generator);
    std::vector<float> reset(std::mt19937 &generator);
    std::tuple<std::vector<float>, double, bool> step(std::vector<float> action);

    void clone_body(b2Body* current, b2Body* other);
};

float demo_heuristic_lander();
#endif
