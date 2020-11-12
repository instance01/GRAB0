#include "lunar_lander.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>


template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

LunarLanderEnv::LunarLanderEnv() {
  max_steps = 500;
  world = new b2World(b2Vec2(0, -9.8));
  continuous = true;

  // State is already normalized well enough.
  expected_mean = {0., 0., 0., 0., 0., 0., 0., 0.};
  expected_stddev = {1., 1., 1., 1., 1., 1., 1., 1.};
}

LunarLanderEnv::LunarLanderEnv(bool continuous) : LunarLanderEnv() {
  // TODO: continuous for now hardcoded to true and ignored.
}

void LunarLanderEnv::clone_body(b2Body* current, b2Body* other) {
  // TODO: Reconsider which things need to be set.
  auto transform = other->GetTransform();
  current->SetAwake(other->IsAwake());
  current->SetTransform(transform.p, transform.q.GetAngle());
  current->SetLinearVelocity(other->GetLinearVelocity());
  current->SetAngularVelocity(other->GetAngularVelocity());
  current->SetLinearDamping(other->GetLinearDamping());
  current->SetAngularDamping(other->GetAngularDamping());
  current->SetGravityScale(other->GetGravityScale());
  current->SetEnabled(other->IsEnabled());

  auto vel = other->GetLinearVelocity();
  current->SetLinearVelocity(vel);
  current->SetTransform(transform.p, transform.q.GetAngle());
  current->SetSleepTime(other->GetSleepTime());
}

LunarLanderEnv::LunarLanderEnv(LunarLanderEnv &other) {
  // There is a blog post regarding this copy constructor.
  // The gist is that bodies are cloned manually in clone_body() and a hack to
  // allow the to sleep is needed.
  auto generator = other.generator;
  world = new b2World(b2Vec2(0, -9.8));
  // We reset manually to get a clean initial world.
  _reset(generator);
  leg1 = _create_leg(0);
  leg2 = _create_leg(1);
  initialized = true;

  // Create the exact same moon as in the other environement.
  height = other.height;
  _create_moon();

  max_steps = other.max_steps;
  steps = other.steps;
  generator = other.generator;

  expected_stddev = other.expected_stddev;
  expected_mean = other.expected_mean;

  if (!other.initialized) {
    return;
  }

  helipad_y = other.helipad_y;
  continuous = other.continuous;
  prev_shaping = other.prev_shaping;
  game_over = other.game_over;
  leg1_ground_contact = other.leg1_ground_contact;
  leg2_ground_contact = other.leg2_ground_contact;

  clone_body(leg1, other.leg1);
  clone_body(leg2, other.leg2);
  clone_body(lander, other.lander);

  // TODO Any way to get rid of this weird hack? 2 iterations work too btw.
  // for (int i = 0; i < 3; ++i) {
  for (int i = 0; i < 10; ++i) {
    world->Step(1.0 / FPS, 6 * 30, 2 * 30);
    world->Step(1.0 / FPS, 6 * 30, 2 * 30);
    world->Step(1.0 / FPS, 6 * 30, 2 * 30);
    world->Step(1.0 / FPS, 6 * 30, 2 * 30);
    world->Step(1.0 / FPS, 6 * 30, 2 * 30);
    world->Step(1.0 / FPS, 6 * 30, 2 * 30);

    clone_body(leg1, other.leg1);
    clone_body(leg2, other.leg2);
    clone_body(lander, other.lander);
  }
}

void
LunarLanderEnv::destroy() {
  if (initialized) {
    world->DestroyBody(moon);
    world->DestroyBody(leg1);
    world->DestroyBody(leg2);
    world->DestroyBody(lander);
    delete listener;
  }
}

LunarLanderEnv::~LunarLanderEnv() {
  destroy();
  delete world;
}

void
ContactDetector::BeginContact(b2Contact* contact) {
  if (env->lander == contact->GetFixtureA()->GetBody() || env->lander == contact->GetFixtureB()->GetBody()) {
    env->game_over = true;
  }
  if (env->leg1 == contact->GetFixtureA()->GetBody() || env->leg1 == contact->GetFixtureB()->GetBody()) {
    env->leg1_ground_contact = true;
  }
  if (env->leg2 == contact->GetFixtureA()->GetBody() || env->leg2 == contact->GetFixtureB()->GetBody()) {
    env->leg2_ground_contact = true;
  }
}

void
ContactDetector::EndContact(b2Contact* contact) {
  if (env->leg1 == contact->GetFixtureA()->GetBody() || env->leg1 == contact->GetFixtureB()->GetBody()) {
    env->leg1_ground_contact = false;
  }
  if (env->leg2 == contact->GetFixtureA()->GetBody() || env->leg2 == contact->GetFixtureB()->GetBody()) {
    env->leg2_ground_contact = false;
  }
}

b2Body*
LunarLanderEnv::_create_leg(int x) {
  // x shall be 0 or 1.
  int i = -1;
  if (x == 1)
    i = 1;

  float initial_y = VIEWPORT_H / SCALE;

  b2BodyDef bodyDef;
  bodyDef.type = b2_dynamicBody;
  bodyDef.position.Set(VIEWPORT_W / SCALE / 2 - i * LEG_AWAY / SCALE, initial_y);
  bodyDef.angle = i * 0.05;
  auto leg = world->CreateBody(&bodyDef);

  b2PolygonShape polygonShape;
  polygonShape.SetAsBox(LEG_W / SCALE, LEG_H / SCALE);
  b2FixtureDef fixtureDef_;
  fixtureDef_.shape = &polygonShape;
  fixtureDef_.density = 1.0;
  b2Filter filter;
  filter.categoryBits = 0x0020;
  filter.maskBits = 0x001;
  fixtureDef_.filter = filter;
  fixtureDef_.restitution = 0;
  leg->CreateFixture(&fixtureDef_);

  auto joint = b2RevoluteJointDef();
  joint.Initialize(lander, leg, leg->GetPosition()); //b2Vec2(0, 0));
  joint.localAnchorA = b2Vec2(0, 0);
  joint.localAnchorB = b2Vec2(i * LEG_AWAY / SCALE, LEG_DOWN / SCALE);
  joint.enableMotor = true;
  joint.enableLimit = true;
  joint.maxMotorTorque = LEG_SPRING_TORQUE;
  joint.motorSpeed = +0.3 * i;

  if (i == -1) {
    joint.lowerAngle = +0.9 - 0.5;
    joint.upperAngle = +0.9;
  } else {
    joint.lowerAngle = -0.9;
    joint.upperAngle = -0.9 + 0.5;
  }

  world->CreateJoint(&joint);

  return leg;
}

void
LunarLanderEnv::_create_moon() {
  std::vector<float> chunk_x;
  for (int i = 0; i < CHUNKS; ++i)
    chunk_x.push_back(W / (CHUNKS-1)*i);
  float helipad_x1 = chunk_x[CHUNKS/2-1];
  float helipad_x2 = chunk_x[CHUNKS/2+1];
  helipad_y = H / 4;
  height[CHUNKS / 2 - 2] = helipad_y;
  height[CHUNKS / 2 - 1] = helipad_y;
  height[CHUNKS / 2 + 0] = helipad_y;
  height[CHUNKS / 2 + 1] = helipad_y;
  height[CHUNKS / 2 + 2] = helipad_y;
  std::vector<float> smooth_y;
  smooth_y.push_back(0.33*(height[CHUNKS] + height[0] + height[1]));
  for (int i = 1; i < CHUNKS; ++i) {
    smooth_y.push_back(0.33*(height[i-1] + height[i+0] + height[i+1]));
  }

  b2EdgeShape edgeShape;
  edgeShape.SetTwoSided(b2Vec2(0, 0), b2Vec2(W, 0));
  b2BodyDef bodyDef;
  bodyDef.type = b2_staticBody;
  moon = world->CreateBody(&bodyDef);
  moon->CreateFixture(&edgeShape, 0);
  for (int i = 0; i < CHUNKS - 1; ++i) {
    b2Vec2 p1 = b2Vec2(chunk_x[i], smooth_y[i]);
    b2Vec2 p2 = b2Vec2(chunk_x[i+1], smooth_y[i+1]);

    b2EdgeShape edgeShape_;
    edgeShape_.SetTwoSided(p1, p2);
    b2FixtureDef fixtureDef;
    fixtureDef.shape = &edgeShape_;
    fixtureDef.density = 0;
    fixtureDef.friction = 0.1;
    moon->CreateFixture(&fixtureDef);
  }
}

void
LunarLanderEnv::_reset(std::mt19937 &generator_) {
  steps = 0;
  generator = generator_;
  // TODO: Here's a small leak. We never delete this and also when resetting multiple times we lose the pointer.
  listener = new ContactDetector(this);
  world->SetContactListener(listener);
  game_over = false;
  prev_shaping = 0.0f;
  leg1_ground_contact = false;
  leg2_ground_contact = false;

  // height is generated for the moon which is created in _create_moon afterwards.
  // This makes it possible to override height (used in copy constructor).
  std::uniform_real_distribution<> uniform_distr(0, H / 2);
  height = std::vector<float>(CHUNKS + 1);
  std::generate(std::begin(height), std::end(height), [&]{ return uniform_distr(generator); });

  float initial_y = VIEWPORT_H / SCALE;

  // TODO just simplyfing for debugging:
  //initial_y = 5.0;

  b2BodyDef bodyDef_;
  bodyDef_.type = b2_dynamicBody;
  bodyDef_.position.Set(VIEWPORT_W / SCALE / 2, initial_y);
  bodyDef_.angle = 0.0;
  lander = world->CreateBody(&bodyDef_);

  b2PolygonShape polygonShape;
  polygonShape.Set(LANDER_POLY, 6);
  b2FixtureDef fixtureDef_;
  fixtureDef_.shape = &polygonShape;
  fixtureDef_.density = 5.0;
  fixtureDef_.friction = 0.1;
  b2Filter filter;
  filter.categoryBits = 0x0010;
  filter.maskBits = 0x001;
  fixtureDef_.filter = filter;
  fixtureDef_.restitution = 0;
  lander->CreateFixture(&fixtureDef_);
}

std::vector<float>
LunarLanderEnv::reset(std::mt19937 &generator_) {
  if (initialized) {
    destroy();
  }
  _reset(generator_);
  _create_moon();

  // TODO just simplyfing for debugging:
  //INITIAL_RANDOM = 100.0f;
  std::uniform_real_distribution<> uniform_distr_(-INITIAL_RANDOM, INITIAL_RANDOM);
  lander->ApplyForceToCenter(b2Vec2(uniform_distr_(generator), uniform_distr_(generator)), true);

  leg1 = _create_leg(0);
  leg2 = _create_leg(1);

  std::vector<float> action = {0.0, 0.0};
  std::vector<float> obs;
  std::tie(obs, std::ignore, std::ignore) = step(action);

  initialized = true;

  return obs;
}

std::tuple<std::vector<float>, double, bool>
LunarLanderEnv::step(std::vector<float> &action) {
  steps += 1;
  if (steps > max_steps)
    game_over = true;

  float main_engine_action = action[0];
  float left_right_engine_action = action[1];

  main_engine_action = std::clamp(main_engine_action, (float) -1.0, (float) 1.0);
  left_right_engine_action = std::clamp(left_right_engine_action, (float) -1.0, (float) 1.0);

  float tip0 = std::sin(lander->GetAngle());
  float tip1 = std::cos(lander->GetAngle());
  float side0 = -tip1;
  float side1 = tip0;
  std::uniform_real_distribution<> uniform_distr(-1, 1);
  float dispersion0 = uniform_distr(generator) / SCALE;
  float dispersion1 = uniform_distr(generator) / SCALE;

  float m_power = 0.0;
  if (main_engine_action > 0.0) {
    m_power = (std::clamp(main_engine_action, (float) 0.0, (float) 1.0) + 1.0) * 0.5;
    // TODO
    // assert m_power >= 0.5 and m_power <= 1.0

    float ox = tip0 * (4 / SCALE + 2 * dispersion0) + side0 * dispersion1;
    float oy = -tip1 * (4 / SCALE + 2 * dispersion0) - side1 * dispersion1;
    b2Vec2 impulse_pos = b2Vec2(lander->GetPosition().x + ox, lander->GetPosition().y + oy);

    lander->ApplyLinearImpulse(
        b2Vec2(-ox * MAIN_ENGINE_POWER * m_power, -oy * MAIN_ENGINE_POWER * m_power),
        impulse_pos,
        true
    );
  }

  float s_power = 0.0;
  if (std::abs(left_right_engine_action) > 0.5) {
    int direction = sgn<float>(left_right_engine_action);
    float s_power = std::clamp(std::abs(left_right_engine_action), (float) 0.5, (float) 1.0);
    // TODO
    // assert s_power >= 0.5 and s_power <= 1.0
    float ox = tip0 * dispersion0 + side0 * (3 * dispersion1 + direction * SIDE_ENGINE_AWAY / SCALE);
    float oy = -tip1 * dispersion0 - side1 * (3 * dispersion1 + direction * SIDE_ENGINE_AWAY / SCALE);
    b2Vec2 impulse_pos = b2Vec2(
        lander->GetPosition().x + ox - tip0 * 17 / SCALE,
        lander->GetPosition().y + oy + tip1 * SIDE_ENGINE_HEIGHT / SCALE
    );
    lander->ApplyLinearImpulse(
        b2Vec2(-ox * SIDE_ENGINE_POWER * s_power, -oy * SIDE_ENGINE_POWER * s_power),
        impulse_pos,
        true
    );
  }

  world->Step(1.0 / FPS, 6 * 30, 2 * 30);

  auto pos = lander->GetPosition();
  auto vel = lander->GetLinearVelocity();
  std::vector<float> state {
      (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
      (pos.y - (helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2),
      vel.x * (VIEWPORT_W / SCALE / 2) / FPS,
      vel.y * (VIEWPORT_H / SCALE / 2) / FPS,
      lander->GetAngle(),
      (float) 20.0 * lander->GetAngularVelocity() / FPS,
      (float) leg1_ground_contact,
      (float) leg2_ground_contact
  };

  float reward = 0;
  float shaping = -100 * std::sqrt(state[0] * state[0] + state[1] * state[1]);
  shaping -= 100 * std::sqrt(state[2] * state[2] + state[3] * state[3]);
  shaping -= 100 * std::abs(state[4]) + 10 * state[6] + 10 * state[7];

  if (prev_shaping != 0.0) {
    reward = shaping - prev_shaping;
  }
  prev_shaping = shaping;

  reward -= m_power * 0.30;
  reward -= s_power * 0.03;

  bool done = false;
  if (game_over || std::abs(state[0]) >= 1.0) {
    done = true;
    reward = -100;
  }
  if (!(lander->IsAwake())) {
    std::cout << "## WIN ##" << std::endl;
    done = true;
    reward = +100;
  }

  auto ret = std::make_tuple(state, reward, done);
  return ret;
}

std::vector<float> heuristic(std::vector<float> state) {
  float angle_targ = state[0] * 0.5 + state[2] * 1.0;
  if (angle_targ > 0.4) angle_targ = 0.4;
  if (angle_targ < -0.4) angle_targ = -0.4;
  float hover_targ = 0.55 * std::abs(state[0]);

  float angle_todo = (angle_targ - state[4]) * 0.5 - (state[5]) * 1.0;
  float hover_todo = (hover_targ - state[1]) * 0.5 - (state[3]) * 0.5;

  if (state[6] || state[7]) {
    angle_todo = 0;
    hover_todo = -(state[3]) * 0.5;
  }

  float a1 = hover_todo * 20 - 1;
  float a2 = -angle_todo * 20;
  a1 = std::clamp(a1, (float) -1.0, (float) 1.0);
  a2 = std::clamp(a2, (float) -1.0, (float) 1.0);
  std::vector<float> a = {a1, a2};
  return a;
}

float demo_heuristic_lander() {
  // Used for testing the environment and whether cloning works.
  std::random_device rd;
  std::mt19937 generator(rd());

  LunarLanderEnv* env = new LunarLanderEnv(true);
  float total_reward = 0;
  int steps = 0;
  auto s = env->reset(generator);
  int i = 0;
  while (true) {
    auto a = heuristic(s);
    i += 1;
    if (i % 1 == 0 && i < 250) {
      auto env_ = new LunarLanderEnv(*env);

      // std::vector<float> s2;
      // std::tie(s2, std::ignore, std::ignore) = env->step(a);

      // std::cout << "state! " << s2[0] << " " << s2[1] << " " << s2[2] << " "<< s2[3];
      // std::cout << " " << s2[4] << " " << s2[5] << " " << s2[6] << " " << s2[7] << " ";
      // std::cout << " : " << env->lander->GetLinearVelocity().x << " " << env->lander->GetLinearVelocity().y;
      // std::cout << " : " << a[0] << " " << a[1];
      // std::cout << std::endl;

      env = env_;
    }
    float r;
    bool done;
    std::tie(s, r, done) = env->step(a);
    total_reward += r;

    // std::cout << "state- " << s[0] << " " << s[1] << " " << s[2] << " "<< s[3];
    // std::cout << " " << s[4] << " " << s[5] << " " << s[6] << " " << s[7] << " ";
    // std::cout << " : " << env->lander->GetLinearVelocity().x << " " << env->lander->GetLinearVelocity().y;
    // std::cout << " : " << a[0] << " " << a[1];
    // std::cout << std::endl;

    auto pos = env->lander->GetPosition();
    std::cout << "POS " << pos.x << " " << pos.y << std::endl;
    if (steps % 50 == 0 || done) {
      std::cout << "|> step " << steps << " |> total_reward " << total_reward << std::endl;
    }
    steps += 1;
    if (steps > 500 || done)
      break;
  }
  return total_reward;
}
