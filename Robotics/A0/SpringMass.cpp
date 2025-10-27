#include "SpringMass.hpp"

// define gravity constant
const double SpringMass::GRAVITY = 10;
const double SpringMass::SPRING_CONST = 7;
const double SpringMass::MASS = 30;

static inline Vec2d make_vec(double x, double y) {
  Vec2d v; v.x = x; v.y = y; return v;
}
// TODO SpringMass constructor
SpringMass::SpringMass(double pos_init, double vel_init,
                       double pos_eqm, double vel_eqm)
{
  m_pos_init = pos_init;
  m_vel_init = vel_init;
  m_pos_eqm  = pos_eqm;
  m_vel_eqm  = vel_eqm;

  m_state = Vec2d(m_pos_init, m_vel_init);
  current_time = 0;
  state_list.clear();
  state_list.push_back(m_state);
}
SpringMass::~SpringMass() {}

// TODO SpringMass simulation step
int SpringMass::step() {
  double k = SPRING_CONST;
  double m = MASS;

  double v_next = m_state.y - (k / m) * (m_state.x - m_pos_eqm);
  double x_next = m_state.x + v_next;

  m_state.x = x_next;
  m_state.y = v_next;

  ++current_time;
  state_list.push_back(m_state);
  return current_time;
}

// TODO SpringMass configuration getter
bool SpringMass::getConfiguration(int t, Vec2d& state) const {
  if (t < 0 || t > current_time)
    return false;
  state = state_list[t];
  return true;
}

// TODO SpringMass current simulation time getter
int SpringMass::getCurrentSimulationTime() const {
  return current_time;
}