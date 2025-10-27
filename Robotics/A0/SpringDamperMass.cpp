#include "SpringDamperMass.hpp"

// TODO Define your methods here
SpringDamperMass::SpringDamperMass(double pos_init, double vel_init,
                                   double pos_eqm, double vel_eqm,
                                   double _damping_coeff)
: SpringMass(pos_init, vel_init, pos_eqm, vel_eqm),
  damping_coeff(_damping_coeff)
{}

int SpringDamperMass::step() {
  double k = SPRING_CONST;
  double m = MASS;
  double b = damping_coeff;

  double v_next = m_state.y - (b / m) * m_state.y - (k / m) * (m_state.x - m_pos_eqm);
  double x_next = m_state.x + v_next;

  m_state.x = x_next;
  m_state.y = v_next;

  ++current_time;
  state_list.push_back(m_state);
  return current_time;
}