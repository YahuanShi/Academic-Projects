#ifndef SPRING_DAMPER_MASS__H__
#define SPRING_DAMPER_MASS__H__

#include "SpringMass.hpp"

class SpringDamperMass : public SpringMass {
public: 

  /**
   * @brief Initialize object initial position and velocity,  
   * object position and velocity when the spring is unstretched (equilibrium state)
   * and the damping coefficient
   */
  SpringDamperMass(double pos_init, double vel_init,
                   double pos_eqm, double vel_eqm,
                   double _damping_coeff);

  // TODO define your methods here

private:
  /**
   * Damping coefficient for damper
   */ 
  double damping_coeff;
  
  // TODO define additional members (if necessary) here
};


#endif  // SPRING_DAMPER_MASS__H__
