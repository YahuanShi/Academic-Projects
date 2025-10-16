// This is an header guard it prevents
// the header from being inserted more than one time
#ifndef SpringMass__H__
#define SpringMass__H__

#include <vector>

/**
 * A two-dimensional array structure
 * 
 * By convention, x denotes the position of the object and y denotes the velocity.
 */
struct Vec2d {
  double x;
  double y;

  // TODO define members, methods, constructors etc.
};

/**
 * @brief SpringMass simulation class
 * 
 * Used to simulate the trajectory of a spring mass system,
 * spawned at t=0 at some initial position with some
 * initial velocity.
 */
class SpringMass {
  
public:
  
  /**
   * @brief Constructor for the SpringMass object from initial position
   * and velocity of the object, position and velocity when the spring
   * is unstretched (equilibrium state)
   */
  SpringMass(double pos_init, double vel_init, double pos_eqm, double vel_eqm);
  
  /**
   * @brief Destructor for the SpringMass object
   */
  virtual ~SpringMass();
  
  /**
   * @brief Runs a step in the spring mass simulation
   * 
   * @return last time step t 
   */
  virtual int step();

  /**
   * @brief Get the position and velocity of the object at time t.
   * 
   * Constaints: t must be the current time step, or a time step
   * that has already passed. If these constraints are fullfilled,
   * the method returns true and stores position and velocity
   * in a Vec2d struct, otherwise the methods returns false.
   */
  virtual bool getConfiguration(int t, Vec2d& state) const;
  
  /**
   * @brief Return the current simulation time t
   */
  virtual int getCurrentSimulationTime() const;
  
  
protected:
  /**
   * @brief Given constants - use this for your simulation
   */
  static const double GRAVITY;
  static const double SPRING_CONST;
  static const double MASS;

  // TODO define your methods and variables here

};

#endif // SpringMass__H__
