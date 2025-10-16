#include <iostream>
#include <cmath>

#include "SpringMass.hpp"
#include "SpringDamperMass.hpp"

using std::cout;
using std::endl;

// Return 0 if the two values are equal, else return 1
template<typename T = double>
int check_value(T expectedValue, T value) {
  if (round(value) == round(expectedValue))
    return 0;
  
  cout << "Wrong result: (Expected value: " << expectedValue
       << " != " << value << ")" << endl;
  return 1;
}

int main (int argc, char *argv[]) {
  // Test case 1: Spring mass system
  {
    SpringMass mass(200., 0., 161., 0.);
    int t;

    do {
      t = mass.step();
    } while (t < 2000 && t > 0);

    // Get the current state at time t
    Vec2d s;
    mass.getConfiguration(t, s);
    // cout << "Current state: " << s.x << " " << s.y << endl;

    double expected[] = {141., -19.};
    if (check_value(2000, mass.getCurrentSimulationTime()))
      return 1;
    if (check_value(expected[0], floor(s.x)))
      return 1;
    if (check_value(expected[1], floor(s.y)))
      return 1;
    cout << "Test 1 passed." << endl;
  }

  // Test case 2: Spring mass damper system
  {
    SpringDamperMass mass(200., 0., 161., 0., 1.);
    int t;

    do {
      t = mass.step();
    } while (t < 2000 && t > 0);

    // Get the current state at time t
    Vec2d s1, s2;
    mass.getConfiguration(10, s1);
    mass.getConfiguration(t, s2);

    double expected1[] = {174., 16.};
    double expected2[] = {161., 0.};
    if (check_value(expected1[0], floor(s1.x)))
      return 1;
    if (check_value(expected1[1], floor(s1.y)))
      return 1;
    if (check_value(expected2[0], round(s2.x)))
      return 1;
    if (check_value(expected2[1], round(s2.y)))
      return 1;
    cout << "Test 2 passed." << endl;
  }

  cout << "All tests passed (7 assertions in 2 test cases)." << endl;
  return 0;
}
