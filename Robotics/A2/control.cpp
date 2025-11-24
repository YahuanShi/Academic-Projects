// controlDLL.cpp : Defines the entry point for the DLL application.

#include "servo.h"
#include "param.h"
#include "control.h"
// #include "UiAgent.h"
#include "PrVector.h"
#include "PrMatrix.h"
#include "Utils.h" // misc. utility functions, such as toRad, toDeg, etc.
#include <math.h>
#include <algorithm>
using std::min;
using std::max;


struct CubicSpline {
  double t0 , tf ;
  PrVector a0 , a1 , a2 , a3 ;
};

CubicSpline spline;
bool control_done = true;

// *******************************************************************
// DOF Mapping Helpers
// *******************************************************************

double computeTf(GlobalVariables& gv)
{
  std::vector<float> tf_vec;
  double tf_rel;
  PrVector delta_q = gv.qd - gv.q;
  spline.t0 = gv.curTime;
  spline.a0 = gv.q;
  spline.a1 = PrVector(3);
  // calc tf for every q and then take the max
  for (int i = 0; i < 3; i++) {
    float tf_1 = abs((2/(3*gv.dqmax[i])) * (gv.qd[i] - gv.q[i]));
    float tf_2 = sqrt(abs((6/gv.ddqmax[i]) * (gv.qd[i] - gv.q[i])));
    tf_vec.push_back(std::max(tf_1, tf_2));
  }
  tf_rel = *(std::max_element(tf_vec.begin(), tf_vec.end()));
  spline.tf = tf_rel + spline.t0;
  spline.a2 = (3/pow(tf_rel,2)) * delta_q;
  spline.a3 = -(2/pow(tf_rel,3)) * delta_q;

  return spline.tf;
}

void PrintDebug(GlobalVariables& gv);

// *******************************************************************
// Initialization functions
// *******************************************************************

void InitControl(GlobalVariables& gv) 
{
  // This code runs before the first servo loop
}

void PreprocessControl(GlobalVariables& gv)
{
  // This code runs on every servo loop, just before the control law
   
  if ((gv.dof == 3) || (gv.dof == 6)) {
    // get the correct joint angles depending on the current mode:
    double q1,q2,q3;
    if (gv.dof == 3) {
      q1 = gv.q[0];
      q2 = gv.q[1];
      q3 = gv.q[2];
    } else if (gv.dof == 6) {
      q1 = gv.q[1];
      q2 = gv.q[2];
      q3 = gv.q[4];
    }

    // Variable that holds the torque exerted by gravity for each joint
    PrVector3 g123 = PrVector3(0,0,0);

    // Compute g123 here!
    float r1 = R2;
    float r2 = 0.189738;
    float r3 = R6;
    float l1 = L2;
    float l2 = L3;
    float l3 = L6;
    float m1 = M2;
    float m2 = M3 + M4 + M5;
    float m3 = M6;
    float g = -9.81;

    float c1 = cos(q1);
    float s12 = sin(q1 + q2);
    float s123 = sin(q1 + q2 + q3);

    g123[0] = g * (m1 * r1 * c1 + m2 * (l1 * c1 + r2 * s12) + m3 * (l1 * c1 + l2 * s12 + r3 * s123));
    g123[1] = g * (m2 * r2 * s12 + m3 * (l2 * s12 + r3 * s123));
    g123[2] = g * (m3 * r3 * s123);

    // maps the torques to the right joint indices depending on the current mode:
  
}}

void PostprocessControl(GlobalVariables& gv) 
{
  // This code runs on every servo loop, just after the control law
}

void initFloatControl(GlobalVariables& gv) 
{
  // Control Initialization Code Here
}

void initOpenControl(GlobalVariables& gv) 
{
  // Control Initialization Code Here
}

void initNjholdControl(GlobalVariables& gv) 
{
  // Control Initialization Code Here
}

void initJholdControl(GlobalVariables& gv) 
{
  // Control Initialization Code Here
}

void initNjmoveControl(GlobalVariables& gv) 
{
  // Control Initialization Code Here
}

void initJmoveControl(GlobalVariables& gv) 
{
  // Control Initialization Code Here
}

void initNjgotoControl(GlobalVariables& gv) 
{
  // Control Initialization Code Here
} 

void initJgotoControl(GlobalVariables& gv) 
{
  // Control Initialization Code Here
}

void initNjtrackControl(GlobalVariables& gv)
{
  computeTf(gv);
}

void initJtrackControl(GlobalVariables& gv) 
{
  // Control Initialization Code Here
}

void initNxtrackControl(GlobalVariables& gv) 
{
  // Control Initialization Code Here
}

void initXtrackControl(GlobalVariables& gv) 
{
  // Control Initialization Code Here
} 

void initNholdControl(GlobalVariables& gv) 
{
  // Control Initialization Code Here
}

void initHoldControl(GlobalVariables& gv) 
{
  // Control Initialization Code Here
}

void initNgotoControl(GlobalVariables& gv) 
{
  // Control Initialization Code Here
} 

void initGotoControl(GlobalVariables& gv) 
{
  // Control Initialization Code Here
} 

void initNtrackControl(GlobalVariables& gv) 
{
  // Control Initialization Code Here
}

void initTrackControl(GlobalVariables& gv) 
{
  // Control Initialization Code Here
} 

void initPfmoveControl(GlobalVariables& gv) 
{
  // Control Initialization Code Here
} 

void initLineControl(GlobalVariables& gv) 
{
  // Control Initialization Code Here
}

void initProj1Control(GlobalVariables& gv) 
{
  PrVector qd = gv.qd;
  qd[0] = 0.096;
  qd[1] = 0.967;
  qd[2] = -1.016;
  gv.qd = qd;
  initNjtrackControl(gv);
}

void initProj2Control(GlobalVariables& gv) 
{
  spline.t0 = gv.curTime;
}

void initProj3Control(GlobalVariables& gv) 
{
  spline.t0 = gv.curTime;
}


// *******************************************************************
// Control laws
// *******************************************************************

void noControl(GlobalVariables& gv)
{
}

void floatControl(GlobalVariables& gv)
{
  gv.tau = gv.G;
  // this only works on the real robot unless the function is changed to use cout
  // the handed in solution must not contain any printouts
  // PrintDebug(gv);
}

void openControl(GlobalVariables& gv)
{
  floatControl(gv);  // Remove this line when you implement this controller
}

void njholdControl(GlobalVariables& gv) 
{
  floatControl(gv);  // Remove this line when you implement this controller
}

void jholdControl(GlobalVariables& gv) 
{
  floatControl(gv);  // Remove this line when you implement this controller
}

void njmoveControl(GlobalVariables& gv)
{
  gv.tau = gv.kp * (gv.qd - gv.q);  //P-controller
}

void jmoveControl(GlobalVariables& gv)
{
  floatControl(gv);  // Remove this line when you implement this controller
}

void njgotoControl(GlobalVariables& gv) 
{
  gv.tau = gv.kp * (gv.qd - gv.q) + gv.G; //P-controller with gravity compensation
}

void jgotoControl(GlobalVariables& gv) 
{
  gv.tau = gv.kp * (gv.qd - gv.q) + gv.G - gv.kv * gv.dq; //PD-controller with gravity compensation
}

void njtrackControl(GlobalVariables& gv) 
{
  if (gv.curTime > spline.tf) {
    floatControl(gv);
    return;
  }
  double t = gv.curTime - spline.t0;
  PrVector qd  = spline.a0 + spline.a1*t + spline.a2*t*t + spline.a3*t*t*t;
  PrVector dqd = spline.a1 + 2*spline.a2*t + 3*spline.a3*t*t;
  gv.tau = -gv.kp*(gv.q-qd) - gv.kv*(gv.dq-dqd) + gv.G;
}

void jtrackControl(GlobalVariables& gv)
{
  floatControl(gv);  // Remove this line when you implement this controller
}

void nxtrackControl(GlobalVariables& gv) 
{
  floatControl(gv);  // Remove this line when you implement this controller
}

void xtrackControl(GlobalVariables& gv) 
{
  floatControl(gv);  // Remove this line when you implement this controller
}

void nholdControl(GlobalVariables& gv) 
{
  floatControl(gv);  // Remove this line when you implement this controller
}

void holdControl(GlobalVariables& gv) 
{
  floatControl(gv);  // Remove this line when you implement this controller
}

void ngotoControl(GlobalVariables& gv) 
{
  floatControl(gv);  // Remove this line when you implement this controller
}

void gotoControl(GlobalVariables& gv) 
{
  floatControl(gv);  // Remove this line when you implement this controller
}

void ntrackControl(GlobalVariables& gv) 
{
  floatControl(gv);  // Remove this line when you implement this controller
}

void trackControl(GlobalVariables& gv) 
{
  floatControl(gv);  // Remove this line when you implement this controller
}

void pfmoveControl(GlobalVariables& gv) 
{
  floatControl(gv);  // Remove this line when you implement this controller
}

void lineControl(GlobalVariables& gv)
{
  floatControl(gv);  // Remove this line when you implement this controller
}

void proj1Control(GlobalVariables& gv) 
{
  njtrackControl(gv);
  return;
}

void proj2Control(GlobalVariables& gv) 
{
  //-- End-effector Trajectory --//

  // Circle Parameters
  double radius = 0.2;
  double x_center[2] = {0.6, 0.35};

  // Trajectory Time
  double t = gv.curTime - spline.t0;

  // Velocity Magnitude
  double vel = (2.0 * M_PI) / 5.0;

  // Rotation to End-Effector Space
  gv.xd[0] =  x_center[0] + radius * cos(vel * t);
  gv.xd[1] =  x_center[1] + radius * sin(vel * t);
  gv.xd[2] =  0.0;

  // The derivative gives us the velocity component
  gv.dxd[0] = -radius * vel * sin(vel * t);
  gv.dxd[1] =  radius * vel * cos(vel * t);
  gv.dxd[2] =  0.0;

  //-- Operational Space Controller (OSC) --//
  
  // Control Law (PD)
  PrVector endeff_F = -gv.kp * (gv.x - gv.xd) - gv.kv * (gv.dx - gv.dxd);
  
  // Jacobian
  gv.Jtranspose.multiply(endeff_F, gv.tau);

  // Gravity Compensation
  gv.tau -= gv.G;
}

void proj3Control(GlobalVariables& gv) 
{
  //-- End-effector Trajectory --//

  // Circle Parameters
  double radius = 0.2;
  double x_center[2] = {0.6, 0.35};
  double circle_iters = 3.0;

  // Velocity and Acceleration Magnitudes
  double vel_max = (2.0 * M_PI) / 5.0;
  double acc_max = (2.0 * M_PI) / 25.0;

  // Trajectory Times and Distances
  double total_traj_len = 2 * M_PI * circle_iters;

  double t = gv.curTime - spline.t0;
  double t_blend = vel_max / acc_max;  // Time to reach vel_max
  
  double blend_dist = acc_max * pow(t_blend, 2);  // Total blend distance for acceleration and deceleration phases
  double const_dist = total_traj_len - blend_dist;

  double t_const = const_dist / vel_max;
  double total_time = 2 * t_blend + t_const;
  
  // Trajectory Phase Identification
  double target_pos, target_vel;
  if (t < t_blend)  // Acceleration Phase (parabolic)
  {
    target_vel = acc_max * t;
    target_pos = 0.5 * acc_max * pow(t, 2);
  }
  else if (t < total_time - t_blend)  // Constant Velocity Phase (linear)
  {
    target_vel = vel_max;
    target_pos = 0.5 * acc_max * pow(t_blend, 2) + vel_max * (t - t_blend);
  }
  else if (t < total_time)  // Deceleration Phase (parabolic)
  {
    target_vel = vel_max - acc_max * (t - (total_time - t_blend));
    target_pos = total_traj_len - 0.5 * acc_max * pow(total_time - t, 2);
  }
  else
  {
    target_vel = 0.0;
    target_pos = total_traj_len;
  }

  // Rotation to End-Effector Space
  gv.xd[0] =  x_center[0] + radius * cos(target_pos);
  gv.xd[1] =  x_center[1] + radius * sin(target_pos);
  gv.xd[2] =  0.0;

  gv.dxd[0] = -radius * target_vel * sin(target_pos);
  gv.dxd[1] =  radius * target_vel * cos(target_pos);
  gv.dxd[2] =  0.0;

  // Control Law (PD)
  PrVector endeff_F = -gv.kp * (gv.x - gv.xd) - gv.kv * (gv.dx - gv.dxd);
  
  // Jacobian
  gv.Jtranspose.multiply(endeff_F, gv.tau);

  // Gravity Compensation
  gv.tau -= gv.G;
}

// *******************************************************************
// Debug function
// *******************************************************************

void PrintDebug(GlobalVariables& gv)
{
  // Replace this code with any debug information you'd like to get
  // when you type "pdebug" at the prompt.
  printf( "This sample code prints the torque and mass\n" );
  gv.tau.display( "tau" );
  gv.A.display( "A" );
}

#ifdef WIN32
// *******************************************************************
// XPrintf(): Replacement for printf() which calls ui->VDisplay()
// whenever the ui object is available.  See utility/XPrintf.h.
// *******************************************************************

int XPrintf( const char* fmt, ... )
{
  int returnValue;
  va_list argptr;
  va_start( argptr, fmt );

  returnValue = vprintf( fmt, argptr );

  va_end( argptr );
  return returnValue;
}
#endif //#ifdef WIN32

/********************************************************

END OF DEFAULT STUDENT FILE 

ADD HERE ALL STUDENT DEFINED AND AUX FUNCTIONS 

*******************************************************/
