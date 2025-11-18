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
    double t0, tf;
    PrVector a0, a1, a2, a3;
};
CubicSpline spline;

PrVector F;  // operational-space control force

// *******************************************************************
// DOF Mapping Helpers
// *******************************************************************

inline void extract3DOF(GlobalVariables& gv, PrVector& q, PrVector& dq)
{
    q  = PrVector(3);
    dq = PrVector(3);

    if (gv.dof == 3) {
        q = gv.q;
        dq = gv.dq;
    } else {
        q[0]=gv.q[1]; q[1]=gv.q[2]; q[2]=gv.q[4];
        dq[0]=gv.dq[1]; dq[1]=gv.dq[2]; dq[2]=gv.dq[4];
    }
}

inline void writeTau3DOF(GlobalVariables& gv, const PrVector& tau3)
{
    if (gv.dof == 3) {
        gv.tau = tau3;
    } else {
        gv.tau = PrVector(gv.dof);
        gv.tau[1]=tau3[0];
        gv.tau[2]=tau3[1];
        gv.tau[4]=tau3[2];
    }
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
    if (gv.dof == 3) {
      gv.G[0] = g123[0];
      gv.G[1] = g123[1];
      gv.G[2] = g123[2];
    } else if (gv.dof == 6) {
      gv.G[1] = g123[0];
      gv.G[2] = g123[1];
      gv.G[4] = g123[2];
    }
    // printing example, do not leave print inthe handed in solution 
    // printVariable(g123, "g123");
  } else {
    gv.G = PrVector(gv.G.size());
  }   
}

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

void initNjtrackControl(GlobalVariables& gv) // B1: Cubic Spline Initialization
{
  PrVector q(3), dq(3), qd(3);
    extract3DOF(gv, q, dq);

    if (gv.dof == 3) qd = gv.qd;
    else {
        qd[0]=gv.qd[1]; qd[1]=gv.qd[2]; qd[2]=gv.qd[4];
    }

    spline.t0 = gv.curTime;
    double T = 3.0;
    spline.tf = gv.curTime + T;

    spline.a0 = q;
    spline.a1 = PrVector(3);        // zero velocity
    spline.a2 = 3*(qd - q)/(T*T);
    spline.a3 = -2*(qd - q)/(T*T*T);
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
  // Control Initialization Code Here
}

void initProj2Control(GlobalVariables& gv) // C1: Circle Trajectory Init
{
  spline.t0 = gv.curTime;
}

void initProj3Control(GlobalVariables& gv) // C4: Parabolic Blend (3 full circles = 6π radians)
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

void njtrackControl(GlobalVariables& gv) // B2: Joint-Space PD + Gravity Compensation
{
  if (gv.curTime > spline.tf) {
        writeTau3DOF(gv, -gv.G);
        return;
    }

    double t = gv.curTime - spline.t0;

    PrVector qd  = spline.a0 + spline.a1*t + spline.a2*t*t + spline.a3*t*t*t;
    PrVector dqd = spline.a1 + 2*spline.a2*t + 3*spline.a3*t*t;

    PrVector q(3), dq(3);
    extract3DOF(gv, q, dq);

    PrVector tau3 = -gv.kp*(q-qd) - gv.kv*(dq-dqd) - gv.G;
    writeTau3DOF(gv, tau3);
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

void proj1Control(GlobalVariables& gv) // B2: Joint-Space PD + Gravity Compensation
{
  njtrackControl(gv);
}

void proj2Control(GlobalVariables& gv) // C2: Operational Space Circular Tracking
{
  double w = 2*M_PI/5;
    double t = gv.curTime - spline.t0;

    gv.xd[0] = 0.6 + 0.2*cos(w*t);
    gv.xd[1] = 0.35+ 0.2*sin(w*t);
    gv.xd[2] = 0;

    gv.dxd[0] = -0.2*w*sin(w*t);
    gv.dxd[1] =  0.2*w*cos(w*t);
    gv.dxd[2] = 0;

    F = -gv.kp*(gv.x-gv.xd) - gv.kv*(gv.dx-gv.dxd);
    gv.Jtranspose.multiply(F, gv.tau);
    gv.tau = gv.tau - gv.G;
}

void proj3Control(GlobalVariables& gv) 
{
  double ddB = 2*M_PI/25;
    double dB  = 2*M_PI/5;
    double total = 6*M_PI;

    double t = gv.curTime - spline.t0;
    double tb = dB/ddB;

    double beta, betaDot;

    if (t < tb) {
        betaDot = ddB*t;
        beta = 0.5*ddB*t*t;
    }
    else if (t < tb + (total - ddB*tb*tb)/dB) {
        double t2 = t - tb;
        betaDot = dB;
        beta = 0.5*ddB*tb*tb + dB*t2;
    }
    else {
        double t3 = t - tb - (total - ddB*tb*tb)/dB;
        betaDot = dB - ddB*t3;
        beta = total - 0.5*ddB*t3*t3;
    }

    gv.xd[0] = 0.6 + 0.2*cos(beta);
    gv.xd[1] = 0.35+ 0.2*sin(beta);
    gv.xd[2] = 0;

    gv.dxd[0] = -0.2*betaDot*sin(beta);
    gv.dxd[1] =  0.2*betaDot*cos(beta);
    gv.dxd[2] = 0;

    F = -gv.kp*(gv.x-gv.xd) - gv.kv*(gv.dx-gv.dxd);
    gv.Jtranspose.multiply(F, gv.tau);
    gv.tau = gv.tau - gv.G;
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
