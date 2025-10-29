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

void initNjtrackControl(GlobalVariables& gv) 
{
  // Control Initialization Code Here
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

void initProj2Control(GlobalVariables& gv) 
{
  // Control Initialization Code Here
}

void initProj3Control(GlobalVariables& gv) 
{
  // Control Initialization Code Here
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
  gv.tau = gv.kp * (gv.qd - gv.q) - gv.G; //P-controller with gravity compensation
}

void jgotoControl(GlobalVariables& gv) 
{
  gv.tau = gv.kp * (gv.qd - gv.q) + gv.G - gv.kv * gv.dq; //PD-controller with gravity compensation
}

void njtrackControl(GlobalVariables& gv) 
{
  floatControl(gv);  // Remove this line when you implement this controller
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
  floatControl(gv);  // Remove this line when you implement proj1Control
}

void proj2Control(GlobalVariables& gv) 
{
  floatControl(gv);  // Remove this line when you implement proj2Control
}

void proj3Control(GlobalVariables& gv) 
{
  floatControl(gv);  // Remove this line when you implement proj3Control
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
