#ifndef _YOUR_PLANNER_H_
#define _YOUR_PLANNER_H_

#ifndef M_PI
#define M_PI           3.14159265358979323846
#endif

#include "RrtConConBase.h"
#include <map>
#include <iostream>
#include "YourSampler.h"

using namespace ::rl::plan;

/**
 * Custom bidirectional RRT-Connect planner with local sampling,
 * goal-directed sampling, early-stop connect, intermediate nodes,
 * and failed-connection pruning.
 */
class YourPlanner : public RrtConConBase
{
public:
    YourPlanner();

    virtual ~YourPlanner();

    virtual ::std::string getName() const;

    bool solve();

    /** Per-function call counts and timing for profiling. */
    std::map<std::string, double> functionCalls;

    /** Custom sampler with local and volume sampling support. */
    YourSampler* sampler;

protected:
    /** Sample a configuration locally around a random node in the tree. */
    void choose(::rl::math::Vector& chosen, Tree& tree);

    /** Sample a configuration locally around the node closest to the other tree's root. */
    void chooseN(::rl::math::Vector& chosen, Tree& tree, ::rl::math::Vector& target);

    /** Connect nearest node towards chosen, with early-stop and intermediate nodes. */
    RrtConConBase::Vertex connect(Tree& tree, const Neighbor& nearest, const ::rl::math::Vector& chosen);

    /** Single extend step (delegates to base class). */
    RrtConConBase::Vertex extend(Tree& tree, const Neighbor& nearest, const ::rl::math::Vector& chosen);

    /** Find nearest neighbour, skipping nodes that exceeded maxFailedConnections. */
    RrtConConBase::Neighbor nearest(const Tree& tree, const ::rl::math::Vector& chosen);

private:
    // --- Extension toggles ---
    bool useConnectV2;              // Early-stop connect when moving away from goal
    bool useLocalSampling;          // Sample locally around existing tree nodes
    bool useGoalDirectedSampling;   // Sample around node closest to the other tree's root
    bool useVolumeSampling = false; // Randomize vector length for volume sampling

    // --- Tunable parameters ---
    float goalBias = 0;             // Probability of goal-directed sampling per iteration
    int intermediateNodeInterval;   // Add intermediate vertex every N connect steps
    int maxFailedConnections;       // Skip node in nearest() after N failed connections

    ::rl::math::Vector target;
};

#endif // _YOUR_PLANNER_H_
