#include "YourPlanner.h"
#include <rl/plan/SimpleModel.h>

YourPlanner::YourPlanner() :
        RrtConConBase(),
        functionCalls({}),
        useConnectV2(true),
        sampler(NULL),
        useLocalSampling(true),
        useGoalDirectedSampling(true),
        goalBias(0.15),
        useVolumeSampling(true),
        intermediateNodeInterval(5),
        maxFailedConnections(10)
{
}

YourPlanner::~YourPlanner()
{
}

::std::string
YourPlanner::getName() const
{
    ::std::string name = "Your Planner";
    if (useConnectV2) name.append(" CoV2");
    if (useVolumeSampling) name.append(" Vol");
    if (useLocalSampling) name.append(" ADD");
    if (useGoalDirectedSampling) name.append(" ADDW");
    if (goalBias > 0.) {
        name.append(" gbias:");
        name.append(std::to_string(goalBias));
    }
    if (intermediateNodeInterval > 0) {
        name.append(" CN: ");
        name.append(std::to_string(intermediateNodeInterval));
    }
    if (maxFailedConnections > 0) {
        name.append(" Stop: ");
        name.append(std::to_string(maxFailedConnections));
    }
    return name;
}

//////////////////////////////////////////////////////////////////////////////
// choose: Sample a configuration locally around a random tree node
//////////////////////////////////////////////////////////////////////////////
void
YourPlanner::choose(::rl::math::Vector& chosen, Tree& tree)
{
    functionCalls["choose_n"] += 1;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

    if (useLocalSampling)
    {
        // Generate a local offset vector and add it to a random tree node
        chosen = this->sampler->generateUnit(8., useVolumeSampling);

        u_int64_t len = ::boost::num_vertices(tree);
        ::rl::math::Vector rq;

        int idx = (int) (len * this->sampler->rand());
        Vertex vert = ::boost::vertex(idx, tree);
        rq = *(tree)[vert].q;

        chosen += rq;
        this->model->clip(chosen);
    } else
    {
        // Fallback: uniform random sampling
        chosen = this->sampler->generate();
    }

    std::chrono::steady_clock::time_point stop = std::chrono::steady_clock::now();
    functionCalls["choose_time"] += std::chrono::duration_cast< std::chrono::duration<double>>(stop - start).count() * 1000;
}

//////////////////////////////////////////////////////////////////////////////
// chooseN: Sample locally around the node closest to the other tree's root
//////////////////////////////////////////////////////////////////////////////
void
YourPlanner::chooseN(::rl::math::Vector& chosen, Tree& tree, ::rl::math::Vector& target)
{
    functionCalls["choose_n"] += 1;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

    // Generate a local offset vector
    chosen = this->sampler->generateUnit(8., useVolumeSampling);

    // Find node in the current tree closest to the other tree's root
    ::rl::math::Vector nq;
    nq = *tree[this->nearest(tree, target).first].q;
    chosen += nq;

    this->model->clip(chosen);

    std::chrono::steady_clock::time_point stop = std::chrono::steady_clock::now();
    functionCalls["choose_time"] += std::chrono::duration_cast< std::chrono::duration<double>>(stop - start).count() * 1000;
}

//////////////////////////////////////////////////////////////////////////////
// nearest: Find nearest neighbour, skipping exhausted nodes
//////////////////////////////////////////////////////////////////////////////
RrtConConBase::Neighbor
YourPlanner::nearest(const Tree& tree, const ::rl::math::Vector& chosen)
{
    functionCalls["nearest_n"] += 1;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

    Neighbor p(Vertex(), (::std::numeric_limits< ::rl::math::Real >::max)());

    for (VertexIteratorPair i = ::boost::vertices(tree); i.first != i.second; ++i.first)
    {
        // Constant check first for short-circuit evaluation
        if (maxFailedConnections != 0 && tree[*i.first].connectedCollidedCounter >= maxFailedConnections) continue;

        ::rl::math::Real d = this->model->transformedDistance(chosen, *tree[*i.first].q);
        if (d < p.second)
        {
            p.first = *i.first;
            p.second = d;
        }
    }

    p.second = this->model->inverseOfTransformedDistance(p.second);

    std::chrono::steady_clock::time_point stop = std::chrono::steady_clock::now();
    functionCalls["nearest_time"] += std::chrono::duration_cast< std::chrono::duration<double>>(stop - start).count() * 1000;
    return p;
}

//////////////////////////////////////////////////////////////////////////////
// connect: Greedily extend towards chosen, with early-stop and intermediate nodes
//////////////////////////////////////////////////////////////////////////////
RrtConConBase::Vertex
YourPlanner::connect(Tree& tree, const Neighbor& nearest, const ::rl::math::Vector& chosen)
{
    functionCalls["connect_n"] += 1;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

    // --- First extend step ---
    ::rl::math::Real distance = nearest.second;
    ::rl::math::Real step = distance;
    bool reached = false;

    if (step <= this->delta)
    {
        reached = true;
    }
    else
    {
        step = this->delta;
    }

    ::rl::plan::VectorPtr last = ::std::make_shared< ::rl::math::Vector >(this->model->getDof());

    // Interpolate from nearest node towards chosen
    this->model->interpolate(*tree[nearest.first].q, chosen, step / distance, *last);

    this->model->setPosition(*last);
    this->model->updateFrames();

    if (this->model->isColliding())
    {
        tree[nearest.first].connectedCollidedCounter++;
        return NULL;
    }

    ::rl::math::Vector next(this->model->getDof());

    if (reached) {
        *last = chosen;
    }

    // --- Further extend steps ---

    // Hoist loop-invariant: determine goal reference once
    const ::rl::math::Vector* goalRef = NULL;
    ::rl::math::Real distanceCachedToGoal = 0;
    if (useConnectV2 && !reached) {
        goalRef = ((&this->tree[0]) == (&tree)) ? this->goal : this->start;
        // Pre-compute distance for the initial *last position
        distanceCachedToGoal = this->model->transformedDistance(*last, *goalRef);
    }

    int addVertexCounter = 0;
    while (!reached)
    {
        distance = this->model->distance(*last, chosen);
        step = distance;

        if (step <= this->delta)
        {
            reached = true;
        }
        else
        {
            step = this->delta;
        }

        // Interpolate from last towards chosen
        this->model->interpolate(*last, chosen, step / distance, next);

        this->model->setPosition(next);
        this->model->updateFrames();

        if (this->model->isColliding())
        {
            tree[nearest.first].connectedCollidedCounter++;
            break;
        }

        // Early-stop: if extending is moving away from the overall goal,
        // probabilistically stop to avoid wasting collision queries
        if (useConnectV2) {
            ::rl::math::Real distanceNextToGoal = this->model->transformedDistance(next, *goalRef);

            if (distanceCachedToGoal < distanceNextToGoal) {
                if (this->sampler->rand() > 0.85) break;
            }

            // Cache for next iteration: next becomes last
            distanceCachedToGoal = distanceNextToGoal;
        }

        // Periodically add intermediate vertices to enrich the tree
        addVertexCounter++;
        if(intermediateNodeInterval > 0 && addVertexCounter >= intermediateNodeInterval) {
            Vertex connected = this->addVertex(tree, last);
            this->addEdge(nearest.first, connected, tree);
            addVertexCounter = 0;
        }

        *last = next;
    }

    // Add the final reached/stopped vertex to the tree
    Vertex connected = this->addVertex(tree, last);
    this->addEdge(nearest.first, connected, tree);

    std::chrono::steady_clock::time_point stop = std::chrono::steady_clock::now();
    functionCalls["connect_time"] += std::chrono::duration_cast< std::chrono::duration<double>>(stop - start).count() * 1000;

    return connected;
}

//////////////////////////////////////////////////////////////////////////////
// extend: Single extend step (delegates to base class)
//////////////////////////////////////////////////////////////////////////////
RrtConConBase::Vertex
YourPlanner::extend(Tree& tree, const Neighbor& nearest, const ::rl::math::Vector& chosen)
{
    functionCalls["extend_n"] += 1;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

    RrtConConBase::Vertex result = RrtConConBase::extend(tree, nearest, chosen);

    std::chrono::steady_clock::time_point stop = std::chrono::steady_clock::now();
    functionCalls["extend_time"] += std::chrono::duration_cast< std::chrono::duration<double>>(stop - start).count() * 1000;
    return result;
}

//////////////////////////////////////////////////////////////////////////////
// solve: Bidirectional RRT-Connect main loop
//////////////////////////////////////////////////////////////////////////////
bool
YourPlanner::solve()
{
    functionCalls = {};
    functionCalls["solve_n"] += 1;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

    this->time = ::std::chrono::steady_clock::now();

    // Initialize both trees with start and goal as roots
    this->begin[0] = this->addVertex(this->tree[0], ::std::make_shared< ::rl::math::Vector >(*this->start));
    this->begin[1] = this->addVertex(this->tree[1], ::std::make_shared< ::rl::math::Vector >(*this->goal));

    Tree* a = &this->tree[0];
    Tree* b = &this->tree[1];

    ::rl::math::Vector chosen(this->model->getDof());

    while ((::std::chrono::steady_clock::now() - this->time) < this->duration)
    {
        // Grow tree a then connect b, then swap roles
        for (::std::size_t j = 0; j < 2; ++j)
        {
            // Sample a random configuration with goal bias
            if(this->sampler->rand() > goalBias)
            {
                // Local sampling around a random node in tree a
                this->choose(chosen, *a);
            }
            else
            {
                if(useGoalDirectedSampling)
                {
                    // Sample around the node closest to the other tree's root
                    if(j == 0) target = *this->goal;
                    if(j == 1) target = *this->start;
                    this->chooseN(chosen, *a, target);
                }
                else
                {
                    // Directly use the other tree's root as target
                    if(j == 0) chosen = *this->goal;
                    if(j == 1) chosen = *this->start;
                }
            }

            // Find nearest neighbour in tree a and connect towards the sample
            Neighbor aNearest = this->nearest(*a, chosen);
            Vertex aConnected = this->connect(*a, aNearest, chosen);

            if (NULL != aConnected)
            {
                // Try to connect tree b towards the newly added node in tree a
                Neighbor bNearest = this->nearest(*b, *(*a)[aConnected].q);
                Vertex bConnected = this->connect(*b, bNearest, *(*a)[aConnected].q);

                if (NULL != bConnected)
                {
                    // Check if both trees are now connected
                    if (this->areEqual(*(*a)[aConnected].q, *(*b)[bConnected].q))
                    {
                        this->end[0] = &this->tree[0] == a ? aConnected : bConnected;
                        this->end[1] = &this->tree[1] == b ? bConnected : aConnected;

                        std::chrono::steady_clock::time_point stop = std::chrono::steady_clock::now();
                        functionCalls["solve_time"] += std::chrono::duration_cast< std::chrono::duration<double>>(stop - start).count() * 1000;

                        for (const auto& n : functionCalls) std::cout << n.first << " = " << n.second << "; ";
                        std::cout << '\n';

                        return true;
                    }
                }
            }

            // Swap the roles of a and b
            using ::std::swap;
            swap(a, b);
        }
    }

    return false;
}