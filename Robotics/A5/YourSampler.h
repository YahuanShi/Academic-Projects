#ifndef _YOURSAMPLER_H_
#define _YOURSAMPLER_H_

#include <rl/plan/Sampler.h>
#include <random>

namespace rl
{
    namespace plan
    {
        /**
         * Custom sampling strategy with local offset and volume sampling support.
         */
        class YourSampler : public Sampler
        {
        public:
            YourSampler();

            virtual ~YourSampler();

            /** Uniform random sampling within joint limits. */
            ::rl::math::Vector generate();

            virtual void seed(const ::std::mt19937::result_type& value);

            /** Returns a uniform random number in [0, 1). */
            ::std::uniform_real_distribution< ::rl::math::Real>::result_type rand();

            /**
             * Generate a random unit-direction vector scaled by 'scale',
             * optionally with randomized length for volume sampling.
             */
            ::rl::math::Vector generateUnit(::rl::math::Real scale, bool volumeSampling);

        protected:
            ::std::uniform_real_distribution< ::rl::math::Real> randDistribution;

            ::std::mt19937 randEngine;

        private:

        };
    }
}

#endif // _YOURSAMPLER_H_
