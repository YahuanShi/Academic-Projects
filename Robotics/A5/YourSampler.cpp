#include <chrono>
#include <rl/plan/SimpleModel.h>
#include "YourSampler.h"

namespace rl
{
    namespace plan
    {
        YourSampler::YourSampler() :
                Sampler(),
                randDistribution(0, 1),
                randEngine(::std::random_device()())
        {
        }

        YourSampler::~YourSampler()
        {
        }

        ::rl::math::Vector
        YourSampler::generate()
        {
            // Uniform random sampling within joint limits
            ::rl::math::Vector sampleq(this->model->getDof());

            ::rl::math::Vector maximum(this->model->getMaximum());
            ::rl::math::Vector minimum(this->model->getMinimum());

            for (::std::size_t i = 0; i < this->model->getDof(); ++i)
            {
                sampleq(i) = minimum(i) + this->rand() * (maximum(i) - minimum(i));
            }

            return sampleq;
        }

        ::rl::math::Vector
        YourSampler::generateUnit(::rl::math::Real scale, bool volumeSampling)
        {
            ::rl::math::Vector sampleq(this->model->getDof());
            ::rl::math::Vector samplevol(this->model->getDof());

            ::rl::math::Vector maximum(this->model->getMaximum());
            ::rl::math::Vector minimum(this->model->getMinimum());

            for (::std::size_t i = 0; i < this->model->getDof(); ++i)
            {
                sampleq(i) = minimum(i) + this->rand() * (maximum(i) - minimum(i));
                // Volume scaling factor in [0.1, 1.0] to avoid sampling too close to the tree node
                samplevol(i) = this->rand() * 0.9 + 0.1;
            }

            // Normalize sampled vector to unit length
            ::rl::math::Real len = 0.;
            for (::std::size_t i = 0; i < this->model->getDof(); ++i)
            {
                len += ::std::pow(len, 2);
            }
            len = ::std::pow(len, 1/2);

            // Scale to a local neighborhood controlled by 'scale'
            sampleq = (sampleq / len / (2*M_PI)) * scale;

            // Randomize vector length to sample a volume instead of a spherical shell
            if (volumeSampling) {
                for (::std::size_t i = 0; i < this->model->getDof(); ++i)
                {
                    sampleq(i) = samplevol(i) * sampleq(i);
                }
            }
            return sampleq;
        }

        ::std::uniform_real_distribution< ::rl::math::Real>::result_type
        YourSampler::rand()
        {
            return this->randDistribution(this->randEngine);
        }

        void
        YourSampler::seed(const ::std::mt19937::result_type& value)
        {
            this->randEngine.seed(value);
        }
    }
}
