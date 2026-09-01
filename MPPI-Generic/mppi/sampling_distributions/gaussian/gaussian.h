#pragma once
/**
 * Created by Bogdan Vlahov on 3/24/2023
 **/

#include <mppi/sampling_distributions/sampling_distribution.h>

#include <vector>

namespace mppi
{
namespace sampling_distributions
{
__global__ void setGaussianControls(const float* __restrict__ mean_d, const float* __restrict__ std_dev_d,
                                    float* __restrict__ control_samples_d, const int control_dim,
                                    const int num_timesteps, const int num_rollouts, const int num_distributions,
                                    const int optimization_stride, const float std_dev_decay,
                                    const float pure_noise_percentage, const bool time_specific_std_dev = false);

// Set the default number of distributions to 2 since that is currently the most we would use
template <int C_DIM, int MAX_DISTRIBUTIONS_T = 2>
struct GaussianParamsImpl : public SamplingParams<C_DIM>
{
  static const int MAX_DISTRIBUTIONS = MAX_DISTRIBUTIONS_T;
  float std_dev[C_DIM * MAX_DISTRIBUTIONS] MPPI_ALIGN(sizeof(float4)) = { 0.0f };
  float control_cost_coeff[C_DIM] MPPI_ALIGN(sizeof(float4)) = { 0.0f };
  float pure_noise_trajectories_percentage = 0.01f;
  float std_dev_decay = 1.0f;
  // Kernel launching params
  dim3 rewrite_controls_block_dim = dim3(32, 16, 1);
  int sum_strides = 32;
  // Various flags
  bool time_specific_std_dev = false;

  GaussianParamsImpl(int num_rollouts = 1, int num_timesteps = 1, int num_distributions = 1)
    : SamplingParams<C_DIM>::SamplingParams(num_rollouts, num_timesteps, num_distributions)
  {
    for (int i = 0; i < this->CONTROL_DIM * MAX_DISTRIBUTIONS; i++)
    {
      std_dev[i] = 1.0f;
    }
  }

  void copyStdDevToDistribution(const int src_distribution_idx, const int dest_distribution_idx)
  {
    bool src_out_of_distribution = src_distribution_idx >= MAX_DISTRIBUTIONS;
    if (src_out_of_distribution || dest_distribution_idx >= MAX_DISTRIBUTIONS)
    {
      printf("%s Distribution %d is out of range. There are only %d total distributions\n",
             src_out_of_distribution ? "Src" : "Dest",
             src_out_of_distribution ? src_distribution_idx : dest_distribution_idx, MAX_DISTRIBUTIONS);
      return;
    }
    float* std_dev_src = std_dev[this->CONTROL_DIM * src_distribution_idx];
    float* std_dev_dest = std_dev[this->CONTROL_DIM * dest_distribution_idx];
    for (int i = 0; i < this->CONTROL_DIM; i++)
    {
      std_dev_dest[i] = std_dev_src[i];
    }
  }
};

template <int C_DIM>
using GaussianParams = GaussianParamsImpl<C_DIM, 2>;

template <int C_DIM, int MAX_TIMESTEPS = 1, int MAX_DISTRIBUTIONS_T = 2>
struct GaussianTimeVaryingStdDevParams : public GaussianParamsImpl<C_DIM, MAX_DISTRIBUTIONS_T>
{
  float std_dev[C_DIM * MAX_TIMESTEPS * MAX_DISTRIBUTIONS_T] = { 0.0f };
  GaussianTimeVaryingStdDevParams(int num_rollouts = 1, int num_timesteps = 1, int num_distributions = 1)
    : GaussianParamsImpl<C_DIM, MAX_DISTRIBUTIONS_T>::GaussianParamsImpl(num_rollouts, num_timesteps, num_distributions)
  {
    this->time_specific_std_dev = true;
    for (int i = 0; i < this->CONTROL_DIM * MAX_TIMESTEPS * this->MAX_DISTRIBUTIONS; i++)
    {
      std_dev[i] = 1.0f;
    }
  }

  void copyStdDevToDistribution(const int src_distribution_idx, const int dest_distribution_idx)
  {
    bool src_out_of_distribution = src_distribution_idx >= this->MAX_DISTRIBUTIONS;
    if (src_out_of_distribution || dest_distribution_idx >= this->MAX_DISTRIBUTIONS)
    {
      printf("%s Distribution %d is out of range. There are only %d total distributions\n",
             src_out_of_distribution ? "Src" : "Dest",
             src_out_of_distribution ? src_distribution_idx : dest_distribution_idx, this->MAX_DISTRIBUTIONS);
      return;
    }
    float* std_dev_src = std_dev[this->CONTROL_DIM * this->num_timesteps * src_distribution_idx];
    float* std_dev_dest = std_dev[this->CONTROL_DIM * this->num_timesteps * dest_distribution_idx];
    for (int i = 0; i < this->CONTROL_DIM * this->num_timesteps; i++)
    {
      std_dev_dest[i] = std_dev_src[i];
    }
  }
};

template <class CLASS_T, template <int> class PARAMS_TEMPLATE = GaussianParams, class DYN_PARAMS_T = DynamicsParams>
class GaussianDistributionImpl : public SamplingDistribution<CLASS_T, PARAMS_TEMPLATE, DYN_PARAMS_T>
{
public:
  using PARENT_CLASS = SamplingDistribution<CLASS_T, PARAMS_TEMPLATE, DYN_PARAMS_T>;
  using SAMPLING_PARAMS_T = typename PARENT_CLASS::SAMPLING_PARAMS_T;
  using control_array = typename PARENT_CLASS::control_array;
  static const int CONTROL_DIM = PARENT_CLASS::CONTROL_DIM;
  typedef Eigen::Matrix<float, CONTROL_DIM, CONTROL_DIM> TEST_TYPE;

  GaussianDistributionImpl(cudaStream_t stream = 0);
  GaussianDistributionImpl(const SAMPLING_PARAMS_T& params, cudaStream_t stream = 0);

  ~GaussianDistributionImpl()
  {
    freeCudaMem();
  }

  __host__ virtual std::string getSamplingDistributionName() const override
  {
    return "Gaussian";
  }

  __host__ void allocateCUDAMemoryHelper();

  __host__ __device__ float computeFeedbackCost(const float* __restrict__ u_fb, float* __restrict__ theta_d,
                                                const int t, const int distribution_idx, const float lambda = 1.0,
                                                const float alpha = 0.0);

  __host__ float computeFeedbackCost(const Eigen::Ref<const control_array>& u_fb, const int t,
                                     const int distribution_idx, const float lambda = 1.0, const float alpha = 0.0);

  __host__ __device__ float computeLikelihoodRatioCost(const float* __restrict__ u, float* __restrict__ theta_d,
                                                       const int sample_index, const int t, const int distribution_idx,
                                                       const float lambda = 1.0, const float alpha = 0.0);

  __host__ float computeLikelihoodRatioCost(const Eigen::Ref<const control_array>& u, const int sample_index,
                                            const int t, const int distribution_idx, const float lambda = 1.0,
                                            const float alpha = 0.0);

  __host__ void copyImportanceSamplerToDevice(const float* importance_sampler, const int& distribution_idx,
                                              bool synchronize = true);

  __host__ void freeCudaMem();

  __host__ void generateSamples(const int& optimization_stride, const int& iteration_num, curandGenerator_t& gen,
                                bool synchronize = true);

  __host__ void paramsToDevice(bool synchronize = true);

  __host__ void setHostOptimalControlSequence(float* optimal_control_trajectory, const int& distribution_idx,
                                              bool synchronize = true);

  __host__ void setNumDistributions(const int num_distributions, bool synchronize = false)
  {
    if (num_distributions > SAMPLING_PARAMS_T::MAX_DISTRIBUTIONS)
    {
      this->logger_->error("GaussianParams can't handle more than %d distributions but %d were requested\n",
                           SAMPLING_PARAMS_T::MAX_DISTRIBUTIONS, num_distributions);
      throw std::out_of_range("Can't set num distributions higher than allowed in params");
    }
    PARENT_CLASS::setNumDistributions(num_distributions, synchronize);
  }

  __host__ void updateDistributionParamsFromDevice(const float* trajectory_weights_d, float normalizer,
                                                   const int& distribution_i, bool synchronize = false) override;

protected:
  float* std_dev_d_ = nullptr;
  float* control_means_d_ = nullptr;
  std::vector<float> means_;
};

template <class DYN_PARAMS_T>
class GaussianDistribution
  : public GaussianDistributionImpl<GaussianDistribution<DYN_PARAMS_T>, GaussianParams, DYN_PARAMS_T>
{
public:
  using PARENT_CLASS = GaussianDistributionImpl<GaussianDistribution, GaussianParams, DYN_PARAMS_T>;
  using SAMPLING_PARAMS_T = typename PARENT_CLASS::SAMPLING_PARAMS_T;

  GaussianDistribution(cudaStream_t stream = 0) : PARENT_CLASS(stream)
  {
  }
  GaussianDistribution(const SAMPLING_PARAMS_T& params, cudaStream_t stream = 0) : PARENT_CLASS(params, stream)
  {
  }
};

template <int C_DIM, int MAX_DISTRIBUTIONS_T>
const int GaussianParamsImpl<C_DIM, MAX_DISTRIBUTIONS_T>::MAX_DISTRIBUTIONS;
}  // namespace sampling_distributions
}  // namespace mppi

#include <mppi/core/mppi_common.h>
#include <mppi/utils/cuda_math_utils.h>
#include <mppi/utils/math_utils.h>

namespace mppi
{
namespace sampling_distributions
{
#define GAUSSIAN_TEMPLATE template <class CLASS_T, template <int> class PARAMS_TEMPLATE, class DYN_PARAMS_T>
#define GAUSSIAN_CLASS GaussianDistributionImpl<CLASS_T, PARAMS_TEMPLATE, DYN_PARAMS_T>

__global__ void setGaussianControls(const float* __restrict__ mean_d, const float* __restrict__ std_dev_d,
                                    float* __restrict__ control_samples_d, const int control_dim,
                                    const int num_timesteps, const int num_rollouts, const int num_distributions,
                                    const int optimization_stride, const float std_dev_decay,
                                    const float pure_noise_percentage, const bool time_specific_std_dev)
{
  const int trajectory_index = threadIdx.x + blockDim.x * blockIdx.x;
  const int distribution_index = threadIdx.z + blockDim.z * blockIdx.z;
  const int time_index = threadIdx.y + blockDim.y * blockIdx.y;
  const bool valid_index =
      trajectory_index < num_rollouts && time_index < num_timesteps && distribution_index < num_distributions;
  const auto& num_timesteps_block = blockDim.y;
  const auto& num_rollouts_block = blockDim.x;
  const int shared_noise_index = threadIdx.z * num_timesteps_block * num_rollouts_block * control_dim +
                                 threadIdx.x * num_timesteps_block * control_dim + threadIdx.y * control_dim;
  const int global_noise_index =
      min(distribution_index, num_distributions) * num_timesteps * num_rollouts * control_dim +
      min(trajectory_index, num_rollouts) * num_timesteps * control_dim + min(time_index, num_timesteps) * control_dim;
  const int shared_mean_index = distribution_index * num_timesteps * control_dim + time_index * control_dim;
  // Std Deviation setup
  int std_dev_size = num_distributions * control_dim;
  int shared_std_dev_index = threadIdx.z * num_timesteps * control_dim + threadIdx.y * control_dim;
  int global_std_dev_index = min(distribution_index, num_distributions) * num_timesteps * control_dim +
                             min(time_index, num_timesteps) * control_dim;
  shared_std_dev_index = time_specific_std_dev ? shared_std_dev_index : 0;
  global_std_dev_index = time_specific_std_dev ? global_std_dev_index : 0;
  std_dev_size = time_specific_std_dev ? num_timesteps * std_dev_size : std_dev_size;

  // local variables
  int i, j, k;

  // Shared memory setup
  /**
   * @brief Shared memory setup
   * This kernel has three shared memory arrays, mean_shared, std_dev_shared, and control_samples_shared.
   * In order to prevent memory alignment issues, the memory is being over-allocated to ensure that they are start on
   * the float4 boundary mean_shared - size should be num_timesteps * num_distributions * control_dim std_dev_shared =
   * num_distributions * control_dim if time_specific_std_dev is false std_dev_shared = num_distributions *
   * num_timesteps * control_dim if time_specific_std_dev is true control_samples_shared = BLOCKSIZE_X * BLOCKSIZE_Y *
   * BLOCKSIZE_Z * control_dim
   *
   */
  extern __shared__ float entire_buffer[];
  // Create memory_aligned shared memory pointers
  float* mean_shared = entire_buffer;
  float* std_dev_shared = &mean_shared[mppi::math::nearest_multiple_4(num_timesteps * num_distributions * control_dim)];
  float* control_samples_shared = &std_dev_shared[mppi::math::nearest_multiple_4(std_dev_size)];
  if (control_dim % 4 == 0)
  {
    // Step 1: copy means into shared memory
    for (i = threadIdx.z; i < num_distributions; i += blockDim.z)
    {
      for (j = threadIdx.y; j < num_timesteps; j += blockDim.y)
      {
        const int mean_index = (i * num_timesteps + j) * control_dim;
        float4* mean_shared4 = reinterpret_cast<float4*>(&mean_shared[mean_index]);
        const float4* mean_d4 = reinterpret_cast<const float4*>(&mean_d[mean_index]);
        for (k = threadIdx.x; k < control_dim / 4; k += blockDim.x)
        {
          mean_shared4[k] = mean_d4[k];
        }
      }
    }

    // Step 2: load std_dev to shared memory
    const float4* std_dev_d4 = reinterpret_cast<const float4*>(&std_dev_d[global_std_dev_index]);
    float4* std_dev_shared4 = reinterpret_cast<float4*>(&std_dev_shared[shared_std_dev_index]);
    for (i = threadIdx.x; i < control_dim / 4; i += blockDim.x)
    {
      std_dev_shared4[i] = std_dev_decay * std_dev_d4[i];
    }

    // Step 3: load noise into shared memory
    float4* control_samples_shared4 = reinterpret_cast<float4*>(&control_samples_shared[shared_noise_index]);
    float4* control_samples_d4 = reinterpret_cast<float4*>(&control_samples_d[global_noise_index]);
    // Create const pointre to mean in shared memory as it shouldn't change henceforth
    const float4* mean_shared4 = reinterpret_cast<const float4*>(&mean_shared[shared_mean_index]);
    for (i = 0; valid_index && i < control_dim / 4; i++)
    {
      control_samples_shared4[i] = control_samples_d4[i];
    }

    __syncthreads();  // wait for all copying from global to shared memory to finish
    // Step 4: do mean + variance calculations
    if (valid_index && (trajectory_index == 0 || time_index < optimization_stride))
    {  // 0 noise trajectory
      for (i = 0; i < control_dim / 4; i++)
      {
        control_samples_shared4[i] = mean_shared4[i];
      }
    }
    else if (valid_index && trajectory_index >= (1.0f - pure_noise_percentage) * num_rollouts)
    {  // doing zero mean trajectories
      for (i = 0; i < control_dim / 4; i++)
      {
        control_samples_shared4[i] = std_dev_shared4[i] * control_samples_shared4[i];
      }
    }
    else if (valid_index)
    {
      for (i = 0; i < control_dim / 4; i++)
      {
        control_samples_shared4[i] = mean_shared4[i] + std_dev_shared4[i] * control_samples_shared4[i];
      }
    }

    // save back to global memory
    for (i = 0; valid_index && i < control_dim / 4; i++)
    {
      control_samples_d4[i] = control_samples_shared4[i];
    }
  }
  else if (control_dim % 2 == 0)
  {
    // Step 1: copy means into shared memory
    for (i = threadIdx.z; i < num_distributions; i += blockDim.z)
    {
      for (j = threadIdx.y; j < num_timesteps; j += blockDim.y)
      {
        const int mean_index = (i * num_timesteps + j) * control_dim;
        float2* mean_shared2 = reinterpret_cast<float2*>(&mean_shared[mean_index]);
        const float2* mean_d2 = reinterpret_cast<const float2*>(&mean_d[mean_index]);
        for (k = threadIdx.x; k < control_dim / 2; k += blockDim.x)
        {
          mean_shared2[k] = mean_d2[k];
        }
      }
    }

    // Step 2: load std_dev to shared memory
    const float2* std_dev_d2 = reinterpret_cast<const float2*>(&std_dev_d[global_std_dev_index]);
    float2* std_dev_shared2 = reinterpret_cast<float2*>(&std_dev_shared[shared_std_dev_index]);
    for (i = threadIdx.x; i < control_dim / 2; i += blockDim.x)
    {
      std_dev_shared2[i] = std_dev_decay * std_dev_d2[i];
    }

    // Step 3: load noise into shared memory
    float2* control_samples_shared2 = reinterpret_cast<float2*>(&control_samples_shared[shared_noise_index]);
    float2* control_samples_d2 = reinterpret_cast<float2*>(&control_samples_d[global_noise_index]);
    // Create const pointer to mean in shared memory as it shouldn't change henceforth
    const float2* mean_shared2 = reinterpret_cast<const float2*>(&mean_shared[shared_mean_index]);
    for (i = 0; valid_index && i < control_dim / 2; i++)
    {
      control_samples_shared2[i] = control_samples_d2[i];
    }

    __syncthreads();  // wait for all copying from global to shared memory to finish
    // Step 4: do mean + variance calculations
    if (valid_index && (trajectory_index == 0 || time_index < optimization_stride))
    {  // 0 noise trajectory
      for (i = 0; i < control_dim / 2; i++)
      {
        control_samples_shared2[i] = mean_shared2[i];
      }
    }
    else if (valid_index && trajectory_index >= (1.0f - pure_noise_percentage) * num_rollouts)
    {  // doing zero mean trajectories
      for (i = 0; i < control_dim / 2; i++)
      {
        control_samples_shared2[i] = std_dev_shared2[i] * control_samples_shared2[i];
      }
    }
    else if (valid_index)
    {
      for (i = 0; i < control_dim / 2; i++)
      {
        control_samples_shared2[i] = mean_shared2[i] + std_dev_shared2[i] * control_samples_shared2[i];
      }
    }

    // save back to global memory
    for (i = 0; valid_index && i < control_dim / 2; i++)
    {
      control_samples_d2[i] = control_samples_shared2[i];
    }
  }
  else
  {  // No memory alignment to take advantage of
    // Step 1: copy means into shared memory
    for (i = threadIdx.z; i < num_distributions; i += blockDim.z)
    {
      for (j = threadIdx.y; j < num_timesteps; j += blockDim.y)
      {
        const int mean_index = (i * num_timesteps + j) * control_dim;
        for (k = threadIdx.x; k < control_dim; k += blockDim.x)
        {
          mean_shared[mean_index + k] = mean_d[mean_index + k];
        }
      }
    }
    // __syncthreads();
    // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0)
    // {
    //   printf("Mean: ");
    //   for (i = 0; i < num_timesteps; i++)
    //   {
    //     printf("%f, ", mean_d[i]);
    //   }
    //   printf("\n");
    // }
    // __syncthreads();

    // Step 2: load std_dev to shared memory
    for (i = threadIdx.x; i < control_dim; i += blockDim.x)
    {
      std_dev_shared[shared_std_dev_index + i] = std_dev_decay * std_dev_d[global_std_dev_index + i];
    }

    // Step 3: load noise into shared memory
    for (i = 0; valid_index && i < control_dim; i++)
    {
      control_samples_shared[shared_noise_index + i] = control_samples_d[global_noise_index + i];
    }
    // __syncthreads();
    // if (trajectory_index == 10 && time_index == 20)
    // {
    //   printf("Control noise %d at time %d: ", trajectory_index, time_index);
    //   for (i = 0; i < control_dim; i++)
    //   {
    //     printf("%f, ", control_samples_shared[shared_noise_index + i]);
    //   }
    //   printf("\n std_dev_decay: %f, optimization_stride: %d\n", std_dev_decay, optimization_stride);
    //   printf("Std Dev: %f\n", std_dev_shared[shared_std_dev_index]);
    // }
    // __syncthreads();

    __syncthreads();  // wait for all copying from global to shared memory to finish
    // Step 4: do mean + variance calculations
    if (valid_index && (trajectory_index == 0 || time_index < optimization_stride))
    {  // 0 noise trajectory
      for (i = 0; i < control_dim; i++)
      {
        control_samples_shared[shared_noise_index + i] = mean_shared[shared_mean_index + i];
      }
    }
    else if (valid_index && trajectory_index >= (1.0f - pure_noise_percentage) * num_rollouts)
    {  // doing zero mean trajectories
      for (i = 0; i < control_dim; i++)
      {
        control_samples_shared[shared_noise_index + i] =
            std_dev_shared[shared_std_dev_index + i] * control_samples_shared[shared_noise_index + i];
      }
    }
    else if (valid_index)
    {
      for (i = 0; i < control_dim; i++)
      {
        control_samples_shared[shared_noise_index + i] =
            mean_shared[shared_mean_index + i] +
            std_dev_shared[shared_std_dev_index + i] * control_samples_shared[shared_noise_index + i];
      }
    }
    __syncthreads();
    // save back to global memory
    for (i = 0; valid_index && i < control_dim; i++)
    {
      control_samples_d[global_noise_index + i] = control_samples_shared[shared_noise_index + i];
    }
  }
}

GAUSSIAN_TEMPLATE
GAUSSIAN_CLASS::GaussianDistributionImpl(cudaStream_t stream) : PARENT_CLASS::SamplingDistribution(stream)
{
}

GAUSSIAN_TEMPLATE
GAUSSIAN_CLASS::GaussianDistributionImpl(const SAMPLING_PARAMS_T& params, cudaStream_t stream)
  : PARENT_CLASS::SamplingDistribution(params, stream)
{
}

GAUSSIAN_TEMPLATE
__host__ void GAUSSIAN_CLASS::allocateCUDAMemoryHelper()
{
  if (this->GPUMemStatus_)
  {
    if (std_dev_d_)
    {
#if defined(CUDART_VERSION) && CUDART_VERSION > 11200
      HANDLE_ERROR(cudaFreeAsync(std_dev_d_, this->stream_));
#else
      HANDLE_ERROR(cudaFree(std_dev_d_));
#endif
    }
    if (control_means_d_)
    {  // deallocate previous memory control trajectory means
#if defined(CUDART_VERSION) && CUDART_VERSION > 11200
      HANDLE_ERROR(cudaFreeAsync(control_means_d_, this->stream_));
#else
      HANDLE_ERROR(cudaFree(control_means_d_));
#endif
    }

    int std_dev_size = CONTROL_DIM * this->getNumDistributions();
    if (this->params_.time_specific_std_dev)
    {
      std_dev_size *= this->getNumTimesteps();
    }
#if defined(CUDART_VERSION) && CUDART_VERSION > 11200
    HANDLE_ERROR(cudaMallocAsync((void**)&std_dev_d_, sizeof(float) * std_dev_size, this->stream_));
    HANDLE_ERROR(cudaMallocAsync((void**)&control_means_d_,
                                 sizeof(float) * this->getNumDistributions() * this->getNumTimesteps() * CONTROL_DIM,
                                 this->stream_));
#else
    HANDLE_ERROR(cudaMalloc((void**)&std_dev_d_, sizeof(float) * std_dev_size));
    HANDLE_ERROR(cudaMalloc((void**)&control_means_d_,
                            sizeof(float) * this->getNumDistributions() * this->getNumTimesteps() * CONTROL_DIM));
#endif
    means_.resize(this->getNumDistributions() * this->getNumTimesteps() * CONTROL_DIM);
    // Ensure that the device side point knows where the the standard deviation memory is located
    HANDLE_ERROR(cudaMemcpyAsync(&this->sampling_d_->std_dev_d_, &std_dev_d_, sizeof(float*), cudaMemcpyHostToDevice,
                                 this->stream_));
    HANDLE_ERROR(cudaMemcpyAsync(&this->sampling_d_->control_means_d_, &control_means_d_, sizeof(float*),
                                 cudaMemcpyHostToDevice, this->stream_));
  }
}

GAUSSIAN_TEMPLATE
__host__ void GAUSSIAN_CLASS::freeCudaMem()
{
  if (this->GPUMemStatus_)
  {
    HANDLE_ERROR(cudaFree(control_means_d_));
    HANDLE_ERROR(cudaFree(std_dev_d_));
    control_means_d_ = nullptr;
    std_dev_d_ = nullptr;
  }
  PARENT_CLASS::freeCudaMem();
}

GAUSSIAN_TEMPLATE
void GAUSSIAN_CLASS::paramsToDevice(bool synchronize)
{
  PARENT_CLASS::paramsToDevice(false);
  if (this->GPUMemStatus_)
  {
    if (this->params_.time_specific_std_dev)
    {
      HANDLE_ERROR(cudaMemcpyAsync(this->std_dev_d_, this->params_.std_dev,
                                   sizeof(float) * CONTROL_DIM * this->getNumTimesteps() * this->getNumDistributions(),
                                   cudaMemcpyHostToDevice, this->stream_));
    }
    else
    {
      HANDLE_ERROR(cudaMemcpyAsync(this->std_dev_d_, this->params_.std_dev,
                                   sizeof(float) * CONTROL_DIM * this->getNumDistributions(), cudaMemcpyHostToDevice,
                                   this->stream_));
    }
    if (synchronize)
    {
      HANDLE_ERROR(cudaStreamSynchronize(this->stream_));
    }
  }
}

GAUSSIAN_TEMPLATE
__host__ void GAUSSIAN_CLASS::generateSamples(const int& optimization_stride, const int& iteration_num,
                                              curandGenerator_t& gen, bool synchronize)
{
  if (this->params_.use_same_noise_for_all_distributions)
  {
    HANDLE_CURAND_ERROR(curandGenerateNormal(
        gen, this->control_samples_d_, this->getNumTimesteps() * this->getNumRollouts() * CONTROL_DIM, 0.0f, 1.0f));
    for (int i = 1; i < this->getNumDistributions(); i++)
    {
      HANDLE_ERROR(cudaMemcpyAsync(
          &this->control_samples_d_[this->getNumRollouts() * this->getNumTimesteps() * CONTROL_DIM * i],
          this->control_samples_d_, sizeof(float) * this->getNumRollouts() * this->getNumTimesteps() * CONTROL_DIM,
          cudaMemcpyDeviceToDevice, this->stream_));
    }
  }
  else
  {
    HANDLE_CURAND_ERROR(curandGenerateNormal(
        gen, this->control_samples_d_,
        this->getNumTimesteps() * this->getNumRollouts() * this->getNumDistributions() * CONTROL_DIM, 0.0f, 1.0f));
  }
  const int BLOCKSIZE_X = this->params_.rewrite_controls_block_dim.x;
  const int BLOCKSIZE_Y = this->params_.rewrite_controls_block_dim.y;
  const int BLOCKSIZE_Z = this->params_.rewrite_controls_block_dim.z;
  /**
   * Generate noise samples with mean added
   **/
  dim3 control_writing_grid;
  control_writing_grid.x = mppi::math::int_ceil(this->getNumRollouts(), BLOCKSIZE_X);
  control_writing_grid.y = mppi::math::int_ceil(this->getNumTimesteps(), BLOCKSIZE_Y);
  control_writing_grid.z = mppi::math::int_ceil(this->getNumDistributions(), BLOCKSIZE_Z);
  unsigned int std_dev_mem_size = this->getNumDistributions() * CONTROL_DIM;
  // Allocate shared memory for std_deviations per timestep or constant across the trajectory
  std_dev_mem_size = mppi::math::nearest_multiple_4(
      this->params_.time_specific_std_dev ? std_dev_mem_size * this->getNumTimesteps() : std_dev_mem_size);
  unsigned int shared_mem_size =
      std_dev_mem_size +
      mppi::math::nearest_multiple_4(this->getNumDistributions() * this->getNumTimesteps() * CONTROL_DIM) +
      mppi::math::nearest_multiple_4(BLOCKSIZE_X * BLOCKSIZE_Y * BLOCKSIZE_Z * CONTROL_DIM);
  shared_mem_size *= sizeof(float);
  // std::cout << "Shared mem size: " << shared_mem_size << " bytes. BLOCKSIZE_X: " << BLOCKSIZE_X
  //           << ", BLOCKSIZE_Y: " << BLOCKSIZE_Y << ", BLOCKSIZE_Z: " << BLOCKSIZE_Z
  //           << "Grid: (" << control_writing_grid.x << ", " << control_writing_grid.y
  //           << ", " << control_writing_grid.z << ")" << std::endl;
  setGaussianControls<<<control_writing_grid, this->params_.rewrite_controls_block_dim, shared_mem_size,
                        this->stream_>>>(
      this->control_means_d_, this->std_dev_d_, this->control_samples_d_, CONTROL_DIM, this->getNumTimesteps(),
      this->getNumRollouts(), this->getNumDistributions(), optimization_stride,
      powf(this->params_.std_dev_decay, iteration_num), this->params_.pure_noise_trajectories_percentage,
      this->params_.time_specific_std_dev);

  HANDLE_ERROR(cudaGetLastError());
  if (synchronize)
  {
    HANDLE_ERROR(cudaStreamSynchronize(this->stream_));
  }
}

GAUSSIAN_TEMPLATE
__host__ void GAUSSIAN_CLASS::updateDistributionParamsFromDevice(const float* trajectory_weights_d, float normalizer,
                                                                 const int& distribution_i, bool synchronize)
{
  if (distribution_i >= this->getNumDistributions())
  {
    this->logger_->error(
        "Updating distributional params for distribution %d out of %d total. Distribution out of bounds.\n",
        distribution_i, this->getNumDistributions());
    return;
  }
  float* control_samples_i_d =
      &(this->control_samples_d_[distribution_i * this->getNumRollouts() * this->getNumTimesteps() * CONTROL_DIM]);
  float* control_mean_i_d = &(this->control_means_d_[distribution_i * this->getNumTimesteps() * CONTROL_DIM]);
  mppi::kernels::launchWeightedReductionKernel<CONTROL_DIM>(trajectory_weights_d, control_samples_i_d, control_mean_i_d,
                                                            normalizer, this->getNumTimesteps(), this->getNumRollouts(),
                                                            this->params_.sum_strides, this->stream_, synchronize);
  HANDLE_ERROR(cudaMemcpyAsync(&means_[distribution_i * this->getNumTimesteps() * CONTROL_DIM], control_mean_i_d,
                               sizeof(float) * this->getNumTimesteps() * CONTROL_DIM, cudaMemcpyDeviceToHost,
                               this->stream_));
  if (synchronize)
  {
    HANDLE_ERROR(cudaStreamSynchronize(this->stream_));
  }
}

GAUSSIAN_TEMPLATE
__host__ void GAUSSIAN_CLASS::setHostOptimalControlSequence(float* optimal_control_trajectory,
                                                            const int& distribution_i, bool synchronize)
{
  if (distribution_i >= this->getNumDistributions())
  {
    this->logger_->error(
        "Asking for optimal control sequence from distribution %d out of %d total. Distribution out of bounds.\n",
        distribution_i, this->getNumDistributions());
    return;
  }

  HANDLE_ERROR(cudaMemcpyAsync(
      optimal_control_trajectory, &(this->control_means_d_[this->getNumTimesteps() * CONTROL_DIM * distribution_i]),
      sizeof(float) * this->getNumTimesteps() * CONTROL_DIM, cudaMemcpyDeviceToHost, this->stream_));
  if (synchronize)
  {
    HANDLE_ERROR(cudaStreamSynchronize(this->stream_));
  }
}

GAUSSIAN_TEMPLATE
__host__ __device__ float GAUSSIAN_CLASS::computeLikelihoodRatioCost(const float* __restrict__ u,
                                                                     float* __restrict__ theta_d,
                                                                     const int sample_index, const int t,
                                                                     const int distribution_idx, const float lambda,
                                                                     const float alpha)
{
  SAMPLING_PARAMS_T* params_p = (SAMPLING_PARAMS_T*)theta_d;
  const int distribution_i = distribution_idx >= params_p->num_distributions ? 0 : distribution_idx;
  float* std_dev = &(params_p->std_dev[CONTROL_DIM * distribution_i]);
  if (params_p->time_specific_std_dev)
  {
    std_dev = &(params_p->std_dev[(distribution_i * params_p->num_timesteps + t) * CONTROL_DIM]);
  }
  float* mean = &(this->control_means_d_[(params_p->num_timesteps * distribution_i + t) * CONTROL_DIM]);
  float* control_cost_coeff = params_p->control_cost_coeff;

  float cost = 0.0f;
  int i, step;
  mppi::p1::getParallel1DIndex<mppi::p1::Parallel1Dir::THREAD_Y>(i, step);

  if (CONTROL_DIM % 4 == 0)
  {
    float4 cost_i = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 mean_i, std_dev_i, u_i, control_cost_coeff_i;
    for (; i < CONTROL_DIM / 4; i += step)
    {
      if (sample_index >= (1.0f - params_p->pure_noise_trajectories_percentage) * params_p->num_rollouts)
      {
        mean_i = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      }
      else
      {
        mean_i = reinterpret_cast<float4*>(mean)[i];  // read mean value from global memory only once
      }
      std_dev_i = reinterpret_cast<float4*>(std_dev)[i];
      u_i = reinterpret_cast<const float4*>(u)[i];
      control_cost_coeff_i = reinterpret_cast<float4*>(control_cost_coeff)[i];
      // cost_i += control_cost_coeff_i * mean_i * (mean_i + 2 * (u_i - mean_i)) / (std_dev_i * std_dev_i);
      cost_i += control_cost_coeff_i * mean_i * (mean_i - 2.0f * u_i) / (std_dev_i * std_dev_i);  // Proper way
    }
    cost += cost_i.x + cost_i.y + cost_i.z + cost_i.w;
  }
  else if (CONTROL_DIM % 2 == 0)
  {
    float2 cost_i = make_float2(0.0f, 0.0f);
    float2 mean_i, std_dev_i, u_i, control_cost_coeff_i;
    for (; i < CONTROL_DIM / 2; i += step)
    {
      if (sample_index >= (1.0f - params_p->pure_noise_trajectories_percentage) * params_p->num_rollouts)
      {
        mean_i = make_float2(0.0f, 0.0f);
      }
      else
      {
        mean_i = reinterpret_cast<float2*>(mean)[i];  // read mean value from global memory only once
      }
      std_dev_i = reinterpret_cast<float2*>(std_dev)[i];
      u_i = reinterpret_cast<const float2*>(u)[i];
      control_cost_coeff_i = reinterpret_cast<float2*>(control_cost_coeff)[i];
      // cost_i += control_cost_coeff_i * mean_i * (mean_i + 2 * (u_i - mean_i)) / (std_dev_i * std_dev_i);
      cost_i += control_cost_coeff_i * mean_i * (mean_i - 2.0f * u_i) / (std_dev_i * std_dev_i);  // Proper way
    }
    cost += cost_i.x + cost_i.y;
  }
  else
  {
    float mean_i;
    for (; i < CONTROL_DIM; i += step)
    {
      if (sample_index >= (1.0f - params_p->pure_noise_trajectories_percentage) * params_p->num_rollouts)
      {
        mean_i = 0.0f;
      }
      else
      {
        mean_i = mean[i];  // read mean value from global memory only once
      }
      cost += control_cost_coeff[i] * mean_i * (mean_i - 2.0f * u[i]) / (std_dev[i] * std_dev[i]);  // Proper way
      // float noise = u[i] - mean_i;
      // cost += control_cost_coeff[i] * mean_i * (u[i] + noise) / (std_dev[i] * std_dev[i]); // Way in cost kernel
    }
  }
  return 0.5f * lambda * (1.0f - alpha) * cost;
}

GAUSSIAN_TEMPLATE
__host__ __device__ float GAUSSIAN_CLASS::computeFeedbackCost(const float* __restrict__ u_fb,
                                                              float* __restrict__ theta_d, const int t,
                                                              const int distribution_idx, const float lambda,
                                                              const float alpha)
{
  SAMPLING_PARAMS_T* params_p = (SAMPLING_PARAMS_T*)theta_d;
  const int distribution_i = distribution_idx >= params_p->num_distributions ? 0 : distribution_idx;
  float* std_dev = &(params_p->std_dev[CONTROL_DIM * distribution_i]);
  if (params_p->time_specific_std_dev)
  {
    std_dev = &(params_p->std_dev[(distribution_i * params_p->num_timesteps + t) * CONTROL_DIM]);
  }
  float* control_cost_coeff = params_p->control_cost_coeff;

  float cost = 0.0f;
  int i, step;
  mppi::p1::getParallel1DIndex<mppi::p1::Parallel1Dir::THREAD_Y>(i, step);

  if (CONTROL_DIM % 4 == 0)
  {
    float4 cost_i = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 std_dev_i, control_cost_coeff_i, u_fb_i;
    for (; i < CONTROL_DIM / 4; i += step)
    {
      std_dev_i = reinterpret_cast<float4*>(std_dev)[i];
      u_fb_i = reinterpret_cast<const float4*>(u_fb)[i];
      control_cost_coeff_i = reinterpret_cast<float4*>(control_cost_coeff)[i];
      cost_i += control_cost_coeff_i * (u_fb_i * u_fb_i) / (std_dev_i * std_dev_i);
    }
    cost += cost_i.x + cost_i.y + cost_i.z + cost_i.w;
  }
  else if (CONTROL_DIM % 2 == 0)
  {
    float2 cost_i = make_float2(0.0f, 0.0f);
    float2 std_dev_i, control_cost_coeff_i, u_fb_i;
    for (; i < CONTROL_DIM / 2; i += step)
    {
      std_dev_i = reinterpret_cast<float2*>(std_dev)[i];
      control_cost_coeff_i = reinterpret_cast<float2*>(control_cost_coeff)[i];
      u_fb_i = reinterpret_cast<const float2*>(u_fb)[i];
      cost_i += control_cost_coeff_i * (u_fb_i * u_fb_i) / (std_dev_i * std_dev_i);
    }
    cost += cost_i.x + cost_i.y;
  }
  else
  {
    for (; i < CONTROL_DIM; i += step)
    {
      cost += control_cost_coeff[i] * (u_fb[i] * u_fb[i]) / (std_dev[i] * std_dev[i]);
    }
  }
  return 0.5f * lambda * (1.0f - alpha) * cost;
}

GAUSSIAN_TEMPLATE
__host__ float GAUSSIAN_CLASS::computeFeedbackCost(const Eigen::Ref<const control_array>& u_fb, const int t,
                                                   const int distribution_idx, const float lambda, const float alpha)
{
  float cost = 0.0f;
  const int distribution_i = distribution_idx >= this->params_.num_distributions ? 0 : distribution_idx;
  float* std_dev = &(this->params_.std_dev[CONTROL_DIM * distribution_i]);
  if (this->params_.time_specific_std_dev)
  {
    std_dev = &(this->params_.std_dev[(distribution_i * this->getNumTimesteps() + t) * CONTROL_DIM]);
  }
  for (int i = 0; i < CONTROL_DIM; i++)
  {
    cost += this->params_.control_cost_coeff[i] * u_fb(i) * u_fb(i) / (std_dev[i] * std_dev[i]);
  }
  return 0.5f * lambda * (1.0f - alpha) * cost;
}

GAUSSIAN_TEMPLATE
__host__ float GAUSSIAN_CLASS::computeLikelihoodRatioCost(const Eigen::Ref<const control_array>& u,
                                                          const int sample_index, const int t,
                                                          const int distribution_idx, const float lambda,
                                                          const float alpha)
{
  float cost = 0.0f;
  const int distribution_i = distribution_idx >= this->params_.num_distributions ? 0 : distribution_idx;
  const int mean_index = (distribution_i * this->getNumTimesteps() + t) * CONTROL_DIM;
  float* mean = &(this->means_[mean_index]);
  float* std_dev = &(this->params_.std_dev[CONTROL_DIM * distribution_i]);
  control_array zero_mean = control_array::Zero();
  if (sample_index >= (1.0f - this->params_.pure_noise_trajectories_percentage) * this->params_.num_rollouts)
  {
    mean = zero_mean.data();
  }
  if (this->params_.time_specific_std_dev)
  {
    std_dev = &(this->params_.std_dev[(distribution_i * this->getNumTimesteps() + t) * CONTROL_DIM]);
  }
  for (int i = 0; i < CONTROL_DIM; i++)
  {
    cost += this->params_.control_cost_coeff[i] * mean[i] * (mean[i] - 2.0f * u(i)) /
            (std_dev[i] * std_dev[i]);  // Proper way
  }
  return 0.5f * lambda * (1.0f - alpha) * cost;
}

GAUSSIAN_TEMPLATE
__host__ void GAUSSIAN_CLASS::copyImportanceSamplerToDevice(const float* importance_sampler,
                                                            const int& distribution_idx, bool synchronize)
{
  HANDLE_ERROR(cudaMemcpyAsync(&control_means_d_[this->getNumTimesteps() * CONTROL_DIM * distribution_idx],
                               importance_sampler, sizeof(float) * this->getNumTimesteps() * CONTROL_DIM,
                               cudaMemcpyHostToDevice, this->stream_));
  if (synchronize)
  {
    HANDLE_ERROR(cudaStreamSynchronize(this->stream_));
  }
}
#undef GAUSSIAN_TEMPLATE
#undef GAUSSIAN_CLASS

}  // namespace sampling_distributions
}  // namespace mppi
