/*
 * Created on Sun Sep 28 2020 by Bogdan
 */

#ifndef FEEDBACK_CONTROLLERS_DDP_CUH_
#define FEEDBACK_CONTROLLERS_DDP_CUH_

#include <mppi/feedback_controllers/feedback.h>
#include <mppi/ddp/ddp_model_wrapper.h>
#include <mppi/ddp/ddp_tracking_costs.h>
#include <mppi/ddp/ddp.h>
#include <mppi/ddp/util.h>
#include <mppi/utils/math_utils.h>

template <class DYN_T>
struct DDPParams
{
  using StateCostWeight = typename TrackingCostDDP<ModelWrapperDDP<DYN_T>>::StateCostWeight;
  using Hessian = typename TrackingTerminalCost<ModelWrapperDDP<DYN_T>>::Hessian;
  using ControlCostWeight = typename TrackingCostDDP<ModelWrapperDDP<DYN_T>>::ControlCostWeight;

  StateCostWeight Q = StateCostWeight::Identity();
  Hessian Q_f = Hessian::Identity();
  ControlCostWeight R = ControlCostWeight::Identity();
  int num_iterations = 1;
};

template <class DYN_T, int N_TIMESTEPS>
struct DDPFeedbackState : GPUState
{
  static const int FEEDBACK_SIZE = DYN_T::CONTROL_DIM * DYN_T::STATE_DIM * N_TIMESTEPS;
  static const int NUM_TIMESTEPS = N_TIMESTEPS;

  /**
   * Variables
   **/
  float fb_gain_traj_[FEEDBACK_SIZE] MPPI_ALIGN(16) = { 0.0 }; // ensure it is aligned to 16 bytes

  /**
   * Methods
   **/
  bool isEqual(const DDPFeedbackState<DYN_T, N_TIMESTEPS>& other) const
  {
    for (int i = 0; i < FEEDBACK_SIZE; i++)
    {
      if (this->fb_gain_traj_[i] != other.fb_gain_traj_[i])
      {
        return false;
      }
    }
    return true;
  }
};

/**
 * Needed for Test in base_plant_tester.cu
 **/
template <class DYN_T, int N_TIMESTEPS>
bool operator==(const DDPFeedbackState<DYN_T, N_TIMESTEPS>& lhs, const DDPFeedbackState<DYN_T, N_TIMESTEPS>& rhs)
{
  return lhs.isEqual(rhs);
};

/**
 * DDP GPU Controller class starting point. This class is where the actual
 * methods for DDP on the GPU are implemented but it is not used directly since
 * setting up the GPU_FB_T value would be painful
 */
template <class GPU_FB_T, class DYN_T, int NUM_TIMESTEPS = 1>
class DeviceDDPImpl : public GPUFeedbackController<GPU_FB_T, DYN_T, DDPFeedbackState<DYN_T, NUM_TIMESTEPS>>
{
public:
  using PARAMS_T = DDPFeedbackState<DYN_T, NUM_TIMESTEPS>;
  // static const int SHARED_MEM_REQUEST_BLK_BYTES = DYN_T::CONTROL_DIM * DYN_T::STATE_DIM;
  DeviceDDPImpl(int num_timesteps, cudaStream_t stream = 0);
  DeviceDDPImpl(cudaStream_t stream = 0)
    : GPUFeedbackController<GPU_FB_T, DYN_T, DDPFeedbackState<DYN_T, NUM_TIMESTEPS>>(stream){};

  void allocateCUDAMemory(){};
  void deallocateCUDAMemory(){};

  __device__ void k(const float* __restrict__ x_act, const float* __restrict__  x_goal, const int t, float* __restrict__  theta, float* __restrict__  control_output);

protected:
  // Needed for allocating memory for feedback gains
  int num_timesteps_ = 1;
};

/**
 * Alias class for DDP GPU Controller. This sets up the class derivation correctly and is
 * used inside of the CPU version of DDP
 */
template <class DYN_T, int NUM_TIMESTEPS>
class DeviceDDP : public DeviceDDPImpl<DeviceDDP<DYN_T, NUM_TIMESTEPS>, DYN_T, NUM_TIMESTEPS>
{
public:
  DeviceDDP(int num_timesteps, cudaStream_t stream = 0)
    : DeviceDDPImpl<DeviceDDP<DYN_T, NUM_TIMESTEPS>, DYN_T, NUM_TIMESTEPS>(num_timesteps, stream){};

  DeviceDDP(cudaStream_t stream = 0) : DeviceDDPImpl<DeviceDDP<DYN_T, NUM_TIMESTEPS>, DYN_T, NUM_TIMESTEPS>(stream){};
};

/**
 * CPU Class for DDP. This is what the user should interact with
 */
template <class DYN_T, int NUM_TIMESTEPS>
class DDPFeedback : public FeedbackController<DeviceDDP<DYN_T, NUM_TIMESTEPS>, DDPParams<DYN_T>, NUM_TIMESTEPS>
{
public:
  /**
   * Aliases
   **/
  typedef util::EigenAlignedVector<float, DYN_T::CONTROL_DIM, DYN_T::STATE_DIM> feedback_gain_trajectory;
  typedef FeedbackController<DeviceDDP<DYN_T, NUM_TIMESTEPS>, DDPParams<DYN_T>, NUM_TIMESTEPS> PARENT_CLASS;

  using control_array = typename PARENT_CLASS::control_array;
  using state_array = typename PARENT_CLASS::state_array;
  using state_trajectory = typename PARENT_CLASS::state_trajectory;
  using control_trajectory = typename PARENT_CLASS::control_trajectory;
  using INTERNAL_STATE_T = typename PARENT_CLASS::TEMPLATED_FEEDBACK_STATE;
  using feedback_gain_matrix = typename DYN_T::feedback_matrix;
  using square_state_matrix = typename DDPParams<DYN_T>::StateCostWeight;
  using square_control_matrix = typename DDPParams<DYN_T>::ControlCostWeight;

  /**
   * Variables
   **/
  std::shared_ptr<ModelWrapperDDP<DYN_T>> ddp_model_;
  std::shared_ptr<TrackingCostDDP<ModelWrapperDDP<DYN_T>>> run_cost_;
  std::shared_ptr<TrackingTerminalCost<ModelWrapperDDP<DYN_T>>> terminal_cost_;
  std::shared_ptr<DDP<ModelWrapperDDP<DYN_T>>> ddp_solver_;
  OptimizerResult<ModelWrapperDDP<DYN_T>> result_;

  control_array control_min_;
  control_array control_max_;
  DYN_T* model_;

  DDPFeedback(DYN_T* model, float dt, int num_timesteps = NUM_TIMESTEPS, cudaStream_t stream = 0);

  void setParams(const DDPParams<DYN_T>& params) override;

  void initTrackingController();

  control_array k_(const Eigen::Ref<const state_array>& x_act, const Eigen::Ref<const state_array>& x_goal, int t,
                   INTERNAL_STATE_T& fb_state)
  {
    int index = DYN_T::STATE_DIM * DYN_T::CONTROL_DIM * t;
    Eigen::Map<feedback_gain_matrix> fb_gain(&(fb_state.fb_gain_traj_[index]));
    control_array u_output = fb_gain * (x_act - x_goal);
    return u_output;
  }

  feedback_gain_trajectory getFeedbackGainsEigen()
  {
    return result_.feedback_gain;
  }

  void computeFeedback(const Eigen::Ref<const state_array>& init_state,
                       const Eigen::Ref<const state_trajectory>& goal_traj,
                       const Eigen::Ref<const control_trajectory>& control_traj);
};

template <class GPU_FB_T, class DYN_T, int NUM_TIMESTEPS>
DeviceDDPImpl<GPU_FB_T, DYN_T, NUM_TIMESTEPS>::DeviceDDPImpl(int num_timesteps, cudaStream_t stream)
  : num_timesteps_(num_timesteps)
  , GPUFeedbackController<GPU_FB_T, DYN_T, DDPFeedbackState<DYN_T, NUM_TIMESTEPS>>(stream)
{
}

template <class GPU_FB_T, class DYN_T, int NUM_TIMESTEPS>
__device__ void DeviceDDPImpl<GPU_FB_T, DYN_T, NUM_TIMESTEPS>::k(const float* __restrict__ x_act,
                                                                 const float* __restrict__ x_goal, const int t,
                                                                 float* __restrict__ theta,
                                                                 float* __restrict__ control_output)
{
  float* fb_gain_t = &(this->state_.fb_gain_traj_[DYN_T::STATE_DIM * DYN_T::CONTROL_DIM * t]);
  float e = 0;
  for (int i = 0; i < DYN_T::STATE_DIM; i++)
  {
    e = x_act[i] - x_goal[i];
    if (DYN_T::CONTROL_DIM % 4 == 0)
    {  // load 4 floats in at a time to save on global memory reads
      float4* fb_gain_t4 = reinterpret_cast<float4*>(&fb_gain_t[i * DYN_T::CONTROL_DIM]);
      for (int j = 0; j < DYN_T::CONTROL_DIM / 4; j++)
      {
        reinterpret_cast<float4*>(control_output)[j] = fb_gain_t4[j] * e;
      }
    }
    else if (DYN_T::CONTROL_DIM % 2 == 0)
    {  // load 2 floats in at a time to save on global memory reads
      float2* fb_gain_t2 = reinterpret_cast<float2*>(&fb_gain_t[i * DYN_T::CONTROL_DIM]);
      for (int j = 0; j < DYN_T::CONTROL_DIM / 2; j++)
      {
        reinterpret_cast<float2*>(control_output)[j] = fb_gain_t2[j] * e;
      }
    }
    else
    {
      for (int j = 0; j < DYN_T::CONTROL_DIM; j++)
      {
        control_output[j] += fb_gain_t[i * DYN_T::CONTROL_DIM + j] * e;
      }
    }
  }
}

/**
 * CPU Class for DDP Methods
 */
template <class DYN_T, int NUM_TIMESTEPS>
DDPFeedback<DYN_T, NUM_TIMESTEPS>::DDPFeedback(DYN_T* model, float dt, int num_timesteps, cudaStream_t stream)
{
  model_ = model;
  this->dt_ = dt;
  this->num_timesteps_ = std::max(num_timesteps, NUM_TIMESTEPS);
  this->gpu_controller_->freeCudaMem();  // Remove allocated CUDA mem from default constructor
  this->gpu_controller_ = std::make_shared<DeviceDDP<DYN_T, NUM_TIMESTEPS>>(this->num_timesteps_, stream);
}

template <class DYN_T, int NUM_TIMESTEPS>
void DDPFeedback<DYN_T, NUM_TIMESTEPS>::initTrackingController()
{
  util::DefaultLogger logger;
  bool verbose = false;
  ddp_model_ = std::make_shared<ModelWrapperDDP<DYN_T>>(model_);
  ddp_solver_ = std::make_shared<DDP<ModelWrapperDDP<DYN_T>>>(this->dt_, this->num_timesteps_,
                                                              this->params_.num_iterations, &logger, verbose);

  result_ = OptimizerResult<ModelWrapperDDP<DYN_T>>();
  result_.feedback_gain = feedback_gain_trajectory(this->num_timesteps_);
  for (int i = 0; i < this->num_timesteps_; i++)
  {
    result_.feedback_gain[i] = DYN_T::feedback_matrix::Zero();
  }

  run_cost_ =
      std::make_shared<TrackingCostDDP<ModelWrapperDDP<DYN_T>>>(this->params_.Q, this->params_.R, this->num_timesteps_);
  terminal_cost_ = std::make_shared<TrackingTerminalCost<ModelWrapperDDP<DYN_T>>>(this->params_.Q_f);
}

template <class DYN_T, int NUM_TIMESTEPS>
void DDPFeedback<DYN_T, NUM_TIMESTEPS>::setParams(const DDPParams<DYN_T>& params)
{
  this->params_ = params;
  run_cost_ =
      std::make_shared<TrackingCostDDP<ModelWrapperDDP<DYN_T>>>(this->params_.Q, this->params_.R, this->num_timesteps_);
  terminal_cost_ = std::make_shared<TrackingTerminalCost<ModelWrapperDDP<DYN_T>>>(this->params_.Q_f);
}

template <class DYN_T, int NUM_TIMESTEPS>
void DDPFeedback<DYN_T, NUM_TIMESTEPS>::computeFeedback(const Eigen::Ref<const state_array>& init_state,
                                                        const Eigen::Ref<const state_trajectory>& goal_traj,
                                                        const Eigen::Ref<const control_trajectory>& control_traj)
{
  run_cost_->setTargets(goal_traj.data(), control_traj.data(), this->num_timesteps_);

  terminal_cost_->xf = run_cost_->traj_target_x_.col(this->num_timesteps_ - 1);

  // update control ranges
  for (int i = 0; i < DYN_T::CONTROL_DIM; i++)
  {
    control_min_(i) = model_->control_rngs_[i].x;
    control_max_(i) = model_->control_rngs_[i].y;
  }

  result_ =
      ddp_solver_->run(init_state, control_traj, *ddp_model_, *run_cost_, *terminal_cost_, control_min_, control_max_);

  // Copy Feedback Gains into Feedback State
  for (size_t i = 0; i < result_.feedback_gain.size(); i++)
  {
    int i_index = i * DYN_T::STATE_DIM * DYN_T::CONTROL_DIM;
    for (size_t j = 0; j < DYN_T::CONTROL_DIM * DYN_T::STATE_DIM; j++)
    {
      this->getFeedbackStatePointer()->fb_gain_traj_[i_index + j] = result_.feedback_gain[i].data()[j];
    }
  }
  // Actually put new feedback gain trajectory onto the GPU
  // this->gpu_controller_->copyToDevice();
}

#endif  // FEEDBACK_CONTROLLERS_DDP_CUH_
