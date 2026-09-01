#include"mppi-generic-controller.h"

















struct GPUState
{
};

/**
 * This is the begining of the mess. So we have a GPU Feedback class that houses the methods and data
 * that are used on the GPU. The first template argument is for that cool thing that Jason knows the name of.
 * The second template argument is the dynamics class so that we can automatically pull out the state and control
 * dims needed. The final template argument is the state class. This state class is what will be passed back and forth
 * from the CPU to the GPU and vice versa (if necessary). For example, this would be your k_p, k_d, and k_i terms for a
 * PID, or the trajectory of feedback gains for DDP.
 *
 * Things a new controller will need:
 *   - a k method. This is how to use the feedback controller on the GPU
 *   - a GPU_STATE_T class. This should contain the relevant data to transfer from CPU to GPU and vice versa
 * Optional Things to implement:
 *   - copyFromDevice(), copyToDevice(), paramsToDevice(): if your data structure FEEDBACK_STATE_T is complex, you will
 * need to implement these yourself
 *   - allocateCUDAMemory(), deallocateCUDAMemory(): If your feedback class needs some CUDA memory beyond the
 * FEEDBACK_STATE_T, you will need to create and clear it here.
 */
template <class GPU_FB_T, class TEMPLATED_DYNAMICS, class GPU_STATE_T>
class GPUFeedbackController : public Managed
{
public:
  /**
   * Type Aliasing
   */
  using DYN_T = TEMPLATED_DYNAMICS;
  using FEEDBACK_STATE_T = GPU_STATE_T;

  GPU_FB_T* feedback_d_ = nullptr;

  /**
   * Constructors
   */

  GPUFeedbackController() = default;

  GPUFeedbackController(cudaStream_t stream = 0) : Managed(stream)
  {
    this->SHARED_MEM_REQUEST_GRD_BYTES = 0;
    this->SHARED_MEM_REQUEST_BLK_BYTES = 0;
  }

  /**
   * =================== METHODS THAT SHOULD NOT BE OVERWRITTEN ================
   */
  virtual ~GPUFeedbackController()
  {
    freeCudaMem();
  };

  // Overwrite of Managed->GPUSetup to call allocateCUDAMemory as well
  void GPUSetup()
  {
    GPU_FB_T& derived = static_cast<GPU_FB_T&>(*this);
    if (!GPUMemStatus_)
    {
      feedback_d_ = Managed::GPUSetup(derived);
      derived->allocateCUDAMemory();
    }
    else
    {
      this->logger_->debug("Feedback Controller GPU Memory already set\n");
    }
    derived->copyToDevice();
  }
  void freeCudaMem()
  {
    if (GPUMemStatus_)
    {
      GPU_FB_T& derived = static_cast<GPU_FB_T&>(*this);
      derived->deallocateCUDAMemory();
      cudaFree(feedback_d_);
      GPUMemStatus_ = false;
      feedback_d_ = nullptr;
    }
  }

  void setFeedbackState(const FEEDBACK_STATE_T& state)
  {
    state_ = state;
    if (GPUMemStatus_)
    {
      GPU_FB_T& derived = static_cast<GPU_FB_T&>(*this);
      derived.copyToDevice();
    }
  }

  __host__ __device__ FEEDBACK_STATE_T getFeedbackState()
  {
    return state_;
  }

  __host__ __device__ FEEDBACK_STATE_T* getFeedbackStatePointer()
  {
    return &state_;
  }

  /**
   * ==================== NECESSARY METHODS TO OVERWRITE =====================
   */
  __device__ void k(const float* __restrict__ x_act, const float* __restrict__  x_goal, const int t, float* __restrict__  theta, float* __restrict__  control_output)
  {
  }
  /**
   * ===================== OPTIONAL METHODS TO OVERWRITE ======================
   */
  /**
   * Only needed to allocate/deallocate additional CUDA memory when appropriate,
   * GPU pointer is already handled.
   */
  void allocateCUDAMemory()
  {
  }
  void deallocateCUDAMemory()
  {
  }

  __device__ void initializeFeedback(const float* __restrict__ x, const float* __restrict__ u, float* __restrict__ theta, const float t, const float dt)
  {}

  // Abstract method to copy information to GPU
  // void copyToDevice() {}

  // Copies the params to the device at the moment
  void copyToDevice(bool synchronize = true)
  {
    if (GPUMemStatus_)
    {
      HANDLE_ERROR(
          cudaMemcpyAsync(&feedback_d_->state_, &state_, sizeof(FEEDBACK_STATE_T), cudaMemcpyHostToDevice, stream_));
      if (synchronize)
      {
        HANDLE_ERROR(cudaStreamSynchronize(stream_));
      }
    }
  }
  // Method to return potential diagnostic information from GPU
  void copyFromDevice(bool synchronize = true)
  {
  }

protected:
  FEEDBACK_STATE_T state_;
};

/**
 * Steps to making a new one
 * Create the GPUFeedback class as an impl class like costs but is still templated on DYN
 * The actual GPUFeedback_act class will then be templated on DYN and inherit from the GPUFeedbackImpl
 * Write the feedback controller to use the GPUFeedback_act as thee GPU_FEEDBACK_T template option
 * It will then automatically create the right pointer
 */
template <class GPU_FB_T, class PARAMS_T, int NUM_TIMESTEPS>
class FeedbackController
{
public:
  // Type Defintions and aliases
  typedef typename GPU_FB_T::DYN_T DYN_T;
  // typedef FEEDBACK_STATE_T TEMPLATED_FEEDBACK_STATE;
  typedef PARAMS_T TEMPLATED_PARAMS;
  typedef GPU_FB_T TEMPLATED_GPU_FEEDBACK;
  typedef typename GPU_FB_T::FEEDBACK_STATE_T TEMPLATED_FEEDBACK_STATE;
  static const int FB_TIMESTEPS = NUM_TIMESTEPS;

  using state_array = typename DYN_T::state_array;
  using control_array = typename DYN_T::control_array;
  typedef Eigen::Matrix<float, DYN_T::CONTROL_DIM,
                        NUM_TIMESTEPS> control_trajectory;  // A control trajectory
  typedef Eigen::Matrix<float, DYN_T::STATE_DIM,
                        NUM_TIMESTEPS> state_trajectory;  // A state trajectory

  // Constructors and Generators
  FeedbackController(float dt = 0.01, int num_timesteps = NUM_TIMESTEPS, cudaStream_t stream = 0)
    : dt_(dt), num_timesteps_(num_timesteps)
  {
    gpu_controller_ = std::make_shared<GPU_FB_T>(stream);
    auto logger = std::make_shared<mppi::util::MPPILogger>();
    setLogger(logger);
  }

  virtual ~FeedbackController()
  {
    freeCudaMem();
  };

  virtual __host__ void GPUSetup()
  {
    gpu_controller_->GPUSetup();
  }

  virtual __host__ void freeCudaMem()
  {
    gpu_controller_->freeCudaMem();
  }

  virtual __host__ void initTrackingController() = 0;

  virtual __host__ void setParams(const PARAMS_T& params)
  {
    params_ = params;
  }

  PARAMS_T getParams()
  {
    return params_;
  }

  // CPU Methods
  /**
   * Compute feedback control method that should not be overwritten.
   * Input:
   *  - x_act: the state where the system is
   *  - x_goal: the state we want to be at
   *  - index: the number of timesteps from the initial time we are
   */
  virtual __host__ control_array k(const Eigen::Ref<const state_array>& x_act, const Eigen::Ref<const state_array>& x_goal,
                          int t)
  {
    TEMPLATED_FEEDBACK_STATE* gpu_feedback_state = getFeedbackStatePointer();
    return k_(x_act, x_goal, t, *gpu_feedback_state);
  }
  /**
   * Feeback Control Method to overwrite.
   */
  virtual __host__ control_array k_(const Eigen::Ref<const state_array>& x_act, const Eigen::Ref<const state_array>& x_goal,
                           int t, TEMPLATED_FEEDBACK_STATE& fb_state) = 0;

  // might not be a needed method
  virtual __host__ void computeFeedback(const Eigen::Ref<const state_array>& init_state,
                               const Eigen::Ref<const state_trajectory>& goal_traj,
                               const Eigen::Ref<const control_trajectory>& control_traj) = 0;

  // TODO Construct a default version of this method that uses the state_ variable automatically
  virtual __host__ control_array interpolateFeedback_(const Eigen::Ref<const state_array>& state,
                                             const Eigen::Ref<const state_array>& goal_state, double rel_time,
                                             TEMPLATED_FEEDBACK_STATE& fb_state)
  {
    int lower_idx = (int)(rel_time / dt_);
    int upper_idx = lower_idx + 1;
    double alpha = (rel_time - lower_idx * dt_) / dt_;

    control_array u_fb =
        (1 - alpha) * k_(state, goal_state, lower_idx, fb_state) + alpha * k_(state, goal_state, upper_idx, fb_state);

    return u_fb;
  }

  virtual __host__ control_array interpolateFeedback(const Eigen::Ref<const state_array>& state,
                                            const Eigen::Ref<const state_array>& goal_state, double rel_time)
  {
    TEMPLATED_FEEDBACK_STATE* fb_state = getFeedbackStatePointer();
    return interpolateFeedback_(state, goal_state, rel_time, *fb_state);
  }

  GPU_FB_T* getDevicePointer()
  {
    return gpu_controller_->feedback_d_;
  }

  std::shared_ptr<GPU_FB_T> getHostPointer()
  {
    return gpu_controller_;
  }

  void bindToStream(cudaStream_t stream)
  {
    gpu_controller_->bindToStream(stream);
  }

  /**
   * Calls GPU version
   */
  void copyToDevice(bool synchronize = true)
  {
    this->gpu_controller_->copyToDevice(synchronize);
  }

  TEMPLATED_FEEDBACK_STATE getFeedbackState()
  {
    return this->gpu_controller_->getFeedbackState();
  }

  TEMPLATED_FEEDBACK_STATE* getFeedbackStatePointer()
  {
    return this->gpu_controller_->getFeedbackStatePointer();
  }

  void setFeedbackState(const TEMPLATED_FEEDBACK_STATE& gpu_fb_state)
  {
    this->gpu_controller_->setFeedbackState(gpu_fb_state);
  }

  float getDt()
  {
    return dt_;
  }
  void setDt(float dt)
  {
    dt_ = dt;
  }

  __host__ void setLogger(const mppi::util::MPPILoggerPtr& logger)
  {
    logger_ = logger;
    gpu_controller_->setLogger(logger);
  }

  __host__ void setLogLevel(const mppi::util::LOG_LEVEL& level)
  {
    logger_->setLogLevel(level);
    gpu_controller_->setLogLevel(level);
  }

  __host__ mppi::util::MPPILoggerPtr getLogger()
  {
    return logger_;
  }

  __host__ mppi::util::MPPILoggerPtr getLogger() const
  {
    return logger_;
  }

protected:
  std::shared_ptr<GPU_FB_T> gpu_controller_;
  float dt_;
  int num_timesteps_;
  PARAMS_T params_;
  mppi::util::MPPILoggerPtr logger_ = nullptr;
};















// helpful macros to use the enum setup
#ifndef E_INDEX
#define E_INDEX(ENUM, enum_val) static_cast<int>(ENUM::enum_val)
#endif
#ifndef S_INDEX
#define S_IND_CLASS(CLASS, enum_val) E_INDEX(CLASS::StateIndex, enum_val)
#define S_INDEX(enum_val) S_IND_CLASS(PARENT_CLASS::DYN_PARAMS_T, enum_val)
#endif

#ifndef C_INDEX
#define C_IND_CLASS(CLASS, enum_val) E_INDEX(CLASS::ControlIndex, enum_val)
#define C_INDEX(enum_val) C_IND_CLASS(PARENT_CLASS::DYN_PARAMS_T, enum_val)
#endif

#ifndef O_INDEX
#define O_IND_CLASS(CLASS, enum_val) E_INDEX(CLASS::OutputIndex, enum_val)
#define O_INDEX(enum_val) O_IND_CLASS(PARENT_CLASS::DYN_PARAMS_T, enum_val)
#endif

struct DynamicsParams
{
  enum class StateIndex : int
  {
    POS_X = 0,
    NUM_STATES
  };
  enum class ControlIndex : int
  {
    VEL_X = 0,
    NUM_CONTROLS
  };
  enum class OutputIndex : int
  {
    POS_X = 0,
    NUM_OUTPUTS
  };
};

template <typename T>
using paramsInheritsFrom = typename std::is_base_of<DynamicsParams, T>;

namespace MPPI_internal
{
template <class CLASS_T, class PARAMS_T>
class Dynamics : public Managed
{
  static_assert(paramsInheritsFrom<PARAMS_T>::value, "Dynamics PARAMS_T does not inherit from DynamicsParams");

public:
  //  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  static const int STATE_DIM = S_IND_CLASS(PARAMS_T, NUM_STATES);
  static const int CONTROL_DIM = C_IND_CLASS(PARAMS_T, NUM_CONTROLS);
  static const int OUTPUT_DIM = O_IND_CLASS(PARAMS_T, NUM_OUTPUTS);
  typedef CLASS_T DYN_T;
  typedef PARAMS_T DYN_PARAMS_T;

  /**
   * useful typedefs
   */
  typedef Eigen::Matrix<float, CONTROL_DIM, 1> control_array;                 // Control at a time t
  typedef Eigen::Matrix<float, STATE_DIM, 1> state_array;                     // State at a time t
  typedef Eigen::Matrix<float, OUTPUT_DIM, 1> output_array;                   // Output at a time t
  typedef Eigen::Matrix<float, STATE_DIM, STATE_DIM> dfdx;                    // Jacobian wrt x
  typedef Eigen::Matrix<float, STATE_DIM, CONTROL_DIM> dfdu;                  // Jacobian wrt u
  typedef Eigen::Matrix<float, CONTROL_DIM, STATE_DIM> feedback_matrix;       // Feedback matrix
  typedef Eigen::Matrix<float, STATE_DIM, STATE_DIM + CONTROL_DIM> Jacobian;  // Jacobian of x and u

  typedef std::map<std::string, Eigen::VectorXf> buffer_trajectory;

  // protected constructor prevent anyone from trying to construct a Dynamics
protected:
  /**
   * sets the default control ranges to -infinity and +infinity
   */
  Dynamics(cudaStream_t stream = 0) : Managed(stream)
  {
    setDefaultControlRanges();
  }

  /**
   * sets the control ranges to the passed in value
   * @param control_rngs
   * @param stream
   */
  Dynamics(std::array<float2, CONTROL_DIM>& control_rngs, cudaStream_t stream = 0) : Managed(stream)
  {
    setControlRanges(control_rngs);
  }

  Dynamics(PARAMS_T& params, std::array<float2, CONTROL_DIM>& control_rngs, cudaStream_t stream = 0) : Managed(stream)
  {
    setParams(params);
    setControlRanges(control_rngs);
  }

  Dynamics(PARAMS_T& params, cudaStream_t stream = 0) : Managed(stream)
  {
    setParams(params);
    setDefaultControlRanges();
  }

public:
  // This variable defines what the zero control is
  // For most systems, it should be zero but for things like a quadrotor,
  // it should be the command to hover
  control_array zero_control_ = control_array::Zero();

  /**
   * Destructor must be virtual so that children are properly
   * destroyed when called from a Dynamics reference
   */
  virtual ~Dynamics()
  {
    freeCudaMem();
  }

  virtual std::string getDynamicsModelName() const
  {
    return "Dynamics model name not set";
  }

  /**
   * Allocates all of the GPU memory
   */
  void GPUSetup()
  {
    CLASS_T* derived = static_cast<CLASS_T*>(this);
    if (!GPUMemStatus_)
    {
      model_d_ = Managed::GPUSetup(derived);
    }
    else
    {
      this->logger_->debug("%s: GPU Memory already set\n", derived->getDynamicsModelName().c_str());
    }
    derived->paramsToDevice();
  }

  void setZeroControl(const Eigen::Ref<const control_array>& zero_control)
  {
    zero_control_ = zero_control;
  }
  control_array getZeroControl() const
  {
    return zero_control_;
  }

  std::array<float2, CONTROL_DIM> getControlRanges()
  {
    std::array<float2, CONTROL_DIM> result;
    for (int i = 0; i < CONTROL_DIM; i++)
    {
      result[i] = control_rngs_[i];
    }
    return result;
  }
  __host__ __device__ float2* getControlRangesRaw()
  {
    return control_rngs_;
  }

  void setControlRanges(std::array<float2, CONTROL_DIM>& control_rngs, bool synchronize = true)
  {
    for (int i = 0; i < CONTROL_DIM; i++)
    {
      control_rngs_[i].x = control_rngs[i].x;
      control_rngs_[i].y = control_rngs[i].y;
    }
    if (GPUMemStatus_)
    {
      HANDLE_ERROR(cudaMemcpyAsync(this->model_d_->control_rngs_, this->control_rngs_, CONTROL_DIM * sizeof(float2),
                                   cudaMemcpyHostToDevice, stream_));
      if (synchronize)
      {
        HANDLE_ERROR(cudaStreamSynchronize(stream_));
      }
    }
  }

  void setControlDeadbands(std::array<float, CONTROL_DIM>& control_deadband, bool synchronize = true)
{
  for (int i = 0; i < CONTROL_DIM; i++)
  {
    control_deadband_[i] = control_deadband[i];
  }
  if (GPUMemStatus_)
  {
    HANDLE_ERROR(cudaMemcpyAsync(this->model_d_->control_deadband_, this->control_deadband_,
                                 CONTROL_DIM * sizeof(float), cudaMemcpyHostToDevice, stream_));
    if (synchronize)
    {
      HANDLE_ERROR(cudaStreamSynchronize(stream_));
    }
  }
}


  void setParams(const PARAMS_T& params)
  {
    params_ = params;
    if (GPUMemStatus_)
    {
      CLASS_T& derived = static_cast<CLASS_T&>(*this);
      derived.paramsToDevice();
    }
  }

  __device__ __host__ PARAMS_T getParams()
  {
    return params_;
  }

  /*
   *
   */
  void freeCudaMem()
  {
    if (GPUMemStatus_)
    {
      HANDLE_ERROR(cudaFree(model_d_));
      GPUMemStatus_ = false;
      model_d_ = nullptr;
    }
  }

  /**
   *
   */
  void printState(float* state);

  /**
   *
   */
  // TODO should not assume it is going to cout, pass in stream
  void printParams();

  /**
   *
   */

  void paramsToDevice(bool synchronize)
  {
    if (GPUMemStatus_)
    {
      HANDLE_ERROR(cudaMemcpyAsync(&model_d_->params_, &params_, sizeof(PARAMS_T), cudaMemcpyHostToDevice, stream_));

      HANDLE_ERROR(cudaMemcpyAsync(&model_d_->control_rngs_, &control_rngs_, CONTROL_DIM * sizeof(float2),
                                   cudaMemcpyHostToDevice, stream_));
      if (synchronize)
      {
        HANDLE_ERROR(cudaStreamSynchronize(stream_));
      }
    }
  }

  /**
   * loads the .npz at given path
   * @param model_path
   */
  void loadParams(const std::string& model_path);

  /**
   * updates the internals of the dynamics model
   * @param description
   * @param data
   */
  // TODO generalize
  void updateModel(std::vector<int> description, std::vector<float> data);

  /**
   * compute the Jacobians with respect to state and control
   *
   * @param state   input of current state, passed by reference
   * @param control input of currrent control, passed by reference
   * @param A       output Jacobian wrt state, passed by reference
   * @param B       output Jacobian wrt control, passed by reference
   */
  bool computeGrad(const Eigen::Ref<const state_array>& state = state_array(),
                   const Eigen::Ref<const control_array>& control = control_array(), Eigen::Ref<dfdx> A = dfdx(),
                   Eigen::Ref<dfdu> B = dfdu())
  {
    return false;
  }
  /**
   * enforces control constraints
   * @param state
   * @param control
   */
  void enforceConstraints(Eigen::Ref<state_array> state, Eigen::Ref<control_array> control)
  {
    for (int i = 0; i < CONTROL_DIM; i++)
    {
      if (fabsf(control[i]) < this->control_deadband_[i])
      {
        control[i] = this->zero_control_[i];
      }
      else
      {
        control[i] += this->control_deadband_[i] * -mppi::math::sign(control[i]);
      }
      control[i] = fminf(fmaxf(this->control_rngs_[i].x, control[i]), this->control_rngs_[i].y);
    }
  }

  /**
   * updates the current state using s_der
   * @param s state
   * @param s_der
   */
  DEPRECATED void updateState(Eigen::Ref<state_array> state, Eigen::Ref<state_array> state_der, const float dt)
  {
    CLASS_T* derived = static_cast<CLASS_T*>(this);
    derived->updateState(state, state, state_der, dt);
  }

  void updateState(const Eigen::Ref<const state_array> state, Eigen::Ref<state_array> next_state,
                   Eigen::Ref<state_array> state_der, const float dt)
  {
    next_state = state + state_der * dt;
  }

  void step(Eigen::Ref<state_array> state, Eigen::Ref<state_array> next_state, Eigen::Ref<state_array> state_der,
            const Eigen::Ref<const control_array>& control, Eigen::Ref<output_array> output, const float t,
            const float dt)
  {
    CLASS_T* derived = static_cast<CLASS_T*>(this);
    derived->computeStateDeriv(state, control, state_der);
    derived->updateState(state, next_state, state_der, dt);
    derived->stateToOutput(next_state, output);
  }

  void stateToOutput(const Eigen::Ref<const state_array>& state, Eigen::Ref<output_array> output)
  {
    // TODO this is a hack
    for (int i = 0; i < OUTPUT_DIM && i < STATE_DIM; i++)
    {
      output[i] = state[i];
    }
  }

  __host__ __device__ void stateToOutput(const float* __restrict__ state, float* __restrict__ output)

  {
    // TODO this is a hack
    int p_index, step;
    mppi::p1::getParallel1DIndex<mppi::p1::Parallel1Dir::THREAD_Y>(p_index, step);
    for (int i = p_index; i < OUTPUT_DIM && i < STATE_DIM; i += step)
    {
      output[i] = state[i];
    }
  }

  __host__ __device__ void outputToState(const float* __restrict__ output, float* __restrict__ state)

  {
    // TODO this is a hack
    int p_index, step;
    mppi::p1::getParallel1DIndex<mppi::p1::Parallel1Dir::THREAD_Y>(p_index, step);
    for (int i = p_index; i < OUTPUT_DIM && i < STATE_DIM; i += step)
    {
      state[i] = output[i];
    }
  }


  state_array getZeroState() const
  {
    return state_array::Zero();
  }

  __host__ __device__ void getZeroState(float* state) const
  {
    int p_index, step;
    mppi::p1::getParallel1DIndex<mppi::p1::Parallel1Dir::THREAD_Y>(p_index, step);
    for (int i = p_index; i < STATE_DIM; i += step)
    {
      state[i] = 0.0f;
    }
  }

  /**
   * does a linear interpolation of states
   * @param state_1
   * @param state_2
   * @param alpha
   * @return
   */
  state_array interpolateState(const Eigen::Ref<const state_array>& state_1,
                               const Eigen::Ref<const state_array>& state_2, const float alpha)
  {
    return (1 - alpha) * state_1 + alpha * state_2;
  }

  /**
   * computes a specific state error
   * @param pred_state
   * @param true_state
   * @return
   */
  state_array computeStateError(const Eigen::Ref<const state_array>& pred_state,
                                const Eigen::Ref<const state_array>& true_state)
  {
    return pred_state - true_state;
  }

  /**
   * computes the section of the state derivative that comes form the dyanmics
   * @param state
   * @param control
   * @param state_der
   */
  void computeDynamics(const Eigen::Ref<const state_array>& state, const Eigen::Ref<const control_array>& control,
                       Eigen::Ref<state_array> state_der);

  /**
   * computes the parts of the state that are based off of kinematics
   * @param s state
   * @param s_der
   */
  void computeKinematics(const Eigen::Ref<const state_array>& state, Eigen::Ref<state_array> s_der){};

  /**
   * computes the full state derivative by calling computeKinematics then computeDynamics
   * @param state
   * @param control
   * @param state_der
   */
  void computeStateDeriv(const Eigen::Ref<const state_array>& state, const Eigen::Ref<const control_array>& control,
                         Eigen::Ref<state_array> state_der)
  {
    CLASS_T* derived = static_cast<CLASS_T*>(this);
    derived->computeKinematics(state, state_der);
    derived->computeDynamics(state, control, state_der);
  }

  /**
   * computes the section of the state derivative that comes form the dyanmics
   * @param state
   * @param control
   * @param state_der
   * @param theta_s shared memory that can be used when computation is computed across the same block
   */
  __device__ void computeDynamics(float* state, float* control, float* state_der, float* theta_s = nullptr);

  /**
   * computes the parts of the state that are based off of kinematics
   * parallelized on X only
   * @param state
   * @param state_der
   */
  __device__ void computeKinematics(float* state, float* state_der){};

  /**
   * @param state
   * @param control
   * @param state_der
   * @param theta_s shared memory that can be used when computation is computed across the same block
   */
  __device__ inline void computeStateDeriv(float* state, float* control, float* state_der, float* theta_s)
  {
    CLASS_T* derived = static_cast<CLASS_T*>(this);
    // only propagate a single state, i.e. thread.y = 0
    // find the change in x,y,theta based off of the rest of the state
    if (threadIdx.y == 0)
    {
      derived->computeKinematics(state, state_der);
    }
    derived->computeDynamics(state, control, state_der, theta_s);
  }

  __device__ inline void step(float* state, float* next_state, float* state_der, float* control, float* output,float* theta_s, const float t, const float dt)
  {
    CLASS_T* derived = static_cast<CLASS_T*>(this);
    derived->computeStateDeriv(state, control, state_der, theta_s);
    __syncthreads();
    derived->updateState(state, next_state, state_der, dt);
    __syncthreads();
    derived->stateToOutput(next_state, output);
  }

  /**
   * applies the state derivative
   * @param state
   * @param state_der
   * @param dt
   */
  DEPRECATED __device__ void updateState(float* state, float* state_der, const float dt)
  {
    CLASS_T* derived = static_cast<CLASS_T*>(this);
    derived->updateState(state, state, state_der, dt);
  }

  __device__ void updateState(float* state, float* next_state, float* state_der, const float dt)
  {
    int i, p_index, step;
    mppi::p1::getParallel1DIndex<mppi::p1::Parallel1Dir::THREAD_Y>(p_index, step);
    // Add the state derivative time dt to the current state.
    for (i = p_index; i < STATE_DIM; i += step)
    {
      next_state[i] = state[i] + state_der[i] * dt;
    }
  }

  /**
   * enforces control constraints
   */
  __device__ void enforceConstraints(float* state, float* control)

  {
    // TODO should control_rngs_ be a constant memory parameter
    int i, p_index, step;
    mppi::p1::getParallel1DIndex<mppi::p1::Parallel1Dir::THREAD_Y>(p_index, step);
    // parallelize setting the constraints with y dim
    for (i = p_index; i < CONTROL_DIM; i += step)
    {
      if (fabsf(control[i]) < this->control_deadband_[i])
      {
        control[i] = this->zero_control_[i];
      }
      else
      {
        control[i] += this->control_deadband_[i] * -mppi::math::sign(control[i]);
      }
      control[i] = fminf(fmaxf(this->control_rngs_[i].x, control[i]), this->control_rngs_[i].y);
    }
  }


  /**
   * Method to allow setup of dynamics on the GPU. This is needed for
   * initializing the memory of an LSTM for example
   */
  void initializeDynamics(const Eigen::Ref<const state_array>& state, const Eigen::Ref<const control_array>& control,
                          Eigen::Ref<output_array> output, float t_0, float dt)
  {
    for (int i = 0; i < OUTPUT_DIM && i < STATE_DIM; i++)
    {
      output[i] = state[i];
    }
  }

  /**
   * Method to allow setup of dynamics on the GPU. This is needed for
   * initializing the memory of an LSTM for example
   */
  __device__ void initializeDynamics(float* state, float* control, float* output, float* theta_s, float t_0, float dt)
  {
    for (int i = 0; i < OUTPUT_DIM && i < STATE_DIM; i++)
    {
      output[i] = state[i];
    }
  }

  /**
   * Method to compute an emergency stopping control
   */
  virtual void getStoppingControl(const Eigen::Ref<const state_array>& state, Eigen::Ref<control_array> u)
  {
    u.setZero();
  }

  /**
   * Method to enforce a leash on the initial state, which depends on type of dynamics.
   */
  virtual void enforceLeash(const Eigen::Ref<const state_array>& state_true,
                            const Eigen::Ref<const state_array>& state_nominal,
                            const Eigen::Ref<const state_array>& leash_values, Eigen::Ref<state_array> state_output)
  {
    for (int i = 0; i < DYN_T::STATE_DIM; i++)
    {
      float diff = fabsf(state_nominal[i] - state_true[i]);

      if (leash_values[i] < diff)
      {
        float leash_dir = fminf(fmaxf(state_nominal[i] - state_true[i], -leash_values[i]), leash_values[i]);
        state_output[i] = state_true[i] + leash_dir;
      }
      else
      {
        state_output[i] = state_nominal[i];
      }
    }
  }

  virtual bool checkRequiresBuffer()
  {
    return requires_buffer_;
  }

  bool updateFromBuffer(const buffer_trajectory& buffer)
  {
    return true;
  }

  bool checkIfKeysInBuffer(const buffer_trajectory& buffer, std::vector<std::string>& needed_keys)
  {
    bool missing_key = false;
    for (const auto& key : needed_keys)
    {
      if (buffer.find(key) == buffer.end())
      {  // Print out all missing keys
        this->logger_->warning("Could not find key %s in init buffer for %s\n", key.c_str(),
                               this->getDynamicsModelName().c_str());
        missing_key = true;
      }
    }
    return missing_key;
  }

  bool checkIfKeysInMap(const std::map<std::string, float>& map, std::vector<std::string>& needed_keys)
  {
    bool missing_key = false;
    for (const auto& key : needed_keys)
    {
      if (map.find(key) == map.end())
      {  // Print out all missing keys
        this->logger_->warning("Could not find key %s in state map for %s\n", key.c_str(),
                               this->getDynamicsModelName().c_str());
        missing_key = true;
      }
    }
    return missing_key;
  }

  virtual state_array stateFromMap(const std::map<std::string, float>& map) = 0;

  // control ranges [.x, .y]
  float2 control_rngs_[CONTROL_DIM];
  float control_deadband_[CONTROL_DIM] = { 0.0f };

  // device pointer, null on the device
  CLASS_T* model_d_ = nullptr;

protected:
  // generic parameter structure
  PARAMS_T params_;

  bool requires_buffer_ = false;

private:

  void setDefaultControlRanges()
  {
    for (int i = 0; i < CONTROL_DIM; i++)
    {
      control_rngs_[i].x = -FLT_MAX;
      control_rngs_[i].y = FLT_MAX;
    }
  }

};

template <class CLASS_T, class PARAMS_T>
const int Dynamics<CLASS_T, PARAMS_T>::STATE_DIM;

template <class CLASS_T, class PARAMS_T>
const int Dynamics<CLASS_T, PARAMS_T>::CONTROL_DIM;

template <class CLASS_T, class PARAMS_T>
const int Dynamics<CLASS_T, PARAMS_T>::OUTPUT_DIM;
}  // namespace MPPI_internal










template <int C_DIM>
struct CostParams
{
  static const int CONTROL_DIM = C_DIM;
  float control_cost_coeff[C_DIM];
  float discount = 1.0f;
  CostParams()
  {
    // Default set all controls to 1
    for (int i = 0; i < C_DIM; ++i)
    {
      control_cost_coeff[i] = 1.0f;
    }
  }
};

// https://cboard.cprogramming.com/cplusplus-programming/122412-crtp-how-pass-type.html
template <class CLASS_T, class PARAMS_T, class DYN_PARAMS_T = DynamicsParams>
class Cost : public Managed
{
public:
  //  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
   * typedefs for access to templated class from outside classes
   */
  using ControlIndex = typename DYN_PARAMS_T::ControlIndex;
  using OutputIndex = typename DYN_PARAMS_T::OutputIndex;
  using TEMPLATED_DYN_PARAMS = DYN_PARAMS_T;
  static const int CONTROL_DIM = E_INDEX(ControlIndex, NUM_CONTROLS);
  static const int OUTPUT_DIM = E_INDEX(OutputIndex, NUM_OUTPUTS);  // TODO
  typedef CLASS_T COST_T;
  typedef PARAMS_T COST_PARAMS_T;
  typedef Eigen::Matrix<float, CONTROL_DIM, 1> control_array;             // Control at a time t
  typedef Eigen::Matrix<float, CONTROL_DIM, CONTROL_DIM> control_matrix;  // Control at a time t
  typedef Eigen::Matrix<float, OUTPUT_DIM, 1> output_array;               // Output at a time t

  Cost() = default;
  /**
   * Destructor must be virtual so that children are properly
   * destroyed when called from a basePlant reference
   */
  virtual ~Cost()
  {
    freeCudaMem();
  }

  virtual std::string getCostFunctionName() const
  {
    return "cost function name not set";
  }

  void GPUSetup()
  {
    CLASS_T* derived = static_cast<CLASS_T*>(this);
    if (!GPUMemStatus_)
    {
      cost_d_ = Managed::GPUSetup<CLASS_T>(derived);
    }
    else
    {
      this->logger_->debug("%s: GPU Memory already set.\n", derived->getCostFunctionName().c_str());
    }
    derived->paramsToDevice();
  }


  bool getDebugDisplayEnabled()
  {
    return false;
  }

  /**
   * Updates the cost parameters
   * @param params
   */
  void setParams(const PARAMS_T& params)
  {
    params_ = params;
    if (GPUMemStatus_)
    {
      CLASS_T& derived = static_cast<CLASS_T&>(*this);
      derived.paramsToDevice();
    }
  }

  __host__ __device__ PARAMS_T getParams()
  {
    return params_;
  }

  void paramsToDevice()
  {
    if (GPUMemStatus_)
    {
      HANDLE_ERROR(cudaMemcpyAsync(&cost_d_->params_, &params_, sizeof(PARAMS_T), cudaMemcpyHostToDevice, stream_));
      HANDLE_ERROR(cudaStreamSynchronize(stream_));
    }
  }

  /**
   *
   * @param description
   * @param data
   */
  void updateCostmap(std::vector<int> description, std::vector<float> data){};

  /**
   * deallocates the allocated cuda memory for an object
   */
  void freeCudaMem()
  {
    if (GPUMemStatus_)
    {
      cudaFree(cost_d_);
      GPUMemStatus_ = false;
      cost_d_ = nullptr;
    }
  }

  /**
   * Computes the feedback control cost on CPU for RMPPI
   */
  float computeFeedbackCost(const Eigen::Ref<const control_array> fb_u, const Eigen::Ref<const control_array> std_dev,
                            const float lambda = 1.0f, const float alpha = 0.0f)
  {
    float cost = 0.0f;
    for (int i = 0; i < CONTROL_DIM; i++)
    {
      cost += params_.control_cost_coeff[i] * SQ(fb_u(i)) / SQ(std_dev(i));
    }

    return 0.5f * lambda * (1.0f - alpha) * cost;
  }

  /**
   * Computes the control cost on CPU. This is the normal control cost calculation
   * in MPPI and Tube-MPPI
   */
  float computeControlCost(const Eigen::Ref<const control_array> u, int timestep, int* crash)
  {
    return 0.0f;
  }
  // =================== METHODS THAT SHOULD HAVE NO DEFAULT ==========================
  /**
   * Computes the state cost on the CPU. Should be implemented in subclasses
   */
  float computeStateCost(const Eigen::Ref<const output_array> y, int timestep, int* crash_status)
  {
    throw std::logic_error("SubClass did not implement computeStateCost");
  }

  /**
   *
   * @param s current state as a float array
   * @return state cost on GPU
   */
  __device__ float computeStateCost(float* y, int timestep, float* theta_c, int* crash_status);

  /**
   * Computes the state cost on the CPU. Should be implemented in subclasses
   */
  float terminalCost(const Eigen::Ref<const output_array> y)
  {
    throw std::logic_error("SubClass did not implement terminalCost");
  }

  /**
   *
   * @param s terminal state as float array
   * @return terminal cost on GPU
   */
  __device__ float terminalCost(float* y, float* theta_c);

  /**
   * Method to allow setup of costs on the CPU. This is needed for
   * initializing the memory of an LSTM for example
   */
  void initializeCosts(const Eigen::Ref<const output_array>& output, const Eigen::Ref<const control_array>& control,
                       float t_0, float dt)
  {
  }

  /**
   * Method to allow setup of costs on the GPU. This is needed for
   * initializing the memory of an LSTM for example
   */
  __device__ void initializeCosts(float* output, float* control, float* theta_c, float t_0, float dt)
  {
  }

  // ================ END OF METHODS WITH NO DEFAULT ===========================

  // =================== METHODS THAT SHOULD NOT BE OVERWRITTEN ================
  /**
   * Computes the feedback control cost on GPU used in RMPPI. There is an
   * assumption that we are provided std_dev and the covriance matrix is
   * diagonal.
   */
  __device__ float computeFeedbackCost(float* fb_u, float* std_dev, float lambda = 1.0f, float alpha = 0.0f)
  {
    float cost = 0.0f;
    for (int i = 0; i < CONTROL_DIM; i++)
    {
      cost += params_.control_cost_coeff[i] * SQ(fb_u[i] / std_dev[i]);
    }
    return 0.5f * lambda * (1.0f - alpha) * cost;
  }

  /**
   * Computes the normal control cost for MPPI and Tube-MPPI
   * 0.5 * lambda * (u*^T \Sigma^{-1} u*^T + 2 * u*^T \Sigma^{-1} (u*^T + noise))
   * On the GPU, u = u* + noise already, so we need the following to create
   * the original cost:
   * 0.5 * lambda * (u - noise)^T \Sigma^{-1} (u + noise)
   */
  __device__ float computeControlCost(float* u, int timestep, float* theta_c, int* crash)
  {
    return 0.0f;
  }
  // =================== END METHODS THAT SHOULD NOT BE OVERWRITTEN ============

  // =================== METHODS THAT CAN BE OVERWRITTEN =======================
  float computeRunningCost(const Eigen::Ref<const output_array> y, const Eigen::Ref<const control_array> u,
                           int timestep, int* crash)
  {
    CLASS_T* derived = static_cast<CLASS_T*>(this);

    return derived->computeStateCost(y, timestep, crash) +
           derived->computeControlCost(u, timestep, crash);
  }

  __device__ float computeRunningCost(float* y, float* u,
                                      int timestep, float* theta_c, int* crash)
  {
    if (threadIdx.y == 0)
    {
      CLASS_T* derived = static_cast<CLASS_T*>(this);
      return derived->computeStateCost(y, timestep, theta_c, crash) +
             derived->computeControlCost(u, timestep, theta_c, crash);
    }
    else
    {
      return 0.0f;
    }
  }

  // =================== END METHODS THAT CAN BE OVERWRITTEN ===================

  inline __host__ __device__ PARAMS_T getParams() const
  {
    return params_;
  }

  CLASS_T* cost_d_ = nullptr;

protected:
  PARAMS_T params_;
};

template <class CLASS_T, class PARAMS_T, class DYN_PARAMS_T>
const int Cost<CLASS_T, PARAMS_T, DYN_PARAMS_T>::CONTROL_DIM;

template <class CLASS_T, class PARAMS_T, class DYN_PARAMS_T>
const int Cost<CLASS_T, PARAMS_T, DYN_PARAMS_T>::OUTPUT_DIM;





//struct AckermannDynamicsParams : public DynamicsParams
//{
//    enum class StateIndex : int
//    {
//        POS_X = 0,          // позиция X [м]
//        POS_Y,              // позиция Y [м]
//        PSI,                // угол рысканья (курс) [рад]
//        V,                  // продольная скорость [м/с]
//        STEER_ANGLE,        // фактический угол поворота колёс [рад]
//        STEER_ANGLE_RATE,   // угловая скорость поворота колёс [рад/с]
//        NUM_STATES
//    };

//    enum class ControlIndex : int
//    {
//        V_CMD = 0,          // желаемая скорость [м/с]
//        W_CMD,              // желаемая угловая скорость корпуса [рад/с]
//        NUM_CONTROLS
//    };

//    enum class OutputIndex : int
//    {
//        POS_X = 0,
//        POS_Y,
//        PSI,
//        V,
//        STEER_ANGLE,
//        STEER_ANGLE_RATE,
//        NUM_OUTPUTS
//    };

//    // Геометрия
//    float wheelbase = 0.33f;          // колёсная база [м]
//    float min_turning_r = 0.6f;       // минимальный радиус поворота [м]

//    // Ограничения по скорости
//    float vx_max = 2.0f;               // макс. скорость вперёд [м/с]
//    float vx_min = -0.5f;              // макс. скорость назад [м/с]
//    float ax_max = 1.5f;                // макс. ускорение [м/с²]
//    float ax_min = -2.0f;               // макс. замедление [м/с²]

//    // Ограничения рулевого управления
//    float max_steer_angle = 0.52f;      // макс. угол поворота колёс [рад] ≈ 30°
//    float max_steer_speed = 4.0f;       // макс. скорость поворота руля [рад/с]
//    float max_steer_accel = 20.0f;      // макс. угловое ускорение руля [рад/с²] (опционально)

//    // Ограничение угловой скорости (устойчивость)
//    float wz_max = 2.5f;                // макс. угловая скорость корпуса [рад/с]

//    // Коэффициенты времени
//    float kv = 10.0f;                    // коэффициент реакции скорости (1/с)
//    float kv_steer = 15.0f;               // коэффициент реакции руля (1/с) – используется при отсутствии измерения скорости
//    float ka_steer = 15.0f;
//};

//class AckermannDynamics : public MPPI_internal::Dynamics<AckermannDynamics, AckermannDynamicsParams>
//{
//public:
//    using state_array   = typename Dynamics::state_array;
//    using control_array = typename Dynamics::control_array;
//    using output_array  = typename Dynamics::output_array;
//    using params_t      = AckermannDynamicsParams;

//    AckermannDynamics(cudaStream_t stream = 0) : Dynamics(stream)
//    {
//        this->SHARED_MEM_REQUEST_GRD_BYTES = sizeof(params_t);
//    }

//    // ------------------------------------------------------------------------
//    // Хост-версия computeStateDeriv (без разделяемой памяти)
//    // ------------------------------------------------------------------------
//    void computeStateDeriv(const state_array& state,
//                           const control_array& control,
//                           state_array& state_der)
//    {
//        const params_t& p = params_;

//        // Извлекаем состояние
//        float psi         = state[static_cast<int>(params_t::StateIndex::PSI)];
//        float v           = state[static_cast<int>(params_t::StateIndex::V)];
//        float steer       = state[static_cast<int>(params_t::StateIndex::STEER_ANGLE)];
//        float steer_rate  = state[static_cast<int>(params_t::StateIndex::STEER_ANGLE_RATE)];

//        // Команды управления
//        float v_cmd       = control[static_cast<int>(params_t::ControlIndex::V_CMD)];
//        float w_cmd       = control[static_cast<int>(params_t::ControlIndex::W_CMD)];

//        // ---- 1. Продольная динамика ----
//        float target_v = fmaxf(p.vx_min, fminf(p.vx_max, v_cmd));
//        float accel = (target_v - v) * p.kv;
//        state_der[static_cast<int>(params_t::StateIndex::V)] =
//            fmaxf(p.ax_min, fminf(p.ax_max, accel));

//        // ---- 2. Обратная кинематика: желаемый угол поворота из w_cmd ----
//        float target_steer = 0.0f;
//        const float eps = 0.05f;  // зона нечувствительности по скорости
//        if (fabsf(v) > eps) {
//            target_steer = atan2f(w_cmd * p.wheelbase, v);
//        }
//        // Ограничиваем физическими пределами
//        target_steer = fmaxf(-p.max_steer_angle, fminf(p.max_steer_angle, target_steer));

//        // ---- 3. Динамика рулевого управления (второго порядка) ----
//        // У нас есть состояние: steer и steer_rate.
//        // Команда влияет на ускорение руля (или можно использовать модель первого порядка).
//        // Здесь используем простой пропорциональный регулятор по ошибке угла,
//        // с ограничениями по скорости и ускорению.
//        float steer_error = target_steer - steer;

//        // Желаемая скорость руля от П-регулятора
//        float desired_rate = steer_error * p.kv_steer;
//        // Ограничиваем желаемую скорость максимумом
//        desired_rate = fmaxf(-p.max_steer_speed, fminf(p.max_steer_speed, desired_rate));

//        // Ускорение руля (скорость изменения steer_rate)
//        // Простая модель: steer_rate стремится к desired_rate с некоторой постоянной времени.
//        // Здесь мы задаём производную steer_rate прямо пропорционально ошибке скорости.
//        // Альтернативно: steer_rate derivative = ограничение ускорения * знак ошибки.
//        float rate_error = desired_rate - steer_rate;
//        float steer_accel = rate_error * p.ka_steer;
//        // Применить ограничение ускорения при необходимости
//        steer_accel = fmaxf(-p.max_steer_accel, fminf(p.max_steer_accel, steer_accel));

//        state_der[static_cast<int>(params_t::StateIndex::STEER_ANGLE)] = steer_rate;
//        state_der[static_cast<int>(params_t::StateIndex::STEER_ANGLE_RATE)] = steer_accel;

//        // ---- 4. Кинематика корпуса ----
//        state_der[static_cast<int>(params_t::StateIndex::POS_X)] = v * cosf(psi);
//        state_der[static_cast<int>(params_t::StateIndex::POS_Y)] = v * sinf(psi);

//        // Фактическая угловая скорость от текущего угла поворота (не от команды!)
//        float wz = (v / p.wheelbase) * tanf(steer);
//        // Применяем ограничение угловой скорости (устойчивость + мин. радиус)
//        float max_wz_radius = (fabsf(v) > eps) ? fabsf(v) / p.min_turning_r : 1e6f;
//        float final_limit_wz = fminf(p.wz_max, max_wz_radius);
//        state_der[static_cast<int>(params_t::StateIndex::PSI)] =
//            fmaxf(-final_limit_wz, fminf(final_limit_wz, wz));
//    }

//    // ------------------------------------------------------------------------
//    // Устройственная версия computeStateDeriv с поддержкой разделяемой памяти
//    // ------------------------------------------------------------------------
//    __device__ void computeStateDeriv(float* state, float* control,
//                                      float* state_der, float* theta_s = nullptr)
//    {
//        params_t* p = &params_;
//        if (this->SHARED_MEM_REQUEST_GRD_BYTES != 0 && theta_s != nullptr)
//            p = (params_t*)theta_s;

//        // Извлекаем состояние
//        float psi   = state[static_cast<int>(params_t::StateIndex::PSI)];
//        float v     = state[static_cast<int>(params_t::StateIndex::V)];
//        float steer = state[static_cast<int>(params_t::StateIndex::STEER_ANGLE)];
//        float steer_rate = state[static_cast<int>(params_t::StateIndex::STEER_ANGLE_RATE)];

//        // Команды управления
//        float v_cmd = control[static_cast<int>(params_t::ControlIndex::V_CMD)];
//        float w_cmd = control[static_cast<int>(params_t::ControlIndex::W_CMD)];

//        // ---- 1. Продольная динамика ----
//        float target_v = fmaxf(p->vx_min, fminf(p->vx_max, v_cmd));
//        float accel = (target_v - v) * p->kv;
//        state_der[static_cast<int>(params_t::StateIndex::V)] =
//            fmaxf(p->ax_min, fminf(p->ax_max, accel));

//        // ---- 2. Обратная кинематика: желаемый угол поворота ----
//        float target_steer = 0.0f;
//        const float eps = 0.05f;
//        if (fabsf(v) > eps) {
//            target_steer = atan2f(w_cmd * p->wheelbase, v);
//        }
//        target_steer = fmaxf(-p->max_steer_angle, fminf(p->max_steer_angle, target_steer));

//        // ---- 3. Динамика рулевого управления (второго порядка) ----
//        float steer_error = target_steer - steer;
//        float desired_rate = steer_error * p->kv_steer;
//        desired_rate = fmaxf(-p->max_steer_speed, fminf(p->max_steer_speed, desired_rate));

//        float rate_error = desired_rate - steer_rate;
//        float steer_accel = rate_error * p->ka_steer;
//        // Опционально применить ограничение ускорения:
//        steer_accel = fmaxf(-p->max_steer_accel, fminf(p->max_steer_accel, steer_accel));

//        state_der[static_cast<int>(params_t::StateIndex::STEER_ANGLE)] = steer_rate;
//        state_der[static_cast<int>(params_t::StateIndex::STEER_ANGLE_RATE)] = steer_accel;

//        // ---- 4. Кинематика корпуса ----
//        state_der[static_cast<int>(params_t::StateIndex::POS_X)] = v * cosf(psi);
//        state_der[static_cast<int>(params_t::StateIndex::POS_Y)] = v * sinf(psi);

//        float wz = (v / p->wheelbase) * tanf(steer);
//        float max_wz_radius = (fabsf(v) > eps) ? fabsf(v) / p->min_turning_r : 1e6f;
//        float final_limit_wz = fminf(p->wz_max, max_wz_radius);
//        state_der[static_cast<int>(params_t::StateIndex::PSI)] =
//            fmaxf(-final_limit_wz, fminf(final_limit_wz, wz));
//    }

//    // ------------------------------------------------------------------------
//    // Устройство: обновление состояния (интегрирование + нормализация)
//    // ------------------------------------------------------------------------
//    __device__ void updateState(float* state, float* next_state, float* state_der, const float dt)
//    {
//        int tdy = threadIdx.y;
//        for (int i = tdy; i < STATE_DIM; i += blockDim.y)
//        {
//            next_state[i] = state[i] + state_der[i] * dt;
//        }
//        __syncthreads();

//        // Нормализуем угол рысканья
//        next_state[static_cast<int>(params_t::StateIndex::PSI)] =
//            angle_utils::normalizeAngle(next_state[static_cast<int>(params_t::StateIndex::PSI)]);

//        // Опционально: ограничиваем угол поворота физическими пределами (уже обеспечено динамикой, но для безопасности)
//        float* steer = &next_state[static_cast<int>(params_t::StateIndex::STEER_ANGLE)];
//        *steer = fmaxf(-params_.max_steer_angle, fminf(params_.max_steer_angle, *steer));

//        // Скорость руля не ограничиваем — это состояние может быть любым в пределах.
//    }

//    // ------------------------------------------------------------------------
//    // Устройство: полный шаг (вычисление производных + обновление)
//    // ------------------------------------------------------------------------
//    __device__ void step(float* state, float* next_state, float* state_der, float* control,
//                         float* output, float* theta_s, const float t, const float dt)
//    {
//        // Вычисляем производные
//        computeStateDeriv(state, control, state_der, theta_s);
//        // Интегрируем и нормализуем
//        updateState(state, next_state, state_der, dt);
//        // Копируем состояние в выход (предполагаем, что выход совпадает с состоянием)
//        int tdy = threadIdx.y;
//        for (int i = tdy; i < OUTPUT_DIM; i += blockDim.y)
//            output[i] = next_state[i];
//    }

//    // ------------------------------------------------------------------------
//    // Устройство: инициализация разделяемой памяти параметрами
//    // ------------------------------------------------------------------------
//    __device__ void initializeDynamics(float* state, float* control, float* output,
//                                       float* theta_s, float t_0, float dt)
//    {
//        if (this->SHARED_MEM_REQUEST_GRD_BYTES != 0 && theta_s != nullptr)
//            *(params_t*)theta_s = params_;
//    }
//};


struct AckermannDynamicsParams : public DynamicsParams
{
    enum class StateIndex : int
    {
        POS_X = 0,          // позиция X [м]
        POS_Y,              // позиция Y [м]
        PSI,                // угол рысканья (курс) [рад]
        V,                  // продольная скорость [м/с]
        STEER_ANGLE,        // фактический угол поворота колёс [рад]
        NUM_STATES
    };

    enum class ControlIndex : int
    {
        V_CMD = 0,          // желаемая скорость [м/с]
        W_CMD,              // желаемая угловая скорость корпуса [рад/с]
        NUM_CONTROLS
    };

    enum class OutputIndex : int
    {
        POS_X = 0,
        POS_Y,
        PSI,
        V,
        STEER_ANGLE,
        NUM_OUTPUTS
    };

    // Геометрия
    float wheelbase = 0.33f;          // колёсная база [м]

    // Ограничения по скорости
    float vx_max = 2.0f;               // макс. скорость вперёд [м/с]
    float vx_min = -0.5f;              // макс. скорость назад [м/с]
    float ax_max = 1.5f;                // макс. ускорение [м/с²]
    float ax_min = -2.0f;               // макс. замедление [м/с²]

    // Ограничения рулевого управления (первый порядок)
    float max_steer_angle = 0.52f;      // макс. угол поворота колёс [рад] ≈ 30°
    float max_steer_speed = 4.0f;       // макс. скорость поворота руля [рад/с]

    // Ограничение угловой скорости (устойчивость)
    float wz_max = 2.5f;                // макс. угловая скорость корпуса [рад/с]

    // Коэффициенты времени
    float kv = 10.0f;                    // коэффициент реакции скорости (1/с)
    float kv_steer = 15.0f;              // коэффициент реакции руля (1/с)

    // Порог скорости для обратной кинематики
    float eps_v = 0.05f;                  // зона нечувствительности по скорости [м/с]

    // Минимальный угол поворота при трогании с места
    float min_steer_at_start = 0.01f;      // минимальный угол для начала движения [рад]
};

class AckermannDynamics : public MPPI_internal::Dynamics<AckermannDynamics, AckermannDynamicsParams>
{
public:
    using state_array   = typename Dynamics::state_array;
    using control_array = typename Dynamics::control_array;
    using output_array  = typename Dynamics::output_array;
    using params_t      = AckermannDynamicsParams;

    AckermannDynamics(cudaStream_t stream = 0) : Dynamics(stream)
    {
        this->SHARED_MEM_REQUEST_GRD_BYTES = sizeof(params_t);
    }

    // ------------------------------------------------------------------------
    // Вспомогательная функция для вычисления желаемого угла поворота
    // ------------------------------------------------------------------------
    float computeTargetSteer(float v, float w_cmd, const params_t& p) const
    {
        float target_steer = 0.0f;

        if (fabsf(v) > p.eps_v) {
            // Нормальный режим: скорость достаточна для кинематики Аккермана
            target_steer = atan2f(w_cmd * p.wheelbase, v);
        } else {
            // Режим старта/остановки: скорость близка к нулю
            // В этом случае угол поворота выбирается исходя из желаемого направления
            // поворота, но с ограничением, чтобы при начале движения не было рывка

            if (fabsf(w_cmd) > 0.01f) {
                // Есть команда на поворот - ставим колёса в соответствующее положение
                // Знак угла соответствует знаку w_cmd
                float steer_magnitude = fminf(p.min_steer_at_start,
                                              fabsf(w_cmd) * p.wheelbase / p.eps_v);
                target_steer = (w_cmd > 0) ? steer_magnitude : -steer_magnitude;
            } else {
                // Нет команды на поворот - колёса прямо
                target_steer = 0.0f;
            }
        }

        // Ограничиваем физическими пределами
        return fmaxf(-p.max_steer_angle, fminf(p.max_steer_angle, target_steer));
    }

    // ------------------------------------------------------------------------
    // Хост-версия computeDynamics
    // ------------------------------------------------------------------------
    void computeDynamics(const Eigen::Ref<const state_array>& state,
                         const Eigen::Ref<const control_array>& control,
                         Eigen::Ref<state_array> state_der)
    {
        const params_t& p = params_;

        // Извлекаем состояние
        float psi = state[static_cast<int>(params_t::StateIndex::PSI)];
        float v   = state[static_cast<int>(params_t::StateIndex::V)];
        float steer = state[static_cast<int>(params_t::StateIndex::STEER_ANGLE)];

        // Команды управления
        float v_cmd = control[static_cast<int>(params_t::ControlIndex::V_CMD)];
        float w_cmd = control[static_cast<int>(params_t::ControlIndex::W_CMD)];

        // ---- 1. Продольная динамика (первый порядок) ----
        float target_v = fmaxf(p.vx_min, fminf(p.vx_max, v_cmd));
        float accel = (target_v - v) * p.kv;
        state_der[static_cast<int>(params_t::StateIndex::V)] =
            fmaxf(p.ax_min, fminf(p.ax_max, accel));

        // ---- 2. Рулевое управление (первый порядок) ----
        float target_steer = computeTargetSteer(v, w_cmd, p);

        float steer_rate = (target_steer - steer) * p.kv_steer;
        steer_rate = fmaxf(-p.max_steer_speed, fminf(p.max_steer_speed, steer_rate));
        state_der[static_cast<int>(params_t::StateIndex::STEER_ANGLE)] = steer_rate;

        // ---- 3. Кинематика корпуса ----
        state_der[static_cast<int>(params_t::StateIndex::POS_X)] = v * cosf(psi);
        state_der[static_cast<int>(params_t::StateIndex::POS_Y)] = v * sinf(psi);

        // Угловая скорость от текущего угла поворота
        // При v ≈ 0 угловая скорость тоже близка к 0
        float wz = 0.0f;
        if (fabsf(v) > p.eps_v) {
            wz = (v / p.wheelbase) * tanf(steer);
        }

        // Опциональное ограничение угловой скорости
        if (fabsf(wz) > p.wz_max) {
            wz = (wz > 0) ? p.wz_max : -p.wz_max;
        }
        state_der[static_cast<int>(params_t::StateIndex::PSI)] = wz;
    }

    // ------------------------------------------------------------------------
    // Устройственная версия computeDynamics
    // ------------------------------------------------------------------------
    __device__ void computeDynamics(float* state, float* control,
                                    float* state_der, float* theta_s = nullptr)
    {
        params_t* p = &params_;
        if (this->SHARED_MEM_REQUEST_GRD_BYTES != 0 && theta_s != nullptr)
            p = (params_t*)theta_s;

        // Извлекаем состояние
        float psi = state[static_cast<int>(params_t::StateIndex::PSI)];
        float v   = state[static_cast<int>(params_t::StateIndex::V)];
        float steer = state[static_cast<int>(params_t::StateIndex::STEER_ANGLE)];

        // Команды управления
        float v_cmd = control[static_cast<int>(params_t::ControlIndex::V_CMD)];
        float w_cmd = control[static_cast<int>(params_t::ControlIndex::W_CMD)];

        // ---- 1. Продольная динамика ----
        float target_v = fmaxf(p->vx_min, fminf(p->vx_max, v_cmd));
        float accel = (target_v - v) * p->kv;
        state_der[static_cast<int>(params_t::StateIndex::V)] =
            fmaxf(p->ax_min, fminf(p->ax_max, accel));

        // ---- 2. Рулевое управление (первый порядок) ----
        float target_steer;
        if (fabsf(v) > p->eps_v) {
            target_steer = atan2f(w_cmd * p->wheelbase, v);
        } else {
            if (fabsf(w_cmd) > 0.01f) {
                float steer_magnitude = fminf(p->min_steer_at_start,
                                              fabsf(w_cmd) * p->wheelbase / p->eps_v);
                target_steer = (w_cmd > 0) ? steer_magnitude : -steer_magnitude;
            } else {
                target_steer = 0.0f;
            }
        }
        target_steer = fmaxf(-p->max_steer_angle, fminf(p->max_steer_angle, target_steer));

        float steer_rate = (target_steer - steer) * p->kv_steer;
        steer_rate = fmaxf(-p->max_steer_speed, fminf(p->max_steer_speed, steer_rate));
        state_der[static_cast<int>(params_t::StateIndex::STEER_ANGLE)] = steer_rate;

        // ---- 3. Кинематика корпуса ----
        state_der[static_cast<int>(params_t::StateIndex::POS_X)] = v * cosf(psi);
        state_der[static_cast<int>(params_t::StateIndex::POS_Y)] = v * sinf(psi);

        float wz = 0.0f;
        if (fabsf(v) > p->eps_v) {
            wz = (v / p->wheelbase) * tanf(steer);
        }
        if (fabsf(wz) > p->wz_max) {
            wz = (wz > 0) ? p->wz_max : -p->wz_max;
        }
        state_der[static_cast<int>(params_t::StateIndex::PSI)] = wz;
    }

    // ------------------------------------------------------------------------
    // Хост-версия updateState (интегрирование + нормализация)
    // ------------------------------------------------------------------------
    void updateState(const Eigen::Ref<const state_array>& state,
                     Eigen::Ref<state_array> next_state,
                     const Eigen::Ref<const state_array>& state_der,
                     const float dt)
    {
        next_state = state + state_der * dt;

        // Нормализуем угол рысканья
        next_state[static_cast<int>(params_t::StateIndex::PSI)] =
            angle_utils::normalizeAngle(next_state[static_cast<int>(params_t::StateIndex::PSI)]);

        // Дополнительная страховка: ограничиваем угол поворота
        float* steer = &next_state[static_cast<int>(params_t::StateIndex::STEER_ANGLE)];
        *steer = fmaxf(-params_.max_steer_angle, fminf(params_.max_steer_angle, *steer));
    }

    // ------------------------------------------------------------------------
    // Устройственная версия updateState
    // ------------------------------------------------------------------------
    __device__ void updateState(float* state, float* next_state,
                                float* state_der, const float dt)
    {
        int tdy = threadIdx.y;
        // Параллельное копирование всех компонент состояния
        for (int i = tdy; i < STATE_DIM; i += blockDim.y)
        {
            next_state[i] = state[i] + state_der[i] * dt;
        }
        __syncthreads();

        // Нормализацию и ограничения выполняем одной нитью
        if (tdy == 0)
        {
            // Нормализуем угол рысканья
            next_state[static_cast<int>(params_t::StateIndex::PSI)] =
                angle_utils::normalizeAngle(next_state[static_cast<int>(params_t::StateIndex::PSI)]);

            // Ограничиваем угол поворота
            float* steer = &next_state[static_cast<int>(params_t::StateIndex::STEER_ANGLE)];
            *steer = fmaxf(-params_.max_steer_angle, fminf(params_.max_steer_angle, *steer));
        }
    }

    // ------------------------------------------------------------------------
    // Хост-версия step (полный шаг симуляции)
    // ------------------------------------------------------------------------
    void step(Eigen::Ref<state_array> state, Eigen::Ref<state_array> next_state,
              Eigen::Ref<state_array> state_der, const Eigen::Ref<const control_array>& control,
              Eigen::Ref<output_array> output, const float t, const float dt)
    {
        computeDynamics(state, control, state_der);
        updateState(state, next_state, state_der, dt);
        // Копируем состояние в выход (OUTPUT_DIM = STATE_DIM)
        output = next_state;
    }

    // ------------------------------------------------------------------------
    // Устройственная версия step
    // ------------------------------------------------------------------------
    __device__ void step(float* state, float* next_state, float* state_der, float* control,
                         float* output, float* theta_s, const float t, const float dt)
    {
        computeDynamics(state, control, state_der, theta_s);
        updateState(state, next_state, state_der, dt);

        // Копируем состояние в выход
        int tdy = threadIdx.y;
        for (int i = tdy; i < OUTPUT_DIM; i += blockDim.y)
            output[i] = next_state[i];
    }

    // ------------------------------------------------------------------------
    // Инициализация разделяемой памяти параметрами
    // ------------------------------------------------------------------------
    __device__ void initializeDynamics(float* state, float* control, float* output,
                                       float* theta_s, float t_0, float dt)
    {
        if (this->SHARED_MEM_REQUEST_GRD_BYTES != 0 && theta_s != nullptr)
            *(params_t*)theta_s = params_;
    }

    // ------------------------------------------------------------------------
    // Преобразование карты (ROS-совместимость)
    // ------------------------------------------------------------------------
    state_array stateFromMap(const std::map<std::string, float>& map)
    {
        state_array s = state_array::Zero();
        s[static_cast<int>(params_t::StateIndex::POS_X)] = map.at("POS_X");
        s[static_cast<int>(params_t::StateIndex::POS_Y)] = map.at("POS_Y");
        s[static_cast<int>(params_t::StateIndex::PSI)] = map.at("YAW");
        s[static_cast<int>(params_t::StateIndex::V)] = map.at("VEL_X");
        s[static_cast<int>(params_t::StateIndex::STEER_ANGLE)] = map.at("STEER_ANGLE");
        return s;
    }
};

extern "C"
__host__
void Run_MPPI_Core(uint8_t * input, uint8_t *output)
{

}
