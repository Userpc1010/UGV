//
// Created by jason on 1/5/22.
//

#ifndef MPPIGENERIC_TEXTURE_HELPER_CUH
#define MPPIGENERIC_TEXTURE_HELPER_CUH

#include <mppi/utils/managed.h>
#include <mppi/utils/cuda_math_utils.h>
#include <mppi/utils/math_utils.h>

#include <array>

template <class DATA_T>
struct TextureParams
{
  cudaExtent extent;

  cudaArray* array_d = nullptr;
  cudaTextureObject_t tex_d = 0;
  cudaChannelFormatDesc channelDesc;
  cudaResourceDesc resDesc;
  cudaTextureDesc texDesc;

  float3 origin;
  float3 rotations[3];
  float3 resolution;

  bool use = false;        // indicates that the texture is to be used or not, separate from allocation
  bool allocated = false;  // indicates that the texture has been allocated on the GPU
  bool update_data = false;
  bool update_mem = false;  // indicates the GPU structure should be updated at the next convenient time
  bool update_params = false;

  TextureParams()
  {
    extent = make_cudaExtent(1, 1, 1);
    resDesc.resType = cudaResourceTypeArray;
    channelDesc = cudaCreateChannelDesc<DATA_T>();

    // clamp
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.addressMode[2] = cudaAddressModeClamp;
    texDesc.borderColor[0] = 0.0f;
    texDesc.borderColor[1] = 0.0f;
    texDesc.borderColor[2] = 0.0f;
    texDesc.borderColor[3] = 0.0f;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;

    origin = make_float3(0.0, 0.0, 0.0);
    rotations[0] = make_float3(1, 0, 0);
    rotations[1] = make_float3(0, 1, 0);
    rotations[2] = make_float3(0, 0, 1);
    resolution = make_float3(1, 1, 1);
  }
};

template <class TEX_T, class DATA_T>
class TextureHelper : public Managed
{
protected:
  TextureHelper(int number, cudaStream_t stream = 0);

public:
  virtual ~TextureHelper();

  void GPUSetup();

  static void freeCudaMem(TextureParams<DATA_T>& texture);
  virtual void freeCudaMem();

  /**
   * helper method to deallocate the index before allocating new ones
   */
  virtual void allocateCudaTexture(int index);
  /**
   * helper method to create a cuda texture
   * @param index
   */
  virtual void createCudaTexture(int index, bool sync = true);

  /**
   * Copies texture information to the GPU version of the object
   */
  virtual void copyToDevice(bool synchronize = false);

  /**
   *
   */
  virtual void addNewTexture(const cudaExtent& extent);

  __host__ __device__ void bodyOffsetToWorldPose(const float3& offset, const float3& body_pose, const float3& rotation,
                                                 float3& output);
  __host__ __device__ void worldPoseToMapPose(const int index, const float3& input, float3& output);
  __host__ __device__ void mapPoseToTexCoord(const int index, const float3& input, float3& output);
  __host__ __device__ void worldPoseToTexCoord(const int index, const float3& input, float3& output);
  __host__ __device__ void bodyOffsetWorldToTexCoord(const int index, const float3& offset, const float3& body_pose,
                                                     const float3& rotation, float3& output);
  __host__ __device__ DATA_T queryTextureAtWorldOffsetPose(const int index, const float3& input, const float3& offset,
                                                           const float3& rotation);
  __host__ __device__ DATA_T queryTextureAtWorldPose(const int index, const float3& input);
  __host__ __device__ DATA_T queryTextureAtMapPose(const int index, const float3& input);

  virtual void updateOrigin(int index, float3 new_origin);
  virtual void updateRotation(int index, std::array<float3, 3>& new_rotation);
  virtual void updateResolution(int index, float resolution);
  virtual void updateResolution(int index, float3 resolution);
  virtual bool setExtent(int index, cudaExtent& extent);
  virtual void copyDataToGPU(int index, bool sync = false) = 0;
  virtual void copyParamsToGPU(int index, bool sync = false);
  virtual void enableTexture(int index)
  {
    this->textures_buffer_[index].update_params = true;
    this->textures_buffer_[index].use = true;
  }
  virtual void disableTexture(int index)
  {
    this->textures_buffer_[index].update_params = true;
    this->textures_buffer_[index].use = false;
  }
  __device__ __host__ bool checkTextureUse(int index) const
  {
    return this->textures_d_[index].use;
  }

  void updateAddressMode(int index, cudaTextureAddressMode mode);
  void updateAddressMode(int index, int layer, cudaTextureAddressMode mode);

  void updateBorder(int index, float value);
  void updateBorder(int index, int layer, float value);

  std::vector<TextureParams<DATA_T>> getTextures()
  {
    return textures_;
  }
  std::vector<TextureParams<DATA_T>> getBufferTextures()
  {
    return textures_buffer_;
  }

  std::vector<std::vector<DATA_T>> getCpuValues()
  {
    return cpu_values_;
  }

  std::vector<std::vector<DATA_T>> getCpuBufferValues()
  {
    return cpu_buffer_values_;
  }

  void updateDataAtIndex(int index)
  {
    this->textures_buffer_[index].update_data = true;
  }

  __device__ __host__ int size()
  {
    return size_;
  }

  TEX_T* ptr_d_ = nullptr;

  __host__ __device__ float3 getOrigin(int index) const
  {
    return this->textures_d_[index].origin;
  }

  __host__ __device__ float3 getResolution(int index) const
  {
    return this->textures_d_[index].resolution;
  }

  __host__ __device__ cudaExtent getExtent(int index) const
  {
    return this->textures_d_[index].extent;
  }

protected:
  // std::mutex buffer_lck_;

  // stores the values we actually use
  std::vector<TextureParams<DATA_T>> textures_;
  // buffer that can be updated async, does not have correct cuda memory locations
  std::vector<TextureParams<DATA_T>> textures_buffer_;

  // memory allocation can vary depending on 1D, 2D, or 3D implementation
  std::vector<std::vector<DATA_T>> cpu_values_;
  std::vector<std::vector<DATA_T>> cpu_buffer_values_;

  // helper, on CPU points to vector data (textures_.data()), on GPU points to device copy (params_d_ variable)
  TextureParams<DATA_T>* textures_d_ = nullptr;

  // device pointer to the parameters malloced memory
  TextureParams<DATA_T>* params_d_ = nullptr;
  int size_ = 0;
};

template <class TEX_T, class DATA_T>
TextureHelper<TEX_T, DATA_T>::TextureHelper(int number, cudaStream_t stream) : Managed(stream), size_(number)
{
  textures_.resize(number);
  textures_buffer_.resize(number);
  cpu_values_.resize(number);
  cpu_buffer_values_.resize(number);
  textures_d_ = textures_.data();
}

template <class TEX_T, class DATA_T>
TextureHelper<TEX_T, DATA_T>::~TextureHelper()
{
  freeCudaMem();
}

template <class TEX_T, class DATA_T>
void TextureHelper<TEX_T, DATA_T>::GPUSetup()
{
  if (!GPUMemStatus_)
  {
    TEX_T* derived = static_cast<TEX_T*>(this);
    ptr_d_ = Managed::GPUSetup<TEX_T>(derived);
    // allocates memory to access params on the GPU by pointer
    HANDLE_ERROR(cudaMalloc(&params_d_, sizeof(TextureParams<DATA_T>) * textures_.size()));
    HANDLE_ERROR(cudaMemcpyAsync(&(ptr_d_->textures_d_), &(params_d_), sizeof(TextureParams<DATA_T>*),
                                 cudaMemcpyHostToDevice, this->stream_));
    copyToDevice(true);
  }
  else
  {
    std::cout << "GPU Memory already set" << std::endl;
  }
}

template <class TEX_T, class DATA_T>
void TextureHelper<TEX_T, DATA_T>::freeCudaMem()
{
  if (this->GPUMemStatus_)
  {
    for (int index = 0; index < textures_.size(); index++)
    {
      freeCudaMem(textures_[index]);
    }
    if (params_d_ != nullptr)
    {
      HANDLE_ERROR(cudaFree(params_d_));
    }
    if (ptr_d_ != nullptr)
    {
      HANDLE_ERROR(cudaFree(ptr_d_));
    }
  }
  this->GPUMemStatus_ = false;
  this->params_d_ = nullptr;
  this->ptr_d_ = nullptr;
  CudaCheckError();
}

template <class TEX_T, class DATA_T>
void TextureHelper<TEX_T, DATA_T>::freeCudaMem(TextureParams<DATA_T>& texture)
{
  if (texture.allocated)
  {
    HANDLE_ERROR(cudaFreeArray(texture.array_d));
    HANDLE_ERROR(cudaDestroyTextureObject(texture.tex_d));
    texture.allocated = false;
    texture.array_d = nullptr;
    texture.tex_d = 0;
  }
}

template <class TEX_T, class DATA_T>
void TextureHelper<TEX_T, DATA_T>::allocateCudaTexture(int index)
{
  // if already allocated, deallocate
  if (this->GPUMemStatus_ && textures_[index].allocated)
  {
    freeCudaMem(textures_[index]);
  }
}

template <class TEX_T, class DATA_T>
__host__ __device__ void TextureHelper<TEX_T, DATA_T>::bodyOffsetToWorldPose(const float3& offset,
                                                                             const float3& body_pose,
                                                                             const float3& rotation, float3& output)
{
  mppi::math::bodyOffsetToWorldPoseEuler(offset, body_pose, rotation, output);
}

template <class TEX_T, class DATA_T>
__host__ __device__ void TextureHelper<TEX_T, DATA_T>::worldPoseToMapPose(const int index, const float3& input,
                                                                          float3& output)
{
  float3 diff = make_float3(input.x - textures_d_[index].origin.x, input.y - textures_d_[index].origin.y,
                            input.z - textures_d_[index].origin.z);
  float3* rotation_mat_ptr = textures_d_[index].rotations;
  output.x = (rotation_mat_ptr[0].x * diff.x + rotation_mat_ptr[0].y * diff.y + rotation_mat_ptr[0].z * diff.z);
  output.y = (rotation_mat_ptr[1].x * diff.x + rotation_mat_ptr[1].y * diff.y + rotation_mat_ptr[1].z * diff.z);
  output.z = (rotation_mat_ptr[2].x * diff.x + rotation_mat_ptr[2].y * diff.y + rotation_mat_ptr[2].z * diff.z);
}

template <class TEX_T, class DATA_T>
__host__ __device__ void TextureHelper<TEX_T, DATA_T>::mapPoseToTexCoord(const int index, const float3& input,
                                                                         float3& output)
{
  // printf("res %f %f %f extent %f %f %f\n", textures_d_[index].resolution.x, textures_d_[index].resolution.y,
  // textures_d_[index].resolution.z, textures_d_[index].extent.width, textures_d_[index].extent.depth);
  // from map frame to pixels [m] -> [px]
  output.x = input.x / textures_d_[index].resolution.x;
  output.y = input.y / textures_d_[index].resolution.y;
  output.z = input.z / textures_d_[index].resolution.z;

  // normalize pixel values
  output.x /= textures_d_[index].extent.width;
  output.y /= textures_d_[index].extent.height;
  if (textures_d_[index].extent.depth != 0)
  {
    output.z /= textures_d_[index].extent.depth;
  }
}

template <class TEX_T, class DATA_T>
__host__ __device__ void TextureHelper<TEX_T, DATA_T>::worldPoseToTexCoord(const int index, const float3& input,
                                                                           float3& output)
{
  float3 map;
  worldPoseToMapPose(index, input, map);
  mapPoseToTexCoord(index, map, output);
  // printf("world to map (%f, %f, %f) -> (%f, %f, %f) -> (%f, %f, %f)\n", input.x, input.y, input.z, map.x, map.y,
  // map.z, output.x, output.y, output.z);
}

template <class TEX_T, class DATA_T>
__host__ __device__ void TextureHelper<TEX_T, DATA_T>::bodyOffsetWorldToTexCoord(const int index, const float3& offset,
                                                                                 const float3& body_pose,
                                                                                 const float3& rotation, float3& output)
{
  float3 offset_result;
  bodyOffsetToWorldPose(offset, body_pose, rotation, offset_result);
  worldPoseToTexCoord(index, offset_result, output);
}

template <class TEX_T, class DATA_T>
void TextureHelper<TEX_T, DATA_T>::copyToDevice(bool synchronize)
{
  // TODO lock the buffer
  // copies the buffer to the CPU side version
  for (int i = 0; i < textures_buffer_.size(); i++)
  {
    if (textures_buffer_[i].update_params)
    {
      // copy over params from buffer to object
      textures_[i].origin = textures_buffer_[i].origin;
      textures_[i].rotations[0] = textures_buffer_[i].rotations[0];
      textures_[i].rotations[1] = textures_buffer_[i].rotations[1];
      textures_[i].rotations[2] = textures_buffer_[i].rotations[2];
      textures_[i].resolution = textures_buffer_[i].resolution;
      textures_[i].update_params = true;
      textures_buffer_[i].update_params = false;
    }
    // copy the relevant things over from buffer
    if (textures_buffer_[i].update_data)
    {
      // moves data from cpu buffer to cpu side
      cpu_values_[i] = std::move(cpu_buffer_values_[i]);
      // cpu_buffer_values are resized in the updateTexture method
      textures_[i].update_data = true;
      textures_buffer_[i].update_data = false;
      textures_[i].use = textures_buffer_[i].use;
    }
    if (textures_buffer_[i].update_mem)
    {
      textures_[i].extent = textures_buffer_[i].extent;
      textures_[i].texDesc = textures_buffer_[i].texDesc;
      textures_[i].update_mem = true;
      textures_buffer_[i].update_mem = false;
    }
  }
  // TODO unlock buffer

  if (!this->GPUMemStatus_)
  {
    return;
  }

  // goes through and checks what needs to be copied and does it
  TEX_T* derived = static_cast<TEX_T*>(this);
  for (int i = 0; i < textures_.size(); i++)
  {
    TextureParams<DATA_T>* param = &textures_[i];

    // do the allocation and texture creation
    if (param->update_mem)
    {
      derived->allocateCudaTexture(i);
      derived->createCudaTexture(i, false);
    }
    // if allocated
    if (param->allocated)
    {
      // if we have new parameter values copy it over
      if (param->update_params)
      {
        derived->copyParamsToGPU(i, false);
      }

      // if we have updated data copy it over
      if (param->update_data)
      {
        // copies data to the GPU
        derived->copyDataToGPU(i, false);
      }
    }
  }
  if (synchronize)
  {
    cudaStreamSynchronize(this->stream_);
  }
}

template <class TEX_T, class DATA_T>
void TextureHelper<TEX_T, DATA_T>::createCudaTexture(int index, bool sync)
{
  TextureParams<DATA_T>* cpu_param = &textures_[index];
  cpu_param->resDesc.res.array.array = cpu_param->array_d;

  HANDLE_ERROR(cudaCreateTextureObject(&(cpu_param->tex_d), &cpu_param->resDesc, &cpu_param->texDesc, NULL));

  cpu_param->allocated = true;
  cpu_param->update_mem = false;

  copyParamsToGPU(index, sync);
}

template <class TEX_T, class DATA_T>
void TextureHelper<TEX_T, DATA_T>::addNewTexture(const cudaExtent& extent)
{
  // update the buffer not the actual textures
  textures_buffer_.resize(textures_buffer_.size() + 1);
  textures_.resize(textures_.size() + 1);
  textures_buffer_.back().extent = extent;
  textures_.back().extent = extent;
  size_ = textures_.size();

  if (this->GPUMemStatus_)
  {
    TEX_T* derived = static_cast<TEX_T*>(this);
    int index = textures_.size() - 1;

    // TODO resize the device side array that stores textures

    derived->allocateCudaTexture(index);
    derived->createCudaTexture(index);
    textures_.back().allocated = true;
  }
}

template <class TEX_T, class DATA_T>
__host__ __device__ DATA_T TextureHelper<TEX_T, DATA_T>::queryTextureAtWorldOffsetPose(const int index,
                                                                                       const float3& input,
                                                                                       const float3& offset,
                                                                                       const float3& rotation)
{
  float3 tex_coords;
  bodyOffsetWorldToTexCoord(index, offset, input, rotation, tex_coords);
  TEX_T* derived = static_cast<TEX_T*>(this);
  return derived->queryTexture(index, tex_coords);
}

template <class TEX_T, class DATA_T>
__host__ __device__ DATA_T TextureHelper<TEX_T, DATA_T>::queryTextureAtWorldPose(const int index, const float3& input)
{
  float3 tex_coords;
  worldPoseToTexCoord(index, input, tex_coords);
  TEX_T* derived = static_cast<TEX_T*>(this);
  return derived->queryTexture(index, tex_coords);
}

template <class TEX_T, class DATA_T>
__host__ __device__ DATA_T TextureHelper<TEX_T, DATA_T>::queryTextureAtMapPose(const int index, const float3& input)
{
  float3 tex_coords;
  mapPoseToTexCoord(index, input, tex_coords);
  TEX_T* derived = static_cast<TEX_T*>(this);
  return derived->queryTexture(index, tex_coords);
}

template <class TEX_T, class DATA_T>
void TextureHelper<TEX_T, DATA_T>::updateOrigin(int index, float3 new_origin)
{
  this->textures_buffer_[index].origin = new_origin;
  this->textures_buffer_[index].update_params = true;
}

template <class TEX_T, class DATA_T>
void TextureHelper<TEX_T, DATA_T>::updateRotation(int index, std::array<float3, 3>& new_rotation)
{
  this->textures_buffer_[index].rotations[0] = new_rotation[0];
  this->textures_buffer_[index].rotations[1] = new_rotation[1];
  this->textures_buffer_[index].rotations[2] = new_rotation[2];
  this->textures_buffer_[index].update_params = true;
}

template <class TEX_T, class DATA_T>
void TextureHelper<TEX_T, DATA_T>::updateResolution(int index, float resolution)
{
  this->textures_buffer_[index].resolution.x = resolution;
  this->textures_buffer_[index].resolution.y = resolution;
  this->textures_buffer_[index].resolution.z = resolution;
  this->textures_buffer_[index].update_params = true;
}

template <class TEX_T, class DATA_T>
void TextureHelper<TEX_T, DATA_T>::updateResolution(int index, float3 resolution)
{
  this->textures_buffer_[index].resolution.x = resolution.x;
  this->textures_buffer_[index].resolution.y = resolution.y;
  this->textures_buffer_[index].resolution.z = resolution.z;
  this->textures_buffer_[index].update_params = true;
}

template <class TEX_T, class DATA_T>
bool TextureHelper<TEX_T, DATA_T>::setExtent(int index, cudaExtent& extent)
{
  // checks if the extent has changed and reallocates if yes
  TextureParams<DATA_T>* param = &textures_buffer_[index];
  if (param->extent.width != extent.width || param->extent.height != extent.height ||
      param->extent.depth != extent.depth)
  {
    // flag to update mem next time we should
    param->update_mem = true;
    this->textures_buffer_[index].extent = extent;
    return true;
  }
  return false;
}

template <class TEX_T, class DATA_T>
void TextureHelper<TEX_T, DATA_T>::copyParamsToGPU(int index, bool sync)
{
  TextureParams<DATA_T>* cpu_param = &textures_[index];

  // Copy entire param structure over from CPU to GPU
  HANDLE_ERROR(cudaMemcpyAsync(&(params_d_[index]), cpu_param, sizeof(TextureParams<DATA_T>), cudaMemcpyHostToDevice,
                               this->stream_));
  cpu_param->update_params = false;
  if (sync)
  {
    cudaStreamSynchronize(this->stream_);
  }
}

template <class TEX_T, class DATA_T>
void TextureHelper<TEX_T, DATA_T>::updateAddressMode(int index, cudaTextureAddressMode mode)
{
  this->textures_buffer_[index].texDesc.addressMode[0] = mode;
  this->textures_buffer_[index].texDesc.addressMode[1] = mode;
  this->textures_buffer_[index].texDesc.addressMode[2] = mode;
  this->textures_buffer_[index].update_mem = true;
}

template <class TEX_T, class DATA_T>
void TextureHelper<TEX_T, DATA_T>::updateAddressMode(int index, int layer, cudaTextureAddressMode mode)
{
  this->textures_buffer_[index].texDesc.addressMode[layer] = mode;
  this->textures_buffer_[index].update_mem = true;
}

template <class TEX_T, class DATA_T>
void TextureHelper<TEX_T, DATA_T>::updateBorder(int index, float value)
{
  this->textures_buffer_[index].texDesc.borderColor[0] = value;
  this->textures_buffer_[index].texDesc.borderColor[1] = value;
  this->textures_buffer_[index].texDesc.borderColor[2] = value;
  this->textures_buffer_[index].texDesc.borderColor[3] = value;
  this->textures_buffer_[index].update_mem = true;
}

template <class TEX_T, class DATA_T>
void TextureHelper<TEX_T, DATA_T>::updateBorder(int index, int layer, float value)
{
  this->textures_buffer_[index].texDesc.borderColor[layer] = value;
  this->textures_buffer_[index].update_mem = true;
}

#endif  // MPPIGENERIC_TEXTURE_HELPER_CUH
