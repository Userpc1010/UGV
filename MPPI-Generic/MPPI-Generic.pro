CONFIG += qt

TEMPLATE = lib
TARGET = MPPIGeneric
DEFINES += MPPIGENERIC_LIBRARY

CONFIG += c++17

INCLUDEPATH += /usr/include/eigen3

INCLUDEPATH += $$PWD \
               $$PWD/..


SOURCES += \
    mppi-generic.cpp

HEADERS += \
    mppi-generic.h \
    mppi/controllers/MPPI/mppi_controller.h \
    mppi/controllers/controller.h \
    mppi/core/base_plant.h \
    mppi/core/buffer.h \
    mppi/core/buffered_plant.h \
    mppi/core/mppi_common.h \
    mppi/core/rmppi_kernels.h \
    mppi/cost_functions/NAV2_Compatible_CostFunction.h \
    mppi/cost_functions/cost.h \
    mppi/dynamics/AckermannDynamics.h \
    mppi/dynamics/dynamics.h \
    mppi/feedback_controllers/DDP/ddp.h \
    mppi/feedback_controllers/feedback.h \
    mppi/sampling_distributions/gaussian/gaussian.h \
    mppi/sampling_distributions/sampling_distribution.h \
    mppi/utils/activation_functions.h \
    mppi/utils/angle_utils.h \
    mppi/utils/cuda_math_utils.h \
    mppi/utils/eigen_type_conversions.h \
    mppi/utils/file_utils.h \
    mppi/utils/gpu_err_chk.h \
    mppi/utils/logger.h \
    mppi/utils/managed.h \
    mppi/utils/math_utils.h \
    mppi/utils/matrix_mult_utils.h \
    mppi/utils/nn_helpers/fnn_helper.h \
    mppi/utils/nn_helpers/lstm_helper.h \
    mppi/utils/nn_helpers/lstm_lstm_helper.h \
    mppi/utils/nn_helpers/meta_math.h \
    mppi/utils/numerical_integration.h \
    mppi/utils/parallel_utils.h \
    mppi/utils/risk_utils.h \
    mppi/utils/test_helper.h \
    mppi/utils/texture_helpers/texture_helper.h \
    mppi/utils/texture_helpers/three_d_texture_helper.h \
    mppi/utils/texture_helpers/two_d_texture_helper.h \
    mppi/utils/type_printing.h


#set package support if disabled
QT_CONFIG -= no-pkg-config

CONFIG += link_pkgconfig
PKGCONFIG += eigen3


# CUDA
# nvcc flags (ptxas option verbose is always useful)
NVCCFLAGS = --compiler-options -fno-strict-aliasing -use_fast_math --ptxas-options=-v -Xcompiler -fPIC -std=c++17
# Path to cuda toolkit install
CUDA_DIR = /usr/local/cuda-12.2
# GPU architecture (ADJUST FOR YOUR GPU)
CUDA_GENCODE  = arch=compute_86,code=sm_86
# manually add CUDA sources (ADJUST MANUALLY)
CUDA_SOURCES += mppi-generic-controller.cu
# Path to header and libs files
INCLUDEPATH  += $$CUDA_DIR/include
# libs used in your code
LIBS += -L$$CUDA_DIR/lib64 -lcuda -lcudart -lcublas -lcufft -lcudadevrt -lcurand

cuda.commands = $$CUDA_DIR/bin/nvcc -c -gencode $$CUDA_GENCODE $$NVCCFLAGS \
                $$join(INCLUDEPATH, " -I", "-I") \
                -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
cuda.dependency_type = TYPE_C
cuda.depend_command  = $$CUDA_DIR/bin/nvcc -M ${QMAKE_FILE_NAME} | sed \"s/^.*: //\" #For Qt 5.12.2
cuda.input           = CUDA_SOURCES
cuda.output          = ${OBJECTS_DIR}${QMAKE_FILE_BASE}_cuda.o
# Tell Qt that we want add more stuff to the Makefile
QMAKE_EXTRA_COMPILERS += cuda

DEFINES += CMAKE_USE_CUDA_BARRIERS \
           CMAKE_USE_CUDA_BARRIERS_DYN \
           CMAKE_USE_CUDA_BARRIERS_ROLLOUT

