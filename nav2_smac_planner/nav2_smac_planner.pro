QT -= gui

TEMPLATE = lib
DEFINES += NAV2_SMAC_PLANNER_LIBRARY

CONFIG += c++17

QMAKE_CXXFLAGS += -flto
QMAKE_CXXFLAGS += -O3

# The following define makes your compiler emit warnings if you use
# any Qt feature that has been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

#set package support if disabled
QT_CONFIG -= no-pkg-config

CONFIG += link_pkgconfig
PKGCONFIG += eigen3

HEADERS += \
    a_star.hpp \
    analytic_expansion.hpp \
    angles/angles.h \
    collision_checker.hpp \
    constants.hpp \
    costmap_downsampler.hpp \
    datatypes_smac_planner.h \
    geometry_utils.hpp \
    goal_manager.hpp \
    nav2_costmap_2d/cost_values.hpp \
    nav2_costmap_2d/costmap_2d.hpp \
    nav2_costmap_2d/costmap_2d_ros.hpp \
    nav2_costmap_2d/costmap_math.hpp \
    nav2_costmap_2d/footprint.hpp \
    nav2_costmap_2d/footprint_collision_checker.hpp \
    nav2_costmap_2d/raytrace_line_2d.hpp \
    nav2_costmap_2d/occ_grid_values.hpp \
    nlohmann/adl_serializer.hpp \
    nlohmann/byte_container_with_subtype.hpp \
    nlohmann/detail/abi_macros.hpp \
    nlohmann/detail/conversions/from_json.hpp \
    nlohmann/detail/conversions/to_chars.hpp \
    nlohmann/detail/conversions/to_json.hpp \
    nlohmann/detail/exceptions.hpp \
    nlohmann/detail/hash.hpp \
    nlohmann/detail/input/binary_reader.hpp \
    nlohmann/detail/input/input_adapters.hpp \
    nlohmann/detail/input/json_sax.hpp \
    nlohmann/detail/input/lexer.hpp \
    nlohmann/detail/input/parser.hpp \
    nlohmann/detail/input/position_t.hpp \
    nlohmann/detail/iterators/internal_iterator.hpp \
    nlohmann/detail/iterators/iter_impl.hpp \
    nlohmann/detail/iterators/iteration_proxy.hpp \
    nlohmann/detail/iterators/iterator_traits.hpp \
    nlohmann/detail/iterators/json_reverse_iterator.hpp \
    nlohmann/detail/iterators/primitive_iterator.hpp \
    nlohmann/detail/json_custom_base_class.hpp \
    nlohmann/detail/json_pointer.hpp \
    nlohmann/detail/json_ref.hpp \
    nlohmann/detail/macro_scope.hpp \
    nlohmann/detail/macro_unscope.hpp \
    nlohmann/detail/meta/call_std/begin.hpp \
    nlohmann/detail/meta/call_std/end.hpp \
    nlohmann/detail/meta/cpp_future.hpp \
    nlohmann/detail/meta/detected.hpp \
    nlohmann/detail/meta/identity_tag.hpp \
    nlohmann/detail/meta/is_sax.hpp \
    nlohmann/detail/meta/std_fs.hpp \
    nlohmann/detail/meta/type_traits.hpp \
    nlohmann/detail/meta/void_t.hpp \
    nlohmann/detail/output/binary_writer.hpp \
    nlohmann/detail/output/output_adapters.hpp \
    nlohmann/detail/output/serializer.hpp \
    nlohmann/detail/string_concat.hpp \
    nlohmann/detail/string_escape.hpp \
    nlohmann/detail/string_utils.hpp \
    nlohmann/detail/value_t.hpp \
    nlohmann/json.hpp \
    nlohmann/json_fwd.hpp \
    nlohmann/ordered_map.hpp \
    nlohmann/thirdparty/hedley/hedley.hpp \
    nlohmann/thirdparty/hedley/hedley_undef.hpp \
    node_2d.hpp \
    node_basic.hpp \
    node_hybrid.hpp \
    node_lattice.hpp \
    ompl/base/DiscreteMotionValidator.h \
    ompl/base/GenericParam.h \
    ompl/base/MotionValidator.h \
    ompl/base/ProjectionEvaluator.h \
    ompl/base/ScopedState.h \
    ompl/base/SpaceInformation.h \
    ompl/base/State.h \
    ompl/base/StateSampler.h \
    ompl/base/StateSpace.h \
    ompl/base/StateSpaceTypes.h \
    ompl/base/StateValidityChecker.h \
    ompl/base/TypedSpaceInformation.h \
    ompl/base/TypedStateValidityChecker.h \
    ompl/base/ValidStateSampler.h \
    ompl/base/samplers/UniformValidStateSampler.h \
    ompl/base/spaces/DubinsMotionValidator.h \
    ompl/base/spaces/DubinsStateSpace.h \
    ompl/base/spaces/RealVectorBounds.h \
    ompl/base/spaces/RealVectorStateProjections.h \
    ompl/base/spaces/RealVectorStateSpace.h \
    ompl/base/spaces/ReedsSheppStateSpace.h \
    ompl/base/spaces/SE2StateSpace.h \
    ompl/base/spaces/SO2StateSpace.h \
    ompl/base/spaces/WrapperStateSpace.h \
    ompl/tools/config/MagicConstants.h \
    ompl/util/ClassForward.h \
    ompl/util/Console.h \
    ompl/util/Exception.h \
    ompl/util/GeometricEquations.h \
    ompl/util/ProlateHyperspheroid.h \
    ompl/util/RandomNumbers.h \
    ompl/util/String.h \
    ompl/util/Time.h \
    planner_exceptions.hpp \
    smac_planner_2d.hpp \
    smac_planner_hybrid.hpp \
    smac_planner_lattice.hpp \
    smoother.hpp \
    smoother_utils.hpp \
    thirdparty/robin_hood.h \
    types.hpp \
    utils.hpp

SOURCES += \
    a_star.cpp \
    analytic_expansion.cpp \
    collision_checker.cpp \
    costmap_downsampler.cpp \
    nav2_costmap_2d/costmap_2d.cpp \
    nav2_costmap_2d/costmap_2d_ros.cpp \
    nav2_costmap_2d/footprint.cpp \
    nav2_costmap_2d/footprint_collision_checker.cpp \
    node_2d.cpp \
    node_basic.cpp \
    node_hybrid.cpp \
    node_lattice.cpp \
    ompl/base/samplers/src/UniformValidStateSampler.cpp \
    ompl/base/spaces/scr/DubinsStateSpace.cpp \
    ompl/base/spaces/scr/RealVectorBounds.cpp \
    ompl/base/spaces/scr/RealVectorStateProjections.cpp \
    ompl/base/spaces/scr/RealVectorStateSpace.cpp \
    ompl/base/spaces/scr/ReedsSheppStateSpace.cpp \
    ompl/base/spaces/scr/SE2StateSpace.cpp \
    ompl/base/spaces/scr/SO2StateSpace.cpp \
    ompl/base/spaces/scr/WrapperStateSpace.cpp \
    ompl/base/src/DiscreteMotionValidator.cpp \
    ompl/base/src/GenericParam.cpp \
    ompl/base/src/ProjectionEvaluator.cpp \
    ompl/base/src/SpaceInformation.cpp \
    ompl/base/src/StateSampler.cpp \
    ompl/base/src/StateSpace.cpp \
    ompl/base/src/ValidStateSampler.cpp \
    ompl/util/src/Console.cpp \
    ompl/util/src/GeometricEquations.cpp \
    ompl/util/src/ProlateHyperspheroid.cpp \
    ompl/util/src/RandomNumbers.cpp \
    ompl/util/src/String.cpp \
    ompl/util/src/Time.cpp \
    smac_planner_2d.cpp \
    smac_planner_hybrid.cpp \
    smac_planner_lattice.cpp \
    smoother.cpp

# Default rules for deployment.
unix {
    target.path = /usr/lib
}
!isEmpty(target.path): INSTALLS += target



