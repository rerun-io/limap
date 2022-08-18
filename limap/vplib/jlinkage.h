#ifndef LIMAP_VPLIB_JLINKAGE_H_
#define LIMAP_VPLIB_JLINKAGE_H_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/stl_bind.h>

#include <VPCluster.h>
#include <VPSample.h>

#include "vplib/base_vp_detector.h"

namespace py = pybind11;

namespace limap {

namespace vplib {

class JLinkageConfig: public BaseVPDetectorConfig {
public:
    JLinkageConfig(): BaseVPDetectorConfig() {}
    JLinkageConfig(py::dict dict): BaseVPDetectorConfig(dict) {}
};

class JLinkage: public BaseVPDetector {
public:
    JLinkage(): BaseVPDetector() {}
    JLinkage(const JLinkageConfig& config): config_(config) {}
    JLinkage(py::dict dict): config_(JLinkageConfig(dict)) {}
    JLinkageConfig config_;

    std::vector<int> ComputeVPLabels(const std::vector<Line2d>& lines) const; // cluster id for each line, -1 for no associated vp
    VPResult AssociateVPs(const std::vector<Line2d>& lines) const;

private:
    V3D fitVP(const std::vector<Line2d>& lines) const;
};

} // namespace vplib

} // namespace limap

#endif

