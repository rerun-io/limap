#ifndef LIMAP_VPLIB_VPBASE_H_
#define LIMAP_VPLIB_VPBASE_H_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/stl_bind.h>

#include "_limap/helpers.h"
#include "base/linebase.h"
#include "util/types.h"

namespace py = pybind11;

namespace limap {

namespace vplib {

class VPResult {
public:
    VPResult() {}
    VPResult(const std::vector<int>& labels_, const std::vector<V3D>& vps_): labels(labels_), vps(vps_) {}
    VPResult(const VPResult& input): labels(input.labels), vps(input.vps) {}

    std::vector<int> labels;
    std::vector<V3D> vps;

    size_t count_lines() const { return labels.size(); }
    size_t count_vps() const { return vps.size(); }
    int GetVPLabel(const int& line_id) const { return labels[line_id]; }
    V3D GetVPbyCluster(const int& vp_id) const { return vps[vp_id]; }
    bool HasVP(const int& line_id) const { return GetVPLabel(line_id) >= 0; }
    V3D GetVP(const int& line_id) const { if (HasVP(line_id)) return GetVPbyCluster(GetVPLabel(line_id)); else return V3D(0., 0., 0.); }
};

} // namespace vplib

} // namespace limap

#endif

