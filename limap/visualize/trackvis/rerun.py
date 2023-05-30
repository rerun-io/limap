"""Module providing interactive visualization based on rerun."""
import cv2
import numpy as np
import rerun as rr
from scipy.spatial import transform

from limap.visualize.vis_lines import rerun_get_line_segments
from limap.visualize.vis_utils import test_line_inside_ranges, test_point_inside_ranges

from .base import BaseTrackVisualizer


class RerunTrackVisualizer(BaseTrackVisualizer):
    def __init__(self, tracks, bpt3d_pl=None, bpt3d_vp=None):
        super(RerunTrackVisualizer, self).__init__(tracks)
        self.bpt3d_pl = bpt3d_pl
        self.bpt3d_vp = bpt3d_vp

    def vis_all_lines(self, n_visible_views=4, width=0.01, scale=1.0):
        rr.init("limap line visualization", spawn=True)
        self._log_lines_timeless(n_visible_views, width, scale)

    def vis_reconstruction(
        self,
        imagecols,
        n_visible_views=4,
        width=0.01,
        ranges=None,
        scale=1.0,
        cam_scale=1.0,
    ):
        del cam_scale  # can be adjusted within rerun

        rr.init("limap reconstruction visualization", spawn=True)

        rr.log_view_coordinates("world", up="+Z", timeless=True)

        # all lines (i.e., full reconstruction)
        self._log_lines_timeless(n_visible_views, width, scale, ranges)

        # sequential lines (i.e., as lines are reconstructed)
        self._log_lines_per_frame(n_visible_views, width, scale, ranges)

        # cameras and images
        self._log_camviews(imagecols.get_camviews(), scale, ranges)
        # self._log_camviews_separate(imagecols.get_camviews(), scale, ranges)
        # NOTE doesn't work well right now, wait for future rerun versions, requires:
        #  adjustable camera frustum from code or per-group
        #  adjustable / hideable RGB frame / gizmo
        #  avoid views for each image to pop up initially and on reset

        # TODO visualize detected 2D lines

        # TODO visualize 3D tracks (line candidates + 2D lines)

        # visualize colmap points, and line-point associations
        if self.bpt3d_pl is not None:
            self._log_bpt3d_pl(scale, ranges)

        # visualize vanishing point association
        if self.bpt3d_vp is not None:
            self._log_bpt3d_vp(width, scale, ranges)

    def _log_lines_timeless(self, n_visible_views, width=0.01, scale=1.0, ranges=None):
        lines = self.get_lines_n_visible_views(n_visible_views)
        line_segments = rerun_get_line_segments(lines, ranges=ranges, scale=scale)
        rr.log_line_segments(
            "world/lines",
            line_segments,
            stroke_width=width,
            color=[0.9, 0.1, 0.1],
            timeless=True,
        )

    def _log_lines_per_frame(self, n_visible_views, width=0.01, scale=1.0, ranges=None):
        """Log lines based on when they are visible in n_visible views."""
        for i, track in enumerate(self.tracks):
            if track.count_images() < n_visible_views:
                continue
            line_segments = rerun_get_line_segments(
                [track.line], ranges=ranges, scale=scale
            )
            if len(line_segments) == 0:
                continue
            frame_id = track.GetSortedImageIds()[n_visible_views - 1]
            rr.set_time_sequence("frame_id", frame_id)
            rr.log_line_segments(
                f"world/sequential_lines/#{i}",
                line_segments,
                stroke_width=width,
                color=[0.9, 0.1, 0.1],
            )

    def _log_camviews(self, camviews, scale=1.0, ranges=None):
        for i, camview in enumerate(camviews):
            if ranges is not None:
                if not test_point_inside_ranges(camview.T(), ranges):
                    continue
            bgr_img = cv2.imread(camview.image_name())
            rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            width, height = camview.w(), camview.h()
            rgb_img = cv2.resize(rgb_img, (width, height))
            rr.set_time_sequence("frame_id", i)
            rr.log_image("world/camera/image", rgb_img)
            translation_xyz = camview.T() * scale
            quaternion_xyzw = transform.Rotation.from_matrix(camview.R()).as_quat()
            rr.log_rigid3(
                "world/camera",
                child_from_parent=(translation_xyz, quaternion_xyzw),
            )
            rr.log_view_coordinates("world/camera", xyz="RDF")
            rr.log_pinhole(
                "world/camera/image",
                child_from_parent=camview.K(),
                width=width,
                height=height,
            )

    def _log_camviews_separate(self, camviews, scale=1.0, ranges=None):
        for i, camview in enumerate(camviews):
            if ranges is not None:
                if not test_point_inside_ranges(camview.T(), ranges):
                    continue
            bgr_img = cv2.imread(camview.image_name())
            rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            width, height = camview.w(), camview.h()
            rgb_img = cv2.resize(rgb_img, (width, height))
            rr.set_time_sequence("frame_id", i)
            rr.log_image(f"world/cameras/#{i}/image", rgb_img)
            translation_xyz = camview.T() * scale
            quaternion_xyzw = transform.Rotation.from_matrix(camview.R()).as_quat()
            rr.log_rigid3(
                f"world/cameras/#{i}",
                child_from_parent=(translation_xyz, quaternion_xyzw),
            )
            rr.log_view_coordinates(f"world/cameras/#{i}", xyz="RDF")
            rr.log_pinhole(
                f"world/cameras/#{i}/image",
                child_from_parent=camview.K(),
                width=width,
                height=height,
            )

    def _log_bpt3d_pl(self, scale=1.0, ranges=None):
        points, degrees = [], []
        for idx, ptrack in self.bpt3d_pl.get_dict_points().items():
            p = ptrack.p * scale
            deg = self.bpt3d_pl.pdegree(idx)
            if ranges is not None:
                if not test_point_inside_ranges(p, ranges):
                    continue
            points.append(p)
            degrees.append(deg)
        points_deg0 = [p for p, deg in zip(points, degrees) if deg == 0]
        points_deg1 = [p for p, deg in zip(points, degrees) if deg == 1]
        points_deg2 = [p for p, deg in zip(points, degrees) if deg == 2]
        points_deg3p = [p for p, deg in zip(points, degrees) if deg >= 3]

        rr.log_points(
            "world/points",
            positions=points,
            colors=[0.7, 0.3, 0.3],
            radii=0.01,
            timeless=True,
        )
        rr.log_points(
            "world/pl_associations/deg0",
            positions=points_deg0,
            colors=[0.3, 0.3, 0.3],
            radii=0.01,
            timeless=True,
        )
        rr.log_points(
            "world/pl_associations/deg1",
            positions=points_deg1,
            colors=[0.3, 0.3, 0.9],
            radii=0.03,
            timeless=True,
        )
        rr.log_points(
            "world/pl_associations/deg2",
            positions=points_deg2,
            colors=[0.3, 0.9, 0.3],
            radii=0.05,
            timeless=True,
        )
        rr.log_points(
            "world/pl_associations/deg3p",
            positions=points_deg3p,
            colors=[0.9, 0.3, 0.3],
            radii=0.07,
            timeless=True,
        )

    def _log_bpt3d_vp(self, width=0.01, scale=1.0, ranges=None):
        vp_ids = self.bpt3d_vp.get_point_ids()
        vp_line_sets = {vp_id: [] for vp_id in vp_ids}
        nonvp_line_set = []
        for line_id, ltrack in self.bpt3d_vp.get_dict_lines().items():
            if ranges is not None:
                if not test_line_inside_ranges(ltrack.line, ranges):
                    continue
            vp_ids = self.bpt3d_vp.neighbor_points(line_id)
            if len(vp_ids) == 0:
                nonvp_line_set.append(ltrack.line)
                continue
            assert len(vp_ids) == 1
            vp_id = vp_ids[0]
            vp_line_sets[vp_id].append(ltrack.line)

        if len(nonvp_line_set):
            rr.log_line_segments(
                "world/vp_associations/no_vp",
                rerun_get_line_segments(nonvp_line_set, ranges, scale),
                stroke_width=width,
                color=[0.5, 0.5, 0.5],
                timeless=True,
            )

        for vp_id, vp_line_set in vp_line_sets.items():
            rr.log_line_segments(
                f"world/vp_associations/vp_{vp_id}",
                rerun_get_line_segments(vp_line_set, ranges, scale),
                stroke_width=width,
                timeless=True,
            )
