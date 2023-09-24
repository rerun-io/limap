"""Module providing interactive visualization based on rerun."""
from collections import defaultdict

import cv2
import rerun as rr

from limap.visualize.vis_lines import rerun_get_line_strips
from limap.visualize.vis_utils import test_line_inside_ranges, test_point_inside_ranges

from .base import BaseTrackVisualizer


class RerunTrackVisualizer(BaseTrackVisualizer):
    def __init__(self, tracks, bpt3d_pl=None, bpt3d_vp=None, segments2d_dict=None):
        super(RerunTrackVisualizer, self).__init__(tracks)
        self.bpt3d_pl = bpt3d_pl
        self.bpt3d_vp = bpt3d_vp
        self.segments2d_dict = segments2d_dict

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

        rr.log_view_coordinates("world", up="-Y", timeless=True)

        # all lines (i.e., full reconstruction)
        self._log_lines_timeless(n_visible_views, width, scale, ranges)

        # sequential lines (i.e., as lines are reconstructed)
        self._log_lines_per_frame(n_visible_views, width, scale, ranges)

        # cameras and images
        self._log_camviews(imagecols, scale, ranges)
        # self._log_camviews_separate(imagecols.get_camviews(), scale, ranges)
        # NOTE doesn't work well right now, wait for future rerun versions, requires:
        #  adjustable camera frustum from code or per-group
        #  adjustable / hideable RGB frame / gizmo
        #  avoid views for each image to pop up initially and on reset

        # 2d line detections
        self._log_line_detections()

        # 3d tracks (candidates + detections)
        self._log_tracks(n_visible_views, width, scale, ranges)
        self._log_single_track(0, width, scale, ranges)

        # colmap points and line-point associations
        if self.bpt3d_pl is not None:
            self._log_bpt3d_pl(scale, ranges)

        # vanishing point association
        if self.bpt3d_vp is not None:
            self._log_bpt3d_vp(width, scale, ranges)

    def _log_lines_timeless(self, n_visible_views, width=0.01, scale=1.0, ranges=None):
        lines = self.get_lines_n_visible_views(n_visible_views)
        line_segments = rerun_get_line_strips(lines, ranges=ranges, scale=scale)
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
            line_segments = rerun_get_line_strips(
                [track.line], ranges=ranges, scale=scale
            )
            if len(line_segments) == 0:
                continue
            img_id = track.GetSortedImageIds()[n_visible_views - 1]
            rr.set_time_sequence("img_id", img_id)
            rr.log_line_segments(
                f"world/sequential_lines/#{i}",
                line_segments,
                stroke_width=width,
                color=[0.9, 0.1, 0.1],
            )

    def _log_camviews(self, imagecols, scale=1.0, ranges=None):
        for img_id, camview in imagecols.get_map_camviews().items():
            if ranges is not None:
                if not test_point_inside_ranges(camview.T(), ranges):
                    continue
            bgr_img = cv2.imread(camview.image_name())
            rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            width, height = camview.w(), camview.h()
            rgb_img = cv2.resize(rgb_img, (width, height))
            rr.set_time_sequence("img_id", img_id)
            rr.log_image("world/camera/image", rgb_img)
            rr.log_transform3d(
                "world/camera",
                rr.TranslationAndMat3(camview.T() * scale, camview.R()),
                from_parent=True,
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
            rr.set_time_sequence("img_id", i)
            rr.log_image(f"world/cameras/#{i}/image", rgb_img)
            rr.log_transform3d(
                "world/camera",
                rr.TranslationAndMat3(camview.T() * scale, camview.R()),
                from_parent=True,
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
                rerun_get_line_strips(nonvp_line_set, ranges, scale),
                stroke_width=width,
                color=[0.5, 0.5, 0.5],
                timeless=True,
            )

        for vp_id, vp_line_set in vp_line_sets.items():
            rr.log_line_segments(
                f"world/vp_associations/vp_{vp_id}",
                rerun_get_line_strips(vp_line_set, ranges, scale),
                stroke_width=width,
                timeless=True,
            )

    def _log_tracks(self, n_visible_views, width=0.02, scale=1.0, ranges=None):
        candidate_lines = []
        for track in self.tracks:
            if track.count_images() < n_visible_views:
                continue
            candidate_lines += track.line3d_list

        rr.log_line_segments(
            "world/candidate_lines",
            rerun_get_line_strips(candidate_lines, ranges, scale),
            stroke_width=width * 0.5,
            color=[0.1, 0.9, 0.1],
            timeless=True,
        )

    def _log_single_track(self, track_id, width=0.02, scale=1.0, ranges=None):
        if track_id >= len(self.tracks):
            print("Specified track_id not available.")
            return

        track = self.tracks[track_id]

        line_segments = rerun_get_line_strips(
            [track.line], ranges=ranges, scale=scale
        )
        if len(line_segments) == 0:
            return

        rr.log_line_segments(
            f"world/track_{track_id}/final_line",
            line_segments,
            stroke_width=width,
            color=[1.0, 0.0, 0.0],
            timeless=True,
        )
        min_score = min(track.score_list)
        max_score = max(track.score_list)
        lines_2d_dict = defaultdict(list)
        for line_id, (img_id, line, score, line_2d) in enumerate(
            zip(
                track.image_id_list,
                track.line3d_list,
                track.score_list,
                track.line2d_list,
            )
        ):
            line_segments_3d = rerun_get_line_strips(
                [line], ranges=ranges, scale=scale
            )
            if len(line_segments_3d) == 0:
                continue

            rr.set_time_sequence("img_id", img_id)
            rr.log_line_segments(
                f"world/track_{track_id}/lines/#{line_id}",
                line_segments_3d,
                stroke_width=width * 0.5,
                color=[0.1, (score - min_score) / (max_score - min_score), 0.1],
            )
            lines_2d_dict[img_id].append(line_2d)

        for img_id, lines_2d in lines_2d_dict.items():
            line_segments_2d = rerun_get_line_strips(
                lines_2d, ranges=ranges, scale=scale
            )
            rr.set_time_sequence("img_id", img_id)
            rr.log_line_segments(
                f"world/camera/image/line_track_{track_id}",
                line_segments_2d,
                color=[0.1, (score - min_score) / (max_score - min_score), 0.1],
            )
            if img_id + 1 not in lines_2d_dict:
                rr.set_time_sequence("img_id", img_id + 1)
                rr.log_cleared(f"world/camera/image/line_track_{track_id}")

    def _log_line_detections(self):
        for img_id, segments_2d in self.segments2d_dict.items():
            line_segments_2d = segments_2d.reshape(-1, 2)
            rr.set_time_sequence("img_id", img_id)
            rr.log_line_segments(
                "world/camera/image/detected_lines",
                line_segments_2d,
            )
