"""Module providing interactive visualization based on rerun."""
import os
import sys

import rerun as rr

from .base import BaseTrackVisualizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vis_lines import rerun_get_line_segments
from vis_utils import compute_robust_range_lines


class RerunTrackVisualizer(BaseTrackVisualizer):
    def __init__(self, tracks):
        super(RerunTrackVisualizer, self).__init__(tracks)
        self.reset()

    def vis_all_lines(self, n_visible_views=4, width=2, scale=1.0):
        rr.init("limap line visualization", spawn=True)
        lines = self.get_lines_n_visible_views(n_visible_views)
        line_segments = rerun_get_line_segments(lines, scale=scale)
        rr.log_line_segments(
            "lines", line_segments, stroke_width=width, color=[1.0, 0.0, 0.0]
        )

    def vis_reconstruction(
        self,
        imagecols,
        n_visible_views=4,
        width=2,
        ranges=None,
        scale=1.0,
        cam_scale=1.0,
    ):
        rr.init("limap reconstruction visualization", spawn=True)

        # lines
        lines = self.get_lines_n_visible_views(n_visible_views)
        line_segments = rerun_get_line_segments(lines, ranges=ranges, scale=scale)
        rr.log_line_segments(
            "lines", line_segments, stroke_width=width, color=[1.0, 0.0, 0.0]
        )

        lranges = compute_robust_range_lines(lines)
        scale_cam_geometry = abs(lranges[1, :] - lranges[0, :]).max()

        # TODO log images

        # TODO how to log sequence-mode reconstruction

        # TODO visualize other data stored in output (keypoints, detected 2D lines)
