// Post step: undo thin-wall crossings caused by stylization.
//
// Solid-shell meshes (a teapot spout with an inner and an outer surface)
// have pairs of surfaces a short distance apart that face opposite ways.
// Nothing in the stylization energy ties them together, so they can be
// pushed through each other. Using the rest pose, every vertex is paired
// with the opposite wall behind it (a short ray cast along -normal); after
// stylization, wherever a wall got thinner than min_thickness times its rest
// thickness (or crossed), both sides are pushed apart along the normal, with
// the correction smoothed over the surface. Repeats until nothing is thin.
//
// The same is done for narrow gaps in front of a vertex (an opposite-facing
// surface across air, e.g. a lid resting on a rim, or the bore of a spout),
// so touching parts do not sink into each other.

#pragma once

#include "cubic_stylizer.h"

namespace cubify {

struct ThinWallResult {
  RowMatX3d V;
  int walls = 0;           // wall + gap pairs found in the rest pose
  int crossed_before = 0;  // of those, how many had crossed it
  int crossed_after = 0;
  int moved = 0;           // vertices the fix moved
  int passes = 0;
  double max_move = 0.0;
};

// V_rest: pose the walls are measured in (before stylization); V: stylized
// positions to fix; min_thickness: fraction of the rest thickness a wall
// is pushed back to; max_wall: thickest wall considered, as a fraction of
// the rest bounding-box diagonal.
ThinWallResult fix_thin_walls(const RowMatX3d& V_rest, const RowMatX3d& V, const RowMatX3i& F,
                              double min_thickness, double max_wall, int iterations,
                              ThreadPool& pool);

}  // namespace cubify
