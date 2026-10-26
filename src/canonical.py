"""Step 4: rotate a part mesh into a canonical pose before rendering.

The same part can arrive in any orientation (long side along X in one file,
along Z in another), and fixed camera views then see different pictures.
Inertia axes (PCA) are the textbook fix but fail on CAD data: an L-bracket's
inertia axes are tilted ~24 degrees from its legs, and parts with two equal
moments (cylinders, square plates; 114 of 187 here) have no defined axes.

CAD parts are built from flat faces, so their own axes come from the faces:
    1. axis A = the flat-face normal direction with the most area
    2. axis B = the heaviest normal direction perpendicular to A; if there is
       none (a pin: only its end faces are flat), the rotation about A that
       gives the smallest 2D bounding box
    3. no flat faces at all (freeform): fall back to inertia axes
Then the axes are ordered by extent (longest -> X, shortest -> Z, so flat
parts lie flat), X and Y are flipped so the center of mass sits on their
positive side, and Z = X cross Y keeps the frame right-handed: the part is
never mirrored, so left- and right-hand versions of a part stay distinct.
"""
import numpy as np
import trimesh

PERP_TOL = np.cos(np.radians(80))  # |dot| below this counts as perpendicular
MIN_B_SHARE = 0.02  # a perpendicular face direction needs this share of total area to define B


def _fold(normals):
    """Map n and -n to the same direction: largest-magnitude component positive."""
    idx = np.abs(normals).argmax(axis=1)
    sign = np.sign(normals[np.arange(len(normals)), idx])
    return normals * sign[:, None]


def _face_directions(mesh):
    """Distinct flat-face directions with their total area, heaviest first."""
    folded = _fold(mesh.face_normals)
    keys = np.round(folded * 200).astype(np.int64)  # ~0.3 degree bins; triangles of one CAD face share a normal
    uniq, inverse = np.unique(keys, axis=0, return_inverse=True)
    area = np.bincount(inverse.ravel(), weights=mesh.area_faces)
    dirs = np.zeros((len(uniq), 3))
    np.add.at(dirs, inverse.ravel(), folded * mesh.area_faces[:, None])
    norm = np.linalg.norm(dirs, axis=1)
    keep = norm > 1e-12  # a bin whose folded normals cancel (sign tie at the fold) has no direction
    dirs, area = dirs[keep] / norm[keep, None], area[keep]
    order = np.argsort(-area)
    return dirs[order], area[order] / mesh.area


def _min_box_about(points, a):
    """Unit vector perpendicular to `a` that minimizes the 2D bounding box of the points around `a`."""
    u = np.cross(a, [1, 0, 0] if abs(a[0]) < 0.9 else [0, 1, 0])
    u /= np.linalg.norm(u)
    v = np.cross(a, u)
    p = np.c_[points @ u, points @ v]
    p = p[np.unique(np.round(p, 6), axis=0, return_index=True)[1]]  # fewer points, same box

    def box_area(angles):
        c, s = np.cos(angles), np.sin(angles)
        x = p[:, :1] * c + p[:, 1:] * s  # (n_points, n_angles)
        y = -p[:, :1] * s + p[:, 1:] * c
        return np.ptp(x, axis=0) * np.ptp(y, axis=0)

    coarse = np.radians(np.arange(0, 90, 0.5))
    best = coarse[np.argmin(box_area(coarse))]
    fine = best + np.radians(np.arange(-0.5, 0.5, 0.01))  # refine: the coarse grid alone drifts by up to 0.25 deg
    best = fine[np.argmin(box_area(fine))]
    return np.cos(best) * u + np.sin(best) * v


def canonical_frame(mesh):
    """Return (rotation rows, origin, method). Canonical coordinates = (v - origin) @ rotation.T"""
    dirs, share = _face_directions(mesh)
    flat = share > 1e-3
    if flat.any():
        a = dirs[0]
        perp = flat & (np.abs(dirs @ a) < PERP_TOL) & (share >= MIN_B_SHARE)
        if perp.any():
            b = dirs[np.argmax(perp)]
            method = "faces"
        else:
            b = _min_box_about(mesh.vertices, a)
            method = "face+minbox"
        b = b - (b @ a) * a  # make exactly perpendicular
        b /= np.linalg.norm(b)
        axes = np.array([a, b, np.cross(a, b)])
    else:
        axes = mesh.principal_inertia_vectors
        method = "inertia"

    local = mesh.vertices @ axes.T
    lo, hi = local.min(axis=0), local.max(axis=0)
    diag = np.linalg.norm(hi - lo)
    # Sign from the mass distribution: area-weighted centroid vs bbox center.
    com = (mesh.triangles_center * mesh.area_faces[:, None]).sum(axis=0) / mesh.area
    offset = com @ axes.T - (lo + hi) / 2

    # Longest extent first. Extents within 1% count as equal (square cross-sections);
    # then the axis with the more lopsided mass goes first, so the order doesn't
    # depend on which face direction happened to be found first.
    extent = np.round((hi - lo) / diag, 2)
    order = np.lexsort((-np.abs(offset) / diag, -extent))
    axes, offset = axes[order], offset[order]

    # Point every axis toward the heavier side. Symmetric axes (offset ~ 0) can go
    # either way; if the result is left-handed, flip the least certain axis rather
    # than mirror the part, so left- and right-hand versions stay distinct.
    axes = axes * np.where(offset < 0, -1, 1)[:, None]
    if np.linalg.det(axes) < 0:
        weakest = np.argmin(np.abs(offset))
        axes[weakest] = -axes[weakest]

    local = mesh.vertices @ axes.T  # bbox center in the final, possibly flipped, frame
    origin = axes.T @ ((local.min(axis=0) + local.max(axis=0)) / 2)
    return axes, origin, method


def canonicalize(mesh):
    """Return a copy of the mesh in canonical pose (bbox centered at the origin) and the method used."""
    axes, origin, method = canonical_frame(mesh)
    out = mesh.copy()
    out.vertices = (mesh.vertices - origin) @ axes.T
    return out, method


def random_rotation(rng):
    """Uniform random 3D rotation matrix."""
    return trimesh.transformations.random_rotation_matrix(rng.random(3))[:3, :3]


def axis_swap_rotation(rng):
    """A random non-identity rotation from the 24 that map the world axes onto each other."""
    while True:
        perm = rng.permutation(3)
        signs = rng.choice([-1, 1], size=3)
        r = np.zeros((3, 3))
        r[np.arange(3), perm] = signs
        if np.linalg.det(r) > 0 and not np.allclose(r, np.eye(3)):
            return r
