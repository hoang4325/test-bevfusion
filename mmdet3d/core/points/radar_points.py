# Added by UMONS-Numediart, Ratha SIV in 2026.

from .base_points import BasePoints
import torch

def transform_radar_points(
    points: torch.Tensor,
    attribute_dims: dict,
    rot_mat_T: torch.Tensor,
    translation: torch.Tensor = None,
) -> torch.Tensor:
    """Pure central helper to transform radar points (Task 1).

    Ensures mathematical consistency for rotation and flip operations by
    treating them both as linear transformations applied via the same
    2x2 sub-matrix.

    - Applies translation ONLY to coordinates (x, y, z).
    - Applies 2x2 linear matrix ONLY to horizontal coordinates, and
      horizontal velocity vectors (vx, vy, vx_comp, vy_comp). 

    Returns a cloned tensor; prevents in-place mutation of caller tensors.
    """
    pts = points.clone()

    # 1. Apply affine transformation to coordinates
    pts[:, :3] = pts[:, :3] @ rot_mat_T
    if translation is not None:
        pts[:, :3] += translation

    # 2. Apply purely linear 2x2 transformation to velocity (NO translation)
    # This magically handles both rotations (cos, sin) and flips (-1, 1).
    rot2d_T = rot_mat_T[:2, :2]
    ad = attribute_dims or {}

    def _transform_vel(c_x, c_y):
        n = pts.shape[1]
        if c_x is not None and c_y is not None and c_x < n and c_y < n:
            vel = pts[:, [c_x, c_y]]
            pts[:, [c_x, c_y]] = vel @ rot2d_T

    _transform_vel(ad.get("vx"), ad.get("vy"))
    _transform_vel(ad.get("vx_comp"), ad.get("vy_comp"))

    return pts


class RadarPoints(BasePoints):
    """Points of instances in radar coordinates.
    Args:
        tensor (torch.Tensor | np.ndarray | list): a N x points_dim matrix.
        points_dim (int): Number of the dimension of a point.
            Each row is (x, y, z). Default to 3.
        attribute_dims (dict): Dictionary to indicate the meaning of extra
            dimension. Default to None.
    Attributes:
        tensor (torch.Tensor): Float matrix of N x points_dim.
        points_dim (int): Integer indicating the dimension of a point.
            Each row is (x, y, z, ...).
        attribute_dims (bool): Dictionary to indicate the meaning of extra
            dimension. Default to None.
        rotation_axis (int): Default rotation axis for points rotation.
    """

    def __init__(self, tensor, points_dim=3, attribute_dims=None):
        super(RadarPoints, self).__init__(
            tensor, points_dim=points_dim, attribute_dims=attribute_dims
        )
        self.rotation_axis = 2

    def flip(self, bev_direction="horizontal"):
        """Flip points in BEV along given direction via affine matrix."""
        # Unify flip into a linear matrix multiplication to guarantee exactly
        # identical behavior for coordinates and velocity.
        if bev_direction == "horizontal":
            rot_mat_T = self.tensor.new_tensor([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
        elif bev_direction == "vertical":
            rot_mat_T = self.tensor.new_tensor([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
        else:
            rot_mat_T = torch.eye(3, device=self.tensor.device)
        
        self.tensor = transform_radar_points(self.tensor, self.attribute_dims, rot_mat_T)

    def jitter(self, amount):
        """Add random positional jitter to xyz coordinates only."""
        self.tensor = self.tensor.clone()  # Clone input to avoid alias mutation
        noise = torch.randn_like(self.tensor[:, :3]) * amount
        self.tensor[:, :3] += noise

    def scale(self, scale_factor):
        """Scale point spatial coordinates by ``scale_factor`` (but NOT velocity)."""
        self.tensor = self.tensor.clone()
        self.tensor[:, :3] *= scale_factor

    def translate(self, trans_vector):
        """Translate strictly coordinates, via the pure central helper."""
        if not isinstance(trans_vector, torch.Tensor):
            trans_vector = self.tensor.new_tensor(trans_vector)
        # Using the helper enforces that velocity NEVER receives translation
        rot_mat_T = torch.eye(3, device=self.tensor.device)
        self.tensor = transform_radar_points(
            self.tensor, self.attribute_dims, rot_mat_T, translation=trans_vector
        )

    def rotate(self, rotation, axis=None):
        """Rotate points and velocity vectors via the pure central helper."""
        if not isinstance(rotation, torch.Tensor):
            rotation = self.tensor.new_tensor(rotation)
        assert rotation.shape == torch.Size([3, 3]) or rotation.numel() == 1, \
            f"invalid rotation shape {rotation.shape}"

        if axis is None:
            axis = self.rotation_axis

        if rotation.numel() == 1:
            rot_sin = torch.sin(rotation)
            rot_cos = torch.cos(rotation)
            if axis == 1:
                rot_mat_T = rotation.new_tensor(
                    [[rot_cos, 0, -rot_sin], [0, 1, 0], [rot_sin, 0, rot_cos]]
                )
            elif axis == 2 or axis == -1:
                rot_mat_T = rotation.new_tensor(
                    [[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]]
                )
            elif axis == 0:
                # Correct Euler Rx matrix (fixes previous det=0 bug)
                rot_mat_T = rotation.new_tensor(
                    [[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]]
                )
            else:
                raise ValueError("axis should in range")
            rot_mat_T = rot_mat_T.T
        elif rotation.numel() == 9:
            rot_mat_T = rotation
        else:
            raise NotImplementedError

        # Delegate purely to the helper to prevent in-place mutation
        self.tensor = transform_radar_points(self.tensor, self.attribute_dims, rot_mat_T)

        return rot_mat_T

    def in_range_bev(self, point_range):
        """Check whether the points are in the given range.
        Args:
            point_range (list | torch.Tensor): The range of point
                in order of (x_min, y_min, x_max, y_max).
        Returns:
            torch.Tensor: Indicating whether each point is inside \
                the reference range.
        """
        in_range_flags = (
            (self.tensor[:, 0] > point_range[0])
            & (self.tensor[:, 1] > point_range[1])
            & (self.tensor[:, 0] < point_range[2])
            & (self.tensor[:, 1] < point_range[3])
        )
        return in_range_flags

    def convert_to(self, dst, rt_mat=None):
        """Convert self to ``dst`` mode.
        Args:
            dst (:obj:`CoordMode`): The target Point mode.
            rt_mat (np.ndarray | torch.Tensor): The rotation and translation
                matrix between different coordinates. Defaults to None.
                The conversion from `src` coordinates to `dst` coordinates
                usually comes along the change of sensors, e.g., from camera
                to LiDAR. This requires a transformation matrix.
        Returns:
            :obj:`BasePoints`: The converted point of the same type \
                in the `dst` mode.
        """
        from mmdet3d.core.bbox import Coord3DMode

        return Coord3DMode.convert_point(point=self, src=Coord3DMode.LIDAR, dst=dst, rt_mat=rt_mat)