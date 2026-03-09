import argparse

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke-test BEVFusion tri-modal forward")
    parser.add_argument(
        "--config",
        default="configs/nuscenes/det/transfusion/secfpn/camera+lidar+radar/swint_v0p075/convfuser.yaml",
        help="config file path",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-cams", type=int, default=6)
    parser.add_argument("--num-lidar-points", type=int, default=4096)
    parser.add_argument("--num-radar-points", type=int, default=2048)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    return parser.parse_args()


def identity(batch_size, n=None, device="cpu"):
    base = torch.eye(4, device=device).float()
    if n is None:
        return base.unsqueeze(0).repeat(batch_size, 1, 1)
    return base.unsqueeze(0).unsqueeze(0).repeat(batch_size, n, 1, 1)


def random_points(num_points, dim, pc_range, device):
    points = torch.randn(num_points, dim, device=device).float()
    points[:, 0] = torch.empty(num_points, device=device).uniform_(pc_range[0], pc_range[3])
    points[:, 1] = torch.empty(num_points, device=device).uniform_(pc_range[1], pc_range[4])
    points[:, 2] = torch.empty(num_points, device=device).uniform_(max(0.5, pc_range[2]), pc_range[5])
    return points


def build_dummy_batch(
    cfg,
    batch_size,
    num_cams,
    num_lidar_points,
    num_radar_points,
    device,
    box_type_3d_cls,
):
    image_h, image_w = cfg.image_size
    pc_range = cfg.point_cloud_range

    lidar_dim = int(cfg.model.encoders.lidar.backbone.in_channels)
    radar_dim = int(cfg.model.encoders.radar.backbone.pts_voxel_encoder.in_channels)

    img = torch.randn(batch_size, num_cams, 3, image_h, image_w, device=device).float()
    points = [random_points(num_lidar_points, lidar_dim, pc_range, device) for _ in range(batch_size)]
    radar = [random_points(num_radar_points, radar_dim, pc_range, device) for _ in range(batch_size)]

    camera2ego = identity(batch_size, num_cams, device=device)
    lidar2ego = identity(batch_size, device=device)
    lidar2camera = identity(batch_size, num_cams, device=device)
    camera2lidar = identity(batch_size, num_cams, device=device)
    img_aug_matrix = identity(batch_size, num_cams, device=device)
    lidar_aug_matrix = identity(batch_size, device=device)

    camera_intrinsics = identity(batch_size, num_cams, device=device)
    camera_intrinsics[..., 0, 0] = 500.0
    camera_intrinsics[..., 1, 1] = 500.0
    camera_intrinsics[..., 0, 2] = image_w / 2.0
    camera_intrinsics[..., 1, 2] = image_h / 2.0

    lidar2image = camera_intrinsics.clone()

    metas = [{"box_type_3d": box_type_3d_cls, "token": f"dummy-{idx}"} for idx in range(batch_size)]

    return dict(
        img=img,
        points=points,
        camera2ego=camera2ego,
        lidar2ego=lidar2ego,
        lidar2camera=lidar2camera,
        lidar2image=lidar2image,
        camera_intrinsics=camera_intrinsics,
        camera2lidar=camera2lidar,
        img_aug_matrix=img_aug_matrix,
        lidar_aug_matrix=lidar_aug_matrix,
        metas=metas,
        radar=radar,
    )


def main():
    args = parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, use --device cpu for a CPU-only smoke test.")
    try:
        from torchpack.utils.config import configs
    except ImportError as exc:
        raise RuntimeError(
            "torchpack is required for recursive config loading. "
            "Please activate the BEVFusionx environment first."
        ) from exc

    try:
        from mmcv import Config
        from mmdet3d.core.bbox import LiDARInstance3DBoxes
        from mmdet3d.models import build_model
        from mmdet3d.utils import recursive_eval
    except ImportError as exc:
        raise RuntimeError(
            "mmcv/mmdet3d dependencies are not available. "
            "Please run this script inside the BEVFusionx runtime environment."
        ) from exc

    configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(configs), filename=args.config)

    model = build_model(cfg.model)
    model.to(device)
    model.eval()

    required = {"camera", "lidar", "radar"}
    available = set(model.encoders.keys())
    if required != available:
        raise ValueError(f"Expected encoders {sorted(required)}, got {sorted(available)}")

    batch = build_dummy_batch(
        cfg,
        args.batch_size,
        args.num_cams,
        args.num_lidar_points,
        args.num_radar_points,
        device,
        LiDARInstance3DBoxes,
    )

    with torch.no_grad():
        outputs = model(**batch)

    if not isinstance(outputs, list) or len(outputs) != args.batch_size:
        raise AssertionError(f"Unexpected output type/size: type={type(outputs)}, len={len(outputs)}")

    print("Smoke test passed.")
    print(f"sensor_order: {model.sensor_order}")
    print(f"batch size: {len(outputs)}")
    print(f"output keys[0]: {list(outputs[0].keys())}")


if __name__ == "__main__":
    main()
