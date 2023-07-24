"""Model config for DINO."""
_base_ = [
    "../../../../../recipes/stages/detection/incremental.py",
]
model = dict(
    type="CustomDINO",
    backbone=dict(type="EfficientNet", arch="b0", out_indices=(3, 4, 5)),
    neck=dict(
        type="ChannelMapper",
        in_channels=[40, 112, 320],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type="GN", num_groups=32),
        num_outs=4,
    ),
    bbox_head=dict(
        type="CustomDINOHead",
        num_query=900,
        num_classes=80,
        in_channels=2048,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=True,
        transformer=dict(
            type="CustomDINOTransformer",
            encoder=dict(
                type="DetrTransformerEncoder",
                num_layers=6,
                transformerlayers=dict(
                    type="BaseTransformerLayer",
                    attn_cfgs=dict(type="MultiScaleDeformableAttention", embed_dims=256, dropout=0.0),
                    feedforward_channels=2048,
                    ffn_dropout=0.0,
                    operation_order=("self_attn", "norm", "ffn", "norm"),
                ),
            ),
            decoder=dict(
                type="DINOTransformerDecoder",
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type="DetrTransformerDecoderLayer",
                    attn_cfgs=[
                        dict(type="MultiheadAttention", embed_dims=256, num_heads=8, dropout=0.0),
                        dict(type="MultiScaleDeformableAttention", embed_dims=256, dropout=0.0),
                    ],
                    feedforward_channels=2048,
                    ffn_dropout=0.0,
                    operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
                ),
            ),
        ),
        positional_encoding=dict(
            type="SinePositionalEncoding", num_feats=128, normalize=True, offset=0.0, temperature=20
        ),
        loss_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
        loss_bbox=dict(type="L1Loss", loss_weight=5.0),
        loss_iou=dict(type="GIoULoss", loss_weight=2.0),
        dn_cfg=dict(
            label_noise_scale=0.5,
            box_noise_scale=1.0,  # 0.4 for DN-DETR
            group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=100),
        ),
    ),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type="HungarianAssigner",
            cls_cost=dict(type="FocalLossCost", weight=1.0),
            reg_cost=dict(type="BBoxL1Cost", weight=5.0, box_format="xywh"),
            iou_cost=dict(type="IoUCost", iou_mode="giou", weight=2.0),
        )
    ),
    test_cfg=dict(max_per_img=300),
)
# optimizer
optimizer = dict(
    _delete_=True,
    type="AdamW",
    lr=1e-4,
    weight_decay=0.0001,
)
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
load_from = "https://download.openmmlab.com/mmclassification/v0/efficientnet/\
efficientnet-b0_3rdparty_8xb32_in1k_20220119-a7e2a0b1.pth"
resume_from = None
ignore = False
