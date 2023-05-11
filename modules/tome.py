import launch

if not launch.is_installed("tomesd"):
    print("Cannot import tomesd, please install it manually following the instructions on https://github.com/dbolya/tomesd",file=sys.stderr)
else:
    import tomesd
    from modules.shared import opts

    def apply_tome(sd_model, hires=False):
        ratio = opts.token_merging_ratio
        if hires:
            ratio = opts.token_merging_ratio_hr
        
        tomesd.apply_patch(
            sd_model,
            ratio=ratio,
            max_downsample=opts.token_merging_maximum_down_sampling,
            sx=opts.token_merging_stride_x,
            sy=opts.token_merging_stride_y,
            use_rand=opts.token_merging_random,
            merge_attn=opts.token_merging_merge_attention,
            merge_crossattn=opts.token_merging_merge_cross_attention,
            merge_mlp=opts.token_merging_merge_mlp
        )
    
    def remove_tome(sd_model):
        tomesd.remove_patch(sd_model)