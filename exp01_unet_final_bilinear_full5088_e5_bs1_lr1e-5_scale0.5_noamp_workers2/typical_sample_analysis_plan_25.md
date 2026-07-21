# Typical Sample Analysis Plan

## Selection Rule

The ranking metric is:

```text
delta_dice = sam3_dice - unet_dice
delta_iou  = sam3_iou  - unet_iou
```

The backend analysis pool uses 25 samples:

```text
SAM3 strong win: 10 images with the largest positive delta_dice
U-Net win:       10 images with the most negative delta_dice
Near tie:         5 images with the smallest absolute delta_dice
```

The PPT or interview deck should only show 6-8 images after visual inspection.

## Output Files

The 25-row sample pool is stored in:

```text
typical_sample_selection_25.csv
```

Use `phenomenon_tags` and `initial_reason` in that CSV as the manual observation fields after the images are available.

## Analysis And Display Split

| Type | Backend count | Display target | Purpose |
| --- | ---: | ---: | --- |
| SAM3 strong win | 10 | 2-3 | Inspect why SAM3 wins strongly and whether U-Net has boundary breaks or missing regions |
| U-Net win | 10 | 2-3 | Inspect where SAM3 loses, such as over-segmentation, boundary shift, or prompt sensitivity |
| Near tie | 5 | 1-2 | Show regular cases where both models are already strong |

The `display_candidate` column is only an initial shortlist. Because the top SAM3 rows contain several frames from the same image prefix `79a7691a90b1`, the final deck should avoid showing too many near-duplicate frames unless they reveal a clear repeated failure pattern.

## Files To Find After Dataset Is Added

For each selected `image_id`, the local dataset should contain:

```text
Original image: data/imgs/{image_id}.*
GT mask:        data/masks/{image_id}_mask.*
```

The final qualitative figure should use:

```text
Original Image | Ground Truth | U-Net Prediction | SAM3 Prediction
```

If prediction masks are also copied from the server, place or map them consistently before generating figures, for example:

```text
exp01.../predictions/unet/{image_id}.*
exp01.../predictions/sam3/{image_id}.*
```

## Per-Sample Reading Template

Use the same checklist for every image:

```text
1. Boundary fit: which mask follows the GT boundary more closely?
2. Foreground completeness: is the full car body preserved?
3. Background false positive: are road, shadow, reflection, or background objects included?
4. Sample difficulty: complex background, low contrast, shadow, blur, occlusion, or irregular boundary?
5. Initial explanation: why does SAM3 or U-Net perform better here?
```

Suggested phenomenon tags:

```text
boundary_break
missing_foreground
over_segmentation
background_interference
shadow_confusion
boundary_shift
near_tie
clean_regular_case
```

## Final Deck Rule

After checking all 25 images, choose about 6-8 for presentation:

```text
SAM3 strong win: 2-3 images
U-Net win:       2-3 images
Near tie:        1-2 images
Optional failure or abnormal example: 1 image
```

## Writing Boundary

Use cautious wording:

```text
I first screened representative samples by per-image Dice and IoU differences.
The selected examples suggest...
This does not prove a universal rule, but it helps explain the aggregate trend.
```

Avoid:

```text
SAM3 always wins.
U-Net has no value.
One example proves the reason.
```
