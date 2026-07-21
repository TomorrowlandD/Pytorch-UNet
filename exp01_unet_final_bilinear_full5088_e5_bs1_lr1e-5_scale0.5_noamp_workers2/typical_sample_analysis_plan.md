# Typical Sample Analysis Plan

## Selection Rule

The main ranking metric is:

```text
delta_dice = sam3_dice - unet_dice
```

The first analysis pass uses 8 samples:

```text
SAM3 strong win: 3 images with large positive delta_dice
U-Net win:       3 images with most negative delta_dice
Near tie:        2 images with the smallest absolute delta_dice
```

For the SAM3 strong-win group, the top rows contain several frames from the same image prefix `79a7691a90b1`. To avoid a biased visual story, the selected list keeps only one from that sequence and then uses other high-delta prefixes.

## Selected Samples

| Type | image_id | U-Net Dice | SAM3 Dice | Delta Dice | Why selected |
| --- | --- | ---: | ---: | ---: | --- |
| SAM3 strong win | `79a7691a90b1_15` | 0.8512 | 0.9949 | +0.1437 | Largest positive delta |
| SAM3 strong win | `8d5423cb763c_09` | 0.8682 | 0.9959 | +0.1277 | Large delta, different prefix |
| SAM3 strong win | `eeb7eeca738e_10` | 0.8711 | 0.9957 | +0.1246 | Large delta, different prefix |
| U-Net win | `430f0cb5666c_05` | 0.9877 | 0.9775 | -0.0102 | Most negative delta |
| U-Net win | `69915dab0755_13` | 0.9908 | 0.9813 | -0.0095 | Second most negative delta |
| U-Net win | `90b65c521a8b_03` | 0.9944 | 0.9882 | -0.0062 | High U-Net Dice, SAM3 lower |
| Near tie | `28d9a149cb02_11` | 0.9928 | 0.9928 | -0.0000 | Closest absolute delta |
| Near tie | `42a3297ccd4b_10` | 0.9947 | 0.9947 | +0.0000 | High Dice for both models |

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

## Writing Boundary

Use cautious wording:

```text
Initial observation from representative samples...
The selected examples suggest...
This does not prove a universal rule, but it helps explain the aggregate trend.
```

Avoid:

```text
SAM3 always wins.
U-Net has no value.
One example proves the reason.
```
