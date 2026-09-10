# SAE Architecture Ideas

Last updated: 2026-06-26

This note focuses specifically on **architectural** changes that are likely to improve SAE reconstruction performance, especially for the `Field` SAE on ViT activation maps.

The main question behind this file is:

- if the current `Field` SAE is too local or too limited, what architectural changes are most likely to improve recovery of complex activation maps?

## Main principle

The most promising changes are the ones that:

- increase the effective receptive field
- preserve dense spatial detail
- allow the SAE to represent multiple spatial scales at once
- avoid losing resolution too early

For reconstruction, this is usually better than aggressively downsampling or introducing very fancy mechanisms too early.

## Ranked architectural ideas

Ordered from most promising / practical to less urgent or more speculative.

### 1. Deeper plain `Field` SAE

Most honest first architecture upgrade.

Idea:

- keep the current overall `Field` SAE design
- increase the number of residual local blocks
- test depth before introducing more exotic architectural ideas

Why this should come first:

- the current `Field` SAE may simply be too shallow
- for large and complex activation maps, the plain hypothesis “the model is underpowered” is the cleanest one to test first
- this is the least disruptive architecture change
- if deeper plain `Field` already helps, that is easier to interpret than immediately moving to multiscale or ASPP-style blocks

Recommended first settings:

- `d = 2C`, `n_blocks = 6`
- `d = 2C`, `n_blocks = 8`

### 2. Wider plain `Field` SAE

Natural second step after depth.

Idea:

- keep the same structure
- increase hidden width from about `2C` to something like `4C`

Why:

- if the field is complex, the model may also be channel-capacity limited
- width is still a simple expressivity upgrade, without changing the model’s inductive bias too much

Recommended first settings:

- `d = 4C`, `n_blocks = 6`
- `d = 4C`, `n_blocks = 8`

### 3. Multiscale parallel branches

Idea:

- run several spatial kernels in parallel inside one block
- examples:
  - `1x1`
  - `3x3`
  - `5x5`
  - maybe a dilated `3x3`
- concatenate the branch outputs
- fuse them with a final `1x1`

Why this is promising:

- activation maps likely contain structure at different spatial scales
- the current `Field` SAE is mostly single-scale and local
- a multiscale block lets the model choose which spatial scale matters for each feature

Best way to use it:

- replace the current plain residual local block with a multibranch residual block
- use `1x1` bottlenecks before expensive branches to control cost

### 4. Mixed dilations / ASPP-style context

Very strong candidate.

Idea:

- use several dilation rates in parallel or in alternating sequence
- examples:
  - dilation `1`
  - dilation `2`
  - dilation `4`
- fuse the outputs

Why this is promising:

- it increases receptive field without downsampling
- that is important for dense reconstruction
- ViT token-grid features may need broader context than local `5x5` blocks provide

Important caution:

- do not stack the same dilation repeatedly without variation
- repeated same-rate dilation can create gridding artifacts

Best version:

- ASPP-style parallel dilated branches
- or hybrid dilation patterns such as `1,2,5`

### 5. Larger depthwise kernels

Strong and simple candidate.

Idea:

- replace or augment `5x5` depthwise kernels with larger ones
- examples:
  - `7x7`
  - maybe larger if stable and affordable

Why this is promising:

- larger depthwise kernels are cheap relative to dense large kernels
- they provide more context per layer
- this may be especially useful for transformer activation maps, which often contain broader spatial dependencies

Best use:

- combine with residual blocks and `1x1` channel mixing
- possibly mix `5x5` and `7x7` blocks rather than using only one size

### 6. Bottleneck residual blocks

Useful, but not likely to be the main fix by itself.

Idea:

- use a bottleneck-style internal block
- for example:
  - `1x1 reduce`
  - spatial conv or depthwise conv
  - `1x1 expand`
  - residual add

Why it helps:

- allows deeper/wider networks at lower cost
- improves parameter efficiency
- makes multiscale or larger-kernel blocks more affordable

Why it is not the top recommendation:

- it improves efficiency and depth, but does not automatically solve the main receptive-field issue
- it is best used together with multiscale branches or larger kernels

## Multiscale encoder-decoder ideas

These are more invasive than block-level upgrades, but potentially very strong.

### 7. Small U-Net / FPN-like `Field` SAE

Potentially very strong if simpler changes are not enough.

Idea:

- add a small downsample / upsample hierarchy inside the SAE
- merge coarse semantic context back into higher-resolution representations

Why this is promising:

- gives the model both local detail and larger-scale context
- mirrors the idea behind U-Net / FPN:
  - high resolution keeps localization
  - lower resolution gives broader semantic context

Why it is later in the priority list:

- more code
- more tuning
- more opportunities to accidentally hurt reconstruction if the design is careless

### 8. Strided conv downsampling

Possible, but not a first-line choice by itself.

Idea:

- downsample with learned strided convs instead of pure pooling

Why it can help:

- learned downsampling may capture broader context efficiently

Why it is risky:

- reconstruction is sensitive to spatial detail
- downsampling too aggressively can destroy exactly the information you later need to rebuild

Best use:

- only inside a proper encoder-decoder with a strong upsampling path

### 9. Pixel shuffle upsampling

A decoder-side option, but not likely to be the main driver.

Idea:

- decode at low resolution
- then rearrange channels into a finer spatial grid

Why it may help:

- efficient upsampling
- avoids checkerboard artifacts from some transposed-conv designs

Why it is not a top priority:

- it is mainly an implementation choice for decoder quality
- it is not the core missing ingredient if the main issue is insufficient receptive field or insufficient multiscale context

## Lower-priority ideas

### 10. Blur pooling / anti-aliased downsampling

Probably low priority for this exact problem.

Why:

- useful for shift consistency and robustness
- less obviously central to sparse activation recovery quality

### 11. Deformable convolutions

Interesting but too fancy for early-stage fixes.

Why:

- adaptive receptive fields are conceptually attractive
- but the added complexity and instability make this a poor first move

This should come only after simpler multiscale and dilation ideas are tested.

## Best combined architecture directions

The strongest realistic next `Field` SAE upgrades are likely combinations of the above, not isolated tricks.

### Candidate A: conservative upgrade

- keep the current overall encoder-decoder shape
- make it wider and deeper
- replace current residual blocks with bottleneck residual blocks
- use larger kernels such as `7x7`

Why:

- low-risk
- likely better than the current local stack

### Candidate B: strongest near-term upgrade

- use multiscale parallel branches in each residual block
- include `3x3`, `5x5`, and dilated `3x3`
- fuse with `1x1`
- keep residual connections

Why:

- probably the best tradeoff between expressivity and implementation complexity
- directly targets the “single local scale may be insufficient” problem

### Candidate C: more ambitious upgrade

- small U-Net / FPN-like SAE
- preserve a sparse bottleneck in the middle
- use multiscale merging during encode/decode

Why:

- most likely to capture both detail and global-ish context
- probably strongest long-run direction if simpler fixes still leave a gap

## Recommended practical order

If architecture work starts now, the best order is:

1. deeper plain `Field`
2. wider plain `Field`
3. larger depthwise kernels
4. multiscale parallel residual blocks
5. mixed dilation / ASPP-style context
6. bottleneck-ized residual blocks for efficiency
7. only then move to a U-Net / FPN-like SAE

## Bottom line

If only one architecture idea from this note is tried first, it should be:

- **a deeper plain `Field` SAE**

If two are tried together, the best first pair is probably:

- **deeper plain `Field`**
- **wider plain `Field`**

If that still underperforms, the next strongest pair is probably:

- **multiscale branches**
- **mixed dilation or larger depthwise kernels**

So the most likely sequence is:

- first test whether the current model is simply underpowered
- then add more sophisticated multiscale inductive bias only if needed
