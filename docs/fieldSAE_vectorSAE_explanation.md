# Appendix — How VectorSAE and FieldSAE work

This appendix compares the historical **FieldSAE** used in the CNN transition chain with the later clean-redo **ContextVectorSAE** (`vector_context`) requested for the Vector diagram. They are not a matched historical benchmark pair: FieldSAE was the historical chain winner, while ContextVectorSAE belongs to the later corrected evaluation program.

Both models answer the same question:

> Can I replace a full CNN activation map with a small list of learned **feature, location, value** entries, rebuild the activation map, and preserve the rest of the CNN's behavior?

For one CNN checkpoint, the input is an activation map with shape `C × H × W`:

- `C` is the number of channels.
- `H × W` is the spatial grid.
- Each location contains a `C`-number vector.

Both architectures create candidate feature-location values and retain a fixed number for **each image**. Historical FieldSAE used `K = 8C` candidate feature channels and global Top-K over its activation values. ContextVectorSAE uses `K = 4C`, learns a separate selection score and signed value for each candidate, and applies exact global Top-M to the learned scores. The kept entries retain their feature identity, spatial location, and numeric value.

Neither model learns a permanent set of coordinates. Each encoder creates an input-dependent candidate map, then Top-K or Top-M selects the retained pairs separately for each image.

---

## In-text arrow diagrams

### Historical FieldSAE

```text
Full CNN activation map (C × H × W)
        ↓
1×1 channel mixing: turn each location into an internal feature vector
        ↓
Three padded 5×5 residual blocks: each location combines information from nearby locations
        ↓
1×1 output: make K = 8C candidate feature values at every location (K × H × W)
        ↓
Global Top-K: keep only the largest candidate values for this image; set all others to zero
        ↓
Sparse code: a short list of [feature identity, location, value] entries
        ↓
1×1 decoder + three padded 5×5 residual blocks: let retained entries help rebuild nearby locations
        ↓
Dense rebuilt CNN activation map (C × H × W)
```

The padded 5×5 blocks do not make the map smaller. They keep the same `H × W` size while allowing one position to use nearby information. The representation becomes sparse only at the **Global Top-K** step.

### ContextVectorSAE (later clean-redo version)

```text
Full CNN activation map (C × H × W)
        ↓
1×1 channel mixing + local convolution blocks + pooled attention
        ↓
For every feature-location pair, make two learned outputs:
  • a score for whether to keep it
  • a signed value to send if it is kept
        ↓
Exact global Top-M: keep exactly M highest-scoring pairs for this image
        ↓
Sparse code: a short list of [feature identity, location, signed value] entries
        ↓
1×1 direct vector decoder: rebuild each retained entry at its own location
        ↓
Local convolutional refiner: use nearby retained entries to add a spatial correction
        ↓
Dense rebuilt CNN activation map (C × H × W)
```

The main distinction is that ContextVectorSAE explicitly learns a separate keep-score and sent value. FieldSAE's historical version ranked its nonnegative candidate values themselves.

---

## ContextVectorSAE — vector features with cross-location mixing

### One-sentence description

ContextVectorSAE keeps a direct vector-style feature decoder but uses cross-location context when scoring feature-location pairs and when refining the reconstruction.

Standalone diagram assets: [`vector_sae_diagram.tex`](appendix_diagrams/vector_sae_diagram.tex) and [`vector_sae_diagram.jpg`](appendix_diagrams/vector_sae_diagram.jpg).

### How the context-aware VectorSAE works

1. A contextual encoder examines the activation map with local convolution blocks and pooled attention. A candidate score at `(x, y)` can therefore use nearby positions and pooled global context.
2. Two output heads produce a learned selection score and a signed stored value for every feature-location pair.
3. Exact Top-M keeps the entries with the highest scores for that image.
4. A direct `1×1` vector decoder produces a location-wise reconstruction from the retained values.
5. A contextual refiner mixes nearby retained entries and adds a cross-location correction to that direct reconstruction.

The model therefore preserves the intuitive “feature at a location” representation while allowing context to affect both selection and reconstruction.

### Why this is different from the historical Vector baseline

The old VectorSAE used only a `1×1` encoder and `1×1` decoder, so every location was processed independently. ContextVectorSAE was added later to test whether the weakness of that baseline came from the lack of spatial context rather than from vector features themselves.

---

## FieldSAE — a spatially aware learned representation

### One-sentence description

FieldSAE learns feature-location values using nearby positions in the activation map, and it also uses nearby sparse values when rebuilding the activation map.

Standalone diagram assets: [`field_sae_diagram.tex`](appendix_diagrams/field_sae_diagram.tex) and [`field_sae_diagram.jpg`](appendix_diagrams/field_sae_diagram.jpg).

### What “spatially aware” means here

When FieldSAE scores a candidate at `(x, y)`, its score can depend on the activation at `(x, y)` **and on nearby positions**. The local residual blocks repeatedly mix information from small neighborhoods before producing the candidate feature map.

That means FieldSAE can use local arrangements such as:

- a short edge continuing across nearby positions;
- a texture pattern spread over a neighborhood;
- a local combination of channels that only makes sense in context.

After Top-K selection, FieldSAE's decoder also lets nearby retained entries work together when rebuilding the activation. A retained feature at one position can therefore contribute to reconstruction in a local neighborhood rather than only at its exact position.

### What FieldSAE learned and what Top-K selected

```text
The FieldSAE encoder learned:
  • which feature channels to create
  • how each local neighborhood should influence each feature-location value

Top-K selected for each image:
  • which feature-location pairs were retained under the fixed budget
```

FieldSAE did not learn fixed image coordinates. It learned an input-dependent scoring function over all feature-location pairs, then Top-K made the final exact-budget selection.

### Historical result and current interpretation

Historical FieldSAE did outperform the **simple, context-free** VectorSAE at aggressive budgets in early CNN checkpoints. That result does not establish that FieldSAE beats ContextVectorSAE, because ContextVectorSAE was added in the later corrected evaluation program and the full cross-family matrix remains incomplete.

---

## The practical difference in one table

| Question | ContextVectorSAE | Historical FieldSAE |
|---|---|---|
| Does one location see its neighbors? | Yes: local convolutions and pooled attention | Yes: local spatial residual blocks |
| Encoder | Contextual blocks, pooled attention, then score/value heads | 1×1 maps plus three local spatial residual blocks |
| Decoder | Direct 1×1 vector decoder plus contextual refiner | 1×1 maps plus three local spatial residual blocks |
| Candidate selection | Exact Top-M over learned selection scores | Historical global Top-K over feature activations |
| Are locations fixed after training? | No; Top-M varies by image | No; Top-K varies by image |
| Role | Later clean-redo candidate | Historical CNN winner used for the transition chain |

## Important presentation caveat

The historical FieldSAE experiment reported retained fraction relative to an expanded `K = 8C` latent map. Retaining 5% of that map meant retaining about 40% as many coefficient values as the original activation map before counting location indices. The later ContextVectorSAE redo instead fixes an exact active count relative to the original activation map and records index and value bits. Do not compare their reported retained percentages as though they were the same budget.

## Suggested appendix-slide narration

“The original Vector baseline processed locations independently. This ContextVectorSAE keeps vector-style feature atoms but adds cross-location context before selection and during reconstruction. FieldSAE also uses local spatial context throughout. The clean comparison between these context-aware families is still incomplete, so the historical Field-versus-simple-Vector result should not be presented as a verdict on ContextVectorSAE.”
