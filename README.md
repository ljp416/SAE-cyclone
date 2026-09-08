# SAE-cyclone

Companion code for *Sparse Autoencoders Reveal How Cyclone Attributes Are Organized in an AI Weather Model*.

Explore cyclone-related sparse autoencoder (SAE) features in GraphCast using the Dorian-Juliette case at **1 September 2019, 18 UTC**. Start with [analysis.ipynb](analysis.ipynb); example activations and pretrained SAE checkpoints are available in the [release](https://github.com/ljp416/SAE-cyclone/releases/tag/v0.1.0). No training is required.

## Quick start

The [environment](environment.yml) uses Python 3.11 and targets Linux with CUDA 12. Saved-activation analysis runs on CPU; running GraphCast requires an NVIDIA GPU.

```bash
conda env create -f environment.yml
conda activate sae-cyclone
jupyter lab analysis.ipynb
```

Download `example.tar.gz` from the release and extract it in the project folder (`tar -xzf example.tar.gz`). Set `RUN_GRAPHCAST = False` in the first code cell, then run the notebook cells in order to load one processor layer, encode it with an SAE, rank features and draw one map.

## Notebook options

| Setting | Options |
| --- | --- |
| `LAYER` | GraphCast processor layer, `0`-`15` |
| `SEED` | `50108` for all layers; `50109` also available for layer `8` |
| `RANK` | Select one of the top four features: `1`, `2`, `3`, or `4` |
| `LON_RANGE`, `LAT_RANGE` | Optional map bounds within the supplied coastline region |
| `RUN_GRAPHCAST` | Use saved activations (`False`) or run GraphCast (`True`) |

Ranking uses F1 overlap with the simultaneous cyclone target. This single-time example differs from the paper's full-period feature selection; map bounds affect only the displayed region.

The notebook saves `feature.png` and `feature.npz`.

## Optional: run GraphCast

Also download and extract `graphcast.tar.gz` (`tar -xzf graphcast.tar.gz`). Set `RUN_GRAPHCAST = True` to regenerate all 16 processor-layer activations and a six-hour forecast. Configure the source, checkpoint, ERA5 input and statistics paths in the first code cell, using the [compatible GraphCast source](https://github.com/google-deepmind/weathernext/tree/6819a0f19796ca9c34a079855339b37134b8d930).

## Data and references

The release archives extract inputs into `data/` and pretrained weights into `checkpoints/`.

- [GraphCast / WeatherNext](https://github.com/google-deepmind/weathernext/tree/6819a0f19796ca9c34a079855339b37134b8d930)
- [WeatherBench 2 data guide](https://weatherbench2.readthedocs.io/en/latest/data-guide.html)
- [IBTrACS](https://www.ncei.noaa.gov/products/international-best-track-archive)

Please cite the corresponding model and dataset publications when using these resources.

## License

- Project code: [Apache-2.0](LICENSE).
- SAE checkpoints and original project-generated example data: [CC BY 4.0](LICENSE-DATA).
- Third-party materials retain their original licenses and terms. The GraphCast weights used here are [CC BY-NC-SA 4.0](https://github.com/google-deepmind/weathernext/tree/6819a0f19796ca9c34a079855339b37134b8d930#license-and-disclaimers); ERA5, IBTrACS and coastlines retain their source terms.
