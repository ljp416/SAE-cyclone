<h1 align="center">SAE-cyclone</h1>

<p align="center">
  <strong>Tropical Cyclone Attributes in Sparse Autoencoder Features of an AI Weather Model</strong>
</p>

<p align="center">Jiapei Liu · Ruidong Li · Ting Sun · Guangheng Ni</p>

<p align="center">
  <a href="https://arxiv.org/abs/2610.28904"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2610.28904-b31b1b"></a>
  <a href="https://doi.org/10.5281/zenodo.23492703"><img alt="Zenodo DOI" src="https://img.shields.io/badge/Zenodo-10.5281%2Fzenodo.23492703-blue"></a>
  <a href="LICENSE"><img alt="Code: Apache 2.0" src="https://img.shields.io/badge/code-Apache%202.0-green"></a>
  <a href="LICENSE-DATA"><img alt="Data: CC BY 4.0" src="https://img.shields.io/badge/data-CC%20BY%204.0-green"></a>
</p>

Explore cyclone-related sparse autoencoder (SAE) features in GraphCast with [`analysis.ipynb`](analysis.ipynb). The notebook uses the Dorian–Juliette case at **1 September 2019, 18 UTC**, with saved activations and pretrained SAEs. No training is required.

## Quick start

The supplied [environment](environment.yml) targets **Python 3.11, Linux, and CUDA 12**. Analysis of saved activations runs on CPU; optional GraphCast inference requires an NVIDIA GPU.

1. Clone the repository:

   ```bash
   git clone https://github.com/ljp416/SAE-cyclone.git
   cd SAE-cyclone
   ```

2. Download [`example.tar.gz`](https://github.com/ljp416/SAE-cyclone/releases/download/v0.1.0/example.tar.gz) into the project directory, then install the environment and open the notebook:

   ```bash
   tar -xzf example.tar.gz
   conda env create -f environment.yml
   conda activate sae-cyclone
   jupyter lab analysis.ipynb
   ```

3. Set `RUN_GRAPHCAST = False` in the first code cell, then run the cells in order. The notebook ranks SAE features, draws a map, and saves `feature.png` and `feature.npz`.

| Setting | Options |
| --- | --- |
| `LAYER` | Processor layer `0`–`15` |
| `SEED` | `50108` for all layers; `50109` also available at layer `8` |
| `RANK` | Feature rank `1`–`4` |
| `LON_RANGE`, `LAT_RANGE` | Map bounds within the supplied coastline region |

Features are ranked by F1 overlap with the simultaneous cyclone target. This single-time ranking differs from the paper's full-period selection; map bounds affect display only.

<details>
<summary><strong>Optional: run GraphCast</strong></summary>

Download [`graphcast.tar.gz`](https://github.com/ljp416/SAE-cyclone/releases/download/v0.1.0/graphcast.tar.gz) into the project directory and extract it:

```bash
tar -xzf graphcast.tar.gz
```

Set `RUN_GRAPHCAST = True` and configure the GraphCast source, checkpoint, ERA5 input, and statistics paths in the first code cell. Use the [compatible GraphCast source](https://github.com/google-deepmind/weathernext/tree/6819a0f19796ca9c34a079855339b37134b8d930). This regenerates all 16 processor-layer activations and a six-hour forecast.

</details>

## Data and checkpoints

| Resource | Contents |
| --- | --- |
| [Example release · v0.1.0](https://github.com/ljp416/SAE-cyclone/releases/tag/v0.1.0) | Single-time notebook inputs and 17 SAE checkpoints; optional GraphCast inputs and weights |
| [Paper archive · v1.0.0](https://doi.org/10.5281/zenodo.23492703) | Analysis and training code, processed inputs for Figures 1–4, S1–S9 and Tables S1–S3, and 80 SAE checkpoints across five seeds and 16 layers |

To reproduce the paper's figures and tables on CPU, follow the README in the Zenodo archive. Full upstream weather databases and the complete hidden-state training corpus are outside that archive.

## Citation

Please cite the paper when using this work, and the Zenodo archive when using its data, checkpoints, or analysis code.

```bibtex
@misc{liu2026cyclone,
  title         = {Tropical Cyclone Attributes in Sparse Autoencoder Features of an AI Weather Model},
  author        = {Liu, Jiapei and Li, Ruidong and Sun, Ting and Ni, Guangheng},
  year          = {2026},
  eprint        = {2610.28904},
  archivePrefix = {arXiv},
  primaryClass  = {physics.ao-ph},
  doi           = {10.48550/arXiv.2610.28904},
  url           = {https://arxiv.org/abs/2610.28904}
}

@misc{liu2026cyclonearchive,
  title     = {Data, checkpoints, and analysis code for Tropical Cyclone Attributes in Sparse Autoencoder Features of an AI Weather Model},
  author    = {Liu, Jiapei and Li, Ruidong and Sun, Ting and Ni, Guangheng},
  year      = {2026},
  publisher = {Zenodo},
  version   = {1.0.0},
  doi       = {10.5281/zenodo.23492703},
  url       = {https://doi.org/10.5281/zenodo.23492703}
}
```

## Acknowledgements and license

This work uses [GraphCast / WeatherNext](https://github.com/google-deepmind/weathernext/tree/6819a0f19796ca9c34a079855339b37134b8d930), [ERA5 via WeatherBench 2](https://weatherbench2.readthedocs.io/en/latest/data-guide.html), and [IBTrACS](https://www.ncei.noaa.gov/products/international-best-track-archive). Please also cite the corresponding model and dataset publications.

- **Code:** [Apache 2.0](LICENSE).
- **SAE checkpoints and original project-generated data:** [CC BY 4.0](LICENSE-DATA).
- **Third-party materials:** retain their source terms. The GraphCast weights used here are [CC BY-NC-SA 4.0](https://github.com/google-deepmind/weathernext/tree/6819a0f19796ca9c34a079855339b37134b8d930#license-and-disclaimers); ERA5, IBTrACS, and coastlines retain their applicable upstream terms.
