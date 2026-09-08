"""One native 0.25-degree GraphCast step and all requested processor states.

Matches ADEMPTOR issue496's center-time convention, BF16 inference, input and
residual normalization, and zero-based mesh post-residual capture. Haiku
interception returns states from the same forward pass without editing GraphCast.
"""
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np
import xarray as xr


def run_graphcast(graphcast_root, checkpoint, era5, stats_dir, layers, output_dir):
    """Use ERA5 [t-6h,t,t+6h]; save corpus at t and prediction valid at t+6h.

    ``era5`` is one full-global, three-frame NetCDF in GraphCast's batch/time/
    datetime format. ``graphcast_root`` may name the checkout or its parent.
    Output files are layer00.npy ... layer15.npy and prediction.nc.
    """
    root = Path(graphcast_root).resolve()
    if not (root / "graphcast" / "graphcast.py").is_file():
        root = root / "graphcast"
    if not (root / "graphcast" / "graphcast.py").is_file():
        raise FileNotFoundError("GraphCast checkout must contain graphcast/graphcast.py")
    sys.path.insert(0, str(root))
    import haiku as hk
    import jax
    from graphcast import casting, checkpoint as checkpoint_io, data_utils
    from graphcast import graphcast, normalization

    if Path(graphcast.__file__).resolve().parent.parent != root:
        raise RuntimeError("Restart the kernel: a different GraphCast checkout was already imported")
    jax.config.update("jax_default_matmul_precision", "highest")
    layers = tuple(int(layer) for layer in layers)
    if not layers or len(set(layers)) != len(layers) or any(i not in range(16) for i in layers):
        raise ValueError("Select distinct zero-based processor layers from 0 through 15")
    with Path(checkpoint).open("rb") as stream:
        ckpt = checkpoint_io.load(stream, graphcast.CheckPoint)
    if (ckpt.model_config.resolution != 0.25 or ckpt.model_config.gnn_msg_steps != 16
            or ckpt.model_config.latent_size != 512 or len(ckpt.task_config.pressure_levels) != 37):
        raise ValueError("Use the 0.25-degree, 37-level, 16-layer GraphCast checkpoint")
    dataset = xr.load_dataset(era5)
    times = np.asarray(dataset.datetime).reshape(-1).astype("datetime64[ns]")
    if (dataset.sizes.get("batch") != 1 or len(times) != 3
            or not np.all(np.diff(times) == np.timedelta64(6, "h"))):
        raise ValueError("ERA5 must contain one batch of three consecutive six-hour frames")
    if dataset.lat[0] > dataset.lat[-1]:
        dataset = dataset.reindex(lat=dataset.lat[::-1])
    if (dataset.sizes.get("lat") != 721 or dataset.sizes.get("lon") != 1440
            or not np.allclose(dataset.lat, np.linspace(-90, 90, 721))
            or not np.allclose(dataset.lon, np.arange(1440) * 0.25)):
        raise ValueError("Inference requires the complete native 0.25-degree global grid")
    inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
        dataset, target_lead_times=slice("6h", "6h"), **asdict(ckpt.task_config))
    target_attrs = {name: dict(dataset[name].attrs) for name in targets.data_vars}
    del dataset
    stats_dir = Path(stats_dir)
    stats = {name: xr.load_dataset(stats_dir / f"{name}.nc")
             for name in ("diffs_stddev_by_level", "mean_by_level", "stddev_by_level")}

    @hk.transform_with_state
    def forward(inputs, targets_template, forcings):
        captured = []

        def capture(next_method, args, kwargs, context):
            result = next_method(*args, **kwargs)
            if context.module.name == "mesh_gnn" and context.method_name == "_process_step":
                captured.append(result.nodes["mesh_nodes"].features)
            return result

        predictor = graphcast.GraphCast(ckpt.model_config, ckpt.task_config)
        predictor = casting.Bfloat16Cast(predictor)
        predictor = normalization.InputsAndResiduals(predictor, **stats)
        # Exactly one step needs no autoregressive feedback or scan wrapper.
        with hk.intercept_methods(capture):
            prediction = predictor(inputs, targets_template, forcings)
        if len(captured) != 16:
            raise RuntimeError(f"Expected 16 mesh processor states, captured {len(captured)}")
        return prediction, tuple(captured[layer] for layer in layers)

    (prediction, states), _ = jax.jit(forward.apply)(
        ckpt.params, {}, jax.random.PRNGKey(0), inputs, targets * np.nan, forcings)
    prediction, states = jax.device_get((prediction, states))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_files = []
    for layer, state in zip(layers, states):
        state = np.asarray(state)
        if state.shape != (40962, 1, 512) or str(state.dtype) != "bfloat16":
            raise ValueError(f"Layer {layer} does not match the native BF16 mesh corpus")
        if not np.isfinite(state.astype(np.float32)).all():
            raise ValueError(f"Layer {layer} contains non-finite activations")
        path = output_dir / f"layer{layer:02d}.npy"
        np.save(path, state.view("V2"), allow_pickle=False)
        corpus_files.append(str(path))
    prediction = prediction.copy(data={name: np.asarray(value.data)
                                       for name, value in prediction.data_vars.items()})
    if not all(np.isfinite(value).all() for value in prediction.data_vars.values()):
        raise ValueError("GraphCast prediction contains non-finite values")
    prediction = prediction.assign_coords(datetime=(("batch", "time"), times[None, -1:]))
    prediction.attrs.update(forecast_reference_time=str(times[1]), valid_time=str(times[2]))
    for name in prediction.data_vars:
        prediction[name].attrs.update(target_attrs[name])
    prediction_file = output_dir / "prediction.nc"
    prediction.to_netcdf(prediction_file, encoding={
        name: {"zlib": True, "complevel": 1} for name in prediction.data_vars})
    return dict(center_time=str(times[1]), valid_time=str(times[2]),
                corpus_files=corpus_files, prediction_file=str(prediction_file))
