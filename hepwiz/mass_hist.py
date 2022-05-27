#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pickle
import sys
from math import ceil
from typing import Tuple

import click


@click.command()
@click.argument("lhe-path", type=click.Path(exists=True))
@click.argument("pythia-settings", type=click.Path(exists=True))
@click.argument("output", type=click.Path(exists=False, writable=True))
@click.argument("pdgs", type=click.INT, nargs=-1)
@click.option("--pt-cut/--no-pt-cut", default=False, show_default=True)
@click.option("--eta-cut/--no-eta-cut", default=False, show_default=True)
@click.option("--use-flow/--no-use-flow", default=True, show_default=True)
@click.option("--num-bins", type=click.IntRange(min=10), default=200,
              show_default=True)
@click.option("--width", type=click.FloatRange(min=1.0), default=40.0,
              show_default=True)
def main(lhe_path: str, pythia_settings: str, output: str,
         pdgs: Tuple[int, ...], pt_cut: bool, eta_cut: bool, use_flow: bool,
         num_bins: int, width: float) -> None:
    """Showers LHE hard events in parallel to produce a pickled
    dictionary of histograms.

    \b
    Parameters
    ----------
    LHE_PATH : path
        Location of the LHE file to shower.
        May be uncompressed or gzipped.
        May also be stored online as a URL.
    PYTHIA_SETTINGS : path
        File containing settings for Pythia, typically with `.cmnd`
        extension.
    OUTPUT : path
        Location in which to store the pickled histograms.
    PDGS : sequence of ints
        PDG codes for each hard parton for which to produce a histogram.

    \b
    Notes
    -----
    Parallelism is implemented using MPI. To make use of multiple cores,
    `mpiexec` must be used when executing this program.
    eg. `mpiexec -np 40 mass-hist [OPTIONS] [ARGUMENTS] -- [PDGS]`
    The PDGS argument was separated from the others with a double hyphen
    to prevent the CLI interpreting negative PDG codes as options.
    """
    from mpi4py import MPI
    import numpy as np
    from showerpipe.generator import PythiaGenerator
    from showerpipe.lhe import split, count_events
    import graphicle as gcl
    from colliderscope.data import Histogram
    from tqdm import tqdm


    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    
    rng = np.random.default_rng(seed=rank)
    seed = rng.integers(low=0, high=1_000_000_000)
    
    # intialise the histograms
    
    parent_pdg = gcl.PdgArray(np.array(pdgs, dtype='<i'))
    histograms = {
        pdg.name[0]: Histogram(num_bins, pdg.mass, width) for pdg in parent_pdg
    }
    
    total_num_events = count_events(lhe_path)
    stride = ceil(total_num_events / size)
    
    if rank == 0:
        lhe_splits = split(lhe_path, stride)  # 10,000 events
        data = next(lhe_splits)
        for i in range(1, size):
            comm.send(next(lhe_splits), dest=i, tag=10+i)
    else:
        data = comm.recv(source=0, tag=10+rank)
    
    gen = PythiaGenerator(pythia_settings, data, rng_seed=seed)
    gen_ = gen
    if rank == 0:
        gen_ = tqdm(gen)
    for event in gen_:
        graph = gcl.Graphicle.from_numpy(
            pdg=event.pdg,
            pmu=event.pmu,
            status=event.status,
            final=event.final,
            edges=event.edges
        )
        hard_masks = gcl.select.hard_descendants(
                graph, parent_pdg.data, sign_sensitive=True)
        cuts = gcl.MaskGroup()
        cuts["final"] = graph.final.data
        if eta_cut:
            cuts["eta"] = np.abs(graph.pmu.eta) < 2.5
        if pt_cut:
            cuts["pt"] = graph.pmu.pt > 0.5

        for parton_name, hist in histograms.items():
            parton_mask = gcl.MaskGroup()
            parton_mask["cuts"] = cuts
            parton_mask["parton"] = hard_masks[parton_name]
            try:
                flow_weight = None
                if use_flow:
                    flow_array = gcl.calculate.hard_trace(
                            graph, parton_mask, graph.pmu.data["e"])
                    flow_weight = flow_array[parton_name]
                hist.update(gcl.calculate.jet_mass(
                    graph.pmu[parton_mask], flow_weight))
            except NotImplementedError as ne_error:
                print(ne_error)
            except ValueError as val_error:
                print(val_error)
    
    for hist in histograms.values():
        all_counts = np.empty_like(hist.counts)
        comm.Reduce(hist.counts, all_counts, MPI.SUM)
        hist.counts = all_counts
    
    if rank == 0:
        with open(output, "wb") as f:
            pickle.dump(histograms, f)

if __name__ == "__main__":
    sys.exit(main())
