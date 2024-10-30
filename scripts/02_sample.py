#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import argparse


# -----------------------------------------------------------------------------
# Script parameters
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description='Create training dataset from Dedalus simulation data',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--dataDir", default="simuData", help="directory containing simulation data")
parser.add_argument(
    "--inSize", default=1, help="input size", type=int)
parser.add_argument(
    "--outStep", default=1, help="output step", type=int)
parser.add_argument(
    "--inStep", default=5, help="input step", type=int)
parser.add_argument(
    "--outType", default="solution", help="output type in the dataset",
    choices=["solution", "update"])
parser.add_argument(
    "--outScaling", default=1, type=float,
    help="scaling factor for the output (ignored with outType=solution !)")
parser.add_argument(
    "--dataFile", default="dataset.h5", help="name of the dataset HDF5 file")
parser.add_argument(
    "--config", default=None, help="config file, overwriting all parameters specified in it")
parser.add_argument(
    "--dryRun", default=None, action='store_true',
    help="don't extract the data, just print the infos of the expected dataset")
args = parser.parse_args()

# To avoid import when using help ...
sys.path.insert(2, os.getcwd())
from fnop.utils import readConfig
from fnop.data.data_preprocessing import createDataset

if args.config is not None:
    config = readConfig(args.config)
    assert "sample" in config, "config file needs a sample section"
    args.__dict__.update(**config.data)
    args.__dict__.update(**config.sample)
    if "simu" in config and "dataDir" in config.simu:
        args.dataDir = config.simu.dataDir
    if "data" in config:
        for key in ["outType", "outScaling", "dataFile"]:
            if key in config.data: args.__dict__[key] = config.data[key]
kwargs = {**args.__dict__}
kwargs.pop("config")

# -----------------------------------------------------------------------------
# Script execution
# -----------------------------------------------------------------------------
createDataset(**kwargs)
