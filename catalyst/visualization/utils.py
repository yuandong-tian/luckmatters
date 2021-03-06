import pickle
import torch
import os
import sys
import glob
import yaml
from copy import deepcopy
import logging
log = logging.getLogger(__file__)
from tqdm import tqdm

def find_params(data, cond):
    for d in data:
        found = True
        for k, v in cond.items():
            if d["args"][k] != v:
                found = False
        if found:
            return d
    return None

def find_all_params(data, cond):
    all_d = []
    for d in data:
        found = True
        for k, v in cond.items():
            if d["args"][k] != v:
                found = False
        if found:
            all_d.append(d)
    return all_d

def load_configs(folder):
    print(f"Load hydra configs from folder")
    config_filename = os.path.join(folder, "config.yaml")
    if not os.path.exists(config_filename):
       config_filename = (os.path.join(folder, ".hydra/config.yaml"))
    else:
        return None
    print(f"Config file: {config_filename}")
    args = yaml.load(open(config_filename, "r"))
    return dict(args=args,path=folder)


def load_stats(folder, stats_filename="stats.pickle"):
    log.info(f"Load stats from {folder}")
    filename = os.path.join(folder, stats_filename)
    if os.path.exists(filename):
        args = load_configs(folder)
        log.info(f"Stats file: {stats_filename}")
        stats = torch.load(filename)
        return dict(args=args,stats=stats,path=folder)
    else:
        log.warn(f"The {filename} doesn't exist")
        return None


def load_data(root, stats_filename="stats.pickle", use_tqdm=False):
    data = []
    total = 0
    folders = sorted(glob.glob(os.path.join(root, "*")))
    last_prefix = None

    if use_tqdm: folders = tqdm(folders)
    for folder in folders:
        path, folder_name = os.path.split(folder)
        items = folder_name.split("_")
        if len(items) > 1:
            prefix, job_id = items
            if prefix == last_prefix:
                continue
        else:
            job_id = items[0]
            prefix = job_id

        stats = load_stats(folder, stats_filename=stats_filename)
        if stats is not None:
            log.info(f"{len(data)}: {folder}")
            data.append(stats)
            last_prefix = prefix

    return data

def convert_stats_to_summary(folder):
    d = load_stats(folder)
    if d is None:
        log.warn(f"Cannot find stats in {folder}, skip.")
        return

    # Only save the last stat. 
    new_stats = [ d["stats"][0][0], d["stats"][0][-1] ]
    torch.save(new_stats, os.path.join(folder, "summary.pth"))
    
