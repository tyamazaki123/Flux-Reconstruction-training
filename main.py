#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File Name: main.py
Author: Tomoki Yamazaki
Updated by: Sep, 2025
"""
import argparse
import configparser
import os
import sys

from solver import SodShockTubeSolver, InitialCondition#, BoundaryCondition, RHS, LHS

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sod Shock Tube Simulator (FR/CPR, Gauss points)")
    parser.add_argument("config", help="Path to configuration .ini file (e.g., input.ini)")
    args = parser.parse_args()
    config_path = args.config

    # Read configuration file
    if not os.path.isfile(config_path):
        print(f"Error: Config file '{config_path}' not found.", file=sys.stderr)
        sys.exit(1)

    config_parser = configparser.ConfigParser()
    config_parser.read(config_path)

    # Use section "Simulation" if it exists; else DEFAULT
    if "Simulation" in config_parser:
        config = config_parser["Simulation"]
    else:
        config = config_parser.defaults()

    # Output directory
    results_dir = config.get("results_dir", fallback="results")
    os.makedirs(results_dir, exist_ok=True)
    log_file_path = os.path.join(results_dir, "simulation.log")

    with open(log_file_path, "w", encoding="utf-8") as log_file:
        # Initialize solver and auxiliaries
        solver   = SodShockTubeSolver(config)
        init_cond = InitialCondition()

        # Write config to log
        log_file.write("Configuration Parameters:\n")
        for key in config:
            log_file.write(f"  {key} = {config.get(key)}\n")
        log_file.write("\n")

        # Run
        solver.run(init_cond,log_file)

    print(f"Simulation finished.\n- Log:     {log_file_path}\n- Results: {results_dir}")
