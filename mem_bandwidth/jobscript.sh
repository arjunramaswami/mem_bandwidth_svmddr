#!/bin/bash
#SBATCH -A pc2-mitarbeiter
#SBATCH -J "svmddr"
#SBATCH -p fpga
#SBATCH --mem=90000MB 
#SBATCH --time=23:59:00
#SBATCH --constraint=emul

module load intelFPGA_pro/20.3.0 intel_pac/19.2.0_usm devel/CMake
module load toolchain/gompi/2020a

make build_hw_no_interleaving
