#!/bin/bash
#SBATCH -J gromacs_viscosity
#SBATCH -p hfacnormal01
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32

set -e

# Environment setup
module purge
module load compiler/gcc/9.3.0 mpi/intelmpi/2021.3.0
source /public/home/ac9h310k3a/software/gromacs-2023.3-clean-mpi/bin/GMXRC
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# File paths
BASE_DIR="/public/home/ac9h310k3a/Gromacs/PMMA_250424/chain_length/CL/1"
GRO_FILE="${BASE_DIR}/eq.gro"
TOP_FILE="${BASE_DIR}/gro/system.top"

# Run GROMACS workflow
gmx_mpi grompp -f vis.mdp -c ${GRO_FILE} -p ${TOP_FILE} -o shear.tpr -maxwarn 10
srun gmx_mpi mdrun -deffnm shear
echo "31" | gmx_mpi energy -f shear.edr -b 20000 -o viscosity.xvg

echo "Viscosity calculation completed successfully!"