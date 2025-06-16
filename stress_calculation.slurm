#!/bin/bash
#SBATCH -J gromacs_final_working
#SBATCH -p hfacnormal01
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32

set -e
# =====================================================================
#             1. 最终的、简洁的、正确的环境
# =====================================================================
echo "Setting up the NEW, clean, and stable environment..."
module purge

# 只加载编译时使用的那几个模块！这保证了环境的一致性。
module load compiler/gcc/9.3.0
module load mpi/intelmpi/2021.3.0
# 激活您刚刚编译好的、全新的、干净的GROMACS
source /public/home/ac9h310k3a/software/gromacs-2023.3-clean-mpi/bin/GMXRC
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# =====================================================================
#             2. GROMACS 工作流程 (现在将正确运行)
# =====================================================================
TOP_FILE="/public/home/ac9h310k3a/Gromacs/PMMA_250424/chain_length/CL/1/gro/system.top"

gmx_mpi grompp -f ss2.mdp -c eq.gro -p ${TOP_FILE} -o ss.tpr -maxwarn 10
srun gmx_mpi mdrun -v -deffnm ss 
echo -e "Pres-XX\nBox-X" | gmx_mpi energy -f ss.edr -o pressure_xx.xvg

echo "Job Finished Successfully!"