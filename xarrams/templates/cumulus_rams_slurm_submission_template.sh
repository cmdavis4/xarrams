#!/bin/bash
#SBATCH -A {{project_code}}
#SBATCH -p {{queue}}
#SBATCH -N {{n_nodes}}
#SBATCH -J {{run_name}}
#SBATCH -t {{wall_time}}
#SBATCH --mem={{memory}}
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user={{user_email}}
#SBATCH --exclusive
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

module load intel hdf5

# Get current datetime for output files
CURRENT_DT=$(date +%Y%m%d%H%M%S)

# Run and tee output to both datetime-stamped and current files
srun -n {{n_cores}} {{rams_executable_path}} -f {{ramsin_path}} \
  > >(tee -a {{stdout_dir}}/{{run_name}}_${CURRENT_DT}.stdout > {{stdout_dir}}/current.stdout) \
  2> >(tee -a {{stdout_dir}}/{{run_name}}_${CURRENT_DT}.stderr > {{stdout_dir}}/current.stderr)
