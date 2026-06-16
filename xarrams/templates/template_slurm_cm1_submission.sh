#!/bin/bash
#SBATCH -A {{account}}
#SBATCH -p {{queue}}
#SBATCH -N {{n_nodes}}
#SBATCH -J {{run_name}}
#SBATCH -t {{wall_time}}
#SBATCH --mem=0
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=cmdavis4@colostate.edu
#SBATCH --exclusive
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

{% for m in modules -%}
module load {{m}}
{% endfor %}
{%- if prologue %}
{{prologue}}
{% endif %}
# Copy the CM1 executable into our output directory; assume namelist is already there
cp -f {{cm1_dir}}/run/cm1.exe {{run_dir}}/output
cd {{run_dir}}/output

# Get current datetime for output files
CURRENT_DT=$(date +%Y%m%d%H%M%S)
STAMPED_STDOUT={{run_dir}}/stdout/{{run_name}}_${CURRENT_DT}.stdout
STAMPED_STDERR={{run_dir}}/stdout/{{run_name}}_${CURRENT_DT}.stderr
CURRENT_STDOUT={{run_dir}}/stdout/current.stdout
CURRENT_STDERR={{run_dir}}/stdout/current.stderr
# Also copy the namelist as both curent and dated
cp -r {{run_dir}}/output/namelist.input {{stdout_dir}}/current.namelist
cp -r {{run_dir}}/output/namelist.input {{stdout_dir}}/{{run_name}}_${CURRENT_DT}.namelist

# Provenance preamble: executable checksum, command, full namelist contents.
# Captured at job start so the checksum matches the binary that actually runs.
HASHES="##############################"
{
  echo "$HASHES"
  echo "CM1 CHECKSUM: $(md5sum ./cm1.exe | awk '{print $1}')"
  echo "$HASHES"
  echo
  echo "$HASHES"
  echo "         BEGIN COMMAND"
  echo "$HASHES"
  echo "{{mpi_launcher}} {{n_cores}} ./cm1.exe"
  echo "$HASHES"
  echo "          END COMMAND"
  echo "$HASHES"
  echo
  echo "$HASHES"
  echo "         BEGIN NAMELIST"
  echo "$HASHES"
  cat ./namelist.input
  echo
  echo "$HASHES"
  echo "          END NAMELIST"
  echo "$HASHES"
  echo
  echo "$HASHES"
  echo "         BEGIN CM1 OUTPUT"
  echo "$HASHES"
} | tee "$STAMPED_STDOUT" > "$CURRENT_STDOUT"

# Set MPI to use romio, for if we're doing parallel parcel writes
export OMPI_MCA_io=romio321
# disable HCOLL (fixes the MPI_Comm_dup crash)
export OMPI_MCA_coll_hcoll_enable=0
# Run and tee output to both datetime-stamped and current files
{{mpi_launcher}} {{n_cores}} ./cm1.exe \
  > >(tee -a "$STAMPED_STDOUT" >> "$CURRENT_STDOUT") \
  2> >(tee -a "$STAMPED_STDERR" > "$CURRENT_STDERR")
