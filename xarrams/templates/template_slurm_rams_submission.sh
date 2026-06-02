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
# Run from the RAMSIN's directory so that relative paths in the RAMSIN
# (input/, output/, etc., as written by build_rams_directory_structure)
# resolve correctly. Mirrors what xarrams.run_rams does for python-side runs.
cd "$(dirname "{{ramsin_path}}")"

# Get current datetime for output files
CURRENT_DT=$(date +%Y%m%d%H%M%S)
STAMPED_STDOUT={{stdout_dir}}/${CURRENT_DT}.stdout
STAMPED_STDERR={{stdout_dir}}/${CURRENT_DT}.stderr
CURRENT_STDOUT={{stdout_dir}}/current.stdout
CURRENT_STDERR={{stdout_dir}}/current.stderr
# Also copy the ramsin as both current and dated
cp -r {{ramsin_path}} {{stdout_dir}}/current.ramsin
cp -r {{ramsin_path}} {{stdout_dir}}/${CURRENT_DT}.ramsin

# Provenance preamble: executable checksum, command, full RAMSIN contents.
# Captured at job start so the checksum matches the binary that actually runs.
HASHES="##############################"
{
  echo "$HASHES"
  echo "RAMS CHECKSUM: $(md5sum {{rams_executable_path}} | awk '{print $1}')"
  echo "$HASHES"
  echo
  echo "$HASHES"
  echo "         BEGIN COMMAND"
  echo "$HASHES"
  echo "{{mpi_launcher}} {{n_cores}} {{rams_executable_path}} -f {{ramsin_path}}"
  echo "$HASHES"
  echo "          END COMMAND"
  echo "$HASHES"
  echo
  echo "$HASHES"
  echo "         BEGIN RAMSIN"
  echo "$HASHES"
  cat {{ramsin_path}}
  echo
  echo "$HASHES"
  echo "          END RAMSIN"
  echo "$HASHES"
  echo
  echo "$HASHES"
  echo "         BEGIN RAMS OUTPUT"
  echo "$HASHES"
} | tee "$STAMPED_STDOUT" > "$CURRENT_STDOUT"

# Run and tee output to both datetime-stamped and current files
{{mpi_launcher}} {{n_cores}} {{rams_executable_path}} -f {{ramsin_path}} \
  > >(tee -a "$STAMPED_STDOUT" >> "$CURRENT_STDOUT") \
  2> >(tee -a "$STAMPED_STDERR" > "$CURRENT_STDERR")
