#!/bin/bash
#SBATCH -A {{account}}
#SBATCH -p {{queue}}
#SBATCH -N {{n_nodes}}
{% if ntasks_per_node -%}
#SBATCH --ntasks-per-node={{ntasks_per_node}}
{% endif -%}
#SBATCH -J {{run_name}}
#SBATCH -t {{wall_time}}
#SBATCH --mem=0
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=cmdavis4@colostate.edu
#SBATCH --exclusive
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --export=NONE

module purge
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
  echo "{{mpi_launcher}} {{n_cores}} --export=ALL {{rams_executable_path}} -f {{ramsin_path}}"
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

# HDF5 filter plugins (ZFP); must match the HDF5 the executable links
export HDF5_PLUGIN_PATH={{hdf5_plugin_path}}
export OMPI_MCA_io=romio321
{% if romio_hints %}
# ROMIO MPI-IO hints for the parallel HDF5 analysis writes.
# romio_cb_write=enable forces collective buffering: without it ROMIO lets
# every rank write its own small compressed chunks, and the ranks contend for
# shared GPFS blocks, which made the analysis writes several times slower.
ROMIO_HINTS_PATH={{stdout_dir}}/romio_hints.txt
cat > "$ROMIO_HINTS_PATH" <<'END_ROMIO_HINTS'
{% for hint_name, hint_value in romio_hints.items() -%}
{{hint_name}} {{hint_value}}
{% endfor -%}
END_ROMIO_HINTS
export ROMIO_HINTS="$ROMIO_HINTS_PATH"
{% endif %}

# Run and tee output to both datetime-stamped and current files
{{mpi_launcher}} {{n_cores}} --export=ALL {{rams_executable_path}} -f {{ramsin_path}} \
  > >(tee -a "$STAMPED_STDOUT" >> "$CURRENT_STDOUT") \
  2> >(tee -a "$STAMPED_STDERR" > "$CURRENT_STDERR")
