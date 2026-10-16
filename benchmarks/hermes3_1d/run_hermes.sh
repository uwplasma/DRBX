#!/bin/bash
# Usage: [DT=timestep] [FROM=coarser_run_dir] run_hermes.sh HERMES_SRC OUTROOT SRC NY [EIMULT] [NOUT]
# Runs one Hermes-3 case (1 rank, nice 10) from deck.BOUT.inp into OUTROOT/h_<SRC>_<NY>_ei<EIMULT>/.
# FROM: start from the steady state of a coarser run, interpolated in y (restart_from.py).
set -euo pipefail
H=$1; OUT=$2; SRC=$3; NY=$4; EI=${5:-100000}; NOUT=${6:-300}; DT=${DT:-1000000}
here=$(cd "$(dirname "$0")" && pwd)
FAC=$(python3 -c "from math import pi,sqrt
ny=$NY; a=0.9; yx=pi*(2-0.1-sqrt(1.9**2-4*a/3))/a
print(repr(10/sum((30/ny)*(1+a*(1-y/pi)) for y in (2*pi*(j+0.5)/ny for j in range(ny)) if y<yx)))")
d=$OUT/h_${SRC}_${NY}_ei${EI}; mkdir -p "$d/json_database"
cp "$here/json_database/"*.json "$H"/json_database/AMJUEL_H.4_2.1.8.json "$H"/json_database/AMJUEL_H.10_2.1.8.json "$d/json_database/"
sed -e "s/@NY@/$NY/; s/@SRC@/$SRC/g; s/@FAC@/$FAC/g; s/@EIMULT@/$EI/; s/@NOUT@/$NOUT/; s/@DT@/$DT/; s|@JSONDIR@|$d/json_database|" \
    "$here/deck.BOUT.inp" > "$d/BOUT.inp"
rm -f "$d"/BOUT.dmp.*.nc "$d"/BOUT.restart.*.nc
extra=""
if [ -n "${FROM:-}" ]; then
  (cd "$d" && nice -n 10 mpirun -np 1 "$H/build/hermes-3" -d . nout=0 > run0.log 2>&1)
  python -I "$here/restart_from.py" "$FROM" "$d"; rm -f "$d"/BOUT.dmp.*.nc
  extra="restart=true append=false"
fi
echo "start $(date -Is) load $(cut -d' ' -f1-3 /proc/loadavg)" > "$d/load.txt"
cd "$d" && nice -n 10 mpirun -np 1 "$H/build/hermes-3" -d . $extra > run.log 2>&1
echo "end $(date -Is) load $(cut -d' ' -f1-3 /proc/loadavg)" >> "$d/load.txt"
