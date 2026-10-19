#!/bin/bash
# Usage: run_all_hermes.sh HERMES_SRC OUTROOT  -- the Hermes-3 matrix, one job at a time.
# Sources [m^-3 s^-1] = DRBX steady particle source / 10 m at n_up = 2, 4, 7e19 (ny = 800).
# ny = 100, 200 start from the deck's initial profiles (300 outputs); ny = 400, 800, 1600 start
# from the next-coarser steady state (restart_from.py) and run 260 outputs.
here=$(cd "$(dirname "$0")" && pwd)
for ny in 100 200 400 800 1600; do
  for src in 2.2311e21 6.8845e21 1.04033e22; do
    if [ "$ny" -le 200 ]; then
      DT=1000000 "$here/run_hermes.sh" "$1" "$2" $src $ny 100000 300 || echo "FAILED $src $ny"
    else
      FROM="$2/h_${src}_$((ny / 2))_ei100000" DT=1000000 "$here/run_hermes.sh" "$1" "$2" $src $ny 100000 260 || echo "FAILED $src $ny"
    fi
    python -I "$here/extract_hermes.py" "$2/h_${src}_${ny}_ei100000" "$2/h_${src}_${ny}_ei100000.json"
  done
done
