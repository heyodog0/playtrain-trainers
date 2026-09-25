#!/bin/bash
# U16a: for 4 init seeds (own params + own batch each), official fp32 and bf16 and port fp32 and bf16,
# all on H100 with identical inputs per seed.
E=/n/netscratch/gershman_lab/Everyone/truong/dv3-equiv
cd $E
for s in 1 2 3 4; do
  a=$(sbatch --parsable --export=ALL,WARMUP=0,SEED=$s,DTYPE=float32,OUT_OVERRIDE=sweep/s$s/off_fp32 equiv_official_gpu.sbatch)
  b=$(sbatch --parsable --dependency=afterok:$a --export=ALL,WARMUP=0,SEED=$s,DTYPE=bfloat16,INPUTS=$E/sweep/s$s/off_fp32,OUT_OVERRIDE=sweep/s$s/off_bf16 equiv_official_gpu.sbatch)
  c=$(sbatch --parsable --dependency=afterok:$a --export=ALL,WARMUP=0,DTYPE=float32,OFF_DIR=$E/sweep/s$s/off_fp32,PORT_OUT=$E/sweep/s$s/port_fp32,PORT_SRC=port_src_bf16 equiv_port_gpu.sbatch)
  d=$(sbatch --parsable --dependency=afterok:$a --export=ALL,WARMUP=0,DTYPE=bfloat16,OFF_DIR=$E/sweep/s$s/off_fp32,PORT_OUT=$E/sweep/s$s/port_bf16,PORT_SRC=port_src_bf16 equiv_port_gpu.sbatch)
  echo "seed $s: $a $b $c $d"
done
