#!/bin/bash
# Lab 2 Data Collection Script
# Run this from inside the container at /mca-compiler-labs/workspace/lab_2

OUT=/share/lab2_output
mkdir -p $OUT

INCLUDE=-I/usr/riscv32-unknown-elf/riscv32-unknown-elf/include/
TARGET="-target riscv32 -mabi=ilp32"
CLANG_FLAGS="$TARGET -S -emit-llvm -O2 -mllvm -disable-llvm-optzns $INCLUDE"

echo "========== STARTING LAB 2 DATA COLLECTION ==========" | tee $OUT/log.txt

# ─────────────────────────────────────────────
# HELPER: rebuild merged.ll with current matrix.h
# ─────────────────────────────────────────────
build_merged() {
    clang main.c $CLANG_FLAGS -fno-addrsig -o main.ll
    clang matrix.c $CLANG_FLAGS -fno-addrsig -o matrix.ll
    llvm-link main.ll matrix.ll -S -o merged.ll
}

# ─────────────────────────────────────────────
# PART 2 — set MDIM=10
# ─────────────────────────────────────────────
echo "" | tee -a $OUT/log.txt
echo "===== PART 2: MDIM=10 =====" | tee -a $OUT/log.txt
sed -i 's/#define MDIM .*/#define MDIM 10/' matrix.h
build_merged

# Save baseline merged.ll
cp merged.ll $OUT/merged_baseline.ll

# --- Q2.1: internalize ---
echo "" | tee -a $OUT/log.txt
echo "--- Q2.1: internalize ---" | tee -a $OUT/log.txt
opt merged.ll -passes=internalize -S -o $OUT/opt_internalize.ll
diff merged.ll $OUT/opt_internalize.ll > $OUT/diff_internalize.txt
echo "diff saved to diff_internalize.txt" | tee -a $OUT/log.txt

opt merged.ll -passes=internalize -internalize-public-api-list=main -S -o $OUT/opt_internalize_fixed.ll
diff merged.ll $OUT/opt_internalize_fixed.ll > $OUT/diff_internalize_fixed.txt
echo "fixed internalize diff saved" | tee -a $OUT/log.txt

# --- Q2.2: inline ---
echo "" | tee -a $OUT/log.txt
echo "--- Q2.2: inline ---" | tee -a $OUT/log.txt
opt merged.ll -passes=inline -S -o $OUT/opt_inline.ll
diff merged.ll $OUT/opt_inline.ll > $OUT/diff_inline.txt
echo "inline diff saved" | tee -a $OUT/log.txt

# --- Q2.3: loop-unroll (MDIM=10) ---
echo "" | tee -a $OUT/log.txt
echo "--- Q2.3: loop-unroll (MDIM=10) ---" | tee -a $OUT/log.txt
opt merged.ll -passes=loop-unroll -unroll-count=10 -S -o $OUT/opt_unroll.ll
diff merged.ll $OUT/opt_unroll.ll > $OUT/diff_unroll.txt

# Save accumulate() sections
grep -A 200 "define.*accumulate" merged.ll        | head -200 > $OUT/accumulate_baseline.txt
grep -A 200 "define.*accumulate" $OUT/opt_unroll.ll | head -200 > $OUT/accumulate_unrolled.txt
echo "accumulate sections saved" | tee -a $OUT/log.txt

# --- Q2.4: mem2reg ---
echo "" | tee -a $OUT/log.txt
echo "--- Q2.4: mem2reg ---" | tee -a $OUT/log.txt
opt merged.ll -passes=mem2reg -S -o $OUT/opt_mem2reg.ll
diff merged.ll $OUT/opt_mem2reg.ll > $OUT/diff_mem2reg.txt

# Save first block of main()
grep -A 60 "define.*main" merged.ll          | head -60 > $OUT/main_baseline.txt
grep -A 60 "define.*main" $OUT/opt_mem2reg.ll | head -60 > $OUT/main_mem2reg.txt
echo "main() blocks saved" | tee -a $OUT/log.txt

# --- Q2.5: instcombine ---
echo "" | tee -a $OUT/log.txt
echo "--- Q2.5: instcombine ---" | tee -a $OUT/log.txt
opt merged.ll -passes=instcombine -S -o $OUT/opt_instcombine.ll
diff merged.ll $OUT/opt_instcombine.ll > $OUT/diff_instcombine.txt

# Save accumulate() icmp lines
grep -n "icmp\|cmp\|br" merged.ll              > $OUT/cmp_baseline.txt
grep -n "icmp\|cmp\|br" $OUT/opt_instcombine.ll > $OUT/cmp_instcombine.txt

# Save full accumulate() for context
grep -A 200 "define.*accumulate" merged.ll             | head -200 > $OUT/accumulate_instcombine_baseline.txt
grep -A 200 "define.*accumulate" $OUT/opt_instcombine.ll | head -200 > $OUT/accumulate_instcombine_opt.txt
echo "instcombine sections saved" | tee -a $OUT/log.txt

# ─────────────────────────────────────────────
# PART 3 — set MDIM=100
# ─────────────────────────────────────────────
echo "" | tee -a $OUT/log.txt
echo "===== PART 3: MDIM=100 =====" | tee -a $OUT/log.txt
sed -i 's/#define MDIM .*/#define MDIM 100/' matrix.h
build_merged

run_set() {
    local name=$1
    local passes=$2
    echo "" | tee -a $OUT/log.txt
    echo "--- Set $name: $passes ---" | tee -a $OUT/log.txt
    opt merged.ll -passes="$passes" -S -o opt_${name}.ll
    llc opt_${name}.ll -o opt_${name}.s
    /opt/riscv32-wo-double/bin/riscv32-unknown-elf-gcc opt_${name}.s -o bin_${name}
    echo "Timing for Set $name ($passes):" | tee -a $OUT/log.txt
    { time qemu-riscv32 ./bin_${name} ; } 2>&1 | tee -a $OUT/log.txt $OUT/timing_${name}.txt
}

run_set "set1" "inline,loop-unroll"
run_set "set2" "inline,mem2reg"
run_set "set3" "mem2reg,loop-unroll"

echo "" | tee -a $OUT/log.txt
echo "========== DONE — all output in $OUT ==========" | tee -a $OUT/log.txt
