// Optimized ARMv7-A NEON signed int8 dot-product leaf for Rockchip RV1103 (Cortex-A7)
// AAPCS compliant: r0/r1 pointers (16-byte aligned), r2 length (multiple of 16); signed int32 result in r0.
#if (defined(__arm__) || defined(__thumb__)) && (defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(RUVI_KERNEL_RV1103_NEON))
    .syntax unified
    .arch armv7-a
    .fpu neon-vfpv4
    .text
    .align 4
    .global ruvi_kernel_neon_dot_raw
    .type ruvi_kernel_neon_dot_raw, %function
ruvi_kernel_neon_dot_raw:
    // Zero 32-bit vector accumulators: q3 (d6, d7) and q8 (d16, d17)
    // Both registers are caller-saved in AAPCS (d0-d7, d16-d31).
    vmov.i32 q3, #0
    vmov.i32 q8, #0

    // blocks = length / 16 (r2 >> 4)
    lsrs r3, r2, #4
    beq .Ldone_zero

    // n4 = blocks / 4 (r12 = counter for 64-byte chunks)
    lsrs r12, r3, #2
    beq .Lrem_blocks

.Lloop64:
    // Block 0 (16 bytes): load into q0, q1; compute into q2; accumulate into q3
    vld1.8 {d0, d1}, [r0]!
    vld1.8 {d2, d3}, [r1]!
    vmull.s8 q2, d0, d2
    vmlal.s8 q2, d1, d3
    vpadal.s16 q3, q2

    // Block 1 (16 bytes): load into q9, q10; compute into q11; accumulate into q8
    vld1.8 {d18, d19}, [r0]!
    vld1.8 {d20, d21}, [r1]!
    vmull.s8 q11, d18, d20
    vmlal.s8 q11, d19, d21
    vpadal.s16 q8, q11

    // Block 2 (16 bytes): load into q0, q1; compute into q2; accumulate into q3
    vld1.8 {d0, d1}, [r0]!
    vld1.8 {d2, d3}, [r1]!
    vmull.s8 q2, d0, d2
    vmlal.s8 q2, d1, d3
    vpadal.s16 q3, q2

    // Block 3 (16 bytes): load into q9, q10; compute into q11; accumulate into q8
    vld1.8 {d18, d19}, [r0]!
    vld1.8 {d20, d21}, [r1]!
    vmull.s8 q11, d18, d20
    vmlal.s8 q11, d19, d21
    vpadal.s16 q8, q11

    subs r12, r12, #1
    bne .Lloop64

.Lrem_blocks:
    // Remainder 16-byte blocks (blocks & 3)
    ands r3, r3, #3
    beq .Lreduce

.Lloop16:
    vld1.8 {d0, d1}, [r0]!
    vld1.8 {d2, d3}, [r1]!
    vmull.s8 q2, d0, d2
    vmlal.s8 q2, d1, d3
    vpadal.s16 q3, q2
    subs r3, r3, #1
    bne .Lloop16

.Lreduce:
    // Combine dual accumulators
    vadd.s32 q3, q3, q8

    // Horizontal reduction of 4x int32 lanes to scalar in r0
    vpadd.s32 d6, d6, d7
    vpadd.s32 d6, d6, d6
    vmov.32 r0, d6[0]
    bx lr

.Ldone_zero:
    movs r0, #0
    bx lr

    .size ruvi_kernel_neon_dot_raw, .-ruvi_kernel_neon_dot_raw
#endif

#if defined(__linux__) && defined(__ELF__)
    .section .note.GNU-stack,"",%progbits
#endif
