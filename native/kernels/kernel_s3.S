// Independently authored from ESP32-S3 TRM instruction definitions.
// Private leaf ABI: a2/a3 buffers, a4 byte length; signed result in a2.
#if defined(__XTENSA__) && (defined(CONFIG_IDF_TARGET_ESP32S3) || defined(RUVI_KERNEL_ESP32_S3))
    .text
    .align 4
    .global ruvi_kernel_s3_dot_raw
    .type ruvi_kernel_s3_dot_raw, @function
ruvi_kernel_s3_dot_raw:
#ifndef __XTENSA_CALL0_ABI__
    entry a1, 32
#endif
    // Zero 40-bit accumulator via user registers
    movi a14, 0
    wur.accx_0 a14
    wur.accx_1 a14

    // blocks = length / 16 (a4 >> 4)
    srli a5, a4, 4
    // n4 = blocks / 4 (a5 >> 2)
    srli a6, a5, 2
    // nr = blocks & 3
    extui a7, a5, 0, 2

    beqz a6, .Lruvi_rem
.Lruvi_loop4:
    ee.vld.128.ip q0, a2, 16
    ee.vld.128.ip q1, a3, 16
    ee.vld.128.ip q2, a2, 16
    ee.vld.128.ip q3, a3, 16
    ee.vld.128.ip q4, a2, 16
    ee.vld.128.ip q5, a3, 16
    ee.vld.128.ip q6, a2, 16
    ee.vld.128.ip q7, a3, 16
    ee.vmulas.s8.accx q0, q1
    ee.vmulas.s8.accx q2, q3
    ee.vmulas.s8.accx q4, q5
    ee.vmulas.s8.accx q6, q7
    addi a6, a6, -1
    bnez a6, .Lruvi_loop4

.Lruvi_rem:
    beqz a7, .Lruvi_done
.Lruvi_loop1:
    ee.vld.128.ip q0, a2, 16
    ee.vld.128.ip q1, a3, 16
    ee.vmulas.s8.accx q0, q1
    addi a7, a7, -1
    bnez a7, .Lruvi_loop1

.Lruvi_done:
    rur.accx_0 a2
#ifdef __XTENSA_CALL0_ABI__
    ret
#else
    retw
#endif
    .size ruvi_kernel_s3_dot_raw, .-ruvi_kernel_s3_dot_raw
#endif
