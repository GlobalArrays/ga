	.file	"fops.2d.f"
// PGFTN Rel R5.0 -opt 2 -recursive
	.text
	.globl	_accumulatef_v_
	.align	32
_accumulatef_v_:
.a1 = 0
.f1 = 4256
	addu -(.a1+.f1), sp, sp
	st.l fp,(.f1-16)(sp)
	addu (.f1-16), sp, fp
	st.l r1, 4(fp)
	fst.d f2, -4224(fp)
	st.l r4, -4216(fp)
	st.l r5, -4212(fp)
	st.l r6, -4208(fp)
	st.l r7, -4204(fp)
	st.l r8, -4200(fp)
	st.l r9, -4196(fp)
	st.l r10, -4192(fp)
	st.l r11, -4188(fp)
	st.l r12, -4184(fp)
	st.l r13, -4180(fp)
	st.l r14, -4176(fp)
	st.l r15, -4172(fp)
                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
	
	
	
	
	
	
	st.l r22, -4(fp)
	st.l r16, -8(fp)
	st.l r17, -12(fp)
	st.l r18, -16(fp)
	st.l r19, -20(fp)
	st.l r20, -24(fp)
	st.l r21, -28(fp)
// lineno: 0
// lineno: 0
	ld.l -24(fp), r28
	ld.l -4(fp), r30
	ld.l r0(r28), r29
	ld.l r0(r30), r16
	adds 1, r29, r28
	st.l r28, -36(fp)
	adds 1, r16, r30
	st.l r30, -44(fp)
	st.l r29, -32(fp)
	st.l r16, -40(fp)
	mov r0, r11
	ld.l -16(fp), r17
	adds 1, r0, r18
	ld.l r0(r17), r9
	
	subs 0, r9, r0
	
	bnc .B1_334
	ld.l -12(fp), r17
	ld.l -20(fp), r19
	ld.l r0(r17), r10
	shl 3, r28, r20
	subs r10, r18, r0
	shl r0, r18, r17
	bc ..L0
	or r10, r0, r17
..L0:
	adds -1, r17, r12
	subs r19, r20, r19
	adds 1, r29, r21
	shl 3, r21, r20
	adds r20, r19, r19
	addu -8, r19, r21
	st.l r21, -48(fp)
	adds r19, r0, r13
	ld.l -8(fp), r22
	ld.l -28(fp), r23
	fld.d r0(r22), f2
	shl 3, r30, r24
	subs r23, r24, r23
	adds 1, r16, r25
	shl 3, r25, r24
	adds r24, r23, r23
	adds r23, r0, r14
	addu -8, r23, r15
	shl 3, r16, r26
	st.l r26, -52(fp)
	shl 3, r29, r27
	st.l r27, -56(fp)
// lineno: 9
.B1_459:
	subs 0, r10, r0
.DB.B1_459459:	
	
	bnc .B1_340
	adds 10, r0, r28
	subs r10, r28, r0
	
	bc .B1_360
	mov r10, r4
	mov r13, r6
	mov r13, r7
	mov r14, r8
// lineno: 0
.B1_399:
	adds 256, r0, r28
.DB.B1_399399:	
	subs r4, r28, r0
	shl r0, r4, r29
	bc ..L1
	or r28, r0, r29
..L1:
	mov r8, r16
	adds -4160, fp, r17
	adds 1, r0, r19
	mov r29, r18
	fiadd.dd f2, f0, f8
	call ___strin8m
	mov r29, r5
	mov r7, r16
	adds -2112, fp, r17
	adds 1, r0, r19
	call ___streamixp8
	mov r5, r18
	mov r6, r16
	adds -4160, fp, r17
	adds -2112, fp, r18
	adds 1, r0, r20
	call ___add8s
	mov r5, r19
	addu 2048, r6, r6
	addu 2048, r7, r7
	addu 2048, r8, r8
	adds -256, r4, r4
	subs 0, r4, r0
	
	bc.t .DB.B1_399399
	adds 256, r0, r28
// lineno: 0
	br .B1_359
	nop
// lineno: 0
.B1_360:
	mov r15, r7
.DB.B1_360360:	
	ld.l -48(fp), r18
	adds -1, r0, r16
	mov r12, r17
	bla r16, r17,.B1_417
	pfmul.dd f0, f0, f0
// lineno: 0
.B1_417:
	fld.d 8(r7)++, f16
.DB.B1_417417:	
	fld.d 8(r18)++, f20
	
	fmul.dd f2, f16, f18
	
	
	
	fadd.dd f18, f20, f16
	
	
	bla r16, r17, .B1_417
	fst.d f16, r0(r18)
// lineno: 0
.B1_359:
// lineno: 0
.B1_340:
	ld.l -56(fp), r28
.DB.B1_340340:	
	ld.l -48(fp), r29
	ld.l -52(fp), r30
	adds r28, r29, r29
	st.l r29, -48(fp)
	adds r30, r14, r14
	adds r30, r15, r15
	adds r28, r13, r13
	adds 1, r11, r11
	adds -1, r9, r9
	subs 0, r9, r0
	
	bc.t .DB.B1_459459
	subs 0, r10, r0
// lineno: 0
.B1_334:
// lineno: 13
	
	
	
	
	fld.d -4224(fp), f2
	ld.l -4216(fp), r4
	ld.l -4212(fp), r5
	ld.l -4208(fp), r6
	ld.l -4204(fp), r7
	ld.l -4200(fp), r8
	ld.l -4196(fp), r9
	ld.l -4192(fp), r10
	ld.l -4188(fp), r11
	ld.l -4184(fp), r12
	ld.l -4180(fp), r13
	ld.l -4176(fp), r14
	ld.l -4172(fp), r15
	adds .a1+16, fp, r31
	ld.l 4(fp), r1
	ld.l 0(fp), fp
	bri r1
	mov r31, sp
// STATIC VARIABLES
// COMMON BLOCKS
	.text
	.globl	_dcopy2d_v_
	.align	32
_dcopy2d_v_:
.a2 = 0
.f2 = 4224
	addu -(.a2+.f2), sp, sp
	st.l fp,(.f2-16)(sp)
	addu (.f2-16), sp, fp
	st.l r1, 4(fp)
	st.l r4, -4192(fp)
	st.l r5, -4188(fp)
	st.l r6, -4184(fp)
	st.l r7, -4180(fp)
	st.l r8, -4176(fp)
	st.l r9, -4172(fp)
	st.l r10, -4168(fp)
	st.l r11, -4164(fp)
	st.l r12, -4160(fp)
	st.l r13, -4156(fp)
	st.l r14, -4152(fp)
	st.l r15, -4148(fp)
                                                     
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
	
	
	
	
	
	
	st.l r16, -4(fp)
	st.l r17, -8(fp)
	st.l r18, -12(fp)
	st.l r19, -16(fp)
	st.l r20, -20(fp)
	st.l r21, -24(fp)
// lineno: 0
// lineno: 0
	ld.l -16(fp), r28
	ld.l -24(fp), r30
	ld.l r0(r28), r29
	ld.l r0(r30), r16
	adds 1, r29, r28
	st.l r28, -32(fp)
	adds 1, r16, r30
	st.l r30, -40(fp)
	st.l r29, -28(fp)
	st.l r16, -36(fp)
	mov r0, r10
	ld.l -8(fp), r17
	adds 1, r0, r18
	ld.l r0(r17), r8
	
	subs 0, r8, r0
	
	bnc .B2_331
	ld.l -4(fp), r17
	ld.l -20(fp), r19
	ld.l r0(r17), r9
	shl 3, r30, r20
	subs r9, r18, r0
	shl r0, r18, r17
	bc ..L2
	or r9, r0, r17
..L2:
	adds -1, r17, r11
	subs r19, r20, r19
	adds 1, r16, r21
	shl 3, r21, r20
	adds r20, r19, r19
	adds r19, r0, r12
	ld.l -12(fp), r22
	shl 3, r28, r23
	subs r22, r23, r22
	adds 1, r29, r24
	shl 3, r24, r23
	adds r23, r22, r22
	adds r22, r0, r13
	addu -8, r22, r14
	addu -8, r19, r15
	shl 3, r29, r25
	st.l r25, -44(fp)
	shl 3, r16, r26
	st.l r26, -48(fp)
// lineno: 21
.B2_449:
	subs 0, r9, r0
.DB.B2_449449:	
	
	bnc .B2_337
	adds 10, r0, r28
	subs r9, r28, r0
	
	bc .B2_357
	mov r9, r4
	mov r12, r6
	mov r13, r7
// lineno: 0
.B2_388:
	adds 512, r0, r28
.DB.B2_388388:	
	subs r4, r28, r0
	shl r0, r4, r29
	bc ..L3
	or r28, r0, r29
..L3:
	mov r7, r16
	adds -4144, fp, r17
	adds 1, r0, r19
	mov r29, r18
	call ___streamixp8
	mov r29, r5
	adds -4144, fp, r16
	adds 1, r0, r19
	mov r6, r17
	call ___streamoxp8
	mov r5, r18
	addu 4096, r6, r6
	addu 4096, r7, r7
	adds -512, r4, r4
	subs 0, r4, r0
	
	bc.t .DB.B2_388388
	adds 512, r0, r28
// lineno: 0
	br .B2_356
	nop
// lineno: 0
.B2_357:
	mov r14, r6
.DB.B2_357357:	
	mov r15, r18
	adds -1, r0, r16
	mov r11, r17
	bla r16, r17,.B2_408
	pfmul.dd f0, f0, f0
// lineno: 0
.B2_408:
	fld.d 8(r6)++, f16
.DB.B2_408408:	
	
	
	bla r16, r17, .B2_408
	fst.d f16, 8(r18)++
// lineno: 0
.B2_356:
// lineno: 0
.B2_337:
	ld.l -44(fp), r28
.DB.B2_337337:	
	ld.l -48(fp), r29
	adds r28, r13, r13
	adds r28, r14, r14
	adds r29, r12, r12
	adds r29, r15, r15
	adds 1, r10, r10
	adds -1, r8, r8
	subs 0, r8, r0
	
	bc.t .DB.B2_449449
	subs 0, r9, r0
// lineno: 0
.B2_331:
// lineno: 25
	
	
	
	
	ld.l -4192(fp), r4
	ld.l -4188(fp), r5
	ld.l -4184(fp), r6
	ld.l -4180(fp), r7
	ld.l -4176(fp), r8
	ld.l -4172(fp), r9
	ld.l -4168(fp), r10
	ld.l -4164(fp), r11
	ld.l -4160(fp), r12
	ld.l -4156(fp), r13
	ld.l -4152(fp), r14
	ld.l -4148(fp), r15
	adds .a2+16, fp, r31
	ld.l 4(fp), r1
	ld.l 0(fp), fp
	bri r1
	mov r31, sp
// STATIC VARIABLES
// COMMON BLOCKS
	.text
	.globl	_icopy2d_v_
	.align	32
_icopy2d_v_:
.a3 = 0
.f3 = 4240
	addu -(.a3+.f3), sp, sp
	st.l fp,(.f3-16)(sp)
	addu (.f3-16), sp, fp
	st.l r1, 4(fp)
	st.l r4, -4208(fp)
	st.l r5, -4204(fp)
	st.l r6, -4200(fp)
	st.l r7, -4196(fp)
	st.l r8, -4192(fp)
	st.l r9, -4188(fp)
	st.l r10, -4184(fp)
	st.l r11, -4180(fp)
	st.l r12, -4176(fp)
	st.l r13, -4172(fp)
	st.l r14, -4168(fp)
	st.l r15, -4164(fp)
                                                     
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
	
	
	
	
	
	
	st.l r16, -4(fp)
	st.l r17, -8(fp)
	st.l r18, -12(fp)
	st.l r19, -16(fp)
	st.l r20, -20(fp)
	st.l r21, -24(fp)
// lineno: 0
// lineno: 0
	ld.l -16(fp), r28
	ld.l -24(fp), r30
	ld.l r0(r28), r29
	ld.l r0(r30), r16
	adds 1, r29, r28
	st.l r28, -32(fp)
	adds 1, r16, r30
	st.l r30, -40(fp)
	st.l r29, -28(fp)
	st.l r16, -36(fp)
	mov r0, r10
	ld.l -8(fp), r17
	adds 1, r0, r18
	ld.l r0(r17), r8
	
	subs 0, r8, r0
	
	bnc .B3_331
	ld.l -4(fp), r17
	ld.l -20(fp), r19
	ld.l r0(r17), r9
	shl 2, r30, r20
	subs r9, r18, r0
	shl r0, r18, r17
	bc ..L4
	or r9, r0, r17
..L4:
	adds -1, r17, r11
	subs r19, r20, r19
	adds 1, r16, r21
	shl 2, r21, r20
	adds r20, r19, r19
	adds r19, r0, r12
	adds 1, r29, r22
	shl 2, r22, r15
	ld.l -12(fp), r23
	shl 2, r28, r24
	subs r23, r24, r23
	adds r15, r23, r28
	adds r28, r0, r13
	addu -4, r19, r14
	st.l r23, -52(fp)
	shl 2, r16, r25
	st.l r25, -44(fp)
	shl 2, r29, r26
	st.l r26, -48(fp)
// lineno: 33
.B3_446:
	subs 0, r9, r0
.DB.B3_446446:	
	
	bnc .B3_337
	adds 10, r0, r28
	subs r9, r28, r0
	
	bc .B3_357
	mov r9, r4
	mov r12, r6
	mov r13, r7
// lineno: 0
.B3_388:
	adds 1024, r0, r28
.DB.B3_388388:	
	subs r4, r28, r0
	shl r0, r4, r29
	bc ..L5
	or r28, r0, r29
..L5:
	mov r7, r16
	adds -4160, fp, r17
	adds 1, r0, r19
	mov r29, r18
	call ___streamixp4
	mov r29, r5
	adds -4160, fp, r16
	adds 1, r0, r19
	mov r6, r17
	call ___streamoxp4
	mov r5, r18
	addu 4096, r6, r6
	addu 4096, r7, r7
	adds -1024, r4, r4
	subs 0, r4, r0
	
	bc.t .DB.B3_388388
	adds 1024, r0, r28
// lineno: 0
	br .B3_356
	nop
// lineno: 0
.B3_357:
	mov r14, r6
.DB.B3_357357:	
	mov r15, r18
	ld.l -52(fp), r19
	adds -1, r0, r16
	mov r11, r17
	bla r16, r17,.B3_406
	pfmul.dd f0, f0, f0
// lineno: 0
.B3_406:
	ld.l r19(r18), r28
.DB.B3_406406:	
	addu 4, r18, r18
	st.l r28, 4(r6)
	bla r16, r17, .B3_406
	addu 4, r6, r6
// lineno: 0
.B3_356:
// lineno: 0
.B3_337:
	ld.l -44(fp), r28
.DB.B3_337337:	
	ld.l -48(fp), r29
	adds r28, r12, r12
	adds r28, r14, r14
	adds r29, r15, r15
	adds r29, r13, r13
	adds 1, r10, r10
	adds -1, r8, r8
	subs 0, r8, r0
	
	bc.t .DB.B3_446446
	subs 0, r9, r0
// lineno: 0
.B3_331:
// lineno: 37
	
	
	
	
	ld.l -4208(fp), r4
	ld.l -4204(fp), r5
	ld.l -4200(fp), r6
	ld.l -4196(fp), r7
	ld.l -4192(fp), r8
	ld.l -4188(fp), r9
	ld.l -4184(fp), r10
	ld.l -4180(fp), r11
	ld.l -4176(fp), r12
	ld.l -4172(fp), r13
	ld.l -4168(fp), r14
	ld.l -4164(fp), r15
	adds .a3+16, fp, r31
	ld.l 4(fp), r1
	ld.l 0(fp), fp
	bri r1
	mov r31, sp
// STATIC VARIABLES
// COMMON BLOCKS
	.extern	___streamixp4
	.extern	___streamoxp4
	.extern	___streamoxp8
	.extern	___strin8m
	.extern	___streamixp8
	.extern	___add8s
