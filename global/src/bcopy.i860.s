//
//	Credit to Andy Pfiffer for this.
//
//	void bcopy(src, dst, cnt)
//	char	*src, *dst;
//	int	cnt;
//
//	A bcopy() that has better performance characteristics
//	for transfers that are not favorably aligned.
//
//	WARNING: this routine does not handle overlapping copies
//		 that should be performed in backwards order.
//
	.file	"bcopy.s"
	.text
	.align 32
_memcpy_i::
	mov	r17, r31
	mov	r16, r17
	mov	r31, r16
_bcopy_i::
	//
	//	if < 48 bytes to copy, handle the small stuff
	//	in a non-pipelined loop.
	//
	adds	-48, r18, r0
	bc	.small

	//
	//	>= 48 bytes to copy.  Check for 16-byte alignment.
	//
	or	r16, r17, r31
	and	0x000f, r31, r0
	bnc	.mis8

	//
	//	pipelined copy of 16-byte aligned data.
	//
.copy16:
	addu	-16, r16, r16		// autopreincrement
	addu	-16, r17, r17		// autopreincrement
	addu	-48, r18, r18		// 3 in load pipe
	pfld.q	16(r16)++, f8
	addu	-1,  r0, r28		// increment
	pfld.q	16(r16)++, f8
	shr	 6, r18, r29		// number of loops == (cnt / 64)
	pfld.q	16(r16)++, f8
	mov	r29, r30		// save the loop count...
	bla	r28, r29, .primed16
	 nop
.primed16:
	bla	r28, r29, .pump16
	 nop
.drain16:
	pfld.d	0(r16), f16		// drain the 3 outstanding
	pfld.d	0(r16), f20
	pfld.d	0(r16), f24
	fst.q	f16, 16(r17)++
	fst.q	f20, 16(r17)++
	fst.q	f24, 16(r17)++
	addu	16, r16, r16		// undo the autopreincrement
	addu	16, r17, r17		// undo the autopreincrement
.tail16:
	bte	0, r18, .done
	adds	-16, r18, r0
	bc	.tail8
	fld.q	0(r16), f16
	addu	16, r16, r16
	addu	16, r17, r17
	adds	-16, r18, r18
	br	.tail16
	 fst.q	f16, -16(r17)
.pump16:
	pfld.q	16(r16)++, f16
	pfld.q	16(r16)++, f20
	pfld.q	16(r16)++, f24
	pfld.q	16(r16)++, f28
	fst.q	f16, 16(r17)++
	fst.q	f20, 16(r17)++
	fst.q	f24, 16(r17)++
	bla	r28, r29, .pump16
	 fst.q	f28, 16(r17)++
	shl	6, r30, r30		// subtract the number of bytes...
	br	.drain16
	 subs	r18, r30, r18		// ...moved in the loop.

	//
	//	check for 8-byte alignment
	//
.mis8:
	or	r16, r17, r31
	and	0x0007, r31, r0
	bnc	.mis4

	//
	//	there are four cases for (src | dst) & 7 == 0:
	//
	//	src=0 dst=0	-- not applicable (handled above)
	//	src=0 dst=8	-- 8-byte but can't become 16-byte aligned
	//	src=8 dst=0	-- 8-byte but can't become 16-byte aligned
	//	src=8 dst=8	-- can become 16-byte aligned
	//
	xor	r16, r17, r31
	and	0x0008, r31, r0
	bc	.one8			// might become 16-byte aligned

	//
	//	pipelined copy of 8-byte aligned data.
	//
.copy8:
	addu	-8, r16, r16		// autopreincrement
	addu	-8, r17, r17		// autopreincrement
	addu	-24, r18, r18		// 3 in load pipe
	pfld.d	8(r16)++, f8
	addu	-1,  r0, r28		// increment
	pfld.d	8(r16)++, f8
	shr	 6, r18, r29		// number of loops == (cnt / 64)
	pfld.d	8(r16)++, f8
	mov	r29, r30		// save the loop count...
	bla	r28, r29, .primed8
	 nop
.primed8:
	bla	r28, r29, .pump8
	 nop
.drain8:
	pfld.d	0(r16), f16		// drain the 3 outstanding
	pfld.d	0(r16), f18
	pfld.d	0(r16), f20
	fst.d	f16, 8(r17)++
	fst.d	f18, 8(r17)++
	fst.d	f20, 8(r17)++
	addu	8, r16, r16		// undo the autopreincrement
	addu	8, r17, r17		// undo the autopreincrement
.tail8:
	bte	0, r18, .done
	adds	-8, r18, r0
	bc	.tail4
	fld.d	 0(r16), f16
	addu	 8, r16, r16
	addu	 8, r17, r17
	adds	-8, r18, r18
	br	.tail8
	 fst.d	f16, -8(r17)
.pump8:
	pfld.d	8(r16)++, f16
	pfld.d	8(r16)++, f18
	pfld.d	8(r16)++, f20
	pfld.d	8(r16)++, f22
	pfld.d	8(r16)++, f24
	pfld.d	8(r16)++, f26
	pfld.d	8(r16)++, f28
	pfld.d	8(r16)++, f30
	fst.d	f16, 8(r17)++
	fst.d	f18, 8(r17)++
	fst.d	f20, 8(r17)++
	fst.d	f22, 8(r17)++
	fst.d	f24, 8(r17)++
	fst.d	f26, 8(r17)++
	fst.d	f28, 8(r17)++
	bla	r28, r29, .pump8
	 fst.d	f30, 8(r17)++
	shl	6, r30, r30		// subtract the number of bytes...
	br	.drain8
	 subs	r18, r30, r18		// ...moved in the loop.

.one8:
	//
	//	move 1 8-byte word (makes src and dst 16-byte aligned).
	//
	fld.d	0(r16), f16
	addu	8, r16, r16
	addu	8, r17, r17
	addu	-8, r18, r18
	br	_bcopy_i
	 fst.d	f16, -8(r17)

.mis4:
	or	r16, r17, r31
	and	0x0003, r31, r0
	bnc	.mis2

	//
	//	there are four cases for (src | dst) & 3 == 0:
	//
	//	src=0 dst=0	-- not applicable (handled above)
	//	src=0 dst=4	-- 4-byte but can't become 8-byte aligned
	//	src=4 dst=0	-- 4-byte but can't become 8-byte aligned
	//	src=4 dst=4	-- can become 8-, 16-byte aligned
	//
	xor	r16, r17, r31
	and	0x0004, r31, r0
	bc	.one4			// might become 16-byte aligned

	//
	//	pipelined copy of 4-byte aligned data
	//
.copy4:
	addu	-4, r16, r16		// autopreincrement
	addu	-4, r17, r17		// autopreincrement
	addu	-12, r18, r18		// 3 in load pipe
	pfld.l	4(r16)++, f8
	addu	-1,  r0, r28		// increment
	pfld.l	4(r16)++, f8
	shr	 6, r18, r29		// number of loops == (cnt / 64)
	pfld.l	4(r16)++, f8
	mov	r29, r30		// save the loop count...
	bla	r28, r29, .primed4
	 nop
.primed4:
	bla	r28, r29, .pump4
	 nop
.drain4:
	pfld.l	0(r16), f16		// drain the 3 outstanding
	pfld.l	0(r16), f17
	pfld.l	0(r16), f18
	fst.l	f16, 4(r17)++
	fst.l	f17, 4(r17)++
	fst.l	f18, 4(r17)++
	addu	4, r16, r16		// undo the autopreincrement
	addu	4, r17, r17		// undo the autopreincrement
.tail4:
	bte	0, r18, .done
	adds	-4, r18, r0
	bc	.tail2
	fld.l	 0(r16), f16
	addu	 4, r16, r16
	addu	 4, r17, r17
	adds	-4, r18, r18
	br	.tail4
	 fst.l	f16, -4(r17)
.pump4:
	pfld.l	4(r16)++, f16
	pfld.l	4(r16)++, f17
	pfld.l	4(r16)++, f18
	pfld.l	4(r16)++, f19
	pfld.l	4(r16)++, f20
	pfld.l	4(r16)++, f21
	pfld.l	4(r16)++, f22
	pfld.l	4(r16)++, f23
	pfld.l	4(r16)++, f24
	pfld.l	4(r16)++, f25
	pfld.l	4(r16)++, f26
	pfld.l	4(r16)++, f27
	pfld.l	4(r16)++, f28
	pfld.l	4(r16)++, f29
	pfld.l	4(r16)++, f30
	pfld.l	4(r16)++, f31
	fst.l	f16, 4(r17)++
	fst.l	f17, 4(r17)++
	fst.l	f18, 4(r17)++
	fst.l	f19, 4(r17)++
	fst.l	f20, 4(r17)++
	fst.l	f21, 4(r17)++
	fst.l	f22, 4(r17)++
	fst.l	f23, 4(r17)++
	fst.l	f24, 4(r17)++
	fst.l	f25, 4(r17)++
	fst.l	f26, 4(r17)++
	fst.l	f27, 4(r17)++
	fst.l	f28, 4(r17)++
	fst.l	f29, 4(r17)++
	fst.l	f30, 4(r17)++
	bla	r28, r29, .pump4
	 fst.l	f31, 4(r17)++
	shl	6, r30, r30		// subtract the number of bytes...
	br	.drain4
	 subs	r18, r30, r18		// ...moved in the loop.

.one4:
	//
	//	move just 1 4-byte word and try again from the top
	//	(making 8-, and perhaps 16-byte alignment)
	//
	fld.l	0(r16), f16
	addu	4, r16, r16
	addu	4, r17, r17
	addu	-4, r18, r18
	br	_bcopy_i
	 fst.l	f16, -4(r17)

.mis2:
	or	r16, r17, r31
	and	0x0001, r31, r0
	bnc	.mis1

	//
	//	there are four cases for (src | dst) & 1 == 0:
	//
	//	src=0 dst=0	-- not applicable (handled above)
	//	src=0 dst=2	-- 2-byte but can't become 4-byte aligned
	//	src=2 dst=0	-- 2-byte but can't become 4-byte aligned
	//	src=2 dst=2	-- can become 4-, 8-, 16-byte aligned
	//
	xor	r16, r17, r31
	and	0x0002, r31, r0
	bc	.one2			// might become 16-byte aligned

	//
	//	without resorting to additional shifting,
	//	this is the best we can do, moving 2-bytes
	//	at a time.  we do know, however, that there
	//	are >= 48 bytes to move.  (it's convenient
	//	to move only 24 bytes per iteration).
	//
	ld.s	 0(r16), r20
.copy2:
	ld.s	 2(r16), r21
	ld.s	 4(r16), r22
	ld.s	 6(r16), r23
	ld.s	 8(r16), r24
	ld.s	10(r16), r25
	ld.s	12(r16), r26
	ld.s	14(r16), r27
	ld.s	16(r16), r28
	ld.s	18(r16), r29
	ld.s	20(r16), r30
	ld.s	22(r16), r31
	ld.s	 0(r17), r0		// st.s dcache miss is sloooow
	addu	24, r16, r16
	st.s	r20,  0(r17)
	st.s	r21,  2(r17)
	st.s	r22,  4(r17)
	st.s	r23,  6(r17)
	st.s	r24,  8(r17)
	st.s	r25, 10(r17)
	st.s	r26, 12(r17)
	st.s	r27, 14(r17)
	st.s	r28, 16(r17)
	st.s	r29, 18(r17)
	st.s	r30, 20(r17)
	st.s	r31, 22(r17)
	addu	24, r17, r17
	adds	-24, r18, r18
	bc.t	.copy2
	 ld.s	 0(r16), r20
.tail2:
	bte	0, r18, .done
	adds	-2, r18, r0
	bc	.tail1
	ld.s	 0(r16), r20
	addu	 2, r16, r16
	ld.s	 0(r17), r0		// st.s dcache miss is slow
	addu	 2, r17, r17
	adds	-2, r18, r18
	br	.tail2
	 st.s	r20, -2(r17)
.one2:
	ld.s	0(r16), r20
	addu	2, r16, r16
	addu	2, r17, r17
	addu	-2, r18, r18
	br	_bcopy_i
	 st.s	r20, -2(r17)

.mis1:
	xor	r16, r17, r31
	and	0x0001, r31, r0
	bc	.one1			// might become 16-byte aligned

	ld.b	 0(r16), r20
.copy1:
	ld.b	 1(r16), r21
	ld.b	 2(r16), r22
	ld.b	 3(r16), r23
	ld.b	 4(r16), r24
	ld.b	 5(r16), r25
	ld.b	 6(r16), r26
	ld.b	 7(r16), r27
	ld.b	 8(r16), r28
	ld.b	 9(r16), r29
	ld.b	10(r16), r30
	ld.b	11(r16), r31
	ld.b	 0(r17), r0		// st.b dcache miss is sloooow
	addu	12, r16, r16
	st.b	r20,  0(r17)
	st.b	r21,  1(r17)
	st.b	r22,  2(r17)
	st.b	r23,  3(r17)
	st.b	r24,  4(r17)
	st.b	r25,  5(r17)
	st.b	r26,  6(r17)
	st.b	r27,  7(r17)
	st.b	r28,  8(r17)
	st.b	r29,  9(r17)
	st.b	r30, 10(r17)
	st.b	r31, 11(r17)
	addu	12, r17, r17
	adds	-12, r18, r18
	bc.t	.copy1
	 ld.b	 0(r16), r20
.tail1:
	adds	-1, r18, r0
	bc	.done
	ld.b	 0(r16), r20
	addu	 1, r16, r16
	ld.b	 0(r17), r0		// st.b dcache miss is slow
	addu	 1, r17, r17
	adds	-1, r18, r18
	br	.tail1
	 st.b	r20, -1(r17)
.one1:
	ld.b	0(r16), r20
	addu	1, r16, r16
	addu	1, r17, r17
	addu	-1, r18, r18
	br	_bcopy_i		// try again -- with better alignment
	 st.b	r20, -1(r17)

.small:
	bte	0, r18, .done
	or	r16, r17, r31
	and	0x000f, r31, r0	
	bc.t	.tail16
	 nop
	and	0x0007, r31, r0
	bc.t	.tail8
	 nop
	and	0x0003, r31, r0
	bc.t	.tail4
	 nop
	and	0x0001, r31, r0
	bc.t	.tail2
	 nop
	br	.tail1
	 nop
.done:
	bri	r1
	 nop
