// $Header: /tmp/hpctools/ga/tcgmsg/ipcv4.0/memcpy.s,v 1.1.1.1 1994-03-29 06:44:48 d3g681 Exp $
//
//     RJH
//     C entry is same as standard library routine memcpy
//
//     char *Memcpy (s1, s2, n)
//     char *s1, *s2;
//     int n;
//
//     Memcpy() copies n characters from memory area s2 to s1.   It
//     returns s1. 
//     Standard library routine achieves about 3.8 Mbyte/s.
//     This does 38.2 Mbyte/s for 8 byte aligned input and output
//               21.8 Mbyte/s for 4 ....
//                6.3 Mbyte/s for unaligned data
//     The theoretical peak on the FX2800 is 80/2=40Mb/s for data
//     in the shared cache.
//
//     FORTRAN entry is
//
//     subroutine memcpy(a, b, n)
//
	.text
	.globl		_Memcpy   // Fortran name 
	.globl		_memcpy_  // C name
	.align		16
//
//	FORTRAN entry ... r18 is passed by reference ... load it in
//
_memcpy_:
	ld.l	0(r18), r18
//
//	C entry
//
_Memcpy:
	mov	r16, r19	// save r19 in return register
	adds	-1, r0, r20	// store -1 in r20
//
	or	r19, r17, r22	// or addresses together
	and	7, r22, r0	//
	bc	aligned8	// skip to 8 byte aligned code
	and	3, r22, r0	//
	bc	aligned4	// skip to 4 byte aligned code
	br	aligned1	// skip to 1 byte aligned code
	  nop
//
//	code for eight byte alignment ... four way unrolled doubles (32 bytes)
//	38.2 Mbyte/s = full speed if input is cachable
//
aligned8:
	shr	5, r18, r21	// r21 = r18/32
	shl	5, r21, r22
	subs	r18, r22, r18	// r18 = remainder
	adds	-1, r21, r21// bla does 0,...,r21-1
	bc	aligned4	// skip if r21 < 1
	adds	-8, r19, r19	// prepare for autoinc
	bla	r20, r21, loop8a
	  adds	-8, r17, r17	// prepare for autoinc
loop8a:	fld.d	8(r17)++, f8	// get 8 bytes
	fld.d	8(r17)++, f10	// get 8 bytes
	fld.d	8(r17)++, f12	// get 8 bytes
	fld.d	8(r17)++, f14	// get 8 bytes
	fst.d	f8, 8(r19)++	// store 8 bytes
	fst.d	f10, 8(r19)++	// store 8 bytes
	fst.d	f12, 8(r19)++	// store 8 bytes
	bla	r20, r21, loop8a// decrement and branch
	  fst.d	f14, 8(r19)++	// store 8 bytes
//
	adds	8, r19, r19
	adds	8, r17, r17	// undo autoinc offsets and fall thru
//
//	code for 4 byte aligned ... 4 way unrolled integer copy (16 bytes)
//	21.8 Mbytes/s = about half speed if input is cachable
//
aligned4:
	shr	4, r18, r21	// r21 = r18/16
	shl	4, r21, r22
	subs	r18, r22, r18	// r18 = remainder
	adds	-1, r21, r21// bla does 0,...,r21-1
	bc	aligned1	// skip if r21 < 1
	bla	r20, r21, loop4a
	  nop
loop4a:	ld.l	0(r17), r22	// get 4 bytes
	ld.l	4(r17), r23	// get 4 bytes
	ld.l	8(r17), r24	// get 4 bytes
	ld.l	12(r17), r25	// get 4 bytes
	adds	16, r17, r17	// increment address
	st.l	r22, 0(r19)	// store 4 bytes
	st.l	r23, 4(r19)	// store 4 bytes
	st.l	r24, 8(r19)	// store 4 bytes
	st.l	r25, 12(r19)	// store 4 bytes
	bla	r20, r21, loop4a// decrement and branch
	  adds	16, r19, r19	// increment address in delay slot
//
//	2 byte aligned ... slower than single bytes ... deleted
//
//	code for general alignment ... 4 way unrolled byte copy
//	6.3 Mbytes/s if input is cachable
//
aligned1:
	shr	2, r18, r21	// r21 = r18/4
	shl	2, r21, r22
	subs	r18, r22, r18	// r18 = remainder
	adds	-1, r21, r21// bla does 0,...,r21-1
	bc	done1a		// skip if r21 < 1
	bla	r20, r21, loop1a
	  nop
loop1a: ld.b	0(r17), r22	// get byte
	ld.b	1(r17), r23	// get byte
	ld.b	2(r17), r24	// get byte
	ld.b	3(r17), r25	// get byte
	adds	4, r17, r17	// increment address
	st.b	r22, 0(r19)	// store byte
	st.b	r23, 1(r19)	// store byte
	st.b	r24, 2(r19)	// store byte
	st.b	r25, 3(r19)	// store byte
	bla	r20, r21, loop1a
	  adds	4, r19, r19	// increment address in delay slot
//
//	tidy up loop for single byte copy
//
done1a:	adds	-1, r18, r18	// bla does 0,...,r18-1
	bc	done		// skip if r18<1
	bla	r20, r18, loop1b
	  nop
loop1b: ld.b	0(r17), r22	// get byte
	adds	1, r17, r17	// increment address
	st.b	r22, 0(r19)	// store byte
	bla	r20, r18, loop1b	// decrement and branch
	  adds	1, r19, r19	// increment address in delay slot
//
done:	bri	r1
	  nop
