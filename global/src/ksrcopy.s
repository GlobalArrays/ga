	.file	"/home2/d3h325/g/global/src/"
	.file	"ksrcopy.c"
	.vstamp 7
# KSR1 ccom  -OLM -X28 -X92 -X115 -X151 -X153 -X155 -X156 -X157 -X158 -X159
# 	 -X172 -X187 -X205
# ccom: 1.2.1.3-1.1.2 

	.text

	.data
	.def	CopyTo$TXT;	.val	CopyTo$TXT;	.scl	2;	.endef

	.text
	.def	CopyTo;	.val	CopyTo;	.scl	2;	.type	513;	.endef
CopyTo$TXT:
   finop				;  cxnop
   finop				;  cxnop
	.def	.bf;	.val	.;	.scl	101;	.line	4;	.endef
	.ln	2	# 5
   itstle8	0, %i4			;  movb8_8	%i2, %c7
   add8.ntr	127, %i4, %i9		;  movb8_8	%i3, %c6
   selsc8	%i4, %i9, %i9		;  cxnop
   ash8.ntr	-7, %i9, %i9		;  cxnop
   itstge8	0, %i9			;  cxnop
   movi8	0, %i5			;  bcs.qn	@citst, .L5
   finop                                ;  pcsp.ex.nbl  128(%c6)
   finop                                ;  pcsp.ex.nbl  256(%c6)
.L6:
   finop                                ;  pcsp.ex.nbl  384(%c6)
   add8.ntr	1, %i5, %i5		;  ld8		0(%c7), %f0
   itstge8	%i5, %i9		;  ld8		8(%c7), %f1
   add8.ntr	33, %i31, %i31		;  ld8		16(%c7), %f5
   finop				;  ld8		24(%c7), %f6
   finop				;  ld8		32(%c7), %f7
   finop				;  ld8		40(%c7), %f8
   finop				;  ld8		48(%c7), %f9
   finop				;  ld8		56(%c7), %f10
   finop				;  sadd8.ntr	0, %c6, 128, %c6
   finop				;  st8		%f0, -128(%c6)
   finop				;  st8		%f1, -120(%c6)
   finop				;  st8		%f5, -112(%c6)
   finop				;  st8		%f6, -104(%c6)
   finop				;  st8		%f7, -96(%c6)
   finop				;  st8		%f8, -88(%c6)
   finop				;  st8		%f9, -80(%c6)
   finop				;  st8		%f10, -72(%c6)
   finop				;  ld8		120(%c7), %f10
   finop				;  ld8		112(%c7), %f9
   finop				;  ld8		104(%c7), %f8
   finop				;  ld8		96(%c7), %f7
   finop				;  ld8		88(%c7), %f6
   finop				;  ld8		80(%c7), %f5
   finop				;  ld8		72(%c7), %f1
   finop				;  ld8		64(%c7), %f0
   finop				;  st8		%f5, -48(%c6)
   finop				;  st8		%f6, -40(%c6)
   finop				;  st8		%f1, -56(%c6)
   finop				;  st8		%f0, -64(%c6)
   finop				;  st8		%f7, -32(%c6)
   finop				;  st8		%f8, -24(%c6)
   finop				;  st8		%f9, -16(%c6)
   finop				;  bcc.qn	@citst, .L6
   finop				;  sadd8.ntr	0, %c7, 128, %c7
   finop				;  st8		%f10, -8(%c6)
.L5:
   itstle8	0, %i4			;  cxnop
   clrh8	7, %i4, %i4		;  cxnop
   sub8.ntr	%i4, 128, %i9		;  cxnop
   selsc8	%i4, %i9, %i9		;  cxnop
   itsteq8	0, %i4			;  cxnop
   selsc8	%i4, %i9, %i9		;  cxnop
   itstle8	0, %i9			;  cxnop
   add8.ntr	7, %i9, %i5		;  cxnop
   selsc8	%i9, %i5, %i5		;  cxnop
   ash8.ntr	-3, %i5, %i9		;  cxnop
   itstge8	0, %i9			;  cxnop
   movi8	0, %i5			;  bcs.qt	@citst, .L1
.L3:
   add8.ntr	1, %i5, %i5		;  ld8		0(%c7), %f0
   itstge8	%i5, %i9		;  sadd8.ntr	0, %c6, 8, %c6
   add8.ntr	3, %i31, %i31		;  bcc.qn	@citst, .L3
   finop				;  sadd8.ntr	0, %c7, 8, %c7
   finop				;  st8		%f0, -8(%c6)
.L1:
   add8.ntr	3, %i31, %i31		;  jmp		32(%c14)
   finop				;  cxnop
   finop				;  cxnop
	.def	.ef;	.val	.;	.scl	101;	.line	51;	.endef
	.def	CopyTo;	.scl	-1;	.endef

	.data
# i	%i5	local
# m	%i9	local
# a	%f0	local
# b	%f1	local
# c	%f5	local
# d	%f6	local
# e	%f7	local
# f	%f8	local
# g	%f9	local
# h	%f10	local
# src	%c7	local
# dest	%c6	local
	.half	  0x0, 0x0, 0x0, 0x0
.L10:
CopyTo:	.word	  CopyTo$TXT

# src0	%c7	local
# dest0	%c6	local
# n	%i4	local

	.text

	.data
	.def	CopyFrom$TXT;	.val	CopyFrom$TXT;	.scl	2;	.endef

	.text
	.def	CopyFrom;	.val	CopyFrom;	.scl	2;	.type	513;	.endef
CopyFrom$TXT:
   finop				;  cxnop
   finop				;  cxnop
	.def	.bf;	.val	.;	.scl	101;	.line	60;	.endef
	.ln	2	# 61
   itstle8	0, %i4			;  movb8_8	%i2, %c7
   add8.ntr	127, %i4, %i9		;  movb8_8	%i3, %c6
   selsc8	%i4, %i9, %i9		;  cxnop
   ash8.ntr	-7, %i9, %i9		;  cxnop
   itstge8	0, %i9			;  cxnop
   movi8	0, %i5			;  bcs.qn	@citst, .L17
   finop                                ;  pcsp.ro.nbl   128(%c7)
   finop                                ;  pcsp.ro.nbl   256(%c7)
.L18:
   finop                                ;  pcsp.ro.nbl   384(%c7)
   add8.ntr	1, %i5, %i5		;  ld8		0(%c7), %f0
   itstge8	%i5, %i9		;  ld8		8(%c7), %f1
   add8.ntr	33, %i31, %i31		;  ld8		16(%c7), %f5
   finop				;  ld8		24(%c7), %f6
   finop				;  ld8		32(%c7), %f7
   finop				;  ld8		40(%c7), %f8
   finop				;  ld8		48(%c7), %f9
   finop				;  ld8		56(%c7), %f10
   finop				;  sadd8.ntr	0, %c6, 128, %c6
   finop				;  st8		%f0, -128(%c6)
   finop				;  st8		%f1, -120(%c6)
   finop				;  st8		%f5, -112(%c6)
   finop				;  st8		%f6, -104(%c6)
   finop				;  st8		%f7, -96(%c6)
   finop				;  st8		%f8, -88(%c6)
   finop				;  st8		%f9, -80(%c6)
   finop				;  st8		%f10, -72(%c6)
   finop				;  ld8		120(%c7), %f10
   finop				;  ld8		112(%c7), %f9
   finop				;  ld8		104(%c7), %f8
   finop				;  ld8		96(%c7), %f7
   finop				;  ld8		88(%c7), %f6
   finop				;  ld8		80(%c7), %f5
   finop				;  ld8		72(%c7), %f1
   finop				;  ld8		64(%c7), %f0
   finop				;  st8		%f5, -48(%c6)
   finop				;  st8		%f6, -40(%c6)
   finop				;  st8		%f1, -56(%c6)
   finop				;  st8		%f0, -64(%c6)
   finop				;  st8		%f7, -32(%c6)
   finop				;  st8		%f8, -24(%c6)
   finop				;  st8		%f9, -16(%c6)
   finop				;  bcc.qn	@citst, .L18
   finop				;  sadd8.ntr	0, %c7, 128, %c7
   finop				;  st8		%f10, -8(%c6)
.L17:
   itstle8	0, %i4			;  cxnop
   clrh8	7, %i4, %i4		;  cxnop
   sub8.ntr	%i4, 128, %i9		;  cxnop
   selsc8	%i4, %i9, %i9		;  cxnop
   itsteq8	0, %i4			;  cxnop
   selsc8	%i4, %i9, %i9		;  cxnop
   itstle8	0, %i9			;  cxnop
   add8.ntr	7, %i9, %i5		;  cxnop
   selsc8	%i9, %i5, %i5		;  cxnop
   ash8.ntr	-3, %i5, %i9		;  cxnop
   itstge8	0, %i9			;  cxnop
   movi8	0, %i5			;  bcs.qt	@citst, .L13
.L15:
   add8.ntr	1, %i5, %i5		;  ld8		0(%c7), %f0
   itstge8	%i5, %i9		;  sadd8.ntr	0, %c6, 8, %c6
   add8.ntr	3, %i31, %i31		;  bcc.qn	@citst, .L15
   finop				;  sadd8.ntr	0, %c7, 8, %c7
   finop				;  st8		%f0, -8(%c6)
.L13:
   add8.ntr	3, %i31, %i31		;  jmp		32(%c14)
   finop				;  cxnop
   finop				;  cxnop
	.def	.ef;	.val	.;	.scl	101;	.line	51;	.endef
	.def	CopyFrom;	.scl	-1;	.endef

	.data
# i	%i5	local
# m	%i9	local
# a	%f0	local
# b	%f1	local
# c	%f5	local
# d	%f6	local
# e	%f7	local
# f	%f8	local
# g	%f9	local
# h	%f10	local
# src	%c7	local
# dest	%c6	local
	.half	  0x0, 0x0, 0x0, 0x0
.L22:
CopyFrom:	.word	  CopyFrom$TXT

# src0	%c7	local
# dest0	%c6	local
# n	%i4	local

	.text

	.data
	.def	Accum$TXT;	.val	Accum$TXT;	.scl	2;	.endef

	.text
	.def	Accum;	.val	Accum;	.scl	2;	.type	513;	.endef
Accum$TXT:
   finop				;  cxnop
   finop				;  cxnop
	.def	.bf;	.val	.;	.scl	101;	.line	116;	.endef
	.ln	2	# 117
   movin8	%f0			;  movout8	%i2
   itstle8	0, %i5			;  ssub8.ntr	0, %sp, 128, %sp
   add8.ntr	15, %i5, %i10		;  movb8_8	%i3, %c7
   selsc8	%i5, %i10, %i10		;  movb8_8	%i4, %c6
   ash8.ntr	-4, %i10, %i10		;  st8		%f16, 88(%sp)
   itstge8	0, %i10			;  st8		%f17, 80(%sp)
   movi8	0, %i9			;  st8		%f18, 72(%sp)
   add8.ntr	13, %i31, %i31		;  st8		%f19, 64(%sp)
   finop				;  st8		%f20, 56(%sp)
   finop				;  st8		%f21, 48(%sp)
   finop				;  bcs.qn	@citst, .L29
   finop				;  st8		%f22, 40(%sp)
   finop				;  st8		%f23, 32(%sp)
   finop                                ;  pcsp.ex.nbl  128(%c6)
   finop                                ;  pcsp.ex.nbl  256(%c6)
.L30:
   finop                                ;  pcsp.ex.nbl  384(%c6)
   add8.ntr	1, %i9, %i9		;  ld8		0(%c7), %f1
   itstge8	%i9, %i10		;  ld8.ex	0(%c6), %f12
   add8.ntr	53, %i31, %i31		;  ld8		8(%c7), %f5
   fmad8.tr	%f1, %f0, %f12		;  ld8.ex	8(%c6), %f13
   finop				;  ld8.ex	32(%c6), %f1
   fmad8.tr	%f5, %f0, %f13		;  ld8		32(%c7), %f8
   finop				;  ld8.ex	40(%c6), %f5
   finop				;  ld8		40(%c7), %f9
   fmad8.tr	%f8, %f0, %f1		;  ld8		16(%c7), %f6
   finop				;  ld8.ex	16(%c6), %f14
   fmad8.tr	%f9, %f0, %f5		;  ld8		24(%c7), %f7
   fmad8.tr	%f6, %f0, %f14		;  ld8.ex	24(%c6), %f15
   finop				;  ld8.ex	48(%c6), %f6
   fmad8.tr	%f7, %f0, %f15		;  ld8		48(%c7), %f10
   finop				;  ld8.ex	56(%c6), %f7
   finop				;  ld8		56(%c7), %f11
   fmad8.tr	%f10, %f0, %f6		;  ld8.ex	64(%c6), %f16
   finop				;  ld8.ex	72(%c6), %f17
   fmad8.tr	%f11, %f0, %f7		;  ld8.ex	80(%c6), %f18
   finop				;  ld8.ex	88(%c6), %f19
   finop				;  ld8.ex	96(%c6), %f20
   finop				;  ld8.ex	104(%c6), %f21
   finop				;  ld8.ex	112(%c6), %f22
   finop				;  ld8.ex	120(%c6), %f23
   finop				;  st8		%f12, 0(%c6)
   finop				;  st8		%f13, 8(%c6)
   finop				;  st8		%f14, 16(%c6)
   finop				;  st8		%f15, 24(%c6)
   finop				;  st8		%f1, 32(%c6)
   finop				;  st8		%f5, 40(%c6)
   finop				;  st8		%f6, 48(%c6)
   finop				;  st8		%f7, 56(%c6)
   finop				;  ld8		120(%c7), %f11
   finop				;  ld8		112(%c7), %f10
   finop				;  ld8		104(%c7), %f9
   fmad8.tr	%f11, %f0, %f23		;  ld8		96(%c7), %f8
   fmad8.tr	%f10, %f0, %f22		;  ld8		88(%c7), %f7
   fmad8.tr	%f9, %f0, %f21		;  ld8		80(%c7), %f6
   fmad8.tr	%f8, %f0, %f20		;  ld8		72(%c7), %f5
   fmad8.tr	%f7, %f0, %f19		;  ld8		64(%c7), %f1
   fmad8.tr	%f6, %f0, %f18		;  sadd8.ntr	0, %c6, 128, %c6
   fmad8.tr	%f5, %f0, %f17		;  sadd8.ntr	0, %c7, 128, %c7
   fmad8.tr	%f1, %f0, %f16		;  cxnop
   finop				;  cxnop
   finop				;  st8		%f21, -24(%c6)
   finop				;  st8		%f20, -32(%c6)
   finop				;  st8		%f19, -40(%c6)
   finop				;  st8		%f18, -48(%c6)
   finop				;  st8		%f22, -16(%c6)
   finop				;  st8		%f17, -56(%c6)
   finop				;  bcc.qn	@citst, .L30
   finop				;  st8		%f16, -64(%c6)
   finop				;  st8		%f23, -8(%c6)
.L29:
   itstle8	0, %i5			;  cxnop
   clrh8	4, %i5, %i5		;  cxnop
   sub8.ntr	%i5, 16, %i9		;  cxnop
   selsc8	%i5, %i9, %i10		;  cxnop
   itsteq8	0, %i5			;  cxnop
   selsc8	%i5, %i10, %i10		;  cxnop
   itstge8	0, %i10			;  cxnop
   movi8	0, %i9			;  bcs.qt	@citst, .L25
.L27:
   add8.ntr	1, %i9, %i9		;  ld8		0(%c7), %f9
   itstge8	%i9, %i10		;  ld8.ex	0(%c6), %f8
   add8.ntr	5, %i31, %i31		;  sadd8.ntr	0, %c6, 8, %c6
   fmad8.tr	%f0, %f9, %f8		;  sadd8.ntr	0, %c7, 8, %c7
   finop				;  bcc.qn	@citst, .L27
   finop				;  cxnop
   finop				;  st8		%f8, -8(%c6)
   add8.ntr	11, %i31, %i31		;  ld8		88(%sp), %f16
   finop				;  ld8		80(%sp), %f17
   finop				;  ld8		72(%sp), %f18
   finop				;  ld8		64(%sp), %f19
   finop				;  ld8		56(%sp), %f20
   finop				;  ld8		48(%sp), %f21
   finop				;  ld8		40(%sp), %f22
   finop				;  ld8		32(%sp), %f23
   finop				;  jmp		32(%c14)
   finop				;  sadd8.ntr	0, %sp, 128, %sp
   finop				;  cxnop
.L25:
   add8.ntr	11, %i31, %i31		;  ld8		88(%sp), %f16
   finop				;  ld8		80(%sp), %f17
   finop				;  ld8		72(%sp), %f18
   finop				;  ld8		64(%sp), %f19
   finop				;  ld8		56(%sp), %f20
   finop				;  ld8		48(%sp), %f21
   finop				;  ld8		40(%sp), %f22
   finop				;  ld8		32(%sp), %f23
   finop				;  jmp		32(%c14)
   finop				;  sadd8.ntr	0, %sp, 128, %sp
   finop				;  cxnop
	.def	.ef;	.val	.;	.scl	101;	.line	50;	.endef
	.def	Accum;	.scl	-1;	.endef

	.data
# i	%i9	local
# m	%i10	local
# a	%f1	local
# b	%f5	local
# c	%f6	local
# d	%f7	local
# e	%f8	local
# f	%f9	local
# g	%f10	local
# h	%f11	local
# src	%c7	local
# dest	%c6	local
	.half	  0x0, 0xff0000, 0x0, 0x0
.L34:
Accum:	.word	  Accum$TXT

# alpha	%f0	local
# src0	%c7	local
# dest0	%c6	local
# n	%i5	local

	.text

	.data

	.align  	128
.L37:
	.globl  	Accum
	.globl  	Accum$TXT
	.globl  	CopyFrom
	.globl  	CopyFrom$TXT
	.globl  	CopyTo
	.globl  	CopyTo$TXT

	.text
