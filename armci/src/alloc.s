	.set noat
	.set noreorder
	.data
	.lcomm	$$1$info 32
	.lcomm	$$2$offsets 512
	.data
	.lcomm	$$3$armci_base_map 8
	.data
$$1:
$$2:
	.ascii	"smp count too large\X00"

	.byte	0 : 4
$$3:
	.ascii	"smp count bad\X00"

	.byte	0 : 2
$$4:
	.ascii	"getaddressconf failed\X00"

	.byte	0 : 2
$$5:
	.ascii	"******armci_base_map = %p \X0A\X00"

	.byte	0 : 4
$$6:
	.ascii	">shmem_reserve failed\X00"

	.byte	0 : 2
$$7:
	.ascii	"reserved address = %p\X0A\X00"

	.byte	0 : 1
$$8:
	.ascii	"smp count too large\X00"

	.byte	0 : 4
$$9:
	.ascii	"%d i=%d> before =%p %p reserved=%ld size=%ld bytes=%ld pid=%ld\X0A\X00"

$$10:
	.ascii	"failed:end=\X00"

	.byte	0 : 4
$$11:
	.ascii	"%d: fixup p=%d peer=%d %p ->>> %p\X0A\X00"

	.byte	0 : 5
	.sdata
$$12:
$$4$armci_reserve_bytes:
	.quad	0x0 # .quad 0
	.byte	0 : 8
	.rdata
$$13:
	.extern armci_clus_first 4
	.extern armci_clus_last 4
	.extern armci_me 4
	.extern armci_master 4
	.text
	.arch	generic
	.align 4
	.file 1 "alloc.c"
	.loc 1 34
 #     34 void armci_init_alloc(size_t reserve_bytes, int slaves)
	.globl  armci_init_alloc
	.ent 	armci_init_alloc
	.loc 1 34
armci_init_alloc:
	.context full
	ldah	$gp, ($27)!gpdisp!1
	unop
	lda	$gp, ($gp)!gpdisp!1
	unop
L$27:
	lda	$sp, -272($sp)
	.loc 1 40
 #     35 {
 #     36 struct addressconf config[AC_N_AREAS];
 #     37 armci_alloc_t *pinfo = & info;
 #     38 char *ptr;
 #     39 int rc;
 #     40 size_t pagesize=getpagesize();
	ldq	$27, getpagesize($gp)!literal!2										   # 000040
	.loc 1 34
	stq	$26, ($sp)												   # 000034
	stq	$9, 8($sp)
	stq	$10, 16($sp)
	stq	$11, 24($sp)
	.mask 0x04000E00,-272
	.fmask 0x00000000,0
	.frame  $sp, 272, $26
	.prologue 1
	mov	$16, $9
	.loc 1 35
	sextl	$17, $10												   # 000035
	.loc 1 40
	jsr	$26, ($27), getpagesize!lituse_jsr!2									   # 000040
	ldah	$gp, ($26)!gpdisp!3
	.loc 1 42
 #     41 
 #     42     if(MAX_SMP_SLAVES<slaves) ARMCI_Error("smp count too large",slaves);
	cmple	$10, 64, $1												   # 000042
	.loc 1 40
	stq	$0, 32($sp)												   # 000040
	.loc 1 42
	mov	$10, $17												   # 000042
	.loc 1 40
	lda	$gp, ($gp)!gpdisp!3											   # 000040
	.loc 1 42
	bne	$1, L$28												   # 000042
	ldq	$27, ARMCI_Error($gp)!literal!4
	ldq	$16, $$2+144($gp)!literal!5
	lda	$16, -144($16)!lituse_base!5
	jsr	$26, ($27), ARMCI_Error!lituse_jsr!4
	ldah	$gp, ($26)!gpdisp!6
	lda	$gp, ($gp)!gpdisp!6
L$28:
	.loc 1 43
 #     43     if(slaves<1) ARMCI_Error("smp count bad",slaves);
	bgt	$10, L$33												   # 000043
	mov	$10, $17
	ldq	$27, ARMCI_Error($gp)!literal!7
	ldq	$16, $$3+120($gp)!literal!8
	lda	$16, -120($16)!lituse_base!8
	jsr	$26, ($27), ARMCI_Error!lituse_jsr!7
	ldah	$gp, ($26)!gpdisp!9
	lda	$gp, ($gp)!gpdisp!9
L$33:
	.loc 1 45
 #     44 
 #     45     rc = getaddressconf(config, sizeof(config));
	lda	$16, 40($sp)												   # 000045
	mov	224, $17
	ldq	$27, getaddressconf($gp)!literal!10
	jsr	$26, ($27), getaddressconf!lituse_jsr!10
	ldah	$gp, ($26)!gpdisp!11
	mov	$0, $17
	lda	$gp, ($gp)!gpdisp!11
	.loc 1 46
 #     46     if(rc<0)ARMCI_Error("getaddressconf failed",rc);
	bge	$0, L$35												   # 000046
	ldq	$27, ARMCI_Error($gp)!literal!12
	ldq	$16, $$4+104($gp)!literal!13
	lda	$16, -104($16)!lituse_base!13
	jsr	$26, ($27), ARMCI_Error!lituse_jsr!12
	ldah	$gp, ($26)!gpdisp!14
	lda	$gp, ($gp)!gpdisp!14
L$35:
	.loc 1 48
 #     47 
 #     48     pinfo->base = (char*) config[AC_MMAP_DATA].ac_base;
	ldq	$0, 216($sp)												   # 000048
	.loc 1 52
 #     49     /*pinfo->base = 0x30000000000;*/
 #     50     if(DEBUG)printf("******armci_base_map = %p \n",pinfo->base);
 #     51 
 #     52     bzero(offsets,sizeof(offsets));
	mov	512, $17												   # 000052
	ldq	$27, bzero($gp)!literal!15
	.loc 1 37
	ldq	$11, $$1$info+32($gp)!literal!16									   # 000037
	lda	$11, -32($11)!lituse_base!16
	.loc 1 48
	stq	$0, ($11)												   # 000048
	.loc 1 52
	lda	$16, 32($11)												   # 000052
	jsr	$26, ($27), bzero!lituse_jsr!15
	ldah	$gp, ($26)!gpdisp!17
	.loc 1 55
 #     53 
 #     54 
 #     55     if(reserve_bytes%pagesize) reserve_bytes=RESERVE_BYTES;
	mov	$9, $16													   # 000055
	ldq	$17, 32($sp)
	mov	1, $2
	.loc 1 52
	lda	$gp, ($gp)!gpdisp!17											   # 000052
	.loc 1 55
	sll	$2, 31, $2												   # 000055
	ldq	$27, _OtsRemainder64Unsigned($gp)!literal!18
	jsr	$26, ($27), _OtsRemainder64Unsigned!lituse_jsr!18
	ldah	$gp, ($26)!gpdisp!19
	cmovne	$0, $2, $9
	.loc 1 56
 #     56     pinfo->reserved = reserve_bytes/slaves;
	mov	$10, $17												   # 000056
	.loc 1 55
	lda	$gp, ($gp)!gpdisp!19											   # 000055
	.loc 1 56
	ldq	$27, _OtsDivide64Unsigned($gp)!literal!20								   # 000056
	mov	$9, $16
	jsr	$26, ($27), _OtsDivide64Unsigned!lituse_jsr!20
	ldah	$gp, ($26)!gpdisp!21
	.loc 1 59
 #     57 
 #     58     /* roundup size to reserve per peer */
 #     59     pinfo->reserved >>= 22;
	srl	$0, 22, $0												   # 000059
	.loc 1 62
 #     60     pinfo->reserved <<= 22;
 #     61 
 #     62     pinfo->ptr = (char*) shmem_reserve(pinfo->base, reserve_bytes, 0);
	ldq	$16, ($11)												   # 000062
	mov	$9, $17
	.loc 1 56
	lda	$gp, ($gp)!gpdisp!21											   # 000056
	.loc 1 60
	sll	$0, 22, $0												   # 000060
	.loc 1 62
	clr	$18													   # 000062
	ldq	$27, shmem_reserve($gp)!literal!22
	.loc 1 60
	stq	$0, 16($11)												   # 000060
	unop
	.loc 1 62
	jsr	$26, ($27), shmem_reserve!lituse_jsr!22									   # 000062
	ldah	$gp, ($26)!gpdisp!23
	stq	$0, 8($11)
	.loc 1 63
 #     63     if(!pinfo->ptr)ARMCI_Error(">shmem_reserve failed",0);
	clr	$17													   # 000063
	.loc 1 62
	lda	$gp, ($gp)!gpdisp!23											   # 000062
	.loc 1 63
	bne	$0, L$36												   # 000063
	ldq	$27, ARMCI_Error($gp)!literal!24
	ldq	$16, $$6+48($gp)!literal!25
	lda	$16, -48($16)!lituse_base!25
	jsr	$26, ($27), ARMCI_Error!lituse_jsr!24
	ldah	$gp, ($26)!gpdisp!26
	lda	$gp, ($gp)!gpdisp!26
	.loc 1 66
 #     64 
 #     65     if(DEBUG)printf("reserved address = %p\n",pinfo->ptr);
 #     66 }
L$36:															   # 000066
	ldq	$26, ($sp)
	ldq	$9, 8($sp)
	ldq	$10, 16($sp)
	ldq	$11, 24($sp)
	lda	$sp, 272($sp)
	ret	($26)
	.end 	armci_init_alloc
	unop
	unop
	.loc 1 34
	.loc 1 68
 #     67 
 #     68 char *armci_region_getcore(size_t bytes)
	.globl  armci_region_getcore
	.ent 	armci_region_getcore
	.loc 1 68
armci_region_getcore:
	.context full
	ldah	$gp, ($27)!gpdisp!27
	unop
	lda	$gp, ($gp)!gpdisp!27
	unop
L$39:
	lda	$sp, -16($sp)
	.loc 1 71
 #     69 {
 #     70      if (armci_reserve_bytes==0){
 #     71          armci_reserve_bytes =1;
	mov	1, $7													   # 000071
	.loc 1 72
 #     72          armci_init_alloc(-1,armci_clus_last-armci_clus_first+1);
	ldq	$6, armci_clus_first($gp)!literal!28									   # 000072
	ldq	$5, armci_clus_last($gp)!literal!29
	.loc 1 70
	ldq	$3, $$4$armci_reserve_bytes($gp)!literal!30								   # 000070
	ldq	$4, ($3)!lituse_base!30
	.loc 1 68
	stq	$26, ($sp)												   # 000068
	.mask 0x04000000,-16
	.fmask 0x00000000,0
	.frame  $sp, 16, $26
	.prologue 1
	stq	$16, 8($sp)
	.loc 1 72
	mov	-1, $16													   # 000072
	ldl	$5, ($5)!lituse_base!29
	ldl	$6, ($6)!lituse_base!28
	unop
	.loc 1 70
	bne	$4, L$40												   # 000070
	.loc 1 71
	stq	$7, ($3)!lituse_base!30											   # 000071
	.loc 1 72
	subl	$5, $6, $5												   # 000072
	addl	$5, 1, $17
	bsr	$26, L$27
	.loc 1 73
 #     73      }
L$40:															   # 000073
	.loc 1 74
 #     74      return malloc(bytes);
	ldq	$27, malloc($gp)!literal!31										   # 000074
	ldq	$16, 8($sp)
	unop
	jsr	$26, ($27), malloc!lituse_jsr!31
	ldah	$gp, ($26)!gpdisp!32
	.loc 1 75
 #     75 }
	ldq	$26, ($sp)												   # 000075
	lda	$sp, 16($sp)
	.loc 1 74
	.context none
	lda	$gp, ($gp)!gpdisp!32											   # 000074
	.loc 1 75
	ret	($26)													   # 000075
	.end 	armci_region_getcore
	unop
	unop
	.loc 1 68
	.loc 1 78
 #     76 
 #     77 
 #     78 void exchange_info(int n, long *info)
	.globl  exchange_info
	.ent 	exchange_info
	.loc 1 78
exchange_info:
	.context full
	ldah	$gp, ($27)!gpdisp!33
	unop
	lda	$gp, ($gp)!gpdisp!33
	unop
L$18:
	.loc 1 79
 #     79 {
	sextl	$16, $21												   # 000079
	.loc 1 81
 #     80 int i;
 #     81      for(i=0; i< n; i++)
	ble	$21, L$42												   # 000081
	.loc 1 78
	lda	$sp, -48($sp)												   # 000078
	stq	$26, ($sp)
	stq	$9, 8($sp)
	stq	$10, 16($sp)
	stq	$11, 24($sp)
	stq	$12, 32($sp)
	stq	$13, 40($sp)
	.mask 0x04003E00,-48
	.fmask 0x00000000,0
	.frame  $sp, 48, $26
	.prologue 1
	clr	$11
	sextl	$16, $10
	mov	$17, $9
	.loc 1 82
 #     82          if(i!=(armci_me-armci_master)){
	ldq	$13, armci_me($gp)!literal!34										   # 000082
	ldq	$12, armci_master($gp)!literal!35
	unop
	unop
	.loc 1 81
L$23:															   # 000081
	.loc 1 82
	ldl	$3, ($13)												   # 000082
	ldl	$4, ($12)
	.loc 1 84
 #     83            //printf("%d i=%d puting %ld\n",armci_me, i, info[armci_me]);
 #     84            shmem_put(info+armci_me-armci_master,info+armci_me-armci_master,1,i)
	mov	1, $18													   # 000084
	mov	$11, $19
	ldq	$27, shmem_put($gp)!literal!36
	.loc 1 82
	subl	$3, $4, $5												   # 000082
	.loc 1 84
	s8addq	$4, $31, $4												   # 000084
	s8addq	$3, $9, $3
	.loc 1 82
	xor	$11, $5, $5												   # 000082
	.loc 1 84
	subq	$3, $4, $17												   # 000084
	subq	$3, $4, $16
	unop
	.loc 1 82
	beq	$5, L$25												   # 000082
	.loc 1 84
	jsr	$26, ($27), shmem_put!lituse_jsr!36									   # 000084
	ldah	$gp, ($26)!gpdisp!37
	lda	$gp, ($gp)!gpdisp!37
	.loc 1 86
 #     85 ;
 #     86          }
L$25:															   # 000086
	.loc 1 81
	addl	$11, 1, $11												   # 000081
	cmplt	$11, $10, $0
	bne	$0, L$23
	.loc 1 87
 #     87 }
	ldq	$26, ($sp)												   # 000087
	ldq	$9, 8($sp)
	ldq	$10, 16($sp)
	ldq	$11, 24($sp)
	ldq	$12, 32($sp)
	ldq	$13, 40($sp)
	lda	$sp, 48($sp)
L$42:
	.context none
	ret	($26)
	.end 	exchange_info
	unop
	.loc 1 78
	.loc 1 90
 #     88 
 #     89 
 #     90 int armci_region_register(int p, void **pinout, long pid, size_t bytes)
	.globl  armci_region_register
	.ent 	armci_region_register
	.loc 1 90
armci_region_register:
	.context full
	ldah	$gp, ($27)!gpdisp!38
	unop
	lda	$gp, ($gp)!gpdisp!38
	unop
L$7:
	.loc 1 98
 #     91 {
 #     92 int i;
 #     93 void *end,*ptr,*save=*pinout;
 #     94 armci_alloc_t *pinfo = & info;
 #     95 char *ref = (char*)0x140000000;
 #     96 size_t rgn_size, map_size=0;
 #     97 
 #     98      if(!*pinout)return 0;
	ldq	$3, ($17)												   # 000098
	.loc 1 91
	sextl	$16, $23												   # 000091
	.loc 1 105
 #     99 
 #    100 #if 0
 #    101      printf("%d: trying to map for peer=%d pid=%ld bytes=%ld %p\n",armci_me, p,
 #    102  pid, bytes, *pinout);
 #    103 #endif
 #    104 
 #    105      if(MAX_SMP_SLAVES<p) ARMCI_Error("smp count too large",p);
	ldq	$27, ARMCI_Error($gp)!literal!39									   # 000105
	unop
	.loc 1 98
	beq	$3, L$44												   # 000098
	.loc 1 105
	cmple	$23, 64, $4												   # 000105
	.loc 1 90
	lda	$sp, -64($sp)												   # 000090
	stq	$26, ($sp)
	stq	$9, 8($sp)
	stq	$10, 16($sp)
	stq	$11, 24($sp)
	stq	$12, 32($sp)
	.mask 0x04001E00,-64
	.fmask 0x00000000,0
	.frame  $sp, 64, $26
	.prologue 1
	mov	$17, $9
	stq	$18, 48($sp)
	clr	$11
	sextl	$16, $10
	stq	$19, 56($sp)
	.loc 1 105
	bne	$4, L$11												   # 000105
	ldq	$16, $$8($gp)!literal!40
	mov	$23, $17
	jsr	$26, ($27), ARMCI_Error!lituse_jsr!39
	ldah	$gp, ($26)!gpdisp!41
	lda	$gp, ($gp)!gpdisp!41
L$11:
	.loc 1 109
 #    106      if(!offsets[p]){
 #    107 
 #    108         /* map memory allocated by others in my address space */
 #    109         rgn_size = ((char*)*pinout)+bytes-ref+getpagesize();
	ldq	$2, 56($sp)												   # 000109
	.loc 1 106
	ldq	$12, $$1$info+32($gp)!literal!42									   # 000106
	.loc 1 109
	ldq	$27, getpagesize($gp)!literal!43									   # 000109
	.loc 1 106
	s8addq	$10, $12, $10												   # 000106
	ldq	$0, ($10)
	bne	$0, L$13
	.loc 1 109
	ldq	$1, ($9)												   # 000109
	mov	5, $11
	addq	$1, $2, $1
	sll	$11, 30, $11
	subq	$1, $11, $11
	jsr	$26, ($27), getpagesize!lituse_jsr!43
	.loc 1 112
 #    110 
 #    111         /* allign size on 4MB boundary (an overkill) */
 #    112         map_size = rgn_size +4*1024*1024*1024L -1;
	mov	-1, $1													   # 000112
	.loc 1 109
	ldah	$gp, ($26)!gpdisp!44											   # 000109
	addq	$11, $0, $19
	.loc 1 118
 #    113         map_size >>= 22;
 #    114         map_size <<= 22;
 #    115 
 #    116         if(armci_me==0)
 #    117 	   if(DEBUG)printf("%d i=%d> before =%p %p reserved=%ld size=%ld bytes=%ld pid=%ld\n",armci_me,p, pinfo->ptr, *pinout, m
 # ap_size,rgn_size, bytes, pid); 
 #    118         end = shmem_map(pinfo->ptr, ref, map_size,rgn_size, pid, &ptr);
	mov	5, $17													   # 000118
	.loc 1 112
	zapnot	$1, 15, $1												   # 000112
	.loc 1 109
	lda	$gp, 4($gp)!gpdisp!44											   # 000109
	.loc 1 118
	ldl	$20, 48($sp)												   # 000118
	ldq	$16, -24($12)
	.loc 1 112
	addq	$19, $1, $1												   # 000112
	.loc 1 118
	ldq	$27, shmem_map($gp)!literal!45										   # 000118
	sll	$17, 30, $17
	lda	$21, 40($sp)
	.loc 1 113
	srl	$1, 22, $1												   # 000113
	.loc 1 114
	sll	$1, 22, $11												   # 000114
	.loc 1 118
	mov	$11, $18												   # 000118
	jsr	$26, ($27), shmem_map!lituse_jsr!45
	ldah	$gp, ($26)!gpdisp!46
	.loc 1 119
 #    119         if(!end) ARMCI_Error("failed:end=",0);
	clr	$17													   # 000119
	.loc 1 118
	lda	$gp, ($gp)!gpdisp!46											   # 000118
	.loc 1 119
	bne	$0, L$15												   # 000119
	ldq	$27, ARMCI_Error($gp)!literal!47
	ldq	$16, $$10-88($gp)!literal!48
	lda	$16, 88($16)!lituse_base!48
	jsr	$26, ($27), ARMCI_Error!lituse_jsr!47
	ldah	$gp, ($26)!gpdisp!49
	lda	$gp, ($gp)!gpdisp!49
L$15:
	.loc 1 121
 #    120         /* offsets[p] = (((char*)*pinout) -ref);*/
 #    121         offsets[p] = pinfo->ptr - ref;
	ldq	$0, -24($12)												   # 000121
	mov	5, $1
	sll	$1, 30, $1
	subq	$0, $1, $0
	stq	$0, ($10)
	.loc 1 124
 #    122 
 #    123 
 #    124      }
L$13:															   # 000124
	.loc 1 126
 #    125  /**pinout = pinfo->ptr + offsets[p];*/
 #    126      *pinout = offsets[p] + (char*)*pinout;
	ldq	$3, ($9)												   # 000126
	ldq	$10, ($10)
	addq	$3, $10, $3
	.loc 1 134
 #    127      pinfo->ptr += map_size;
 #    128 #if 0
 #    129      if(armci_me==0)printf("%d: peer=%d pid=%ld ptr=%p mapped to %p off=%ld\n",
 #    130                             armci_me, p, pid, save, *pinout,offsets[p]);
 #    131 #endif
 #    132 
 #    133      return 0;
 #    134 }
	ldq	$26, ($sp)												   # 000134
	ldq	$10, 16($sp)
	.loc 1 133
	clr	$0													   # 000133
	.loc 1 126
	stq	$3, ($9)												   # 000126
	.loc 1 127
	ldq	$4, -24($12)												   # 000127
	addq	$4, $11, $4
	.loc 1 134
	ldq	$9, 8($sp)												   # 000134
	ldq	$11, 24($sp)
	.loc 1 127
	stq	$4, -24($12)												   # 000127
	.loc 1 134
	ldq	$12, 32($sp)												   # 000134
	lda	$sp, 64($sp)
	ret	($26)
	unop
	unop
	unop
L$44:
	.loc 1 133
	clr	$0													   # 000133
	.loc 1 134
	ret	($26)													   # 000134
	.end 	armci_region_register
	unop
	unop
	.loc 1 90
	.loc 1 137
 #    135 
 #    136 
 #    137 void armci_region_fixup(int proc, void **pinout)
	.globl  armci_region_fixup
	.ent 	armci_region_fixup
	.loc 1 137
armci_region_fixup:
	.context full
	ldah	$gp, ($27)!gpdisp!50
	unop
	lda	$gp, ($gp)!gpdisp!50
	unop
	.frame  $sp, 0, $26
	.prologue 1
L$2:
	.loc 1 138
 #    138 {
	sextl	$16, $16												   # 000138
	.loc 1 140
 #    139 int p = proc -armci_clus_first;
 #    140     if(armci_me == proc) return;
	ldq	$4, armci_me($gp)!literal!51										   # 000140
	.loc 1 139
	ldq	$3, armci_clus_first($gp)!literal!52									   # 000139
	.loc 1 143
 #    141     if(DEBUG)
 #    142 	printf("%d: fixup p=%d peer=%d %p ->>> %p\n",armci_me, proc, p, *pinout, (char*)*pinout-offsets[p]);
 #    143     *pinout = (char*)*pinout - offsets[p];
	ldq	$6, $$1$info+32($gp)!literal!53										   # 000143
	lda	$6, ($6)!lituse_base!53
	.loc 1 140
	ldl	$4, ($4)!lituse_base!51											   # 000140
	.loc 1 139
	ldl	$3, ($3)!lituse_base!52											   # 000139
	.loc 1 140
	xor	$4, $16, $4												   # 000140
	.loc 1 139
	subl	$16, $3, $3												   # 000139
	.loc 1 140
	beq	$4, L$5													   # 000140
	.loc 1 143
	s8addq	$3, $6, $3												   # 000143
	ldq	$5, ($17)
	ldq	$3, ($3)
	subq	$5, $3, $3
	stq	$3, ($17)
	.loc 1 144
 #    144 }
L$5:															   # 000144
	unop
	ret	($26)
	.end 	armci_region_fixup
	.loc 1 137
