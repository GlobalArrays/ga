/* $Id: barrier.KSR.c,v 1.4 1999-07-28 00:27:02 d3h325 Exp $ */
/****************************************************************\ 
 Fast barriers for the KSR. 
 The dynamic f-way algorithm by Grunwald & Vajracharya.
 Adopted for multiprocessing environment from the original code. 
 J. Nieplocha, 01.26.94.
\****************************************************************/ 
 
#define NULL    0
#define P 	256
#define LogP	5
#define FANIN   6

typedef unsigned char boolean;
#define False ((boolean) 0)
#define True ((boolean) 1)

typedef unsigned char type_enum;
#define ROOT ((type_enum) 1)

typedef union {
	long whole;
	boolean parts[FANIN];
} whole_and_parts;

typedef struct treenode_t{
        boolean *myPlace;        /* pointer to my flag on the parent's word  */
	struct treenode_t *parent;       /* pointer to my parent */
	whole_and_parts childSoFar;      /* children that has arrived so far */
	whole_and_parts childExpected[2];/* children expected to arrive */
	unsigned char level;
	type_enum type;
        char padding[86];
} treenode_t;

/*
__align128 __shared treenode_t tree[P/FANIN][LogP]; 
__align128 __shared boolean global_sense = True;
*/

volatile treenode_t   *tree;                /* tree data structure */
volatile boolean *global_sense;             /* central sense used for wakeup */

__private boolean local_sense = False;    /* local private sense */
__private treenode_t *myleaf;                           
__private boolean *myStartingPlace;




int  KSRbarrier_mem_req()
{
/* memory for tree +  "global sense" 128-byte alligned */
return(sizeof(treenode_t)*(P/FANIN)*LogP + 128);
}




void KSRbarrier_init(numCpus, vpid,fan, shm_ptr)
int numCpus,vpid,fan;
char *shm_ptr;
{
extern void ga_error();

short fanin = fan;
int parentIndex;
int numLeft=numCpus;
treenode_t *node;
short i;

   if((int)shm_ptr%128)
      ga_error("barrier_init: shmem ptr not alligned ",(long)shm_ptr);
   tree = (treenode_t *) shm_ptr;

   global_sense = (boolean*) (shm_ptr + sizeof(treenode_t)*(P/FANIN)*LogP);

   if((int)global_sense%128)
      ga_error("barrier_init: global sense not alligned ",(long)global_sense);

   *global_sense = True; /* everybody does it for safety reasons */

   myleaf = &tree[(vpid/fanin)*LogP];
   node = myleaf;
   myStartingPlace = &(node->childSoFar.parts[(vpid%fanin)]);
   for(i=0;i<LogP-1;i++) {
      node->level = i;
      node->childExpected[1].parts[vpid%fanin] = True;
      node->childSoFar.parts[vpid%fanin] = True;
      parentIndex = vpid/fanin;
      numLeft = (numLeft%fanin) ? numLeft/fanin + 1: numLeft/fanin;
      if (numLeft <= 1) {
            node->type = ROOT;
            break;
      }
      node->parent = &tree[(parentIndex/fanin)*LogP + i+1];
      node->myPlace = &((node->parent)->childSoFar.parts[parentIndex%fanin]);
      node = node->parent;
      vpid = parentIndex;
   }
}




void KSRbarrier()
{
    volatile treenode_t *node = myleaf;

    *myStartingPlace = local_sense;
    for(;;) {
         if (node->childSoFar.whole != node->childExpected[local_sense].whole) {
    	    while (*global_sense != local_sense);
    	    break;
    	 }
         if (node->type == ROOT) {
    	    if (*global_sense == local_sense) break;
    	    *global_sense = local_sense;
    	    break;
         }
    	 *node->myPlace = local_sense;
     	 node = node->parent;
    }
    local_sense ^= True;
}
