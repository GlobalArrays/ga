#include <stdio.h>
#include <pthread.h>
#include "sndrcv.h"
#include "sndrcvP.h"

#include "ksr.h"

#define LOCK(x)         _gspwt(x)
#define UNLOCK(x)       _rsp(x)

#define MINIMUM(x,y)    ((x)<(y)?(x):(y))
#define UNALIGNED(x)    (((unsigned long) (x)) % sizeof(long))

/*
 * Pool of message slots
 */

extern message_slot_t (*SR_msg_slot)[][KSR_NUM_SLOTS];

/*
 * Pool of free message slots
 */

extern message_slot_list_t (*SR_free_msg_slots)[];

/*
 * Pool of message headers
 */

extern message_hdr_t (*SR_msg_header)[][KSR_NUM_HEADERS];

/*
 * Pool of free message headers
 */

extern message_hdr_list_t (*SR_free_msg_headers)[];

/*
 * List of received message headers for each thread
 */

extern message_hdr_list_t (*SR_received_msg_headers)[];

void KSR_BindProcess();

void KSR_MapBufferSpace();

void KSR_InitBuffer();

long KSR_MatchMessage();

void KSR_rcv_local();

void KSR_snd_local();

/*
 * Pool of message slots
 */

static message_slot_t (*SR_msg_slot)[][KSR_NUM_SLOTS];

/*
 * Pool of free message slots
 */

static message_slot_list_t (*SR_free_msg_slots)[];

/*
 * Pool of message headers
 */

static message_hdr_t (*SR_msg_header)[][KSR_NUM_HEADERS];

/*
 * Pool of free message headers
 */

static message_hdr_list_t (*SR_free_msg_headers)[];

/*
 * List of received message headers for each thread
 */

static message_hdr_list_t (*SR_received_msg_headers)[];


void KSR_BindProcess(me)
     long me;
/*
  KSR_BindProcess -- Bind this process to a KSR processor
*/
{
  long proc;

  if (DEBUG_) 
    {printf("KSRBP: me = %ld\n", me); fflush(stdout);}
  
  /* Initialize processor set */
  (void) psm_init();
  
  if (DEBUG_)
    {printf("KSRBP: me = %ld psm_init done\n", me); fflush(stdout);}
  
  /* Bind myself to a processor */
  proc = psm_bind(pthread_self(), me);
  
  if (DEBUG_)
    {printf("%2ld: Bound to processor %ld\n", NODEID_(), proc); fflush(stdout);}
}

/*
 * KSR_MapBufferSpace -- Layout the buffer space onto shared memory
 */

void KSR_MapBufferSpace(masterid, nslave)
    long masterid;
    long nslave;
{
    /* Map the buffer space data structures onto the shared memory */
    SR_msg_slot = (message_slot_t (*)[][KSR_NUM_SLOTS]) SR_proc_info[masterid].shmem;
    SR_free_msg_slots = (message_slot_list_t (*)[])
        ((char *) SR_msg_slot + nslave * KSR_NUM_SLOTS * sizeof(message_slot_t));
    SR_msg_header = (message_hdr_t (*)[][KSR_NUM_HEADERS])
        ((char *) SR_free_msg_slots + nslave * sizeof(message_slot_list_t));
    SR_free_msg_headers = (message_hdr_list_t (*)[])
        ((char *) SR_msg_header + nslave * KSR_NUM_HEADERS * sizeof(message_hdr_t));
    SR_received_msg_headers = (message_hdr_list_t (*)[])
        ((char *) SR_free_msg_headers + nslave * sizeof(message_hdr_list_t));
}

/*
 * KSR_InitBufferSpace -- Initialize header and slot free lists
 */

void KSR_InitBufferSpace()
{
    long i;
    long local_me = SR_proc_info[NODEID_()].slaveid;
    message_hdr_t *curr_hdr;
    message_slot_t *curr_slot;

    /* Initialize header free list */
    (*SR_free_msg_headers)[local_me].list = &(*SR_msg_header)[local_me][0];
    for (i = 0, curr_hdr = &(*SR_msg_header)[local_me][0]; i < KSR_NUM_HEADERS-1; i++, curr_hdr++)
        curr_hdr->next = curr_hdr + 1;
    curr_hdr->next = NULL;
    
    /* Initialize slot free list */
    (*SR_free_msg_slots)[local_me].list = &(*SR_msg_slot)[local_me][0];
    for (i = 0, curr_slot = &(*SR_msg_slot)[local_me][0]; i < KSR_NUM_SLOTS-1; i++, curr_slot++)
        curr_slot->next = curr_slot + 1;
    curr_slot->next = NULL;
}

/*
 * KSR_MatchMessage -- Determine if a message of given type has been received
 */

long KSR_MatchMessage(next_node, me, type)
    long next_node;
    long me;
    long type;
{
    long local_me = SR_proc_info[me].slaveid;
    long local_next_node = SR_proc_info[next_node].slaveid;
    long found;
    message_hdr_t *curr;

    /* Lock the received message list for this process */
    LOCK(&(*SR_received_msg_headers)[local_me].list);

    /* Search for a message of the given type */
    curr = (*SR_received_msg_headers)[local_me].list;
    for (found = FALSE; !found && curr != NULL; curr = curr->next)
        if (curr->from == local_next_node && curr->type == type)
            found = TRUE;

    /* Release the received message list */
    UNLOCK(&(*SR_received_msg_headers)[local_me].list);

    return(found);
}

 /*
  * Macro to copy data
  */
 
 #define Copy(src, dest, n) \
 { \
     register void *_s_ = (src); \
     register void *_d_ = (dest); \
     register long _length_ = (long) (n); \
     register long _i_, _l_; \
     register long _r00_, _r01_, _r02_, _r03_; \
     register long _r04_, _r05_, _r06_, _r07_; \
     register long _r08_, _r09_, _r10_, _r11_; \
     register long _r12_, _r13_, _r14_, _r15_; \
  \
     if (UNALIGNED(_s_) || UNALIGNED(_d_)) \
     { \
         _l_ = _length_ / 16; \
         for (_i_ = 0; _i_ < _l_; _i_++, _length_ -= 16) \
         { \
             _r00_ = *((char *)_s_ +  0); \
             _r01_ = *((char *)_s_ +  1); \
             _r02_ = *((char *)_s_ +  2); \
             _r03_ = *((char *)_s_ +  3); \
             _r04_ = *((char *)_s_ +  4); \
             _r05_ = *((char *)_s_ +  5); \
             _r06_ = *((char *)_s_ +  6); \
             _r07_ = *((char *)_s_ +  7); \
             _r08_ = *((char *)_s_ +  8); \
             _r09_ = *((char *)_s_ +  9); \
             _r10_ = *((char *)_s_ + 10); \
             _r11_ = *((char *)_s_ + 11); \
             _r12_ = *((char *)_s_ + 12); \
             _r13_ = *((char *)_s_ + 13); \
             _r14_ = *((char *)_s_ + 14); \
             _r15_ = *((char *)_s_ + 15); \
             ((char *)_s_) += 16; \
             *((char *)_d_ +  0) = _r00_; \
             *((char *)_d_ +  1) = _r01_; \
             *((char *)_d_ +  2) = _r02_; \
             *((char *)_d_ +  3) = _r03_; \
             *((char *)_d_ +  4) = _r04_; \
             *((char *)_d_ +  5) = _r05_; \
             *((char *)_d_ +  6) = _r06_; \
             *((char *)_d_ +  7) = _r07_; \
             *((char *)_d_ +  8) = _r08_; \
             *((char *)_d_ +  9) = _r09_; \
             *((char *)_d_ + 10) = _r10_; \
             *((char *)_d_ + 11) = _r11_; \
             *((char *)_d_ + 12) = _r12_; \
             *((char *)_d_ + 13) = _r13_; \
             *((char *)_d_ + 14) = _r14_; \
             *((char *)_d_ + 15) = _r15_; \
             ((char *)_d_) += 16; \
         } \
         for (_i_ = 0; _i_ < _length_; _i_++) \
             *((char *)_d_)++ = *((char *)_s_)++; \
     } \
     else \
     { \
         _l_ = _length_ / sizeof(subpage); \
         for (_i_ = 0; _i_ < _l_; _i_++, _length_ -= sizeof(subpage)) \
         { \
             _r00_ = *((long *)_s_ +  0); \
             _r01_ = *((long *)_s_ +  1); \
             _r02_ = *((long *)_s_ +  2); \
             _r03_ = *((long *)_s_ +  3); \
             _r04_ = *((long *)_s_ +  4); \
             _r05_ = *((long *)_s_ +  5); \
             _r06_ = *((long *)_s_ +  6); \
             _r07_ = *((long *)_s_ +  7); \
             _r08_ = *((long *)_s_ +  8); \
             _r09_ = *((long *)_s_ +  9); \
             _r10_ = *((long *)_s_ + 10); \
             _r11_ = *((long *)_s_ + 11); \
             _r12_ = *((long *)_s_ + 12); \
             _r13_ = *((long *)_s_ + 13); \
             _r14_ = *((long *)_s_ + 14); \
             _r15_ = *((long *)_s_ + 15); \
             ((char *)_s_) += sizeof(subpage); \
             *((long *)_d_ +  0) = _r00_; \
             *((long *)_d_ +  1) = _r01_; \
             *((long *)_d_ +  2) = _r02_; \
             *((long *)_d_ +  3) = _r03_; \
             *((long *)_d_ +  4) = _r04_; \
             *((long *)_d_ +  5) = _r05_; \
             *((long *)_d_ +  6) = _r06_; \
             *((long *)_d_ +  7) = _r07_; \
             *((long *)_d_ +  8) = _r08_; \
             *((long *)_d_ +  9) = _r09_; \
             *((long *)_d_ + 10) = _r10_; \
             *((long *)_d_ + 11) = _r11_; \
             *((long *)_d_ + 12) = _r12_; \
             *((long *)_d_ + 13) = _r13_; \
             *((long *)_d_ + 14) = _r14_; \
             *((long *)_d_ + 15) = _r15_; \
             ((char *)_d_) += sizeof(subpage); \
         } \
         _l_ = _length_ / sizeof(long); \
         for (_i_ = 0; _i_ < _l_; _i_++, _length_ -= sizeof(long)) \
             *((long *)_d_)++ = *((long *)_s_)++; \
         for (_i_ = 0; _i_ < _length_; _i_++) \
             *((char *)_d_)++ = *((char *)_s_)++; \
     } \
 }
 
 /*
  * KSR_rcv_local -- Receive a message via shared memory
  */
 
 void KSR_rcv_local(type, buf, lenbuf, lenmes, nodeselect, nodefrom)
     long *type;
     char *buf;
     long *lenbuf;
     long *lenmes;
     long *nodeselect;
     long *nodefrom;
 {
     message_hdr_t *header, *curr, *last, *new_last;
     message_slot_t *slot, *old;
     long i, length;
     long me = NODEID_();
     long local_me = SR_proc_info[me].slaveid;
     long local_node = SR_proc_info[*nodeselect].slaveid;
 
     if (DEBUG_)
     {
 	printf("%2ld rcv_local type=%d buf=%016x lenbuf=%d node=%d\n",
 	       me, *type, buf, *lenbuf, *nodeselect);
 	(void) fflush(stdout);
     }
 
     /* Lock the received msg header list */
     LOCK(&(*SR_received_msg_headers)[local_me].list);
 
     /* Get a copy of the recevied msg header list pointer */
     last = curr = (*SR_received_msg_headers)[local_me].list;
 
     /* Unlock the received msg header list */
     UNLOCK(&(*SR_received_msg_headers)[local_me].list);
 
     /* Loop until we find the message */
     while (1)
     {
 	/* Go to the end of the received list */
 	if (curr != NULL)
 	{
 	    while (curr->next != NULL)
 	    {
 		/* Establish backlink pointer */
 		curr->next->prev = curr;
 		curr = curr->next;
 	    }
 	}
 
 	/* Search back for the appropriate message */
 	while (curr != last)
 	{
 	    /* Check this message */
 	    if (curr->type == (long) *type && curr->from == local_node)
 	    {
 		/* Remove message from list */
 		curr->prev->next = curr->next;
 
 		/* Return this message */
 		header = curr;
 		goto ReceivedHeader;
 	    }
 	    curr = curr->prev;
 	}
 
 	/* Reached what was the last message received */
 	if (curr != NULL && curr->type == (long) *type && curr->from == local_node)
 	{
 	    /* Lock the received msg header list */
 	    LOCK(&(*SR_received_msg_headers)[local_me].list);
 
 	    /* Check if new messages have arrived */
 	    if (last == (*SR_received_msg_headers)[local_me].list)
 	    {
 		/* No new messages have arrived, update list pointer */
 		(*SR_received_msg_headers)[local_me].list = curr->next;
 	    }
 	    else
 	    {
 		/* Find the oldest new message */
 		curr = (*SR_received_msg_headers)[local_me].list;
 		while (curr->next != last)
 		    curr = curr->next;
 
 		/* Remove the original last message */
 		curr->next = last->next;
 
 		/* Make this the current message */
 		curr = last;
 	    }
 
 	    /* Unlock the received msg header list */
 	    UNLOCK(&(*SR_received_msg_headers)[local_me].list);
 
 	    /* Return the header */
 	    header = curr;
 	    goto ReceivedHeader;
 	}
 	else
 	{
 	    /* Wait for more messages to arrive */
 	    do
 	    {
 		/* Delay */
 		for (i=0; i<100; i++);
 
 		/* Lock the recieved msg header list */
 		LOCK(&(*SR_received_msg_headers)[local_me].list);
 
 		/* Get a copy of the list pointer */
 		new_last = (*SR_received_msg_headers)[local_me].list;
 
 		/* Unlock the recieved msg header list */
 		UNLOCK(&(*SR_received_msg_headers)[local_me].list);
 	    } while (last == new_last);
 
 	    /* Start over */
 	    last = curr = new_last;
 	}
     }
 
 ReceivedHeader:
 
     /* Check length of message */
     if (header->length <= MSG_HDR_DATA)
     {
 	/* Copy the data over */
 	Copy(header->data, buf, header->length);
     }
     else
     {
 	/* Wait for the first slot to arrive */
 	slot = header->slot;
 
 	/* Loop until the slot becomes valid */
 	LOCK(slot);
 	UNLOCK(slot);
 
 	/* Receive each slot */
 	for (i = 0, length = header->length; length > 0; i++, length -= KSR_SLOT_SIZE)
 	{
 	    /* Copy the next slot */
 	    Copy(slot->data, &buf[i * KSR_SLOT_SIZE], MINIMUM(KSR_SLOT_SIZE, length));
 
 	    /* Save pointer to old slot */
 	    old = slot;
 
 	    /* Get next slot if more slots to copy */
 	    if (length > KSR_SLOT_SIZE)
 	    {
 		/* Wait for next slot */
 		slot = slot->next;
 
 		/* Loop until slot becomes valid */
 		LOCK(slot);
 		UNLOCK(slot);
 	    }
 	    else
 		slot = NULL;
 
 	    /* Lock the free msg slot list */
 	    LOCK(&(*SR_free_msg_slots)[local_me].list);
 
 	    /* Put the old slot on the free msg slot list */
 	    old->next = (*SR_free_msg_slots)[local_me].list;
 	    (*SR_free_msg_slots)[local_me].list = old;
 
 	    /* Unlock the free msg slot list */
 	    UNLOCK(&(*SR_free_msg_slots)[local_me].list);
 	}
     }
 
     /* Lock the free msg header list */
     LOCK(&(*SR_free_msg_headers)[local_me].list);
 
     /* Free the header */
     header->next = (*SR_free_msg_headers)[local_me].list;
     (*SR_free_msg_headers)[local_me].list = header;
 
     /* Unlock the free msg header list */
     UNLOCK(&(*SR_free_msg_headers)[local_me].list);
 
     /* Update results */
     *lenmes = *lenbuf;
     *nodefrom = *nodeselect;
 }
 
 /*
  * KSR_snd_local -- Send a message via shared memory
  */
 
 void KSR_snd_local(type, buf, lenbuf, node)
     long *type;
     char *buf;
     long *lenbuf;
     long *node;
 {
     message_hdr_t *header;
     message_slot_t *slot;
     long i, length, found, cell;
     long me = NODEID_();
     long local_me = SR_proc_info[me].slaveid;
     long threads = SR_clus_info[SR_proc_info[me].clusid].nslave;
     long local_node = SR_proc_info[*node].slaveid;
 
     if (DEBUG_)
     {
 	printf("%2ld snd_local type=%d buf=%016x lenbuf=%d node=%d\n",
 	       me, *type, buf, *lenbuf, *node);
 	(void) fflush(stdout);
     }
 
     /*
      * Search for a free message header starting with my own free list and
      * continuing with other cell's lists
      */
     for (cell = local_me, found = FALSE; !found; cell = (cell + 1) % threads)
     {
 	/* Lock this cell's free msg header list */
 	LOCK(&(*SR_free_msg_headers)[cell].list);
 
 	/* Check if there are any free headers here */
 	if ((*SR_free_msg_headers)[cell].list != NULL)
 	{
 	    /* Take the first header off of the list */
 	    found = TRUE;
 	    header = (*SR_free_msg_headers)[cell].list;
 	    (*SR_free_msg_headers)[cell].list = header->next;
 	}
 
 	/* Unlock this cell's free msg header list */
 	UNLOCK(&(*SR_free_msg_headers)[cell].list);
     }
 
     /* Must have found something */
     if (!found)
 	Error("snd_local: Out of message headers", (int) 0);
 
     /* Fill in the header */
     header->type = (long) *type;
     header->from = (long) me;
     header->length = (long) *lenbuf;
     header->next = NULL;
     header->prev = NULL;
 
     /* Check size of message to send */
     if (header->length <= MSG_HDR_DATA)
     {
 	/* No slots needed */
 	header->slot = NULL;
 
 	/* Small message: Send data in header */
 	Copy(buf, header->data, header->length);
 
 	/* Get received msg list of recipient in atomic state */
 	LOCK(&(*SR_received_msg_headers)[local_node].list);
 
 	/* Add this header to the list */
 	header->next = (*SR_received_msg_headers)[local_node].list;
 	(*SR_received_msg_headers)[local_node].list = header;
 
 	/* Release atomic state of received msg list of recipient */
 	UNLOCK(&(*SR_received_msg_headers)[local_node].list);
     }
     else
     {
 	/*
 	 * Search for a free message slot starting with my own free list
 	 * and continuing with other cell's lists
 	 */
 	for (cell = local_me, found = FALSE; !found; cell = (cell + 1) % threads)
 	{
 	    /* Lock this cell's free msg slot list */
 	    LOCK(&(*SR_free_msg_slots)[cell].list);
 
 	    /* Check if there are any free slots here */
 	    if ((*SR_free_msg_slots)[cell].list != NULL)
 	    {
 		/* Take the first slot off of the list */
 		found = TRUE;
 		slot = (*SR_free_msg_slots)[cell].list;
 		(*SR_free_msg_slots)[cell].list = slot->next;
 	    }
 
 	    /* Unlock this cell's free msg slot list */
 	    UNLOCK(&(*SR_free_msg_slots)[cell].list);
 	}
 
 	/* Must have found something */
 	if (!found)
 	    Error("snd_local: Out of message slots", (int) 0);
 
 	/* This is the first slot of the message */
 	header->slot = slot;
 
 	/* Lock this slot while data is copied in */
 	LOCK(slot);
 
 	/* Lock the recipients received msg header list */
 	LOCK(&(*SR_received_msg_headers)[local_node].list);
 
 	/* Add this header to the list */
 	header->next = (*SR_received_msg_headers)[local_node].list;
 	(*SR_received_msg_headers)[local_node].list = header;
 
 	/* Release the recipients recevied message header list */
 	UNLOCK(&(*SR_received_msg_headers)[local_node].list);
 
 	/* Send the data with slots */
 	for (i = 0, length = header->length; length > 0; i++, length -= KSR_SLOT_SIZE)
 	{
 	    /* Fill in the slot header */
 	    if (length > KSR_SLOT_SIZE)
 	    {
 		/*
 		 * Search for another free message slot starting with my own
 		 * free list and continuing with other cell's lists
 		 */
 		for (cell = local_me, found = FALSE; !found; cell = (cell + 1) % threads)
 		{
 		    /* Lock the free msg slot list */
 		    LOCK(&(*SR_free_msg_slots)[cell].list);
 
 		    /* Check if there are any free slots here */
 		    if ((*SR_free_msg_slots)[cell].list != NULL)
 		    {
 			/* Take the first slot off of the list */
 			found = TRUE;
 			slot->next = (*SR_free_msg_slots)[cell].list;
 			(*SR_free_msg_slots)[cell].list = slot->next->next;
 		    }
 
 		    /* Unlock the free msg slot list */
 		    UNLOCK(&(*SR_free_msg_slots)[cell].list);
 		}
 
 		/* Must have found something */
 		if (!found)
 		    Error("snd_local: Out of message slots", (int) 0);
 
 		/* Lock the new slot until the data is copied into it */
 		LOCK(slot->next);
 	    }
 	    else
 		slot->next = NULL;
 
 	    /* Copy the data to the slot */
 	    Copy(&buf[i * KSR_SLOT_SIZE], slot->data, MINIMUM(KSR_SLOT_SIZE, length));
 
 	    /* Unlock the sent slot */
 	    UNLOCK(slot);
 
 	    /* Update slot pointer */
 	    slot = slot->next;
 	}
     }
 }
 
