#ifndef _KSRTCGMSG_H
#define _KSRTCGMSG_H

#define KSR_NUM_HEADERS    256

#define KSR_NUM_SLOTS      256

#define KSR_SLOT_SIZE      16384

typedef __align128 unsigned char subpage[128];

/*
 * Message slot type
 */

typedef struct message_slot_t
{
    __align128 struct message_slot_t *next;
    char data[KSR_SLOT_SIZE];
} message_slot_t;

/*
 * List of message slots
 */

typedef struct
{
    __align128 message_slot_t *list;
} message_slot_list_t;

/*
 * Message header type
 */

typedef struct message_hdr_t
{
    __align128 long type;
    long from;
    long length;
    struct message_hdr_t *next;
    struct message_hdr_t *prev;
    message_slot_t *slot;
    char *dataptr;

    /*
     * This macro should give the amount of unused space in this structure
     */

#define MSG_HDR_DATA (9 * sizeof(long))

    char data[MSG_HDR_DATA];
} message_hdr_t;

/*
 * List of message headers
 */

typedef struct
{
    __align128 message_hdr_t *list;
} message_hdr_list_t;

/*
 * Define constants for buffer space
 */

#define KSR_SHMEM_BUF_SIZE KSR_NUM_SLOTS * sizeof(message_slot_t) + \
                           sizeof(message_slot_list_t) + \
                           KSR_NUM_HEADERS * sizeof(message_hdr_t) + \
                           sizeof(message_hdr_list_t) + \
                           sizeof(message_hdr_list_t)
#endif

