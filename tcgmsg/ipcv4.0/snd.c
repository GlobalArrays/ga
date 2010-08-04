#if HAVE_CONFIG_H
#   include "config.h"
#endif

#if HAVE_STDIO_H
#   include <stdio.h>
#endif
#if HAVE_STRINGS_H
#   include <strings.h>
#endif
#if HAVE_STRING_H
#   include <string.h>
#endif
#if HAVE_SYS_SELECT_H
#   include <sys/select.h>
#endif
#if HAVE_SYS_TYPES_H
#   include <sys/types.h>
#endif
#if HAVE_SYS_TIME_H
#   include <sys/time.h>
#endif
#if HAVE_MEMORY_H
#    include <memory.h>
#endif

extern void abort(void);

#ifdef USE_VAMPIR
#   include "tcgmsg_vampir.h"
#endif

extern void Error();

#include "sndrcv.h"
#include "sndrcvP.h"
#include "sockets.h"

#if HAVE_RPC_XDR_H
#   include "xdrstuff.h"
#endif

#include "sema.h"
#include "shmem.h"

#ifdef EVENTLOG
#   include "evlog.h"
#endif

extern void ListenOnSock(int sock);
extern int AcceptConnection(int sock);


/**
 * Print out the SR_proc_info structure array for this process.
 */
void PrintProcInfo()
{
    Integer i;

    (void) fprintf(stderr,"Process info for node %ld: \n",NODEID_());

    for (i=0; i<NNODES_(); i++) {
        (void) fprintf(stderr,"[%ld] = {\n  clusid = %-8ld\n  slaveid = %-8ld\n  local = %-8ld\n  sock = %-8d\n  shmem = %-8p\n  shmem_size = %-8ld\n  shmem_id = %-8ld\n  buffer = %-8p\n  buflen = %-8ld\n  header = %-8p\n  semid = %-8ld\n  sem_read = %-8ld\n  sem_written = %-8ld\n  n_rcv = %-8ld\n  nb_rcv = %-8ld\n  t_rcv = %-8ld\n  n_snd = %-8ld\n  nb_snd = %-8ld\n  t_snd = %-8ld\n  peeked = %-8ld}\n",
                       i,
                       SR_proc_info[i].clusid,
                       SR_proc_info[i].slaveid,
                       SR_proc_info[i].local,
                       SR_proc_info[i].sock,
                       SR_proc_info[i].shmem,
                       SR_proc_info[i].shmem_size,
                       SR_proc_info[i].shmem_id,
                       SR_proc_info[i].buffer,
                       SR_proc_info[i].buflen,
                       SR_proc_info[i].header,
                       SR_proc_info[i].semid,
                       SR_proc_info[i].sem_read,
                       SR_proc_info[i].sem_written,
                       SR_proc_info[i].n_rcv,
                       (Integer)  SR_proc_info[i].nb_rcv,
                       (Integer)  SR_proc_info[i].t_rcv,
                       SR_proc_info[i].n_snd,
                       (Integer)  SR_proc_info[i].nb_snd,
                       (Integer)  SR_proc_info[i].t_snd,
                       SR_proc_info[i].peeked);
    }

    (void) fflush(stderr);
}


/**
 * Print out the contents of a message header aInteger with info message.
 */
static void PrintMessageHeader(char *info, MessageHeader *header)
{
    (void) printf("%2ld:%s: type=%ld, from=%ld, to=%ld, len=%ld, tag=%ld\n",
                  NODEID_(),info, header->type, header->nodefrom, 
                  header->nodeto, header->length, header->tag);
    (void) fflush(stdout);
}


static int DummyRoutine()
{
    int i, sum=0;
    for(i=0; i<10; i++) {
        sum += i;
    }
    return sum;
}


static Integer flag(Integer *p)
{
    return *p;
}


/**
 * Wait until the value pointed to by p equals value.
 * Since *ptr is volatile but cannot usually declare this
 * include another level of procedure call to protect
 * against compiler optimization.
 */
static void Await(Integer *p, Integer value)
{
    int nspin = 0;
    if (DEBUG_) {
        printf("%2ld: Await p=%p, value=%ld\n", NODEID_(), p, value);
        fflush(stdout);
    }

    for (; flag(p) != value; nspin++) {
        if (nspin < 100) {
            (void) DummyRoutine();
        }
        else {
            USleep((Integer) 10000);
        }
    }
}


static void rcv_local(
        Integer *type,
        char *buf,
        Integer *lenbuf,
        Integer *lenmes,
        Integer *nodeselect,
        Integer *nodefrom)
{
    Integer me = NODEID_();
    Integer node = *nodeselect;
    MessageHeader *head = SR_proc_info[node].header;
    Integer buflen = SR_proc_info[node].buflen;
    char *buffer = SR_proc_info[node].buffer;
    Integer nodeto, len;
    Integer semid = SR_proc_info[node].semid;
    Integer sem_read = SR_proc_info[node].sem_read;
    Integer sem_written = SR_proc_info[node].sem_written;
    Integer semid_to = SR_proc_info[me].semid;
    Integer sem_pend = SR_proc_info[me].sem_pend;

    /* Error checking */

    if ( (buffer == (char *) NULL) || (head == (MessageHeader *) NULL) ) {
        Error("rcv_local: invalid shared memory", (Integer) node);
    }

    if ( (semid < 0)
            || (sem_read < 0)
            || (sem_written < 0)
            || (semid_to < 0)
            || (sem_pend < 0) ) {
        Error("rcv_local: invalid semaphore set", (Integer) node);
    }

    SemWait(semid_to, sem_pend);
    Await(&head->nodeto, me);    /* Still have this possible spin */
    SemWait(semid, sem_written);

    /* Now have a message for me ... check the header info and
       copy the first block of the message. */

    if (DEBUG_) {
        PrintMessageHeader("rcv_local ",head);
    }

    nodeto = head->nodeto;    /* Always me ... history here */
    head->nodeto = -1;

    *nodefrom = head->nodefrom;

    if (head->type != *type) {
        PrintMessageHeader("rcv_local ",head);
        /* printf("rcv_local: type mismatch ... strong typing enforced\n"); */
        /* abort(); */
        Error("rcv_local: type mismatch ... strong typing enforced",
                (Integer) *type);
    }

    *lenmes = len = head->length;

    if ( *lenmes > *lenbuf ) {
        Error("rcv_local: message too Integer for buffer", (Integer) *lenmes);
    }
    if (nodeto != me) {
        Error("rcv_local: message meant for someone else?", (Integer) nodeto);
    }

    if (len) {
        (void) memcpy(buf, buffer, (len > buflen) ? buflen : len);
    }

    SemPost(semid, sem_read);

    len -= buflen;
    buf += buflen;

    /* Copy the remainder of the message */

    while (len > 0) {
        SemWait(semid, sem_written);
        (void) memcpy(buf, buffer, (len > buflen) ? buflen : len);
        SemPost(semid, sem_read);
        len -= buflen;
        buf += buflen;
    }
}


#ifdef SHMEM
static void snd_local(
        Integer *type,
        char *buf,
        Integer *lenbuf,
        Integer *node)
{
    Integer me = NODEID_();
    MessageHeader *head = SR_proc_info[me].header;
    Integer buflen = SR_proc_info[me].buflen;
    Integer len = *lenbuf;
    char *buffer = SR_proc_info[me].buffer;
    Integer tag = SR_proc_info[*node].n_snd;
    Integer semid = SR_proc_info[me].semid;
    Integer sem_read = SR_proc_info[me].sem_read;
    Integer sem_written = SR_proc_info[me].sem_written;
    Integer semid_to = SR_proc_info[*node].semid;
    Integer sem_pend = SR_proc_info[*node].sem_pend;

    /* Error checking */

    if ( (buffer == (char *) NULL) || (head == (MessageHeader *) NULL) ) {
        Error("snd_local: invalid shared memory", (Integer) *node);
    }

    if ( (semid < 0)
            || (semid_to < 0)
            || (sem_read < 0)
            || (sem_written < 0) ) {
        Error("snd_local: invalid semaphore set", (Integer) *node);
    }

    /* Check that final segment of last message has been consumed */

    SemWait(semid, sem_read);

    /* Fill in message header */

    head->nodefrom = (char) me;
    head->type = *type;
    head->length = *lenbuf;
    head->tag = tag;
    head->nodeto = (char) *node;

    if (DEBUG_) {
        PrintMessageHeader("snd_local ",head);
        (void) fflush(stdout);
    }

    /* Copy the first piece of the message so that send aInteger with
       header to minimize use of semaphores. Also need to send header
       even for messages of zero length */

    if (len) {
        (void) memcpy(buffer, buf, (len > buflen) ? buflen : len);
    }

    SemPost(semid, sem_written);
    SemPost(semid_to, sem_pend);

    len -= buflen;
    buf += buflen;

    while (len > 0) {
        SemWait(semid, sem_read);
        (void) memcpy(buffer, buf, (len > buflen) ? buflen : len);
        SemPost(semid, sem_written);
        len -= buflen;
        buf += buflen;
    }
}    


/**
 * synchronous send to remote process
 *
 * Integer *type     = user defined integer message type (input)
 * char *buf      = data buffer (input)
 * Integer *lenbuf   = length of buffer in bytes (input)
 * Integer *node     = node to send to (input)
 *
 * for zero length messages only the header is sent
 */
static void snd_remote(
    Integer *type,
    char *buf,
    Integer *lenbuf,
    Integer *node)
{
#define SHORT_MSG_BUF_SIZE (2048 + 40)
    static char fudge[SHORT_MSG_BUF_SIZE]; 
    MessageHeader header;
    Integer me=NODEID_();
    int sock=SR_proc_info[*node].sock;
    Integer len;
#ifdef SOCK_FULL_SYNC
    char sync=0;
#endif

    if ( sock < 0 )
        Error("snd_remote: sending to process without socket", (Integer) *node);

    header.nodefrom = me;
    header.nodeto   = *node;
    header.type     = *type;
    header.length   = *lenbuf;
    header.tag      = SR_proc_info[*node].n_snd;

    /* header.length is the no. of items if XDR is used or just the
       number of bytes */

#if HAVE_RPC_XDR_H
    if ( *type & MSGDBL )
        header.length = *lenbuf / sizeof(DoublePrecision);
    else if ( *type & MSGINT ) 
        header.length = *lenbuf / sizeof(Integer);
    else if ( *type & MSGCHR )
        header.length = *lenbuf / sizeof(char);
    else
        header.length = *lenbuf;
#else
    header.length = *lenbuf;
#endif

    if (DEBUG_)
        PrintMessageHeader("snd_remote",&header);

#ifndef HAVE_RPC_XDR_H
    /* Combine header and messages less than a certain size to avoid
     * performance problem on (older?) linuxes */
    if ((*lenbuf + sizeof(header)) <= sizeof(fudge)) {
        memcpy(fudge,(char *) &header, sizeof(header));
        memcpy(fudge+sizeof(header), buf, *lenbuf);
        if ( (len = WriteToSocket(sock, fudge, sizeof(header)+*lenbuf)) != 
                (sizeof(header)+*lenbuf))
            Error("snd_remote: writing message to socket",
                    (Integer) (len+100000*(sock + 1000* *node)));
        return;
    }
#endif

#if HAVE_RPC_XDR_H
    (void) WriteXdrLong(sock, (Integer *) &header, 
                        (Integer) (sizeof(header)/sizeof(Integer)));
#else
    if ( (len = WriteToSocket(sock, (char *) &header, (Integer) sizeof(header)))
            != sizeof(header) )
        Error("snd_remote: writing header to socket", len);
#endif

    if (*lenbuf)  {
#if HAVE_RPC_XDR_H
        if ( *type & MSGDBL )
            (void) WriteXdrDouble(sock, (DoublePrecision *) buf, header.length);
        else if ( *type & MSGINT )
            (void) WriteXdrLong(sock, (Integer *) buf, header.length);
        else if ( *type & MSGCHR )
            (void) WriteXdrChar(sock, (char *) buf, header.length);
        else if ( (len = WriteToSocket(sock, buf, header.length)) != 
                header.length)
            Error("snd_remote: writing message to socket",
                    (Integer) (len+100000*(sock + 1000* *node)));
#else
        if ( (len = WriteToSocket(sock, buf, header.length)) != 
                header.length)
            Error("snd_remote: writing message to socket",
                    (Integer) (len+100000*(sock + 1000* *node)));
#endif
    }

#ifdef SOCK_FULL_SYNC
    /* this read (and write in rcv_remote) of an acknowledgment 
       forces synchronous */

    if ( ReadFromSocket(sock, &sync, (Integer) 1) != 1)
        Error("snd_remote: reading acknowledgement",
                (Integer) (len+100000*(sock + 1000* *node)));
#endif
}
#endif /* SHMEM */


/**
 * mostly syncrhonous send
 *
 * Integer *type     = user defined integer message type (input)
 * char *buf     = data buffer (input)
 * Integer *lenbuf   = length of buffer in bytes (input)
 * Integer *node     = node to send to (input)
 * Integer *sync    = flag for sync/async ... IGNORED
 *
 * for zero length messages only the header is sent
 */
void SND_(
        Integer *type,
        void *buf,
        Integer *lenbuf,
        Integer *node,
        Integer *sync)
{
    Integer me=NODEID_();
    Integer nproc=NNODES_();
#ifdef TCGMSG_TIMINGS
    DoublePrecision start;
#endif
#ifdef USE_VAMPIR
    vampir_begin(TCGMSG_SND,__FILE__,__LINE__);
    vampir_send(me,*node,*lenbuf,*type);
#endif

    /* Error checking */

    if (*node == me)
        Error("SND_: cannot send message to self", (Integer) me);

    if ( (*node < 0) || (*node > nproc) )
        Error("SND_: out of range node requested", (Integer) *node);

    if ( (*lenbuf < 0) || (*lenbuf > BIG_MESSAGE) )
        Error("SND_: message length out of range", (Integer) *lenbuf);

#ifdef EVENTLOG
    evlog(EVKEY_BEGIN,     EVENT_SND,
            EVKEY_MSG_LEN,  (int) *lenbuf,
            EVKEY_MSG_FROM, (int)  me,
            EVKEY_MSG_TO,   (int) *node,
            EVKEY_MSG_TYPE, (int) *type,
            EVKEY_MSG_SYNC, (int) *sync,
            EVKEY_LAST_ARG);
#endif

    /* Send via shared memory or sockets */

#ifdef TCGMSG_TIMINGS
    start = TCGTIME_();
#endif

#ifdef SHMEM
    if (SR_proc_info[*node].local){
        snd_local(type, buf, lenbuf, node);
    } else {
        snd_remote(type, buf, lenbuf, node);
    }
#endif

    /* Collect statistics */

    SR_proc_info[*node].n_snd += 1;
    SR_proc_info[*node].nb_snd += *lenbuf;

#ifdef TCGMSG_TIMINGS
    SR_proc_info[*node].t_snd += TCGTIME_() - start;
#endif

#ifdef EVENTLOG
    evlog(EVKEY_END, EVENT_SND, EVKEY_LAST_ARG);
#endif
#ifdef USE_VAMPIR
    vampir_end(TCGMSG_SND,__FILE__,__LINE__);
#endif
}    

static Integer MatchMessage(header, me, type)
    MessageHeader *header;
    Integer me, type;
    /*
       Wrapper round check on if header is to me and of required
       type so that compiler does not optimize out fetching
       header info from shared memory.
       */
{
    return (Integer) ((header->nodeto == me) && (header->type == type));
}

static Integer NextReadyNode(type)
    Integer type;
    /*
       Select a node from which input is pending ... also match the
       desired type.

       next_node is maintained as the last node that NextReadyNode chose
       plus one modulo NNODES_(). This aids in ensuring fairness.

       First use select to get info about the sockets and then loop
       through processes looking either at the bit in the fd_set for
       the socket (remote process) or the message header in the shared
       memory buffer (local process).

       This may be an expensive operation but fairness seems important.

       If only sockets are in use, just block in select until data is
       available.  
       */
{
    static Integer  next_node = 0;

    Integer  nproc = NNODES_();
    Integer  me = NODEID_();
    int i, nspin = 0;

    if (!SR_using_shmem) {
        int list[MAX_PROCESS];
        int nready;
        nready = WaitForSockets(SR_nsock,SR_socks,list);
        if (nready == 0) 
            Error("NextReadyNode: nready = 0\n", 0);

        /* Insert here type checking logic ... not yet done */

        return SR_socks_proc[list[0]];
    }

    /* With both local and remote processes end up with a busy wait
       as no way to wait for both a semaphore and a socket.
       Moderate this slightly by having short timeout in select */

    while (1) {

        for(i=0; i<nproc; i++, next_node = (next_node + 1) % nproc) {

            if (next_node == me) {
                ;  /* can't receive from self */
            }
            else if (SR_proc_info[next_node].local) {
                /* Look for local message */

                if (MatchMessage(SR_proc_info[next_node].header, me, type))
                    break;
            }
            else if (SR_proc_info[next_node].sock >= 0) {
                /* Look for message over socket */

                int sock = SR_proc_info[next_node].sock;

                /* Have we already peeked at this socket? */

                if (SR_proc_info[next_node].peeked) {
                    if (SR_proc_info[next_node].head_peek.type == type)
                        break;
                }
                else if (PollSocket(sock)) {
                    /* Data is available ... let's peek at it */
#if HAVE_RPC_XDR_H
                    (void) ReadXdrLong(sock, 
                                       (Integer *) &SR_proc_info[next_node].head_peek,
                                       (Integer) (sizeof(MessageHeader)/sizeof(Integer)));
#else
                    if (ReadFromSocket(sock, 
                                (char *) &SR_proc_info[next_node].head_peek,
                                (Integer) sizeof(MessageHeader))
                            != sizeof(MessageHeader) )
                        Error("NextReadyNode: reading header from socket", next_node);
#endif
                    SR_proc_info[next_node].peeked = TRUE;
                    if (DEBUG_)
                        PrintMessageHeader("peeked_at ",
                                &SR_proc_info[next_node].head_peek);

                    if (SR_proc_info[next_node].head_peek.type == type)
                        break;
                }
            }
        }
        if (i < nproc)       /* If found a node skip out of the while loop */
            break;

        nspin++;         /* Compromise between low latency and low cpu use */
        if (nspin < 10)
            continue;
        else if (nspin < 100)
            USleep((Integer) 1000);
        else if (nspin < 600)
            USleep((Integer) 10000);
        else
            USleep((Integer) 100000);
    }

    i = next_node;
    next_node = (next_node + 1) % nproc;

    return (Integer) i;
}

Integer PROBE_(type, node)
    Integer *type, *node;
    /*
       Return 1/0 (TRUE/FALSE) if a message of the given type is available
       from the given node.  If the node is specified as -1, then all nodes
       will be examined.  Some attempt is made at ensuring fairness.

       First use select to get info about the sockets and then loop
       through processes looking either at the bit in the fd_set for
       the socket (remote process) or the message header in the shared
       memory buffer (local process).

       This may be an expensive operation but fairness seems important.
       */
{
    Integer  nproc = NNODES_();
    Integer  me = NODEID_();
    int i, proclo, prochi;

#ifdef USE_VAMPIR
    vampir_begin(TCGMSG_PROBE,__FILE__,__LINE__);
#endif

    if (*node == me)
        Error("PROBE_ : cannot recv message from self, msgtype=", *type);

    if (*node == -1) {        /* match anyone */
        proclo = 0;
        prochi = nproc-1;
    }
    else
        proclo = prochi = *node;

    for(i=proclo; i<=prochi; i++) {

        if (i == me) {
            ;  /* can't receive from self */
        }
        else if (SR_proc_info[i].local) {
            /* Look for local message */

            if (MatchMessage(SR_proc_info[i].header, me, *type))
                break;
        }
        else if (SR_proc_info[i].sock >= 0) {
            /* Look for message over socket */

            int sock = SR_proc_info[i].sock;

            /* Have we already peeked at this socket? */

            if (SR_proc_info[i].peeked) {
                if (SR_proc_info[i].head_peek.type == *type)
                    break;
            }
            else if (PollSocket(sock)) {
                /* Data is available ... let's peek at it */
#if HAVE_RPC_XDR_H
                (void) ReadXdrLong(sock, 
                                   (Integer *) &SR_proc_info[i].head_peek,
                                   (Integer) (sizeof(MessageHeader)/sizeof(Integer)));
#else
                if (ReadFromSocket(sock, 
                            (char *) &SR_proc_info[i].head_peek,
                            (Integer) sizeof(MessageHeader))
                        != sizeof(MessageHeader) )
                    Error("NextReadyNode: reading header from socket", (Integer) i);
#endif
                SR_proc_info[i].peeked = TRUE;
                if (DEBUG_)
                    PrintMessageHeader("peeked_at ",
                            &SR_proc_info[i].head_peek);

                if (SR_proc_info[i].head_peek.type == *type)
                    break;
            }
        }
    }

#ifdef USE_VAMPIR
    vampir_end(TCGMSG_PROBE,__FILE__,__LINE__);
#endif
    if (i <= prochi)
        return 1;
    else
        return 0;
}


/**
 * synchronous receive of data
 *
 * Integer *type        = user defined type of received message (input)
 * char *buf        = data buffer (output)
 * Integer *lenbuf      = length of buffer in bytes (input)
 * Integer *lenmes      = length of received message in bytes (output)
 * (exceeding receive buffer is hard error)
 * Integer *nodeselect  = node to receive from (input)
 * -1 implies that any pending message may be received
 *
 * Integer *nodefrom    = node message is received from (output)
 */
static void rcv_remote(type, buf, lenbuf, lenmes, nodeselect, nodefrom)
    Integer *type;
    char *buf;
    Integer *lenbuf;
    Integer *lenmes;
    Integer *nodeselect;
    Integer *nodefrom;
{
    Integer me = NODEID_();
    Integer node = *nodeselect;
    int sock = SR_proc_info[node].sock;
    Integer len;
    MessageHeader header;
#ifdef SOCK_FULL_SYNC
    char sync = 0;
#endif

    if ( sock < 0 ) {
        Error("rcv_remote: receiving from process without socket",
                (Integer) node);
    }

    /* read the message header and check contents */

    if (SR_proc_info[node].peeked) {
        /* Have peeked at this socket ... get message header from buffer */

        if (DEBUG_) {
            printf("%2ld: rcv_remote message has been peeked at\n", me);
        }

        (void) memcpy((char *) &header, (char *) &SR_proc_info[node].head_peek,
                      sizeof(MessageHeader));
        SR_proc_info[node].peeked = FALSE;
    }
    else {
#if HAVE_RPC_XDR_H
        (void) ReadXdrLong(sock, (Integer *) &header,
                           (Integer) (sizeof(header)/sizeof(Integer)));
#else
        if ( (len = ReadFromSocket(sock, (char *) &header, (Integer) sizeof(header)))
                != sizeof(header) ) {
            Error("rcv_remote: reading header from socket", len);
        }
#endif
    }

    if (DEBUG_) {
        PrintMessageHeader("rcv_remote",&header);
    }

    if (header.nodeto != me) {
        PrintMessageHeader("rcv_remote",&header);
        Error("rcv_remote: got message meant for someone else",
                (Integer) header.nodeto);
    }

    *nodefrom = header.nodefrom;
    if (*nodefrom != node) {
        Error("rcv_remote: got message from someone on incorrect socket",
                (Integer) *nodefrom);
    }

    if (header.type != *type) {
        PrintMessageHeader("rcv_remote",&header);
        printf("rcv_remote: type mismatch ... strong typing enforced\n");
        abort();
        Error("rcv_remote: type mismatch ... strong typing enforced", (Integer) *type);
    }

#if HAVE_RPC_XDR_H
    if ( *type & MSGDBL ) {
        *lenmes = header.length * sizeof(DoublePrecision);
    }
    else if ( *type & MSGINT ) {
        *lenmes = header.length * sizeof(Integer);
    }
    else if ( *type & MSGCHR ) {
        *lenmes = header.length * sizeof(char);
    }
    else {
        *lenmes = header.length; 
    }
#else
    *lenmes = header.length; 
#endif

    if ( (*lenmes < 0) || (*lenmes > BIG_MESSAGE) || (*lenmes > *lenbuf) ) {
        PrintMessageHeader("rcv_remote",&header);
        (void) fprintf(stderr, "rcv_remote err: lenbuf=%ld\n",*lenbuf);
        Error("rcv_remote: message length out of range",(Integer) *lenmes);
    }

    if (*lenmes > 0) {
#if HAVE_RPC_XDR_H
        if ( *type & MSGDBL ) {
            (void) ReadXdrDouble(sock, (DoublePrecision *) buf, header.length);
        }
        else if ( *type & MSGINT ) {
            (void) ReadXdrLong(sock, (Integer *) buf, header.length);
        }
        else if ( *type & MSGCHR ) {
            (void) ReadXdrChar(sock, (char *) buf, header.length);
        }
        else if ( (len = ReadFromSocket(sock, buf, *lenmes)) != *lenmes) {
            Error("rcv_remote: reading message from socket",
                    (Integer) (len+100000*(sock+ 1000* *nodefrom)));
        }
#else
        if ( (len = ReadFromSocket(sock, buf, *lenmes)) != *lenmes) {
            Error("rcv_remote: reading message from socket",
                    (Integer) (len+100000*(sock+ 1000* *nodefrom)));
        }
#endif
    }

    /* this write (and read in snd_remote) makes the link synchronous */

#ifdef SOCK_FULL_SYNC
    if ( WriteToSocket(sock, &sync, (Integer) 1) != 1) {
        Error("rcv_remote: writing sync to socket", (Integer) node);
    }
#endif

}


/**
 * Integer *type        = user defined type of received message (input)
 * char *buf        = data buffer (output)
 * Integer *lenbuf      = length of buffer in bytes (input)
 * Integer *lenmes      = length of received message in bytes (output)
 * (exceeding receive buffer is hard error)
 * Integer *nodeselect  = node to receive from (input)
 * -1 implies that any pending message may be received
 *
 * Integer *nodefrom    = node message is received from (output)
 * Integer *sync        = 0 for asynchronous, 1 for synchronous (NOT USED)
 */
void RCV_(
        Integer *type,
        void *buf,
        Integer *lenbuf,
        Integer *lenmes,
        Integer *nodeselect,
        Integer *nodefrom,
        Integer *sync)
{
    Integer me = NODEID_();
    Integer nproc = NNODES_();
    Integer node;
#ifdef TCGMSG_TIMINGS
    DoublePrecision start;
#endif
#ifdef USE_VAMPIR
    vampir_begin(TCGMSG_RCV,__FILE__,__LINE__);
#endif

#ifdef EVENTLOG
    evlog(EVKEY_BEGIN,     EVENT_RCV,
            EVKEY_MSG_FROM, (int) *nodeselect,
            EVKEY_MSG_TO,   (int)  me,
            EVKEY_MSG_TYPE, (int) *type,
            EVKEY_MSG_SYNC, (int) *sync,
            EVKEY_LAST_ARG);
#endif

    /* Assign the desired node or the next ready node */

#ifdef TCGMSG_TIMINGS
    start = TCGTIME_();
#endif

    if (*nodeselect == -1) {
        node = NextReadyNode(*type);
    }
    else {
        node = *nodeselect;
    }

    /* Check for some errors ... need more checking here ...
       note that the overall master process has id nproc */

    if (node == me) {
        Error("RCV_: cannot receive message from self", (Integer) me);
    }

    if ( (node < 0) || (node > nproc) ) {
        Error("RCV_: out of range node requested", (Integer) node);
    }

    /* Receive the message ... use shared memory, switch or socket */

    if (SR_proc_info[node].local) {
        rcv_local(type, buf, lenbuf, lenmes, &node, nodefrom);
    } else {
        rcv_remote(type, buf, lenbuf, lenmes, &node, nodefrom);
    }

    /* Collect statistics */

    SR_proc_info[node].n_rcv += 1;
    SR_proc_info[node].nb_rcv += *lenmes;

#ifdef TCGMSG_TIMINGS
    SR_proc_info[node].t_rcv += TCGTIME_() - start;
#endif

#ifdef EVENTLOG
    evlog(EVKEY_END, EVENT_RCV,
            EVKEY_MSG_FROM, (int) node,
            EVKEY_MSG_LEN, (int) *lenmes,
            EVKEY_LAST_ARG);
#endif
#ifdef USE_VAMPIR
    vampir_recv(me,node,*lenmes,*type);
    vampir_end(TCGMSG_RCV,__FILE__,__LINE__);
#endif
}    


/**
 * Make a socket connection between processes a and b via the
 * process c to which both are already connected.
 */
void RemoteConnect(Integer a, Integer b, Integer c)
{
    Integer me = NODEID_();
    Integer nproc = NNODES_();
    Integer type = TYPE_CONNECT;  /* Overriden below */
    char cport[8];
    Integer tmp, lenmes, nodefrom, clusid, lenbuf, sync=1;
    int sock, port;
    Integer lport;

    if ((a == b) || (a == c) || (b == c) ) {
        return; /* Gracefully ignore redundant connections */
    }

    if ( (me != a) && (me != b) && (me != c) ) {
        return; /* I'm not involved in this connection */
    }

    if (a < b) {
        tmp = a; a = b; b = tmp;
    }

    type = (a + nproc*b) | MSGINT;  /* Create a unique type */

    if (DEBUG_) {
        (void) printf("RC a=%ld, b=%ld, c=%ld, me=%ld\n",a,b,c,me);
        (void) fflush(stdout);
    }

    if (a == me) {
        CreateSocketAndBind(&sock, &port);  /* Create port */
        if (DEBUG_) {
            (void) printf("RC node=%ld, sock=%d, port=%d\n",me, sock, port);
            (void) fflush(stdout);
        }
        lport = port;
        lenbuf = sizeof lport;
        ListenOnSock(sock);
        /* Port to intermediate */
        SND_(&type, (char *) &lport, &lenbuf, &c, &sync);
        /* Accept connection and save socket info */
        SR_proc_info[b].sock = AcceptConnection(sock);
    }
    else if (b == me) {
        clusid = SR_proc_info[a].clusid;
        lenbuf = sizeof lport;
        RCV_(&type, (char *) &lport, &lenbuf, &lenmes, &c, &nodefrom, &sync);
        port = lport;
        (void) sprintf(cport,"%d",port);
        lenbuf = strlen(cport) + 1;
        if (lenbuf > sizeof cport)
            Error("RemoteConnect: cport too small", (Integer) lenbuf);
        SR_proc_info[a].sock = 
            CreateSocketAndConnect(SR_clus_info[clusid].hostname, cport); 
    }
    else if (c == me) {
        lenbuf = sizeof lport;
        RCV_(&type, (char *) &lport, &lenbuf, &lenmes, &a, &nodefrom, &sync);
        SND_(&type, (char *) &lport, &lenbuf, &b, &sync);
    }
}
