/* $Header: /tmp/hpctools/ga/tcgmsg/ipcv4.0/xdrtest.c,v 1.3 1995-02-24 02:18:09 d3h325 Exp $ */

#include <stdio.h>
#include <rpc/types.h>
#include <rpc/xdr.h>

#if defined(ULTRIX) || defined(SGI) || defined(DECOSF)
extern void *malloc();
#else
extern char *malloc();
#endif

static char *xdrbuf;

static XDR xdrs;

int main(argc, argv)
   int argc;
   char **argv;
{
   long data[4];
   long len;
   long *temp=data;

   if (argc != 2)
     return 1;

   xdrbuf = malloc(4096);

   if (strcmp(argv[1], "encode") == 0) {
    xdrmem_create(&xdrs, xdrbuf, 4096, XDR_ENCODE);
    (void) fprintf(stderr," encode xdr_setpos=%d\n",
                   xdr_setpos(&xdrs, (u_int) 0));
    (void) scanf("%ld %ld %ld %ld", data, data+1, data+2, data+3);
    (void) fprintf(stderr,"encode Input longs %ld, %ld, %ld, %ld\n",
                      data[0], data[1], data[2], data[3]);
    len = 4;
    (void) fprintf(stderr,"encode xdr_array=%d\n",
      xdr_array(&xdrs, (char **) &temp, &len, (u_int) 4096,
                (u_int) sizeof(long), xdr_long));
    len = 4*4 + 4;
    (void) fprintf(stderr,"encode len=%ld\n", len);
    (void) fwrite(&len, 4, 1, stdout);
    (void) fwrite(xdrbuf, 1, len, stdout);
    (void) fprintf(stderr,"encode data written\n");
    return 0;
   }
   else {
    xdrmem_create(&xdrs, xdrbuf, 4096, XDR_DECODE);
    (void) fprintf(stderr," decode xdr_setpos=%d\n",
                   xdr_setpos(&xdrs, (u_int) 0));
    (void) fread(&len, 4, 1, stdin);
    (void) fprintf(stderr,"decode len=%ld\n", len);
    (void) fread(xdrbuf, 1, len, stdin);
    (void) fprintf(stderr,"decode data read\n");
    (void) fprintf(stderr,"decode xdr_array=%d\n",
      xdr_array(&xdrs, (char **) &temp, &len, (u_int) 4096,
                (u_int) sizeof(long), xdr_long));
    (void) fprintf(stderr,"decode Input longs %ld, %ld, %ld, %ld\n",
                      data[0], data[1], data[2], data[3]);
    return 0;
   }
}
