#include "macommon.h"
#include "global.h"
#include "DRA.h"
#include "DRAp.h"


int main(argc, argv)
int argc;
char **argv;
{
#define LEN 10
int from, to, type;
Integer idata[LEN];
int fd;
Integer i, ii, imax, offset, status;
DoublePrecision ddata[LEN];

   if(argc < 2){
     printf("program prints data from a binary file to screen\n");
     printf("usage: dbg_read.x <filename> <type> <from> <to>\n");
     printf("type: 1 - integer 2 - double \n <from> <to> -range of elements (0, ...)\n");
     return(1);
   }

   type = atoi(argv[2]);
   from = atoi(argv[3]);
   to   = atoi(argv[4]);

   if(from < 0 || to < from) {printf("range error\n"); return 1;}
   if(!(fd = dra_el_open(argv[1],DRA_R))){printf("not found\n"); return 1;} 

   switch (type){

   case 1: 
           for(i=from; i<= to; i+= LEN){
               imax = MIN(i+LEN-1,to);
               offset = sizeof(Integer)*i;
               status=dra_el_read(idata, sizeof(Integer), imax -i+1, fd, offset);
               if(!status)printf("error read failed\n");
               for(ii=0;ii<imax-i+1;ii++) printf("%4ld ",idata[ii]);
               printf("\n");
           }
           break;
   case 2: 
           for(i=from; i<= to; i+= LEN){ 
               imax = MIN(i+LEN-1,to);
               offset = sizeof(DoublePrecision)*i;
               status=dra_el_read(ddata, sizeof(DoublePrecision), imax -i+1, fd, offset);
               if(!status)printf("error read failed\n");
               for(ii=0;ii<imax-i+1;ii++) printf("%lf ",ddata[ii]);
               printf("\n");
           }
           break;
   default: printf("type error\n"); return 1;
   }

}
