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
     printf("program writes test data to a binary file\n");
     printf("usage: dbg_write.x <filename> <type> <from> <to>\n");
     printf("type: 1 - integer 2 - double \n <from> <to> -range of elements (0, ...)\n");
     return(1);
   }

   type = atoi(argv[2]);
   from = atoi(argv[3]);
   to   = atoi(argv[4]);

   if(from < 0 || to < from) {printf("range error\n"); return 1;}
   if(!(fd = dra_el_open(argv[1],DRA_W))){printf("not found\n"); return 1;} 

   switch (type){

   case 1: 
           for(i=from; i<= to; i+= LEN){
               imax = MIN(i+LEN-1,to);
               offset = sizeof(Integer)*i;
               for(ii=0;ii<imax-i+1;ii++) idata[ii]=ii;
               status=dra_el_write(idata, sizeof(Integer), imax-i+1, fd, offset);
               if(!status)printf("error write failed\n");
           }
           break;
   case 2: 
           for(i=from; i<= to; i+= LEN){ 
               imax = MIN(i+LEN-1,to);
               offset = sizeof(DoublePrecision)*i;
               for(ii=0;ii<imax-i+1;ii++) ddata[ii]=1.*ii;
               status=dra_el_write(ddata,sizeof(DoublePrecision), imax -i+1, fd,offset);
               if(!status)printf("error write failed\n");
           }
           break;
   default: printf("type error\n"); return 1;
   }

}
