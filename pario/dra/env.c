#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define MAX_NUM_SERV 64

/*\ get number of I/O servers from optional environmental variable DRA_NUM_SERV
$Id: env.c,v 1.1 1997-12-07 11:14:18 d3e129 Exp $
\*/
int drai_get_num_serv()
{
int  val=-1;
char *str;

     str = getenv("DRA_NUM_SERV");
     if(str==NULL)val = 0;
     else{
         val = atoi(str);
         if(val<1 || val >MAX_NUM_SERV)val =0;
     }
     return val;
}  
