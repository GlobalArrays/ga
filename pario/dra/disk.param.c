/**********************************************************************/
/* store and retrieve parameters of disk arrays                       */ 
/*       -- at present time, we use additional file for parameters    */
/**********************************************************************/


#include <stdio.h>
#include <unistd.h>
#include "global.h"
#include "drap.h"

#define MAX_HD_NAME_LEN 100
#define HD_NAME_EXT_LEN 10
#define HD_EXT ".info"




/*\ Retrive parameters of a disk array from the disk
\*/
void dai_read_param(char* filename,Integer d_a)
{
FILE *fd;
char param_filename[MAX_HD_NAME_LEN];
Integer len;
Integer me=ga_nodeid_();
Integer brd_type=DRA_BRD_TYPE, orig, dra_hndl=d_a+DRA_OFFSET;
long input;
char dummy[80];

  ga_sync_();
    
  if(!me){ /* only process 0 reads param file */

    /* build param file name */
    len = strlen(filename);
    if(len+HD_NAME_EXT_LEN >= MAX_HD_NAME_LEN)
       dai_error("dai_read_param: filename too long:",len);
    strcpy(param_filename,filename);
    strcat(param_filename,HD_EXT);

    if(!(fd=fopen(param_filename,"r")))
                    dai_error("dai_read_param: open failed",0);   

    if(!fscanf(fd,"%ld", &input))  dai_error("dai_read_param:dim1",0);
    DRA[dra_hndl].dim1 = (Integer) input;
    if(!fscanf(fd,"%ld",&input))   dai_error("dai_read_param:dim2",0);
    DRA[dra_hndl].dim2 = (Integer) input;

    if(!fscanf(fd,"%ld",&input))   dai_error("dai_read_param:type",0);
    DRA[dra_hndl].type = (Integer) input;
    if(!fscanf(fd,"%ld",&input))   dai_error("dai_read_param:layout",0);
    DRA[dra_hndl].layout = (Integer) input;

    if(!fscanf(fd,"%ld",&input))   dai_error("dai_read_param:chunk1",0);
    DRA[dra_hndl].chunk1 = (Integer) input;
    if(!fscanf(fd,"%ld",&input))   dai_error("dai_read_param:chunk2",0);
    DRA[dra_hndl].chunk2 = (Integer) input;

    fgets(dummy,80,fd); /*advance to next line*/
    if(!fgets(DRA[dra_hndl].name,DRA_MAX_NAME,fd))dai_error("dai_read_param:name",0);

    if(fclose(fd))dai_error("dai_read_param: fclose failed",0);
  }

  /* process 0 broadcasts data to everybody else                            */
  /* for 6 Integers there shouldn't be alignement padding in the structure */
  /* the integers are followed by array name */
  len = 6*sizeof(Integer)+DRA_MAX_NAME+1; orig =0;
  ga_brdcst_(&brd_type, DRA + dra_hndl, &len, &orig);
  
  ga_sync_();
}
  

  
/*\ Store parameters of a disk array on the disk
\*/
void dai_write_param(char* filename,Integer d_a)
{
Integer len;
FILE *fd;
char param_filename[MAX_HD_NAME_LEN];
Integer me=ga_nodeid_(), dra_hndl=d_a+DRA_OFFSET;

  ga_sync_();
   
  if(!me){ /* only process 0 writes param file */

    /* build param file name */
    len = strlen(filename);
    if(len + HD_NAME_EXT_LEN >= MAX_HD_NAME_LEN)
       dai_error("dai_write_param: filename too long:",len);
    strcpy(param_filename,filename);
    strcat(param_filename,HD_EXT);

    if(! (fd = fopen(param_filename,"w")) )
                                dai_error("dai_write_param:open failed",0);

    if(!fprintf(fd,"%ld ",(long)DRA[dra_hndl].dim1)) 
                                dai_error("dai_write_param:dim1",0);
    if(!fprintf(fd,"%ld ",(long)DRA[dra_hndl].dim2)) 
                                dai_error("dai_write_param:dim2",0);
    if(!fprintf(fd,"%ld ",(long)DRA[dra_hndl].type)) 
                                dai_error("dai_write_param:type",0);
    if(!fprintf(fd,"%ld ",(long)DRA[dra_hndl].layout))
                                dai_error("dai_write_param:layout",0);
    if(!fprintf(fd,"%ld ",(long)DRA[dra_hndl].chunk1))
                                dai_error("dai_write_param:chunk1",0);
    if(!fprintf(fd,"%ld ",(long)DRA[dra_hndl].chunk2))
                                dai_error("dai_write_param:chunk2",0);
    if(!fprintf(fd,"\n%s\n",(long)DRA[dra_hndl].name))
                                dai_error("dai_write_param:name",0);

    if(fclose(fd))dai_error("dai_write_param: fclose failed",0);
  }

  ga_sync_();
}



/*\ Delete info file
\*/
void dai_delete_param(char* filename,Integer d_a)
{
char param_filename[MAX_HD_NAME_LEN];
int len;
Integer me=ga_nodeid_();

  ga_sync_();

  if(!me){ /* only process 0 reads param file */

    /* build param file name */
    len = strlen(filename);
    if(len+HD_NAME_EXT_LEN >= MAX_HD_NAME_LEN)
       dai_error("dai_read_param: filename too long:",len);
    strcpy(param_filename,filename);
    strcat(param_filename,HD_EXT);

    if(unlink(param_filename)) dai_error("dai_delete_param failed",d_a);
  }
}
