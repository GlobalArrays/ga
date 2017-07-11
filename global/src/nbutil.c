#if HAVE_CONFIG_H
#   include "config.h"
#endif

#include "globalp.h"
#include "base.h"
#if HAVE_STDIO_H
#   include <stdio.h>
#endif
#define DEBUG 0
#include <pthread.h>
#include <limits.h>

typedef struct struct_armcihdl_t{
    armci_hdl_t* handle;
    struct struct_armcihdl_t *next;
    int index;
}ga_armcihdl_t;

static ga_armcihdl_t head={NULL,NULL,0};
static ga_armcihdl_t *tail=&head;
pthread_mutex_t mutex;
static unsigned next_hdl = 1; 
static int loop_flag = 0;

void add_hdl(Integer *nbhandle);
ga_armcihdl_t *get_hdl(Integer *nbhandle);
void remove_hdl(Integer *nbhandle);

armci_hdl_t* get_armci_nbhandle(Integer *nbhandle){
  return get_hdl(nbhandle)->handle;
}
int nga_wait_internal(Integer *nbhandle){
  if(!ARMCI_Wait(get_hdl(nbhandle)->handle))
  {
    remove_hdl(nbhandle);  
    return 0;
  }
  return 1;

}

int nga_test_internal(Integer *nbhandle){
  if(!ARMCI_Test(get_hdl(nbhandle)->handle))
  {
    remove_hdl(nbhandle);  
    return 0;
  }
  return 1;
}

void ga_init_nbhandle(Integer *nbhandle)
{
  pthread_mutex_lock(&mutex);
    *nbhandle = next_hdl;
    if(next_hdl == UINT_MAX)
     {
       loop_flag = 1;
       next_hdl = 0;
     }
    next_hdl++;
    if(loop_flag)
      if(get_hdl(nbhandle))
        nga_wait_internal(nbhandle);
    add_hdl(nbhandle);
  pthread_mutex_unlock(&mutex);
  ARMCI_INIT_HANDLE(get_hdl(nbhandle)->handle);
}

void add_hdl(Integer *nbhandle)
{
  ga_armcihdl_t *temp = (ga_armcihdl_t*) malloc(sizeof(ga_armcihdl_t));
  temp->handle = (armci_hdl_t*) malloc(sizeof(armci_hdl_t)); 
  temp->index = *nbhandle;
  temp->next = NULL;
  tail->next=temp;
  tail = temp;
}

ga_armcihdl_t *get_hdl(Integer *nbhandle)
{
  pthread_mutex_lock(&mutex);
  ga_armcihdl_t *temp = &head;
  while(temp->next != NULL && temp->index != *nbhandle)
    temp = temp->next;
  if(temp->index == *nbhandle)
  {
  pthread_mutex_unlock(&mutex);
    return temp;
  }
  else
  {
  pthread_mutex_unlock(&mutex);
    return 0; 
  }
}

void remove_hdl(Integer *nbhandle)
{
  pthread_mutex_lock(&mutex);
  ga_armcihdl_t *temp = &head;
  ga_armcihdl_t *temp2;
  while(temp->next != NULL && temp->next->index != *nbhandle)
    temp = temp->next;
  if(temp->next->index == *nbhandle)
  {
    if(temp->next->next == NULL)
      tail = temp;
    temp2 = temp->next;
    temp->next = temp->next->next;
    free(temp2->handle);
    free(temp2);
  }
  pthread_mutex_unlock(&mutex);
}
