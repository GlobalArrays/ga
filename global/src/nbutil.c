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

typedef struct struct_armcihdl_t{
    armci_hdl_t* handle;
    struct struct_armcihdl_t *next;
    struct struct_armcihdl_t *prev;
    int index;
}ga_armcihdl_t;

static ga_armcihdl_t head={NULL,NULL,NULL,0};
pthread_mutex_t mutex;
static unsigned next_hdl = 1; 

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
    next_hdl++;
    add_hdl(nbhandle);
  pthread_mutex_unlock(&mutex);
  ARMCI_INIT_HANDLE(get_hdl(nbhandle)->handle);
}

void add_hdl(Integer *nbhandle)
{
  ga_armcihdl_t *temp = &head;
  while(temp->next != NULL)
    temp = temp->next;
  temp->next = (ga_armcihdl_t*) malloc(sizeof(ga_armcihdl_t));
  ga_armcihdl_t *save = temp;
  temp = temp->next;
  temp->handle = (armci_hdl_t*) malloc(sizeof(armci_hdl_t)); 
  temp->index = *nbhandle;
  temp->next = NULL;
  temp->prev= save;
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
  while(temp->next != NULL && temp->index != *nbhandle)
    temp = temp->next;
  if(temp->index == *nbhandle)
  {
    if(temp->next != NULL)
      temp->next->prev=temp->prev;
    temp->prev->next=temp->next;
    free(temp->handle);
    free(temp);
  }
  pthread_mutex_unlock(&mutex);
}
