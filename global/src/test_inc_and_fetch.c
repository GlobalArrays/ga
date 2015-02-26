#include <stdio.h>

//__sync_add_and_fetch (type *ptr, type value, ...)

static unsigned int tag1; /* RACE */  
static unsigned int tag2; /* RACE */  

unsigned int get_next_tag1(void)
{
    return((++tag1));
}

unsigned int get_next_tag2(void)
{
    return __sync_add_and_fetch(&tag2,1);
}

int main(void)
{
    for (unsigned int i=0; i<1000; i++) {
        printf("%u, %u, %u\n", i, get_next_tag1(), get_next_tag2() );
    }
    return 0;
}
