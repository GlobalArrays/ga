
static int locked_slot;
struct locked_array_info_t{
       Integer g_a;
       Integer ilo;
       Integer ihi;
       Integer jlo;
       Integer jhi;
       Integer padding[3];
}

volatile double dummy=0.;

static void wait_awhile()
{
int n = 100;
    while(n--) dummy++;
}
      


void ga_shmem_lock(g_a, proc, ilo, ihi, jlo, jhi)
Integer g_a, proc, ilo, ihi, jlo, jhi;
{
    int entering_critical = 0; 

    while (! entering_critical){

        lock(proc);

        if(n_locked_sections[proc] == 0){

             entering_critical = 1;
             locked_slot[proc] = 0;

        } else if(n_locked_sections[proc] < MAX_LOCKED){

               entering_critical = 1;

               /* need to search entire array */
               for(slot = 0; slot <  MAX_LOCKED; slot++){
                   if(locked_array_info[slot].ilo < 0) locked_slot[proc] = slot;
                   else if (OVERLAPS(locked_array_info[slot], g_a,ilo, ihi, jlo, jhi) {
                           entering_critical = 0;
                           break;
                   }
               }
        }

        if( entering_critical ) {
            locked_array_info[slot].g_a = g_a;
            locked_array_info[slot].ilo = ilo;
            locked_array_info[slot].ihi = ihi;
            locked_array_info[slot].jlo = jlo;
            locked_array_info[slot].jhi = jhi;
            n_locked_sections[proc]++;
        } 
        
        unlock(proc);

        if(!entering_critical ) wait_awhile();

    }
}


ga_shmem_unlock(g_a, proc, ilo, ihi, jlo, jhi)
Integer g_a, proc, ilo, ihi, jlo, jhi;
{

        lock(proc);
        locked_array_info[locked_slot].ilo = -1;
        n_locked_sections[proc]--;
        unlock(proc);
}
