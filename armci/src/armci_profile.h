/* $Id: armci_profile.h,v 1.3 2004-07-20 02:26:10 manoj Exp $ */

#define ARMCI_PROFILE_GET          1
#define ARMCI_PROFILE_PUT          2
#define ARMCI_PROFILE_ACC          3
#define ARMCI_PROFILE_NBGET        4
#define ARMCI_PROFILE_NBPUT        5
#define ARMCI_PROFILE_NBACC        6
#define ARMCI_PROFILE_BARRIER      7
#define ARMCI_PROFILE_WAIT         8
#define ARMCI_PROFILE_NOTIFY_WAIT  9
#define ARMCI_PROFILE_FENCE        10
#define ARMCI_PROFILE_ALLFENCE     11

#define ARMCI_MAX_DIM 7

extern void armci_profile_init();
extern void armci_profile_terminate();
extern void armci_profile_start_strided(int count[], int stride_levels, 
					int proc, int comm_type);
extern void armci_profile_stop_strided();
extern void armci_profile_start_vector(armci_giov_t darr[], int len, int proc,
				       int comm_type);
extern void armci_profile_stop_vector();
extern void armci_profile_start(int comm_type);
extern void armci_profile_stop();
