#define ARMCI_PROFILE_GET 1
#define ARMCI_PROFILE_PUT 2
#define ARMCI_PROFILE_ACC 3

#define ARMCI_MAX_DIM 7

extern void armci_profile_init();
extern void armci_profile_terminate();
extern void armci_profile_start_strided(int count[], int stride_levels, 
					int proc, int comm_type);
extern void armci_profile_stop_strided();
extern void armci_profile_start_vector(armci_giov_t darr[], int len, int proc,
				       int comm_type);
extern void armci_profile_stop_vector();
