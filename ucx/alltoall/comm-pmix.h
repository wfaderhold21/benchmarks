#ifndef COMM_PMIX_H
#define COMM_PMIX_H

int pmix_data_exchange(uint64_t remote, void *** pack_param);
int pmix_worker_exchange(void *** param_worker_addrs);
int barrier_all(void);
int init_pmix();


#endif
