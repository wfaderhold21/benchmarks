/*
 * transport_detect.h -- Detect loaded UCC TL / UCX transport libraries
 *
 * Uses dl_iterate_phdr to inspect the process address space for
 * libucc_tl_*.so and libuct_*.so symbols, returning comma-separated
 * strings that can be written directly into CSV output.
 */

#ifndef TRANSPORT_DETECT_H
#define TRANSPORT_DETECT_H

typedef struct {
    char *ucc_tls;   /* allocated UCC TL string, caller frees via transport_free() */
    char *ucx_tls;   /* allocated UCX transport string, caller frees via transport_free() */
} transport_info_t;

/* Detect both UCC TLs and UCX transports separately */
transport_info_t transport_detect(void);
void             transport_free(char *s);

#endif /* TRANSPORT_DETECT_H */
