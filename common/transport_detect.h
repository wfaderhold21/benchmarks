/*
 * transport_detect.h -- Detect loaded UCC TL / UCX transport libraries
 *
 * Uses dl_iterate_phdr to inspect the process address space for
 * libucc_tl_*.so and libuct_*.so symbols, returning a comma-separated
 * string that can be written directly into CSV output.
 */

#ifndef TRANSPORT_DETECT_H
#define TRANSPORT_DETECT_H

/* Returns allocated string (comma-separated transport names), caller frees */
char *transport_detect(void);
void  transport_free(char *s);

#endif /* TRANSPORT_DETECT_H */
