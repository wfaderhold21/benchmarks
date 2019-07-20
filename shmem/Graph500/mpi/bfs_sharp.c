/* Copyright (C) 2010 The Trustees of Indiana University.                  */
/*                                                                         */
/* Use, modification and distribution is subject to the Boost Software     */
/* License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at */
/* http://www.boost.org/LICENSE_1_0.txt)                                   */
/*                                                                         */
/*  Authors: Jeremiah Willcock                                             */
/*           Andrew Lumsdaine                                              */

#include "common.h"
#include "oned_csr.h"
#include <mpi.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <limits.h>

#include <shmem.h>
#include <shmemx.h>

int qb1 = 0, qb2 = 0, p2 = 0; 

static oned_csr_graph g;

void make_graph_data_structure(const tuple_graph* const tg) {
  convert_graph_to_oned_csr(tg, &g);
}

void free_graph_data_structure(void) {
  free_oned_csr_graph(&g);
}

int bfs_writes_depth_map(void) {
  return 0;
}

extern int64_t * pred;
int64_t *pred2_buf;
unsigned long * qb1_buf, *qb2_buf;


/**
 * @index   the index to operate on
 * @val     the value to compare to
 * @pe      the pe holding the index
 * @obj     the sharp object all of this occurs on
 */
static inline void sharp_atomic_min(size_t index, int64_t val, int pe, int64_t * ag) 
{
    int64_t cval = 0;
    int64_t lval = 0;
    int done = 0;
    int64_t error = 0;

    //TODO: change this to an atomic fadd
    shmem_long_get(&lval, &ag[index], 1, pe);
    if(val < lval) {
        for(;;) {
            if (val < lval) {
                cval = val;
            } else {
                break;
            }
            error = shmem_long_atomic_compare_swap(&ag[index], lval, cval, pe);
            if(error == lval) {
                break;
            } else {
                lval = error;
            }
        }
    }
}

/**
 * @index   the index to operate on
 * @val     the value to or with
 * @pe      the pe holding the index
 * @obj     the sharp object all of this occurs on
 */
static inline void sharp_atomic_bor(size_t index, uint64_t val, int pe, unsigned long * ag) 
{
    //shmem_ulong_atomic_or(&ag[index], val, pe);
    uint64_t cval = 0;
    uint64_t lval = 0;

//    sharp_comms_get(&lval, sizeof(uint64_t), index*sizeof(uint64_t), pe, ag->id);
    shmem_ulong_get(&lval, &ag[index], 1, pe);
    cval = lval | val;
    if(cval == lval) {
        return;
    }
    for(;;) {
        unsigned long error;
        //error = sharp_comms_put(&cval, sizeof(int64_t), index * sizeof(int64_t), pe, ag->id);            
        error = shmem_ulong_atomic_compare_swap(&ag[index], lval, cval, pe);
//        error = sharp_comms_atomic_cswap64(&cval, lval, index*sizeof(uint64_t), pe, ag->id);
        if(error == lval) {
            break;
        } else {
            lval = error;
            cval = lval | val;
            if (cval == lval)
                break;
        }  
    }
}
/* This BFS represents its queues as bitmaps and uses some data representation
 * tricks to fit with the use of MPI one-sided operations.  It is not much
 * faster than the standard version on the machines I have tested it on, but
 * systems that have good RDMA hardware and good MPI one-sided implementations
 * might get better performance from it.  This code might also be good to
 * translate to UPC, Co-array Fortran, SHMEM, or GASNet since those systems are
 * more designed for one-sided remote memory operations. */
void run_bfs(int64_t root, int64_t* pred) {
  const size_t nlocalverts = g.nlocalverts;
  const int64_t nglobalverts = g.nglobalverts;
//    sharp_group_allocated_t * qb1_ag, * qb2_ag;
//    sharp_group_allocated_t * p_ag, *p2_ag;
  int error = 0;
    int64_t * dummy1, * dummy2, *dummy3;
    double min_time_s, min_time_e, bor_time_s, bor_time_e;

 //   printf("allocating 4 allocation groups with size: %lu\n", (nlocalverts * sizeof(int64_t)));

    if(p2 == 0) {    
#ifdef WITH_HINTS
        pred2_buf = (int64_t *) shmemx_malloc_with_hint(nlocalverts * sizeof(int64_t), SHMEM_HINT_NEAR_NIC_MEM);
#else
        pred2_buf = (int64_t *) shmem_malloc(nlocalverts * sizeof(int64_t));
#endif
//        error = sharp_create_group_allocate(nlocalverts * sizeof(int64_t) * comms_size,
//                                    SHARP_MPI, MPI_COMM_WORLD, comms_size,
//                                    &data_tiers, &pred2_ag);
        //printf("error is %d\n", error);
        p2 = 1;
    } 

//    p_ag = &pred_ag;
//    p2_ag = &pred2_ag;


  /* Set up a second predecessor map so we can read from one and modify the
   * other. */
  int64_t* orig_pred = pred;
    //FIXME: change this to allocation group
  int64_t* pred2 = pred2_buf;// = (int64_t*)xMPI_Alloc_mem(nlocalverts * sizeof(int64_t));
    dummy1 = (int64_t*)xMPI_Alloc_mem(nlocalverts * sizeof(int64_t));

  /* The queues (old and new) are represented as bitmaps.  Each bit in the
   * queue bitmap says to check elts_per_queue_bit elements in the predecessor
   * map for vertices that need to be visited.  In other words, the queue
   * bitmap is an overapproximation of the actual queue; because MPI_Accumulate
   * does not get any information on the result of the update, sometimes
   * elements are also added to the bitmap when they were actually already
   * black.  Because of this, the predecessor map needs to be checked to be
   * sure a given vertex actually needs to be processed. */
  const int elts_per_queue_bit = 4;
  const int ulong_bits = sizeof(unsigned long) * CHAR_BIT;
  int64_t queue_nbits = (nlocalverts + elts_per_queue_bit - 1) / elts_per_queue_bit;
  int64_t queue_nwords = (queue_nbits + ulong_bits - 1) / ulong_bits;
    if(qb1 == 0) {
#ifdef WITH_HINTS
        qb1_buf = (unsigned long *) shmemx_malloc_with_hint(queue_nwords * sizeof(int64_t), SHMEM_HINT_NEAR_NIC_MEM);
#else
        qb1_buf = (unsigned long *) shmem_malloc(queue_nwords * sizeof(int64_t));
#endif
/*        error = sharp_create_group_allocate(queue_nwords * sizeof(int64_t) * comms_size,
                                    SHARP_MPI, MPI_COMM_WORLD, comms_size,
                                    &data_tiers, &queue_bitmap1_ag);*/
        //printf("error is %d\n", error);
        qb1 = 1;
    }
    if(qb2 == 0) {
#ifdef WITH_HINTS
        qb2_buf = (unsigned long *) shmemx_malloc_with_hint(queue_nwords * sizeof(int64_t), SHMEM_HINT_NEAR_NIC_MEM);
#else
        qb2_buf = (unsigned long *) shmem_malloc(queue_nwords * sizeof(int64_t));
#endif
//        error = sharp_create_group_allocate(queue_nwords * sizeof(int64_t) * comms_size,
//                                SHARP_MPI, MPI_COMM_WORLD, comms_size,
//                                &data_tiers, &queue_bitmap2_ag);
        //printf("error is %d\n", error);
        qb2 = 1;
    }

//    qb1_ag = &queue_bitmap1_ag;
//    qb2_ag = &queue_bitmap2_ag;
    //FIXME: change this to allocation group
  unsigned long* queue_bitmap1;// = (unsigned long*)xMPI_Alloc_mem(queue_nwords * sizeof(unsigned long));
    //FIXME: change this to allocation group
  unsigned long* queue_bitmap2;// = (unsigned long*)xMPI_Alloc_mem(queue_nwords * sizeof(unsigned long));
    dummy2 = (unsigned long*)xMPI_Alloc_mem(queue_nwords * sizeof(unsigned long));
    dummy3 = (unsigned long*)xMPI_Alloc_mem(queue_nwords * sizeof(unsigned long));
//    pred = pred_ag.buffer_ptr;
    //pred2 = pred2_ag.buffer_ptr;
    queue_bitmap1 = qb1_buf;
    queue_bitmap2 = qb2_buf;
  memset(queue_bitmap1, 0, queue_nwords * sizeof(unsigned long));
//  memset(queue_bitmap2, 0, queue_nwords * sizeof(unsigned long));

  /* List of local vertices (used as sources in MPI_Accumulate). */
  int64_t* local_vertices = (int64_t*)xMPI_Alloc_mem(nlocalverts * sizeof(int64_t));
  {size_t i; for (i = 0; i < nlocalverts; ++i) local_vertices[i] = VERTEX_TO_GLOBAL(rank, i);}

  /* List of all bit masks for an unsigned long (used as sources in
   * MPI_Accumulate). */
  unsigned long masks[ulong_bits];
  {int i; for (i = 0; i < ulong_bits; ++i) masks[i] = (1UL << i);}

  /* Coding of predecessor map: */
  /* - White (not visited): INT64_MAX */
  /* - Grey (in queue): 0 .. nglobalverts-1 */
  /* - Black (done): -nglobalverts .. -1 */

  /* Set initial predecessor map. */
  {size_t i; for (i = 0; i < nlocalverts; ++i) pred[i] = INT64_MAX;}

  /* Mark root as grey and add it to the queue. */
  if (VERTEX_OWNER(root) == rank) {
    pred[VERTEX_LOCAL(root)] = root;
    queue_bitmap1[VERTEX_LOCAL(root) / elts_per_queue_bit / ulong_bits] |= (1UL << ((VERTEX_LOCAL(root) / elts_per_queue_bit) % ulong_bits));
  }

  /* Create MPI windows on the two predecessor arrays and the two queues. */

    MPI_Barrier(MPI_COMM_WORLD);
  /*MPI_Win pred_win, pred2_win, queue1_win, queue2_win;
  MPI_Win_create(pred, nlocalverts * sizeof(int64_t), sizeof(int64_t), MPI_INFO_NULL, MPI_COMM_WORLD, &pred_win);
  MPI_Win_create(pred2, nlocalverts * sizeof(int64_t), sizeof(int64_t), MPI_INFO_NULL, MPI_COMM_WORLD, &pred2_win);
  MPI_Win_create(queue_bitmap1, queue_nwords * sizeof(unsigned long), sizeof(unsigned long), MPI_INFO_NULL, MPI_COMM_WORLD, &queue1_win);
  MPI_Win_create(queue_bitmap2, queue_nwords * sizeof(unsigned long), sizeof(unsigned long), MPI_INFO_NULL, MPI_COMM_WORLD, &queue2_win);
*/

  while (1) {
    int64_t i;
    /* Clear the next-level queue. */
    memset(queue_bitmap2, 0, queue_nwords * sizeof(unsigned long));

    /* The pred2 array is pred with all grey vertices changed to black. */
    memcpy(pred2, pred, nlocalverts * sizeof(int64_t));
    for (i = 0; i < (int64_t)nlocalverts; ++i) {
      if (pred2[i] >= 0 && pred2[i] < nglobalverts) pred2[i] -= nglobalverts;
    }


    /* Start one-sided operations for this level. */
   MPI_Barrier(MPI_COMM_WORLD);
//    MPI_Win_fence(MPI_MODE_NOPRECEDE, pred2_win);
//    MPI_Win_fence(MPI_MODE_NOPRECEDE, queue2_win);

    /* Step through the words of the queue bitmap. */
    for (i = 0; i < queue_nwords; ++i) {
      unsigned long val = queue_bitmap1[i];
      int bitnum;
      /* Skip any that are all zero. */
      if (!val) { 
            //printf("skipping %d!\n", i);
            continue;       
        }
      /* Scan the bits in the word. */
      for (bitnum = 0; bitnum < ulong_bits; ++bitnum) {
        size_t first_v_local = (size_t)((i * ulong_bits + bitnum) * elts_per_queue_bit);
        if (first_v_local >= nlocalverts) break;
        int bit = (int)((val >> bitnum) & 1);
        /* Skip any that are zero. */
        if (!bit) { 
            //printf("skipping bit: %d\n", bit);
            continue;
        }
        /* Scan the queue elements corresponding to this bit. */
        int qelem_idx;
        for (qelem_idx = 0; qelem_idx < elts_per_queue_bit; ++qelem_idx) {
          size_t v_local = first_v_local + qelem_idx;
          if (v_local >= nlocalverts) continue;
          /* Since the queue is an overapproximation, check the predecessor map
           * to be sure this vertex is grey. */
            //printf("it came this far, pred[v_local]: %ld, nglobalverts: %d\n", pred[v_local], nglobalverts);
          if (pred[v_local] >= 0 && pred[v_local] < nglobalverts) {
            size_t ei, ei_end = g.rowstarts[v_local + 1];
            /* Walk the incident edges. */
            //printf("ei_end: %lu, g.rowstarts[v_local]: %lu\n", ei_end, g.rowstarts[v_local]);
            for (ei = g.rowstarts[v_local]; ei < ei_end; ++ei) {
              int64_t w = g.column[ei];
                //printf("and this far, w: %ld vertex_to_global: %d\n", w, VERTEX_TO_GLOBAL(rank, v_local));
              if (w == VERTEX_TO_GLOBAL(rank, v_local)) continue; /* Self-loop */
              /* Set the predecessor of the other edge endpoint (note use of
               * MPI_MIN and the coding of the predecessor map). */
               //printf("local_vertices[%d]: %ld\n", v_local, local_vertices[v_local]); 
                //printf("atomic min on pe %d, offset: %lu\n", VERTEX_OWNER(w), VERTEX_LOCAL(w)*sizeof(int64_t));
                //min_time_s = MPI_Wtime();
              sharp_atomic_min(VERTEX_LOCAL(w), local_vertices[v_local], VERTEX_OWNER(w), pred2);
                //min_time_e = MPI_Wtime();
     //         MPI_Accumulate(&local_vertices[v_local], 1, MPI_INT64_T, VERTEX_OWNER(w), VERTEX_LOCAL(w), 1, MPI_INT64_T, MPI_MIN, pred2_win);
              /* Mark the endpoint in the remote queue (note that the min may
               * not do an update, so the queue is an overapproximation in this
               * way as well). */
                //bor_time_s = MPI_Wtime();
              sharp_atomic_bor((VERTEX_LOCAL(w) / elts_per_queue_bit / ulong_bits), masks[((VERTEX_LOCAL(w) / elts_per_queue_bit) % ulong_bits)], VERTEX_OWNER(w), queue_bitmap2);
                //bor_time_e = MPI_Wtime();
                //printf("min_time: %g, bor_time: %g\n", min_time_e - min_time_s, bor_time_e - bor_time_s);
          //    MPI_Accumulate(&masks[((VERTEX_LOCAL(w) / elts_per_queue_bit) % ulong_bits)], 1, MPI_UNSIGNED_LONG, VERTEX_OWNER(w), VERTEX_LOCAL(w) / elts_per_queue_bit / ulong_bits, 1, MPI_UNSIGNED_LONG, MPI_BOR, queue2_win);
                shmem_quiet();
            }
          }
        }
      }
    }
    /* End one-sided operations. */
    shmem_barrier_all();
//   MPI_Barrier(MPI_COMM_WORLD);
    //MPI_Win_fence(MPI_MODE_NOSUCCEED, queue2_win);
   // MPI_Win_fence(MPI_MODE_NOSUCCEED, pred2_win);
//   MPI_Barrier(MPI_COMM_WORLD);

    /* Test if there are any elements in the next-level queue (globally); stop
     * if none. */
    int any_set = 0;
    for (i = 0; i < queue_nwords; ++i) {
      if (queue_bitmap2[i] != 0) {any_set = 1;  break;}
    }
    MPI_Allreduce(MPI_IN_PLACE, &any_set, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
    if (!any_set) break;

    /* Swap queues and predecessor maps. */
    //printf("swapping maps. qb1.bp = %p, qb2.bp = %p\n", qb1_ag->buffer_ptr, qb2_ag->buffer_ptr);
//    {sharp_group_allocated_t temp = queue_bitmap1_ag; queue_bitmap1_ag = queue_bitmap2_ag, queue_bitmap2_ag = temp;}
//    {sharp_group_allocated_t temp = pred_ag; pred_ag = pred2_ag, pred2_ag = temp;}
{
    int64_t * temp = pred;
    pred = pred2;
    pred2 = temp;    
}
//    {sharp_group_allocated_t * temp = p_ag; p_ag = p2_ag; p2_ag = temp;}
{
    unsigned long * temp = queue_bitmap1;
    queue_bitmap1 = queue_bitmap2;
    queue_bitmap2 = temp;
}
//    {sharp_group_allocated_t * temp = qb1_ag; qb1_ag = qb2_ag; qb2_ag = temp;}
//    {unsigned long * temp = queue_bitmap1; queue_bitmap1 = queue_bitmap2; queue_bitmap2 = temp;}
//    {int64_t* temp = pred; pred = pred2; pred2 = temp;} 
    //printf("after swapping maps. qb1.bp = %p, qb2.bp = %p\n", qb1_ag->buffer_ptr, qb2_ag->buffer_ptr);
     
    //{MPI_Win temp = queue1_win; queue1_win = queue2_win; queue2_win = temp;}
//    {unsigned long* temp = queue_bitmap1; queue_bitmap1 = queue_bitmap2; queue_bitmap2 = temp;}
   // {MPI_Win temp = pred_win; pred_win = pred2_win; pred2_win = temp;}
//    {int64_t* temp = pred; pred = pred2; pred2 = temp;}*/
  }
    MPI_Barrier(MPI_COMM_WORLD);
//    MPI_Free_mem(local_vertices);
/*  MPI_Win_free(&pred_win);
  MPI_Win_free(&pred2_win);
  MPI_Win_free(&queue1_win);
  MPI_Win_free(&queue2_win); */
  MPI_Free_mem(local_vertices);
  MPI_Free_mem(dummy2);
  MPI_Free_mem(dummy3); 

  /* Clean up the predecessor map swapping since the surrounding code does not
   * allow the BFS to change the predecessor map pointer. */
  if (pred2 != orig_pred) {
    memcpy(orig_pred, pred2, nlocalverts * sizeof(int64_t));
//    sharp_destroy_allocation_group(&pred2_ag);
   // MPI_Free_mem(dummy1);
    //memset(pred2_ag.buffer_ptr, 0, nlocalverts * sizeof(int64_t));
  } else {
//    sharp_destroy_allocation_group(&pred_ag);
//    MPI_Free_mem(pred);
    memset(pred, 0, nlocalverts * sizeof(int64_t));
  }

  /* Change from special coding of predecessor map to the one the benchmark
   * requires. */
  size_t i;
  for (i = 0; i < nlocalverts; ++i) {
    if (orig_pred[i] < 0) {
      orig_pred[i] += nglobalverts;
    } else if (orig_pred[i] == INT64_MAX) {
      orig_pred[i] = -1;
    }
  }
}

void get_vertex_distribution_for_pred(size_t count, const int64_t* vertex_p, int* owner_p, size_t* local_p) {
  const int64_t* restrict vertex = vertex_p;
  int* restrict owner = owner_p;
  size_t* restrict local = local_p;
  ptrdiff_t i;
#pragma omp parallel for
  for (i = 0; i < (ptrdiff_t)count; ++i) {
    owner[i] = VERTEX_OWNER(vertex[i]);
    local[i] = VERTEX_LOCAL(vertex[i]);
  }
}

int64_t vertex_to_global_for_pred(int v_rank, size_t v_local) {
  return VERTEX_TO_GLOBAL(v_rank, v_local);
}

size_t get_nlocalverts_for_pred(void) {
  return g.nlocalverts;
}
