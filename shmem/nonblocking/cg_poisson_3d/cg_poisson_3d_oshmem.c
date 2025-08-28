/*
 * Copyright (c) 2006 The Trustees of Indiana University and Indiana
 *                    University Research and Technology
 *                    Corporation.  All rights reserved.
 * Copyright (c) 2006 The Technical University of Chemnitz. All 
 *                    rights reserved.
 *
 *  Authors:
 *    Peter Gottschling <pgottsch@osl.iu.edu>
 *    Torsten Hoefler <htor@cs.indiana.edu>
 *
 * Ported to OpenSHMEM by:
 *    [Your Name]
 *
 * Debug Output:
 *   Compile with DEBUG=1 to enable verbose debug output:
 *   make DEBUG=1
 *   or
 *   gcc -DDEBUG=1 -o cg_poisson_3d_oshmem cg_poisson_3d_oshmem.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <shmem.h>
#include <shmemx.h>
#include <mpi.h>
#include <math.h>
#include <sys/time.h>

/* Debug macro - compile with DEBUG=1 to enable debug output */
#ifndef DEBUG
#define DEBUG 0
#endif

#define DEBUG_PRINT(fmt, ...) do { if (DEBUG) printf(fmt, ##__VA_ARGS__); } while(0)

/* ugly global variables */
double gt1, gt2, gt3, gt4;
static long pSync[SHMEM_ALLTOALL_SYNC_SIZE];
static long max_pSync[SHMEM_REDUCE_SYNC_SIZE];
static size_t max_pWrk[SHMEM_REDUCE_MIN_WRKDATA_SIZE];

struct array_3d
{
  double  *array;
  int     x_dim, y_dim, z_dim,
          z_inc;                /* must be x_dim * y_dim */
  /* correspondingly y_inc == x_dim and x_inc == 1 */
};

/* plane within a 3D array */
struct plane
{
  double  *first;               /* pointer to first element in 3D array */
  int     major_dim, minor_dim, /* dimensions of plane */
          major_inc, minor_inc, /* increments in both directions */
          par_inc;              /* increment parallel to the plane */
};

/* Information about Cartesian grid */
struct grid_info
{
  /* neighbors */
  int left, right, top, bottom, front, back;
};

/* Set of buffers to send to or to receive from neighbors */
struct buffer_set
{
  double *start;
  int    left, right, top, bottom, front, back;
  int    d_left, d_right, d_top, d_bottom, d_front, d_back;
};

/* All communication related data */
struct comm_data_t
{
  struct grid_info   info;
  struct buffer_set  send_buffers, recv_buffers;
  int                non_blocking;
  int                rank;
  int                npes;
  int                dims[3];
  int                coords[3];
  shmem_req_h        req;  /* OpenSHMEM request handle for non-blocking operations */
  /* Buffer management for non-blocking operations */
  double            *padded_send;
  double            *padded_recv;
  size_t            nelems;
};
struct comm_data_t    comm_data;

/* =========
   Functions
   ========= 
*/

static inline int vector_size(struct array_3d *a)
{ 
  return a->x_dim * a->y_dim * a->z_dim;
}

static inline int x_plane_size(struct array_3d *a)
{
  return a->y_dim * a->z_dim;
}

static inline int y_plane_size(struct array_3d *a)
{
  return a->x_dim * a->z_dim;
}

static inline int z_plane_size(struct array_3d *a)
{
  return a->x_dim * a->y_dim;
}

/* block size of <n>th processor out of num_proc with <total> overall block size 
   n in [0, num_proc-1] */
static inline int nth_block_size(int n, int num_proc, int total)
{
  return total / num_proc + (total % num_proc > n);
}

void mult_boundaries(struct array_3d *v, struct buffer_set *recv_buffers);

void alloc_buffer_set(struct buffer_set *bs, struct grid_info *gi, struct array_3d *a);

void alloc_buffer_set(struct buffer_set *bs, struct grid_info *gi, struct array_3d *a) 
{
  /* memory must be contiguous in buffers (would be easier otherwise) */
  int     total;
  
  /* all sizes and displacements initialized to 0 */
  bs->left= bs->right= bs->front= bs->back= bs->top= bs->bottom= 0;
  bs->d_left= bs->d_right= bs->d_front= bs->d_back= bs->d_top= bs->d_bottom= 0;

  total= 0;
  if (gi->left != -1) {
    bs->left= x_plane_size(a);
    bs->d_left= total;
    total+= x_plane_size(a);
  }
  if (gi->right != -1) {
    bs->right= x_plane_size(a);
    bs->d_right= total;
    total+= x_plane_size(a);
  }

  if (gi->front != -1) {
    bs->front= y_plane_size(a);
    bs->d_front= total;
    total+= y_plane_size(a);
  }
  if (gi->back != -1) {
    bs->back= y_plane_size(a);
    bs->d_back= total;
    total+= y_plane_size(a);
  }

  if (gi->top != -1) {
    bs->top= z_plane_size(a);
    bs->d_top= total;
    total+= z_plane_size(a);
  }
  if (gi->bottom != -1) {
    bs->bottom= z_plane_size(a);
    bs->d_bottom= total;
    total+= z_plane_size(a);
  }
  
  if (total > 0) {
    bs->start = (double*) shmem_malloc(total * sizeof(double) * comm_data.npes);
    if (bs->start == NULL) {
      printf("Error: Failed to allocate buffer of size %zu bytes\n", total * sizeof(double) * comm_data.npes);
      shmem_finalize();
      exit(1);
    }
    
    if (comm_data.rank == 0) {
      DEBUG_PRINT("Buffer allocation: %d elements, %zu bytes per PE\n", total, total * sizeof(double));
    }
  } else {
    bs->start = NULL;
  } 
}

static inline double av(struct array_3d *a, int x, int y, int z)
{
  return a->array[z * a->z_inc + y * a->x_dim + x];
}

/* same as pointer */
static inline double* ap(struct array_3d *a, int x, int y, int z)
{
  return &a->array[z * a->z_inc + y * a->x_dim + x];
}

/* Returns the <x>th plane in x direction */
void x_plane(struct array_3d *a, int x, struct plane *p)
{
  assert(0 <= x && x < a->x_dim);

  p->first= a->array + x;
  p->major_dim= a->z_dim;
  p->minor_dim= a->y_dim;
  p->major_inc= a->z_inc;
  p->minor_inc= a->x_dim;
  p->par_inc= 1;
}

void left_plane(struct array_3d *a, struct plane *p)
{
  x_plane(a, 0, p);
}

void right_plane(struct array_3d *a, struct plane *p)
{
  x_plane(a, a->x_dim - 1, p);
}

/* Returns the <y>th plane in y direction */
void y_plane(struct array_3d *a, int y, struct plane *p)
{
  assert(0 <= y && y < a->y_dim);

  p->first= a->array + y * a->x_dim;
  p->major_dim= a->z_dim;
  p->minor_dim= a->x_dim;
  p->major_inc= a->z_inc;
  p->minor_inc= 1;
  p->par_inc=   a->x_dim;
}

void front_plane(struct array_3d *a, struct plane *p)
{
  y_plane(a, 0, p);
}

void back_plane(struct array_3d *a, struct plane *p)
{
  y_plane(a, a->y_dim - 1, p);
}

/* Returns the <z>th plane in z direction */
void z_plane(struct array_3d *a, int z, struct plane *p)
{
  assert(0 <= z && z < a->z_dim);

  p->first= a->array + z * a->z_inc;
  p->major_dim= a->y_dim;
  p->minor_dim= a->x_dim;
  p->major_inc= a->x_dim;
  p->minor_inc= 1;
  p->par_inc=   a->z_inc;
}

void bottom_plane(struct array_3d *a, struct plane *p)
{
  z_plane(a, 0, p);
}

void top_plane(struct array_3d *a, struct plane *p)
{
  z_plane(a, a->z_dim - 1, p);
}

void print_array_3d(struct array_3d *a)
{
  int x, y, z;
  for (z= 0; z < a->z_dim; z++) {
    printf("z= %i\n", z);
    for (y= 0; y < a->y_dim; y++) {
      for (x= 0; x < a->x_dim; x++) {
        printf("%f ", av(a, x, y, z));
      }
      printf("\n");
    }
    printf("\n");
  }
}

void parallel_print_array_3d(struct array_3d *a, struct comm_data_t *comm_data)
{
  int i;
  for (i= 0; i < comm_data->npes; i++) {
    shmem_barrier_all();
    if (i == comm_data->rank) {
      printf("PE %i:\n", comm_data->rank);
      print_array_3d(a);
    }
  }
}

void init_array_3d(struct array_3d *a, int xd, int yd, int zd)
{
  a->x_dim= xd;
  a->y_dim= yd;
  a->z_dim= zd;
  a->z_inc= xd * yd;
  a->array = (double*) shmem_malloc(vector_size(a) * sizeof(double) * comm_data.npes);
  if (a->array == NULL) {
    printf("Error: Failed to allocate array of size %zu bytes\n", vector_size(a) * sizeof(double) * comm_data.npes);
    shmem_finalize();
    exit(1);
  }
}

void iota_array_3d(struct array_3d *a)
{
  int i;
  for (i= 0; i < vector_size(a); i++) {
    a->array[i]= i;
  }
}

void set_array_3d(struct array_3d *a, double value)
{
  int i;
  for (i= 0; i < vector_size(a); i++) {
    a->array[i]= value;
  }
}

void set_plane(struct plane *p, double value)
{
  int i, j;
  for (i= 0; i < p->major_dim; i++) {
    for (j= 0; j < p->minor_dim; j++) {
      p->first[i * p->major_inc + j * p->minor_inc]= value;
    }
  }
}

static inline void check_same_dimensions(struct array_3d *v1, struct array_3d *v2)
{
  assert(v1->x_dim == v2->x_dim && v1->y_dim == v2->y_dim && v1->z_dim == v2->z_dim);
}

void volume_mult(struct array_3d *v_in, struct array_3d *v_out, struct comm_data_t *comm_data)
{
  int x, y, z;
  double *v= v_in->array;
  double *w= v_out->array;
  int x_dim= v_in->x_dim;
  int y_dim= v_in->y_dim;
  int z_dim= v_in->z_dim;
  int z_inc= v_in->z_inc;
  int y_inc= x_dim;
  int x_inc= 1;

  check_same_dimensions(v_in, v_out);

  /* multiply inner volume */
  if (comm_data->rank == 0) {
    DEBUG_PRINT("Multiplying inner volume in volume_mult...\n");
  }
  for (z= 1; z < z_dim - 1; z++) {
    for (y= 1; y < y_dim - 1; y++) {
      for (x= 1; x < x_dim - 1; x++) {
        w[z * z_inc + y * y_inc + x * x_inc]= 
          6.0 * v[z * z_inc + y * y_inc + x * x_inc] -
          v[z * z_inc + y * y_inc + (x-1) * x_inc] -
          v[z * z_inc + y * y_inc + (x+1) * x_inc] -
          v[z * z_inc + (y-1) * y_inc + x * x_inc] -
          v[z * z_inc + (y+1) * y_inc + x * x_inc] -
          v[(z-1) * z_inc + y * y_inc + x * x_inc] -
          v[(z+1) * z_inc + y * y_inc + x * x_inc];
      }
    }
  }

  /* multiply boundaries */
  if (comm_data->rank == 0) {
    DEBUG_PRINT("Multiplying boundaries in volume_mult...\n");
  }
  mult_boundaries(v_in, &comm_data->recv_buffers);
}

void volume_mult_simple(struct array_3d *v_in, struct array_3d *v_out)
{
  int x, y, z;
  double *v= v_in->array;
  double *w= v_out->array;
  int x_dim= v_in->x_dim;
  int y_dim= v_in->y_dim;
  int z_dim= v_in->z_dim;
  int z_inc= v_in->z_inc;
  int y_inc= x_dim;
  int x_inc= 1;

  /* multiply inner volume */
  for (z= 1; z < z_dim - 1; z++) {
    for (y= 1; y < y_dim - 1; y++) {
      for (x= 1; x < x_dim - 1; x++) {
        w[z * z_inc + y * y_inc + x * x_inc]= 
          6.0 * v[z * z_inc + y * y_inc + x * x_inc] -
          v[z * z_inc + y * y_inc + (x-1) * x_inc] -
          v[z * z_inc + y * y_inc + (x+1) * x_inc] -
          v[z * z_inc + (y-1) * y_inc + x * x_inc] -
          v[z * z_inc + (y+1) * y_inc + x * x_inc] -
          v[(z-1) * z_inc + y * y_inc + x * x_inc] -
          v[(z+1) * z_inc + y * y_inc + x * x_inc];
      }
    }
  }
}

void plane_buffer_mult(struct plane *p, double* buffer)
{
  int i, j;
  for (i= 0; i < p->major_dim; i++) {
    for (j= 0; j < p->minor_dim; j++) {
      buffer[i * p->major_dim + j]= p->first[i * p->major_inc + j * p->minor_inc];
    }
  }
}

void plane_buffer_copy(struct plane *p, double* buffer)
{
  int i, j;
  for (i= 0; i < p->major_dim; i++) {
    for (j= 0; j < p->minor_dim; j++) {
      p->first[i * p->major_inc + j * p->minor_inc]= buffer[i * p->major_dim + j];
    }
  }
}

static inline int
nb_rank(int rank, int dim, int dir, int dims[3], int my_coords[3])
{
  int coords[3];
  memcpy(coords, my_coords, 3 * sizeof(int));
  coords[dim]+= dir;
  if (coords[dim] < 0 || coords[dim] >= dims[dim]) {
    return -1;
  }
  return coords[0] + coords[1] * dims[0] + coords[2] * dims[0] * dims[1];
}

void init_processor_grid(struct comm_data_t *comm_data)
{
  int i, j, k;
  int dims[3];
  int coords[3];
  int rank = comm_data->rank;
  int npes = comm_data->npes;
  
  if (rank == 0) {
    DEBUG_PRINT("Initializing processor grid for %d PEs\n", npes);
  }

  /* Find the best 3D grid */
  dims[0] = dims[1] = dims[2] = 0;
  for (i = 1; i <= npes; i++) {
    for (j = 1; j <= npes; j++) {
      for (k = 1; k <= npes; k++) {
        if (i * j * k == npes) {
          if (i * j * k > dims[0] * dims[1] * dims[2]) {
            dims[0] = i;
            dims[1] = j;
            dims[2] = k;
          }
        }
      }
    }
  }

  comm_data->dims[0] = dims[0];
  comm_data->dims[1] = dims[1];
  comm_data->dims[2] = dims[2];
  
  if (rank == 0) {
    DEBUG_PRINT("Processor grid: %d x %d x %d\n", dims[0], dims[1], dims[2]);
  }

  /* Calculate my coordinates */
  coords[0] = rank % dims[0];
  coords[1] = (rank / dims[0]) % dims[1];
  coords[2] = rank / (dims[0] * dims[1]);

  comm_data->coords[0] = coords[0];
  comm_data->coords[1] = coords[1];
  comm_data->coords[2] = coords[2];
  
  if (rank == 0) {
    DEBUG_PRINT("PE coordinates: (%d, %d, %d)\n", coords[0], coords[1], coords[2]);
  }

  /* Find neighbors */
  comm_data->info.left = nb_rank(rank, 0, -1, dims, coords);
  comm_data->info.right = nb_rank(rank, 0, 1, dims, coords);
  comm_data->info.front = nb_rank(rank, 1, -1, dims, coords);
  comm_data->info.back = nb_rank(rank, 1, 1, dims, coords);
  comm_data->info.top = nb_rank(rank, 2, -1, dims, coords);
  comm_data->info.bottom = nb_rank(rank, 2, 1, dims, coords);
  
  if (rank == 0) {
    DEBUG_PRINT("Neighbors: left=%d, right=%d, front=%d, back=%d, top=%d, bottom=%d\n",
           comm_data->info.left, comm_data->info.right, comm_data->info.front,
           comm_data->info.back, comm_data->info.top, comm_data->info.bottom);
  }
}

void fill_buffers(struct array_3d *v, struct buffer_set *send_buffers)
{
  struct plane    v_left_plane, v_right_plane, v_front_plane, v_back_plane,
                  v_top_plane, v_bottom_plane;
  int             offset;

  /* Check if buffer is allocated */
  if (send_buffers->start == NULL) {
    printf("Error: send_buffers->start is NULL\n");
    shmem_finalize();
    exit(1);
  }

  if (send_buffers->left) {
    left_plane(v, &v_left_plane);
    plane_buffer_mult(&v_left_plane, send_buffers->start + send_buffers->d_left);
  }
  if (send_buffers->right) {
    right_plane(v, &v_right_plane);
    plane_buffer_mult(&v_right_plane, send_buffers->start + send_buffers->d_right);
  }
  if (send_buffers->front) {
    front_plane(v, &v_front_plane);
    plane_buffer_mult(&v_front_plane, send_buffers->start + send_buffers->d_front);
  }
  if (send_buffers->back) {
    back_plane(v, &v_back_plane);
    plane_buffer_mult(&v_back_plane, send_buffers->start + send_buffers->d_back);
  }
  if (send_buffers->top) {
    top_plane(v, &v_top_plane);
    plane_buffer_mult(&v_top_plane, send_buffers->start + send_buffers->d_top);
  }
  if (send_buffers->bottom) {
    bottom_plane(v, &v_bottom_plane);
    plane_buffer_mult(&v_bottom_plane, send_buffers->start + send_buffers->d_bottom);
  }
}

void start_send_boundaries(struct array_3d *v, struct comm_data_t *comm_data)
{
  struct buffer_set        send_buffers, recv_buffers;
  struct grid_info         gi;
  int                     i;
  double                  t1, t2;
  static size_t          nelems;
  static long            max_nelems;
  static double          *padded_send = NULL;
  static double          *padded_recv = NULL;
  static size_t          padded_size = 0;

  t1 = MPI_Wtime();

  /* copy data to send buffers */
  fill_buffers(v, &comm_data->send_buffers);

  /* Calculate total number of elements to exchange */
  nelems = comm_data->send_buffers.left + comm_data->send_buffers.right +
           comm_data->send_buffers.front + comm_data->send_buffers.back +
           comm_data->send_buffers.top + comm_data->send_buffers.bottom;

  /* Find maximum elements across all PEs */
  shmem_long_max_to_all(&max_nelems, (long*)&nelems, 1, 0, 0, comm_data->npes, max_pWrk, max_pSync);
  
  /* Add debug output for large problems */
  if (comm_data->rank == 0 && max_nelems > 100000) {
    DEBUG_PRINT("Warning: Large buffer size: %ld elements\n", max_nelems);
    DEBUG_PRINT("Buffer size in MB: %.2f\n", (double)(max_nelems * sizeof(double) * comm_data->npes) / (1024*1024));
  }

  /* Allocate new padded buffers */
  padded_send = (double*) shmem_malloc(max_nelems * sizeof(double));
  padded_recv = (double*) shmem_malloc(max_nelems * sizeof(double) * comm_data->npes);
  padded_size = max_nelems;
  
  /* Check if allocation succeeded */
  if (padded_send == NULL || padded_recv == NULL) {
    printf("Error: Failed to allocate communication buffers of size %ld\n", max_nelems);
    shmem_finalize();
    exit(1);
  }

  /* Copy data to padded buffer and zero out the rest */
  memcpy(padded_send, comm_data->send_buffers.start, nelems * sizeof(double));
  memset(padded_send + nelems, 0, (max_nelems - nelems) * sizeof(double));

  if (comm_data->non_blocking) {
    /* Use non-blocking alltoall for boundary exchange */
    shmemx_alltoall_nb(SHMEM_TEAM_WORLD,
                      padded_recv,                    /* dest */
                      padded_send,                    /* source */
                      max_nelems,                     /* nelems */
                      &comm_data->req);              /* req */
    
    /* For non-blocking, store buffer pointers for later cleanup */
    comm_data->padded_send = padded_send;
    comm_data->padded_recv = padded_recv;
    comm_data->nelems = nelems;
  } else {
    /* Use blocking alltoall for boundary exchange */
    shmem_alltoall64(padded_recv,                    /* target */
                    padded_send,                     /* source */
                    max_nelems,                      /* nelems */
                    0,                               /* PE_start */
                    0,                               /* logPE_stride */
                    comm_data->npes,                 /* PE_size */
                    pSync);                          /* pSync */
    
    /* Copy received data back to original buffer for blocking version */
    memcpy(comm_data->recv_buffers.start, padded_recv, nelems * sizeof(double) * comm_data->npes);
    
    /* Free buffers for blocking version */
    shmem_free(padded_send);
    shmem_free(padded_recv);
  }

  t2 = MPI_Wtime();
  gt1 += t2 - t1;
}

void finish_send_boundaries(struct comm_data_t *comm_data)
{
  double t1, t2;
  t1 = MPI_Wtime();
  
  if (comm_data->non_blocking) {
    /* Wait for non-blocking alltoall to complete */
    shmem_req_wait(&comm_data->req);
    
    /* Now safe to copy data and free buffers */
    memcpy(comm_data->recv_buffers.start, comm_data->padded_recv, comm_data->nelems * sizeof(double) * comm_data->npes);
    
    /* Free buffers after data is copied */
    shmem_free(comm_data->padded_send);
    shmem_free(comm_data->padded_recv);
  }
  
  t2 = MPI_Wtime();
  gt2 += t2 - t1;
}

void mult_boundaries(struct array_3d *v, struct buffer_set *recv_buffers)
{
  struct plane    v_left_plane, v_right_plane, v_front_plane, v_back_plane,
                  v_top_plane, v_bottom_plane;

  /* Check if buffer is allocated */
  if (recv_buffers->start == NULL) {
    printf("Error: recv_buffers->start is NULL\n");
    shmem_finalize();
    exit(1);
  }

  if (recv_buffers->left) {
    left_plane(v, &v_left_plane);
    plane_buffer_copy(&v_left_plane, recv_buffers->start + recv_buffers->d_left);
  }
  if (recv_buffers->right) {
    right_plane(v, &v_right_plane);
    plane_buffer_copy(&v_right_plane, recv_buffers->start + recv_buffers->d_right);
  }
  if (recv_buffers->front) {
    front_plane(v, &v_front_plane);
    plane_buffer_copy(&v_front_plane, recv_buffers->start + recv_buffers->d_front);
  }
  if (recv_buffers->back) {
    back_plane(v, &v_back_plane);
    plane_buffer_copy(&v_back_plane, recv_buffers->start + recv_buffers->d_back);
  }
  if (recv_buffers->top) {
    top_plane(v, &v_top_plane);
    plane_buffer_copy(&v_top_plane, recv_buffers->start + recv_buffers->d_top);
  }
  if (recv_buffers->bottom) {
    bottom_plane(v, &v_bottom_plane);
    plane_buffer_copy(&v_bottom_plane, recv_buffers->start + recv_buffers->d_bottom);
  }
}

void matrix_vector_mult(struct array_3d *v_in, struct array_3d *v_out,
                      struct comm_data_t *comm_data)
{
  double t1, t2;
  t1 = MPI_Wtime();

  check_same_dimensions(v_in, v_out);

  /* multiply inner volume */
  volume_mult_simple(v_in, v_out);

  /* exchange boundaries */
  if (comm_data->rank == 0) {
    DEBUG_PRINT("Starting boundary exchange...\n");
  }
  start_send_boundaries(v_in, comm_data);

  /* multiply boundaries */
  if (comm_data->rank == 0) {
    DEBUG_PRINT("Multiplying boundaries in matrix_vector_mult...\n");
  }
  mult_boundaries(v_in, &comm_data->recv_buffers);

  /* wait for all communication to complete */
  if (comm_data->rank == 0) {
    DEBUG_PRINT("Waiting for communication to complete...\n");
  }
  finish_send_boundaries(comm_data);

  t2 = MPI_Wtime();
  gt3 += t2 - t1;
}

void allocate_buffers(struct comm_data_t *comm_data, struct array_3d *x)
{
  alloc_buffer_set(&comm_data->send_buffers, &comm_data->info, x);
  alloc_buffer_set(&comm_data->recv_buffers, &comm_data->info, x);
}

void allocate_vectors(struct array_3d *x, struct array_3d *b, int argc, char** argv, 
                    struct comm_data_t *comm_data)
{
  int nx, ny, nz;
  int x_dim, y_dim, z_dim;

  if (argc != 4) {
    if (comm_data->rank == 0) {
      printf("Usage: %s <nx> <ny> <nz>\n", argv[0]);
    }
    shmem_finalize();
    exit(1);
  }

  nx= atoi(argv[1]);
  ny= atoi(argv[2]);
  nz= atoi(argv[3]);
  
  if (comm_data->rank == 0) {
    DEBUG_PRINT("Problem size: %d x %d x %d\n", nx, ny, nz);
    DEBUG_PRINT("Total global elements: %d\n", nx * ny * nz);
  }

  x_dim= nth_block_size(comm_data->coords[0], comm_data->dims[0], nx);
  y_dim= nth_block_size(comm_data->coords[1], comm_data->dims[1], ny);
  z_dim= nth_block_size(comm_data->coords[2], comm_data->dims[2], nz);

  init_array_3d(x, x_dim, y_dim, z_dim);
  init_array_3d(b, x_dim, y_dim, z_dim);
  
  if (comm_data->rank == 0) {
    DEBUG_PRINT("Local dimensions: x=%d, y=%d, z_dim=%d\n", x_dim, y_dim, z_dim);
    DEBUG_PRINT("Total local elements: %d\n", x_dim * y_dim * z_dim);
  }
}

void init_vectors(struct array_3d *x, struct array_3d *b, struct comm_data_t *comm_data)
{
  set_array_3d(x, 0.0);
  set_array_3d(b, 1.0);
}

void vector_assign_add(struct array_3d *v, struct array_3d *w)
{
  int i;
  for (i= 0; i < vector_size(v); i++) {
    v->array[i]+= w->array[i];
  }
}

double parallel_dot(struct array_3d *v, struct array_3d *w, struct comm_data_t *comm_data)
{
  int i;
  static double local_dot = 0.0;
  static double global_dot;

  for (i= 0; i < vector_size(v); i++) {
    local_dot+= v->array[i] * w->array[i];
  }

  shmem_double_sum_to_all(&global_dot, &local_dot, 1, 0, 0, comm_data->npes, NULL, 0);

  return global_dot;
}

int conjugate_gradient(struct array_3d *b, struct array_3d *x, double rel_error, struct comm_data_t *comm_data)
{
  struct array_3d       v, q, r;
  double               alpha, beta, rho, rho_old;
  int                  i;
  int                  max_iter= 200;
  double               t1, t2;

  t1 = MPI_Wtime();

  if (comm_data->rank == 0) {
    DEBUG_PRINT("Initializing conjugate gradient arrays...\n");
  }

  init_array_3d(&v, x->x_dim, x->y_dim, x->z_dim);
  init_array_3d(&q, x->x_dim, x->y_dim, x->z_dim);
  init_array_3d(&r, x->x_dim, x->y_dim, x->z_dim);

  /* r = b - A x */
  if (comm_data->rank == 0) {
    DEBUG_PRINT("Calculating initial residual...\n");
  }
  matrix_vector_mult(x, &v, comm_data);
  set_array_3d(&r, 0.0);
  vector_assign_add(&r, b);
  vector_assign_add(&r, &v);

  /* v = r */
  memcpy(v.array, r.array, vector_size(&r) * sizeof(double));

  if (comm_data->rank == 0) {
    DEBUG_PRINT("Calculating initial rho...\n");
  }
  rho= parallel_dot(&r, &r, comm_data);

  if (comm_data->rank == 0) {
    DEBUG_PRINT("Starting main iteration loop...\n");
  }

  for (i= 0; i < max_iter; i++) {
    if (i % 100 == 0 && comm_data->rank == 0) {
      DEBUG_PRINT("Iteration %d: starting matrix-vector multiply...\n", i);
    }
    matrix_vector_mult(&v, &q, comm_data);
    alpha= rho / parallel_dot(&v, &q, comm_data);
    vector_assign_add(x, &v);
    vector_assign_add(&r, &q);
    rho_old= rho;
    rho= parallel_dot(&r, &r, comm_data);
    
    if (i % 100 == 0 && comm_data->rank == 0) {
      DEBUG_PRINT("Iteration %d: residual = %e\n", i, sqrt(rho));
    }
    
    if (sqrt(rho) < rel_error) {
      if (comm_data->rank == 0) {
        DEBUG_PRINT("Converged at iteration %d with residual %e\n", i, sqrt(rho));
      }
      break;
    }
    beta= rho / rho_old;
    vector_assign_add(&v, &r);
  }
  
  if (i >= max_iter && comm_data->rank == 0) {
    DEBUG_PRINT("Warning: Maximum iterations (%d) reached without convergence\n", max_iter);
  }

  shmem_free(v.array);
  shmem_free(q.array);
  shmem_free(r.array);

  t2 = MPI_Wtime();
  gt4 += t2 - t1;

  if (comm_data->rank == 0) {
    DEBUG_PRINT("Conjugate gradient completed in %d iterations\n", i);
  }

  return i;
}

int main(int argc, char** argv) 
{
  struct array_3d       x, b;
//  struct comm_data_t    comm_data;
  int                   iter;
  double                rel_error= 1.0e-6;
  double                t1, t2, t3, t4;
  double                blocking_time, nonblocking_time;
  double                blocking_gt1, blocking_gt2, blocking_gt3, blocking_gt4;
  double                nonblocking_gt1, nonblocking_gt2, nonblocking_gt3, nonblocking_gt4;
  int                   i;

  shmem_init();

  if (shmem_my_pe() == 0) {
    DEBUG_PRINT("Starting CG Poisson 3D solver...\n");
  }

  comm_data.rank = shmem_my_pe();
  comm_data.npes = shmem_n_pes();
  
  /* Initialize buffer management fields */
  comm_data.padded_send = NULL;
  comm_data.padded_recv = NULL;
  comm_data.nelems = 0;

  /* Initialize timing variables */
  gt1 = gt2 = gt3 = gt4 = 0.0;

  /* Initialize pSync arrays */
  for (i = 0; i < SHMEM_ALLTOALL_SYNC_SIZE; i++) {
    pSync[i] = SHMEM_SYNC_VALUE;
  }
  for (i = 0; i < SHMEM_REDUCE_SYNC_SIZE; i++) {
    max_pSync[i] = SHMEM_SYNC_VALUE;
  }

  /* First run with blocking version */
  if (comm_data.rank == 0) {
    DEBUG_PRINT("\n=== Starting Blocking Version ===\n");
  }
  comm_data.non_blocking = 0;
  t1 = MPI_Wtime();

  init_processor_grid(&comm_data);
  allocate_vectors(&x, &b, argc, argv, &comm_data);
  allocate_buffers(&comm_data, &x);
  init_vectors(&x, &b, &comm_data);

  if (comm_data.rank == 0) {
    DEBUG_PRINT("Starting blocking conjugate gradient iterations...\n");
  }
  iter = conjugate_gradient(&b, &x, rel_error, &comm_data);

  t2 = MPI_Wtime();
  blocking_time = t2 - t1;
  
  if (comm_data.rank == 0) {
    DEBUG_PRINT("Blocking version completed in %.6f seconds\n", blocking_time);
  }

  /* Save blocking timing values */
  blocking_gt1 = gt1;
  blocking_gt2 = gt2;
  blocking_gt3 = gt3;
  blocking_gt4 = gt4;

  /* Free resources */
  if (comm_data.rank == 0) {
    DEBUG_PRINT("Cleaning up blocking version resources...\n");
  }
  shmem_free(x.array);
  shmem_free(b.array);
  shmem_free(comm_data.send_buffers.start);
  shmem_free(comm_data.recv_buffers.start);

  /* Reset timing variables */
  gt1 = gt2 = gt3 = gt4 = 0.0;

  /* Now run with non-blocking version */
  if (comm_data.rank == 0) {
    DEBUG_PRINT("\n=== Starting Non-Blocking Version ===\n");
  }
  comm_data.non_blocking = 1;
  t3 = MPI_Wtime();

  init_processor_grid(&comm_data);
  allocate_vectors(&x, &b, argc, argv, &comm_data);
  allocate_buffers(&comm_data, &x);
  init_vectors(&x, &b, &comm_data);

  if (comm_data.rank == 0) {
    DEBUG_PRINT("Starting non-blocking conjugate gradient iterations...\n");
  }
  iter = conjugate_gradient(&b, &x, rel_error, &comm_data);

  t4 = MPI_Wtime();
  nonblocking_time = t4 - t3;
  
  if (comm_data.rank == 0) {
    DEBUG_PRINT("Non-blocking version completed in %.6f seconds\n", nonblocking_time);
  }

  /* Save non-blocking timing values */
  nonblocking_gt1 = gt1;
  nonblocking_gt2 = gt2;
  nonblocking_gt3 = gt3;
  nonblocking_gt4 = gt4;

  if (comm_data.rank == 0) {
    DEBUG_PRINT("Cleaning up non-blocking version resources...\n");
  }

  if (comm_data.rank == 0) {
    printf("\n=== Results ===\n");
    printf("Converged after %i iterations\n", iter);
    printf("\nBlocking version:\n");
    printf("  Boundary exchange start: %f seconds\n", blocking_gt1);
    printf("  Boundary exchange finish: %f seconds\n", blocking_gt2);
    printf("  Matrix-vector multiply: %f seconds\n", blocking_gt3);
    printf("  Conjugate gradient: %f seconds\n", blocking_gt4);
    printf("  Total time: %f seconds\n", blocking_time);
    printf("\nNon-blocking version:\n");
    printf("  Boundary exchange start: %f seconds\n", nonblocking_gt1);
    printf("  Boundary exchange finish: %f seconds\n", nonblocking_gt2);
    printf("  Matrix-vector multiply: %f seconds\n", nonblocking_gt3);
    printf("  Conjugate gradient: %f seconds\n", nonblocking_gt4);
    printf("  Total time: %f seconds\n", nonblocking_time);
    printf("\nSpeedup: %f\n", blocking_time / nonblocking_time);
  }

  shmem_free(x.array);
  shmem_free(b.array);
  shmem_free(comm_data.send_buffers.start);
  shmem_free(comm_data.recv_buffers.start);

  if (comm_data.rank == 0) {
    DEBUG_PRINT("\n=== Program Complete ===\n");
  }

  shmem_finalize();
  return 0;
} 
