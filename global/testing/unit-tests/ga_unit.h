#ifndef GA_UNIT_H
#define GA_UNIT_H

#define GA_PRINT_MSG() printf("Test Completed\n")

#define GA_COMPLETE_MSG() printf("Test Completed\n")

#define GA_ERROR_MSG() printf("GA ERROR\n")

#define GA_ERROR_MSG2() printf("GA ERROR\n")

#define NUM_TYPES 7
int TYPES[NUM_TYPES] = {
    C_INT,
    C_LONG,
    C_LONGLONG,
    C_FLOAT,
    C_DBL,
    C_SCPL,
    C_DCPL,
};

char* TYPE_NAMES[NUM_TYPES] = {
    "C_INT",
    "C_LONG",
    "C_LONGLONG",
    "C_FLOAT",
    "C_DBL",
    "C_SCPL",
    "C_DCPL",
};

enum dist_type {
    DIST_REGULAR=0,
    DIST_CHUNK,
    DIST_IRREGULAR,
    DIST_BLOCK_CYCLIC,
    DIST_SCALAPACK,
    DIST_RESTRICTED,
    NUM_DISTS,
};

int DIST_TYPES[NUM_DISTS] = {
    DIST_REGULAR,
    DIST_CHUNK,
    DIST_IRREGULAR,
    DIST_BLOCK_CYCLIC,
    DIST_SCALAPACK,
    DIST_RESTRICTED,
};

char* DIST_NAMES[NUM_DISTS] = {
    "DIST_REGULAR",
    "DIST_CHUNK",
    "DIST_IRREGULAR",
    "DIST_BLOCK_CYCLIC",
    "DIST_SCALAPACK",
    "DIST_RESTRICTED",
};

#define NUM_SHAPES 3
static int SHAPES_ONE[] = {2,3};
#define    SHAPES_ONE_NDIM 2
#define    SHAPES_ONE_NAME "2x3"
static int SHAPES_TWO[] = {2,3,4};
#define    SHAPES_TWO_NDIM 3
#define    SHAPES_TWO_NAME "2x3x4"
static int SHAPES_THREE[] = {2,3,4,5};
#define    SHAPES_THREE_NDIM 4
#define    SHAPES_THREE_NAME "2x3x4x5"

static int* SHAPES[] = {
    SHAPES_ONE,
    SHAPES_TWO,
    SHAPES_THREE,
    NULL
};
static char* SHAPE_NAMES[] = {
    SHAPES_ONE_NAME,
    SHAPES_TWO_NAME,
    SHAPES_THREE_NAME,
    NULL
};
static int SHAPES_NDIM[] = {
    SHAPES_ONE_NDIM,
    SHAPES_TWO_NDIM,
    SHAPES_THREE_NDIM,
};

//#define OP_TYPES 6
//char operators[OP_TYPES] = {'+', '*', 'max', 'min', 'absmax', 'absmin'};

#define TEST_SETUP    GA_Initialize_args(&argc, &argv)
#define TEST_TEARDOWN GA_Terminate(); MPI_Finalize()

#endif
