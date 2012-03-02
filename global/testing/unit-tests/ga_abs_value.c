#include "ga.h"
#include "mock.h"
#include "ga_unit.h"

static void test(int shape_idx, int type_idx, int dist_idx)
{
    int type = TYPES[type_idx];
    int *dims = SHAPES[shape_idx];
    int ndim = SHAPES_NDIM[shape_idx];
    mock_ga_t *mock_a, *result_a;
    int g_a;

    /* create the local array and result array */
    mock_a = Mock_Create(type, ndim, dims, "mock", NULL);
    result_a = Mock_Create(type, ndim, dims, "mock", NULL);

    /* create the global array */
    g_a = NGA_Create(type, ndim, dims, "g_a", NULL);

    /* create meaningful data for local array */
    mock_data(mock_a, g_a);

    /* init global array with same data as local array */
    mock_to_global(mock_a, g_a);

    /* call the local routine */
    Mock_Abs_value(mock_a);

    /* call the global routine */
    GA_Abs_value(g_a);

    /* get the results from the global array */
    global_to_mock(g_a, result_a);

    /* compare the results */
    if (neq_mock(mock_a, result_a)) {
        GA_Error("failure", 1);
    }

    /* clean up */
    Mock_Destroy(mock_a);
    Mock_Destroy(result_a);
    GA_Destroy(g_a);
}

int main(int argc, char **argv)
{
    TEST_SETUP;

    int shape_idx, type_idx, dist_idx;
    for (shape_idx=0; shape_idx < NUM_SHAPES; ++shape_idx) {
        for (type_idx=0; type_idx < NUM_TYPES; ++type_idx) {
            for (dist_idx=0; dist_idx < NUM_DISTS; ++dist_idx) {
                printf("%s %s %s\n",
                        SHAPE_NAMES[shape_idx],
                        TYPE_NAMES[type_idx],
                        DIST_NAMES[dist_idx]
                        );
                test(shape_idx, type_idx, dist_idx);
            }
        }
    }

    TEST_TEARDOWN;
    return 0;
}
