divert(-1)
  # what kind of data type to test?
  define(m4_test_int, `yes')
  define(m4_test_dbl, `yes')
  define(m4_test_dcpl, `yes')

  # test dimension from which to which?
  define(m4_dim_from, 1)
  define(m4_dim_to, 7)

  # patch related
  define(m4_test_NGA_FILL_PATCH, `yes')
  define(m4_test_NGA_COPY_PATCH, `yes')
  define(m4_test_NGA_SCALE_PATCH, `yes')
  # due to stack limits, GNU m4 might be needed to process this one
  define(m4_test_NGA_ADD_PATCH, `no')
  # depending on the data type, the functions are NGA_IDOT_PATCH,
  # NGA_DDOT_PATCH, and NGA_ZDOT_PATCH, corresponding to Integer,
  # Double, and Double Complex data types, respectively.
  define(m4_test_NGA_DOT_PATCH, `yes')
divert
include(`ngatest_src/ngatest.def')
