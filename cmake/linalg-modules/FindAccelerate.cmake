include( CheckCCompilerFlag )                                                        
check_c_compiler_flag( "-framework Accelerate" COMPILER_RECOGNIZES_ACCELERATE )      
if( COMPILER_RECOGNIZES_ACCELERATE )                                                 
  set( Accelerate_LIBRARIES "-framework Accelerate" CACHE STRING "Accelerate Libraries" FORCE)
  set( Accelerate_lp64_FOUND  TRUE  )                                                
  set( Accelerate_ilp64_FOUND FALSE )                                                
endif()                                                                              

include(FindPackageHandleStandardArgs)                                               
find_package_handle_standard_args( Accelerate                                        
  REQUIRED_VARS Accelerate_LIBRARIES                                                 
  HANDLE_COMPONENTS
) 
