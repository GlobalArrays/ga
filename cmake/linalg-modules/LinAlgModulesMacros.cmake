set( LINALG_MACROS_DIR ${CMAKE_CURRENT_LIST_DIR} )

function( install_linalg_modules _dest_dir )

install( DIRECTORY ${LINALG_MACROS_DIR}
         DESTINATION ${${_dest_dir}} 
         FILES_MATCHING 
           REGEX "Find.*cmake" 
           REGEX "util/.*"
           REGEX ".*git.*"      EXCLUDE
           REGEX ".*examples.*" EXCLUDE
)

endfunction()
