# First record of input is a list of flags.

# Copy rest of input to output obeying nested IF-ELSEIF-ELSE-ENDIF
# statements which must commence at beginning of the line.

# Syntax is simple ... IF flag-list
#                      ELSEIF flag-list
#                      ELSE
#                      ENDIF
# where flag-list is a list of whitespace separated flags. If any of
# the flags in the flag list match those specified in the first line
# of the input the whole flag-list is evaluated as true, otherwise
# it is false.

# Parsing of blocks works if the blocks are logically consistent
# but it won't detect any incorrectly nested blocks etc. unless
# the no. of IF & ENDIF's don't match.

BEGIN {
       TRUE    = 1;
       FALSE   = 0;
       nflag   = 0; 
       iflevel = 0;
       error   = FALSE;
       output[0] = TRUE;
      }

# First record ... read flags
NR == 1 {
         for (i=1; i<=NF; i++) {
           flag[i]=$i; 
           nflag++;
         }
         next;
        }

# ENDIF
/^ENDIF/ {
          iflevel--;
          if(iflevel < 0) {
            printf "!! Error on input line=%d: %s\n", NR, "Incorrect IF nest";
            print;
            error = TRUE;
            exit 1;
          }
          next;
         }

# IF flag-list statement
/^IF/ {
       value = FALSE; # Old awk has no functions ... evaluate flags
       for (i=2; i<=NF; i++)
         for (j=1; j<=nflag; j++)
           if ($i == flag[j])
             value = TRUE; 

       iflevel++; 
       expr[iflevel] = value;
       if ((output[iflevel-1] == TRUE) && (expr[iflevel] == TRUE))
         output[iflevel] = TRUE;
       else
         output[iflevel] = FALSE;
       next;
      }

# ELSEIF flag-list
/^ELSEIF/ {
           value = FALSE; # Old awk has no functions ... evaluate flags
           for (i=2; i<=NF; i++)
             for (j=1; j<=nflag; j++)
               if ($i == flag[j])
                 value = TRUE; 

           if (expr[iflevel] == TRUE)
             output[iflevel] = FALSE  # Already done a block of this IF
           else {
             expr[iflevel] = value;
             if ((output[iflevel-1] == TRUE) && (expr[iflevel] == TRUE))
               output[iflevel] = TRUE;
             else
               output[iflevel] = FALSE;
           }
           next;
          }

# ELSE
/^ELSE/ {
         if (expr[iflevel] == TRUE)
           output[iflevel] = FALSE  # Already done a block of this IF
         else
           output[iflevel] = output[iflevel-1];
         next;
        }

# This prints stuff that has print enabled in this IF block
(output[iflevel] == TRUE) {print}

# Sanity check at the end
END {
     if(iflevel < 0) {
       print "!! Incorrect IF nesting detected at end of input"
       exit 1;
     }
    }

