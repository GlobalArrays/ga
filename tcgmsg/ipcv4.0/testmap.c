/* $Header: /tmp/hpctools/ga/tcgmsg/ipcv4.0/testmap.c,v 1.1.1.1 1994-03-29 06:44:51 d3g681 Exp $ */

#include <stdio.h>

long ncols;
long nrows;


/* Given the col and row no. return the actual process no. */
#define MAP(Row,Col) (ncols*(Row) + (Col))

/* Given the node return the row no. */
#define ROW(Node) ((Node) / ncols)

/* Given the node return the column no. */
#define COL(Node) ((Node) - ncols*((Node)/ncols))

int main(argc, argv)
     int argc;
     char **argv;
{

  long row, col, node, data, type, len, me;

  (void) printf("Input nrows, ncols ");
  (void) scanf("%d %d",&nrows, &ncols);
  
  node = 0;
  type = 1;
  len = 4;

  for (me=0; me<(nrows*ncols); me++)
    (void) printf(" me=%d row=%d col=%d map=%d\n",me,
		  ROW(me),COL(me),MAP(ROW(me),COL(me)));
  return 0;
}
