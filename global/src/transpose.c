#include <stdio.h>

#define MAX 3

void 
trnm(int *a,int n) { 
  int s,*p,*q;
  int i,j,e;
  int nops=0;

  for(i=0,e=n-1; i<n-1 ;++i,--e,a+=n+1)
    for(p=a+1,q=a+n,j=0; j<e ;++j){
      s= *p; *p++ = *q; *q=s; q+=n;
      ++nops;
    }
  printf("\n# of operations in 2d = %d\n", nops);
}

void 
trnm3(int *a,int n) { 
  int s,*p,*q;
  int i,j,k,e,f;
  int nops=0;

  for(k=0,f=n-1; k<n-1 ;++k,--f)
    for(i=0; i<n ;++i,a+=n)
      for(p=a+1+k,q=a+n*n+k,j=0; j<f ;++j){
	s= *p; *p++ = *q; *q=s; q+=n*n;
	++nops;
      }
  printf("\n# of operations in 3d = %d\n", nops);
}

void 
trnm4(int *a,int n) { 
  int s,*p,*q;
  int i,j,k,m,e,f;
  int nops=0;

  for(m=0,f=n-1; m<n-1 ;++m,--f)
    for(k=0; k<n ;++k)
      for(i=0; i<n ;++i,a+=n)
	for(p=a+1+m,q=a+n*n*n+m,j=0; j<f ;++j){
	  s= *p; *p++ = *q; *q=s; q+=n*n*n;
	  ++nops;
	}
  printf("\n# of operations in 4d = %d\n", nops);
}

void
verify2d(int *a, int b[MAX][MAX]) {
  int i,j,sym[MAX][MAX];
  
  for(i=0; i<MAX; i++)
    for(j=0; j<MAX; j++)
	sym[i][j]= *a++ + b[i][j];
  
  for(i=0; i<MAX; i++)
    for(j=0; j<MAX; j++)
	if(sym[i][j] != sym[j][i]) {
	  printf("2D Transpose Failed\n\n");return;
	}
  printf("2D Transpose OK\n\n");
}

void
verify3d(int *a, int b[MAX][MAX][MAX]) {
  int i,j,k,sym[MAX][MAX][MAX];
  
  for(i=0; i<MAX; i++)
    for(j=0; j<MAX; j++)
      for(k=0; k<MAX; k++)
	sym[i][j][k]= *a++ + b[i][j][k];
  
  for(i=0; i<MAX; i++)
    for(j=0; j<MAX; j++)
      for(k=0; k<MAX; k++)
	if(sym[i][j][k] != sym[k][j][i]) {
	  printf("3D Transpose Failed\n\n");return;
	}
  printf("3D Transpose OK\n\n");
}

void
verify4d(int *a, int b[MAX][MAX][MAX][MAX]) {
  int i,j,k,m,sym[MAX][MAX][MAX][MAX];
  
  for(i=0; i<MAX; i++)
    for(j=0; j<MAX; j++)
      for(k=0; k<MAX; k++)
	for(m=0; m<MAX; m++) {
	  printf("a = %d\n", *a);
	  sym[i][j][k][m]= *a++ + b[i][j][k][m];
	}
  
  for(i=0; i<MAX; i++)
    for(j=0; j<MAX; j++)
      for(k=0; k<MAX; k++)
	for(m=0; m<MAX; m++)
	if(sym[i][j][k][m] != sym[m][k][j][i]) {
	  printf("4D Transpose Failed\n\n");return;
	}
  printf("4D Transpose OK\n\n");
}


void 
matrix2d() {
  int a[100], b[MAX][MAX];
  int i,j;
  
  for(i=0; i<MAX; i++)
    for(j=0; j<MAX; j++) {
      a[i*MAX+j] = i*MAX + j+1;
      b[i][j]    = i*MAX + j+1;
    }
  trnm(a, MAX);
  verify2d(a,b);
}

void 
matrix3d() {
  int a[100],b[MAX][MAX][MAX];
  int i,j,k;
  
  for(i=0; i<MAX; i++)
    for(j=0; j<MAX; j++)
      for(k=0; k<MAX; k++) {
	a[i*MAX*MAX+j*MAX+k] = i*MAX*MAX + j*MAX+k+1;
	b[i][j][k]           = i*MAX*MAX + j*MAX+k+1;
      }
  
  trnm3(a, MAX);
  verify3d(a,b);
}

void 
matrix4d() {
  int a[100],b[MAX][MAX][MAX][MAX];
  int i,j,k,m;
  
  for(i=0; i<MAX; i++)
    for(j=0; j<MAX; j++)
      for(k=0; k<MAX; k++)
	for(m=0; m<MAX; m++) {
	  a[i*MAX*MAX*MAX+j*MAX*MAX+k*MAX+m] = i*MAX*MAX*MAX+j*MAX*MAX+k*MAX+m;
	  b[i][j][k][m]                      = i*MAX*MAX*MAX+j*MAX*MAX+k*MAX+m;
	}
  
  trnm4(a, MAX);
  verify4d(a,b);
}



int 
main() {
  matrix2d();
  matrix3d();
  matrix4d();
}
