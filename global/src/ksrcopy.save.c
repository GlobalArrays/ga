void CopyTo(src0,dest0,n)
double *src0, *dest0;
long n;
{
register long i,m;
register double a, b, c, d, e, f, g, h;
double *src, *dest;
     src = src0;
     dest= dest0;
     m =  n/128;
     for (i = 0; i< m; i++){
       a = src[0];
       b = src[1];
       c = src[2];
       d = src[3];
       e = src[4];
       f = src[5];
       g = src[6];
       h = src[7];

       dest[0] = a;
       dest[1] = b;
       dest[2] = c;
       dest[3] = d;
       dest[4] = e;
       dest[5] = f;
       dest[6] = g;
       dest[7] = h;

       a = src[8];
       b = src[9];
       c = src[10];
       d = src[11];
       e = src[12];
       f = src[13];
       g = src[14];
       h = src[15];
       dest[8]  = a;
       dest[9]  = b;
       dest[10] = c;
       dest[11] = d;
       dest[12] = e;
       dest[13] = f;
       dest[14] = g;
       dest[15] = h;

       dest += 16;
       src  += 16;
     }
     m = n%128;
     m = m/8;
     for (i = 0; i< m; i++)
       dest[i] =  src[i];
}


void CopyFrom(src0,dest0,n)
double *src0, *dest0;
long n;
{
register long i,m;
register double a, b, c, d, e, f, g, h;
double *src, *dest;
     src = src0;
     dest= dest0;
     m =  n/128;
     for (i = 0; i< m; i++){
       a = src[0];
       b = src[1];
       c = src[2];
       d = src[3];
       e = src[4];
       f = src[5];
       g = src[6];
       h = src[7];

       dest[0] = a;
       dest[1] = b;
       dest[2] = c;
       dest[3] = d;
       dest[4] = e;
       dest[5] = f;
       dest[6] = g;
       dest[7] = h;

       a = src[8];
       b = src[9];
       c = src[10];
       d = src[11];
       e = src[12];
       f = src[13];
       g = src[14];
       h = src[15];
       dest[8]  = a;
       dest[9]  = b;
       dest[10] = c;
       dest[11] = d;
       dest[12] = e;
       dest[13] = f;
       dest[14] = g;
       dest[15] = h;

       dest += 16;
       src  += 16;
     }
     m = n%128;
     m = m/8;
     for (i = 0; i< m; i++)
       dest[i] =  src[i];
}


void Accum(alpha,src0,dest0,n)
double *src0, *dest0, alpha;
long n;
{
register long i,m;
register double a, b, c, d, e, f, g, h;
double *src, *dest;
     src = src0;
     dest= dest0;
     m =  n/16;
/*     _gspwt(dest0);*/
     for (i = 0; i< m; i++){
       a = src[0];
       b = src[1];
       c = src[2];
       d = src[3];
       e = src[4];
       f = src[5];
       g = src[6];
       h = src[7];

       dest[0] += a*alpha;
       dest[1] += b*alpha;
       dest[2] += c*alpha;
       dest[3] += d*alpha;
       dest[4] += e*alpha;
       dest[5] += f*alpha;
       dest[6] += g*alpha;
       dest[7] += h*alpha;

       a = src[8];
       b = src[9];
       c = src[10];
       d = src[11];
       e = src[12];
       f = src[13];
       g = src[14];
       h = src[15];
       dest[8]  += a*alpha;
       dest[9]  += b*alpha;
       dest[10] += c*alpha;
       dest[11] += d*alpha;
       dest[12] += e*alpha;
       dest[13] += f*alpha;
       dest[14] += g*alpha;
       dest[15] += h*alpha;

       dest += 16;
       src  += 16;
     }
     m = n%16;
     for (i = 0; i< m; i++)
       dest[i] += alpha*src[i];
/*     _rsp(dest0);*/
}

