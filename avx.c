//. /usr/local/INTEL2013.sh
//gcc -mavx -std=c99 avx.c

#include <stdlib.h>
#include <stdio.h>
#include <immintrin.h>
#include <x86intrin.h>

#define ALIGN __attribute__ ((aligned (32)))

int main() {
  __m256d a, b, c, d;

  double ALIGN temp[16];
  for(int i = 0; i < 16; i++) {
    temp[i] = (double) i;
  }
  a = _mm256_load_pd(temp);
  b = _mm256_load_pd(temp+4);
  c = _mm256_load_pd(temp+8);

  //d = _mm256_fmadd_pd(a,b,c);
  //d = __builtin_ia32_vfmaddpd256(a,b,c);
  d = _mm256_add_pd(a,b);
  //d = __builtin_ia32_addpd256(a,b);
  _mm256_store_pd(temp, d);
  printf("%e\n", temp[0]);
  
 return(0);
 }
