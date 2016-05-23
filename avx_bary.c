// /usr/local/gcc-5.1.0/bin/gcc -mavx -std=c99 -O2 avx_bary.c  // -O3 skips loop
// 3.661s vs 0.830s
// --or--
// . /usr/local/INTEL2013.sh
// icc -std=c99 -O1 avx_bary.c  // -O2 skips loop
// 4.076s vs 1.044s
//
// c.f. 1580 ticks for 1e7 reps, or 300 with -O3 !
// unroll is 510, or 80 with -O3

#include <stdlib.h>
#include <stdio.h>
#include <immintrin.h>
//#include <x86intrin.h>
#include <time.h>
#include <stdint.h>

#define ALIGN __attribute__ ((aligned (32)))

struct tetrahedron {
    float ALIGN T[9];
    float ALIGN ori[3];
};

void map(const struct tetrahedron * const restrict self,
         float * restrict dst,
         const float * const restrict src) {
    // Matrix Multiply dst=T*tmp; [3x3].[3x1]
    {
        const float * const T=self->T;
        const float * const o=self->ori;
        #pragma unroll
        for(int k=0;k<3;++k) {
            dst[k]=T[3*k]*(src[0]-o[0])
                +T[3*k+1]*(src[1]-o[1])
                +T[3*k+2]*(src[2]-o[2]);
        }
    }
    dst[3]=1.0f-dst[0]-dst[1]-dst[2];
}

__m256 r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15;

void map_avx(const struct tetrahedron * const restrict self,
         float (* restrict dst)[8],
         float (* const restrict src)[8]) {   // const??
    // Matrix Multiply dst=T*tmp; [3x3].[3x1]
    const float * const T=self->T;
    const float * const ori=self->ori;

    r0 = _mm256_load_ps(src[0]);
    r1 = _mm256_load_ps(src[1]);
    r2 = _mm256_load_ps(src[2]);
    r12 = _mm256_set1_ps(ori[0]);
    r13 = _mm256_set1_ps(ori[1]);
    r14 = _mm256_set1_ps(ori[2]);

    r3  = _mm256_set1_ps(T[0]);
    r4  = _mm256_set1_ps(T[1]);
    r5  = _mm256_set1_ps(T[2]);
    r6  = _mm256_set1_ps(T[3]);
    r7  = _mm256_set1_ps(T[4]);
    r8  = _mm256_set1_ps(T[5]);
    r9  = _mm256_set1_ps(T[6]);
    r10 = _mm256_set1_ps(T[7]);
    r11 = _mm256_set1_ps(T[8]);

    r0 = _mm256_sub_ps(r0,r12);
    r1 = _mm256_sub_ps(r1,r13);
    r2 = _mm256_sub_ps(r2,r14);

    r3  = _mm256_mul_ps(r0,r3);
    r4  = _mm256_mul_ps(r1,r4);
    r5  = _mm256_mul_ps(r2,r5);
    r6  = _mm256_mul_ps(r0,r6);
    r7  = _mm256_mul_ps(r1,r7);
    r8  = _mm256_mul_ps(r2,r8);
    r9  = _mm256_mul_ps(r0,r9);
    r10 = _mm256_mul_ps(r1,r10);
    r11 = _mm256_mul_ps(r2,r11);

    r3 = _mm256_add_ps(r3,r4);
    r6 = _mm256_add_ps(r6,r7);
    r9 = _mm256_add_ps(r9,r10);
    r3 = _mm256_add_ps(r3,r5);
    r6 = _mm256_add_ps(r6,r8);
    r9 = _mm256_add_ps(r9,r11);

    _mm256_store_ps(dst[0],r3);
    _mm256_store_ps(dst[1],r6);
    _mm256_store_ps(dst[2],r9);

    r12 = _mm256_set1_ps(1.0f);
    r12 = _mm256_sub_ps(r12,r3);
    r12 = _mm256_sub_ps(r12,r6);
    r12 = _mm256_sub_ps(r12,r9);
    _mm256_store_ps(dst[3],r12);
}

int main() {

  struct tetrahedron tetrads[5];
  uint64_t i,j,k,n=100000000;

if(1) {

  float r[8][3],lambdas[8][4];

  for(i=0; i<9; i++)  tetrads[0].T[i]=(float)i;//rand();
  for(i=0; i<3; i++)  tetrads[0].ori[i]=(float)i;//rand();
  for(i=0; i<3; i++)
    for(j=0; j<8; j++)
      r[j][i]=(float)(i*j);//rand();

  #pragma nounroll
  for(i=0; i<n; i++) {
    for(j=0; j<8; j++) {
      map(tetrads,lambdas[j],r[j]); } }

  i=6;
  printf("T=\t\t\tori=\tr=\tlambda=\n");
  for(j=0; j<3; j++) {
    for(k=0; k<3; k++)
      printf("%f\t",tetrads[0].T[j*3+k]);
    printf("%f\t",tetrads[0].ori[j]);
    printf("%f\t",r[j][i]);
    printf("%f\n",lambdas[j][i]); }
  printf("\t\t\t\t\t\t\t\t\t\t%f\n",lambdas[3][i]);

} else {

  float ALIGN r[3][8],lambdas[4][8];

  for(i=0; i<9; i++)  tetrads[0].T[i]=(float)i;//rand();
  for(i=0; i<3; i++)  tetrads[0].ori[i]=(float)i;//rand();
  for(i=0; i<8; i++)
    for(j=0; j<3; j++)
      r[j][i]=(float)(i*j);//rand();

  #pragma nounroll
  for(j=0; j<n; j++) {
    map_avx(tetrads,lambdas,r); }

  i=6;
  printf("T=\t\t\tori=\tr=\tlambda=\n");
  for(j=0; j<3; j++) {
    for(k=0; k<3; k++)
      printf("%f\t",tetrads[0].T[j*3+k]);
    printf("%f\t",tetrads[0].ori[j]);
    printf("%f\t",r[j][i]);
    printf("%f\n",lambdas[j][i]); }
  printf("\t\t\t\t\t\t\t\t\t\t%f\n",lambdas[3][i]);

}

 return(0);
 }
