// PATH=$HOME/local/bin:$PATH /usr/local/gcc-5.1.0/bin/gcc -mavx2 -std=c99 -mfma --save-temps -O3 avx_bary2.c  (needs binutils 2.25)
// 0.635s vs 0.392s
// --or--
// . /usr/local/INTEL2013.sh
// icc -std=c99 -O3 avx_bary2.c
// 0.526s vs 0.420s

#include <stdlib.h>
#include <stdio.h>
#include <immintrin.h>
//#include <x86intrin.h>
#include <time.h>
#include <stdint.h>

#define ALIGN __attribute__ ((aligned (32)))
typedef uint16_t TPixel; /* Note (3) */

// unsigned=32-bit int

void interp(TPixel * restrict dst,  // 1
         TPixel * restrict src,  // 1
         const unsigned * const restrict zero,  // 3
         const unsigned * const restrict one,   // 3
         const float * const restrict frac,   // 3    btw [0,1)
         const unsigned * const restrict strides) {  // 3
    float tmp,c00,c10,c01,c11,c0,c1;

    tmp = (1.0f-frac[0]);
    c00 = src[zero[0]*strides[0]+zero[1]*strides[1]+zero[2]*strides[2]]*tmp
        + src[ one[0]*strides[0]+zero[1]*strides[1]+zero[2]*strides[2]]*frac[0];
    c10 = src[zero[0]*strides[0]+ one[1]*strides[1]+zero[2]*strides[2]]*tmp
        + src[ one[0]*strides[0]+ one[1]*strides[1]+zero[2]*strides[2]]*frac[0];
    c01 = src[zero[0]*strides[0]+zero[1]*strides[1]+ one[2]*strides[2]]*tmp
        + src[ one[0]*strides[0]+zero[1]*strides[1]+ one[2]*strides[2]]*frac[0];
    c11 = src[zero[0]*strides[0]+ one[1]*strides[1]+ one[2]*strides[2]]*tmp
        + src[ one[0]*strides[0]+ one[1]*strides[1]+ one[2]*strides[2]]*frac[0];
    tmp = (1.0f-frac[1]);
    c0 = c00*tmp + c10*frac[1];
    c1 = c01*tmp + c11*frac[1];
    *dst = c0*(1.0f-frac[2]) + c1*frac[2];
}

__m256 r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15;
__m256i ir0,ir1,ir2,ir3,ir4,ir5,ir6,ir7,ir8,ir9,ir10,ir11,ir12,ir13,ir14,ir15;

void interp_avx(TPixel * restrict dst,  // 1
         TPixel * restrict src,  // 1
         unsigned (* const restrict zero)[8],  // 3
         unsigned (* const restrict one)[8],   // 3
         float (* const restrict frac)[8],   // 3 
         unsigned * const restrict strides) {  // 3

    ir0  = _mm256_set1_epi32(strides[0]);
    ir1  = _mm256_set1_epi32(strides[1]);
    ir2  = _mm256_set1_epi32(strides[2]);
    ir3  = _mm256_load_si256((__m256i *)(zero+0));
    ir4  = _mm256_load_si256((__m256i *)(zero+1));
    ir5  = _mm256_load_si256((__m256i *)(zero+2));
    ir6  = _mm256_load_si256((__m256i *)(one+0));
    ir7  = _mm256_load_si256((__m256i *)(one+1));
    ir8  = _mm256_load_si256((__m256i *)(one+2));
    ir10 = _mm256_mullo_epi32(ir3,ir0);      // zero[0]*strides[0]
    ir11 = _mm256_mullo_epi32(ir4,ir1);      // zero[1]*strides[1]
    ir12 = _mm256_mullo_epi32(ir5,ir2);      // zero[2]*strides[2]
    ir13 = _mm256_mullo_epi32(ir6,ir0);      // one[0]*strides[0]
    ir14 = _mm256_mullo_epi32(ir7,ir1);      // one[1]*strides[1]
    ir15 = _mm256_mullo_epi32(ir8,ir2);      // one[2]*strides[2]


    ir4 = _mm256_add_epi32(ir11,ir12);   // (        zero[1]+zero[2])*strides[]
    ir5 = _mm256_add_epi32(ir14,ir12);   // (         one[1]+zero[2])*strides[]
    ir6 = _mm256_add_epi32(ir4,ir10);    // (zero[0]+zero[1]+zero[2])*strides[]
    ir7 = _mm256_add_epi32(ir4,ir13);    // ( one[0]+zero[1]+zero[2])*strides[]
    ir8 = _mm256_add_epi32(ir5,ir10);    // (zero[0]+ one[1]+zero[2])*strides[]
    ir9 = _mm256_add_epi32(ir5,ir13);    // ( one[0]+ one[1]+zero[2])*strides[]

    ir6 = _mm256_i32gather_epi32((int*)src,ir6,2);
    ir7 = _mm256_i32gather_epi32((int*)src,ir7,2);
    ir8 = _mm256_i32gather_epi32((int*)src,ir8,2);
    ir9 = _mm256_i32gather_epi32((int*)src,ir9,2);


    ir0 = _mm256_add_epi32(ir11,ir15);
    ir1 = _mm256_add_epi32(ir14,ir15);
    ir2 = _mm256_add_epi32(ir0,ir10);
    ir3 = _mm256_add_epi32(ir0,ir13);
    ir4 = _mm256_add_epi32(ir1,ir10);
    ir5 = _mm256_add_epi32(ir1,ir13);

    ir2 = _mm256_i32gather_epi32((int*)src,ir2,2);
    ir3 = _mm256_i32gather_epi32((int*)src,ir3,2);
    ir4 = _mm256_i32gather_epi32((int*)src,ir4,2);
    ir5 = _mm256_i32gather_epi32((int*)src,ir5,2);


    r0  = _mm256_set1_ps(1.0f);
    r1  = _mm256_load_ps(frac[0]);       // frac
    r0  = _mm256_sub_ps(r0,r1);          // tmp=1-frac
    ir10 = _mm256_set1_epi32(0x0000FFFF);  // assumes little-endian?


    ir6 = _mm256_and_si256(ir6,ir10);
    ir7 = _mm256_and_si256(ir7,ir10);
    ir8 = _mm256_and_si256(ir8,ir10);
    ir9 = _mm256_and_si256(ir9,ir10);

    r6 = _mm256_cvtepi32_ps(ir6);
    r7 = _mm256_cvtepi32_ps(ir7);
    r8 = _mm256_cvtepi32_ps(ir8);
    r9 = _mm256_cvtepi32_ps(ir9);
    r6  = _mm256_mul_ps(r6,r0);
    r7  = _mm256_mul_ps(r7,r1);
    r8  = _mm256_mul_ps(r8,r0);
    r9  = _mm256_mul_ps(r9,r1);
    r12 = _mm256_add_ps(r6,r7);  // c00
    r11 = _mm256_add_ps(r8,r9);  // c10


    ir2 = _mm256_and_si256(ir2,ir10);
    ir3 = _mm256_and_si256(ir3,ir10);
    ir4 = _mm256_and_si256(ir4,ir10);
    ir5 = _mm256_and_si256(ir5,ir10);

    r2 = _mm256_cvtepi32_ps(ir2);
    r3 = _mm256_cvtepi32_ps(ir3);
    r4 = _mm256_cvtepi32_ps(ir4);
    r5 = _mm256_cvtepi32_ps(ir5);
    r2 = _mm256_mul_ps(r2,r0);
    r3 = _mm256_mul_ps(r3,r1);
    r4 = _mm256_mul_ps(r4,r0);
    r5 = _mm256_mul_ps(r5,r1);
    r14 = _mm256_add_ps(r2,r3);  // c01
    r13 = _mm256_add_ps(r4,r5);  // c11


    r0  = _mm256_set1_ps(1.0f);
    r1  = _mm256_load_ps(frac[1]);       // frac
    r2  = _mm256_sub_ps(r0,r1);          // tmp=1-frac
    r12 = _mm256_mul_ps(r12,r2);
    r14 = _mm256_mul_ps(r14,r2);
    r12 = _mm256_fmadd_ps(r1,r11,r12);   // c0
    r14 = _mm256_fmadd_ps(r1,r13 ,r14);  // c1

    r1  = _mm256_load_ps(frac[2]);       // frac
    r2  = _mm256_sub_ps(r0,r1);          // tmp=1-frac
    r12 = _mm256_mul_ps(r12,r2);
    r12 = _mm256_fmadd_ps(r1,r14,r12);
    ir12 = _mm256_cvtps_epi32(r12);
    _mm256_store_si256((__m256i *)dst,ir12);
}

int main() {

  unsigned ALIGN strides[] = {1,100,10000,1000000};
  TPixel *src, dst[16];
  uint64_t i,j,k,l,n=10000000;
  
  src=(TPixel*)malloc(strides[3]*sizeof(TPixel));
  for(i=0; i<strides[3]; i++)
    src[i]=i;

if(0) {

  unsigned ALIGN zero[8][3],one[8][3];
  float ALIGN frac[8][3];
  for(j=0; j<8; j++)
    for(i=0; i<3; i++) {
      zero[j][i]=(unsigned)((float)rand()/(float)RAND_MAX*98.0f);
      one[j][i]=zero[j][i]+1;
      frac[j][i]=(float)rand()/(float)RAND_MAX; }

  #pragma nounroll
  for(i=0; i<n; i++) {
    for(j=0; j<8; j++) {
      interp(dst+j, src, // 1
               zero[j],  // 3
               one[j],   // 3
               frac[j],   // 3
               strides); } } // 3

  printf("zero=\tone=\tfrac=\t\n");
  i=1;
  for(j=0; j<3; j++) {
    printf("%d\t",zero[i][j]);
    printf("%d\t",one[i][j]);
    printf("%f\t\n",frac[i][j]); }
  printf("src=\t\n");
  for(j=0; j<2; j++) { for(k=0; k<2; k++) {
    for(l=0; l<2; l++)
      printf("%d\t", src[(zero[i][0]+j)*strides[0]+
                         (zero[i][1]+k)*strides[1]+
                         (zero[i][2]+l)*strides[2]]);
    printf("\n"); } }
  printf("dst=%d\n",dst[i]);

} else {

  unsigned ALIGN zero[3][8],one[3][8];
  float ALIGN frac[3][8];
  for(j=0; j<3; j++)
    for(i=0; i<8; i++) {
      zero[j][i]=(unsigned)((float)rand()/(float)RAND_MAX*98.0f);
      one[j][i]=zero[j][i]+1;
      frac[j][i]=(float)rand()/(float)RAND_MAX; }

  #pragma nounroll
  for(i=0; i<n; i++) {
    interp_avx(dst, src, // 1
             zero,  // 3
             one,   // 3
             frac,   // 3
             strides); } // 3

  printf("zero=\tone=\tfrac=\t\n");
  i=1;
  for(j=0; j<3; j++) {
    printf("%d\t",zero[j][i]);
    printf("%d\t",one[j][i]);
    printf("%f\t\n",frac[j][i]); }
  printf("src=\t\n");
  for(j=0; j<2; j++) { for(k=0; k<2; k++) {
    for(l=0; l<2; l++)
      printf("%d\t", src[(zero[0][i]+j)*strides[0]+
                         (zero[1][i]+k)*strides[1]+
                         (zero[2][i]+l)*strides[2]]);
    printf("\n"); } }
  printf("dst=%d\n",((int*)dst)[i]);

}

 return(0);
 }
