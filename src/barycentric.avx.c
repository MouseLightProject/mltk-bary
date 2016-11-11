#include <resamplers.h>
#include <matrix.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>

#include<stdio.h>
static void default_reporter(const char* msg, const char* expr, const char* file, int line, const char* function,void* usr) {
    (void)usr;
    printf("%s(%d) - %s()\n\t%s\n\t%s\n",file,line,function,msg,expr);
}

typedef void (*reporter_t)(const char* msg, const char* expr, const char* file, int line, const char* function,void* usr);

static void*      reporter_context_=0;
static reporter_t error_  =&default_reporter;
static reporter_t warning_=&default_reporter;
static reporter_t info_   =&default_reporter;

#define ERR(e,msg)  error_(msg,#e,__FILE__,__LINE__,__FUNCTION__,reporter_context_)
#define WARN(e,msg) warning_(msg,#e,__FILE__,__LINE__,__FUNCTION__,reporter_context_)
#define INFO(e,msg) info_(msg,#e,__FILE__,__LINE__,__FUNCTION__,reporter_context_)
#define ALIGN __attribute__ ((aligned (32)))

#define ASSERT(e) do{if(!(e)) {ERR(e,"Expression evaluated as false."); return 1; }}while(0)

static void useReporters(
    void (*error)  (const char* msg, const char* expr, const char* file, int line, const char* function,void* usr),
    void (*warning)(const char* msg, const char* expr, const char* file, int line, const char* function,void* usr),
    void (*info)   (const char* msg, const char* expr, const char* file, int line, const char* function,void* usr),
    void *usr) {
    error_  =error;
    warning_=warning;
    info_   =info;
    reporter_context_=usr;
}

struct tetrahedron {
    float T[9];
    float ori[3];
};

#define restrict __restrict
#define EPS (1e-5f)
#define BIT(k,i) (((k)>>(i))&1)
#define countof(e) (sizeof(e)/sizeof(*(e)))

/* UTILITIES */

static void tetrahedron(struct tetrahedron *self, const float * const v, const unsigned * idx) {
    const float T[9] = {
        v[3*idx[0]  ]-v[3*idx[3]  ],v[3*idx[1]  ]-v[3*idx[3]  ],v[3*idx[2]  ]-v[3*idx[3]  ],
        v[3*idx[0]+1]-v[3*idx[3]+1],v[3*idx[1]+1]-v[3*idx[3]+1],v[3*idx[2]+1]-v[3*idx[3]+1],
        v[3*idx[0]+2]-v[3*idx[3]+2],v[3*idx[1]+2]-v[3*idx[3]+2],v[3*idx[2]+2]-v[3*idx[3]+2],
    };
    Matrixf.inv33(self->T,T); 
    memcpy(self->ori,v+3*idx[3],sizeof(self->ori));
};

static float sum(const float * const restrict v, const unsigned n) {
    float s=0.0f;
    const float *c=v+n;
    while(c-->v) s+=*c;
    return s;
}

#if 0 // unused?
static float thresh(float *v,const unsigned n,const float thresh) {
    float *c=v+n;
    while(c-->v) *c*=((*c<=-thresh)||(*c>=thresh));
}
#endif

/* 
    Maps source points into barycentric coordinates.

    src must be sized like float[3]
    dst must be sized like float[3] (same size as src)

    dst will hold four lambdas: Must be sized float[4] 
        lambda4 is computed based on the others: lambda4 = 1-lambda1-lambda2-lambda3
*/
static void map(const struct tetrahedron * const restrict self,
                float * restrict dst,
                const float * const restrict src) {
    float tmp[3];
    memcpy(tmp,src,sizeof(float)*3);
    {
        const float * const o = self->ori;
        tmp[0]-=o[0];
        tmp[1]-=o[1];
        tmp[2]-=o[2];
    }
    Matrixf.mul(dst,self->T,3,3,tmp,1);    
    dst[3]=1.0f-sum(dst,3);
}

static unsigned prod(const unsigned * const v,unsigned n) {
    unsigned p=1;
    const unsigned *c=v+n;
    while(c-->v) p*=*c;
    return p;
}

static void cumprod(unsigned * const out, const unsigned * const v, unsigned n) {
    unsigned i;
    out[0]=1;
    for(i=1;i<=n;++i) out[i]=out[i-1]*v[i-1];
}

#if 0 // unused
static unsigned any_greater_than_one(const float * const restrict v,const unsigned n) {
    const float *c=v+n;
    while(c-->v) if(*c>(1.0f+EPS)) return 1;
    return 0;
}
#endif

static unsigned any_less_than_zero(const float * const restrict v,const unsigned n) {
    const float *c=v+n;
    //while(c-->v) if(*c<EPS) return 1; // use this to show off the edges of the middle tetrad
    while(c-->v) if(*c<-EPS) return 1;
    return 0;
}

static unsigned find_best_tetrad(const float * const restrict ls) {
    float v=ls[0];
    unsigned i,argmin=0;
    for(i=1;i<4;++i) {
        if(ls[i]<v) {
            v=ls[i];
            argmin=i;
        }
    }
    if(v>=0.0f)
        return 0;
    return argmin+1;
}

/** 3d only */
static void idx2coord(float * restrict r,unsigned idx,const unsigned * const restrict shape) {
    r[0]=idx%shape[0];  idx/=shape[0];
    r[1]=idx%shape[1];  idx/=shape[1];
    r[2]=idx%shape[2];
}

/*
 * INTERFACE
 */

struct ctx {
    TPixel *src,*dst;
    unsigned src_shape[4],dst_shape[4];
    unsigned src_strides[5],dst_strides[5];
};

int BarycentricAVXinit(struct resampler* self,
                const unsigned * const src_shape,
                const unsigned * const dst_shape,
                const unsigned ndim
               ) {
    ASSERT(ndim==4);  //transform is done in 3D and replicated across the 4th
    memset(self,0,sizeof(*self));
    ASSERT(self->ctx=(struct ctx*)malloc(sizeof(struct ctx)));
    {
        struct ctx * const c=self->ctx;
        //memset(c,0,sizeof(*c));
        memcpy(c->src_shape,src_shape,sizeof(c->src_shape));
        cumprod(c->src_strides,src_shape,4);
        // src just ref'd: no alloc
    }
    {
        struct ctx * const c=self->ctx;
        //memset(c,0,sizeof(*c));
        memcpy(c->dst_shape,dst_shape,sizeof(c->dst_shape));
        cumprod(c->dst_strides,dst_shape,4);
        ASSERT(c->dst=(TPixel*)malloc(c->dst_strides[4]*sizeof(TPixel)));
    }
    return 1;
}

void BarycentricAVXrelease(struct resampler *self) {
    if(self->ctx) {
        struct ctx * c=self->ctx;
        free(c->dst);
        /*free(c->src);   NOT MALLOC'D./  Just refs the input pointer. So not owned by this object.*/
        self->ctx=0;        
    }
}


int BarycentricAVXsource(const struct resampler * self,
                  TPixel * const src)
{
    struct ctx * const ctx=self->ctx;
    ctx->src=src;
    return 1;
}

int BarycentricAVXdestination(struct resampler *self,
                       TPixel * const dst){
    struct ctx * const c=self->ctx;
    memcpy(c->dst,dst,c->dst_strides[4]*sizeof(TPixel));
    return 1;
 }

int BarycentricAVXresult(const struct resampler * const self,
                  TPixel * const dst)
{
    struct ctx * const ctx=self->ctx;
    memcpy(dst,ctx->dst,ctx->dst_strides[4]*sizeof(TPixel));
    return 1;
}

/* THE CRUX */

/* 4 indexes each for 5 tetrads; the first is the center tetrad */
static const unsigned indexes0[5][4]={
        {1,2,4,7},
        {2,4,6,7}, // opposite 1
        {1,4,5,7}, // opposite 2
        {1,2,3,7}, // opposite 4
        {0,1,2,4}  // opposite 7
};
// the orthogonal way to arrange the tetrads
static const unsigned indexes90[5][4]={
        {0,3,5,6},
        {3,5,6,7}, // opposite 0
        {0,4,5,6}, // opposite 3
        {0,2,3,6}, // opposite 5
        {0,1,3,5}  // opposite 6
};
static unsigned (*indexes)[5][4];

/**
    @param cubeverts [in]   An array of floats ordered like float[8][3].
                            Describes the vertices of a three dimensional cube.
                            The vertices must be Morton ordered.  That is, 
                            when bit 0 of the index (0 to 7) is low, that 
                            corresponds to a vertex on the face of the cube that
                            is more minimal in x; 1 on the maximal side.
                            Bit 1 is the y dimension, and bit 2 the z dimension.
*/

#define NTHREADS (8)
#define NSIMD (8)

struct work {
    TPixel *  restrict dst;
    const unsigned * restrict dst_shape;
    const unsigned * restrict dst_strides;

    TPixel *  restrict src;
    const unsigned * restrict src_shape;
    const unsigned * restrict src_strides;

    struct tetrahedron * tetrads;
    int id;
    int method;
};


#include <tictoc.h>
//#define TEST_AVX
//#define TIME_AVX
//#define VERBOSE_AVX
#define MIN(a, b) ((a) < (b) ? (a) : (b))

static void worker(void *param) {
    const struct work * const work=(struct work *)(param);

    {
        TPixel * const restrict dst                         =work->dst;
        const unsigned * const restrict dst_shape           =work->dst_shape;
        const unsigned * const restrict dst_strides         =work->dst_strides;
        TPixel * const restrict src                         =work->src;
        const unsigned * const restrict src_shape           =work->src_shape;
        const unsigned * const restrict src_strides         =work->src_strides;
        const struct tetrahedron * const restrict tetrads   =work->tetrads;
        const int method   =work->method;

        unsigned idst, idst2, idst_max=prod(dst_shape,3); 
        float r[3],lambdas[4];
        unsigned itetrad, skip_it, check;
        float ALIGN r_avx[3][NSIMD], lambdas_avx[4][NSIMD];
        unsigned ALIGN skip_it_avx[NSIMD];
        unsigned ALIGN itetrad_avx[NSIMD];
        const float * const T=tetrads->T;
        const float * const ori=tetrads->ori;
        const float d1 = 1.0f/dst_shape[0];
        const float d2 = 1.0f/dst_shape[1]; 
        const float d3 = 1.0f/dst_shape[2];
        unsigned n;

        const unsigned N = 32;
        unsigned l[2], sh1[2], sh2[2];
        long long mp[2];
        for (n=0; n<2; n++) {
          l[n] = (unsigned)ceilf(log2f((float)dst_shape[n]));
          mp[n] = (long long)(floor((2ll<<(N-1))*((2<<(l[n]-1))-dst_shape[n])/(double)dst_shape[n])+1.0);
          sh1[n] = l[n]<1 ? l[n] : 1;
          sh2[n] = (l[n]-1)>0 ? (l[n]-1) : 0; }
        
        __m256   r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
        __m256i ir0,ir1,ir2,ir3,ir4,ir5,ir6,ir7,ir8,ir9,ir10,ir11,ir12,ir13,ir14,ir15;
        for(idst=work->id;idst<idst_max;idst+=(NTHREADS*NSIMD)) {

           // BJA : == Begin idx2coord ==
           // modulus with integer division.  see granlund and montgomery (1994)

           // idst2 = div(idst, dst_shape[0])
           ir1 = _mm256_set1_epi64x(mp[0]);
           ir3 = _mm256_set_epi64x(idst+6,idst+4,idst+2,idst+0);  // idst
           ir4 = _mm256_set_epi64x(idst+7,idst+5,idst+3,idst+1); 
           ir5 = _mm256_mul_epu32(ir1,ir3);
           ir6 = _mm256_mul_epu32(ir1,ir4);
           ir5 = _mm256_srli_epi64(ir5,N);  // t1
           ir6 = _mm256_srli_epi64(ir6,N);
           ir7 = _mm256_sub_epi64(ir3,ir5);
           ir8 = _mm256_sub_epi64(ir4,ir6);
           ir7 = _mm256_srli_epi64(ir7,sh1[0]);
           ir8 = _mm256_srli_epi64(ir8,sh1[0]);
           ir7 = _mm256_add_epi64(ir7,ir5);
           ir8 = _mm256_add_epi64(ir8,ir6);
           ir7 = _mm256_srli_epi64(ir7,sh2[0]);  // idst2
           ir8 = _mm256_srli_epi64(ir8,sh2[0]);

           // r[0] = idst - idst2 * dst_shape[0]
           ir5 = _mm256_set1_epi64x(dst_shape[0]);
           ir0 = _mm256_mul_epu32(ir7,ir5);
           ir1 = _mm256_mul_epu32(ir8,ir5);
           ir0 = _mm256_sub_epi64(ir3,ir0);  // r[0]
           ir1 = _mm256_sub_epi64(ir4,ir1);

           // r[2] = div(idst2, dst_shape[1])
           ir2 = _mm256_set1_epi64x(mp[1]);
           ir9 = _mm256_mul_epu32(ir2,ir7);
           ir10= _mm256_mul_epu32(ir2,ir8);
           ir9 = _mm256_srli_epi64(ir9,N);  // t1
           ir10= _mm256_srli_epi64(ir10,N);
           ir4 = _mm256_sub_epi64(ir7,ir9);
           ir5 = _mm256_sub_epi64(ir8,ir10);
           ir4 = _mm256_srli_epi64(ir4,sh1[1]);
           ir5 = _mm256_srli_epi64(ir5,sh1[1]);
           ir4 = _mm256_add_epi64(ir4,ir9);
           ir5 = _mm256_add_epi64(ir5,ir10);
           ir4 = _mm256_srli_epi64(ir4,sh2[1]);  // r[2]
           ir5 = _mm256_srli_epi64(ir5,sh2[1]);

           // r[1] = idst2 - r[2] * dst_shape[1]
           ir6 = _mm256_set1_epi64x(dst_shape[1]);
           ir2 = _mm256_mul_epu32(ir4,ir6);
           ir3 = _mm256_mul_epu32(ir5,ir6);
           ir2 = _mm256_sub_epi64(ir7,ir2);
           ir3 = _mm256_sub_epi64(ir8,ir3);

           ir1 = _mm256_slli_epi64(ir1,N);
           ir3 = _mm256_slli_epi64(ir3,N);
           ir5 = _mm256_slli_epi64(ir5,N);
           ir0 = _mm256_blend_epi32(ir0,ir1,0b10101010);
           ir1 = _mm256_blend_epi32(ir2,ir3,0b10101010);
           ir2 = _mm256_blend_epi32(ir4,ir5,0b10101010);
           r0  = _mm256_cvtepi32_ps(ir0);
           r1  = _mm256_cvtepi32_ps(ir1);
           r2  = _mm256_cvtepi32_ps(ir2);

           /* test code for idx2coord
           unsigned ALIGN foo[3][8];
           _mm256_store_si256(foo[0],ir0);
           _mm256_store_si256(foo[1],ir1);
           _mm256_store_si256(foo[2],ir2);
           printf("\tl[0]=%u\n\tmp[0]=%u\n\tsh1[0]=%u\n\tsh2[0]=%u\n",l[0],mp[0],sh1[0],sh2[0]);
           printf("\tl[1]=%u\n\tmp[1]=%u\n\tsh1[1]=%u\n\tsh2[1]=%u\n",l[1],mp[1],sh1[1],sh2[1]);
           printf("\tidst  = %u\n\tshape = %i, %i, %i\n", idst, dst_shape[0], dst_shape[1], dst_shape[2]);
           for(n=0; n<8; n++)
                 printf("\tr7 = %d\tr8 = %d\tr8 = %d\n", foo[0][n],foo[1][n],foo[2][n]);
           exit(0);
           */

           /* the slow easy way to compute idx2coord using the CPU
           for(n=0; n < NSIMD; n++) { 
               idx2coord(r, MIN(idst_max, idst+n), dst_shape);
               r_avx[0][n]=r[0];
               r_avx[1][n]=r[1];
               r_avx[2][n]=r[2]; }
           r0  = _mm256_load_ps(r_avx[0]);
           r1  = _mm256_load_ps(r_avx[1]);
           r2  = _mm256_load_ps(r_avx[2]);
           */

           // BJA : == End idx2coord ==

           #ifdef TEST_AVX                     // ECL : Test the avx idx2coord against cpu version
           
           _mm256_store_ps(r_avx[0],r0);
           _mm256_store_ps(r_avx[1],r1);
           _mm256_store_ps(r_avx[2],r2);
           for(n=0; n < NSIMD; n++) { 
               idx2coord(r, MIN(idst_max, idst+n), dst_shape);
               if(r[0] != r_avx[0][n] || r[1] != r_avx[1][n] || r[2] != r_avx[2][n]) {
                   printf("ERROR: \n");
                   printf("\tl[0]=%u\n\tmp[0]=%u\n\tsh1[0]=%u\n\tsh2[0]=%u\n",l[0],mp[0],sh1[0],sh2[0]);
                   printf("\tl[1]=%u\n\tmp[1]=%u\n\tsh1[1]=%u\n\tsh2[1]=%u\n",l[1],mp[1],sh1[1],sh2[1]);
                   printf("\tidst  = %u\n\tshape = %i, %i, %i\n\tr     = %f, %f, %f\n\tr_avx = %f, %f, %f\n\n",
                          idst+n,
                          dst_shape[0], dst_shape[1], dst_shape[2], 
                          r[0],       r[1],       r[2],
                          r_avx[0][n],r_avx[1][n],r_avx[2][n]);
                   exit(0); }
           }
           #endif


           // ECL : == Begin map ==

           r12 = _mm256_set1_ps(ori[0]); // ECL : Loading all variables
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

           // ECL : r0, r1, and r2 all still contain src[0], src[1], and src[2] respectively

           r12  = _mm256_sub_ps(r0,r12);  // ECL : r0 = src[0]-ori[0]
           r13  = _mm256_sub_ps(r1,r13); 
           r14  = _mm256_sub_ps(r2,r14);

           r3  = _mm256_mul_ps(r12,r3);   //       r3 = T[0] * (src[0]-ori[0]) 
           r4  = _mm256_mul_ps(r13,r4);   //       r4 = T[1] * (src[1]-ori[1])
           r3  = _mm256_add_ps(r3,r4);
           r5  = _mm256_mul_ps(r14,r5);   //       r5 = T[2] * (src[2]-ori[2])
           r3  = _mm256_add_ps(r3,r5);    //       r3 = r3 + r4 + r5 = lambda[0]

           r6  = _mm256_mul_ps(r12,r6);   //       r6 = T[3] * (src[0]-ori[0])
           r7  = _mm256_mul_ps(r13,r7);   //       r7 = T[4] * (src[1]-ori[1])
           r6  = _mm256_add_ps(r6,r7);
           r8  = _mm256_mul_ps(r14,r8);   //       r8 = T[5] * (src[2]-ori[2])
           r6  = _mm256_add_ps(r6,r8);    //       lambda[1]

           r9  = _mm256_mul_ps(r12,r9);   //       r9 = T[6] * (src[0]-ori[0])
           r10 = _mm256_mul_ps(r13,r10);  //       r10= T[7] * (src[1]-ori[1])
           r9  = _mm256_add_ps(r9,r10);  
           r11 = _mm256_mul_ps(r14,r11);  //       r11= T[8] * (src[2]-ori[2])
           r9  = _mm256_add_ps(r9,r11);   //       lambda[2]
           
           r12 = _mm256_set1_ps(1.0f);
           r12 = _mm256_sub_ps(r12,r3);
           r12 = _mm256_sub_ps(r12,r6);
           r12 = _mm256_sub_ps(r12,r9);   // ECL : r12 = 1.0f - lambdas[0] - lambdas[1] - lambdas[2] = lambda[3] 

           // ECL : == End map ==
             
            #ifdef TEST_AVX
           _mm256_store_ps(lambdas_avx[0],r3);  
           _mm256_store_ps(lambdas_avx[1],r6);   
           _mm256_store_ps(lambdas_avx[2],r9);  

           _mm256_store_ps(lambdas_avx[3],r12);
            for(n=0;n<NSIMD;n++) {
              r[0] = r_avx[0][n];
              r[1] = r_avx[1][n];
              r[2] = r_avx[2][n];
              map(tetrads, lambdas, r);
              if(fabs(lambdas[0]-lambdas_avx[0][n])>1e-6 ||
                 fabs(lambdas[1]-lambdas_avx[1][n])>1e-6 ||
                 fabs(lambdas[2]-lambdas_avx[2][n])>1e-6 ||
                 fabs(lambdas[3]-lambdas_avx[3][n])>1e-6) 
                 printf("ERROR: \n\t\t\t\t\tlambdas     =  %.10f,  %.10f,  %.10f,  %.10f\n\
                                  \tlambdas_avx =  %.10f,  %.10f,  %.10f,  %.10f\n\
                                  \tdiff        =  %.10f,  %.10f,  %.10f,  %.10f\n\n",
                             lambdas[0],       lambdas[1],       lambdas[2],       lambdas[3],
                         lambdas_avx[0][n],lambdas_avx[1][n],lambdas_avx[2][n],lambdas_avx[3][n],
                         lambdas[0]-lambdas_avx[0][n],
                         lambdas[1]-lambdas_avx[1][n],
                         lambdas[2]-lambdas_avx[2][n],
                         lambdas[3]-lambdas_avx[3][n]);
            }
            #endif
             

            // ECL : == Begin find_best_tetrad ==
 
            ir4 = _mm256_set1_epi32(1);             // ECL : Setting first min for n to be 1

            r5  = _mm256_cmp_ps(r6,r3,_CMP_LT_OQ);  // ECL : r5[n] should now either be 1's wherever
                                                    //    lambda[1][n] < lambda[0][n], else 0's
            r15 = _mm256_min_ps(r6,r3);             //       r15 = min of lambda[1] and lambda[0]
            ir5 = _mm256_castps_si256(r5);        
            ir7 = _mm256_set1_epi32(2);             // ECL : ir7 = a register of ints = 2
            ir5 = _mm256_and_si256(ir5,ir7);        //       ir5[n] = 2 if lambda[1][n] < lambda[0][n], or else 0
            ir4 = _mm256_max_epi32(ir4,ir5);        //       Because of the order we are going in (1 to 4), 
                                                    //    we want the largest numbers. (ie if ir4[n] initialy 
                                                    //    had 1 stored, but ir5[n] had 2, this would 
                                                    //    indicate that lambda[1] had a lower value than lambda[0],
                                                    //    so we would now store 2 in ir4[n].

            r5  = _mm256_cmp_ps(r9,r15,_CMP_LT_OQ); // ECL : Same logic as above
            r15 = _mm256_min_ps(r9,r15);
            ir5 = _mm256_castps_si256(r5);
            ir7 = _mm256_set1_epi32(3);
            ir5 = _mm256_and_si256(ir5,ir7);
            ir4 = _mm256_max_epi32(ir4,ir5);

            r5  = _mm256_cmp_ps(r12,r15,_CMP_LT_OQ);
            r15 = _mm256_min_ps(r12,r15);
            ir5 = _mm256_castps_si256(r5);
            ir7 = _mm256_set1_epi32(4);
            ir5 = _mm256_and_si256(ir5,ir7);
            ir4 = _mm256_max_epi32(ir4,ir5);

            // ECL : This block just removes any positive values, so if there were no negative 
            // numbers for a particular n, itetrad[n] would be 0.
            r7  = _mm256_set1_ps(0.0f);   // BJA:  ECL had an -EPS here
            r5  = _mm256_cmp_ps(r15,r7,_CMP_LT_OQ);
            ir5 = _mm256_castps_si256(r5);
            ir4 = _mm256_and_si256(ir4,ir5);
            
            _mm256_store_si256((__m256i *)itetrad_avx,ir4);
            
            // ECL : == End find_best_tetrad ==


            #ifdef TEST_AVX
            for(n=0;n<NSIMD;n++) {
              idx2coord(r, MIN(idst_max, idst+n), dst_shape);
              map(tetrads, lambdas, r);
              itetrad = find_best_tetrad(lambdas);
              if(itetrad_avx[n] != itetrad) { 
                printf("ERROR:\n\titetrad     = %i\n\titetrad_avx = %i\n", itetrad, itetrad_avx[n]);
                printf("\t\t\t\tlambdas     =  %.20f,  %.20f,  %.20f,  %.20f\n\
                        \tlambdas_avx =  %.20f,  %.20f,  %.20f,  %.20f\n\n",
                         lambdas[0],       lambdas[1],       lambdas[2],       lambdas[3],
                     lambdas_avx[0][n],lambdas_avx[1][n],lambdas_avx[2][n],lambdas_avx[3][n]);
              }
            }
            #endif


            // ECL : We now want to check if the 8 itetrads in itetrad_avx are the same, as
            //       this will make the logic easier.

            // ECL : ir4 = a1, a2, a3, a4, a5, a6, a7, a8
            //       ir5 = a2, a3, a4, a1, a6, a7, a8, a5
            //       ir8 = a5, a6, a7, a8, a1, a2, a3, a4
            //       so if ir4 = ir5 = ir8, we get
            //       a1 = a2 = a5 = a6 = a7 = a8 = a3 = a4 
            ir5 = _mm256_shuffle_epi32(ir4, 0b10010011);
            ir8 = _mm256_permute2x128_si256(ir4, ir4, 0b00000001);
            ir5 = _mm256_xor_si256(ir4, ir5);
            ir8 = _mm256_xor_si256(ir4, ir8);
            ir5 = _mm256_or_si256(ir5, ir8); // ECL : If all numbers were equal, ir5 now holds all zeros
            unsigned tmp[8];
            _mm256_store_si256((__m256i *)tmp, ir5);
            unsigned check = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
            const float * const T2=(tetrads + itetrad_avx[0])->T;
            const float * const ori2=(tetrads + itetrad_avx[0])->ori;
            

            // ECL : == Begin second map ==

            //r0  = _mm256_load_ps(r_avx[0]);
            //r1  = _mm256_load_ps(r_avx[1]);
            //r2  = _mm256_load_ps(r_avx[2]);
            switch(check) {
               case 0 : // ECL : 0 implies all numbers were equal, so just take the correct tetrad and re-map
                   r12 = _mm256_set1_ps(ori2[0]);
                   r13 = _mm256_set1_ps(ori2[1]);
                   r14 = _mm256_set1_ps(ori2[2]);
                   r3  = _mm256_set1_ps(T2[0]);
                   r4  = _mm256_set1_ps(T2[1]);
                   r5  = _mm256_set1_ps(T2[2]);
                   r6  = _mm256_set1_ps(T2[3]);
                   r7  = _mm256_set1_ps(T2[4]);
                   r8  = _mm256_set1_ps(T2[5]);
                   r9  = _mm256_set1_ps(T2[6]);
                   r10 = _mm256_set1_ps(T2[7]);
                   r11 = _mm256_set1_ps(T2[8]);
                   break;
               
               default : // ECL : Otherwise we have to take the correct tetrad on an individual case, thus we gather 
                   ir15= _mm256_set1_epi32(12);
                   ir15= _mm256_mullo_epi32(ir4,ir15);
                   r12 = _mm256_i32gather_ps(&ori[0],ir15,4);              
                   r13 = _mm256_i32gather_ps(&ori[1],ir15,4);
                   r14 = _mm256_i32gather_ps(&ori[2],ir15,4);
                   r3  = _mm256_i32gather_ps(&T[0],ir15,4);
                   r4  = _mm256_i32gather_ps(&T[1],ir15,4);
                   r5  = _mm256_i32gather_ps(&T[2],ir15,4);
                   r6  = _mm256_i32gather_ps(&T[3],ir15,4);
                   r7  = _mm256_i32gather_ps(&T[4],ir15,4);
                   r8  = _mm256_i32gather_ps(&T[5],ir15,4);
                   r9  = _mm256_i32gather_ps(&T[6],ir15,4);
                   r10 = _mm256_i32gather_ps(&T[7],ir15,4);
                   r11 = _mm256_i32gather_ps(&T[8],ir15,4); 
                   break;
            }
            r0  = _mm256_sub_ps(r0,r12);   // ECL : r0 = src[0]-ori[0]
            r1  = _mm256_sub_ps(r1,r13); 
            r2  = _mm256_sub_ps(r2,r14);

            r3  = _mm256_mul_ps(r0,r3);    //       r3 = T[0] * (src[0]-ori[0]) 
            r4  = _mm256_mul_ps(r1,r4);    //       r4 = T[1] * (src[1]-ori[1])
            r3  = _mm256_add_ps(r3,r4);
            r5  = _mm256_mul_ps(r2,r5);    //       r5 = T[2] * (src[2]-ori[2])
            r3  = _mm256_add_ps(r3,r5);    //       r3 = r3 + r4 + r5 = lambda[0]

            r6  = _mm256_mul_ps(r0,r6);    //       r6 = T[3] * (src[0]-ori[0])
            r7  = _mm256_mul_ps(r1,r7);    //       r7 = T[4] * (src[1]-ori[1])
            r6  = _mm256_add_ps(r6,r7);
            r8  = _mm256_mul_ps(r2,r8);    //       r8 = T[5] * (src[2]-ori[2])
            r6  = _mm256_add_ps(r6,r8);    //       r6 =  lambda[1]

            r9  = _mm256_mul_ps(r0,r9);    //       r9 = T[6] * (src[0]-ori[0])
            r10 = _mm256_mul_ps(r1,r10);   //       r10= T[7] * (src[1]-ori[1])
            r9  = _mm256_add_ps(r9,r10);  
            r11 = _mm256_mul_ps(r2,r11);   //       r11= T[8] * (src[2]-ori[2])
            r9  = _mm256_add_ps(r9,r11);   //       r9 = lambda[2]

            r12 = _mm256_set1_ps(1.0f);
            r12 = _mm256_sub_ps(r12,r3);
            r12 = _mm256_sub_ps(r12,r6);
            r12 = _mm256_sub_ps(r12,r9);
            #ifdef TEST_AVX
            _mm256_store_ps(lambdas_avx[0],r3);
            _mm256_store_ps(lambdas_avx[1],r6);
            _mm256_store_ps(lambdas_avx[2],r9);
            _mm256_store_ps(lambdas_avx[3],r12);
            #endif

            // ECL : == End second map ==


            // ECL : == Begin skip_it_avx ==

            r4  = _mm256_min_ps(r3,r6);
            r5  = _mm256_min_ps(r9,r12);
            r7  = _mm256_set1_ps(-EPS);
            r4  = _mm256_min_ps(r4,r5);            // ECL : r4 = smallest lambda
            r5  = _mm256_cmp_ps(r4,r7,_CMP_LT_OQ); // ECL : if a lambda is negative, skip it in the interp.
            ir5 = _mm256_castps_si256(r5);
            _mm256_store_si256((__m256i *)skip_it_avx,ir5);

            // ECL : == End skip_it_avx ==

            // ECL : == Begin interp ==
            //       We make the interpolation to amenable to AVX by unrolling the inner loops of the 
            //     original version. As such, we now accumulate three s variables (one for each dimension)
            //     12 w variables, etc.
            r0 = _mm256_setzero_ps();                            // ECL : r0 will hold s0
            r1 = _mm256_setzero_ps();                            //       r1 will hold s1
            r2 = _mm256_setzero_ps();                            //       r2 will hold s2
            r4 = _mm256_set1_ps((float)src_shape[0]);
            r5 = _mm256_set1_ps((float)src_shape[1]);
            r7 = _mm256_set1_ps((float)src_shape[2]);
            
            ir4 = _mm256_mullo_epi32(ir4, _mm256_set1_epi32(4)); // ECL : Since indexes is stored in 
            ir0 = _mm256_i32gather_epi32(**indexes+0,ir4,4);     //     row-major ordering, indexes+n
            ir1 = _mm256_i32gather_epi32(**indexes+1,ir4,4);     //     grabs the nth collumn of indexes.
            ir2 = _mm256_i32gather_epi32(**indexes+2,ir4,4);     //     Since each row has 4 elements, we 
            ir3 = _mm256_i32gather_epi32(**indexes+3,ir4,4);     //     multiply itetrad by 4.
            
            ir5 = _mm256_set1_epi32(1);
            
            ir6 = _mm256_and_si256(ir0, ir5);                    // ECL : ir6 = (idx>>0 & 1) = BIT(idx,0)
            ir7 = _mm256_and_si256(ir1, ir5);
            ir8 = _mm256_and_si256(ir2, ir5);
            ir9 = _mm256_and_si256(ir3, ir5);

            r11 = _mm256_mul_ps(_mm256_cvtepi32_ps(ir6), r3);    // ECL : r11 = w*BIT(idx,0)
            r13 = _mm256_mul_ps(_mm256_cvtepi32_ps(ir7), r6);
            r14 = _mm256_mul_ps(_mm256_cvtepi32_ps(ir8), r9);
            r15 = _mm256_mul_ps(_mm256_cvtepi32_ps(ir9), r12);
            
            r0  = _mm256_add_ps(r11,r13);                        // ECL : r0/s0 accumulates values
            r0  = _mm256_add_ps(r0,r14);
            r0  = _mm256_add_ps(r0,r15);
 
            ir6 = _mm256_srli_epi32(ir0,0b00000001);             // ECL : ir6 = idx>>1
            ir7 = _mm256_srli_epi32(ir1,0b00000001);
            ir8 = _mm256_srli_epi32(ir2,0b00000001);
            ir9 = _mm256_srli_epi32(ir3,0b00000001);
            ir6 = _mm256_and_si256(ir6, ir5);                    // ECL : ir6 = BIT(idx,1)
            ir7 = _mm256_and_si256(ir7, ir5);
            ir8 = _mm256_and_si256(ir8, ir5);
            ir9 = _mm256_and_si256(ir9, ir5);

            r11 = _mm256_mul_ps(_mm256_cvtepi32_ps(ir6), r3);    // ECL : r11 = w*BIT(idx,1)
            r13 = _mm256_mul_ps(_mm256_cvtepi32_ps(ir7), r6);
            r14 = _mm256_mul_ps(_mm256_cvtepi32_ps(ir8), r9);
            r15 = _mm256_mul_ps(_mm256_cvtepi32_ps(ir9), r12);
            r1  = _mm256_add_ps(r11,r13);                      
            r1  = _mm256_add_ps(r1,r14);
            r1  = _mm256_add_ps(r1,r15);
            
            ir6 = _mm256_srli_epi32(ir0,0b00000010);             // ECL : ir6 = idx>>2
            ir7 = _mm256_srli_epi32(ir1,0b00000010);
            ir8 = _mm256_srli_epi32(ir2,0b00000010);
            ir9 = _mm256_srli_epi32(ir3,0b00000010);
            ir6 = _mm256_and_si256(ir6, ir5);
            ir7 = _mm256_and_si256(ir7, ir5);
            ir8 = _mm256_and_si256(ir8, ir5);
            ir9 = _mm256_and_si256(ir9, ir5);

            r11 = _mm256_mul_ps(_mm256_cvtepi32_ps(ir6), r3);
            r13 = _mm256_mul_ps(_mm256_cvtepi32_ps(ir7), r6);
            r14 = _mm256_mul_ps(_mm256_cvtepi32_ps(ir8), r9);
            r15 = _mm256_mul_ps(_mm256_cvtepi32_ps(ir9), r12);
            r2  = _mm256_add_ps(r11,r13);
            r2  = _mm256_add_ps(r2,r14);
            r2  = _mm256_add_ps(r2,r15);
            
            r0 = _mm256_mul_ps(r0,r4);                           // ECL : r0=s0*d
            r1 = _mm256_mul_ps(r1,r5);
            r2 = _mm256_mul_ps(r2,r7);
            
            r10 = _mm256_setzero_ps();                           // ECL : if r0/s0 is less than zero, set it to zero
            r0 = _mm256_max_ps(r0, r10);
            r1 = _mm256_max_ps(r1, r10);
            r2 = _mm256_max_ps(r2, r10);
            
            r10 = _mm256_set1_ps(((float)src_shape[0])-1.0f);
            r0 = _mm256_min_ps(r0, r10);                         // ECL : if r0/s0 is greater than d-1, set it to d-1

            r10 = _mm256_set1_ps(((float)src_shape[1])-1.0f);
            r1 = _mm256_min_ps(r1, r10);
            
            r10 = _mm256_set1_ps(((float)src_shape[2])-1.0f);
            r2 = _mm256_min_ps(r2, r10);

            r13 = _mm256_floor_ps(r0);                           // ECL : floor of s
            r14 = _mm256_floor_ps(r1);
            r15 = _mm256_floor_ps(r2);
            
            ir3 = _mm256_cvtps_epi32(r13);                       // ECL : zero[0]
            ir4 = _mm256_cvtps_epi32(r14);                       //       zero[1]
            ir5 = _mm256_cvtps_epi32(r15);                       //       zero[2]
            
            float ALIGN frac_avx[3][NSIMD];
            unsigned ALIGN tmp2[NSIMD];
            unsigned i4, i4src, i4dst;
            switch(method) {
               
              case 0:
                ir0  = _mm256_set1_epi32(src_strides[0]);
                ir1  = _mm256_set1_epi32(src_strides[1]);
                ir2  = _mm256_set1_epi32(src_strides[2]);
                
                ir0 = _mm256_mullo_epi32(ir0,ir3);
                ir1 = _mm256_mullo_epi32(ir1,ir4);
                ir2 = _mm256_mullo_epi32(ir2,ir5);
                
                ir12 = _mm256_add_epi32(ir0,ir1);
                ir12 = _mm256_add_epi32(ir12,ir2);

                _mm256_store_si256((__m256i *)tmp2,ir12);
                for(idst2=0; idst2<NSIMD; idst2++)
                {
                    if((idst+idst2)>idst_max)  break;                // ECL : If we surpass idst_max, we stop
                    if(skip_it_avx[idst2]>0)  continue;              //       Skipping "bad" pixels

                    for(i4=0; i4<dst_shape[3]; i4++) {
                      i4src=i4*src_strides[3];
                      i4dst=i4*dst_strides[3];
                      dst[idst+idst2+i4dst] = src[tmp2[idst2]+i4src]; }
                }
                break;

              case 1:
                r0 = _mm256_sub_ps(r0,r13);                      // ECL : frac[0]
                r1 = _mm256_sub_ps(r1,r14);                      //       frac[1]
                r2 = _mm256_sub_ps(r2,r15);                      //       frac[2]
            
                ir9 = _mm256_set1_epi32(1);
                ir6 = _mm256_add_epi32(ir3,ir9);                 //       one[0]
                ir7 = _mm256_add_epi32(ir4,ir9);                 //       one[1]
                ir8 = _mm256_add_epi32(ir5,ir9);                 //       one[2]

                // ECL : If zero > source_shape for any number, we add it to skip_it_avx
                r15 = _mm256_cmp_ps(_mm256_cvtepi32_ps(ir3), r4, _CMP_GE_OQ);                   
                r15 = _mm256_and_ps(r15, _mm256_cmp_ps(_mm256_cvtepi32_ps(ir4), r5, _CMP_GE_OQ));
                r15 = _mm256_and_ps(r15, _mm256_cmp_ps(_mm256_cvtepi32_ps(ir5), r7, _CMP_GE_OQ));
                ir9 = _mm256_load_si256((__m256i *)skip_it_avx);
                ir9 = _mm256_or_si256(ir9, _mm256_cvtps_epi32(r15));
                _mm256_store_si256((__m256i *)skip_it_avx, ir9);
                ir11 = _mm256_set1_epi32(1);
 
                // ECL : If one > source shape, subtract 1
                ir10 = _mm256_add_epi32(ir6, ir11);
                ir9 = _mm256_cmpgt_epi32(ir10, _mm256_cvtps_epi32(r4));
                ir9 = _mm256_and_si256(ir9, _mm256_set1_epi32(1));
                ir6 = _mm256_sub_epi32(ir6, ir9);
           
                ir10 = _mm256_add_epi32(ir7, ir11);
                ir9 = _mm256_cmpgt_epi32(ir10, _mm256_cvtps_epi32(r5));
                ir9 = _mm256_and_si256(ir9, _mm256_set1_epi32(1));
                ir7 = _mm256_sub_epi32(ir7, ir9);

                ir10 = _mm256_add_epi32(ir8, ir11);
                ir9 = _mm256_cmpgt_epi32(ir10, _mm256_cvtps_epi32(r7));
                ir9 = _mm256_and_si256(ir9, _mm256_set1_epi32(1));
                ir8 = _mm256_sub_epi32(ir8, ir9);

                ir12 = _mm256_set1_epi32(src_strides[0]);
                ir13 = _mm256_set1_epi32(src_strides[1]);
                ir14 = _mm256_set1_epi32(src_strides[2]);
                ir10 = _mm256_mullo_epi32(ir3,ir12);              // zero[0]*strides[0]
                ir11 = _mm256_mullo_epi32(ir6,ir12);              //  one[0]*strides[0]
                ir4  = _mm256_mullo_epi32(ir4,ir13);              // zero[1]*strides[1]
                ir7  = _mm256_mullo_epi32(ir7,ir13);              //  one[1]*strides[1]
                ir5  = _mm256_mullo_epi32(ir5,ir14);              // zero[2]*strides[2]
                ir8  = _mm256_mullo_epi32(ir8,ir14);              //  one[2]*strides[2]
                ir12 = _mm256_add_epi32(ir4,ir5);                 // (zero[1]+zero[2])*strides[]
                ir13 = _mm256_add_epi32(ir7,ir5);                 // ( one[1]+zero[2])*strides[]
                ir14 = _mm256_add_epi32(ir4,ir8);                 // (zero[1]+ one[2])*strides[]
                ir15 = _mm256_add_epi32(ir7,ir8);                 // ( one[1]+ one[2])*strides[]

                for(i4=0; i4<dst_shape[3]; i4++) {
                  i4src=i4*src_strides[3];
                  i4dst=i4*dst_strides[3];

                  ir6 = _mm256_add_epi32(ir10,ir12);              // c00a = (zero[0]+zero[1]+zero[2])*strides[]
                  ir7 = _mm256_add_epi32(ir11,ir12);              // c00b = ( one[0]+zero[1]+zero[2])*strides[]
                  ir8 = _mm256_add_epi32(ir10,ir13);              // c10a = (zero[0]+ one[1]+zero[2])*strides[]
                  ir9 = _mm256_add_epi32(ir11,ir13);              // c10b = ( one[0]+ one[1]+zero[2])*strides[]

                  ir3 = _mm256_set1_epi32(i4src);     // i4src
                  ir6 = _mm256_add_epi32(ir6,ir3);
                  ir7 = _mm256_add_epi32(ir7,ir3);
                  ir8 = _mm256_add_epi32(ir8,ir3);
                  ir9 = _mm256_add_epi32(ir9,ir3);

                  ir6 = _mm256_i32gather_epi32((int*)src,ir6,2);
                  ir7 = _mm256_i32gather_epi32((int*)src,ir7,2);
                  ir8 = _mm256_i32gather_epi32((int*)src,ir8,2);
                  ir9 = _mm256_i32gather_epi32((int*)src,ir9,2);

                  ir3 = _mm256_set1_epi32(0x0000FFFF);            // assumes little-endian?
                  ir6 = _mm256_and_si256(ir6,ir3);
                  ir7 = _mm256_and_si256(ir7,ir3);
                  ir8 = _mm256_and_si256(ir8,ir3);
                  ir9 = _mm256_and_si256(ir9,ir3);

                  r3 = _mm256_set1_ps(1.0f);
                  r3 = _mm256_sub_ps(r3,r0);                      // remfrac[0]
                  r6 = _mm256_cvtepi32_ps(ir6);
                  r7 = _mm256_cvtepi32_ps(ir7);
                  r8 = _mm256_cvtepi32_ps(ir8);
                  r9 = _mm256_cvtepi32_ps(ir9);
                  r6 = _mm256_mul_ps(r6,r3);
                  r7 = _mm256_mul_ps(r7,r0);
                  r8 = _mm256_mul_ps(r8,r3);
                  r9 = _mm256_mul_ps(r9,r0);
                  r4 = _mm256_add_ps(r6,r7);                      // c00
                  r5 = _mm256_add_ps(r8,r9);                      // c10

                  ir6 = _mm256_add_epi32(ir10,ir14);              // c01a = (zero[0]+zero[1]+zero[2])*strides[]
                  ir7 = _mm256_add_epi32(ir11,ir14);              // c01b = ( one[0]+zero[1]+zero[2])*strides[]
                  ir8 = _mm256_add_epi32(ir10,ir15);              // c11a = (zero[0]+ one[1]+zero[2])*strides[]
                  ir9 = _mm256_add_epi32(ir11,ir15);              // c11b = ( one[0]+ one[1]+zero[2])*strides[]

                  ir3 = _mm256_set1_epi32(i4src);     // i4src
                  ir6 = _mm256_add_epi32(ir6,ir3);
                  ir7 = _mm256_add_epi32(ir7,ir3);
                  ir8 = _mm256_add_epi32(ir8,ir3);
                  ir9 = _mm256_add_epi32(ir9,ir3);

                  ir6 = _mm256_i32gather_epi32((int*)src,ir6,2);
                  ir7 = _mm256_i32gather_epi32((int*)src,ir7,2);
                  ir8 = _mm256_i32gather_epi32((int*)src,ir8,2);
                  ir9 = _mm256_i32gather_epi32((int*)src,ir9,2);

                  ir3 = _mm256_set1_epi32(0x0000FFFF);            // assumes little-endian?
                  ir6 = _mm256_and_si256(ir6,ir3);
                  ir7 = _mm256_and_si256(ir7,ir3);
                  ir8 = _mm256_and_si256(ir8,ir3);
                  ir9 = _mm256_and_si256(ir9,ir3);

                  r3 = _mm256_set1_ps(1.0f);
                  r3 = _mm256_sub_ps(r3,r0);                      // remfrac[0]
                  r6 = _mm256_cvtepi32_ps(ir6);
                  r7 = _mm256_cvtepi32_ps(ir7);
                  r8 = _mm256_cvtepi32_ps(ir8);
                  r9 = _mm256_cvtepi32_ps(ir9);
                  r6 = _mm256_mul_ps(r6,r3);
                  r7 = _mm256_mul_ps(r7,r0);
                  r8 = _mm256_mul_ps(r8,r3);
                  r9 = _mm256_mul_ps(r9,r0);
                  r6 = _mm256_add_ps(r6,r7);                      // c01
                  r7 = _mm256_add_ps(r8,r9);                      // c11

                  r3 = _mm256_set1_ps(1.0f);
                  r3 = _mm256_sub_ps(r3,r1);                      // remfrac[1]
                  r4 = _mm256_mul_ps(r4,r3);
                  r6 = _mm256_mul_ps(r6,r3);
                  r4 = _mm256_fmadd_ps(r1,r5,r4);                 // c0
                  r5 = _mm256_fmadd_ps(r1,r7,r6);                 // c1

                  r3 = _mm256_set1_ps(1.0f);
                  r3 = _mm256_sub_ps(r3,r2);                      // remfrac[2]
                  r4 = _mm256_mul_ps(r4,r3);
                  r4 = _mm256_fmadd_ps(r2,r5,r4);

                  ir4 = _mm256_cvtps_epi32(r4);                   // final result
                  _mm256_store_si256((__m256i *)tmp2,ir4);

                  for(idst2=0; idst2<NSIMD; idst2++)
                  {
                      if((idst+idst2)>idst_max)  break;                // ECL : If we surpass idst_max, we stop
                      if(skip_it_avx[idst2]>0)  continue;              //       Skipping "bad" pixels

                      dst[idst+idst2+i4dst] = tmp2[idst2];                
                  }
              }
              break;
            } 
           
        }
    }
}

#include <thread.h>

int BarycentricAVXresample(struct resampler * const self,
                     const float * const cubeverts,
                     const int orientation,
                     const int method) {
    /* Approach

    1. Build tetrahedra from cube vertices
    2. Over pixel indexes for dst, for central tetrad
        1. map to lambdas
        2. check for oob/best tetrad.
        3. For best tetrad
           1. map to uvw
           2. sample source
    */

    struct tetrahedron tetrads[5];
    thread_t ts[NTHREADS]={0};
    struct work jobs[NTHREADS]={0};
    unsigned i,j;

    struct ctx * const ctx=self->ctx;
    TPixel * const restrict         dst         = ctx->dst;
    const unsigned * const restrict dst_shape   = ctx->dst_shape;
    const unsigned * const restrict dst_strides = ctx->dst_strides;
    TPixel * const restrict         src         = ctx->src;
    const unsigned * const restrict src_shape   = ctx->src_shape;
    const unsigned * const restrict src_strides = ctx->src_strides;
    indexes = orientation==0 ? &indexes0 : &indexes90;
    
    for(i=0;i<5;i++)
        tetrahedron(tetrads+i,cubeverts,(*indexes)[i]); // TODO: VERIFY the indexing on "indexes" works correctly here
    
    for(i=0;i<NTHREADS;++i)
    {
        const struct work job={dst,dst_shape,dst_strides,src,src_shape,src_strides,tetrads,i*NSIMD,method};
        jobs[i]=job;
        ts[i]=thread_create(worker,jobs+i);
    }
    for(i=0;i<NTHREADS;++i) thread_join(ts+i,-1);
    for(i=0;i<NTHREADS;++i) thread_release(ts+i);
    return 1;
}

/* Internal testing */

static unsigned eq(const float *a,const float *b,int n) {
    int i;
    for(i=0;i<n;++i) if(a[i]!=b[i]) return 0;
    return 1;
}

static int test_testrahedron(void) {
    const unsigned idx[]={1,4,5,7};
    float v[8*3]={0};
    int i;
    struct tetrahedron ans;
    struct tetrahedron expected={
            {18.0f,19.0f,20.0f}
    };
    for(i=0;i<8;++i) {
        v[3*i+0]=(float)BIT(i,0);
        v[3*i+1]=(float)BIT(i,1);
        v[3*i+2]=(float)BIT(i,2);
    }
    tetrahedron(&ans,v,idx);
    ASSERT(eq(ans.ori,v+3*7,3));
    return 0;
}

static int test_sum(void) {
    float v[]={4.0f,2.0f,4.0f,3.5f};
    float d=sum(v,4)-13.5f;
    ASSERT(d*d<1e-5f);
    return 0;
}

static int test_map(void) {
    WARN(0,"TODO");
    return 0;
}

static int test_prod(void) {
    unsigned v[]={4,2,4,3};
    ASSERT(prod(v,4)==4*2*4*3);
    return 0;
}

#if 0 
static int test_any_greater_than_one(void) {
    float yes[]={4.0f,-0.1f,4.0f,3.5f};
    float  no[]={-4.0f, -2.0f,-4.0f,-3.5f};
    ASSERT(any_greater_than_one(yes,4)==1);
    ASSERT(any_greater_than_one( no,4)==0);
    return 0;
}
#endif

static int test_find_best_tetrad(void) {
    float first[]={0.1f,0.5f,0.7f,0.3f};
    float third[]={0.1f,0.5f,-0.7f,0.3f};
    float  last[]={0.1f,0.5f,0.7f,-0.3f};
    ASSERT(find_best_tetrad(first)==0);
    ASSERT(find_best_tetrad(third)==3);
    ASSERT(find_best_tetrad( last)==4);
    return 0;
}

static int test_idx2coord(void) {
    float r[3];
    unsigned shape[3]={7,13,11};
    float expected[]={4.0f,2.0f,5.0f};
    idx2coord(r,4+7*(2+13*5),shape);
    ASSERT(eq(expected,r,3));
    return 0;
}

static int (*tests[])(void)={
    test_testrahedron,
    test_sum,
    test_map,
    test_prod,
    //test_any_greater_than_one,
    test_find_best_tetrad,
    test_idx2coord,
};

int BarycentricAVXrunTests() {
    int i;
    int nfailed=0;
    for(i=0;i<countof(tests);++i) {
        nfailed+=tests[i]();
    }
    return nfailed;
}
