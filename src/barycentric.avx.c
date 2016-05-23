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
    unsigned src_shape[3],dst_shape[3];
    unsigned src_strides[4],dst_strides[4];
};

int BarycentricAVXinit(struct resampler* self,
                const unsigned * const src_shape,
                const unsigned * const dst_shape,
                const unsigned ndim
               ) {
    ASSERT(ndim==3);
    memset(self,0,sizeof(*self));
    ASSERT(self->ctx=(struct ctx*)malloc(sizeof(struct ctx)));
    {
        struct ctx * const c=self->ctx;
        //memset(c,0,sizeof(*c));
        memcpy(c->src_shape,src_shape,sizeof(c->src_shape));
        cumprod(c->src_strides,src_shape,3);
        // src just ref'd: no alloc
    }
    {
        struct ctx * const c=self->ctx;
        //memset(c,0,sizeof(*c));
        memcpy(c->dst_shape,dst_shape,sizeof(c->dst_shape));
        cumprod(c->dst_strides,dst_shape,3);
        ASSERT(c->dst=(TPixel*)malloc(c->dst_strides[3]*sizeof(TPixel)));
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
    memcpy(c->dst,dst,c->dst_strides[3]*sizeof(TPixel));
    return 1;
 }

int BarycentricAVXresult(const struct resampler * const self,
                  TPixel * const dst)
{
    struct ctx * const ctx=self->ctx;
    memcpy(dst,ctx->dst,ctx->dst_strides[3]*sizeof(TPixel));
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

#define NTHREADS (1)
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

__m256 r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15;
__m256i ir0,ir1,ir2,ir3,ir4,ir5,ir6,ir7,ir8,ir9,ir10,ir11,ir12,ir13,ir14,ir15;

#include <tictoc.h>
#define TEST_AVX
#define ALIGN __attribute__ ((aligned (32)))
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
        float r[3],lambdas0[4],lambdas[4];
        unsigned itetrad, skip_it, check;
        float ALIGN r_avx[3][NSIMD], lambdas_avx[4][NSIMD];
        unsigned ALIGN skip_it_avx[NSIMD];
        unsigned ALIGN itetrad_avx[NSIMD];
        for(idst=work->id;idst<idst_max;idst+=(NTHREADS*NSIMD)) {

            #ifdef TIME_AVX
              TicTocTimer t=tic();
              printf("NULL %10gs\t%s\n",toc(&t),"");
              t=tic();
            #endif

            // idx2coord(r,idst,dst_shape);
            ir0  = _mm256_set_epi32(
                MIN(idst_max,idst+7),MIN(idst_max,idst+6),MIN(idst_max,idst+5),MIN(idst_max,idst+4),
                MIN(idst_max,idst+3),MIN(idst_max,idst+2),MIN(idst_max,idst+1),idst);
            r3  = _mm256_set1_ps((float)(dst_shape[0]));
            r4  = _mm256_set1_ps((float)(dst_shape[1]));
            r5  = _mm256_set1_ps((float)(dst_shape[2]));
            r0   = _mm256_cvtepi32_ps(ir0);
            r1   = _mm256_div_ps(r0,r3);
            r2   = _mm256_div_ps(r1,r4);
            r1   = _mm256_floor_ps(r1);
            r2   = _mm256_floor_ps(r2);

            r0   = _mm256_div_ps(r0,r3);
            r1   = _mm256_div_ps(r1,r4);
            r2   = _mm256_div_ps(r2,r5);
            r6   = _mm256_floor_ps(r0);
            r7   = _mm256_floor_ps(r1);
            r8   = _mm256_floor_ps(r2);
            r0   = _mm256_sub_ps(r0,r6);
            r1   = _mm256_sub_ps(r1,r7);
            r2   = _mm256_sub_ps(r2,r8);
            r0   = _mm256_mul_ps(r3,r0);  // r[0]
            r1   = _mm256_mul_ps(r4,r1);  // r[1]
            r2   = _mm256_mul_ps(r5,r2);  // r[2]

            // not enough registers for map()
            _mm256_store_ps(r_avx[0],r0);
            _mm256_store_ps(r_avx[1],r1);
            _mm256_store_ps(r_avx[2],r2);
            #ifdef TEST_AVX
              printf("r_avx=%f,%f,%f\n",r_avx[0][0],r_avx[1][0],r_avx[2][0]);
            #endif

            // map(tetrads,lambdas,r);             // Map center tetrahedron
            const float * const T=tetrads->T;
            const float * const ori=tetrads->ori;

            //r0 = _mm256_load_ps(r_avx[0]);
            //r1 = _mm256_load_ps(r_avx[1]);
            //r2 = _mm256_load_ps(r_avx[2]);
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
            r3 = _mm256_add_ps(r3,r5);   // lambda[0]
            r6 = _mm256_add_ps(r6,r8);   // lambda[1]
            r9 = _mm256_add_ps(r9,r11);  // lambda[2]

            r12 = _mm256_set1_ps(1.0f);
            r12 = _mm256_sub_ps(r12,r3);
            r12 = _mm256_sub_ps(r12,r6);
            r12 = _mm256_sub_ps(r12,r9);   // lambda[3]
            #ifdef TEST_AVX
              float foo[NSIMD], lambdas0_avx[4];
              _mm256_storeu_ps(foo,r3);   lambdas0_avx[0]=foo[0];
              _mm256_storeu_ps(foo,r6);   lambdas0_avx[1]=foo[0];
              _mm256_storeu_ps(foo,r9);   lambdas0_avx[2]=foo[0];
              _mm256_storeu_ps(foo,r12);  lambdas0_avx[3]=foo[0];
              printf("lambda0_avx=%f,%f,%f,%f\n",
                  lambdas0_avx[0],lambdas0_avx[1],lambdas0_avx[2],lambdas0_avx[3]);
            #endif

            // itetrad=find_best_tetrad(lambdas);
            ir4  = _mm256_set1_epi32(1);

            r5 = _mm256_cmp_ps(r6,r3,_CMP_LT_OQ);
            r15 = _mm256_min_ps(r6,r3);
            ir5 = _mm256_castps_si256(r5);
            ir7  = _mm256_set1_epi32(2);
            ir5 = _mm256_and_si256(ir5,ir7);
            ir4  = _mm256_max_epi32(ir4,ir5);

            r5 = _mm256_cmp_ps(r9,r15,_CMP_LT_OQ);
            r15 = _mm256_min_ps(r9,r15);
            ir5 = _mm256_castps_si256(r5);
            ir7  = _mm256_set1_epi32(3);
            ir5 = _mm256_and_si256(ir5,ir7);
            ir4  = _mm256_max_epi32(ir4,ir5);

            r5 = _mm256_cmp_ps(r12,r15,_CMP_LT_OQ);
            r15 = _mm256_min_ps(r12,r15);
            ir5 = _mm256_castps_si256(r5);
            ir7  = _mm256_set1_epi32(4);
            ir5 = _mm256_and_si256(ir5,ir7);
            ir4  = _mm256_max_epi32(ir4,ir5);

            r7 = _mm256_set1_ps(0.0f);
            r5 = _mm256_cmp_ps(r15,r7,_CMP_LT_OQ);
            ir5 = _mm256_castps_si256(r5);
            ir4 = _mm256_and_si256(ir4,ir5);

            _mm256_store_si256((__m256i *)itetrad_avx,ir4);

            /*
            r4 = _mm256_min_ps(r3,r6);
            r5 = _mm256_min_ps(r9,r12);
            r7 = _mm256_set1_ps(0.0f);
            r4 = _mm256_min_ps(r4,r5);
            r5 = _mm256_cmp_ps(r4,r7,_CMP_LT_OQ);
            ir5 = _mm256_castps_si256(r5);

            //could use _mm256_testz_si256 to skip below...
            //faster?  depends on volume fraction of inner tetrahedron
            //nope.  only 1/8 are in inner, 1/8 on border, 3/4 outside
            unsigned inner[NSIMD];
            ir5 = _mm256_castps_si256(r5);
            _mm256_storeu_si256((__m256i *)inner,ir5);
            printf("INNER %d\n",
                (inner[0]==0)+(inner[1]==0)+(inner[2]==0)+(inner[3]==0)+
                (inner[4]==0)+(inner[5]==0)+(inner[6]==0)+(inner[7]==0));

            r7  = _mm256_cmp_ps(r4,r3,_CMP_EQ_OQ);
            ir7 = _mm256_castps_si256(r7);
            ir8 = _mm256_set1_epi32(1);
            ir7 = _mm256_and_si256(ir5,ir7);
            ir7 = _mm256_and_si256(ir8,ir7);

            r10  = _mm256_cmp_ps(r4,r6,_CMP_EQ_OQ);
            ir10 = _mm256_castps_si256(r10);
            ir8  = _mm256_set1_epi32(2);
            ir10 = _mm256_and_si256(ir5,ir10);
            ir10 = _mm256_and_si256(ir8,ir10);

            r11  = _mm256_cmp_ps(r4,r9,_CMP_EQ_OQ);
            ir11 = _mm256_castps_si256(r11);
            ir8  = _mm256_set1_epi32(3);
            ir11 = _mm256_and_si256(ir5,ir11);
            ir11 = _mm256_and_si256(ir8,ir11);

            r13  = _mm256_cmp_ps(r4,r12,_CMP_EQ_OQ);
            ir13 = _mm256_castps_si256(r13);
            ir8  = _mm256_set1_epi32(4);
            ir13 = _mm256_and_si256(ir5,ir13);
            ir13 = _mm256_and_si256(ir8,ir13);

            ir7  = _mm256_min_epi32(ir7,ir10);
            ir11 = _mm256_min_epi32(ir11,ir13);
            ir7  = _mm256_min_epi32(ir7,ir11);  // itetrad
            _mm256_store_si256((__m256i *)itetrad_avx,ir7);
            */
            #ifdef TEST_AVX
              printf("itetrad_avx=%u\n", itetrad_avx[0]);
            #endif

            // if(itetrad>0) {
            //     map(tetrads+itetrad,lambdas,r);   // Map best tetrahedron
            // }

            // could make this into function as it's repeated above

            // const float * const T=tetrads->T;
            // const float * const ori=tetrads->ori;

            r0 = _mm256_load_ps(r_avx[0]);
            r1 = _mm256_load_ps(r_avx[1]);
            r2 = _mm256_load_ps(r_avx[2]);
            ir15 = _mm256_set1_epi32(12);
            ir15 = _mm256_mul_epi32(ir7,ir15);
            r12 = _mm256_i32gather_ps(ori+0,ir15,4);
            r13 = _mm256_i32gather_ps(ori+1,ir15,4);
            r14 = _mm256_i32gather_ps(ori+2,ir15,4);

            r3  = _mm256_i32gather_ps(T+0,ir15,4);
            r4  = _mm256_i32gather_ps(T+1,ir15,4);
            r5  = _mm256_i32gather_ps(T+2,ir15,4);
            r6  = _mm256_i32gather_ps(T+3,ir15,4);
            r7  = _mm256_i32gather_ps(T+4,ir15,4);
            r8  = _mm256_i32gather_ps(T+5,ir15,4);
            r9  = _mm256_i32gather_ps(T+6,ir15,4);
            r10 = _mm256_i32gather_ps(T+7,ir15,4);
            r11 = _mm256_i32gather_ps(T+8,ir15,4);

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
            r3 = _mm256_add_ps(r3,r5);   // lambda[0]
            r6 = _mm256_add_ps(r6,r8);   // lambda[1]
            r9 = _mm256_add_ps(r9,r11);  // lambda[2]

            r12 = _mm256_set1_ps(1.0f);
            r12 = _mm256_sub_ps(r12,r3);
            r12 = _mm256_sub_ps(r12,r6);
            r12 = _mm256_sub_ps(r12,r9);   // lambda[3]
            _mm256_store_ps(lambdas_avx[0],r3);
            _mm256_store_ps(lambdas_avx[1],r6);
            _mm256_store_ps(lambdas_avx[2],r9);
            _mm256_store_ps(lambdas_avx[3],r12);
            #ifdef TEST_AVX
              printf("lambda_avx=%f,%f,%f,%f\n",
                  lambdas_avx[0][0],lambdas_avx[1][0],lambdas_avx[2][0],lambdas_avx[3][0]);
            #endif

            // if(any_less_than_zero(lambdas,4)) // other boundary
            //     continue;
            r4 = _mm256_min_ps(r3,r6);
            r5 = _mm256_min_ps(r9,r12);
            r7 = _mm256_set1_ps(-EPS);
            r4 = _mm256_min_ps(r4,r5);
            r5 = _mm256_cmp_ps(r4,r7,_CMP_LT_OQ);
            ir5 = _mm256_castps_si256(r5);
            _mm256_store_si256((__m256i *)skip_it_avx,ir5);
            //printf("NEGATIVE %d\n",
            //    (skip_it_avx[0]>0)+(skip_it_avx[1]>0)+(skip_it_avx[2]>0)+(skip_it_avx[3]>0)+
            //    (skip_it_avx[4]>0)+(skip_it_avx[5]>0)+(skip_it_avx[6]>0)+(skip_it_avx[7]>0));
            #ifdef TEST_AVX
              printf("any_less_than_zero_avx=%u\n", skip_it_avx[0]);
            #endif
            
            #ifdef TIME_AVX
              printf("AVX %10gs\t%s\n",toc(&t),"");
              t=tic();
            #endif

            #if defined(TEST_AVX) || defined(TIME_AVX)
              idx2coord(r,idst,dst_shape);
              map(tetrads,lambdas0,r);             // Map center tetrahedron
              itetrad=find_best_tetrad(lambdas0);
              if(itetrad>0) {
                  map(tetrads+itetrad,lambdas,r);   // Map best tetrahedron
              }
              skip_it=any_less_than_zero(lambdas,4); // other boundary
            #endif
            #ifdef TEST_AVX
              printf("r=%f,%f,%f\n",r[0],r[1],r[2]);
              printf("lambda0=%f,%f,%f,%f\n",lambdas0[0],lambdas0[1],lambdas0[2],lambdas0[3]);
              printf("itetrad=%u\n",itetrad);
              printf("lambda=%f,%f,%f,%f\n",lambdas[0],lambdas[1],lambdas[2],lambdas[3]);
              printf("any_less_than_zero=%u\n", skip_it);
              if((r[0]!=r_avx[0][0]) ||
                 (r[1]!=r_avx[1][0]) ||
                 (r[2]!=r_avx[2][0])) printf("ERROR rs not equal\n");
              if(itetrad!=itetrad_avx[0]) printf("ERROR itetrads not equal\n");
              if((lambdas[0]!=lambdas_avx[0][0]) ||
                 (lambdas[1]!=lambdas_avx[1][0]) ||
                 (lambdas[2]!=lambdas_avx[2][0])) printf("ERROR lambdas' not equal, itetrad=%d\n",itetrad);
              if((skip_it==0) != (skip_it_avx[0]==0)) printf("ERROR any_less_than_zeros not equal\n");
            #endif
            #ifdef TIME_AVX
              printf("CPU %10gs\t%s\n",toc(&t),"");
              t=tic();
            #endif
            
            // Map source index
            for(idst2=0; idst2<NSIMD; idst2++)
            {
                if((idst+idst2)>idst_max)  break;
                if(skip_it_avx[idst2]>0)  continue;

                if(method==0) { //  nathan's original nearest neighbor
                  unsigned idim,ilambda,isrc=0;
                  for(idim=0;idim<3;++idim) {
                      float s=0.0f;
                      const float d=(float)(src_shape[idim]);
                      for(ilambda=0;ilambda<4;++ilambda) {
                          const float      w=lambdas_avx[ilambda][idst2];
                          const unsigned idx=(*indexes)[itetrad_avx[idst2]][ilambda];
                          s+=w*BIT(idx,idim);
                      }
                      s*=d;
                      s=(s<0.0f)?0.0f:(s>(d-1))?(d-1):s;
                      isrc+=src_strides[idim]*((unsigned)s); // important to floor here.  can't change order of sums
                  }
                  dst[idst+idst2]=src[isrc]; }

                else if(method==1) { // ben's trilinear
                  float frac[3],tmp,c00,c10,c01,c11,c0,c1;
                  unsigned zero[3],one[3];
                  unsigned idim,ilambda;
                  for(idim=0;idim<3;++idim) {
                    float s=0.0f;
                    const float d=(float)(src_shape[idim]);
                    for(ilambda=0;ilambda<4;++ilambda) {
                      const float      w=lambdas_avx[ilambda][idst2];
                      const unsigned idx=(*indexes)[itetrad_avx[idst2]][ilambda];
                      s+=w*BIT(idx,idim); }
                    s*=d;
                    s=(s<0.0f)?0.0f:(s>(d-1))?(d-1):s;
                    frac[idim] = modff(s,&tmp);
                    zero[idim]=(unsigned)tmp;
                    one[idim]=zero[idim]+1; }
                  if(zero[0]>=src_shape[0] || zero[1]>=src_shape[1] || zero[2]>=src_shape[2]) continue;
                  if(one[0]>=src_shape[0]) one[0]--;  // otherwise pixels are black at edges, not sure why
                  if(one[1]>=src_shape[1]) one[1]--;
                  if(one[2]>=src_shape[2]) one[2]--;
                  tmp = (1.0f-frac[0]);
                  c00 = src[zero[0]*src_strides[0]+zero[1]*src_strides[1]+zero[2]*src_strides[2]]*tmp
                      + src[ one[0]*src_strides[0]+zero[1]*src_strides[1]+zero[2]*src_strides[2]]*frac[0];
                  c10 = src[zero[0]*src_strides[0]+ one[1]*src_strides[1]+zero[2]*src_strides[2]]*tmp
                      + src[ one[0]*src_strides[0]+ one[1]*src_strides[1]+zero[2]*src_strides[2]]*frac[0];
                  c01 = src[zero[0]*src_strides[0]+zero[1]*src_strides[1]+ one[2]*src_strides[2]]*tmp
                      + src[ one[0]*src_strides[0]+zero[1]*src_strides[1]+ one[2]*src_strides[2]]*frac[0];
                  c11 = src[zero[0]*src_strides[0]+ one[1]*src_strides[1]+ one[2]*src_strides[2]]*tmp
                      + src[ one[0]*src_strides[0]+ one[1]*src_strides[1]+ one[2]*src_strides[2]]*frac[0];
                  tmp = (1.0f-frac[1]);
                  c0 = c00*tmp + c10*frac[1];
                  c1 = c01*tmp + c11*frac[1];
                  dst[idst+idst2] = c0*(1.0f-frac[2]) + c1*frac[2]; }
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
    thread_t ts[8]={0};
    struct work jobs[8]={0};
    unsigned i;

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
