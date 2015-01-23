#include <resamplers.h>
#include <stdlib.h>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstring>

#define EPS (1e-5f)
#define BIT(k,i) (((k)>>(i))&1)

/*
 *  LOGGING
 */

#include <stdio.h>
#ifdef _MSC_VER
#define  _CRT_SECURE_NO_WARNINGS
#define snprintf _snprintf
#endif

static void default_reporter(const char* msg, const char* expr, const char* file, int line, const char* function,void* usr) {
    (void)usr;
    printf("%s(%d) - %s()\n\t%s\n\t%s\n",file,line,function,msg,expr);
}

typedef void (*reporter_t)(const char* msg, const char* expr, const char* file, int line, const char* function,void* usr);

static void*      reporter_context_=0;
static reporter_t error_  =&default_reporter;
static reporter_t warning_=&default_reporter;
static reporter_t info_   =&default_reporter;

void BarycentricGPUuseReporters(
    void (*error)  (const char* msg, const char* expr, const char* file, int line, const char* function,void* usr),
    void (*warning)(const char* msg, const char* expr, const char* file, int line, const char* function,void* usr),
    void (*info)   (const char* msg, const char* expr, const char* file, int line, const char* function,void* usr),
    void *usr) {
    error_  =error;
    warning_=warning;
    info_   =info;
    reporter_context_=usr;
}

#define ERR(e,...)  do{char msg[1024]; snprintf(msg,sizeof(msg),__VA_ARGS__); error_  (msg,#e,__FILE__,__LINE__,__FUNCTION__,reporter_context_);}while(0)
#define WARN(e,...) do{char msg[1024]; snprintf(msg,sizeof(msg),__VA_ARGS__); warning_(msg,#e,__FILE__,__LINE__,__FUNCTION__,reporter_context_);}while(0)
#define INFO(...)   do{char msg[1024]; snprintf(msg,sizeof(msg),__VA_ARGS__); info_   (msg,"Info",__FILE__,__LINE__,__FUNCTION__,reporter_context_);}while(0)

#define ASSERT(e) do{if(!(e)) {ERR(e,"Expression evaluated as false."); throw 1; }} while(0)
#define CUTRY(e)  do{cudaError_t ecode=(e); if(ecode!=cudaSuccess) {ERR(e,cudaGetErrorString(ecode)); throw 1; }} while(0)

/*
 *  PRELUDE
 */

#define countof(e) (sizeof(e)/sizeof(*(e)))

#define restrict __restrict__

template<typename t> cudaChannelFormatKind channelformatkind(void);
template<> cudaChannelFormatKind channelformatkind<uint8_t >(void) {return cudaChannelFormatKindUnsigned;}
template<> cudaChannelFormatKind channelformatkind<uint16_t>(void) {return cudaChannelFormatKindUnsigned;}
template<> cudaChannelFormatKind channelformatkind<uint32_t>(void) {return cudaChannelFormatKindUnsigned;}
template<> cudaChannelFormatKind channelformatkind<uint64_t>(void) {return cudaChannelFormatKindUnsigned;}
template<> cudaChannelFormatKind channelformatkind< int8_t >(void) {return cudaChannelFormatKindSigned;}
template<> cudaChannelFormatKind channelformatkind< int16_t>(void) {return cudaChannelFormatKindSigned;}
template<> cudaChannelFormatKind channelformatkind< int32_t>(void) {return cudaChannelFormatKindSigned;}
template<> cudaChannelFormatKind channelformatkind< int64_t>(void) {return cudaChannelFormatKindSigned;}
template<> cudaChannelFormatKind channelformatkind<float   >(void) {return cudaChannelFormatKindFloat;}
template<> cudaChannelFormatKind channelformatkind<double  >(void) {return cudaChannelFormatKindFloat;}

static unsigned prod(const unsigned * const v,unsigned n) {
    unsigned p=1;
    const unsigned *c=v+n;
    while(c-->v) p*=*c;
    return p;
}

/*
 *  INTERFACE
 */

texture<TPixel,cudaTextureType3D,cudaReadModeElementType> in;

struct ctx_t {
    cudaArray *src;
    TPixel    *dst;          // cuda device pointer
    unsigned   dst_shape[3]; // output shape
    unsigned   src_shape[3]; // input shape
};

struct tetrahedron {
    float T[9];
    float ori[3];
};

struct work {
    TPixel   *dst;          // cuda device pointer
    unsigned  src_shape[3];
    unsigned  dst_shape[3];
    struct tetrahedron tetrads[5];
};

int BarycentricGPUinit(struct resampler* self,
                const unsigned * const src_shape,
                const unsigned * const dst_shape,
                const unsigned ndim) {
    try {
        memset(self,0,sizeof(*self));
        ASSERT(ndim==3);
        ctx_t *c=new ctx_t;
        self->ctx=c;
        // dst
        memcpy(c->dst_shape,dst_shape,sizeof(c->dst_shape));
        CUTRY(cudaMalloc(&c->dst,sizeof(TPixel)*prod(dst_shape,3)));
        CUTRY(cudaMemset( c->dst,0,sizeof(TPixel)*prod(dst_shape,3)));

        // src
        memcpy(c->src_shape,src_shape,sizeof(*src_shape)*ndim);
      
        in.addressMode[0]=cudaAddressModeClamp;
        in.addressMode[1]=cudaAddressModeClamp;
        in.addressMode[2]=cudaAddressModeClamp;
        in.filterMode=cudaFilterModePoint;
        in.normalized=1;

        cudaChannelFormatDesc d=cudaCreateChannelDesc(sizeof(TPixel)*8,0,0,0,channelformatkind<TPixel>());
        const struct cudaExtent extent=make_cudaExtent(src_shape[0],src_shape[1],src_shape[2]);
        CUTRY(cudaMalloc3DArray(&c->src,&d,extent));

    } catch(int) {
        return 0;
    }
    return 1;
}

void BarycentricGPUrelease(struct resampler* self) {
    try {
        if(self->ctx) {
            ctx_t *c=(ctx_t*)self->ctx;
            CUTRY(cudaFree(c->dst));
            cudaFreeArray(c->src);
            delete c;
            self->ctx=0;
        }
    }
    catch(...) {;}
}

int BarycentricGPUsource(struct resampler* self,TPixel * const src) {
    try {
        ctx_t *c=(ctx_t*)self->ctx;

        cudaChannelFormatDesc d=cudaCreateChannelDesc(sizeof(TPixel)*8,0,0,0,channelformatkind<TPixel>());
        const struct cudaExtent extent=make_cudaExtent(c->src_shape[0],c->src_shape[1],c->src_shape[2]);

        // copy data to 3D array
        cudaMemcpy3DParms copy={0};
        copy.srcPtr   = make_cudaPitchedPtr(src,c->src_shape[0]*sizeof(TPixel),c->src_shape[0],c->src_shape[1]);
        copy.dstArray = c->src;
        copy.extent   = extent;
        copy.kind     = cudaMemcpyHostToDevice;
        CUTRY(cudaMemcpy3D(&copy));
        // bind to texture
        CUTRY(cudaBindTextureToArray(&in,c->src,&d));
        
    } catch(int) { /* CUTRY macro throws ints */
        return 0;
    }
    return 1;
}

int BarycentricGPUdestination(struct resampler* self,TPixel * const dst) {
    try {
        ctx_t *c=(ctx_t*)self->ctx;
        CUTRY(cudaMemcpy(c->dst,dst,sizeof(TPixel)*prod(c->dst_shape,3),cudaMemcpyHostToDevice));
    } catch(int) { /* CUTRY macro throws ints */
        return 0;
    }
    return 1;
}

int BarycentricGPUresult(const struct resampler* self, TPixel * const dst) {
    try {
        ctx_t *ctx=(ctx_t*)self->ctx;
        CUTRY(cudaMemcpy(dst,ctx->dst,sizeof(TPixel)*prod(ctx->dst_shape,3),cudaMemcpyDeviceToHost));
    } catch(int) {
        return 0;
    }
    return 1;
}

static float* inv33(float * restrict inverse,const float * const restrict t) {
#define det(a,b,c,d) (t[a]*t[d]-t[b]*t[c])
    const float n=t[0]*det(4,5,7,8)-t[1]*det(3,5,6,8)+t[2]*det(3,4,6,7);
    float invT[3][3]={
            {det(4,5,7,8)/n,det(2,1,8,7)/n,det(1,2,4,5)/n},
            {det(5,3,8,6)/n,det(0,2,6,8)/n,det(2,0,5,3)/n},
            {det(3,4,6,7)/n,det(1,0,7,6)/n,det(0,1,3,4)/n},
    };
#undef det
    memcpy(inverse,invT,sizeof(invT));
    return inverse;
}

/*
static __device__ float* mul(float * restrict dst,
                  const float * const restrict lhs,const unsigned nrows_lhs,const unsigned ncols_lhs,
                  const float * const restrict rhs,const unsigned ncols_rhs) {
    unsigned i,j,k;
    for(k=0;k<nrows_lhs;++k) {
        for(j=0;j<ncols_rhs;++j) {
            float s=0.0f;
            for(i=0;i<ncols_lhs;++i) {
                s+=lhs[ncols_lhs*k+i]*rhs[ncols_rhs*i+j];
            }
            dst[ncols_rhs*k+j]=s;
        }
    }
    return dst;
}
*/

static void tetrahedron(struct tetrahedron *self, const float * const v, const unsigned * idx) {
    const float T[9] = {
        v[3*idx[0]  ]-v[3*idx[3]  ],v[3*idx[1]  ]-v[3*idx[3]  ],v[3*idx[2]  ]-v[3*idx[3]  ],
        v[3*idx[0]+1]-v[3*idx[3]+1],v[3*idx[1]+1]-v[3*idx[3]+1],v[3*idx[2]+1]-v[3*idx[3]+1],
        v[3*idx[0]+2]-v[3*idx[3]+2],v[3*idx[1]+2]-v[3*idx[3]+2],v[3*idx[2]+2]-v[3*idx[3]+2],
    };
    inv33(self->T,T); 
    memcpy(self->ori,v+3*idx[3],sizeof(self->ori));
};

/* KERNEL */

inline __device__ unsigned prod(const dim3 a)                  {return a.x*a.y*a.z;}
inline __device__ unsigned stride(const uint3 a, const dim3 b) {return a.x+b.x*(a.y+b.y*a.z);}
inline __device__ unsigned sum(const uint3 a)                  {return a.x+a.y+a.z;}

inline __device__ 
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

inline __device__
unsigned find_best_tetrad(const float * const restrict ls) {
    float v=ls[0];
    unsigned argmin=1;
    if(ls[1]<v) {
        v=ls[1];
        argmin=2;
    }
    if(ls[2]<v) {
        v=ls[2];
        argmin=3;
    }
    if(ls[3]<v) {
        v=ls[3];
        argmin=4;
    }

    if(v>=0.0f)
        return 0;
    return argmin;
}

#define any_less_than_zero4(v) ((v[0]<-EPS)||(v[1]<-EPS)||(v[2]<-EPS)||(v[3]<-EPS))       

/* 4 indexes each for 5 tetrads; the first is the center tetrad */
static __constant__ unsigned indexes_k[5][4]={
        {1,2,4,7},
        {2,4,6,7}, // opposite 1
        {1,4,5,7}, // opposite 2
        {1,2,3,7}, // opposite 4
        {0,1,2,4}  // opposite 7
};
static const unsigned indexes[5][4]={
        {1,2,4,7},
        {2,4,6,7}, // opposite 1
        {1,4,5,7}, // opposite 2
        {1,2,3,7}, // opposite 4
        {0,1,2,4}  // opposite 7
};


static __constant__ struct work work;

template<
    unsigned BX,
    unsigned BY,
    unsigned WORK
>
__global__
void
__launch_bounds__(BX*BY,1) // max threads, min blocks
resample_k(void) {
    const unsigned ox=threadIdx.x+blockIdx.x*BX;
    const unsigned oy=(threadIdx.y+blockIdx.y*BY)*WORK;
    const unsigned idst0=ox+oy*BX;

    for(unsigned idst=idst0;idst<(idst0+WORK*BX);idst+=BX) {
        float lambdas[4];
        unsigned itetrad;
        {
            float r[3];
            {   unsigned idx=idst;
                r[0]=idx%work.dst_shape[0];  idx/=work.dst_shape[0];
                r[1]=idx%work.dst_shape[1];  idx/=work.dst_shape[1];
                r[2]=idx%work.dst_shape[2];
            }
        
            map(work.tetrads,lambdas,r);             // Map center tetrahedron
                
            itetrad=find_best_tetrad(lambdas);
            if(itetrad>0) {
                map(work.tetrads+itetrad,lambdas,r);   // Map best tetrahedron
            }
        }
                
        if(!any_less_than_zero4(lambdas)) { // other boundary
            // Map source index
            float r[3];
            #pragma unroll
            for(int idim=0;idim<3;++idim) {
                float s=0.0f;
                for(int ilambda=0;ilambda<4;++ilambda) {
                    const float      w=lambdas[ilambda];
                    const unsigned idx=indexes_k[itetrad][ilambda];
                    s+=w*BIT(idx,idim);                        
                }
                r[idim]=s;
            }

            work.dst[idst]=max(work.dst[idst],tex3D(in,r[0],r[1],r[2]));
        }
    }
}


static unsigned nextdim(unsigned n, unsigned limit, unsigned *rem)
{ unsigned v=limit,c=limit,low=n/limit,argmin=0,min=limit;
  *rem=0;  
  if(n<limit) return n;
  for(c=low+1;c<limit&&v>0;c++)
  { v=(unsigned)(c*ceil(n/(float)c)-n);
    if(v<min)
    { min=v;
      argmin=c;
    }
  }
  *rem= (min!=0);
  return argmin;
}

int BarycentricGPUresample(struct resampler * const self,
                     const float * const cubeverts) {

    struct work arg={0};
    ctx_t * const c=(ctx_t*)self->ctx;
    printf("sizeof work: %d\n",(int)sizeof(arg));
    arg.dst=c->dst;
    memcpy(arg.dst_shape,c->dst_shape,sizeof(arg.dst_shape));
    memcpy(arg.src_shape,c->src_shape,sizeof(arg.dst_shape));
    
    for(unsigned i=0;i<5;i++)
        tetrahedron(arg.tetrads+i,cubeverts,indexes[i]);

    try {
        const unsigned BX=32*4,BY=1,WORK=32,N=prod(arg.dst_shape,3);
        dim3 threads(BX,BY),
             blocks(1,N/BX/BY/WORK);
        CUTRY(cudaMemcpyToSymbol(work,&arg,sizeof(arg),0));
        CUTRY(cudaGetLastError());
        resample_k<BX,BY,WORK><<<blocks,threads>>>();
        CUTRY(cudaGetLastError());
    } catch(int) {
        return 0;
    }
    return 1;
}

/*           */
/* INTERFACE */
/*           */

/*       */
/* TESTS */
/*       */


/* simpleTransformWithTexture */
#if 0
namespace simpleTransformWithTexture {

    texture<float,cudaTextureType2D,cudaReadModeElementType> src;

    __global__ void simpleTransformWithTexture_k(float *dst,int w,int h,float th) {
        unsigned int x=__umul24(blockIdx.x,blockDim.x)+threadIdx.x;
        unsigned int y=__umul24(blockIdx.y,blockDim.y)+threadIdx.y;
        float u = (x/(float)w)-0.5f,
              v = (y/(float)h)-0.5f,
             tu = u*cosf(th)-v*sinf(th)+0.5f,
             tv=u*sinf(th)+v*cosf(th)+0.5f;
        {
            unsigned int i=__umul24(y,w)+x;
            dst[i]=tex2D(src,tu,tv);
        }
        
    }

    static int simpleTransformWithTexture(void) {
        const int w=256,h=256;
        try { /* CUTRY macro throws ints */
            cudaArray *a;

            src.addressMode[0]=cudaAddressModeWrap;
            src.addressMode[1]=cudaAddressModeWrap;
            src.filterMode=cudaFilterModeLinear;
            src.normalized=1;
            { 
                cudaChannelFormatDesc d=cudaCreateChannelDesc(32,0,0,0,cudaChannelFormatKindFloat);
                CUTRY(cudaMallocArray(&a,&d,w,h));
                CUTRY(cudaBindTextureToArray(src,a,d));

            }
            // --> should copy source data in at this point (cudaMemcpyToArray) <-- 

            float *out;
            CUTRY(cudaMalloc(&out,w*h*sizeof(float)));

            dim3 block(16,16),
                 grid((w+block.x-1)/block.x,
                      (h+block.y-1)/block.y);
            simpleTransformWithTexture_k<<<grid,block>>>(out,w,h,15*3.14159f/180.0f);

            // clean up
            cudaFree(out);
            cudaFreeArray(a);

        } catch(...) {
            return 1;
        }
        return 0;
    }

} // end namespace simpleTransformWithTexture
#endif

/* Test directory */


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
    /* unused?
    struct tetrahedron expected={
            {18.0f,19.0f,20.0f}
    };
    */
    for(i=0;i<8;++i) {
        v[3*i+0]=(float)BIT(i,0);
        v[3*i+1]=(float)BIT(i,1);
        v[3*i+2]=(float)BIT(i,2);
    }
    tetrahedron(&ans,v,idx);
    ASSERT(eq(ans.ori,v+3*7,3));
    return 0;
}

static int (*tests[])(void)={
    //simpleTransformWithTexture::simpleTransformWithTexture,
    test_testrahedron,
};

int BarycentricGPUrunTests() {
    int i;
    int nfailed=0;
    for(i=0;i<countof(tests);++i) {
        nfailed+=tests[i]();
    }
    return nfailed;
}
