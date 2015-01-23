#include <resamplers.h>
#include <stdio.h>
#include <stdlib.h>  
#include <nd.h>
#include <tictoc.h>

#define ASSERT(e)  do{if(!(e)) {printf("%s(%d): %s()(\n\tExpression evaluated as false.\n\t%s\n",__FILE__,__LINE__,__FUNCTION__,#e); exit(1); }}while(0)
#define TIME(e)    do{TicTocTimer t=tic(); {e;} printf("TIME %10fs\t%s\n",toc(&t),#e);} while(0)
#define countof(e) (sizeof(e)/sizeof(*(e)))

static unsigned eq(const TPixel * const a, const TPixel * const b,unsigned n) {
    unsigned i;
    for(i=0;i<n;++i) {
        if(a[i]!=b[i])
            return 0;
    }
    return 1;
}

#define NX (512)
#define NY (512)
#define NZ (512)


const unsigned src_shape []={NX,NY,NZ};
const unsigned src_stride[]={1,NX,NX*NY};
const unsigned dst_shape []={NX,NY,NZ};
const unsigned dst_stride[]={1,NX,NX*NY};


#define s (0.5f)

#if 1 
    // 30 degrees
    #define sn (0.5f)
    #define cs (0.866f)
#elif 0
    // 5 degrees
    #define sn (0.0872f)
    #define cs (0.9962)
#elif 1 
    // 0 degrees
    #define sn (0.0f)
    #define cs (1.0f)
#endif

#define tx(x,y) (( cs*((x)-0.5f)-sn*((y)-0.5f) )*s+0.5f)
#define ty(x,y) (( sn*((x)-0.5f)+cs*((y)-0.5f) )*s+0.5f)

const float cube[]={
    NX*tx(0,0),NY*ty(0,0),0.0*NZ,
    NX*tx(1,0),NY*ty(1,0),0.0*NZ,
    NX*tx(0,1),NY*ty(0,1),0.0*NZ,
    NX*tx(1,1),NY*ty(1,1),0.0*NZ,
     0, 0,NZ,
    NX, 0,NZ,
     0,NY,NZ,
    NX,NY,NZ,
};

#undef s

#undef cs
#undef sn

#undef tx
#undef ty

int main(int argc,char* argv[]) {

    TPixel *src,*dst;

    ASSERT(src=(TPixel*)malloc(sizeof(TPixel)*NX*NY*NZ));
    ASSERT(dst=(TPixel*)calloc(NX*NY*NZ,sizeof(TPixel)));


    {

        unsigned x,y,z,i=0;
        for(z=0;z<NZ;++z) for(y=0;y<NY;++y) for(x=0;x<NX;++x,++i) {
            src[i]=(x)^(y)^(z);
        }

    }

    {
        struct resampler r;
        TIME(ASSERT( BarycentricGPUinit  (&r,src_shape,dst_shape,3)));
        TIME(ASSERT( BarycentricGPUsource(&r,src)));
        TIME(ASSERT( BarycentricGPUdestination(&r,dst)));
        TIME(ASSERT( BarycentricGPUresample(&r,cube)));
        TIME(ASSERT( BarycentricGPUresult(&r,dst)));
                     BarycentricGPUrelease(&r);
    }
    

#if 1
    ndioAddPluginPath("plugins");
    {   
        const size_t shape_sz[]={NX,NY,NZ};
        nd_t v=ndref(ndreshape(ndcast(ndinit(),nd_u8),3,shape_sz),src,nd_static);
        ndioClose(ndioWrite(ndioOpen("src.tif",NULL,"w"),v));
    }
    {
        const size_t shape_sz[]={NX,NY,NZ};
        nd_t v=ndref(ndreshape(ndcast(ndinit(),nd_u8),3,shape_sz),dst,nd_static);
        ndioClose(ndioWrite(ndioOpen("dst.tif",NULL,"w"),v));
    }
#endif

    return 0;
}
