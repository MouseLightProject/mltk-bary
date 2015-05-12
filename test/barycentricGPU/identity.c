#include <resamplers.h>
#include <stdio.h>
#include <stdlib.h>  
#include <nd.h>
#include <tictoc.h>
#include <math.h>

#define ASSERT(e) do{if(!(e)) {printf("%s(%d): %s()(\n\tExpression evaluated as false.\n\t%s\n",__FILE__,__LINE__,__FUNCTION__,#e); exit(2); }}while(0)
#define TIME(e) do{TicTocTimer t=tic(); {e;} printf("TIME %10fs\t%s\n",toc(&t),#e);} while(0)
#define countof(e) (sizeof(e)/sizeof(*(e)))

static unsigned eq(const TPixel * const a, const TPixel * const b,unsigned n) {
    unsigned i;
    for(i=0;i<n;++i) {
        if(a[i]!=b[i])
            return 0;
    }
    return 1;
}

TPixel src[64*64*64];
TPixel dst[64*64*64];

const unsigned src_shape[]={64,64,64};
const unsigned src_stride[]={1,64,64*64};
const unsigned dst_shape[]={64,64,64};
const unsigned dst_stride[]={1,64,64*64};

const float cube[]={
     0, 0, 0,
    64, 0, 0,
     0,64, 0,
    64,64, 0,
     0, 0,64,
    64, 0,64,
     0,64,64,
    64,64,64,
};

int main(int argc,char* argv[]) {

    {
        unsigned x,y,z,i=0,scale=(8*sizeof(TPixel)-(int)log2(64));
        for(z=0;z<64;++z) for(y=0;y<64;++y) for(x=0;x<64;++x,++i) {
            src[i]=((x/3)^(y/3)^(z/3))<<scale;
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
        const size_t shape_sz[]={64,64,64};
        nd_t v=ndref(ndreshape(ndcast(ndinit(),nd_u16),3,shape_sz),src,nd_static);
        ndioClose(ndioWrite(ndioOpen("src.tif",NULL,"w"),v));
    }
    {
        const size_t shape_sz[]={64,64,64};
        nd_t v=ndref(ndreshape(ndcast(ndinit(),nd_u16),3,shape_sz),dst,nd_static);
        ndioClose(ndioWrite(ndioOpen("dst.tif",NULL,"w"),v));
    }
#endif


    ASSERT(eq(src,dst,countof(src)));
    return 0;
}
