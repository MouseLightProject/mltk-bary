#include <resamplers.h>

#include <stdio.h>
#include <stdlib.h>
#define ASSERT(e) do{if(!(e)) {printf("%s(%d): %s()(\n\tExpression evaluated as false.\n\t%s\n",__FILE__,__LINE__,__FUNCTION__,#e); exit(1); }}while(0)

int main(int argc,char*argv[]) {
    ASSERT(0==BarycentricAVXrunTests());
    return 0;
}
