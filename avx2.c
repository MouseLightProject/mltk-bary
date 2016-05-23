// get AVX intrinsics
// #include <immintrin.h>
//#include <x86intrin.h>
//#include <stdio.h>
#include <stdlib.h>
#include <stdio.h>
#include <immintrin.h>
#include <x86intrin.h>
// get CPUID capability
// #include <intrin.h>
// written for clarity, not conciseness
#define OSXSAVEFlag (1UL<<27)
#define AVXFlag    ((1UL<<28)|OSXSAVEFlag)
#define VAESFlag   ((1UL<<25)|AVXFlag|OSXSAVEFlag)
#define FMAFlag    ((1UL<<12)|AVXFlag|OSXSAVEFlag)
#define CLMULFlag  ((1UL<< 1)|AVXFlag|OSXSAVEFlag)
/*
int DetectFeature(unsigned int feature)
       {
       int CPUInfo[4], InfoType=1; //, ECX = 1;
       __cpuidex(CPUInfo, 1, 1);       // read the desired CPUID format
       unsigned int ECX = CPUInfo[2];  // the output of CPUID in the ECX register.
       if ((ECX & feature) != feature) // Missing feature
       return 0;
__int64_t val = _xgetbv(0);
if ((val&6) != 6)
       return 0;
return 1;
}
*/

int main(int argc, const char * argv[]) {
  printf("AVX: %d\n",_may_i_use_cpu_feature(_FEATURE_AVX));
  printf("AVX2: %d\n",_may_i_use_cpu_feature(_FEATURE_AVX2));
  //printf("AVX: %d",DetectFeature(AVXFlag));
}
