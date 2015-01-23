#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

typedef uint8_t TPixel; /* Note (3) */

struct resampler {
    void *ctx;
};

int BarycentricCPUinit(struct resampler*,
                const unsigned * const,
                const unsigned * const,
                const unsigned);
int BarycentricCPUsource(const struct resampler *,
                  TPixel * const);
int BarycentricCPUdestination(struct resampler *,
                       TPixel * const);
int BarycentricCPUresult(const struct resampler * const,
                  TPixel * const);
int BarycentricCPUresample(struct resampler * const,
                    const float * const);
void BarycentricCPUrelease(struct resampler *);
void BarycentricCPUuseReporters( void (*)  (const char*, const char*, const char*, int, const char*,void*),
                          void (*)(const char*, const char*, const char*, int, const char*,void*),
                          void (*)   (const char*, const char*, const char*, int, const char*,void*),
                          void *);
int BarycentricCPUrunTests(void);


int BarycentricGPUinit(struct resampler*,
                const unsigned * const,
                const unsigned * const,
                const unsigned);
int BarycentricGPUsource(struct resampler*,
                  TPixel * const);
int BarycentricGPUdestination(struct resampler *,
                       TPixel * const);
int BarycentricGPUresult(const struct resampler * const,
                  TPixel * const);
int BarycentricGPUresample(struct resampler * const,
                    const float * const);
void BarycentricGPUrelease(struct resampler *);
void BarycentricGPUuseReporters( void (*)  (const char*, const char*, const char*, int, const char*,void*),
                          void (*)(const char*, const char*, const char*, int, const char*,void*),
                          void (*)   (const char*, const char*, const char*, int, const char*,void*),
                          void *);
int  BarycentricGPUrunTests(void);

#ifdef __cplusplus
} // extern "C"
#endif


/* TODO

1. refactor resample call to support
  
   - load src once
   - render many outputs

2. Add ability to query capabilities (max dim,max shape,allowable pixel types)

*/

/* NOTES

   1. Renderer will not handle all possible inputs. See Todo (2).
      Assumes dst and src (of course) are allocated by caller.

      ndim MUST BE 3

   2. runTests() is here so that static utility functions can be run through
      their paces.  The utility functions are static because they're private
      parts of the interface; I don't want to worry about namespace pollution.

   3. Figure out how to generalize the interface over pixel types later.
      It's probably not necessary, and if it is, it will be simple to adapt
      the existing code.

   4. GPU renderer assumes transfer to/from RAM is desired.  See Todo (1).

      Recommend implementing another interface if src,dst are supposed to 
      be device pointers.  Esp since there may be additional requirements on 
      the type of gpu storage there...

      There's probably a way of generalizing so code gets reused on the 
      backend.  The interface here can stay the same and the caller just
      chooses the right implementaiton for what they want.

*/
