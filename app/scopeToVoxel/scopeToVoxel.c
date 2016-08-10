// ECL : How I compiled on my machine prior to cmake
//
// gcc -I/home/legrose/env_local/include/tilebase -I/home/legrose/env_local/include/nd -I/home/legrose/src/tilebase -I/home/legrose/src/nd -I/home/legrose/src/mltk-bary/src -I/home/legrose/src/mltk-bary -L/usr/local/cuda-7.5/lib64 -L/home/legrose/env_local/lib -L/home/legrose/env_local/build/mltk-bary -fPIC -mavx2 -mfma otherFunctions.h /home/legrose/src/mltk-bary/src/barycentric.cpu.c finder.c -ltilebase -lnd -lcudart -lcufft -lengine -Wl,-rpath=/home/legrose/env_local/lib -Wl,-rpath=/usr/local/cuda-7.5/lib64 -Wl,-rpath=/home/legrose/env_local/build/mltk-bary

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <limits.h>
#include <inttypes.h>
#include <tilebase.h>
#include <resamplers.h>

#define VERSION (1.0)
#define NAME "scopeToVoxel"

FILE *outFile, *inFile, *logFile;
int quiet = 0;

static void printHelp() {
  fprintf(stdout, "\n%s -help:\n", NAME);
  fprintf(stdout, "\t-i/-input:  provide an input file for %s to parse. If none specified, default is \"-i=input.txt\"\n");
  fprintf(stdout, "\t-o/-output: provide an output file for %s to print results. If none specified, default is \"-o=output.txt\"\n");
  fprintf(stdout, "\t-l/-log:    provide a log file for %s to print runtime information. If none specified, default is \"-l=stdout\"\n");
  fprintf(stdout, "\t-q/-quiet:  run program with minimal output to screen\n\n");
}


// ECL : Just a small utility used in guessing a subvolume
static int64_t dist2(const int64_t x1, const int64_t y1, const int64_t z1, 
                     const int64_t x2, const int64_t y2, const int64_t z2){
  return ((x1-x2)*(x1-x2))+((y1-y2)*(y1-y2))+((z1-z2)*(z1-z2));
}


tile_t findTile(tiles_t *ymlFile, char *tilePath) {
  unsigned idx, size = TileBaseCount(*ymlFile);
  for(idx = 0; idx < size; idx++){
    tile_t t = TileBaseIndex(*ymlFile, idx);
    if(strcmp(TilePath(t), tilePath) == 0)
      return t;
  }
  return 0;
}




static float det3x3(const float matrix[9]) {
  return matrix[0] * matrix[4] * matrix[8] + 
         matrix[1] * matrix[5] * matrix[6] +
         matrix[2] * matrix[3] * matrix[7] -
         matrix[2] * matrix[4] * matrix[6] -
         matrix[1] * matrix[3] * matrix[8] -
         matrix[0] * matrix[5] * matrix[7];
}




static float det4x4(const float matrix[16]) {
  float a[9] = {matrix[ 5], matrix[ 6], matrix[ 7],
                matrix[ 9], matrix[10], matrix[11],
                matrix[13], matrix[14], matrix[15]};

  float b[9] = {matrix[ 4], matrix[ 6], matrix[ 7],
                matrix[ 8], matrix[10], matrix[11],
                matrix[12], matrix[14], matrix[15]};

  float c[9] = {matrix[ 4], matrix[ 5], matrix[ 7],
                matrix[ 8], matrix[ 9], matrix[11],
                matrix[12], matrix[13], matrix[15]};

  float d[9] = {matrix[ 4], matrix[ 5], matrix[ 6],
                matrix[ 8], matrix[ 9], matrix[10],
                matrix[12], matrix[13], matrix[14]};

  return (matrix[0] * det3x3(a)) - (matrix[1] * det3x3(b)) + (matrix[2] * det3x3(c)) - (matrix[3] * det3x3(d)); 
}


float err = 0.0001f;
// ECL : Calculates if a point is within a tetrahedron using determinants. 
//
//       Note: this fuction acually can also be modified to tell which plane the point is outside of,
//     based on if there was a sign change on d1-d4. 
static int pointInTet(const float point[3], const float tet[12]) {
  float D0[16] = {tet[ 0], tet[ 1], tet[ 2], 1,
                  tet[ 3], tet[ 4], tet[ 5], 1,
                  tet[ 6], tet[ 7], tet[ 8], 1,
                  tet[ 9], tet[10], tet[11], 1};

  float D1[16] = {point[0], point[1], point[2], 1,
                   tet[ 3],  tet[ 4],  tet[ 5], 1,
                   tet[ 6],  tet[ 7],  tet[ 8], 1,
                   tet[ 9],  tet[10],  tet[11], 1};
  
  float D2[16] = { tet[ 0],  tet[ 1],  tet[ 2], 1,
                  point[0], point[1], point[2], 1,
                   tet[ 6],  tet[ 7],  tet[ 8], 1,
                   tet[ 9],  tet[10],  tet[11], 1};
  
  float D3[16] = { tet[ 0],  tet[ 1],  tet[ 2], 1,
                   tet[ 3],  tet[ 4],  tet[ 5], 1,
                  point[0], point[1], point[2], 1,
                   tet[ 9],  tet[10],  tet[11], 1};
  
  float D4[16] = { tet[ 0],  tet[ 1],  tet[ 2], 1,
                   tet[ 3],  tet[ 4],  tet[ 5], 1,
                   tet[ 6],  tet[ 7],  tet[ 8], 1,
                  point[0], point[1], point[2], 1};
  
  if(det4x4(D0) > 0){
    if(det4x4(D1) < 0 + err || det4x4(D2) < 0 + err || det4x4(D3) < 0 + err || det4x4(D4) < 0 + err)
      return 0;
    else
      return 1;
  }
  else {
    if(det4x4(D1) > 0 - err || det4x4(D2) > 0 - err || det4x4(D3) > 0 - err || det4x4(D4) > 0 - err)
      return 0;
    else
      return 1;
  }
}
// indexes 0
//{1,2,4,7},
//{2,4,6,7}, // opposite 1
//{1,4,5,7}, // opposite 2
//{1,2,3,7}, // opposite 4
//{0,1,2,4} 

// indexes 90
//{0,3,5,6},
//{3,5,6,7}, // opposite 0
//{0,4,5,6}, // opposite 3
//{0,2,3,6}, // opposite 5
//{0,1,3,5}  // opposite 6

// ECL : This checks if the point is within any tetrahedron. Works based on indexes 
static int pointInCube(const float point[3], const float * const cubeVerts, int orientation) {
  if(orientation == 0) {
    float t1[12] = { cubeVerts[ 3], cubeVerts[ 4], cubeVerts[ 5],
                     cubeVerts[ 6], cubeVerts[ 7], cubeVerts[ 8],
                     cubeVerts[12], cubeVerts[13], cubeVerts[14],
                     cubeVerts[21], cubeVerts[22], cubeVerts[23] };

    float t2[12] = { cubeVerts[ 6], cubeVerts[ 7], cubeVerts[ 8],
                     cubeVerts[12], cubeVerts[13], cubeVerts[14],
                     cubeVerts[18], cubeVerts[19], cubeVerts[20],
                     cubeVerts[21], cubeVerts[22], cubeVerts[23] };

    float t3[12] = { cubeVerts[ 3], cubeVerts[ 4], cubeVerts[ 5],
                     cubeVerts[12], cubeVerts[13], cubeVerts[14],
                     cubeVerts[15], cubeVerts[16], cubeVerts[17],
                     cubeVerts[21], cubeVerts[22], cubeVerts[23] };

    float t4[12] = { cubeVerts[ 3], cubeVerts[ 4], cubeVerts[ 5],
                     cubeVerts[ 6], cubeVerts[ 7], cubeVerts[ 8],
                     cubeVerts[ 9], cubeVerts[10], cubeVerts[11],
                     cubeVerts[21], cubeVerts[22], cubeVerts[23] };

    float t5[12] = { cubeVerts[ 0], cubeVerts[ 1], cubeVerts[ 2],
                     cubeVerts[ 3], cubeVerts[ 4], cubeVerts[ 5],
                     cubeVerts[ 6], cubeVerts[ 7], cubeVerts[ 8],
                     cubeVerts[12], cubeVerts[13], cubeVerts[14] };
  return pointInTet(point, t1)
       + pointInTet(point, t2)
       + pointInTet(point, t3)
       + pointInTet(point, t4)
       + pointInTet(point, t5);

  } else {
    float t1[12] = { cubeVerts[ 0], cubeVerts[ 1], cubeVerts[ 2],
                     cubeVerts[ 9], cubeVerts[10], cubeVerts[11],
                     cubeVerts[15], cubeVerts[16], cubeVerts[17],
                     cubeVerts[18], cubeVerts[19], cubeVerts[20] };

    float t2[12] = { cubeVerts[ 9], cubeVerts[10], cubeVerts[11],
                     cubeVerts[15], cubeVerts[16], cubeVerts[17],
                     cubeVerts[18], cubeVerts[19], cubeVerts[20],
                     cubeVerts[21], cubeVerts[22], cubeVerts[23] };

    float t3[12] = { cubeVerts[ 0], cubeVerts[ 1], cubeVerts[ 2],
                     cubeVerts[12], cubeVerts[13], cubeVerts[14],
                     cubeVerts[15], cubeVerts[16], cubeVerts[17],
                     cubeVerts[18], cubeVerts[19], cubeVerts[20] };

    float t4[12] = { cubeVerts[ 0], cubeVerts[ 1], cubeVerts[ 2],
                     cubeVerts[ 6], cubeVerts[ 7], cubeVerts[ 8],
                     cubeVerts[ 9], cubeVerts[10], cubeVerts[11],
                     cubeVerts[18], cubeVerts[19], cubeVerts[20] };

    float t5[12] = { cubeVerts[ 0], cubeVerts[ 1], cubeVerts[ 2],
                     cubeVerts[ 3], cubeVerts[ 4], cubeVerts[ 5],
                     cubeVerts[ 9], cubeVerts[10], cubeVerts[11],
                     cubeVerts[15], cubeVerts[16], cubeVerts[17] };
    return pointInTet(point, t1) 
         + pointInTet(point, t2)
         + pointInTet(point, t3)
         + pointInTet(point, t4)
         + pointInTet(point, t5);
  }
}

void testPIC() {
  unsigned i;
  float cube[24] = { 0.0f, 0.0f, 0.0f, 
                     2.0f, 0.0f, 0.0f,
                     0.0f, 2.0f, 0.0f, 
                     2.0f, 2.0f, 0.0f,
                     0.0f, 0.0f, 2.0f, 
                     2.0f, 0.0f, 2.0f,
                     0.0f, 2.0f, 2.0f, 
                     2.0f, 2.0f, 2.0f};
  for(i = 0; i < 1000000; i++) {
    float point[3] = {((float)rand()/(float)(RAND_MAX)) * 2.0f,
                      ((float)rand()/(float)(RAND_MAX)) * 2.0f,
                      ((float)rand()/(float)(RAND_MAX)) * 2.0f };
    int check = pointInCube(point, cube, 0);
    if(check == 0)
      printf("ERROROROROROROROR\n");
  }
}

int getTransformAt(tile_t inputTile, 
                   unsigned cornerID, 
                   const int64_t const * inCoords_nm, 
                   float *outTransform, 
                   float *outCoords) {

  float um2nm = 1000;
  int64_t * coords = TileCoordinates(inputTile);
  unsigned numCoords = TileSizeXlims(inputTile) * TileSizeYlims(inputTile) * TileSizeZlims(inputTile);
  
  if(cornerID < 0 || cornerID > numCoords) {
    //printf("DEBUG: getTransformAt---Quit because cornerID was less than 0 or greater than numCoords\n");
    return 0;
  }
 
  unsigned xID = cornerID % (TileSizeXlims(inputTile));
  unsigned yID = (cornerID/TileSizeXlims(inputTile)) % (TileSizeYlims(inputTile));
  unsigned zID = (cornerID/(TileSizeXlims(inputTile)*TileSizeYlims(inputTile))) % (TileSizeZlims(inputTile));

  // ECL : ^SIGN^: Bounds checking
  if(xID <= 0 | yID <= 0 | zID >= TileSizeZlims(inputTile) - 1){
    //printf("DEBUG: getTransformAt---Quit becausse xID, yID, or zID was bad\n");
    return 0;
  }
  
  unsigned xDist = 3;
  unsigned yDist = TileSizeXlims(inputTile)*xDist;
  unsigned zDist = TileSizeYlims(inputTile)*yDist;
  
  cornerID*=3;
  // ECL : Get the corners of the subvolume in Morton order. 000 means all minimal values, 001 means maximal x minimal
  //     y and z, etc.
  //
  //       ^SIZE^: The logic will need to change here
  int64_t transform[24] = 
  { coords[cornerID + 0     - 0     -     0 + 0], coords[cornerID + 0     - 0      -     0 + 1], 
    coords[cornerID + 0     - 0     -     0 + 2],  //000 

    coords[cornerID + 0     - 0     - xDist + 0], coords[cornerID + 0     - 0      - xDist + 1], 
    coords[cornerID + 0     - 0     - xDist + 2],  //001

    coords[cornerID + 0     - yDist -     0 + 0], coords[cornerID + 0     - yDist  -     0 + 1], 
    coords[cornerID + 0     - yDist -     0 + 2],  //010

    coords[cornerID + 0     - yDist - xDist + 0], coords[cornerID + 0     - yDist  - xDist + 1], 
    coords[cornerID + 0     - yDist - xDist + 2],  //011

    coords[cornerID + zDist - 0     -     0 + 0], coords[cornerID + zDist - 0      -     0 + 1], 
    coords[cornerID + zDist - 0     -     0 + 2],  //100

    coords[cornerID + zDist - 0     - xDist + 0], coords[cornerID + zDist - 0      - xDist + 1], 
    coords[cornerID + zDist - 0     - xDist + 2],  //101

    coords[cornerID + zDist - yDist -     0 + 0], coords[cornerID + zDist - yDist  -     0 + 1], 
    coords[cornerID + zDist - yDist -     0 + 2],  //110

    coords[cornerID + zDist - yDist - xDist + 0], coords[cornerID + zDist - yDist  - xDist + 1], 
    coords[cornerID + zDist - yDist - xDist + 2] };//111 
  
  // ECL : Grab the origin point of the subvolume
  int64_t ori[3] = {transform[0], transform[1], transform[2]};

  // ECL : Get the transform in relative space. outTransform now contains cubeVerts.
  unsigned i;
  for(i = 0; i < 8; i++) {
    outTransform[i * 3 + 0] = (float)(transform[i * 3 + 0] - ori[0]);
    outTransform[i * 3 + 1] = (float)(transform[i * 3 + 1] - ori[1]);
    outTransform[i * 3 + 2] = (float)(transform[i * 3 + 2] - ori[2]); 
  }
  
  // ECL : Do the same here for input coordinate
  outCoords[0] = (float)(inCoords_nm[0] - ori[0]);
  outCoords[1] = (float)(inCoords_nm[1] - ori[1]);
  outCoords[2] = (float)(inCoords_nm[2] - ori[2]);
  return 1;
}

int checkShell(tile_t inputTile, 
               unsigned cornerID, 
               const int64_t const * inCoords_nm, 
               float *outTransform, 
               float *outCoords, 
               unsigned *cornerIDOut) {

  unsigned numCoords = TileSizeXlims(inputTile) * TileSizeYlims(inputTile) * TileSizeZlims(inputTile);

  // ECL : If checkShell was passed in a bad value, quit
  if(cornerID < 0 || cornerID > numCoords) {
    return 0;
  }

  unsigned xID = cornerID % (TileSizeXlims(inputTile));
  unsigned yID = (cornerID/TileSizeXlims(inputTile)) % (TileSizeYlims(inputTile));
  unsigned zID = (cornerID/(TileSizeXlims(inputTile)*TileSizeYlims(inputTile))) % (TileSizeZlims(inputTile));
  int orientation;
  
  if((xID + yID + zID) % 2 == 0) 
    orientation = 90;
  else 
    orientation = 0;
  
  // ECL : This is currently hardcoded checking of each surrounding subvolume ... this surely can be more clever
   
  //       ^SIGN^: the *MinorDist's will need to be changed.
  int xMinor = (xID > 0); // ECL : Are we on the lower boundary of xlims?
  int xMinorDist = 1;     // ECL : Value to add to cornerID to get lower x value

  int yMinor = (yID > 0);
  int yMinorDist = TileSizeXlims(inputTile);

  int zMinor = (zID > 0);
  int zMinorDist = 0 - (TileSizeXlims(inputTile)*TileSizeYlims(inputTile));


  int xMajor = (xID < TileSizeXlims(inputTile)); // ECL : Upper boundary
  int xMajorDist = 0 - xMinorDist;

  int yMajor = (yID < TileSizeYlims(inputTile));
  int yMajorDist = 0 - yMinorDist;

  int zMajor = (zID < TileSizeZlims(inputTile));
  int zMajorDist = 0 - zMinorDist;
  if(xMinor) { 
    getTransformAt(inputTile, cornerID + xMinorDist, inCoords_nm, outTransform, outCoords);
    if(pointInCube(outCoords, outTransform, orientation) > 0) {
      *cornerIDOut = cornerID + xMinorDist;
      return 1;
    }
  }
  if(xMinor && yMinor) {
    getTransformAt(inputTile, cornerID + xMinorDist + yMinorDist, inCoords_nm, outTransform, outCoords);
    if(pointInCube(outCoords, outTransform, orientation) > 0) {
      *cornerIDOut =  cornerID + xMinorDist + yMinorDist;
      return 1;
    }
  }
  if(xMinor && yMajor) {
    getTransformAt(inputTile, cornerID + xMinorDist + yMajorDist, inCoords_nm, outTransform, outCoords);
    if(pointInCube(outCoords, outTransform, orientation) > 0) {
      *cornerIDOut =  cornerID + xMinorDist + yMajorDist;
      return 1;
    }
  }
  if(xMinor && zMinor) {
    getTransformAt(inputTile, cornerID + xMinorDist + zMinorDist, inCoords_nm, outTransform, outCoords);
    if(pointInCube(outCoords, outTransform, orientation) > 0) {
      *cornerIDOut =  cornerID + xMinorDist + zMinorDist;
      return 1;
    }
  }
  if(xMinor && zMajor) {
    getTransformAt(inputTile, cornerID + xMinorDist + zMajorDist, inCoords_nm, outTransform, outCoords);
    if(pointInCube(outCoords, outTransform, orientation) > 0) {
      *cornerIDOut =  cornerID + xMinorDist + zMajorDist;
      return 1;
    }
  }
  if(xMinor && yMinor && zMinor) {
    getTransformAt(inputTile, cornerID + xMinorDist + yMajorDist + zMinorDist, inCoords_nm, outTransform, outCoords);
    if(pointInCube(outCoords, outTransform, orientation) > 0) {
      *cornerIDOut =  cornerID + xMinorDist + yMajorDist + zMinorDist;
      return 1;
    }
  }
  if(xMinor && yMinor && zMajor) {
    getTransformAt(inputTile, cornerID + xMinorDist + yMajorDist + zMajorDist, inCoords_nm, outTransform, outCoords);
    if(pointInCube(outCoords, outTransform, orientation) > 0) {
      *cornerIDOut =  cornerID + xMinorDist + yMajorDist + zMajorDist;
      return 1;
    }
  }
  if(xMinor && yMajor && zMinor) {
    getTransformAt(inputTile, cornerID + xMinorDist + yMajorDist + zMinorDist, inCoords_nm, outTransform, outCoords);
    if(pointInCube(outCoords, outTransform, orientation) > 0) {
      *cornerIDOut =  cornerID + xMinorDist + yMajorDist + zMinorDist;
      return 1;
    }
  }
  if(xMinor && yMajor && zMajor) {
    getTransformAt(inputTile, cornerID + xMinorDist + yMajorDist + zMajorDist, inCoords_nm, outTransform, outCoords);
    if(pointInCube(outCoords, outTransform, orientation) > 0) {
      *cornerIDOut =  cornerID + xMinorDist + yMajorDist + zMajorDist;
      return 1;
    }
  }
  if(xMajor) {
    getTransformAt(inputTile, cornerID + xMajorDist, inCoords_nm, outTransform, outCoords);
    if(pointInCube(outCoords, outTransform, orientation) > 0) {
      *cornerIDOut =  cornerID + xMajorDist;
      return 1;
    }
  }
  if(xMajor && yMinor) {
    getTransformAt(inputTile, cornerID + xMajorDist + yMinorDist, inCoords_nm, outTransform, outCoords);
    if(pointInCube(outCoords, outTransform, orientation) > 0) {
      *cornerIDOut =  cornerID + xMajorDist + yMinorDist;
      return 1;
    }
  }
  if(xMajor && yMajor) {
    getTransformAt(inputTile, cornerID + xMajorDist + yMajorDist, inCoords_nm, outTransform, outCoords);
    if(pointInCube(outCoords, outTransform, orientation) > 0) {
      *cornerIDOut =  cornerID + xMajorDist + yMajorDist;
      return 1;
    }
  }
  if(xMajor && zMinor) {
    getTransformAt(inputTile, cornerID + xMajorDist + zMinorDist, inCoords_nm, outTransform, outCoords);
    if(pointInCube(outCoords, outTransform, orientation) > 0) {
      *cornerIDOut =  cornerID + xMajorDist + zMinorDist;
      return 1;
    }
  }
  if(xMajor && zMajor) {
    getTransformAt(inputTile, cornerID + xMajorDist + zMajorDist, inCoords_nm, outTransform, outCoords);
    if(pointInCube(outCoords, outTransform, orientation) > 0) {
      *cornerIDOut =  cornerID + xMajorDist + zMajorDist;
      return 1;
    }
  }
  if(xMajor && yMinor && zMinor) {
    getTransformAt(inputTile, cornerID + xMajorDist + yMajorDist + zMinorDist, inCoords_nm, outTransform, outCoords);
    if(pointInCube(outCoords, outTransform, orientation) > 0) {
      *cornerIDOut =  cornerID + xMajorDist + yMajorDist + zMinorDist;
      return 1;
    }
  }
  if(xMajor && yMinor && zMajor) {
    getTransformAt(inputTile, cornerID + xMajorDist + yMajorDist + zMajorDist, inCoords_nm, outTransform, outCoords);
    if(pointInCube(outCoords, outTransform, orientation) > 0) {
      *cornerIDOut =  cornerID + xMajorDist + yMajorDist + zMajorDist;
      return 1;
    }
  }
  if(xMajor && yMajor && zMinor) {
    getTransformAt(inputTile, cornerID + xMajorDist + yMajorDist + zMinorDist, inCoords_nm, outTransform, outCoords);
    if(pointInCube(outCoords, outTransform, orientation) > 0) {
      *cornerIDOut =  cornerID + xMajorDist + yMajorDist + zMinorDist;
      return 1;
    }
  }
  if(xMajor && yMajor && zMajor) {
    getTransformAt(inputTile, cornerID + xMajorDist + yMajorDist + zMajorDist, inCoords_nm, outTransform, outCoords);
    if(pointInCube(outCoords, outTransform, orientation) > 0) {
      *cornerIDOut =  cornerID + xMajorDist + yMajorDist + zMajorDist;
      return 1;
    }
  }
  
  if(yMinor) {
    getTransformAt(inputTile, cornerID + yMinorDist, inCoords_nm, outTransform, outCoords);
    if(pointInCube(outCoords, outTransform, orientation) > 0) {
      *cornerIDOut =  cornerID + yMinorDist;
      return 1;
    }
  }
  if(yMinor & zMinor) {
    getTransformAt(inputTile, cornerID + yMinorDist + zMinorDist, inCoords_nm, outTransform, outCoords);
    if(pointInCube(outCoords, outTransform, orientation) > 0) {
      *cornerIDOut =  cornerID + yMinorDist + zMinorDist;
      return 1;
    }
  }
  if(yMinor & zMajor) {
    getTransformAt(inputTile, cornerID + yMinorDist + zMajorDist, inCoords_nm, outTransform, outCoords);
    if(pointInCube(outCoords, outTransform, orientation) > 0) {
      *cornerIDOut =  cornerID + yMinorDist + zMajorDist;
      return 1;
    }
  }
  
  if(yMajor) {
    getTransformAt(inputTile, cornerID + yMajorDist, inCoords_nm, outTransform, outCoords);
    if(pointInCube(outCoords, outTransform, orientation) > 0) {
      *cornerIDOut =  cornerID + yMajorDist;
      return 1;
    }
  }
  if(yMajor & zMinor) {
    getTransformAt(inputTile, cornerID + yMajorDist + zMinorDist, inCoords_nm, outTransform, outCoords);
    if(pointInCube(outCoords, outTransform, orientation) > 0) {
      *cornerIDOut =  cornerID + yMajorDist + zMinorDist;
      return 1;
    }
  }
  if(yMajor & zMajor) {
    getTransformAt(inputTile, cornerID + yMajorDist + zMajorDist, inCoords_nm, outTransform, outCoords);
    if(pointInCube(outCoords, outTransform, orientation) > 0) {
      *cornerIDOut =  cornerID + yMajorDist + zMajorDist;
      return 1;
    }
  }
  if(zMinor) {
    getTransformAt(inputTile, cornerID + zMinorDist, inCoords_nm, outTransform, outCoords);
    if(pointInCube(outCoords, outTransform, orientation) > 0) {
      *cornerIDOut =  cornerID + zMinorDist;
      return 1;
    }
  }
  if(zMajor) {
    getTransformAt(inputTile, cornerID + zMajorDist, inCoords_nm, outTransform, outCoords);
    if(pointInCube(outCoords, outTransform, orientation) > 0) {
      *cornerIDOut =  cornerID + zMajorDist;
      return 1;
    }
  }
  return 0;
}



int computePixel(float out[3], tile_t inputTile, const float inCoordinates[3]) {

  float um2nm = 1000;
  float transform[24], inCoords[3];
  int orientation;
  int64_t inCoords_nm[3] = {(int64_t)(inCoordinates[0] * um2nm), 
                            (int64_t)(inCoordinates[1] * um2nm), 
                            (int64_t)(inCoordinates[2] * um2nm)};
 
  unsigned numCoords = TileSizeXlims(inputTile) * TileSizeYlims(inputTile) * TileSizeZlims(inputTile), 
                       i, 
                       cornerID = -1;
  int64_t * coords = TileCoordinates(inputTile), bestDist = LONG_MAX; 

  
  // ECL : To first guess which sub-volume the inputed location lies in, we find the nearest coordinates within the tile
  //       
  //       Note: At this point, cornerID points to an xyz coordinate. To get an actual position in coordinates,
  //     we will need to first multiply cornerID by 3
  for(i = 0; i < numCoords; i++) {
    if(dist2(inCoords_nm[0], inCoords_nm[1], inCoords_nm[2], coords[i*3], coords[i*3+1], coords[i*3+2]) < bestDist ) {
      cornerID = i;
      bestDist = dist2(inCoords_nm[0], inCoords_nm[1], inCoords_nm[2], coords[i*3], coords[i*3+1], coords[i*3+2]);
    }
  }
  unsigned cornerIDNew = cornerID;
  // ECL : I have left in this debug to show how to print int64_t. It's horrible.
  if(!quiet) 
    fprintf(logFile, "\tnear coord = %" PRId64 " %" PRId64 " %" PRId64 "\n", coords[cornerID*3+0], coords[cornerID*3+1], coords[cornerID*3+2]);
  fflush(logFile); 
  // ECL : Because we want the transform to start with the minimal xyz values, we find the Morton corner 
  //     from the guessed corner.
  //
  //       Note: this is the first place where we work off of the assumption that x and y coordinates decrease with 
  //     each step while z coordinates increase. If weird errors are occuring on a yaml file, check if that 
  //     is the case. If it is not, then signs will need to change. I will mark these areas with ^SIGN^ for 
  //     easier searching. I check this assumption first, and print a warning to the logfile if the assumption
  //     is incorrect.

  if(coords[0 + 0] < coords[3 + 0] && !quiet) 
    fprintf(logFile, "WARNING: x is increasing instead of decreasing\n");
  fflush(logFile);

  if(coords[0 + 1] < coords[TileSizeXlims(inputTile) + 1] && !quiet)
    fprintf(logFile, "WARNING: y is increasing instead of decreasing\n");
  fflush(logFile);

  if(coords[0 + 2] > coords[(TileSizeXlims(inputTile)*TileSizeYlims(inputTile)) + 2] && !quiet)
    fprintf(logFile, "WARNING: z is decreasing instead of increasing\n");
  fflush(logFile);

  if(coords[cornerID*3+0] > inCoords_nm[0]) cornerIDNew += 1; 
  if(coords[cornerID*3+1] > inCoords_nm[1]) cornerIDNew += TileSizeXlims(inputTile);
  if(coords[cornerID*3+2] > inCoords_nm[2]) cornerIDNew -= TileSizeXlims(inputTile) * TileSizeYlims(inputTile);
  // ECL : We now want to find which pixel volume our coordinate volume maps to
  unsigned xID = cornerIDNew % (TileSizeXlims(inputTile));
  unsigned yID = (cornerIDNew/TileSizeXlims(inputTile)) % (TileSizeYlims(inputTile));
  unsigned zID = (cornerIDNew/(TileSizeXlims(inputTile)*TileSizeYlims(inputTile))) % (TileSizeZlims(inputTile));
  // ECL : We check to see if we are lying on a boundry, and move to the closest subvolume if so
  //
  //       ^SIGN^: Bounds checking only checks lower bounds on x and y and upper bounds on z, and corrects
  //     based on the assumption. 
  if(xID == 0 | yID == 0 | zID == TileSizeZlims(inputTile) - 1) { 
     if(!quiet)
       fprintf(logFile, "INFO: point is in a boundary subtile\n");
       
     if(xID == 0){
       if(!quiet)
         fprintf(logFile, "INFO: Correcting xID\n");
       cornerIDNew+=1;
       xID++;
     }
     if(yID == 0) {
       if(!quiet)
         fprintf(logFile, "INFO: Correcting yID\n");
       cornerIDNew+=TileSizeXlims(inputTile);
       yID++;
     }
     if(zID == TileSizeZlims(inputTile) - 1){
       if(!quiet)
         fprintf(logFile, "INFO: Correcting zID\n");
       cornerIDNew-=TileSizeXlims(inputTile) * TileSizeYlims(inputTile);
       zID--;
     }
  }
  cornerID = cornerIDNew;
  // ECL : We grab the source shape from the xyz lims
  //
  //       ^SIGN^: Logic will need to change
  unsigned src_shape[3] = {TileXlims(inputTile)[xID    ] - TileXlims(inputTile)[xID - 1],
                           TileYlims(inputTile)[yID    ] - TileYlims(inputTile)[yID - 1],
                           TileZlims(inputTile)[zID + 1] - TileZlims(inputTile)[zID    ]}; 
  
  
  // ECL : This should not return false since we corrected for bounds above. If it fails, something weird is going on
  if(getTransformAt(inputTile, cornerID, inCoords_nm, transform, inCoords) == 0) {
    if(!quiet) 
      fprintf(logFile, "ERROR: Unexpected result from getTransformAt\n");
    return 0; 
  }
  fflush(logFile);
  // ECL : I may have this orientation incorrect, but I have not gotten issues thus far. If a bug occurs, this may be
  //     the cause. Orientation is also calculated in checkShell and the brute force below.
  orientation = ((xID + yID + zID) % 2 == 0)?90:0;
  int found = 0;
  int check = pointInCube(inCoords, transform, orientation);
  if(!quiet){
     
     fprintf(logFile, "INFO: inCoords  = %f, %f, %f\n", inCoords[0], inCoords[1], inCoords[2]);
     fprintf(logFile, "INFO: transform = \n\t%f, %f, %f\n", transform[0], transform[1], transform[2]);
     fprintf(logFile, "\t%f, %f, %f\n", transform[3], transform[4], transform[5]);
     fprintf(logFile, "\t%f, %f, %f\n", transform[6], transform[7], transform[8]);
     fprintf(logFile, "\t%f, %f, %f\n", transform[9], transform[10], transform[11]);
     fprintf(logFile, "\t%f, %f, %f\n", transform[12], transform[13], transform[14]);
     fprintf(logFile, "\t%f, %f, %f\n", transform[15], transform[16], transform[17]);
     fprintf(logFile, "\t%f, %f, %f\n", transform[18], transform[19], transform[20]);
     fprintf(logFile, "\t%f, %f, %f\n", transform[21], transform[22], transform[23]);
     fprintf(logFile, "INFO: pointInCube initially = %i\n", check);
  }

  if(!check) {  
    if(!quiet)
      fprintf(logFile, "INFO: Point was not in initial cube\n");
    if(!checkShell(inputTile, cornerID, inCoords_nm, transform, inCoords, &cornerIDNew)) {
      if(!quiet){
        fprintf(logFile, "INFO: checkShell could not find point\n");
        fprintf(logFile, "INFO: Attempting brute force search of tile\n");
      }
      fflush(logFile);
      unsigned cornerIter;
      // ECL : We were not able to find the point in out guessed subvolume or any surrounding subvolume, so we
      //     brute force checking every subvolume in the tile. This does not seem to happen often, and so
      //     performance remains reasonable.
      for(cornerIter = 0; cornerIter < numCoords; cornerIter++) {
        orientation = (cornerIter % 2 == 0)?90:0;
        if(getTransformAt(inputTile, cornerIter, inCoords_nm, transform, inCoords) == 0)
          continue;
        if(pointInCube(inCoords, transform, orientation) == 0) {
          continue;
         } else { 
            found = 1;
            cornerIDNew = cornerIter;
            break;
         }
      }
      if(found == 0)
        if(!quiet) 
          fprintf(logFile, "INFO: point not found in tile\n");
        return 0;
    } 
  }
  float relative[3];
  BarycentricCPUpixel(relative, transform, src_shape, orientation, inCoords);
  // ECL : We need to recalculate the ID values if we found the subvolume by brute force or checkShell.
  if(cornerID != cornerIDNew) {
    xID = cornerIDNew % (TileSizeXlims(inputTile));
    yID = (cornerIDNew/TileSizeXlims(inputTile)) % (TileSizeYlims(inputTile));
    zID = (cornerIDNew/(TileSizeXlims(inputTile)*TileSizeYlims(inputTile))) % (TileSizeZlims(inputTile));
  } 
  // ECL : The output pixel needs to be corrected to the specific subvolume.
  //
  //       ^SIZE^: Logic will need to change here
  out[0] = TileXlims(inputTile)[xID] - relative[0];
  out[1] = TileYlims(inputTile)[yID] - relative[1];
  out[2] = TileZlims(inputTile)[zID] + relative[2];
  return 1;
}




int main(int argc, char *argv[]) {

//=================================== ARGUMENT HANDLING ===================================
  char *inFileName=NULL, *outFileName=NULL, *logFileName=NULL, *line;
  line = malloc(1024);
  argc--;
  *argv++;
  while(argc--){
    if(**argv == '-')
    {
      char *arg = (*argv+1);
      switch(*arg) {
        case 'h':
          if(strcmp(arg, "h") == 0 || strcmp(arg, "help") == 0){
            printHelp();
            return 1;
          }
          break;
        case 'q':
          if(strcmp(arg, "q") == 0 || strcmp(arg, "quiet") == 0){
            quiet = 1;
          }
          break;

        case 'i':
          if(strcmp(strtok(arg, "="), "i") == 0 || strcmp(strtok(arg, "="), "input") == 0) 
            inFileName = strtok(NULL, "=");
          else {
            fprintf(stderr, "ERROR: Unrecognized argument -- \n%s is not a registered argument. See -help for more information\n",*argv);
            return 0;
          }
          break;
        case 'o':
          if(strcmp(strtok(arg, "="), "o") == 0 || strcmp(strtok(arg, "="), "output") == 0) 
            outFileName = strtok(NULL, "=");
          else {
            fprintf(stderr, "ERROR: Unrecognized argument -- \n%s is not a registered argument. See -help for more information\n",*argv);
            return 0;
          }
         break;    
        case 'l':
          if(strcmp(strtok(arg, "="), "l") == 0 || strcmp(strtok(arg, "="), "log") == 0) 
            logFileName = strtok(NULL, "=");
          else {
            fprintf(stderr, "ERROR: Unrecognized argument -- \n%s is not a registered argument. See -help for more information\n",*argv);
            return 0;
          }
          break;
        default:
          fprintf(stderr, "ERROR: Unrecognized argument -- \n%s is not a registered argument. See -help for more information\n",*argv);
          return 0;
          break;
      }
      *argv++;
    }
  }
  
  if(logFileName) {
    logFile = fopen(logFileName, "w");
    if(!logFile) {
      fprintf(stderr, "ERROR: Cannot open log file %s\n", logFileName);
      return 0;
    }
  } else logFile = stdout;
    
  if(!outFileName) outFileName = "output.txt";
  outFile = fopen(outFileName, "w");
  if(!outFile) {
    fprintf(stderr, "ERROR: Cannot open output file %s\n", outFileName);
    fclose(logFile);
    return 0;
  }
  
  if(!inFileName)  inFileName  = "input.txt";
  fprintf(logFile, "INFO: Reading input file \"%s\"\n", inFileName);
  inFile = fopen(inFileName, "r");
  if(!inFile) {
    fprintf(stderr, "ERROR: Cannot open input file %s\n", inFileName);
    fclose(logFile);
    fclose(outFile);
    return 0;
  }
//================================= END ARGUMENT HANDLING =================================


// ECL : Get the first line of the input file which has the path to the yml file
  if(!quiet)
    fprintf(logFile, "INFO: Reading line 1 of input\n");
  if(fgets(line, 1024, inFile) == NULL) {
    fprintf(stderr, "ERROR: Input file \"%s\" is empty\n", inFileName);
    fclose(logFile);
    fclose(outFile);
    fclose(inFile);
    return 0;
    
  }
  unsigned c = 0, k;
  while(line[c] != '\n') {
    c++;
  }
  char *path = malloc(c);
  for(k = 0; k < c; k++) {
    path[k] = line[k];
  }
  if(!quiet)
    fprintf(logFile, "INFO: Path set to \"%s\"\n", path);
 
  c = 0;
  fgets(line, 1024, inFile);
  while(line[c] != '\n') {
    c++;
  }
  char *otherFile = malloc(c);
  for(k = 0; k < c; k++) {
    otherFile[k] = line[k];
  }
  
// ECL : path now contains the path to the yml file
  

  if(!quiet)
    fprintf(logFile, "INFO: Opening \"%s/tilebase.cache.yml\"\n", path);  
  tiles_t yml = TileBaseOpen(path,NULL);
  if(!yml) {
    fprintf(stderr, "ERROR: Could not open yml file at \"%s\"\n", path);
    fclose(logFile);
    fclose(outFile);
    fclose(inFile);
  }
  else if(!quiet)
    fprintf(logFile, "INFO: Opened \"%s/tilebase.cache.yml\"\n", path);

  float inCoords_um[3];
  char *tilePath;
  tilePath = malloc(100);
  int coordNum;
  unsigned lineNumber = 3;
  
  // ECL : Formatting output to make line numbers match input
  fprintf(outFile, "\n");
  fprintf(outFile, "%s\n", otherFile);
  // ECL : Begin parsing the input file
  while(fgets(line, 1024, inFile) != NULL) {
    if(!quiet)
      fprintf(logFile, "INFO: Reading line %i of input: \n\t %s",lineNumber, line);
    // ECL : If the line was formatted incorrectly, print a warning to the output file and skip
    if(sscanf(line, "%i, %f, %f, %f, %s", &coordNum, inCoords_um, inCoords_um+1, inCoords_um+2, tilePath) != 5) { 
      if(!quiet)
        fprintf(stderr, "WARNING: Encountered incorrect format of input file at line %i\n", lineNumber);
      if(!quiet)
        fprintf(logFile, "WARNING: The line \"%s\" is not formatted correctly\n", line);
      fprintf(outFile, "WARNING: Line format error\n");
    } else {
      tile_t t = findTile(&yml, tilePath);
      if(!t) { 
      // ECL : Read the input file correctly, but provided tile path was not found
        if(!quiet)
          fprintf(stderr, "WARNING: Tile \"%s\" not found\n", tilePath); 
        if(!quiet)
          fprintf(logFile, "WARNING: Searched for tile with path \"%s,\" but no was tile found\n", tilePath);
        fprintf(outFile, "WARNING: Tile not found\n");
      } else {
        fprintf(logFile, "INFO: Tile \"%s\" found\n", tilePath);
        float tmp[3];
        fflush(outFile);
        fflush(logFile);
        // ECL : An error will orruc if a point is not actually contained within a volume
        if(computePixel(tmp, t, inCoords_um) == 1)
          fprintf(outFile, "%i, %f, %f, %f\n", coordNum, tmp[0], tmp[1], tmp[2]);
        else {
          //unsigned found = 0;
          //unsigned idx, size = TileBaseCount(yml);
          //for(idx = 0; idx < size; idx++){
          //  tile_t t2 = TileBaseIndex(yml, idx);
          //  if(computePixel(tmp, t2, inCoords_um) == 1){
          //    fprintf(outFile, "%f, %f, %f, %s\n", tmp[0], tmp[1], tmp[2], TilePath(t2));
          //    fflush(outFile);
          //    found = 1;
          //    break;
          //  }
          //}
          //if(found == 0) { 
          //  fprintf(outFile, "!!!POINT NOT FOUND IN YML FILE\n");
          //  fflush(outFile);
          //}
          if(!quiet)
            fprintf(stderr, "ERROR: Point not found in volume at line %i\n", lineNumber); 
          fprintf(outFile, "ERROR: Point mismatch\n");
          if(!quiet) {
            fprintf(logFile, "ERROR: Searched for point within selected volume but point was found");
            fprintf(logFile, " to be outside the bounds of \"%s\".\n", tilePath);
          }
        }
      }
    }
    lineNumber++;
  }
  fflush(logFile);
  free(line);
  free(path);
  free(tilePath);
  if(!quiet)
    fprintf(logFile, "INFO: Finished reading input file\n");

  if(!quiet)
    fprintf(logFile, "INFO: Closing all text files\n");
  fclose(inFile);
  fclose(outFile);
  if(!quiet)
    fprintf(logFile, "INFO: Closed all text files\n");

  if(!quiet)
    fprintf(logFile, "INFO: Closing yml file\n");
  TileBaseClose(yml); 
  if(!quiet)
    fprintf(logFile, "INFO: Closed yml file\n");
 
  fprintf(logFile, "INFO: Done\n");
  fclose(logFile);

  return 1;
}
