#ifndef DOUBLE4_H
#define DOUBLE4_H

struct double4 {
  double x, y, z, w;
};

struct double3 {
  double x, y, z;
};

typedef float2 DS;

struct DS4 {
  DS x, y, z, w;
  DS4 operator=(DS4 val) {
    x = val.x;
    y = val.y;
    z = val.z;
    w = val.w;
    return val;
  }
};

struct DS2 {
  DS x, y;
  DS2 operator=(DS2 val) {
    x = val.x;
    y = val.y;
    return val;
  }
};

inline DS to_DS(double a){
  DS b;
  b.x = (float)a;
  b.y = (float)(a - b.x);
  return b;
}

inline double to_double(DS a) {
  double b;
  b = (double)((double)a.x + (double)a.y);
  return b;
}

typedef struct _vel
{
  double u, v, w;
} Vel;


#endif
