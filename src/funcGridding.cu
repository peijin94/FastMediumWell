/**
*    @file funcGridding.cu
*    @author Peijin Zhang
*    The kernel of gridding function
*    reference: Holger Rapp 2009-03-11.
*/

#include <math.h>
#include <stdio.h>
// #include "helper_math.h"  /// Math API

#include "funcGridding.cuh"
#include "cuda.h"
#include "cufft.h"


#define WIDTH 6
#define NCGF 12
#define HWIDTH 3
#define STEP 4

__device__ __constant__ float cgf[32];

extern "C" {
    __global__ void SimpleGridding(float2 *Grd, float2 *bm, float2 *sf, int *cnt, float *d_u, float *d_v, float *d_re,
        float *d_im, int nu, float du, int gcount, int umax, int vmax){
        // gridding function
        // Grd: gridded data output
        // bm:  beam in the gridding, (fft of the dirty beam)
        // sf:  binary beam in the gridding
        // cnt: counter of the gridding
        // d_u: u coordinate of the visibility
        // d_v: v coordinate of the visibility
        // d_re: real part of the visibility
        // d_im: imaginary part of the visibility
        // nu:  number of u pixels
        // du:  size of u pixel
        // gcount: number of visibilities
        // umax:  maximum u pixel
        // vmax:  maximum v pixel

      int iu = blockDim.x*blockIdx.x + threadIdx.x;
      int iv = blockDim.y*blockIdx.y + threadIdx.y;
      int u0 = 0.5*nu;
      if(iu >= u0 && iu <= u0+umax && iv <= u0+vmax){
        for (int ivis = 0; ivis < gcount; ivis++){
          float mu = d_u[ivis];
          float mv = d_v[ivis];
          int hflag = 1;
          if (mu < 0){
            hflag = -1;
            mu = -1*mu;
            mv = -1*mv;
          }
          float uu, vv;
          uu = mu/du + u0;
          vv = mv/du + u0;
          int cnu=abs(iu-uu),cnv=abs(iv-vv);
          int ind = iv*nu+iu;
          if (cnu < HWIDTH && cnv < HWIDTH){
            float wgt = cgf[int(round(4.6*cnu+NCGF-0.5))]*cgf[int(round(4.6*cnv+NCGF-0.5))];
            Grd[ind].x +=       wgt*d_re[ivis];
            Grd[ind].y += hflag*wgt*d_im[ivis];
            cnt[ind]   += 1;
            bm [ind].x += wgt;
            sf[ind].x   = 1;
            sf[ind].y   = 1;
           }
          // deal with points&pixels close to u=0 boundary
          if (iu-u0 < HWIDTH && mu/du < HWIDTH) {
            mu = -1*mu;
            mv = -1*mv;
            uu = mu/du+u0;
            vv = mv/du+u0;
            cnu=abs(iu-uu),cnv=abs(iv-vv);
            if (cnu < HWIDTH && cnv < HWIDTH){
              float wgt = cgf[int(round(4.6*cnu+NCGF-0.5))]*cgf[int(round(4.6*cnv+NCGF-0.5))];
              Grd[ind].x +=          wgt*d_re[ivis];
              Grd[ind].y += -1*hflag*wgt*d_im[ivis];
              cnt[ind]   += 1;
              bm [ind].x += wgt;
              sf[ind].x   = 1;
              sf[ind].y   = 1;            }
          }
        }
      }
    }
}