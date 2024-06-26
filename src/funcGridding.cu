/**
 *    @file funcGridding.cu
 *    @author Peijin Zhang
 *    The kernel of gridding function
 *    reference: Holger Rapp 2009-03-11. 
 * (https://github.com/astroumd/miriad/blob/master/src/subs/mapper.for)
 * (https://github.com/astroumd/miriad/blob/master/src/subs/grid.for
 */

 #include <math.h>
 #include <stdio.h>
 #include "funcGridding.cuh"
 #include "cuda.h"
 
 #include "cufft.h"
 #include "cuComplex.h"
 
// cgf: convolutional gridding function

 extern "C" {
	
   __global__ void SimpleGridding(float2 * Grd, float2 * bm, \
	 float2 * sf, int * cnt, float * d_u, float * d_v, float * d_re,
	 float * d_im, float * cgf, int WIDTH, int HWIDTH, int NCGF, 
	 int N_u, float du, int gcount, int umax, int vmax, 
	 int batch_size_img) {
	 
		// gridding function
	 // Grd: gridded data output
	 // bm:  beam in the gridding, (fft of the dirty beam)
	 // sf:  binary beam in the gridding
	 // cnt: counter of the gridding
	 // d_u: u coordinate of the visibility
	 // d_v: v coordinate of the visibility
	 // d_re: real part of the visibility
	 // d_im: imaginary part of the visibility
	 // N_u:  Number of u 
	 // du:  size of u pixel
	 // gcount: Number of visibilities
	 // umax:  maximum u pixel
	 // vmax:  maximum v pixel
 
	 int iu = blockDim.x * blockIdx.x + threadIdx.x;
	 int iv_block = threadIdx.y; 
	 int u0 = 0.5 * N_u; // center of u and v
	 int iv;

	 for (iv=iv_block*batch_size_img; iv<(iv_block+1)*batch_size_img; iv++) {
		if (iu >= u0 && iu <= u0 + umax && iv <= u0 + vmax) {
			// consider u>0
			for (int ivis = 0; ivis < gcount; ivis++) {
			float mu = d_u[ivis];
			float mv = d_v[ivis];
			int hflag = 1;
			if (mu < 0) { // for u<0, do conjugate
				hflag = -1;
				mu = -1 * mu;
				mv = -1 * mv;
			}
			float uu, vv; // u, v in pixel space, but still decimal
			uu = mu / du + u0;
			vv = mv / du + u0;

			int cnu = abs(iu - uu), cnv = abs(iv - vv); 
			// distance of the pixel to the visibility
			
			if (cnu < HWIDTH && cnv < HWIDTH) {
				int ind = iv * N_u + iu;
				float wgt = cgf[int(round((NCGF-1.0)/WIDTH * cnu + (NCGF-1.0)/2))] *\
								cgf[int(round((NCGF-1.0)/WIDTH * cnv + (NCGF-1.0)/2))];
				Grd[ind].x += wgt * d_re[ivis];
				Grd[ind].y += hflag * wgt * d_im[ivis];
				cnt[ind] += 1;
				bm[ind].x += wgt;
				sf[ind].x = 1;
				sf[ind].y = 1;
			}
			
			// deal with points&pixels close to u=0 boundary
			if (iu - u0 < HWIDTH && mu / du < HWIDTH) {
				int ind = iv * N_u + iu;
				mu = -1 * mu;
				mv = -1 * mv;
				uu = mu / du + u0;
				vv = mv / du + u0;
				cnu = abs(iu - uu), cnv = abs(iv - vv);
				if (cnu < HWIDTH && cnv < HWIDTH) {
				float wgt = cgf[int(round(4.6 * cnu + NCGF - 0.5))] * cgf[int(round(4.6 * cnv + NCGF - 0.5))];
				Grd[ind].x += wgt * d_re[ivis];
				Grd[ind].y += -1 * hflag * wgt * d_im[ivis];
				cnt[ind] += 1;
				bm[ind].x += wgt;
				sf[ind].x = 1;
				sf[ind].y = 1;
				}
			}
			}
		}
   }
	}


	__global__ void SimpleGriddingBatch(float2 * Grd3d, float * flag_2d, float2 * bm, 
		float2 * sf, int * cnt, float * d_u, float * d_v, float * d_re,
		float * d_im, float * cgf, int WIDTH, int HWIDTH, int NCGF, 
		int N_u, float du, int gcount, int gcount_total, int umax, int vmax, 
		int batch_size_img, int batch_size_vis) {
		
		// gridding function, for folded visibilities (for example LWA fastvis)
		// Grd: gridded data output
		// bm:  beam in the gridding, (fft of the dirty beam)
		// sf:  binary beam in the gridding
		// cnt: counter of the gridding
		// d_u: u coordinate of the visibility
		// d_v: v coordinate of the visibility
		// d_re: real part of the visibility
		// d_im: imaginary part of the visibility
		// N_u:  Number of u 
		// du:  size of u pixel
		// gcount: Number of visibilities
		// umax:  maximum u pixel
		// vmax:  maximum v pixel
	
		int iu = blockDim.x * blockIdx.x + threadIdx.x;
		int iv_block = threadIdx.y; 
		int u0 = 0.5 * N_u; // center of u and v
		int iv;
 
		for (iv=iv_block*batch_size_img; iv<(iv_block+1)*batch_size_img; iv++) {
		 if (iu >= u0 && iu <= u0 + umax && iv <= u0 + vmax) {
			 // consider u>0
			 for (int ivis = 0; ivis < gcount; ivis++) {
			 float mu = d_u[ivis];
			 float mv = d_v[ivis];
			 int hflag = 1;
			 if (mu < 0) { // for u<0, do conjugate
				 hflag = -1;
				 mu = -1 * mu;
				 mv = -1 * mv;
			 }
			 float uu, vv; // u, v in pixel space, but still decimal
			 uu = mu / du + u0;
			 vv = mv / du + u0;
 
			 int cnu = abs(iu - uu), cnv = abs(iv - vv); 
			 // distance of the pixel to the visibility
			 
			 if (cnu < HWIDTH && cnv < HWIDTH) {
				 int ind = iv * N_u + iu;
				 float wgt = cgf[int(round((NCGF-1.0)/WIDTH * cnu + (NCGF-1.0)/2))] *\
								 cgf[int(round((NCGF-1.0)/WIDTH * cnv + (NCGF-1.0)/2))];
				 for (int i = 0; i < batch_size_vis; i++) {
				 	Grd3d[ind+i*N_u*N_u].x += wgt * d_re[ivis+i*gcount] * flag_2d[ivis+i*gcount];
				 	Grd3d[ind+i*N_u*N_u].y += hflag * wgt * d_im[ivis+i*gcount] * flag_2d[ivis+i*gcount];
				 }
				 cnt[ind] += 1;
				 bm[ind].x += wgt;
				 sf[ind].x = 1;
				 sf[ind].y = 1;
			 }
			 
			 // deal with points&pixels close to u=0 boundary
			 if (iu - u0 < HWIDTH && mu / du < HWIDTH) {
				 int ind = iv * N_u + iu;
				 mu = -1 * mu;
				 mv = -1 * mv;
				 uu = mu / du + u0;
				 vv = mv / du + u0;
				 cnu = abs(iu - uu), cnv = abs(iv - vv);
				 if (cnu < HWIDTH && cnv < HWIDTH) {
				 float wgt = cgf[int(round((NCGF-1.0)/WIDTH * cnu + (NCGF-1.0)/2))] * cgf[int(round((NCGF-1.0)/WIDTH * cnu + (NCGF-1.0)/2))];
				for (int i = 0; i < batch_size_vis; i++) {
				 	Grd3d[ind+i*N_u*N_u].x += 			   wgt * d_re[ivis+i*gcount] * flag_2d[ivis+i*gcount];
				 	Grd3d[ind+i*N_u*N_u].y += -1 * hflag * wgt * d_im[ivis+i*gcount] * flag_2d[ivis+i*gcount];
				}
				 cnt[ind] += 1;
				 bm[ind].x += wgt;
				 sf[ind].x = 1;
				 sf[ind].y = 1;
				 }
			 }
			 }
		 }
		}
	 }



}