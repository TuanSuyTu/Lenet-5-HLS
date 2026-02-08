#include "Conv.h"
#include "Pool.h"
#include "Dense.h"
#include <algorithm>
#include <string.h>
#include <ap_axi_sdata.h>
typedef ap_fixed<32, 16> fxp;
void CNN(fxp InModel[784], fxp &OutModel0, fxp Weights[5738])
{
#pragma HLS INTERFACE m_axi port = InModel offset = slave bundle = gmem0 depth = 784
#pragma HLS INTERFACE m_axi port = Weights offset = slave bundle = gmem1 depth = 5738
#pragma HLS INTERFACE s_axilite port = OutModel0 bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control
	fxp conv2d[2304];
	fxp max_pooling2d[576];
	fxp conv2d_1[512];
	fxp max_pooling2d_1[128];
	fxp flatten[128];
	fxp dense[32];
	fxp dense_1[16];
	Conv2D_0(&InModel[0], conv2d, &Weights[100], &Weights[0]);
	Max_Pool2D_0(conv2d, max_pooling2d);
	Conv2D_1(max_pooling2d, conv2d_1, &Weights[904], &Weights[104]);
	Max_Pool2D_1(conv2d_1, max_pooling2d_1);
	flatten0(max_pooling2d_1, flatten);
	Dense_0(flatten, dense, &Weights[5008], &Weights[912]);
	Dense_1(dense, dense_1, &Weights[5552], &Weights[5040]);
	Dense_2(dense_1, OutModel0, &Weights[5728], &Weights[5568]);
}
