#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
void Max_Pool2D_0(fxp input_MaxPooling[2304], fxp output_MaxPooling[576]){
	int PoolSize = 2;
	int stride = 2;
	int index = 0;
	for (int i = 0; i < 4; i++){
		index = 0;
		for (int z = 0; z < 12; z++){
			for (int y = 0; y < 12; y++){
				fxp max_c = -10;
				for (int h = 0; h < PoolSize; h++){
					for (int w = 0; w < PoolSize; w++){
						int Pool_index = 24 * 24 * i + 24 * h + 24 * stride * z + w + y * stride;
						fxp Pool_value = input_MaxPooling[Pool_index];
						if (Pool_value >= max_c) max_c = Pool_value;
					}
				}
				int outIndex = 12 * 12 * i + index;
				output_MaxPooling[outIndex] = max_c;
				index++;
			}
		}
	}
}
void Max_Pool2D_1(fxp input_MaxPooling[512], fxp output_MaxPooling[128]){
	int PoolSize = 2;
	int stride = 2;
	int index = 0;
	for (int i = 0; i < 8; i++){
		index = 0;
		for (int z = 0; z < 4; z++){
			for (int y = 0; y < 4; y++){
				fxp max_c = -10;
				for (int h = 0; h < PoolSize; h++){
					for (int w = 0; w < PoolSize; w++){
						int Pool_index = 8 * 8 * i + 8 * h + 8 * stride * z + w + y * stride;
						fxp Pool_value = input_MaxPooling[Pool_index];
						if (Pool_value >= max_c) max_c = Pool_value;
					}
				}
				int outIndex = 4 * 4 * i + index;
				output_MaxPooling[outIndex] = max_c;
				index++;
			}
		}
	}
}
void flatten0(fxp input_Flatten[128],fxp output_Flatten[128]){
	int hs = 0;
	for (int i = 0; i < 4; i++){
		for (int j = 0; j < 4; j++){
			for (int k = 0; k < 8; k++){
				output_Flatten[hs] = input_Flatten[4 * i + 4 * 4 * k + j ];
				hs++;
			}
		}
	}
}
