#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
void Conv2D_0(fxp Input_Conv[784],fxp Output_Conv[2304], fxp bias[4], fxp kernel[100]){
	int stride = 1;
	loop_for_channel2D_0:
	for (int n = 0; n < 4; n++){
		loop_for_bp2D_0:
		for (int x = 0; x < 24; x++){
			loop_for_ap2D_0:
			for (int y = 0; y < 24; y++){
				fxp s = 0;
				loop_for_fc_0:
				for (int k = 0; k < 1; k++){
					loop_for_fb_0:
					for (int i = 0; i < 5; i++){
						loop_for_fa_0:
						for (int j = 0; j < 5; j++){
							s=s+(kernel[1*5*5*n+5*5*k+5*i+j])*(Input_Conv[28*28*k+28*(i+x*stride)+j+y*stride]);}
					}
				}
				if ((s+bias[n])<0) Output_Conv[24*24*n+24*x+y]=0; else Output_Conv[24*24*n+24*x+y]=s+bias[n];
			}
		}
	}
}
void Conv2D_1(fxp Input_Conv[576],fxp Output_Conv[512], fxp bias[8], fxp kernel[800]){
	int stride = 1;
	loop_for_channel2D_1:
	for (int n = 0; n < 8; n++){
		loop_for_bp2D_1:
		for (int x = 0; x < 8; x++){
			loop_for_ap2D_1:
			for (int y = 0; y < 8; y++){
				fxp s = 0;
				loop_for_fc_1:
				for (int k = 0; k < 4; k++){
					loop_for_fb_1:
					for (int i = 0; i < 5; i++){
						loop_for_fa_1:
						for (int j = 0; j < 5; j++){
							s=s+(kernel[4*5*5*n+5*5*k+5*i+j])*(Input_Conv[12*12*k+12*(i+x*stride)+j+y*stride]);}
					}
				}
				if ((s+bias[n])<0) Output_Conv[8*8*n+8*x+y]=0; else Output_Conv[8*8*n+8*x+y]=s+bias[n];
			}
		}
	}
}
