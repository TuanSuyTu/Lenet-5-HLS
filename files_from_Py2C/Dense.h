#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
void Dense_0(fxp input_Dense[128],fxp output_Dense[32],fxp bias[32],fxp weight[4096]);
void Dense_1(fxp input_Dense[32],fxp output_Dense[16],fxp bias[16],fxp weight[512]);
void Dense_2(fxp input_Dense[16],fxp &output_Dense_0,fxp bias[10],fxp weight[160]);
