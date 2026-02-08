#include <ap_axi_sdata.h>
typedef ap_fixed<32,16> fxp;
void Conv2D_0(fxp Input_Conv[784],fxp Output_Conv[2304], fxp bias[4], fxp kernel[100]);
void Conv2D_1(fxp Input_Conv[576],fxp Output_Conv[512], fxp bias[8], fxp kernel[800]);
