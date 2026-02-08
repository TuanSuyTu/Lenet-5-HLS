#include <ap_axi_sdata.h>
typedef ap_fixed<32, 16> fxp;
void Dense_0(fxp input_Dense[128], fxp output_Dense[32], fxp bias[32], fxp weight[4096])
{
	fxp out_Dense[32];
loop_for_a_Dense_0:
	for (int i = 0; i < 32; i++)
	{
		fxp s = 0;
	loop_for_b_Dense_0:
		for (int j = 0; j < 128; j++)
		{
			s += input_Dense[j] * weight[j * 32 + i];
		}
		out_Dense[i] = s + bias[i];
	}
	for (int i = 0; i < 32; i++)
	{
		if (out_Dense[i] < 0)
			output_Dense[i] = 0;
		else
			output_Dense[i] = out_Dense[i];
	}
}
void Dense_1(fxp input_Dense[32], fxp output_Dense[16], fxp bias[16], fxp weight[512])
{
	fxp out_Dense[16];
loop_for_a_Dense_1:
	for (int i = 0; i < 16; i++)
	{
		fxp s = 0;
	loop_for_b_Dense_1:
		for (int j = 0; j < 32; j++)
		{
			s += input_Dense[j] * weight[j * 16 + i];
		}
		out_Dense[i] = s + bias[i];
	}
	for (int i = 0; i < 16; i++)
	{
		if (out_Dense[i] < 0)
			output_Dense[i] = 0;
		else
			output_Dense[i] = out_Dense[i];
	}
}
#include <hls_math.h>
void Dense_2(fxp input_Dense[16], fxp &output_Dense_0, fxp bias[10], fxp weight[160])
{
	fxp out_Dense[10];
loop_for_a_Dense_2:
	for (int i = 0; i < 10; i++)
	{
		fxp s = 0;
	loop_for_b_Dense_2:
		for (int j = 0; j < 16; j++)
		{
			s += input_Dense[j] * weight[j * 10 + i];
		}
		out_Dense[i] = s + bias[i];
	}
	int maxindex = 0;
	fxp max = out_Dense[0];
loop_detect:
	for (int i = 0; i < 10; i++)
	{
		if (out_Dense[i] > max)
		{
			max = out_Dense[i];
			maxindex = i;
		}
	}
	fxp sum_exp_x = 0.0;
	for (int i = 0; i < 10; i++)
	{
		sum_exp_x += hls::exp(out_Dense[i] - out_Dense[maxindex]);
	}
	fxp max_value = out_Dense[maxindex];
	for (int i = 0; i < 10; i++)
	{
		out_Dense[i] = hls::exp(out_Dense[i] - max_value) / sum_exp_x;
	}
	fxp maxindex_2 = 0;
	fxp max_2 = out_Dense[0];
	for (int i = 0; i < 10; i++)
	{
		if (out_Dense[i] > max_2)
		{
			max_2 = out_Dense[i];
			maxindex_2 = i;
		}
	}
	output_Dense_0 = maxindex_2;
}
