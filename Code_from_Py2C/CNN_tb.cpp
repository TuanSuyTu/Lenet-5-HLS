#define _CRT_SECURE_NO_WARNINGS
#include <conio.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string>
#include <fstream>
#include <iostream>
#include "CNN.h"
#include "Conv.h"
#include "Pool.h"
#include "Dense.h"
#define NumberOfPicture ...
#define d ...
int main(){
	fxp OutModel0;
	fxp* Weights = (fxp*)malloc(5738 * sizeof(fxp));
	float tmp;
	FILE* Weight = fopen("Float_Weights.txt", "r");
	for (int i = 0; i < 5738; i++){
		fscanf(Weight, "%f", &tmp);
		*(Weights + i)=tmp;
	}
	fclose(Weight);
	////read Input
	fxp* InModel = (fxp*)malloc((NumberOfPicture * d * 28 * 28) * sizeof(fxp));
	FILE* Input = fopen("X.txt", "r");
	for (int i = 0; i < NumberOfPicture * d * 28 * 28; i++){
		fscanf(Input, "%f", &tmp);
		*(InModel + i)=tmp;
	}
	fclose(Input);
	//Read Label
	fxp*Label = (fxp*)malloc((NumberOfPicture) * sizeof(fxp));
	FILE* Output = fopen("Y.txt", "r");
	for (int i = 0; i < NumberOfPicture ; i++)
	{
		fscanf(Output, "%f", &tmp);
		*(Label + i) = tmp;
	}
	fclose(Output);
	fxp*OutArray = (fxp*)malloc((NumberOfPicture) * sizeof(fxp));
	fxp Image[d * 28 * 28] = {};
	for (int i = 0; i < NumberOfPicture ; i++)
	{
		int startIndex = i * d * 28 * 28;
		for (int k = 0; k < d * 28 * 28; k++)
		{
			Image[k] = *(InModel + startIndex + k);
		}
		CNN(Image, OutModel0, Weights);
		*(OutArray + i) = OutModel0;
	}
	float countTrue = 0;
	for (int i = 0; i < NumberOfPicture; i++)
	{
		int labelValue = *(Label + i);
		int PredictValue = *(OutArray + i);
		if (labelValue == PredictValue)
		{
			countTrue = countTrue + 1;
		}
	}
	float accuracy = (float)((countTrue / (NumberOfPicture)) * 100);
	std::cout << "accuracy of Model: " << accuracy << "%\n";
	//std::cout << "Result: " <<  OutModel <<  "\n";
	return 0;
}
