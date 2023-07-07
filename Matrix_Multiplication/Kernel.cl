
//TODO: set your arguments for the kernel. Note that you have to indicate if the argument is global or local. Global arguments are accessable by both host and this target device. While local can only be accessed by the device running this kernel. eg __global int* inputMatrixA, __local int* groupMemory

__kernel void matrixMultiplication(__global int* matrixA, __global int* matrixB, __global int* output, __global int* Size){
	
	//TODO: program your kernel here

	int workItemNum = get_global_id(0);
	int groupNum = get_group_id(0);
	int localGroupID = get_local_id(0);

        //memory buffers
        uint global_addr = workItemNum;
	int size = *Size;
	
	//printf("work item:%i \t work group:%i  \t col:%i \t elementInA:%i \t elementInB:%i \n", workItemNum, groupNum, workItemNum, matrixA[workItemNum], matrixB[workItemNum]);

	for(int i=0; i<size; i++){
		for(int j=0; j<size; j++){
			int result = 0;
			for(int k=0; k<size; k++){
				result += matrixA[i*size+k] * matrixB[k*size+j];
			}
			output[i*size+j] = result;
		}
	}


	barrier(CLK_LOCAL_MEM_FENCE);
}




