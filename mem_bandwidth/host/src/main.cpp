// Copyright (C) 2013-2015 Altera Corporation, San Jose, California, USA. All rights reserved.
// Permission is hereby granted, free of charge, to any person obtaining a copy of this
// software and associated documentation files (the "Software"), to deal in the Software
// without restriction, including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to
// whom the Software is furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.
//
// This agreement shall be governed in all respects by the laws of the State of California and
// by the laws of the United States of America.

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <map>
// ACL specific includes

#define CL_VERSION_2_0
#include "CL/opencl.h"
#include "CL/cl_ext_intelfpga.h"

//#include "ACLHostUtils.h"
#include "AOCLUtils/aocl_utils.h"
//#include "aocl_mmd.h"

#include <time.h>

using namespace aocl_utils;
static const size_t V = 16;
//static size_t vectorSize = 1024*1024*4*16;
static size_t vectorSize = 1024*1024*16;

bool use_prealloc_svm_buffer = true;

double bw;
// 0 - runall, 2 memcopy , 3 read, 4 write, 5 ddr
// 6 - single kernel 2 batch, 7 - single kernel memread and memwrite 
// 8 - single kernel overlap, 9 - separate kernel overlap
// 10 - 2 Batch 
int runkernel = 10;

// ACL runtime configuration
static cl_platform_id platform;
static cl_device_id device;
static cl_context context;
static cl_command_queue queue;
static cl_command_queue queue1, queue2;
static cl_kernel kernel;
static cl_kernel kernel2;
static cl_kernel kernel_read;
static cl_kernel kernel_write;

static cl_program program;
static cl_int status;

#define BACK_BUFFER 1024*1024
#define ACL_ALIGNMENT 1024*4
const char * board_name = "pac_ec00001";

void* acl_aligned_malloc (size_t size) {
	void *result = NULL;
	  posix_memalign (&result, ACL_ALIGNMENT, size);
		return result;
}
void acl_aligned_free (void *ptr) {
	free (ptr);
}

static bool device_has_svm(cl_device_id device) {
   cl_device_svm_capabilities a;
   clGetDeviceInfo(device, CL_DEVICE_SVM_CAPABILITIES, sizeof(cl_uint), &a, NULL);

   if( a & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER)
   	   return true;

   return false;
}

//hdatain = (unsigned int*)clSVMAllocIntelFPGA(context, 0, buf_size, 1024);
void *alloc_fpga_host_buffer(cl_context &in_context, cl_svm_mem_flags mem_flag, size_t size, unsigned alignment);
cl_int set_fpga_buffer_kernel_param(cl_kernel &kernel, int param, void *ptr);

cl_int enqueue_fpga_buffer(cl_command_queue  queue/* command_queue */,
                cl_bool  blocking         /* blocking_map */,
                cl_map_flags  flags    /* flags */,
                void *ptr            /* svm_ptr */,
                size_t len           /* size */,
                cl_uint    num_events       /* num_events_in_wait_list */,
                const cl_event *events  /* event_wait_list */,
                cl_event *the_event        /* event */);
cl_int unenqueue_fpga_buffer(cl_command_queue queue /* command_queue */,
                  void *ptr            /* svm_ptr */,
                  cl_uint  num_events         /* num_events_in_wait_list */,
                  const cl_event *events  /* event_wait_list */,
                  cl_event *the_event        /* event */);
void remove_fpga_buffer(cl_context &context, void *ptr);

// input and output vectors
static unsigned *hdatain, *hdataout, *hdatatemp;
static unsigned *hdatain_2, *hdataout_2, *hdatatemp_2;

cl_mem hdata_ddr_b1, hdata_ddr_b2; // Non-interleaved to Banks 1, 2

static void initializeVector(unsigned* vector, int size) {
  for (int i = 0; i < size; ++i) {
    vector[i] = 0x32103210;
  }
}
static void initializeVector_seq(unsigned* vector, int size) { for (int i = 0; i < size; ++i) {
    vector[i] = i;
  }
}

static void dump_error(const char *str, cl_int status) {
  printf("%s\n", str);
  printf("Error code: %d\n", status);
}

// free the resources allocated during initialization
static void freeResources() {

  if (hdata_ddr_b1)
    clReleaseMemObject(hdata_ddr_b1);
  if (hdata_ddr_b2)
    clReleaseMemObject(hdata_ddr_b2);
  if(kernel)
    clReleaseKernel(kernel);
  if(kernel2)
    clReleaseKernel(kernel2);
  if(kernel_read)
    clReleaseKernel(kernel_read);
  if(kernel_write)
    clReleaseKernel(kernel_write);
  if(program)
    clReleaseProgram(program);
  if(queue)
    clReleaseCommandQueue(queue);
  if(queue1)
    clReleaseCommandQueue(queue1);
  if(queue2)
    clReleaseCommandQueue(queue2);
  if(hdatain)
   remove_fpga_buffer(context,hdatain);
  if(hdataout)
   remove_fpga_buffer(context,hdataout);
  if(hdatain_2)
   remove_fpga_buffer(context,hdatain_2);
  if(hdataout_2)
   remove_fpga_buffer(context,hdataout_2);
  free(hdatatemp);
  free(hdatatemp_2);
  if(context)
    clReleaseContext(context);
}


void *alloc_fpga_host_buffer(cl_context &in_context, cl_svm_mem_flags mem_flag, size_t size, unsigned alignment)
{
#if 0
      if (use_prealloc_svm_buffer) {
			printf("Info: Using preallocated host buffer and custom MPF calls!\n");
         size_t bump = size%ACL_ALIGNMENT ?    (1+(size/ACL_ALIGNMENT))*ACL_ALIGNMENT : size;
         bump = bump + BACK_BUFFER;
         void *ptr = acl_aligned_malloc(bump);
         printf("Address of allocated pointer is %p\n", (void *)ptr);
         aocl_mmd_shared_mem_prepare_buffer( board_name, bump, ptr);
         return ptr;
      } else {
#endif
			printf("Info: Using clSVMAllocIntelFPGA");
		  return clSVMAlloc(in_context, mem_flag, size, alignment);
#if 0
      }
#endif
}

cl_int set_fpga_buffer_kernel_param(cl_kernel &kernel, int param, void *ptr)
{

		return clSetKernelArgSVMPointer(kernel, param, (void*)ptr);

		//return clSetKernelArg(kernel, param, sizeof(cl_mem), &(buffer_map[ptr]));
}

cl_int enqueue_fpga_buffer(cl_command_queue  queue/* command_queue */,
                cl_bool  blocking         /* blocking_map */,
                cl_map_flags  flags    /* flags */,
                void *ptr            /* svm_ptr */,
                size_t len           /* size */,
                cl_uint    num_events       /* num_events_in_wait_list */,
                const cl_event *events  /* event_wait_list */,
                cl_event *the_event        /* event */)
{

  return clEnqueueSVMMap(queue, blocking, flags, ptr, len, num_events, events, the_event);
}


cl_int unenqueue_fpga_buffer(cl_command_queue queue /* command_queue */,
                  void *ptr            /* svm_ptr */,
                  cl_uint  num_events         /* num_events_in_wait_list */,
                  const cl_event *events  /* event_wait_list */,
                  cl_event *the_event        /* event */)
{
	return clEnqueueSVMUnmap(queue, ptr, num_events, events, the_event);
}


void remove_fpga_buffer(cl_context &context, void *ptr)
{
#if 0
      if (use_prealloc_svm_buffer) {
         printf("Info: Using preallocated host buffer and custom MPF calls release buffer!\n");
         aocl_mmd_shared_mem_release_buffer( board_name, ptr);
         acl_aligned_free(ptr);
      } else {
#endif
		   clSVMFree(context,ptr);
#if 0
      }
#endif
}

void cleanup() {

}

int main(int argc, char *argv[]) {
  cl_uint num_platforms;
  cl_uint num_devices;
  int lines = vectorSize/V;
  if ( argc >= 2 ) /* argc should be  >2 for correct execution */
  {
      vectorSize = atoi(argv[1])*V;
      lines = atoi(argv[1]);
  }

  std::string platform_search_string, aocx_name_string;
  if ( argc >= 3 ) // add emu option
  {
    platform_search_string = "Emulation";
    aocx_name_string = "bin/mem_bandwidth_s10_svmddr_emu.aocx";
    printf("3 Arguments given, assuming emulation mode.\n");
  } else {
    platform_search_string = "SDK";
    aocx_name_string = "bin/mem_bandwidth_s10_svmddr_batch3.aocx";
    printf("<3 Arguments given, assuming hardware execution mode.\n");
  }
  printf("Using %s as search string for platform.\n", platform_search_string.c_str());
  printf("Using %s as aocx name.\n", aocx_name_string.c_str());

  if(lines == 0 || lines > 8000000) {
    printf("Invalid Number of cachelines.\n");
    return 1;
  }

  // get the platform ID
  status = clGetPlatformIDs(1, &platform, &num_platforms);
  if(status != CL_SUCCESS) {
    dump_error("Failed clGetPlatformIDs.", status);
    freeResources();
    return 1;
  }
  if(num_platforms != 1) {
    printf("Found %d platforms!\n", num_platforms);
    printf("Trying to select platform %s...\n", platform_search_string.c_str());
    platform = findPlatform(platform_search_string.c_str());
    if(!platform){
      freeResources();
      return 1;
    }
  }

  //platform = findPlatform("SDK");

  // get the device ID
  /*
  status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, &num_devices);
  if(status != CL_SUCCESS) {
    dump_error("Failed clGetDeviceIDs.", status);
    freeResources();
    return 1;
  }
  */
  /*
  if(num_devices != 1) {
    printf("Found %d devices!\n", num_devices);
    freeResources();
    return 1;
  }
  */
  status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
  if(status != CL_SUCCESS) {
    dump_error("Failed clGetDeviceIDs.", status);
    freeResources();
    return 1;
  }
  cl_device_id *devices = (cl_device_id*) malloc(sizeof(cl_device_id) * num_devices);
  clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
  char *value;

  size_t value_sz;
  for(cl_uint j = 0; j < num_devices; j++){
    clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 0, NULL, &value_sz);
    value = (char*) malloc(value_sz);
    clGetDeviceInfo(devices[j], CL_DEVICE_NAME, value_sz, value, NULL);
    printf("%d. Device: %s\n", j+1, value);
    free(value);

    // print device version
    clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, 0, NULL, &value_sz);
    value = (char*) malloc(value_sz);
    clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, value_sz, value, NULL);
    printf("   %d.%d Device version: %s\n", j+1, 1, value);
    free(value);
  }

  printf("-- Choosing first device\n");
  device = devices[0];

  // create a context
  //context = clCreateContext(0, num_devices, devices, NULL, NULL, &status);
  context = clCreateContext(0, 1, &device, NULL, NULL, &status);
  if(status != CL_SUCCESS) {
    dump_error("Failed clCreateContext.", status);
    freeResources();
    return 1;
  }

  if (!device_has_svm(device)) {
    printf("Platform does not use SVM!\n");
    return 0;
  }
  printf("SVM enabled!\n");

  printf("Creating SVM buffers.\n");
  unsigned int buf_size =  vectorSize <= 0 ? 64 : vectorSize*4;

  // allocate and initialize the input vectors
  hdatain =	(unsigned int*)clSVMAlloc(context, CL_MEM_READ_WRITE, buf_size, 1024);
  //hdatain =	(unsigned int*)clSVMAlloc(context, CL_MEM_READ_WRITE, buf_size, 0);
  //hdatain = (unsigned int*)alloc_fpga_host_buffer(context, CL_MEM_READ_WRITE, buf_size, 1024);
  if(hdatain == NULL) {
    dump_error("Failed alloc_fpga_host_buffer.", status);
    freeResources();
    return 1;
  }
  hdataout = (unsigned int*)alloc_fpga_host_buffer(context, 0, buf_size, 1024);
  if(hdataout == NULL) {
    dump_error("Failed alloc_fpga_host_buffer.", status);
    freeResources();
    return 1;
  }
  hdatain_2 = (unsigned int*)alloc_fpga_host_buffer(context, 0, buf_size, 1024);
  if(hdatain == NULL) {
    dump_error("Failed alloc_fpga_host_buffer.", status);
    freeResources();
    return 1;
  }
  hdataout_2 = (unsigned int*)alloc_fpga_host_buffer(context, 0, buf_size, 1024);
  if(hdataout == NULL) {
    dump_error("Failed alloc_fpga_host_buffer.", status);
    freeResources();
    return 1;
  }
  printf("Creating DDR buffers.\n");
  hdatatemp = (unsigned int*)acl_aligned_malloc(buf_size);
  if(hdatatemp == NULL){
    dump_error("Failed acl_aligned_malloc.", status);
    freeResources();
    return 1;
  }

  hdatatemp_2 = (unsigned int*)acl_aligned_malloc(buf_size);
  if(hdatatemp == NULL){
    dump_error("Failed acl_aligned_malloc.", status);
    freeResources();
    return 1;
  }

  hdata_ddr_b1 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_CHANNEL_1_INTELFPGA, buf_size, NULL, &status);
  if(status != CL_SUCCESS) {
    dump_error("Failed clCreateBuffer.", status);
    freeResources();
    return 1;
  }

  hdata_ddr_b2 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_CHANNEL_2_INTELFPGA, buf_size, NULL, &status);
  if(status != CL_SUCCESS) {
    dump_error("Failed clCreateBuffer.", status);
    freeResources();
    return 1;
  }

  printf("Initializing data.\n");
  initializeVector_seq(hdatain, vectorSize);
  initializeVector(hdataout, vectorSize);
  initializeVector_seq(hdatain_2, vectorSize);
  initializeVector(hdataout_2, vectorSize);

  // create a command queue
  queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
  if(status != CL_SUCCESS) {
    dump_error("Failed clCreateCommandQueue.", status);
    freeResources();
    return 1;
  }
  queue1 = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
  if(status != CL_SUCCESS) {
    dump_error("Failed clCreateCommandQueue.", status);
    freeResources();
    return 1;
  }
  queue2 = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
  if(status != CL_SUCCESS) {
    dump_error("Failed clCreateCommandQueue.", status);
    freeResources();
    return 1;
  }

  // create the program

  cl_int kernel_status;

  size_t binsize = 0;
  printf("Trying to load aocx: %s\n", aocx_name_string.c_str());
  unsigned char * binary_file = loadBinaryFile(aocx_name_string.c_str(), &binsize);

  if(!binary_file) {
    dump_error("Failed loadBinaryFile.", status);
    freeResources();
    return 1;
  }
  program = clCreateProgramWithBinary(context, 1, &device, &binsize, (const unsigned char**)&binary_file, &kernel_status, &status);
  if(status != CL_SUCCESS) {
    dump_error("Failed clCreateProgramWithBinary.", status);
    freeResources();
    return 1;
  }
  delete [] binary_file;
  // build the program
  status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
  if(status != CL_SUCCESS) {
    dump_error("Failed clBuildProgram.", status);
    freeResources();
    return 1;
  }

  printf("0 - Creating nop kernel\n");
  kernel = clCreateKernel(program, "nop", &status);
    if(status != CL_SUCCESS) {
      dump_error("Failed clCreateKernel for nop", status);
      freeResources();
      return 1;
    }

    printf("Launching the kernel...\n");

    status = clEnqueueTask(queue, kernel, 0, NULL, NULL);
    if (status != CL_SUCCESS) {
      dump_error("Failed to launch kernel.", status);
      freeResources();
      return 1;
    }
    printf("after kernel nop launch\n");
    clFinish(queue);


// Done kernel launch test
  initializeVector_seq(hdatain, vectorSize);
  initializeVector(hdataout, vectorSize);
  int failures = 0;
  int successes = 0;
  fflush(stdout);
  if (runkernel == 0 || runkernel == 2){
    printf("\n1 - Creating memcopy kernel (Host->Host)\n");
    // create the kernel
    kernel = clCreateKernel(program, "memcopy", &status);
    if(status != CL_SUCCESS) {
      dump_error("Failed clCreateKernel.", status);
      freeResources();
      return 1;
    }

    // set the arguments
    //printf("hdatain[0] %u,  hdatain[1] %u\n", hdatain[0], hdatain[1]);
    //printf("hdatain pointer %p\n", hdatain);
    status = set_fpga_buffer_kernel_param(kernel, 0, (void*)hdatain);
    if(status != CL_SUCCESS) {
      dump_error("Failed set memcopy  arg 0.", status);
      return 1;
    }
    status = set_fpga_buffer_kernel_param(kernel, 1, (void*)hdataout);
    if(status != CL_SUCCESS) {
      dump_error("Failed Set memcopy arg 1.", status);
      freeResources();
      return 1;
    }
    cl_int arg_3 = lines;
    status = clSetKernelArg(kernel, 2, sizeof(cl_int), &(arg_3));
    if(status != CL_SUCCESS) {
      dump_error("Failed Set arg 2.", status);
      freeResources();
      return 1;
    }
    printf("Launching the kernel...\n");
    status = enqueue_fpga_buffer(queue, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE,
       (void *)hdatain,buf_size, 0, NULL, NULL);
    if(status != CL_SUCCESS) {
      dump_error("Failed enqueue_fpga_buffer", status);
      freeResources();
      return 1;
    }
    status = enqueue_fpga_buffer(queue, CL_TRUE,  CL_MAP_READ | CL_MAP_WRITE,
       (void *)hdataout, buf_size, 0, NULL, NULL);
    if(status != CL_SUCCESS) {
      dump_error("Failed enqueue_fpga_buffer", status);
      freeResources();
      return 1;
    }


    const double start_time = getCurrentTimestamp();
    status = clEnqueueTask(queue, kernel, 0, NULL, NULL);
    if (status != CL_SUCCESS) {
      dump_error("Failed to launch kernel.", status);
      freeResources();
      return 1;
    }
    clFinish(queue);
    const double end_time = getCurrentTimestamp();

	  status = unenqueue_fpga_buffer(queue, (void *)hdatain, 0, NULL, NULL);
    if(status != CL_SUCCESS) {
      dump_error("Failed unenqueue_fpga_buffer", status);
      freeResources();
      return 1;
    }
    status = unenqueue_fpga_buffer(queue, (void *)hdataout, 0, NULL, NULL);
    if(status != CL_SUCCESS) {
      dump_error("Failed unenqueue_fpga_buffer", status);
      freeResources();
      return 1;
    }
    clFinish(queue);


    // Wall-clock time taken.
    double time = (end_time - start_time);

    bw = vectorSize / (time * 1000000.0f) * sizeof(unsigned int) * 2;
    printf("Processed %ld unsigned ints in %.4f us\n", vectorSize, time*1000000.0f);
    printf("Read/Write Bandwidth = %.0f MB/s\n", bw);
    printf("Kernel execution is complete.\n");

    // Verify the output
    for(size_t i = 0; i < vectorSize; i++) {
      if(hdatain[i] != hdataout[i]) {
        if (failures < 32) printf("Verification_failure %ld: %d != %d, diff %d, line %ld\n",i, hdatain[i], hdataout[i], hdatain[i]-hdataout[i],i*4/128);
        failures++;
      }else{
        successes++;
      }
    }
  }

  if (runkernel == 0 || runkernel == 5){

    printf("\n5 - Creating memcopy_ddr kernel (DDR->DDR)\n");

    printf("Copying to DDR\n");
    status = clEnqueueWriteBuffer(queue, hdata_ddr_b1, CL_TRUE,
			0, buf_size, hdatain, 0, NULL, NULL);

    // create the kernel
    kernel = clCreateKernel(program, "memcopy_ddr", &status);
    if(status != CL_SUCCESS) {
      dump_error("Failed clCreateKernel.", status);
      freeResources();
      return 1;
    }

    // set the arguments
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &hdata_ddr_b1);
    if(status != CL_SUCCESS) {
      dump_error("Failed set arg 0.", status);
      return 1;
    }
    status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &hdata_ddr_b2);
    if(status != CL_SUCCESS) {
      dump_error("Failed Set arg 1.", status);
      freeResources();
      return 1;
    }

    cl_int arg_3 = lines;
    status = clSetKernelArg(kernel, 2, sizeof(cl_int), &(arg_3));
    if(status != CL_SUCCESS) {
      dump_error("Failed Set arg 2.", status);
      freeResources();
      return 1;
    }

    printf("Launching the kernel...\n");

    const double start_time = getCurrentTimestamp();
    status = clEnqueueTask(queue, kernel, 0, NULL, NULL);
    if (status != CL_SUCCESS) {
      dump_error("Failed to launch kernel.", status);
      freeResources();
      return 1;
    }
    printf("after kernel launch\n");
    clFinish(queue);
    const double end_time = getCurrentTimestamp();

    status = unenqueue_fpga_buffer(queue, (void *)hdatain, 0, NULL, NULL);
    if(status != CL_SUCCESS) {
      dump_error("Failed unenqueue_fpga_buffer", status);
      freeResources();
      return 1;
    }
    status = unenqueue_fpga_buffer(queue, (void *)hdataout, 0, NULL, NULL);
    if(status != CL_SUCCESS) {
      dump_error("Failed unenqueue_fpga_buffer", status);
      freeResources();
      return 1;
    }

	 clFinish(queue);

    // Wall-clock time taken.
    double time = (end_time - start_time);

    bw = vectorSize / (time * 1000000.0f) * sizeof(unsigned int);
    printf("Processed %ld unsigned ints in %.4f us\n", vectorSize, time*1000000.0f);
    printf("Read/Write Bandwidth = %.0f MB/s\n", bw);
    printf("Kernel execution is complete.\n");

    printf("Copying from DDR\n");
    status = clEnqueueReadBuffer(queue, hdata_ddr_b2, CL_TRUE,
			0, buf_size, hdatatemp, 0, NULL, NULL);

    if(status != CL_SUCCESS) {
      dump_error("Failed clEnqueueReadBuffer", status);
      freeResources();
      return 1;
    }
    // Verify the output
    for(size_t i = 0; i < vectorSize; i++) {
      if(hdatain[i] != hdatatemp[i]) {
        if (failures < 32) printf("Verification_failure %ld: %d != %d, diff %d, line %ld\n",i, hdatain[i], hdataout[i], hdatain[i]-hdataout[i],i*4/128);
        failures++;
      }else{
        successes++;
      }
    }
  }

  if (runkernel == 0 || runkernel == 9){
    printf("\n9 - Creating batched memcopy_to_ddr and memcopy_from_ddr kernel (Host->DDR->Host)\n");

    // create the kernel
    kernel = clCreateKernel(program, "memcopy_to_ddr", &status);
    kernel2 = clCreateKernel(program, "memcopy_from_ddr", &status);
    if(status != CL_SUCCESS) {
      dump_error("Failed clCreateKernel.", status);
      freeResources();
      return 1;
    }

    // set the arguments
    status = set_fpga_buffer_kernel_param(kernel, 0, (void*)hdatain);
    if(status != CL_SUCCESS) {
      dump_error("Failed set memcopy_to_ddr  arg 0.", status);
      return 1;
    }
    status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &hdata_ddr_b1);
    if(status != CL_SUCCESS) {
      dump_error("Failed Set memcopy_to_ddr arg 1.", status);
      freeResources();
      return 1;
    }

    cl_int arg_3 = lines;
    status = clSetKernelArg(kernel, 2, sizeof(cl_int), &(arg_3));
    if(status != CL_SUCCESS) {
      dump_error("Failed Set arg 2.", status);
      freeResources();
      return 1;
    }

    status = clSetKernelArg(kernel2, 0, sizeof(cl_mem), &hdata_ddr_b2);
    if(status != CL_SUCCESS) {
      dump_error("Failed set memcopy_from_ddr arg 0.", status);
      return 1;
    }
    status = set_fpga_buffer_kernel_param(kernel2, 1, (void*)hdataout);
    if(status != CL_SUCCESS) {
      dump_error("Failed Set memcopy_from_ddr arg 1.", status);
      freeResources();
      return 1;
    }
    arg_3 = lines;
    status = clSetKernelArg(kernel2, 2, sizeof(cl_int), &(arg_3));
    if(status != CL_SUCCESS) {
      dump_error("Failed Set arg 2.", status);
      freeResources();
      return 1;
    }
    printf("Writing to device\n");
    status = clEnqueueWriteBuffer(queue1, hdata_ddr_b2, CL_TRUE,
			0, buf_size, hdatain_2, 0, NULL, NULL);

    clFinish(queue1);

    status = enqueue_fpga_buffer(queue, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE,
       (void *)hdatain, buf_size, 0, NULL, NULL);
    if(status != CL_SUCCESS) {
      dump_error("Failed enqueue_fpga_buffer", status);
      freeResources();
      return 1;
    }

    status = enqueue_fpga_buffer(queue, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE,
       (void *)hdataout, buf_size, 0, NULL, NULL);
    if(status != CL_SUCCESS) {
      dump_error("Failed enqueue_fpga_buffer", status);
      freeResources();
      return 1;
    }

    printf("Launching the kernel...\n");

    cl_event start_first_batch, end_first_batch, start_second_batch, end_second_batch;
    const double start_time = getCurrentTimestamp();

    // Kernel2: DDR->Host, Kernel: Host->DDR
    status = clEnqueueTask(queue1, kernel, 0, NULL, &start_second_batch);
    if (status != CL_SUCCESS) {
      dump_error("Failed to launch kernel.", status);
      freeResources();
      return 1;
    }

    // Expm to delay writing to global memory by corresponding time of reading
    // from global memory to show the former stalling when both are overlapped
    // usleep(2000);
    status = clEnqueueTask(queue2, kernel2, 0, NULL, &end_first_batch);
    if (status != CL_SUCCESS) {
      dump_error("Failed to launch kernel.", status);
      freeResources();
      return 1;
    }


    clFinish(queue1);
    clFinish(queue2);

    const double end_time = getCurrentTimestamp();

	  status = unenqueue_fpga_buffer(queue, (void *)hdatain, 0, NULL, NULL);
    if(status != CL_SUCCESS) {
      dump_error("Failed unenqueue_fpga_buffer", status);
      freeResources();
      return 1;
    }
    status = unenqueue_fpga_buffer(queue, (void *)hdataout, 0, NULL, NULL);
    if(status != CL_SUCCESS) {
      dump_error("Failed unenqueue_fpga_buffer", status);
      freeResources();
      return 1;
    }
    clFinish(queue);

    status = clEnqueueReadBuffer(queue1, hdata_ddr_b1, CL_TRUE,
			0, buf_size, hdatatemp, 0, NULL, NULL);
    if(status != CL_SUCCESS) {
      dump_error("Failed enqueue_fpga_buffer", status);
      freeResources();
      return 1;
    }
    clFinish(queue1);

    cl_ulong batch1_kernel1_start = 0, batch1_kernel1_end = 0;
    cl_ulong batch1_kernel2_start = 0, batch1_kernel2_end = 0;
    cl_ulong batch2_kernel1_start = 0, batch2_kernel1_end = 0;
    cl_ulong batch2_kernel2_start = 0, batch2_kernel2_end = 0;

    clGetEventProfilingInfo(start_first_batch, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &batch1_kernel1_start, NULL);
    clGetEventProfilingInfo(start_first_batch, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &batch1_kernel1_end, NULL);
    clGetEventProfilingInfo(end_first_batch, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &batch1_kernel2_start, NULL);
    clGetEventProfilingInfo(end_first_batch, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &batch1_kernel2_end, NULL);
    clGetEventProfilingInfo(start_second_batch, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &batch2_kernel1_start, NULL);
    clGetEventProfilingInfo(start_second_batch, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &batch2_kernel1_end, NULL);
    clGetEventProfilingInfo(end_second_batch, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &batch2_kernel2_start, NULL);
    clGetEventProfilingInfo(end_second_batch, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &batch2_kernel2_end, NULL);

    // Time is recorded in nanoseconds
    printf("\n");
    printf("Batch 1: Write Kernel Start  %lu\n", batch1_kernel1_start);
    printf("Batch 1: Write Kernel End    %lu\n", batch1_kernel1_end);
    printf("Batch 1: Read Kernel Start   %lu\n", batch1_kernel2_start);
    printf("Batch 1: Read Kernel End     %lu\n", batch1_kernel2_end);
    printf("Batch 2: Write Kernel Start  %lu\n", batch2_kernel1_start);
    printf("Batch 2: Write Kernel End    %lu\n", batch2_kernel1_end);
    printf("Batch 2: Read Kernel Start   %lu\n", batch2_kernel2_start);
    printf("Batch 2: Read Kernel End     %lu\n", batch2_kernel2_end);
    printf("\n");

    // ns to ms
    double wr_batch1_exec_t = (double)(batch1_kernel1_end - batch1_kernel1_start) * 1e-6;
    double rd_batch1_exec_t = (double)(batch1_kernel2_end - batch1_kernel2_start) * 1e-6;
    double wr_batch2_exec_t = (double)(batch2_kernel1_end - batch2_kernel1_start) * 1e-6;
    double rd_batch2_exec_t = (double)(batch2_kernel2_end - batch2_kernel2_start) * 1e-6;
    printf("Batch 1: Write Kernel Exec Time %lf ms\n", wr_batch1_exec_t);
    printf("Batch 1: Read Kernel Exec Time  %lf ms\n", rd_batch1_exec_t);
    printf("Batch 2: Write Kernel Exec Time %lf ms\n", wr_batch2_exec_t);
    printf("Batch 2: Read Kernel Exec Time  %lf ms\n", rd_batch2_exec_t);
    printf("\n");
    double time = (end_time - start_time);

    bw = vectorSize * sizeof(unsigned int) * 2 / (time * 1000000.0f);
    printf("Processed %ld unsigned ints in %.4f us\n", 2* vectorSize, time*1000000.0f);
    printf("Read/Write Bandwidth = %.0f MB/s\n", bw);
    printf("Kernel execution is complete.\n");

    // Verify the output
    for(size_t i = 0; i < vectorSize; i++) {
      if(hdatain[i] != hdatatemp[i]) {
        if (failures < 32) printf("Verification_failure %ld: %d != %d, diff %d, line %ld\n",i, hdatain[i], hdatatemp[i], hdatain[i]-hdataout[i],i*4/128);
        failures++;
      }else{
        successes++;
      }
    }
    // Verify the output
    for(size_t i = 0; i < vectorSize; i++) {
      if(hdatain_2[i] != hdataout[i]) {
        if (failures < 32) printf("Verification_failure %ld: %d != %d, diff %d, line %ld\n",i, hdatain_2[i], hdataout[i], hdatain_2[i] - hdataout[i],i*4/128);
        failures++;
      }else{
        successes++;
      }
    }
  }

  if(runkernel == 0 || runkernel == 6){

    printf("\n6 - Creating single kernel batched memcopy (Host->DDR->Host)\n");
    // create the kernel
    kernel = clCreateKernel(program, "memcopy_batch", &status);
    if(status != CL_SUCCESS) {
      dump_error("Failed clCreateKernel.", status);
      freeResources();
      return 1;
    }

    // set the arguments
    status = set_fpga_buffer_kernel_param(kernel, 0, (void*)hdatain);
    if(status != CL_SUCCESS) {
      dump_error("Failed set memcopy_batch arg 0.", status);
      return 1;
    }
    status = set_fpga_buffer_kernel_param(kernel, 1, (void*)hdatain_2);
    if(status != CL_SUCCESS) {
      dump_error("Failed set memcopy_batch arg 1.", status);
      return 1;
    }

    status = clSetKernelArg(kernel, 2, sizeof(cl_mem), &hdata_ddr_b1);
    if(status != CL_SUCCESS) {
      dump_error("Failed Set memcopy_batch arg 2.", status);
      freeResources();
      return 1;
    }

    status = clSetKernelArg(kernel, 3, sizeof(cl_mem), &hdata_ddr_b2);
    if(status != CL_SUCCESS) {
      dump_error("Failed Set memcopy_batch arg 3.", status);
      freeResources();
      return 1;
    }
    status = set_fpga_buffer_kernel_param(kernel, 4, (void*)hdataout);
    if(status != CL_SUCCESS) {
      dump_error("Failed Set memcopy_batch arg 4.", status);
      freeResources();
      return 1;
    }
    status = set_fpga_buffer_kernel_param(kernel, 5, (void*)hdataout_2);
    if(status != CL_SUCCESS) {
      dump_error("Failed Set memcopy_batch arg 5.", status);
      freeResources();
      return 1;
    }

    cl_int arg_3 = lines;
    status = clSetKernelArg(kernel, 6, sizeof(cl_int), &(arg_3));
    if(status != CL_SUCCESS) {
      dump_error("Failed Set arg 6.", status);
      freeResources();
      return 1;
    }

    printf("Launching the kernel...\n");
    status = enqueue_fpga_buffer(queue, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE,
       (void *)hdatain, buf_size, 0, NULL, NULL);
    if(status != CL_SUCCESS) {
      dump_error("Failed enqueue_fpga_buffer", status);
      freeResources();
      return 1;
    }
    status = enqueue_fpga_buffer(queue, CL_TRUE,  CL_MAP_READ | CL_MAP_WRITE,
       (void *)hdataout, buf_size, 0, NULL, NULL);
    if(status != CL_SUCCESS) {
      dump_error("Failed enqueue_fpga_buffer", status);
      freeResources();
      return 1;
    }
    status = enqueue_fpga_buffer(queue, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE,
       (void *)hdatain_2, buf_size, 0, NULL, NULL);
    if(status != CL_SUCCESS) {
      dump_error("Failed enqueue_fpga_buffer", status);
      freeResources();
      return 1;
    }
    status = enqueue_fpga_buffer(queue, CL_TRUE,  CL_MAP_READ | CL_MAP_WRITE,
       (void *)hdataout_2, buf_size, 0, NULL, NULL);
    if(status != CL_SUCCESS) {
      dump_error("Failed enqueue_fpga_buffer", status);
      freeResources();
      return 1;
    }

    cl_event start_batch;
    const double start_time = getCurrentTimestamp();
    status = clEnqueueTask(queue1, kernel, 0, NULL, &start_batch);
    if (status != CL_SUCCESS) {
      dump_error("Failed to launch kernel.", status);
      freeResources();
      return 1;
    }

    clFinish(queue1);
    const double end_time = getCurrentTimestamp();

	  status = unenqueue_fpga_buffer(queue, (void *)hdatain, 0, NULL, NULL);
    if(status != CL_SUCCESS) {
      dump_error("Failed unenqueue_fpga_buffer", status);
      freeResources();
      return 1;
    }
    status = unenqueue_fpga_buffer(queue, (void *)hdataout, 0, NULL, NULL);
    if(status != CL_SUCCESS) {
      dump_error("Failed unenqueue_fpga_buffer", status);
      freeResources();
      return 1;
    }
	  status = unenqueue_fpga_buffer(queue, (void *)hdatain_2, 0, NULL, NULL);
    if(status != CL_SUCCESS) {
      dump_error("Failed unenqueue_fpga_buffer", status);
      freeResources();
      return 1;
    }
    status = unenqueue_fpga_buffer(queue, (void *)hdataout_2, 0, NULL, NULL);
    if(status != CL_SUCCESS) {
      dump_error("Failed unenqueue_fpga_buffer", status);
      freeResources();
      return 1;
    }
    clFinish(queue);

    cl_ulong kernel_start = 0.0, kernel_end = 0.0;
    clGetEventProfilingInfo(start_batch, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &kernel_start, NULL);
    clGetEventProfilingInfo(start_batch, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &kernel_end, NULL);

    printf("\n");
    printf("Batch 1: Write Kernel Start  %lf\n", (double)kernel_start);
    printf("Batch 1: Write Kernel End    %lf\n", (double)kernel_end);
    printf("\n");

    printf("Batch 1: Write Kernel Exec Time %lf\n", (double)kernel_end - (double)kernel_start);
    printf("\n");
    double time = (end_time - start_time);

    bw = vectorSize * sizeof(unsigned int) * 2 * 2 / (time * 1000000.0f);
    printf("Processed %ld unsigned ints in %.4f us\n", 2* 2* vectorSize, time*1000000.0f);
    printf("Read/Write Bandwidth = %.0f MB/s\n", bw);
    printf("Kernel execution is complete.\n");

    // Verify the output
    for(size_t i = 0; i < vectorSize; i++) {
      if(hdatain[i] != hdataout[i]) {
        if (failures < 32) printf("Verification_failure %ld: %d != %d, diff %d, line %ld\n",i, hdatain[i], hdataout[i], hdatain[i]-hdataout[i],i*4/128);
        failures++;
      }else{
        successes++;
      }
    }
    // Verify the output
    for(size_t i = 0; i < vectorSize; i++) {
      if(hdatain_2[i] != hdataout_2[i]) {
        if (failures < 32) printf("Verification_failure %ld: %d != %d, diff %d, line %ld\n",i, hdatain[i], hdataout[i], hdatain[i]-hdataout[i],i*4/128);
        failures++;
      }else{
        successes++;
      }
    }
    printf("Read from DDR.\n");
    status = clEnqueueReadBuffer(queue, hdata_ddr_b1, CL_TRUE,
			0, buf_size, hdatatemp, 0, NULL, NULL);

    if(status != CL_SUCCESS) {
      dump_error("Failed clEnqueueReadBuffer", status);
      freeResources();
      return 1;
    }
    // Verify the output
    for(size_t i = 0; i < vectorSize; i++) {
      if(hdatain[i] != hdatatemp[i]) {
        if (failures < 32) printf("Verification_failure %ld: %d != %d, diff %d, line %ld\n",i, hdatain[i], hdataout[i], hdatain[i]-hdataout[i],i*4/128);
        failures++;
      }else{
        successes++;
      }
    }

    status = clEnqueueReadBuffer(queue, hdata_ddr_b2, CL_TRUE,
			0, buf_size, hdatatemp_2, 0, NULL, NULL);

    if(status != CL_SUCCESS) {
      dump_error("Failed clEnqueueReadBuffer", status);
      freeResources();
      return 1;
    }
    // Verify the output
    for(size_t i = 0; i < vectorSize; i++) {
      if(hdatain_2[i] != hdatatemp_2[i]) {
        if (failures < 32) printf("Verification_failure %ld: %d != %d, diff %d, line %ld\n",i, hdatain_2[i], hdataout_2[i], hdatain_2[i]-hdataout_2[i],i*4/128);
        failures++;
      }else{
        successes++;
      }
    }

  }

  if (runkernel == 0 || runkernel == 3){
    printf("\n3 - Creating memread kernel\n");

    kernel_read = clCreateKernel(program, "memread", &status);
    if(status != CL_SUCCESS) {
      dump_error("Failed clCreateKernel.", status);
      freeResources();
      return 1;
    }

    // set the arguments
    status = set_fpga_buffer_kernel_param(kernel_read, 0, (void*)hdatain);
    if(status != CL_SUCCESS) {
      dump_error("Failed set arg 0.", status);
      return 1;
    }
    status = set_fpga_buffer_kernel_param(kernel_read, 1, (void*)hdataout);
    if(status != CL_SUCCESS) {
      dump_error("Failed Set arg 1.", status);
      freeResources();
      return 1;
    }

    cl_int arg_3 = lines;
    status = clSetKernelArg(kernel_read, 2, sizeof(cl_int), &(arg_3));
    if(status != CL_SUCCESS) {
      dump_error("Failed Set arg 2.", status);
      freeResources();
      return 1;
    }
    printf("Launching the kernel...\n");
    status = enqueue_fpga_buffer(queue, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE,
       (void *)hdatain,buf_size, 0, NULL, NULL);
    if(status != CL_SUCCESS) {
      dump_error("Failed enqueue_fpga_buffer", status);
      freeResources();
      return 1;
    }
    status = enqueue_fpga_buffer(queue, CL_TRUE,  CL_MAP_READ | CL_MAP_WRITE,
       (void *)hdataout, buf_size, 0, NULL, NULL);
    if(status != CL_SUCCESS) {
      dump_error("Failed enqueue_fpga_buffer", status);
      freeResources();
      return 1;
    }

    // launch kernel
    const double start_time = getCurrentTimestamp();
    status = clEnqueueTask(queue, kernel_read, 0, NULL, NULL);
    if (status != CL_SUCCESS) {
      dump_error("Failed to launch kernel.", status);
      freeResources();
      return 1;
    }

    clFinish(queue);
    const double end_time = getCurrentTimestamp();

	  status = unenqueue_fpga_buffer(queue, (void *)hdatain, 0, NULL, NULL);
    if(status != CL_SUCCESS) {
      dump_error("Failed unenqueue_fpga_buffer", status);
      freeResources();
      return 1;
    }
    status = unenqueue_fpga_buffer(queue, (void *)hdataout, 0, NULL, NULL);
    if(status != CL_SUCCESS) {
      dump_error("Failed unenqueue_fpga_buffer", status);
      freeResources();
      return 1;
    }

    clFinish(queue);

    // Wall-clock time taken.
    double time = (end_time - start_time);

    bw = vectorSize  / (time * 1000000.0f) * sizeof(unsigned int);
    printf("Processed %ld unsigned ints in %.4f us\n", vectorSize, time*1000000.0f);
    printf("Read Bandwidth = %.0f MB/s\n", bw);
    printf("Kernel execution is complete.\n");

  }

  if (runkernel == 0 || runkernel == 4){
    printf("\n4 - Creating memwrite kernel\n");
    kernel_write = clCreateKernel(program, "memwrite", &status);

    if(status != CL_SUCCESS) {
      dump_error("Failed clCreateKernel.", status);
      freeResources();
      return 1;
    }

    // set the arguments
    status = set_fpga_buffer_kernel_param(kernel_write, 0, (void*)hdatain);
    if(status != CL_SUCCESS) {
      dump_error("Failed set arg 0.", status);
      return 1;
    }
    status = set_fpga_buffer_kernel_param(kernel_write, 1, (void*)hdataout);
    if(status != CL_SUCCESS) {
      dump_error("Failed Set arg 1.", status);
      freeResources();
      return 1;
    }

    cl_int arg_3 = lines;
    status = clSetKernelArg(kernel_write, 2, sizeof(cl_int), &(arg_3));
    if(status != CL_SUCCESS) {
      dump_error("Failed Set arg 2.", status);
      freeResources();
      return 1;
    }
    status = enqueue_fpga_buffer(queue, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE,
       (void *)hdatain,buf_size, 0, NULL, NULL);
    if(status != CL_SUCCESS) {
      dump_error("Failed enqueue_fpga_buffer", status);
      freeResources();
      return 1;
    }
    status = enqueue_fpga_buffer(queue, CL_TRUE,  CL_MAP_READ | CL_MAP_WRITE,
       (void *)hdataout, buf_size, 0, NULL, NULL);
    if(status != CL_SUCCESS) {
      dump_error("Failed enqueue_fpga_buffer", status);
      freeResources();
      return 1;
    }

	  printf("Launching the kernel...\n");

    const double start_time = getCurrentTimestamp();
    status = clEnqueueTask(queue, kernel_write, 0, NULL, NULL);
    if (status != CL_SUCCESS) {
      dump_error("Failed to launch kernel.", status);
      freeResources();
      return 1;
    }
    clFinish(queue);
    const double end_time = getCurrentTimestamp();

    // Wall-clock time taken.
    double time = (end_time - start_time);

    bw = vectorSize  / (time * 1000000.0f) * sizeof(unsigned int);
    printf("Processed %ld unsigned ints in %.4f us\n", vectorSize, time*1000000.0f);
    printf("Write Bandwidth = %.0f MB/s\n", bw);
    printf("Kernel execution is complete.\n");

  }

  if(runkernel == 0 || runkernel == 7){
    printf("\n7 - Creating Single kernel with both memread and memwrite kernel\n");
    kernel_write = clCreateKernel(program, "memread_memwrite", &status);
    if(status != CL_SUCCESS) {
      dump_error("Failed clCreateKernel.", status);
      freeResources();
      return 1;
    }

    // set the arguments
    status = set_fpga_buffer_kernel_param(kernel_write, 0, (void*)hdatain);
    if(status != CL_SUCCESS) {
      dump_error("Failed set arg 0.", status);
      return 1;
    }
    status = set_fpga_buffer_kernel_param(kernel_write, 1, (void*)hdataout);
    if(status != CL_SUCCESS) {
      dump_error("Failed Set arg 1.", status);
      freeResources();
      return 1;
    }

    status = set_fpga_buffer_kernel_param(kernel_write, 2, (void*)hdataout_2);
    if(status != CL_SUCCESS) {
      dump_error("Failed set arg 0.", status);
      return 1;
    }
    cl_int arg_3 = lines;
    status = clSetKernelArg(kernel_write, 3, sizeof(cl_int), &(arg_3));
    if(status != CL_SUCCESS) {
      dump_error("Failed Set arg 2.", status);
      freeResources();
      return 1;
    }

    status = enqueue_fpga_buffer(queue, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE,
       (void *)hdatain,buf_size, 0, NULL, NULL);
    if(status != CL_SUCCESS) {
      dump_error("Failed enqueue_fpga_buffer", status);
      freeResources();
      return 1;
    }
    status = enqueue_fpga_buffer(queue, CL_TRUE,  CL_MAP_READ | CL_MAP_WRITE,
       (void *)hdataout, buf_size, 0, NULL, NULL);
    if(status != CL_SUCCESS) {
      dump_error("Failed enqueue_fpga_buffer", status);
      freeResources();
      return 1;
    }
    status = enqueue_fpga_buffer(queue, CL_TRUE,  CL_MAP_READ | CL_MAP_WRITE,
       (void *)hdataout_2, buf_size, 0, NULL, NULL);
    if(status != CL_SUCCESS) {
      dump_error("Failed enqueue_fpga_buffer", status);
      freeResources();
      return 1;
    }
	  printf("Launching the kernel...\n");

    cl_event start_kernel;
    const double start_time = getCurrentTimestamp();
    status = clEnqueueTask(queue1, kernel_write, 0, NULL, &start_kernel);
    if (status != CL_SUCCESS) {
      dump_error("Failed to launch kernel.", status);
      freeResources();
      return 1;
    }
    clFinish(queue1);
    const double end_time = getCurrentTimestamp();

	  status = unenqueue_fpga_buffer(queue, (void *)hdatain, 0, NULL, NULL);
    if(status != CL_SUCCESS) {
      dump_error("Failed unenqueue_fpga_buffer", status);
      freeResources();
      return 1;
    }
    status = unenqueue_fpga_buffer(queue, (void *)hdataout, 0, NULL, NULL);
    if(status != CL_SUCCESS) {
      dump_error("Failed unenqueue_fpga_buffer", status);
      freeResources();
      return 1;
    }
	  status = unenqueue_fpga_buffer(queue, (void *)hdataout_2, 0, NULL, NULL);
    if(status != CL_SUCCESS) {
      dump_error("Failed unenqueue_fpga_buffer", status);
      freeResources();
      return 1;
    }
    clFinish(queue);

    /*
    for(size_t i = 0; i < vectorSize; i++) {
      if(hdatain_2[i] != 1) {
        if (failures < 32) 
         printf("Verification_failure %d: %d != 1, line %d\n",i, hdataout_2[i], i*4/128);
        failures++;
      }else{
        successes++;
      }
    }
    */

    cl_ulong kernel_start = 0.0, kernel_end = 0.0;
    clGetEventProfilingInfo(start_kernel, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &kernel_start, NULL);
    clGetEventProfilingInfo(start_kernel, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &kernel_end, NULL);

    printf("\n");
    printf("Kernel Start  %lf\n", (double)kernel_start);
    printf("Kernel End    %lf\n", (double)kernel_end);
    printf("\n");

    printf("Write Kernel Exec Time %lf\n", (double)kernel_end - (double)kernel_start);
    printf("\n");
    double time = (end_time - start_time);

    bw = vectorSize  / (time * 1000000.0f) * sizeof(unsigned int);
    printf("Processed %ld unsigned ints in %.4f us\n", vectorSize, time*1000000.0f);
    printf("Write Bandwidth = %.0f MB/s\n", bw);
    printf("Kernel execution is complete.\n");
  }

  if(runkernel == 0 || runkernel == 8){
    printf("\n8 - Overlapping Device Read and Write\n");
    kernel_write = clCreateKernel(program, "memcopy_read_write", &status);
    if(status != CL_SUCCESS) {
      dump_error("Failed clCreateKernel.", status);
      freeResources();
      return 1;
    }

    // set the arguments
    status = set_fpga_buffer_kernel_param(kernel_write, 0, (void*)hdatain);
    if(status != CL_SUCCESS) {
      dump_error("Failed set arg 0.", status);
      return 1;
    }
    status = set_fpga_buffer_kernel_param(kernel_write, 1, (void*)hdataout);
    if(status != CL_SUCCESS) {
      dump_error("Failed Set arg 1.", status);
      freeResources();
      return 1;
    }

    status = clSetKernelArg(kernel_write, 2, sizeof(cl_mem), &hdata_ddr_b1);
    if(status != CL_SUCCESS) {
      dump_error("Failed Set arg 2.", status);
      freeResources();
      return 1;
    }
    status = clSetKernelArg(kernel_write, 3, sizeof(cl_mem), &hdata_ddr_b2);
    if(status != CL_SUCCESS) {
      dump_error("Failed set arg 3.", status);
      return 1;
    }

    cl_int arg_3 = lines;
    status = clSetKernelArg(kernel_write, 4, sizeof(cl_int), &(arg_3));
    if(status != CL_SUCCESS) {
      dump_error("Failed Set arg 4.", status);
      freeResources();
      return 1;
    }

    printf("Copying to DDR\n");
    status = clEnqueueWriteBuffer(queue, hdata_ddr_b2, CL_TRUE,
			0, buf_size, hdatain_2, 0, NULL, NULL);
    clFinish(queue);

    status = enqueue_fpga_buffer(queue, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE,
       (void *)hdatain, buf_size, 0, NULL, NULL);
    if(status != CL_SUCCESS) {
      dump_error("Failed enqueue_fpga_buffer", status);
      freeResources();
      return 1;
    }
    status = enqueue_fpga_buffer(queue, CL_TRUE,  CL_MAP_READ | CL_MAP_WRITE,
       (void *)hdataout, buf_size, 0, NULL, NULL);
    if(status != CL_SUCCESS) {
      dump_error("Failed enqueue_fpga_buffer", status);
      freeResources();
      return 1;
    }
	  printf("Launching the kernel...\n");

    cl_event start_kernel;
    const double start_time = getCurrentTimestamp();
    status = clEnqueueTask(queue1, kernel_write, 0, NULL, &start_kernel);
    if (status != CL_SUCCESS) {
      dump_error("Failed to launch kernel.", status);
      freeResources();
      return 1;
    }
    clFinish(queue1);

    const double end_time = getCurrentTimestamp();

	  status = unenqueue_fpga_buffer(queue, (void *)hdatain, 0, NULL, NULL);
    if(status != CL_SUCCESS) {
      dump_error("Failed unenqueue_fpga_buffer", status);
      freeResources();
      return 1;
    }
    status = unenqueue_fpga_buffer(queue, (void *)hdataout, 0, NULL, NULL);
    if(status != CL_SUCCESS) {
      dump_error("Failed unenqueue_fpga_buffer", status);
      freeResources();
      return 1;
    }
    clFinish(queue1);

    // read from both banks
    printf("Copying from DDR\n");
    status = clEnqueueReadBuffer(queue, hdata_ddr_b1, CL_TRUE,
			0, buf_size, hdatatemp, 0, NULL, NULL);
    clFinish(queue);

    if(status != CL_SUCCESS) {
      dump_error("Failed clEnqueueReadBuffer", status);
      freeResources();
      return 1;
    }
    // Verify the output
    for(size_t i = 0; i < vectorSize; i++) {
      if(hdatain[i] != hdatatemp[i]) {
        if (failures < 32) printf("Verification_failure %ld: %d != %d, diff %d, line %ld\n",i, hdatain[i], hdataout[i], hdatain[i]-hdataout[i],i*4/128);
        failures++;
      }else{
        successes++;
      }
    }

    for(size_t i = 0; i < vectorSize; i++) {
      if(hdatain_2[i] != hdataout[i]) {
        if (failures < 32) printf("Verification_failure %ld: %d != %d, diff %d, line %ld\n",i, hdatain_2[i], hdataout[i], hdatain_2[i]-hdataout[i],i*4/128);
        failures++;
      }else{
        successes++;
      }
    }

    cl_ulong kernel_start = 0.0, kernel_end = 0.0;
    clGetEventProfilingInfo(start_kernel, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &kernel_start, NULL);
    clGetEventProfilingInfo(start_kernel, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &kernel_end, NULL);

    printf("\n");
    printf("Kernel Start  %lf\n", (double)kernel_start);
    printf("Kernel End    %lf\n", (double)kernel_end);
    printf("\n");

    printf("Write Kernel Exec Time %lf\n", (double)kernel_end - (double)kernel_start);
    printf("\n");
    double time = (end_time - start_time);

    bw = vectorSize * 2 / (time * 1000000.0f) * sizeof(unsigned int);
    printf("Processed %ld unsigned ints in %.4f us\n", vectorSize * 2, time*1000000.0f);
    printf("Full Duplex Bandwidth = %.0f MB/s\n", bw);
    printf("Kernel execution is complete.\n");
  }

  if(runkernel == 0 || runkernel == 10){
    printf("\n10 - Working Batch\n");

    // Kernel to copy data from host to DDR
    kernel = clCreateKernel(program, "memcopy_to_ddr", &status);
    status = set_fpga_buffer_kernel_param(kernel, 0, (void*)hdatain);
    if(status != CL_SUCCESS) {
      dump_error("Failed set memcopy_to_ddr  arg 0.", status);
      return 1;
    }
    status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &hdata_ddr_b1);
    if(status != CL_SUCCESS) {
      dump_error("Failed Set memcopy_to_ddr arg 1.", status);
      freeResources();
      return 1;
    }

    cl_int arg_3 = lines;
    status = clSetKernelArg(kernel, 2, sizeof(cl_int), &(arg_3));
    if(status != CL_SUCCESS) {
      dump_error("Failed Set arg 2.", status);
      freeResources();
      return 1;
    }

    // Kernel to overlap writes and reads
    kernel_write = clCreateKernel(program, "memcopy_read_write", &status);
    if(status != CL_SUCCESS) {
      dump_error("Failed clCreateKernel.", status);
      freeResources();
      return 1;
    }

    status = set_fpga_buffer_kernel_param(kernel_write, 0, (void*)hdatain);
    if(status != CL_SUCCESS) {
      dump_error("Failed set arg 0.", status);
      return 1;
    }
    status = set_fpga_buffer_kernel_param(kernel_write, 1, (void*)hdataout);
    if(status != CL_SUCCESS) {
      dump_error("Failed Set arg 1.", status);
      freeResources();
      return 1;
    }

    status = clSetKernelArg(kernel_write, 2, sizeof(cl_mem), &hdata_ddr_b2);
    if(status != CL_SUCCESS) {
      dump_error("Failed Set arg 2.", status);
      freeResources();
      return 1;
    }
    status = clSetKernelArg(kernel_write, 3, sizeof(cl_mem), &hdata_ddr_b1);
    if(status != CL_SUCCESS) {
      dump_error("Failed set arg 3.", status);
      return 1;
    }

    status = clSetKernelArg(kernel_write, 4, sizeof(cl_int), &(arg_3));
    if(status != CL_SUCCESS) {
      dump_error("Failed Set arg 4.", status);
      freeResources();
      return 1;
    }
    // Kernel to copy data from DDR to host
    kernel2 = clCreateKernel(program, "memcopy_from_ddr", &status);
    if(status != CL_SUCCESS) {
      dump_error("Failed clCreateKernel.", status);
      freeResources();
      return 1;
    }
    status = clSetKernelArg(kernel2, 0, sizeof(cl_mem), &hdata_ddr_b1);
    if(status != CL_SUCCESS) {
      dump_error("Failed set memcopy_from_ddr arg 0.", status);
      return 1;
    }
    status = set_fpga_buffer_kernel_param(kernel2, 1, (void*)hdataout);
    if(status != CL_SUCCESS) {
      dump_error("Failed Set memcopy_from_ddr arg 1.", status);
      freeResources();
      return 1;
    }
    arg_3 = lines;
    status = clSetKernelArg(kernel2, 2, sizeof(cl_int), &(arg_3));
    if(status != CL_SUCCESS) {
      dump_error("Failed Set arg 2.", status);
      freeResources();
      return 1;
    }

    printf("Copying to DDR\n");

    status = enqueue_fpga_buffer(queue, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE,
       (void *)hdatain, buf_size, 0, NULL, NULL);
    if(status != CL_SUCCESS) {
      dump_error("Failed enqueue_fpga_buffer", status);
      freeResources();
      return 1;
    }
    status = enqueue_fpga_buffer(queue, CL_TRUE,  CL_MAP_READ | CL_MAP_WRITE,
       (void *)hdataout, buf_size, 0, NULL, NULL);
    if(status != CL_SUCCESS) {
      dump_error("Failed enqueue_fpga_buffer", status);
      freeResources();
      return 1;
    }
	  printf("Launching the kernel...\n");

    cl_event start_kernel, start_kernel_write1, start_kernel_write2, start_kernel2;
    const double start_time = getCurrentTimestamp();
    status = clEnqueueTask(queue1, kernel, 0, NULL, &start_kernel);
    if (status != CL_SUCCESS) {
      dump_error("Failed to launch kernel.", status);
      freeResources();
      return 1;
    }
    clFinish(queue1);

    status = clEnqueueTask(queue1, kernel_write, 0, NULL, &start_kernel_write1);
    if (status != CL_SUCCESS) {
      dump_error("Failed to launch kernel.", status);
      freeResources();
      return 1;
    }
    clFinish(queue1);
    /*
    status = clSetKernelArg(kernel_write, 2, sizeof(cl_mem), &hdata_ddr_b1);
    if(status != CL_SUCCESS) {
      dump_error("Failed Set arg 2.", status);
      freeResources();
      return 1;
    }
    status = clSetKernelArg(kernel_write, 3, sizeof(cl_mem), &hdata_ddr_b2);
    if(status != CL_SUCCESS) {
      dump_error("Failed set arg 3.", status);
      return 1;
    }

    status = clEnqueueTask(queue1, kernel_write, 0, NULL, &start_kernel_write2);
    if (status != CL_SUCCESS) {
      dump_error("Failed to launch kernel.", status);
      freeResources();
      return 1;
    }
    clFinish(queue1);
    */

    status = clEnqueueTask(queue1, kernel2, 0, NULL, &start_kernel2);
    if (status != CL_SUCCESS) {
      dump_error("Failed to launch kernel.", status);
      freeResources();
      return 1;
    }
    clFinish(queue1);
    const double end_time = getCurrentTimestamp();

	  status = unenqueue_fpga_buffer(queue, (void *)hdatain, 0, NULL, NULL);
    if(status != CL_SUCCESS) {
      dump_error("Failed unenqueue_fpga_buffer", status);
      freeResources();
      return 1;
    }
    status = unenqueue_fpga_buffer(queue, (void *)hdataout, 0, NULL, NULL);
    if(status != CL_SUCCESS) {
      dump_error("Failed unenqueue_fpga_buffer", status);
      freeResources();
      return 1;
    }
    clFinish(queue);

    // Verify the output
    for(size_t i = 0; i < vectorSize; i++) {
      if(hdatain[i] != hdataout[i]) {
        if (failures < 32) printf("Verification_failure %ld: %d != %d, diff %d, line %ld\n",i, hdatain[i], hdataout[i], hdatain[i]-hdataout[i],i*4/128);
        failures++;
      }else{
        successes++;
      }
    }

    cl_ulong kernel_start = 0.0, kernel_end = 0.0;
    cl_ulong kernel_write1_start = 0.0, kernel_write1_end = 0.0;
    cl_ulong kernel_write2_start = 0.0, kernel_write2_end = 0.0;
    cl_ulong kernel2_start = 0.0, kernel2_end = 0.0;

    clGetEventProfilingInfo(start_kernel, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &kernel_start, NULL);
    clGetEventProfilingInfo(start_kernel, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &kernel_end, NULL);

    clGetEventProfilingInfo(start_kernel_write1, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &kernel_write1_start, NULL);
    clGetEventProfilingInfo(start_kernel_write1, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &kernel_write1_end, NULL);

    clGetEventProfilingInfo(start_kernel_write2, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &kernel_write2_start, NULL);
    clGetEventProfilingInfo(start_kernel_write2, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &kernel_write2_end, NULL);

    clGetEventProfilingInfo(start_kernel2, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &kernel2_start, NULL);
    clGetEventProfilingInfo(start_kernel2, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &kernel2_end, NULL);

    printf("\n");
    printf("Kernel Start  %lu\n", kernel_start);
    printf("Kernel End    %lu\n", kernel_end);
    printf("Kernel Write1 Start  %lu\n", kernel_write1_start);
    printf("Kernel Write1 End    %lu\n", kernel_write1_end);
    printf("Kernel Write2 Start  %lu\n", kernel_write2_start);
    printf("Kernel Write2 End    %lu\n", kernel_write2_end);
    printf("Kernel2 Start  %lu\n", kernel2_start);
    printf("Kernel2 End    %lu\n", kernel2_end);
    printf("\n");

    double kernel_exec = (double)(kernel_end - kernel_start) * 1e-6;
    double kernel_write1_exec = (double)(kernel_write1_end - kernel_write1_start) * 1e-6;
    double kernel_write2_exec = (double)(kernel_write2_end - kernel_write2_start) * 1e-6;
    double kernel2_exec = (double)(kernel2_end - kernel2_start) * 1e-6;

    printf("Kernel Exec Time %lf ms\n", kernel_exec);
    printf("1 Overlapping Kernel Exec Time  %lf ms\n", kernel_write1_exec);
    printf("2 Overlapping Kernel Exec Time %lf ms\n", kernel_write2_exec);
    printf("Kernel2 Exec Time  %lf ms\n", kernel2_exec);
    printf("\n");

    double time = (end_time - start_time);

    bw = vectorSize * 4 / (time * 1000000.0f) * sizeof(unsigned int);
    printf("Processed %ld unsigned ints in %.4f us\n", vectorSize * 4, time*1000000.0f);
    printf("Full Duplex Bandwidth = %.0f MB/s\n", bw);
    printf("Kernel execution is complete.\n");
  }
  if(failures == 0) {
    printf("Verification finished.\n");
  } else {
    printf("FAILURES %d - successes - %d\n", failures, successes);
  }

  freeResources();

  return 0;
}



