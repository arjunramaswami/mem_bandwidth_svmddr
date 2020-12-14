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

using namespace aocl_utils;
static const size_t V = 16;
//static size_t vectorSize = 1024*1024*4*16;
static size_t vectorSize = 1024*1024*16;

bool use_prealloc_svm_buffer = true;

float bw;
// 0 - runall, 2 memcopy , 3 read, 4 write, 5 ddr
int runkernel = 0;

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
void *alloc_fpga_host_buffer(cl_context &in_context, int some_int, int size, int some_int2);
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

cl_mem hdata_ddr1, hdata_ddr2;

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
  if(kernel_read)
    clReleaseKernel(kernel_read);
  if(kernel_write)
    clReleaseKernel(kernel_write);
  if(program)
    clReleaseProgram(program);
  if(queue)
    clReleaseCommandQueue(queue);
  if(hdatain)
   remove_fpga_buffer(context,hdatain);
  if(hdataout)
   remove_fpga_buffer(context,hdataout);
  if(hdatain_2)
   remove_fpga_buffer(context,hdatain_2);
  if(hdataout_2)
   remove_fpga_buffer(context,hdataout_2);
  free( hdatatemp);
  free( hdatatemp_2);
  if(context)
    clReleaseContext(context);

}


void *alloc_fpga_host_buffer(cl_context &in_context, int some_int, int size, int some_int2)
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
		   return clSVMAlloc(in_context, some_int, size, some_int2);
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
    aocx_name_string = "bin/mem_bandwidth_s10_svmddr_prof.aocx";
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
  status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, &num_devices);
  if(status != CL_SUCCESS) {
    dump_error("Failed clGetDeviceIDs.", status);
    freeResources();
    return 1;
  }
  if(num_devices != 1) {
    printf("Found %d devices!\n", num_devices);
    freeResources();
    return 1;
  }

  // create a context
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
  hdatain = (unsigned int*)alloc_fpga_host_buffer(context, 0, buf_size, 1024);
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
  /*
  hdata_ddr1 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_CHANNEL_2_INTELFPGA, buf_size, NULL, &status);
  if(status != CL_SUCCESS) {
    dump_error("Failed clCreateBuffer.", status);
    freeResources();
    return 1;
  }

  hdata_ddr2 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_CHANNEL_4_INTELFPGA, buf_size, NULL, &status);
  if(status != CL_SUCCESS) {
    dump_error("Failed clCreateBuffer.", status);
    freeResources();
    return 1;
  }
  */

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

  printf("Creating nop kernel\n");
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


  printf("Starting memcopy kernel\n");
// Done kernel launch test
  initializeVector_seq(hdatain, vectorSize);
  initializeVector(hdataout, vectorSize);
  int failures = 0;
  int successes = 0;
  fflush(stdout);
  if (runkernel == 0 || runkernel == 2){
   printf("Creating memcopy kernel (Host->Host)\n");
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
    float time = (end_time - start_time);

    bw = vectorSize / (time * 1000000.0f) * sizeof(unsigned int) * 2;
    printf("Processed %d unsigned ints in %.4f us\n", vectorSize, time*1000000.0f);
    printf("Read/Write Bandwidth = %.0f MB/s\n", bw);
    printf("Kernel execution is complete.\n");

    // Verify the output
    for(size_t i = 0; i < vectorSize; i++) {
      if(hdatain[i] != hdataout[i]) {
        if (failures < 32) printf("Verification_failure %d: %d != %d, diff %d, line %d\n",i, hdatain[i], hdataout[i], hdatain[i]-hdataout[i],i*4/128);
        failures++;
      }else{
        successes++;
      }
    }
  }

  if (runkernel == 0 || runkernel == 5){

    printf("Copying to DDR\n");
    status = clEnqueueWriteBuffer(queue, hdata_ddr_b1, CL_TRUE,
			0, buf_size, hdatain, 0, NULL, NULL);

    printf("Creating memcopy_ddr kernel (DDR->DDR)\n");

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

   // status = enqueue_fpga_buffer(queue, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE,
    //   (void *)hdatain,buf_size, 0, NULL, NULL);
    if(status != CL_SUCCESS) {
      dump_error("Failed enqueue_fpga_buffer", status);
      freeResources();
      return 1;
    }
   // status = enqueue_fpga_buffer(queue, CL_TRUE,  CL_MAP_READ | CL_MAP_WRITE,
   //    (void *)hdataout, buf_size, 0, NULL, NULL);
    if(status != CL_SUCCESS) {
      dump_error("Failed enqueue_fpga_buffer", status);
      freeResources();
      return 1;
    }

    printf("after buffer enqueue\n");


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
    float time = (end_time - start_time);

    bw = vectorSize / (time * 1000000.0f) * sizeof(unsigned int) * 2;
    printf("Processed %d unsigned ints in %.4f us\n", vectorSize, time*1000000.0f);
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
        if (failures < 32) printf("Verification_failure %d: %d != %d, diff %d, line %d\n",i, hdatain[i], hdataout[i], hdatain[i]-hdataout[i],i*4/128);
        failures++;
      }else{
        successes++;
      }
    }

  }
  if (runkernel == 0 || runkernel == 2){
    printf("Creating memcopy_to_ddr and memcopy_from_ddr kernel (Host->DDR->Host)\n");
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
    cl_int arg_3 = lines;
    status = clSetKernelArg(kernel, 2, sizeof(cl_int), &(arg_3));
    if(status != CL_SUCCESS) {
      dump_error("Failed Set arg 2.", status);
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

    cl_event start_first_batch, end_first_batch, start_second_batch, end_second_batch;
    const double start_time = getCurrentTimestamp();
    // printf("Batch1: Write Kernel Start\n");
    status = clEnqueueTask(queue1, kernel, 0, NULL, &start_first_batch);
    if (status != CL_SUCCESS) {
      dump_error("Failed to launch kernel.", status);
      freeResources();
      return 1;
    }

    // printf("Batch1: Read Kernel Start\n");
    status = clEnqueueTask(queue1, kernel2, 0, NULL, &end_first_batch);
    if (status != CL_SUCCESS) {
      dump_error("Failed to launch kernel.", status);
      freeResources();
      return 1;
    }

    status = set_fpga_buffer_kernel_param(kernel, 0, (void*)hdatain_2);
    if(status != CL_SUCCESS) {
      dump_error("Failed set memcopy_to_ddr  arg 0.", status);
      return 1;
    }
    status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &hdata_ddr_b2);
    if(status != CL_SUCCESS) {
      dump_error("Failed Set memcopy_to_ddr arg 1.", status);
      freeResources();
      return 1;
    }
    // set the arguments
    status = clSetKernelArg(kernel2, 0, sizeof(cl_mem), &hdata_ddr_b2);
    if(status != CL_SUCCESS) {
      dump_error("Failed set memcopy_from_ddr arg 0.", status);
      return 1;
    }
    status = set_fpga_buffer_kernel_param(kernel2, 1, (void*)hdataout_2);
    if(status != CL_SUCCESS) {
      dump_error("Failed Set memcopy_from_ddr arg 1.", status);
      freeResources();
      return 1;
    }

    // printf("Batch2: Write Kernel Start\n");
    status = clEnqueueTask(queue2, kernel, 0, NULL, &start_second_batch);
    if (status != CL_SUCCESS) {
      dump_error("Failed to launch kernel.", status);
      freeResources();
      return 1;
    }

    // printf("Batch2: Read Kernel Start\n");
    status = clEnqueueTask(queue2, kernel2, 0, NULL, &end_second_batch);
    if (status != CL_SUCCESS) {
      dump_error("Failed to launch kernel.", status);
      freeResources();
      return 1;
    }
    clFinish(queue1);
    clFinish(queue2);
    // printf("Batch2: Read Kernel End\n");
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

    cl_ulong batch1_kernel1_start = 0.0, batch1_kernel1_end = 0.0;
    cl_ulong batch1_kernel2_start = 0.0, batch1_kernel2_end = 0.0;
    cl_ulong batch2_kernel1_start = 0.0, batch2_kernel1_end = 0.0;
    cl_ulong batch2_kernel2_start = 0.0, batch2_kernel2_end = 0.0;

    clGetEventProfilingInfo(start_first_batch, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &batch1_kernel1_start, NULL);
    clGetEventProfilingInfo(start_first_batch, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &batch1_kernel1_end, NULL);
    clGetEventProfilingInfo(end_first_batch, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &batch1_kernel2_start, NULL);
    clGetEventProfilingInfo(end_first_batch, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &batch1_kernel2_end, NULL);
    clGetEventProfilingInfo(start_second_batch, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &batch2_kernel1_start, NULL);
    clGetEventProfilingInfo(start_second_batch, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &batch2_kernel1_end, NULL);
    clGetEventProfilingInfo(end_second_batch, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &batch2_kernel2_start, NULL);
    clGetEventProfilingInfo(end_second_batch, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &batch2_kernel2_end, NULL);

    printf("\n");
    printf("Batch 1: Write Kernel Start  %lf\n", (double)batch1_kernel1_start);
    printf("Batch 1: Write Kernel End    %lf\n", (double)batch1_kernel1_end);
    printf("Batch 1: Read Kernel Start   %lf\n", (double)batch1_kernel2_start);
    printf("Batch 1: Read Kernel End     %lf\n", (double)batch1_kernel2_end);
    printf("Batch 2: Write Kernel Start  %lf\n", (double)batch2_kernel1_start);
    printf("Batch 2: Write Kernel End    %lf\n", (double)batch2_kernel1_end);
    printf("Batch 2: Read Kernel Start   %lf\n", (double)batch2_kernel2_start);
    printf("Batch 2: Read Kernel End     %lf\n", (double)batch2_kernel2_end);
    printf("\n");

    printf("Batch 1: Write Kernel Exec Time %lf\n", (double)batch1_kernel1_end - (double)batch1_kernel1_start);
    printf("Batch 1: Read Kernel Exec Time  %lf\n", (double)batch1_kernel2_end - (double)batch1_kernel2_start);
    printf("Batch 2: Write Kernel Exec Time %lf\n", (double)batch2_kernel1_end - (double)batch2_kernel1_start);
    printf("Batch 2: Read Kernel Exec Time  %lf\n", (double)batch2_kernel2_end - (double)batch2_kernel2_start);
    printf("\n");
    float time = (end_time - start_time);

    bw = vectorSize * sizeof(unsigned int) * 2 * 2 / (time * 1000000.0f);
    printf("Processed %d unsigned ints in %.4f us\n", 2* 2* vectorSize, time*1000000.0f);
    printf("Read/Write Bandwidth = %.0f MB/s\n", bw);
    printf("Kernel execution is complete.\n");

    // Verify the output
    for(size_t i = 0; i < vectorSize; i++) {
      if(hdatain[i] != hdataout[i]) {
        if (failures < 32) printf("Verification_failure %d: %d != %d, diff %d, line %d\n",i, hdatain[i], hdataout[i], hdatain[i]-hdataout[i],i*4/128);
        failures++;
      }else{
        successes++;
      }
    }
    // Verify the output
    for(size_t i = 0; i < vectorSize; i++) {
      if(hdatain_2[i] != hdataout_2[i]) {
        if (failures < 32) printf("Verification_failure %d: %d != %d, diff %d, line %d\n",i, hdatain[i], hdataout[i], hdatain[i]-hdataout[i],i*4/128);
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
        if (failures < 32) printf("Verification_failure %d: %d != %d, diff %d, line %d\n",i, hdatain[i], hdataout[i], hdatain[i]-hdataout[i],i*4/128);
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
        if (failures < 32) printf("Verification_failure %d: %d != %d, diff %d, line %d\n",i, hdatain_2[i], hdataout_2[i], hdatain_2[i]-hdataout_2[i],i*4/128);
        failures++;
      }else{
        successes++;
      }
    }

  }


  if (runkernel == 0 || runkernel == 3){
   printf("Creating memread kernel\n");
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
    float time = (end_time - start_time);

    bw = vectorSize  / (time * 1000000.0f) * sizeof(unsigned int);
    printf("Processed %d unsigned ints in %.4f us\n", vectorSize, time*1000000.0f);
    printf("Read Bandwidth = %.0f MB/s\n", bw);
    printf("Kernel execution is complete.\n");

  }

  if (runkernel == 0 || runkernel == 4){
   printf("Creating memwrite kernel\n");
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
    float time = (end_time - start_time);

    bw = vectorSize  / (time * 1000000.0f) * sizeof(unsigned int);
    printf("Processed %d unsigned ints in %.4f us\n", vectorSize, time*1000000.0f);
    printf("Write Bandwidth = %.0f MB/s\n", bw);
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



