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
#define REQD_WG_SIZE (1024 * 32)

// Copies 64 byte lines from src to dst
__kernel void
__attribute__((task))
memcopy (
  __global __attribute((buffer_location("host"))) ulong8 * restrict src,
  __global __attribute((buffer_location("host"))) ulong8 * restrict dst,
  int lines)
{
  for(int i = 0; i < lines; i++) {
    dst[i] = src[i];
  }
}

__kernel void
memcopy_read_write (
  __global __attribute((buffer_location("host"))) ulong8 * restrict src1,
  __global __attribute((buffer_location("host"))) ulong8 * restrict src2,
  __global __attribute((buffer_location("device"))) ulong8 * restrict ddr1,
  __global __attribute((buffer_location("device"))) ulong8 * restrict ddr2,
  int lines)
{

  for(int i = 0; i < lines; i++) {
    ddr1[i] = src1[i];
    src2[i] = ddr2[i];
  }

}

__kernel void
__attribute__((task))
memread_memwrite (
  __global __attribute((buffer_location("host"))) ulong8 * restrict src,
  __global __attribute((buffer_location("host"))) ulong8 * restrict dst1,
  __global __attribute((buffer_location("host"))) ulong8 * restrict dst2,
  int lines)
{
  ulong8 sum = (0,0,0,0,0,0,0,0);
  for(int i = 0; i < lines; i++) {
    sum += src[i];
    dst2[i] = (1,1,1,1,1,1,1,1);
  }
  // This prevents the reads from being optimized away
  dst1[0] = sum;
}


// Copies 64 byte lines from src to dst
__kernel void
memcopy_batch (
  __global __attribute((buffer_location("host"))) ulong8 * restrict src1,
  __global __attribute((buffer_location("host"))) ulong8 * restrict src2,
  __global __attribute((buffer_location("device"))) ulong8 * restrict dst_ddr1,
  __global __attribute((buffer_location("device"))) ulong8 * restrict dst_ddr2,
  __global __attribute((buffer_location("host"))) ulong8 * restrict dst_src1,
  __global __attribute((buffer_location("host"))) ulong8 * restrict dst_src2,
  int lines)
{

  #pragma nofusion
  for(int i = 0; i < lines; i++) {
    dst_ddr1[i] = src1[i];
  }

  #pragma nofusion
  for(int i = 0; i < lines; i++) {
    dst_src1[i] = dst_ddr1[i];
    dst_ddr2[i] = src2[i];
  }

  #pragma nofusion
  for(int i = 0; i < lines; i++){
    dst_src2[i] = dst_ddr2[i];
  }
}

// Copies 64 byte lines from src to dst
__kernel void
memcopy_to_ddr (
  __global __attribute((buffer_location("host"))) ulong8 * restrict src,
  __global __attribute((buffer_location("device"))) ulong8 * restrict dst,
  int lines)
{
  for(int i = 0; i < lines; i++) {
    dst[i] = src[i];
  }
}

// Copies 64 byte lines from src to dst
__kernel void
memcopy_from_ddr (
  __global __attribute((buffer_location("device"))) ulong8 * restrict src,
  __global __attribute((buffer_location("host"))) ulong8 * restrict dst,
  int lines)
{
  for(int i = 0; i < lines; i++) {
    dst[i] = src[i];
  }
}

// Copies 64 byte lines from src to dst
__kernel void
memcopy_ddr (
  __global __attribute((buffer_location("device"))) ulong8 * restrict src,
  __global __attribute((buffer_location("device"))) ulong8 * restrict dst,
  int lines)
{
  for(int i = 0; i < lines; i++) {
    dst[i] = src[i];
  }
}

// Reads 64 byte lines from src
__kernel void
memread (
  __global __attribute((buffer_location("host"))) ulong8 * restrict src,
  __global __attribute((buffer_location("host"))) ulong8 * restrict dst,
  int lines)
{
  ulong8 sum = (0,0,0,0,0,0,0,0);
  for(int i = 0; i < lines; i++) {
    sum += src[i];
  }
  // This prevents the reads from being optimized away
  dst[0] = sum;
}

// Writes 64 byte lines to dst
__kernel void
memwrite (
  __global __attribute((buffer_location("host"))) ulong8 * restrict src,
  __global __attribute((buffer_location("host"))) ulong8 * restrict dst,
  int lines)
{
  for(int i = 0; i < lines; i++) {
    dst[i] = (0,0,0,0,0,0,0,0);
  }
}

__kernel void
nop ()
{
}

