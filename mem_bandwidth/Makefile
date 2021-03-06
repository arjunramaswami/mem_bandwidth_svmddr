# Copyright (C) 2013-2014 Altera Corporation, San Jose, California, USA. All rights reserved.
# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the "Software"), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to
# whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.
#
# This agreement shall be governed in all respects by the laws of the State of California and
# by the laws of the United States of America.


# This is a GNU Makefile.

# It can be used to compile an OpenCL program with
# the Altera SDK for OpenCL.
# See README.txt for more information.


# You must configure ALTERAOCLSDKROOT to point the root directory of the Altera SDK for OpenCL
# software installation.
# See http://www.altera.com/literature/hb/opencl-sdk/aocl_getting_started.pdf
# for more information on installing and configuring the Altera SDK for OpenCL.

ifeq ($(VERBOSE),1)
ECHO :=
else
ECHO := @
endif

TARGET := mem_bandwidth

# Where is the Altera SDK for OpenCL software?
ifeq ($(wildcard $(INTELFPGAOCLSDKROOT)),)
$(error Set INTELFPGAOCLSDKROOT to the root directory of the Altera SDK for OpenCL software installation)
endif
ifeq ($(wildcard $(INTELFPGAOCLSDKROOT)/host/include/CL/opencl.h),)
$(error Set INTELFPGAOCLSDKROOT to the root directory of the Altera SDK for OpenCL software installation.)
endif

# Directories.
HOST_DIR := host
TARGET_DIR := bin
INC_DIRS := host/inc ../common/inc ../extlibs/inc   $(ALTERAOCLSDKROOT)/host/include/
INC_DIRS += $(AOCL_BOARD_PACKAGE_ROOT)/source/include

ifneq (,$(prefix))
$(Adding OPAE path since not installed at default location )
INC_DIRS += $(prefix)/include
#LIB_DIRS += $(prefix)/lib -Wl,-rpath-link -Wl,$(prefix)/lib -Wl,-rpath -Wl,$(prefix)/lib \
#				$(prefix)/lib64 -Wl,-rpath-link -Wl,$(prefix)/lib64 -Wl,-rpath -Wl,$(prefix)/lib64
endif


# All include files.
INC_FILES := $(foreach D,$(INC_DIRS),$(wildcard $D/*.h))

# Source files.
SRCS := $(wildcard ../common/src/AOCLUtils/*.cpp) $(wildcard host/src/*.cpp)

# Libraries.
#LIBS := alteracl intel_opae_mmd rt elf MPF
LIBS := rt 
# Compiler.
CXX := g++

# OpenCL compile and link flags.
AOCL_COMPILE_CONFIG := $(shell aocl compile-config)
AOCL_LINK_CONFIG := $(shell aocl link-config)
#AOCL_LINK_CONFIG :=
#AOCL_LINK_CONFIG += -L/cm/shared/opt/intel_oneapi/beta-10/intelfpgadpcpp/2021.1-beta10/board/intel_s10sx_pac/linux64/lib
#AOCL_LINK_CONFIG += -L/cm/shared/opt/intel_oneapi/beta-10/compiler/2021.1-beta10/linux/lib/oclfpga/linux64/lib/
#AOCL_LINK_CONFIG += -L/cm/shared/opt/intel_oneapi/beta-10/compiler/2021.1-beta10/linux/lib/oclfpga/host/linux64/lib

# Compilation flags
ifeq ($(DEBUG),1)
CXXFLAGS += -g -D__USE_XOPEN2K8
else
CXXFLAGS += -O2 -D__USE_XOPEN2K8
endif
CXXFLAGS+= -Wall -DHAVE_CONFIG_H -DTESTB -g

# Make it all!
all : $(TARGET_DIR)/$(TARGET)

# Host executable target.
$(TARGET_DIR)/$(TARGET) : Makefile $(SRCS) $(INC_FILES)
	@[ -d $(TARGET_DIR) ] || mkdir $(TARGET_DIR)
	$(ECHO)$(CXX) $(CPPFLAGS) $(CXXFLAGS) -fPIC $(foreach D,$(INC_DIRS),-I$D) \
			$(AOCL_COMPILE_CONFIG) $(SRCS) $(AOCL_LINK_CONFIG) \
			$(foreach D,$(LIB_DIRS),-L$D -Wl,-rpath-link) \
			$(foreach L,$(LIBS),-l$L) \
			-o $(TARGET_DIR)/$(TARGET)

#### device targets
emu : device/mem_bandwidth_ddr.cl
	aoc -legacy-emulator -march=emulator -board=pac_s10_usm -board-package=${AOCL_BOARD_PACKAGE_ROOT} -o bin/mem_bandwidth_s10_svmddr_batch_emu.aocx device/mem_bandwidth_ddr_batch.cl

run_emu : emu
	bin/mem_bandwidth 1024 emu

report_hw:
	aoc -rtl -board-package=$(AOCL_BOARD_PACKAGE_ROOT) -board=pac_s10_usm -o bin/mem_bandwidth_s10_svmddr_batch_report.out device/mem_bandwidth_ddr_batch.cl

build_hw:
	aoc -board-package=$(AOCL_BOARD_PACKAGE_ROOT) -board=pac_s10_usm -o bin/mem_bandwidth_s10_svmddr.aocx device/mem_bandwidth_ddr.cl

build_hw_no_interleaving:
	aoc -board-package=$(AOCL_BOARD_PACKAGE_ROOT) -board=pac_s10_usm -no-interleaving=device -o bin/mem_bandwidth_s10_svmddr_batch3.aocx device/mem_bandwidth_ddr_batch.cl

profile_hw_no_interleaving:
	aoc -board-package=$(AOCL_BOARD_PACKAGE_ROOT) -board=pac_s10_usm -profile -no-interleaving=device -o bin/mem_bandwidth_s10_svmddr_batch2_prof.aocx device/mem_bandwidth_ddr_batch.cl

run_hw :
	bin/mem_bandwidth 1024000

# Standard make targets
clean :
	$(ECHO)rm -f $(TARGET_DIR)/$(TARGET)

.PHONY : all clean

