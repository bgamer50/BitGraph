#ifndef GPU_GRAPH_TRAVERSAL_SOURCE_H
#define GPU_GRAPH_TRAVERSAL_SOURCE_H

#include <CL/cl.hpp>
#include "CPUGraphTraversalSource.h"

class GPUGraphTraversalSource: public CPUGraphTraversalSource {
    private:
        void compile_min_program() {
            const char* source_str =
                "   __kernel void minimum(__global ulong* sz, __global ulong* values) {  \n"
                "   ulong size = sz[0];             \n"
                "   const int i = get_global_id(0); \n"
                "   int j;                          \n"
                "   for(j = 1; j < size; j=j*2) {   \n"
                "       if(i \% (2*j) == 0) {           \n"
                "          //printf(\"\%d\\n\", values[i]); \n"
                "          int m = values[i];           \n"
                "          if(!(i + j >= size || m < values[i+j])) m = values[i+j];  \n"
                "          values[i] = m;           \n"
                "       }                           \n"
                "       barrier(CLK_GLOBAL_MEM_FENCE); \n"
                "   }\n"
                "}\n";

            auto start = std::chrono::system_clock::now();
            min_program = cl::Program(context, source_str);

            // Continued prep; compiling the device code
            cl_int compile_status = min_program.build({ device }, "");
            if(compile_status) std::cout << "Error during compilation! (" << compile_status << ")" << endl;
            #ifdef VERBOSE
            else std::cout << "Compilation succesful!\n";
            #endif

            std::string name = device.getInfo<CL_DEVICE_NAME>();
            std::string build_log = min_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
            auto end = std::chrono::system_clock::now();
            std::chrono::duration<double> diff = end - start;
            //std::cout << "compile time: " << diff.count() << std::endl;
            //std::cout << name << ":\t" << build_log << "\n";
        }

    public:
        cl::Program min_program;
        cl::Context context;
        cl::Device device;

        GPUGraphTraversalSource(CPUGraph* gr)
        : CPUGraphTraversalSource(gr) {
            std::vector<cl::Platform> platforms;
	        cl::Platform::get(&platforms);

            cl::Platform platform = platforms[0];
            std::vector<cl::Device> devices;
	        platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

            device = devices[0];

            #ifdef VERBOSE
            std::cout << "Using OpenCL platform: \t" << platform.getInfo<CL_PLATFORM_NAME>() << endl;
            std::cout << endl << "Using OpenCL device: \t" << device.getInfo<CL_DEVICE_NAME>() << endl << endl;
            #endif

            context = cl::Context(device);
            compile_min_program();
        }

        virtual GraphTraversal* get_appropriate_traversal();
};

#include "GPUGraphTraversal.h"

GraphTraversal* GPUGraphTraversalSource::get_appropriate_traversal() {
	return new GPUGraphTraversal(this);
}

#endif