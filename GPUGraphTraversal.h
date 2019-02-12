#ifndef GPU_GRAPH_TRAVERSAL_H
#define GPU_GRAPH_TRAVERSAL_H

#include "CPUGraphTraversal.h"
#include <CL/cl.hpp>
#include <list>
#include <math.h>
#include <typeinfo>

class GPUGraphTraversal : public CPUGraphTraversal {
    private:
        cl::Context context;
        cl::Device device;

    public:
        GPUGraphTraversal(GraphTraversalSource* src)
        : CPUGraphTraversal(src) {
            std::cout << "reached gpu graph traversal constructor\n";
            std::vector<cl::Platform> platforms;
	        cl::Platform::get(&platforms);

            cl::Platform platform = platforms[0];
            std::vector<cl::Device> devices;
	        platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

            device = devices[0];

            std::cout << "Using OpenCL platform: \t" << platform.getInfo<CL_PLATFORM_NAME>() << endl;
            std::cout << endl << "Using OpenCL device: \t" << device.getInfo<CL_DEVICE_NAME>() << endl << endl;

            context = cl::Context(device);
        }

        cl::Context getContext() {
            return context;
        }

        cl::Device getDevice() {
            return device;
        }

        /*
            Filters the current traversers using a GPU vector operator Q.
        */
        GraphTraversal* filter(std::function<std::vector<bool>(std::list<Traverser*>*)> q_filter) {
            return this->appendStep(new GPUFilterStep(q_filter));
        }

        /*
            Matches the standard API by returning NaN if there are no traversers to evaluate.
        */
        virtual void execute_min_step(std::list<Traverser*>* traversers, MinStep* min_step) {
            if(traversers->empty()) {
                traversers->push_back(new Traverser(std::numeric_limits<double>::quiet_NaN()));
                return;
            }

            // Attempt to determine value type.
            size_t num_traversers = traversers->size();
            boost::any object = traversers->front()->get();
            Traverser* trv;
            const std::type_info& type = object.type();
            if(type == typeid(int)) {
                std::vector<uint64_t> original_values(num_traversers);
                int k = 0; for(auto it = traversers->begin(); it != traversers->end(); ++it) original_values[k++] = static_cast<uint64_t>(boost::any_cast<int>((*it)->get()));
                trv = find_min(original_values);
            }
            else if(type == typeid(uint16_t)) {
                std::vector<uint64_t> original_values(num_traversers);
                int k = 0; for(auto it = traversers->begin(); it != traversers->end(); ++it) original_values[k++] = static_cast<uint64_t>(boost::any_cast<uint16_t>((*it)->get()));
                trv = find_min(original_values);
            }
            else if(type == typeid(uint32_t)) {
                std::vector<uint64_t> original_values(num_traversers);
                int k = 0; for(auto it = traversers->begin(); it != traversers->end(); ++it) original_values[k++] = static_cast<uint64_t>(boost::any_cast<uint32_t>((*it)->get()));
                trv = find_min(original_values);
            }
            else if(type == typeid(uint64_t)) {
                std::vector<uint64_t> original_values(num_traversers);
                int k = 0; for(auto it = traversers->begin(); it != traversers->end(); ++it) original_values[k++] = boost::any_cast<uint64_t>((*it)->get());
                trv = find_min(original_values);
            }
            else if(type == typeid(std::string)) {
                trv = new Traverser("error");
            }

            traversers->clear();
            traversers->push_back(trv);
        }

        Traverser* find_min(std::vector<uint64_t>& original_values) {
            std::cout << "uint64_t min method selected\n";
            // Prep for the declaration of the actual function
            cl::Context context = getContext();
            cl::Device device = getDevice();
            const char* source_str =
                "   __kernel void minimum(__global ulong* sz, __global ulong* values) {  \n"
                "   ulong size = sz[0];             \n"
                "   const int i = get_global_id(0); \n"
                "   int j;                          \n"
                "   for(j = 1; j < size; j=j*2) {   \n"
                "       if(i \% (2*j) == 0) {           \n"
                "          printf(\"\%d\\n\", values[i]); \n"
                "          int m = values[i];           \n"
                "          if(!(i + j >= size || m < values[i+j])) m = values[i+j];  \n"
                "          values[i] = m;           \n"
                "       }                           \n"
                "       barrier(CLK_GLOBAL_MEM_FENCE); \n"
                "   }\n"
                "}\n";

            
            cl::Program program = cl::Program(context, source_str);

            // Continued prep; compiling the device code
            cl_int compile_status = program.build({ device }, "");
            if(compile_status) std::cout << "Error during compilation! (" << compile_status << ")" << endl;
            else std::cout << "Compilation succesful!\n";
            	
            uint64_t num_values = static_cast<uint64_t>(original_values.size());

            std::string name = device.getInfo<CL_DEVICE_NAME>();
            std::string build_log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
            std::cout << name << ":\t" << build_log << "\n";

            cl::Buffer values = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, num_values * sizeof(cl_ulong), &original_values[0]);
            cl::Buffer size = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_ulong), &num_values);
            std::cout << "Buffers constructed successfully\n";

            cl::Kernel kernel = cl::Kernel(program, "minimum");
            kernel.setArg(0, size);
            kernel.setArg(1, values);

            cl::CommandQueue queue = cl::CommandQueue(context, device);
            queue.enqueueNDRangeKernel(kernel, NULL, num_values, num_values);
            queue.enqueueReadBuffer(values, CL_TRUE, 0, sizeof(cl_ulong), &original_values[0]); // only 1 value we care about

            std::cout << "Results read\n";

            uint64_t min = original_values[0];
            std::cout << "The min is: " << min << "\n";
            return new Traverser(min);
        }
};

#endif
