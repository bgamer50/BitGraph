#ifndef Q_H
#define Q_H

#include "GPUGraphTraversal.h"
#include <functional>
#include <boost/any.hpp>
#include <inttypes.h>
#include <iostream>
#include <string>

template<typename T>
class Q {
	public:
		static std::function<std::vector<bool>(std::list<Traverser*>*)> eq(T val, GPUGraphTraversal* trv) {
			std::cout << "generic eq method selected\n";
			return [val](std::list<Traverser*>* traversers) {
				return std::vector<bool>(traversers->size(), false);
			};
		}
};

template<>
class Q<uint64_t> {
	public:
		static std::function<std::vector<bool>(std::list<Traverser*>*)> eq(uint64_t val, GPUGraphTraversal* trv) {
            std::cout << "uint64_t eq method selected\n";
            // Prep for the declaration of the actual function
            cl::Context context = trv->getContext();
            cl::Device device = trv->getDevice();
            const char* source_str = 
                "	__kernel void filter(__global ulong* expected, __global ulong* values, __global bool* result) {	"
		        "	const int i = get_global_id(0);	" 
                "   printf(\"%lu\\n\", i); "
		        "	result[i] = values[i] == expected[0]; "
		        "}";

            
            cl::Program program = cl::Program(context, source_str);

            // Continued prep; compiling the device code
            cl_int compile_status = program.build({ device }, "");
            if(compile_status) std::cout << "Error during compilation! (" << compile_status << ")" << endl;
            else std::cout << "Compilation succesful!\n";

            // Write the lambda function
			return [val, context, device, program](std::list<Traverser*>* traversers) {
				std::cout << "uint64_t method selected\n";	
                size_t num_traversers = traversers->size();
                std::vector<uint64_t> original_values(num_traversers);
                int k = 0;
                for(auto it = traversers->begin(); it != traversers->end(); ++it) {
                    original_values[k++] = boost::any_cast<uint64_t>((*it)->get());
                }

                std::string name = device.getInfo<CL_DEVICE_NAME>();
                std::string build_log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
                std::cout << name << ":\t" << build_log << "\n";

                cl::Buffer expected = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_ulong), const_cast<uint64_t*>(&val));
                cl::Buffer values = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, num_traversers * sizeof(cl_ulong), &original_values[0]);
                cl::Buffer result = cl::Buffer(context, CL_MEM_WRITE_ONLY, num_traversers * sizeof(cl_bool), NULL);
                bool cpu_result[num_traversers];

                std::cout << "Buffers constructed successfully\n";

                cl::Kernel kernel = cl::Kernel(program, "filter");
                kernel.setArg(0, expected);
                kernel.setArg(1, values);
                kernel.setArg(2, result);

                cl::CommandQueue queue = cl::CommandQueue(context, device);
                queue.enqueueNDRangeKernel(kernel, NULL, num_traversers, num_traversers);
                queue.enqueueReadBuffer(result, CL_TRUE, 0, num_traversers * sizeof(cl_bool), &cpu_result[0]);

                std::cout << "Results read\n";

                std::vector<bool> vect = std::vector<bool>(&cpu_result[0], &cpu_result[num_traversers]);
                for(auto it = vect.begin(); it != vect.end(); ++it) std::cout << (*it ? "true\n" : "false\n");
                return vect;
			};
		}
};

#endif
