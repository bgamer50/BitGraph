#ifndef GPU_GRAPH_TRAVERSAL_H
#define GPU_GRAPH_TRAVERSAL_H

#include "CPUGraphTraversal.h"
#include "GPUGraphTraversalSource.h"
#include <CL/cl.hpp>
#include <list>
#include <math.h>
#include <typeinfo>

#define MIN_GPU_TRAVERSERS_MIN 1000

class GPUGraphTraversal : public CPUGraphTraversal {
    public:
        GPUGraphTraversal(GraphTraversalSource* src)
        : CPUGraphTraversal(src) {
            // empty
        }

        // Generally only called from other GPUGraphTraversals.
        GPUGraphTraversal(GraphTraversalSource* src, GraphTraversal* anonymous_traversal)
		: CPUGraphTraversal(src, anonymous_traversal) {
            // empty
        }

        cl::Context getContext() {
            return dynamic_cast<GPUGraphTraversalSource*>(this->getTraversalSource())->context;
        }

        cl::Device getDevice() {
           return dynamic_cast<GPUGraphTraversalSource*>(this->getTraversalSource())->device;
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

            size_t num_traversers = traversers->size();
            if(num_traversers < MIN_GPU_TRAVERSERS_MIN) {
                return CPUGraphTraversal::execute_min_step(traversers, min_step);
            }

            // Attempt to determine value type.
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
        
        virtual CPUGraphTraversal* from_anonymous_traversal(GraphTraversal* anonymous) {
		    return new GPUGraphTraversal(this->getTraversalSource(), anonymous);
	    }

        Traverser* find_min(std::vector<uint64_t>& original_values) {
            //std::cout << "uint64_t min method selected\n";
            // Prep for the declaration of the actual function

            uint64_t num_values = static_cast<uint64_t>(original_values.size());
            cl::Buffer values = cl::Buffer(this->getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, num_values * sizeof(cl_ulong), &original_values[0]);
            cl::Buffer size = cl::Buffer(this->getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_ulong), &num_values);
            #ifdef VERBOSE
            std::cout << "Buffers constructed successfully\n";
            #endif

            cl::Program& min_program = dynamic_cast<GPUGraphTraversalSource*>(this->getTraversalSource())->min_program;

            cl::Kernel kernel = cl::Kernel(min_program, "minimum");
            kernel.setArg(0, size);
            kernel.setArg(1, values);

            cl::CommandQueue queue = cl::CommandQueue(this->getContext(), this->getDevice());
            queue.enqueueNDRangeKernel(kernel, NULL, num_values, num_values);
            queue.enqueueReadBuffer(values, CL_TRUE, 0, sizeof(cl_ulong), &original_values[0]); // only 1 value we care about

            #ifdef VERBOSE
            std::cout << "Results read\n";
            #endif

            uint64_t min = original_values[0];
            //std::cout << "The min is: " << min << "\n";
            return new Traverser(min);
        }
};

#endif
