#ifndef GPU_PROPERTY_STEP_H
#define GPU_PROPERTY_STEP_H


#define GPU_PROPERTY_STEP 0x21

/**
    Gathers the properties for a particular Vertex.
    Only supports "VALUE" property steps currently.
**/
class GPUPropertyStep: public TraversalStep {
    private:
        std::vector<std::string> keys; //duplicates are allowed, per api

    public:
        GPUPropertyStep(std::vector<std::string> keys) : TraversalStep(MAP, GPU_PROPERTY_STEP) {
            this->keys = keys;
        }

        virtual void apply(GraphTraversal* traversal, TraverserSet& traversers) {
            GPUGraph* graph = static_cast<GPUGraph*>(traversal->getTraversalSource()->getGraph());

            TraverserSet new_traversers;
            for(Traverser& trv : traversers) {
                auto side_effects = trv.get_side_effects();

                // Each traverser must be a Vertex
                GPUReferenceVertex* v = static_cast<GPUReferenceVertex*>(boost::any_cast<Vertex*>(trv.get()));
                
                // Access by GPU Vertex id
                size_t& v_id_gpu = v->gpu_vertex_id;
                for(std::string key : keys) {
                    // TODO path history, maybe
                    new_traversers.push_back(
                        Traverser(graph->get_property(key, v_id_gpu), side_effects)
                    );
                }
            }

            traversers.swap(new_traversers);
        }

        virtual std::string getInfo() {
            return "GPUPropertyStep{}";
        }

};

#endif